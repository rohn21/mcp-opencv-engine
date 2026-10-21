from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP
from utils.logger import logger

# Ensure project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.config.loader import get_config
from core.config.schema import RuntimeMode
from core.filesystem.path_resolver import PathResolver
from core.filesystem.permissions import AccessDeniedError, PathTraversalError

VALID_FILTER_TYPES = {"grayscale", "edges", "blur", "threshold"}


def _get_processing_context() -> tuple[PathResolver, str, float]:
    """Retrieve PathResolver and processing service URL with environment overrides."""
    cfg = get_config()
    resolver = PathResolver(cfg)

    proc_url = os.getenv("PROCESSING_SERVICE_URL")
    if not proc_url:
        if "PYTEST_CURRENT_TEST" in os.environ:
            proc_url = "http://image-processing:8002"
        else:
            proc_url = cfg.service.processing_service_url

    timeout = cfg.service.request_timeout
    return resolver, proc_url.rstrip("/"), timeout


def register_processing_tools(mcp: FastMCP):

    @mcp.tool()
    async def apply_image_filter(image_path: str, filter_type: str) -> str:
        """
        Applies a basic computer vision filter to an image using OpenCV.
        
        USE CASE: Use this when the user wants to change the visual style of an image, 
        extract edges for analysis, or prepare an image for further processing.
        
        PARAMETERS:
        - image_path (str): The absolute path or relative filename to the image.
        - filter_type (str): STRICTLY one of the following 4 options. Do not invent other names:
            1. 'grayscale': Converts the image to black and white.
            2. 'edges': Detects and highlights the edges of objects (Canny edge detection).
            3. 'blur': Applies a Gaussian blur to soften the image.
            4. 'threshold': Converts the image to pure black and white based on a brightness cutoff.
            
        RETURNS: JSON containing the output image path and dimension changes.
        """
        logger.info("Tool: apply_image_filter")
        logger.info(f"Filter: {filter_type}")

        resolver, proc_url, timeout = _get_processing_context()
        cfg = get_config()

        # Sandbox Pre-validation and path resolution
        try:
            target_is_docker = "image-processing" in proc_url or cfg.runtime_mode == RuntimeMode.DOCKER
            resolved_path = resolver.resolve_input(image_path, target_is_docker=target_is_docker)
        except (AccessDeniedError, PathTraversalError) as exc:
            logger.error(f"Security sandbox rejection: {exc}")
            return f"Security Error: {exc}"

        logger.info(f"→ {proc_url.replace('http://', '')}")
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{proc_url}/filter", json={
                    "image_path": resolved_path,
                    "filter_type": filter_type,
                })
        except httpx.ConnectError:
            return f"Error: Unable to connect to processing service at {proc_url}. Is the service running?"
        except httpx.TimeoutException:
            return f"Error: Processing service timed out after {timeout} seconds."

        if resp.status_code != 200:
            return f"Error: Processing service returned {resp.status_code} - {resp.text}"

        data = resp.json()

        # Mount-aware safe translation of output path
        if isinstance(data, dict):
            for out_key in ("output_path", "output_image_path"):
                if out_key in data and data[out_key]:
                    data[out_key] = resolver.to_client_path(data[out_key])

        logger.info("← processing completed")
        return json.dumps(data, indent=2)

    @mcp.tool()
    async def blur_faces_in_image(image_path: str, blur_intensity: int = 99) -> str:
        """
        Detects human faces in an image and applies a heavy Gaussian blur to anonymize them.
        
        USE CASE: Use this when the user wants to protect privacy, anonymize bystanders, 
        or redact identities from a photograph before sharing or analyzing it.
        
        PARAMETERS:
        - image_path (str): The absolute path or relative filename to the image.
        - blur_intensity (int): The size of the blur kernel. MUST be an odd number (e.g., 11, 51, 99). 
          Higher numbers mean a heavier, more unrecognizable blur. Default is 99 (very heavy).
          If an even number is provided, the system will automatically add 1 to make it odd.
          
        RETURNS: JSON containing the path to the anonymized image and the number of faces found.
        """
        logger.info("Tool: blur_faces_in_image")
        logger.info(f"Image: {image_path}")

        resolver, proc_url, timeout = _get_processing_context()
        cfg = get_config()

        # Sandbox Pre-validation and path resolution
        try:
            target_is_docker = "image-processing" in proc_url or cfg.runtime_mode == RuntimeMode.DOCKER
            resolved_path = resolver.resolve_input(image_path, target_is_docker=target_is_docker)
        except (AccessDeniedError, PathTraversalError) as exc:
            logger.error(f"Security sandbox rejection: {exc}")
            return f"Security Error: {exc}"

        logger.info(f"→ {proc_url.replace('http://', '')}")
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{proc_url}/blur_faces", json={
                    "image_path": resolved_path,
                    "blur_intensity": blur_intensity,
                })
        except httpx.ConnectError:
            return f"Error: Unable to connect to processing service at {proc_url}. Is the service running?"
        except httpx.TimeoutException:
            return f"Error: Processing service timed out after {timeout} seconds."

        if resp.status_code != 200:
            return f"Error: Processing service returned {resp.status_code} - {resp.text}"

        data = resp.json()

        # Mount-aware safe translation of output path
        if isinstance(data, dict):
            for out_key in ("output_path", "output_image_path"):
                if out_key in data and data[out_key]:
                    data[out_key] = resolver.to_client_path(data[out_key])

        logger.info(f"← {data.get('faces_found', data.get('faces_detected', 0))} faces anonymized")
        return json.dumps(data, indent=2)
