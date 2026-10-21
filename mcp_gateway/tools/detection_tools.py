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


def _get_service_context() -> tuple[PathResolver, str, str, float]:
    """Retrieve PathResolver and configured URLs with environment overrides."""
    cfg = get_config()
    resolver = PathResolver(cfg)

    # 1. Detection service URL
    det_url = os.getenv("DETECTION_SERVICE_URL")
    if not det_url:
        if "PYTEST_CURRENT_TEST" in os.environ:
            det_url = "http://object-detection:8001"
        else:
            det_url = cfg.service.detection_service_url

    # 2. OCR service URL
    ocr_url = os.getenv("OCR_SERVICE_URL")
    if not ocr_url:
        if "PYTEST_CURRENT_TEST" in os.environ:
            ocr_url = "http://ocr-service:8003"
        else:
            ocr_url = cfg.service.ocr_service_url

    timeout = cfg.service.request_timeout
    return resolver, det_url.rstrip("/"), ocr_url.rstrip("/"), timeout


def register_detection_tools(mcp: FastMCP):

    @mcp.tool()
    async def detect_objects(
        image_path: str,
        confidence: float = 0.25,
        confidence_threshold: Optional[float] = None,
    ) -> str:
        """
        Detects and identifies objects in an image using the YOLOv8 AI model.
        
        USE CASE: Use this when the user wants to know what objects are in an image, 
        count specific items, or get the exact coordinates (bounding boxes) of objects.
        
        PARAMETERS:
        - image_path (str): The absolute path or relative filename to the image.
        - confidence (float): The minimum confidence score (0.0 to 1.0) required for an object 
          to be detected. Lower values (e.g., 0.25) detect more objects but may include false positives. 
          Higher values (e.g., 0.75) are stricter and only return highly confident detections.
          (Also accepts 'confidence_threshold' as an alias).
        - confidence_threshold (float, optional): Alias for confidence parameter.
          
        RETURNS: JSON with detected object names, confidence scores, bounding box coordinates, 
        and the path to the annotated image (with boxes drawn).
        """
        if confidence_threshold is not None:
            confidence = confidence_threshold

        logger.info("Tool: detect_objects")
        logger.info(f"Image: {image_path}")

        resolver, det_url, _, timeout = _get_service_context()
        cfg = get_config()

        # Sandbox Pre-validation and path resolution
        try:
            target_is_docker = "object-detection" in det_url or cfg.runtime_mode == RuntimeMode.DOCKER
            resolved_path = resolver.resolve_input(image_path, target_is_docker=target_is_docker)
        except (AccessDeniedError, PathTraversalError) as exc:
            logger.error(f"Security sandbox rejection: {exc}")
            return f"Security Error: {exc}"

        logger.info(f"→ {det_url.replace('http://', '')}")
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                health = await client.get(f"{det_url}/health")
                if health.status_code == 503:
                    return "Detection service is still starting up. Please try again in 10 seconds."

                resp = await client.post(f"{det_url}/predict", json={
                    "image_path": resolved_path,
                    "confidence_threshold": confidence,
                    "draw_boxes": True,
                })
        except httpx.ConnectError:
            return f"Error: Unable to connect to detection service at {det_url}. Is the service running?"
        except httpx.TimeoutException:
            return f"Error: Detection service timed out after {timeout} seconds."

        if resp.status_code != 200:
            return f"Error: Detection service returned {resp.status_code} - {resp.text}"

        data = resp.json()

        # Mount-aware safe translation of annotated output path
        if isinstance(data, dict):
            for out_key in ("annotated_image_path", "annotated_image"):
                if out_key in data and data[out_key]:
                    data[out_key] = resolver.to_client_path(data[out_key])

        detected_list = data.get("detections") or data.get("objects", [])
        logger.info(f"← {len(detected_list)} objects detected")
        return json.dumps(data, indent=2)

    @mcp.tool()
    async def extract_text_from_image(image_path: str) -> str:
        """
        Extracts all visible text from an image using Optical Character Recognition (OCR).
        
        USE CASE: Use this when the user wants to read text, signs, documents, license plates, 
        or any written words visible inside an image.
        
        PARAMETERS:
        - image_path (str): The absolute path or relative filename to the image.
        
        NOTE: OCR processing is computationally heavy and may take 5-15 seconds for large images.
        
        RETURNS: JSON containing the full concatenated extracted text, and a list of individual 
        text blocks with their specific bounding box coordinates and confidence scores.
        """
        logger.info("Tool: extract_text_from_image")
        logger.info(f"Image: {image_path}")

        resolver, _, ocr_url, timeout = _get_service_context()
        cfg = get_config()
        ocr_timeout = max(timeout, 120.0)

        # Sandbox Pre-validation and path resolution
        try:
            target_is_docker = "ocr-service" in ocr_url or cfg.runtime_mode == RuntimeMode.DOCKER
            resolved_path = resolver.resolve_input(image_path, target_is_docker=target_is_docker)
        except (AccessDeniedError, PathTraversalError) as exc:
            logger.error(f"Security sandbox rejection: {exc}")
            return f"Security Error: {exc}"

        logger.info(f"→ {ocr_url.replace('http://', '')}")
        try:
            async with httpx.AsyncClient(timeout=ocr_timeout) as client:
                resp = await client.post(f"{ocr_url}/extract_text", json={
                    "image_path": resolved_path,
                    "languages": cfg.model.ocr_languages if hasattr(cfg.model, "ocr_languages") else ["en"],
                })
        except httpx.ConnectError:
            return f"Error: Unable to connect to OCR service at {ocr_url}. Is the service running?"
        except httpx.TimeoutException:
            return f"Error: OCR service timed out after {ocr_timeout} seconds."

        if resp.status_code != 200:
            return f"Error: OCR service returned {resp.status_code} - {resp.text}"

        data = resp.json()
        text_preview = data.get("extracted_text") or data.get("text", "")
        logger.info(f"Extracted text:\n{text_preview[:200]}...")
        return json.dumps(data, indent=2)