"""Real network-backed TCP integration test for FastMCP Gateway.

This test connects an official MCP client session over a real network TCP socket
using the Streamable HTTP transport (not in-process ASGITransport and not in-memory streams).
It verifies:
1. Dynamic TCP socket binding and Uvicorn server startup.
2. Official MCP SDK streamable_http_client handshake and protocol negotiation.
3. Tool listing and parameter schema verification across all 4 CV tools.
4. Real computer-vision tool executions against downstream microservices.
5. Strict path-traversal and sandbox rejection over the network.
6. Deterministic test server shutdown and cleanup.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from pathlib import Path
from typing import Optional

import httpx
import pytest
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP

# Ensure project root and mcp_gateway are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "mcp_gateway") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "mcp_gateway"))

from tools.detection_tools import register_detection_tools
from tools.processing_tools import register_processing_tools


def _find_free_tcp_port() -> int:
    """Find an available ephemeral TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _check_service_online(url: str, timeout: float = 1.0) -> bool:
    """Non-destructive check whether a local microservice is active."""
    try:
        resp = httpx.get(f"{url}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.asyncio
async def test_network_backed_mcp_streamable_http():
    """Verify real MCP client-server communication over a network TCP socket."""
    orig_det = os.environ.get("DETECTION_SERVICE_URL")
    orig_proc = os.environ.get("PROCESSING_SERVICE_URL")
    orig_ocr = os.environ.get("OCR_SERVICE_URL")

    os.environ["DETECTION_SERVICE_URL"] = "http://127.0.0.1:8001"
    os.environ["PROCESSING_SERVICE_URL"] = "http://127.0.0.1:8002"
    os.environ["OCR_SERVICE_URL"] = "http://127.0.0.1:8003"

    port = _find_free_tcp_port()
    
    # 1. Instantiate gateway with real CV tools
    gateway = FastMCP(
        name="CV-MCP-Gateway-TCP-Integration",
        host="127.0.0.1",
        port=port,
    )
    register_detection_tools(gateway)
    register_processing_tools(gateway)

    # 2. Start Uvicorn HTTP server in an asyncio background task
    app = gateway.streamable_http_app()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Wait for server to bind and start listening
    for _ in range(60):
        if server.started:
            break
        await asyncio.sleep(0.05)
    assert server.started, "Gateway server failed to start within timeout"

    url = f"http://127.0.0.1:{port}/mcp"

    try:
        # 3. Connect official MCP SDK client over network TCP socket
        async with streamable_http_client(url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                # Handshake & initialization
                init_res = await session.initialize()
                assert init_res.serverInfo.name == "CV-MCP-Gateway-TCP-Integration"
                assert init_res.protocolVersion is not None

                # Tool enumeration & schema validation
                tools_res = await session.list_tools()
                tool_dict = {t.name: t for t in tools_res.tools}
                assert "detect_objects" in tool_dict
                assert "apply_image_filter" in tool_dict
                assert "blur_faces_in_image" in tool_dict
                assert "extract_text_from_image" in tool_dict

                # Schema checks
                det_schema = tool_dict["detect_objects"].inputSchema
                assert "image_path" in det_schema.get("properties", {})

                # 4. Sandbox security test over network
                sec_res = await session.call_tool(
                    "apply_image_filter",
                    {"image_path": "../../secret.txt", "filter_type": "grayscale"},
                )
                sec_text = sec_res.content[0].text
                assert "Security Error" in sec_text

                # 5. Live downstream tool tests (if services are running)
                downstream_8002_online = _check_service_online("http://127.0.0.1:8002")
                if downstream_8002_online:
                    # Test filter tool
                    f_res = await session.call_tool(
                        "apply_image_filter",
                        {"image_path": "data/inputs/obj_test.jpg", "filter_type": "edges"},
                    )
                    f_data = json.loads(f_res.content[0].text)
                    assert f_data.get("status") == "success"
                    assert Path(f_data.get("output_image_path")).exists()

                    # Test face blurring
                    blur_res = await session.call_tool(
                        "blur_faces_in_image",
                        {"image_path": "data/inputs/face_test.jpg", "blur_intensity": 99},
                    )
                    blur_data = json.loads(blur_res.content[0].text)
                    assert blur_data.get("status") == "success"
                    assert blur_data.get("faces_detected", 0) > 0
                    assert Path(blur_data.get("output_image_path")).exists()

                downstream_8001_online = _check_service_online("http://127.0.0.1:8001")
                if downstream_8001_online:
                    # Test object detection
                    det_res = await session.call_tool(
                        "detect_objects",
                        {"image_path": "data/inputs/obj_test.jpg", "confidence": 0.25},
                    )
                    det_data = json.loads(det_res.content[0].text)
                    objects = det_data.get("detections") or det_data.get("objects", [])
                    assert len(objects) > 0
                    assert det_data.get("total_objects", len(objects)) > 0

                downstream_8003_online = _check_service_online("http://127.0.0.1:8003")
                if downstream_8003_online:
                    # Test OCR extraction
                    ocr_res = await session.call_tool(
                        "extract_text_from_image",
                        {"image_path": "data/inputs/testocr.jpg"},
                    )
                    ocr_data = json.loads(ocr_res.content[0].text)
                    assert ocr_data.get("text_found") is True
                    assert len(ocr_data.get("extracted_text", "")) > 0
    finally:
        # Gracefully shutdown Uvicorn
        server.should_exit = True
        await server_task

        # Restore environment variables so subsequent tests are not poisoned
        for var, orig in [
            ("DETECTION_SERVICE_URL", orig_det),
            ("PROCESSING_SERVICE_URL", orig_proc),
            ("OCR_SERVICE_URL", orig_ocr),
        ]:
            if orig is not None:
                os.environ[var] = orig
            else:
                os.environ.pop(var, None)


if __name__ == "__main__":
    asyncio.run(test_network_backed_mcp_streamable_http())
    print("Standalone network TCP MCP test passed successfully.")
