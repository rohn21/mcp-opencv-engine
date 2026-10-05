import json
import pytest
from tools.detection_tools import register_detection_tools

@pytest.mark.asyncio
async def test_extract_text_success(mock_mcp, httpx_mock):
    register_detection_tools(mock_mcp)
    
    mock_data = {"text": "Hello World", "text_blocks": []}
    httpx_mock.add_response(url="http://ocr-service:8003/extract_text", json=mock_data)
    
    result = await mock_mcp.tool_extract_text_from_image(image_path="/test/document.jpg")
    
    result_data = json.loads(result)
    assert result_data["text"] == "Hello World"

@pytest.mark.asyncio
async def test_extract_text_error(mock_mcp, httpx_mock):
    register_detection_tools(mock_mcp)
    
    httpx_mock.add_response(url="http://ocr-service:8003/extract_text", status_code=400, text="Bad Request")
    
    result = await mock_mcp.tool_extract_text_from_image(image_path="/test/document.jpg")
    assert "Error: OCR service returned 400" in result
