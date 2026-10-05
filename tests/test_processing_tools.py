import json
import pytest
from tools.processing_tools import register_processing_tools

@pytest.mark.asyncio
async def test_apply_image_filter_success(mock_mcp, httpx_mock):
    register_processing_tools(mock_mcp)
    
    mock_data = {"output_path": "/outputs/filtered.jpg", "filter_applied": "edges"}
    httpx_mock.add_response(url="http://image-processing:8002/filter", json=mock_data)
    
    result = await mock_mcp.tool_apply_image_filter(image_path="/test/image.jpg", filter_type="edges")
    
    result_data = json.loads(result)
    assert result_data["output_path"] == "/outputs/filtered.jpg"

@pytest.mark.asyncio
async def test_blur_faces_success(mock_mcp, httpx_mock):
    register_processing_tools(mock_mcp)
    
    mock_data = {"output_path": "/outputs/blurred.jpg", "faces_found": 3}
    httpx_mock.add_response(url="http://image-processing:8002/blur_faces", json=mock_data)
    
    result = await mock_mcp.tool_blur_faces_in_image(image_path="/test/people.jpg", blur_intensity=99)
    
    result_data = json.loads(result)
    assert result_data["faces_found"] == 3

@pytest.mark.asyncio
async def test_processing_error(mock_mcp, httpx_mock):
    register_processing_tools(mock_mcp)
    
    httpx_mock.add_response(url="http://image-processing:8002/blur_faces", status_code=500, text="Server Fault")
    
    result = await mock_mcp.tool_blur_faces_in_image(image_path="/test/people.jpg")
    assert "Error: Processing service returned 500" in result
