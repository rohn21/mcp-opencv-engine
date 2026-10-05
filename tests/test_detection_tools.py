import json
import pytest
from tools.detection_tools import register_detection_tools

@pytest.mark.asyncio
async def test_detect_objects_success(mock_mcp, httpx_mock):
    # Register the tools to extract the function onto mock_mcp
    register_detection_tools(mock_mcp)
    
    # Mock the health check and prediction endpoints
    httpx_mock.add_response(url="http://object-detection:8001/health", status_code=200)
    mock_data = {"objects": [{"name": "person", "confidence": 0.99}], "annotated_image": "/path/to/img.jpg"}
    httpx_mock.add_response(url="http://object-detection:8001/predict", json=mock_data)
    
    # Execute the tool directly
    result = await mock_mcp.tool_detect_objects(image_path="/test/image.jpg", confidence=0.5)
    
    # Verify results
    result_data = json.loads(result)
    assert len(result_data["objects"]) == 1
    assert result_data["objects"][0]["name"] == "person"

@pytest.mark.asyncio
async def test_detect_objects_service_starting(mock_mcp, httpx_mock):
    register_detection_tools(mock_mcp)
    
    # Mock a 503 from the health check
    httpx_mock.add_response(url="http://object-detection:8001/health", status_code=503)
    
    result = await mock_mcp.tool_detect_objects(image_path="/test/image.jpg")
    assert "still starting up" in result

@pytest.mark.asyncio
async def test_detect_objects_service_error(mock_mcp, httpx_mock):
    register_detection_tools(mock_mcp)
    
    httpx_mock.add_response(url="http://object-detection:8001/health", status_code=200)
    httpx_mock.add_response(url="http://object-detection:8001/predict", status_code=500, text="Internal Server Error")
    
    result = await mock_mcp.tool_detect_objects(image_path="/test/image.jpg")
    assert "Error: Detection service returned 500" in result
