import json
import sys
from unittest.mock import MagicMock, patch

# Mock mcp.server.fastmcp completely so importing main.py doesn't start anything
class MockFastMCP:
    def __init__(self, *args, **kwargs): pass
    def prompt(self, *args, **kwargs): return lambda f: f
    def resource(self, *args, **kwargs): return lambda f: f
    def tool(self, *args, **kwargs): return lambda f: f
    def run(self, *args, **kwargs): pass

mock_fastmcp_module = MagicMock()
mock_fastmcp_module.FastMCP = MockFastMCP
sys.modules['mcp.server.fastmcp'] = mock_fastmcp_module

import main

def test_full_image_analysis_prompt():
    """Verify the prompt template injects variables correctly."""
    result = main.full_image_analysis("test_image.jpg")
    assert "test_image.jpg" in result
    assert "detect_objects" in result
    assert "blur_faces_in_image" in result

@patch("main.list_available_images")
def test_get_available_images_success(mock_list):
    """Verify the resource JSON output when images exist."""
    mock_list.return_value = ["img1.jpg", "img2.jpg"]
    result = main.get_available_images()
    data = json.loads(result)
    
    assert data["count"] == 2
    assert "img1.jpg" in data["images"]

@patch("main.list_available_images")
def test_get_available_images_empty(mock_list):
    """Verify the fallback string when no images exist."""
    mock_list.return_value = []
    result = main.get_available_images()
    
    assert "No images found" in result
