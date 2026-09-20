import os
import json
from mcp.server.fastmcp import FastMCP
from tools.detection_tools import register_detection_tools
from tools.processing_tools import register_processing_tools
from utils.assets_manager import list_available_images

mcp = FastMCP(
    name="CV-MCP-Gateway",
    version="1.0.0",
)

register_detection_tools(mcp)
register_processing_tools(mcp)

@mcp.resource("images://available")
def get_available_images() -> str:
    """Lists all images currently available in the inputs directory."""
    images = list_available_images()
    if not images:
        return "No images found in the inputs directory."
    return json.dumps({"count": len(images), "images": images}, indent=2)


@mcp.prompt()
def full_image_analysis(image_path: str) -> str:
    """A prompt template for comprehensive image analysis."""
    return f"""
    Please perform a full analysis of the image at: {image_path}
    
    Step 1: Use the 'apply_image_filter' tool with filter_type='edges' to see edges.
    Step 2: Use the 'detect_objects' tool to identify all objects.
    Step 3: Summarize your findings in a structured report including:
       - What objects were detected and their confidence levels
       - General scene description based on detected objects
       - Location of the annotated output image
    """

if __name__ == "__main__":
    mcp.run()
