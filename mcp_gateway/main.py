import os
import sys
import json

# Ensure mcp_gateway directory and project root are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP
from tools.detection_tools import register_detection_tools
from tools.processing_tools import register_processing_tools
from utils.assets_manager import list_available_images

mcp = FastMCP(
    name="CV-MCP-Gateway",
    host="0.0.0.0",
    port=8000,
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
    """Prompt template for agent-driven computer vision analysis."""

    return f"""
Analyze the following image using the available computer-vision tools.

Image:
{image_path}

Your goal is to produce an accurate, evidence-based analysis of the image.

Available tools:

1. detect_objects
   - Detect objects and people in the image.
   - Use this when identifying, counting, or locating objects is relevant.
   - Report detected object names, confidence scores, and bounding boxes.

2. extract_text_from_image
   - Extract visible text from the image using OCR.
   - Use this when the image contains documents, signs, labels,
     license plates, screens, or other potentially readable text.
   - Report the extracted text and relevant OCR confidence information.

3. apply_image_filter
   - Apply an OpenCV image filter when preprocessing or visual analysis
     would improve the requested task.
   - Available filters are:
       - grayscale
       - edges
       - blur
       - threshold
   - Do not invent other filter types.

4. blur_faces_in_image
   - Detect and anonymize human faces.
   - Use this when the user explicitly requests privacy protection,
     anonymization, or face blurring.

Tool-selection rules:

- Do not automatically call every tool.
- Select tools based on the user's request and the content of the image.
- Use multiple tools when their results complement each other.
- Use image preprocessing only when it is useful for the requested task.
- Use OCR when readable text is relevant.
- Use object detection when objects or people need to be identified.
- Use face blurring only when privacy/anonymization is requested.
- If a tool fails, continue with other relevant tools when possible
  and clearly report the failure.
- Never fabricate objects, text, faces, or other visual information.
- Base all conclusions on the actual results returned by the tools.

After completing the relevant analysis, provide the result in this format:

## Scene Summary
Brief description based only on the available computer-vision results.

## Detected Objects
List detected objects or people with confidence scores.
Include bounding-box information when useful.

## Extracted Text
List meaningful text extracted through OCR.
If no useful text was detected, state that clearly.

## Privacy
State whether face anonymization was requested or performed.
If face blurring was not requested, do not perform it.

## Processing Results
Describe any filters or image-processing operations performed
and provide the resulting output paths.

## Combined Observations
Summarize useful findings obtained by combining results from
multiple computer-vision tools.

## Limitations
Mention failed operations, uncertain detections, OCR limitations,
or information that could not be determined.

Do not invent information that is not supported by the computer-vision
tool results.
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MCP Gateway Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        help="Transport type (stdio, streamable-http, or sse). Defaults to stdio or MCP_TRANSPORT env.",
    )
    args, _ = parser.parse_known_args()
    mcp.run(transport=args.transport)
