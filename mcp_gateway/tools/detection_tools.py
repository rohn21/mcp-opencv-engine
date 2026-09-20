import os
import json
import httpx
from mcp.server.fastmcp import FastMCP


DETECTION_URL = os.getenv("DETECTION_SERVICE_URL", "http://object-detection:8001")
OCR_URL = os.getenv("OCR_SERVICE_URL", "http://172.17.0.1:8003")

def register_detection_tools(mcp: FastMCP):
    
    @mcp.tool()
    async def detect_objects(image_path: str, confidence: float = 0.25) -> str:
        """
        Detects and identifies objects in an image using the YOLOv8 AI model.
        
        USE CASE: Use this when the user wants to know what objects are in an image, 
        count specific items, or get the exact coordinates (bounding boxes) of objects.
        
        PARAMETERS:
        - image_path (str): The absolute path to the image file.
        - confidence (float): The minimum confidence score (0.0 to 1.0) required for an object 
          to be detected. Lower values (e.g., 0.25) detect more objects but may include false positives. 
          Higher values (e.g., 0.75) are stricter and only return highly confident detections.
          
        RETURNS: JSON with detected object names, confidence scores, bounding box coordinates, 
        and the path to the annotated image (with boxes drawn).
        """
        async with httpx.AsyncClient(timeout=60.0) as client:

            health = await client.get(f"{DETECTION_URL}/health")
            if health.status_code == 503:
                return "Detection service is still starting up. Please try again in 10 seconds."
            resp = await client.post(f"{DETECTION_URL}/predict", json={
                "image_path": image_path,
                "confidence_threshold": confidence,
                "draw_boxes": True,
            })
        
        if resp.status_code != 200:
            return f"Error: Detection service returned {resp.status_code} - {resp.text}"
        
        return json.dumps(resp.json(), indent=2)

    @mcp.tool()
    async def extract_text_from_image(image_path: str) -> str:
        """
        Extracts all visible text from an image using Optical Character Recognition (OCR).
        
        USE CASE: Use this when the user wants to read text, signs, documents, license plates, 
        or any written words visible inside an image.
        
        PARAMETERS:
        - image_path (str): The absolute path to the image file.
        
        NOTE: OCR processing is computationally heavy and may take 5-15 seconds for large images.
        
        RETURNS: JSON containing the full concatenated extracted text, and a list of individual 
        text blocks with their specific bounding box coordinates and confidence scores.
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{OCR_URL}/extract_text", json={
                "image_path": image_path,
                "languages": ["en"]
            })

        if resp.status_code != 200:
            return f"Error: OCR service returned {resp.status_code} - {resp.text}"

        return json.dumps(resp.json(), indent=2)