import os
import json
import httpx
from mcp.server.fastmcp import FastMCP

PROCESSING_URL = os.getenv("PROCESSING_SERVICE_URL", "http://image-processing:8002")


def register_processing_tools(mcp: FastMCP):

    @mcp.tool()
    async def apply_image_filter(image_path: str, filter_type: str) -> str:
        """
        Applies a basic computer vision filter to an image using OpenCV.
        
        USE CASE: Use this when the user wants to change the visual style of an image, 
        extract edges for analysis, or prepare an image for further processing.
        
        PARAMETERS:
        - image_path (str): The absolute path to the image file.
        - filter_type (str): STRICTLY one of the following 4 options. Do not invent other names:
            1. 'grayscale': Converts the image to black and white.
            2. 'edges': Detects and highlights the edges of objects (Canny edge detection).
            3. 'blur': Applies a Gaussian blur to soften the image.
            4. 'threshold': Converts the image to pure black and white based on a brightness cutoff.
            
        RETURNS: JSON containing the output image path and dimension changes.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{PROCESSING_URL}/filter", json={
                "image_path": image_path,
                "filter_type": filter_type,
            })
        
        if resp.status_code != 200:
            return f"Error: Processing service returned {resp.status_code} - {resp.text}"
        
        data = resp.json()
        return json.dumps(data, indent=2)

    @mcp.tool()
    async def blur_faces_in_image(image_path: str, blur_intensity: int = 99) -> str:
        """
        Detects human faces in an image and applies a heavy Gaussian blur to anonymize them.
        
        USE CASE: Use this when the user wants to protect privacy, anonymize bystanders, 
        or redact identities from a photograph before sharing or analyzing it.
        
        PARAMETERS:
        - image_path (str): The absolute path to the image file.
        - blur_intensity (int): The size of the blur kernel. MUST be an odd number (e.g., 11, 51, 99). 
          Higher numbers mean a heavier, more unrecognizable blur. Default is 99 (very heavy).
          If an even number is provided, the system will automatically add 1 to make it odd.
          
        RETURNS: JSON containing the path to the anonymized image and the number of faces found.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{PROCESSING_URL}/blur_faces", json={
                "image_path": image_path,
                "blur_intensity": blur_intensity,
            })

        if resp.status_code != 200:
            return f"Error: Processing service returned {resp.status_code} - {resp.text}"

        return json.dumps(resp.json(), indent=2)
