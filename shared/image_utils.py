import cv2
import numpy as np
from pathlib import Path

MAX_FILE_SIZE_MB = 20
MAX_DIMENSION_PX = 4000

def validate_image(image_path: str, allowed_base_dir: str) -> tuple[bool, str]:
    """Checks if an image file exists and is readable by OpenCV."""
    path = Path(image_path).resolve()

    if allowed_base_dir:
        base_dir = Path(allowed_base_dir).resolve()
        if not str(path).startswith(str(base_dir)):
            return False, "Security Error: Access denied to path outside allowed directory."

    if not path.exists():
        return False, f"File not found: {image_path}"
    if path.stat().st_size == 0:
        return False, "File is empty."
    
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({file_size_mb:.1f}MB). Max allowed is {MAX_FILE_SIZE_MB}MB."

    return True, "OK"

def get_dimensions(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not decode image")
    if len(img.shape) == 2:
        h, w = img.shape
        return {"width": w, "height": h, "channels": 1}
    h, w, c = img.shape
    return {"width": w, "height": h, "channels": c}
