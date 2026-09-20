import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"

def list_available_images() -> list[str]:
    """Returns paths of all images in the inputs directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = []
    if INPUTS_DIR.exists():
        for f in INPUTS_DIR.iterdir():
            if f.suffix.lower() in extensions:
                images.append(str(f))
    return images
