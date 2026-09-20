from pydantic import BaseModel
from typing import Optional



class DetectionRequest(BaseModel):
    image_path: str
    confidence_threshold: float = 0.25
    draw_boxes: bool = True

class ImageFilterRequest(BaseModel):
    image_path: str
    filter_type: str



class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class DetectedObject(BaseModel):
    class_name: str
    confidence: float
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    status: str
    total_objects: int
    annotated_image_path: Optional[str] = None
    detections: list[DetectedObject]

class FilterResponse(BaseModel):
    status: str
    filter_applied: str
    output_image_path: str
    original_dimensions: dict
    processed_dimensions: dict

class FaceBlurRequest(BaseModel):
    image_path: str
    blur_intensity: int = 99

class FaceBlurResponse(BaseModel):
    status: str
    faces_detected: int
    output_image_path: str

class OCRRequest(BaseModel):
    image_path: str
    languages: list[str] = ["en"]

class TextBlock(BaseModel):
    text: str
    confidence: float
    bbox: dict

class OCRResponse(BaseModel):
    status: str
    text_found: bool
    extracted_text: str
    text_blocks: list[TextBlock]