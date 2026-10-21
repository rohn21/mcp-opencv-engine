from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    image_path: str
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
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
    detections: List[DetectedObject] = Field(default_factory=list)


class FilterResponse(BaseModel):
    status: str
    filter_applied: str
    output_image_path: str
    original_dimensions: Dict[str, Any] = Field(default_factory=dict)
    processed_dimensions: Dict[str, Any] = Field(default_factory=dict)


class FaceBlurRequest(BaseModel):
    image_path: str
    blur_intensity: int = 99


class FaceBlurResponse(BaseModel):
    status: str
    faces_detected: int
    output_image_path: str


class OCRRequest(BaseModel):
    image_path: str
    languages: List[str] = Field(default_factory=lambda: ["en"])


class TextBlock(BaseModel):
    text: str
    confidence: float
    bbox: Dict[str, Any] = Field(default_factory=dict)


class OCRResponse(BaseModel):
    status: str
    text_found: bool
    extracted_text: str
    text_blocks: List[TextBlock] = Field(default_factory=list)
