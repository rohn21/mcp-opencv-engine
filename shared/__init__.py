"""Shared utilities, data transfer schemas, and filesystem validations for MCP OpenCV services."""

from shared.schemas import (
    BoundingBox,
    DetectedObject,
    DetectionRequest,
    DetectionResponse,
    FaceBlurRequest,
    FaceBlurResponse,
    FilterResponse,
    ImageFilterRequest,
    OCRRequest,
    OCRResponse,
    TextBlock,
)

__all__ = [
    "BoundingBox",
    "DetectedObject",
    "DetectionRequest",
    "DetectionResponse",
    "FaceBlurRequest",
    "FaceBlurResponse",
    "FilterResponse",
    "ImageFilterRequest",
    "OCRRequest",
    "OCRResponse",
    "TextBlock",
]
