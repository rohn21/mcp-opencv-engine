"""Test-compatible re-export of canonical shared.image_utils."""

from shared.image_utils import (
    MAX_DIMENSION_PX,
    MAX_FILE_SIZE_MB,
    get_dimensions,
    validate_image,
)

__all__ = [
    "MAX_DIMENSION_PX",
    "MAX_FILE_SIZE_MB",
    "get_dimensions",
    "validate_image",
]
