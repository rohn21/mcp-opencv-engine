"""Environment and capabilities detection module."""

from core.environment.capabilities import (
    EnvironmentProfile,
    GPUBackend,
    OSFamily,
    RuntimeMode,
)
from core.environment.detector import (
    EnvironmentDetector,
    detect_environment,
)

__all__ = [
    "EnvironmentProfile",
    "GPUBackend",
    "OSFamily",
    "RuntimeMode",
    "EnvironmentDetector",
    "detect_environment",
]
