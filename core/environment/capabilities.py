"""Data structures representing system capabilities and environmental profile."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional


OSFamily = Literal["windows", "linux", "darwin"]
GPUBackend = Literal["cuda", "mps", "cpu"]
RuntimeMode = Literal["docker", "native"]


@dataclass(frozen=True)
class EnvironmentProfile:
    """Immutable snapshot of the host operating system, hardware, and runtime capabilities.
    
    Provides honest disclosure of acceleration options and recommended execution modes
    without mutating system state.
    """
    os_family: OSFamily
    os_name: str
    os_version: str
    cpu_arch: str
    python_version: str
    python_executable: str
    has_docker: bool
    docker_version: Optional[str] = None
    docker_compose_available: bool = False
    docker_compose_version: Optional[str] = None
    gpu_backend: GPUBackend = "cpu"
    gpu_name: Optional[str] = None
    gpu_count: int = 0
    tesseract_installed: bool = False
    tesseract_version: Optional[str] = None
    tesseract_path: Optional[str] = None
    total_memory_gb: Optional[float] = None
    recommended_runtime: RuntimeMode = "native"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_windows(self) -> bool:
        """True if running on Windows."""
        return self.os_family == "windows"

    @property
    def is_linux(self) -> bool:
        """True if running on Linux."""
        return self.os_family == "linux"

    @property
    def is_darwin(self) -> bool:
        """True if running on macOS (Darwin)."""
        return self.os_family == "darwin"

    @property
    def has_gpu_acceleration(self) -> bool:
        """True if hardware acceleration (CUDA or Apple Silicon MPS) is available."""
        return self.gpu_backend in ("cuda", "mps")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the environment profile to a dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the profile to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Generate a concise, human-readable summary of the environment."""
        docker_status = "Available" if self.has_docker else "Not detected"
        if self.has_docker and self.docker_version:
            docker_status += f" ({self.docker_version})"
        if self.docker_compose_available:
            docker_status += f" [Compose: {self.docker_compose_version or 'yes'}]"

        gpu_status = f"{self.gpu_backend.upper()}"
        if self.gpu_name:
            gpu_status += f" ({self.gpu_name}, count: {self.gpu_count})"

        tess_status = "Available" if self.tesseract_installed else "Not detected"
        if self.tesseract_version:
            tess_status += f" ({self.tesseract_version})"

        mem_str = f"{self.total_memory_gb:.1f} GB" if self.total_memory_gb is not None else "Unknown"

        lines = [
            "=== MCP OpenCV Engine Environment Profile ===",
            f"OS:                  {self.os_name} ({self.os_family}) {self.os_version}",
            f"Architecture:        {self.cpu_arch}",
            f"Python:              {self.python_version} ({self.python_executable})",
            f"Total Memory:        {mem_str}",
            f"Hardware Backend:    {gpu_status}",
            f"Tesseract OCR:       {tess_status}",
            f"Docker Support:      {docker_status}",
            f"Recommended Runtime: {self.recommended_runtime.upper()}",
            "==============================================",
        ]
        return "\n".join(lines)
