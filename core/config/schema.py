"""Configuration schemas for MCP OpenCV Engine using Pydantic v2."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Profile(str, Enum):
    DEV = "dev"
    PROD = "prod"
    TEST = "test"
    DOCKER = "docker"
    NATIVE = "native"


class RuntimeMode(str, Enum):
    AUTO = "auto"
    NATIVE = "native"
    DOCKER = "docker"


class DeviceTarget(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class MCPTransport(str, Enum):
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable-http"
    SSE = "sse"


def _validate_http_url(url: str, field_name: str) -> str:
    """Validate that the given string is a valid HTTP/HTTPS URL."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"{field_name} must use http or https scheme, got: '{parsed.scheme}'")
    if not parsed.netloc:
        raise ValueError(f"{field_name} must contain a valid host/network location.")
    return url.strip().rstrip("/")


class ServiceConfig(BaseModel):
    """Configuration for microservice endpoints and MCP transport."""
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    host: str = "0.0.0.0"
    gateway_port: int = Field(default=8000, ge=1, le=65535)
    detection_port: int = Field(default=8001, ge=1, le=65535)
    processing_port: int = Field(default=8002, ge=1, le=65535)
    ocr_port: int = Field(default=8003, ge=1, le=65535)

    detection_service_url: str = "http://localhost:8001"
    processing_service_url: str = "http://localhost:8002"
    ocr_service_url: str = "http://localhost:8003"

    mcp_transport: MCPTransport = MCPTransport.STDIO
    request_timeout: float = Field(default=30.0, gt=0.0)

    @field_validator("detection_service_url")
    @classmethod
    def validate_detection_url(cls, v: str) -> str:
        return _validate_http_url(v, "detection_service_url")

    @field_validator("processing_service_url")
    @classmethod
    def validate_processing_url(cls, v: str) -> str:
        return _validate_http_url(v, "processing_service_url")

    @field_validator("ocr_service_url")
    @classmethod
    def validate_ocr_url(cls, v: str) -> str:
        return _validate_http_url(v, "ocr_service_url")


class StorageConfig(BaseModel):
    """Configuration for directory boundaries, path translation, and sandboxing."""
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    data_dir: Path = Field(default_factory=lambda: Path("./data"))
    inputs_dir: Path = Field(default_factory=lambda: Path("./data/inputs"))
    outputs_dir: Path = Field(default_factory=lambda: Path("./data/outputs"))

    allowed_input_directories: List[Path] = Field(default_factory=list)
    allowed_output_directory: Optional[Path] = None
    protected_directories: List[Path] = Field(default_factory=list)

    # Host-to-container mount mappings, e.g. {"C:\\...\\data": "/app/data"}
    docker_mounts: Dict[str, str] = Field(
        default_factory=lambda: {"./data": "/app/data"}
    )

    @model_validator(mode="after")
    def populate_defaults_and_resolve(self) -> StorageConfig:
        # Default allowed input directories to [inputs_dir] if empty
        if not self.allowed_input_directories:
            self.allowed_input_directories = [self.inputs_dir]

        # Default allowed output directory to outputs_dir if None
        if self.allowed_output_directory is None:
            self.allowed_output_directory = self.outputs_dir

        return self


class ModelConfig(BaseModel):
    """Configuration for AI models and vision engines."""
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    yolo_model_name: str = "yolov8n.pt"
    yolo_weights_dir: Optional[Path] = None
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    ocr_engine: Literal["easyocr", "tesseract"] = "easyocr"
    ocr_languages: List[str] = Field(default_factory=lambda: ["en"])
    ocr_use_gpu: bool = False


class AppConfig(BaseModel):
    """Top-level application configuration profile."""
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    profile: Profile = Profile.DEV
    runtime_mode: RuntimeMode = RuntimeMode.AUTO
    device: DeviceTarget = DeviceTarget.AUTO

    service: ServiceConfig = Field(default_factory=ServiceConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
