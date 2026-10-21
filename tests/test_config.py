"""Unit tests for core.config module."""

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from core.config.loader import (
    get_config,
    get_project_root,
    load_config,
    parse_env_file,
    save_config,
)
from core.config.schema import (
    AppConfig,
    DeviceTarget,
    MCPTransport,
    Profile,
    RuntimeMode,
    ServiceConfig,
    StorageConfig,
)


def test_default_config(tmp_path: Path):
    empty_env = tmp_path / "empty.env"
    config = load_config(env_file=empty_env, use_system_env=False)
    assert isinstance(config, AppConfig)
    assert config.profile == Profile.DEV
    assert config.runtime_mode == RuntimeMode.AUTO
    assert config.device == DeviceTarget.AUTO

    # Service defaults
    assert config.service.gateway_port == 8000
    assert config.service.detection_service_url == "http://localhost:8001"
    assert config.service.processing_service_url == "http://localhost:8002"
    assert config.service.ocr_service_url == "http://localhost:8003"
    assert config.service.mcp_transport == MCPTransport.STDIO

    # Storage defaults
    assert len(config.storage.allowed_input_directories) >= 1
    assert config.storage.allowed_output_directory == config.storage.outputs_dir
    assert "./data" in config.storage.docker_mounts


def test_schema_url_validation():
    # Valid URLs
    s = ServiceConfig(detection_service_url="https://secure-vision.internal:8443")
    assert s.detection_service_url == "https://secure-vision.internal:8443"

    # Invalid URLs
    with pytest.raises(ValidationError):
        ServiceConfig(detection_service_url="ftp://invalid-scheme")

    with pytest.raises(ValidationError):
        ServiceConfig(detection_service_url="not-a-valid-url")

    with pytest.raises(ValidationError):
        ServiceConfig(detection_service_url="")


def test_schema_port_and_threshold_validation():
    with pytest.raises(ValidationError):
        ServiceConfig(gateway_port=70000)

    with pytest.raises(ValidationError):
        ServiceConfig(gateway_port=-1)

    with pytest.raises(ValidationError):
        AppConfig(model={"confidence_threshold": 1.5})  # type: ignore


def test_docker_profile_defaults(tmp_path: Path):
    empty_env = tmp_path / "empty.env"
    config = load_config(profile_override=Profile.DOCKER, env_file=empty_env, use_system_env=False)
    assert config.profile == Profile.DOCKER
    assert config.service.detection_service_url == "http://object-detection:8001"
    assert config.service.processing_service_url == "http://image-processing:8002"
    assert config.service.ocr_service_url == "http://ocr-service:8003"
    assert config.service.mcp_transport == MCPTransport.STREAMABLE_HTTP
    assert str(config.storage.data_dir).replace("\\", "/") == "/app/data"


def test_env_variable_overrides():
    env_mock = {
        "DETECTION_SERVICE_URL": "http://127.0.0.1:9001",
        "PROCESSING_SERVICE_URL": "http://127.0.0.1:9002",
        "OCR_SERVICE_URL": "http://127.0.0.1:9003",
        "MCP_GATEWAY_PORT": "8080",
        "MCP_TRANSPORT": "streamable-http",
        "DATA_DIR": "C:/custom_data",
        "ALLOWED_INPUT_DIRS": "C:/custom_data/inputs;D:/photos",
        "ALLOWED_OUTPUT_DIR": "C:/custom_data/outputs",
        "OCR_USE_GPU": "true",
        "OCR_LANGUAGES": "en,fr,de",
        "RUNTIME_MODE": "docker",
    }

    from core.config.loader import _build_env_overrides
    overrides = _build_env_overrides(env_mock)

    config = load_config(use_system_env=False, **overrides)
    assert config.service.detection_service_url == "http://127.0.0.1:9001"
    assert config.service.processing_service_url == "http://127.0.0.1:9002"
    assert config.service.ocr_service_url == "http://127.0.0.1:9003"
    assert config.service.gateway_port == 8080
    assert config.service.mcp_transport == MCPTransport.STREAMABLE_HTTP
    assert config.runtime_mode == RuntimeMode.DOCKER

    assert len(config.storage.allowed_input_directories) == 2
    assert Path("D:/photos") in config.storage.allowed_input_directories
    assert config.storage.allowed_output_directory == Path("C:/custom_data/outputs")

    assert config.model.ocr_use_gpu is True
    assert config.model.ocr_languages == ["en", "fr", "de"]


def test_parse_env_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Comment line\n"
        "DETECTION_PORT=8101\n"
        "MCP_TRANSPORT='sse'\n"
        "DATA_DIR=\"/opt/opencv_data\"\n",
        encoding="utf-8",
    )

    parsed = parse_env_file(env_file)
    assert parsed["DETECTION_PORT"] == "8101"
    assert parsed["MCP_TRANSPORT"] == "sse"
    assert parsed["DATA_DIR"] == "/opt/opencv_data"

    config = load_config(env_file=env_file, use_system_env=False)
    assert config.service.detection_port == 8101
    assert config.service.mcp_transport == MCPTransport.SSE
    assert str(config.storage.data_dir).replace("\\", "/") == "/opt/opencv_data"


def test_json_config_file_and_save(tmp_path: Path):
    cfg_file = tmp_path / "custom_config.json"
    custom_data = {
        "profile": "prod",
        "runtime_mode": "native",
        "service": {
            "gateway_port": 8888,
            "request_timeout": 45.0,
        },
        "model": {
            "yolo_model_name": "yolov8m.pt",
            "confidence_threshold": 0.4,
        },
    }
    cfg_file.write_text(json.dumps(custom_data), encoding="utf-8")

    config = load_config(config_path=cfg_file, use_system_env=False)
    assert config.profile == Profile.PROD
    assert config.runtime_mode == RuntimeMode.NATIVE
    assert config.service.gateway_port == 8888
    assert config.service.request_timeout == 45.0
    assert config.model.yolo_model_name == "yolov8m.pt"
    assert config.model.confidence_threshold == 0.4

    # Test save_config
    out_file = tmp_path / "saved_config.json"
    save_config(config, out_file)
    assert out_file.exists()

    reloaded = load_config(config_path=out_file, use_system_env=False)
    assert reloaded.service.gateway_port == 8888
    assert reloaded.model.yolo_model_name == "yolov8m.pt"


def test_get_config_singleton():
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2

    cfg3 = get_config(reload=True)
    assert isinstance(cfg3, AppConfig)


def test_get_project_root():
    root = get_project_root()
    assert (root / "pytest.ini").exists() or (root / "docker-compose.yml").exists()
