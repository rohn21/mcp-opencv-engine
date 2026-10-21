"""Configuration loader and environment manager for MCP OpenCV Engine."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config.schema import (
    AppConfig,
    DeviceTarget,
    MCPTransport,
    ModelConfig,
    Profile,
    RuntimeMode,
    ServiceConfig,
    StorageConfig,
)


_CACHED_CONFIG: Optional[AppConfig] = None


def get_project_root() -> Path:
    """Discover the canonical project root directory."""
    # Start at directory containing core/
    candidate = Path(__file__).resolve().parent.parent.parent
    if (candidate / "docker-compose.yml").exists() or (candidate / "requirements.txt").exists():
        return candidate

    # Fallback to current working directory if it contains markers
    cwd = Path.cwd().resolve()
    if (cwd / "docker-compose.yml").exists() or (cwd / "requirements.txt").exists():
        return cwd

    return candidate


def parse_env_file(env_path: Path) -> Dict[str, str]:
    """Parse a simple .env file into key-value pairs without requiring third-party libraries."""
    env_vars: Dict[str, str] = {}
    if not env_path.is_file():
        return env_vars

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key:
                        env_vars[key] = val
    except Exception:
        pass
    return env_vars


def _split_path_list(raw: str) -> List[Path]:
    """Parse comma- or semicolon-separated path strings into a list of Path objects."""
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    return [Path(p) for p in parts]


def _build_env_overrides(env_source: Dict[str, str]) -> Dict[str, Any]:
    """Extract and structure engine configuration from environment variables."""
    overrides: Dict[str, Any] = {
        "service": {},
        "storage": {},
        "model": {},
    }

    # Top-level attributes
    if "MCP_PROFILE" in env_source:
        try:
            overrides["profile"] = Profile(env_source["MCP_PROFILE"].lower())
        except ValueError:
            pass

    if "RUNTIME_MODE" in env_source:
        try:
            overrides["runtime_mode"] = RuntimeMode(env_source["RUNTIME_MODE"].lower())
        except ValueError:
            pass

    if "DEVICE" in env_source or "DEVICE_TARGET" in env_source:
        dev_val = env_source.get("DEVICE") or env_source.get("DEVICE_TARGET")
        try:
            overrides["device"] = DeviceTarget(dev_val.lower())  # type: ignore
        except ValueError:
            pass

    # Service overrides
    if "DETECTION_SERVICE_URL" in env_source:
        overrides["service"]["detection_service_url"] = env_source["DETECTION_SERVICE_URL"]
    if "PROCESSING_SERVICE_URL" in env_source:
        overrides["service"]["processing_service_url"] = env_source["PROCESSING_SERVICE_URL"]
    if "OCR_SERVICE_URL" in env_source:
        overrides["service"]["ocr_service_url"] = env_source["OCR_SERVICE_URL"]

    if "MCP_GATEWAY_PORT" in env_source:
        try:
            overrides["service"]["gateway_port"] = int(env_source["MCP_GATEWAY_PORT"])
        except ValueError:
            pass
    if "DETECTION_PORT" in env_source:
        try:
            overrides["service"]["detection_port"] = int(env_source["DETECTION_PORT"])
        except ValueError:
            pass
    if "PROCESSING_PORT" in env_source:
        try:
            overrides["service"]["processing_port"] = int(env_source["PROCESSING_PORT"])
        except ValueError:
            pass
    if "OCR_PORT" in env_source:
        try:
            overrides["service"]["ocr_port"] = int(env_source["OCR_PORT"])
        except ValueError:
            pass

    if "MCP_TRANSPORT" in env_source:
        try:
            overrides["service"]["mcp_transport"] = MCPTransport(env_source["MCP_TRANSPORT"].lower())
        except ValueError:
            pass

    if "SERVICE_TIMEOUT" in env_source:
        try:
            overrides["service"]["request_timeout"] = float(env_source["SERVICE_TIMEOUT"])
        except ValueError:
            pass

    # Storage overrides
    if "DATA_DIR" in env_source:
        overrides["storage"]["data_dir"] = Path(env_source["DATA_DIR"])
        overrides["storage"]["inputs_dir"] = Path(env_source["DATA_DIR"]) / "inputs"
        overrides["storage"]["outputs_dir"] = Path(env_source["DATA_DIR"]) / "outputs"

    if "INPUTS_DIR" in env_source:
        overrides["storage"]["inputs_dir"] = Path(env_source["INPUTS_DIR"])

    if "OUTPUT_DIR" in env_source or "OUTPUTS_DIR" in env_source:
        out_val = env_source.get("OUTPUT_DIR") or env_source.get("OUTPUTS_DIR")
        overrides["storage"]["outputs_dir"] = Path(out_val)  # type: ignore

    if "ALLOWED_INPUT_DIRS" in env_source:
        overrides["storage"]["allowed_input_directories"] = _split_path_list(env_source["ALLOWED_INPUT_DIRS"])

    if "ALLOWED_OUTPUT_DIR" in env_source:
        overrides["storage"]["allowed_output_directory"] = Path(env_source["ALLOWED_OUTPUT_DIR"])

    # Model overrides
    if "YOLO_MODEL" in env_source or "YOLO_MODEL_NAME" in env_source:
        overrides["model"]["yolo_model_name"] = env_source.get("YOLO_MODEL") or env_source.get("YOLO_MODEL_NAME")
    if "OCR_ENGINE" in env_source:
        overrides["model"]["ocr_engine"] = env_source["OCR_ENGINE"].lower()
    if "OCR_USE_GPU" in env_source:
        overrides["model"]["ocr_use_gpu"] = env_source["OCR_USE_GPU"].lower() in ("true", "1", "yes")
    if "OCR_LANGUAGES" in env_source:
        overrides["model"]["ocr_languages"] = [
            lang.strip() for lang in env_source["OCR_LANGUAGES"].split(",") if lang.strip()
        ]

    # Clean empty dictionaries
    for k in list(overrides.keys()):
        if isinstance(overrides[k], dict) and not overrides[k]:
            del overrides[k]

    return overrides


def _apply_profile_defaults(data: Dict[str, Any], profile: Profile) -> None:
    """Apply default service and storage URLs specific to a runtime profile."""
    if profile == Profile.DOCKER:
        service_data = data.setdefault("service", {})
        service_data.setdefault("detection_service_url", "http://object-detection:8001")
        service_data.setdefault("processing_service_url", "http://image-processing:8002")
        service_data.setdefault("ocr_service_url", "http://ocr-service:8003")
        service_data.setdefault("mcp_transport", MCPTransport.STREAMABLE_HTTP)

        storage_data = data.setdefault("storage", {})
        storage_data.setdefault("data_dir", Path("/app/data"))
        storage_data.setdefault("inputs_dir", Path("/app/data/inputs"))
        storage_data.setdefault("outputs_dir", Path("/app/data/outputs"))


def load_config(
    config_path: Optional[Path | str] = None,
    profile_override: Optional[Profile | str] = None,
    runtime_mode_override: Optional[RuntimeMode | str] = None,
    env_file: Optional[Path | str] = None,
    use_system_env: bool = True,
    **explicit_overrides: Any,
) -> AppConfig:
    """Load configuration by merging schema defaults, config files, .env, and environment variables.
    
    Precedence (highest to lowest):
    1. Explicit function arguments (`explicit_overrides`, `profile_override`, `runtime_mode_override`)
    2. OS Environment variables (if `use_system_env=True`)
    3. `.env` file variables
    4. Configuration file (`config.json` or provided path)
    5. Schema and profile defaults
    """
    root = get_project_root()
    merged: Dict[str, Any] = {}

    # 1. Check for configuration file
    found_config_path: Optional[Path] = None
    if config_path:
        found_config_path = Path(config_path)
    else:
        for fname in ("config.json", "mcp_config.json"):
            candidate = root / fname
            if candidate.is_file():
                found_config_path = candidate
                break

    if found_config_path and found_config_path.is_file():
        try:
            with open(found_config_path, "r", encoding="utf-8") as f:
                merged = json.load(f)
        except Exception as exc:
            raise ValueError(f"Failed to parse config file at {found_config_path}: {exc}") from exc

    # 2. Check for .env file
    env_vars: Dict[str, str] = {}
    resolved_env_file = Path(env_file) if env_file else (root / ".env")
    if resolved_env_file.is_file():
        env_vars.update(parse_env_file(resolved_env_file))

    # 3. Layer OS environment variables
    if use_system_env:
        env_vars.update(dict(os.environ))

    env_overrides = _build_env_overrides(env_vars)

    # Merge env_overrides into merged dict
    for key, val in env_overrides.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key].update(val)
        else:
            merged[key] = val

    # 4. Apply explicit profile override if present
    target_profile: Optional[Profile] = None
    if profile_override:
        if isinstance(profile_override, Profile):
            target_profile = profile_override
        else:
            target_profile = Profile(str(profile_override).lower())
        merged["profile"] = target_profile
    elif "profile" in merged:
        try:
            val = merged["profile"]
            target_profile = val if isinstance(val, Profile) else Profile(str(val).lower())
        except ValueError:
            target_profile = Profile.DEV

    if target_profile:
        _apply_profile_defaults(merged, target_profile)

    # 5. Apply runtime mode override
    if runtime_mode_override:
        if isinstance(runtime_mode_override, RuntimeMode):
            merged["runtime_mode"] = runtime_mode_override
        else:
            merged["runtime_mode"] = RuntimeMode(str(runtime_mode_override).lower())

    # 6. Apply explicit overrides
    for k, v in explicit_overrides.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k].update(v)
        else:
            merged[k] = v

    return AppConfig.model_validate(merged)


def get_config(reload: bool = False) -> AppConfig:
    """Get or initialize the global singleton AppConfig instance."""
    global _CACHED_CONFIG
    if _CACHED_CONFIG is None or reload:
        _CACHED_CONFIG = load_config()
    return _CACHED_CONFIG


def save_config(config: AppConfig, target_path: Path | str) -> None:
    """Serialize and save configuration to a JSON file."""
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(config.model_dump_json(indent=2))
