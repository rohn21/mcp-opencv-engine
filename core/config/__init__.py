"""Configuration management architecture for MCP OpenCV Engine."""

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
    ModelConfig,
    Profile,
    RuntimeMode,
    ServiceConfig,
    StorageConfig,
)

__all__ = [
    "AppConfig",
    "DeviceTarget",
    "MCPTransport",
    "ModelConfig",
    "Profile",
    "RuntimeMode",
    "ServiceConfig",
    "StorageConfig",
    "get_config",
    "get_project_root",
    "load_config",
    "parse_env_file",
    "save_config",
]
