"""MCP client configuration manager module."""

from core.client_config.antigravity import AntigravityAdapter
from core.client_config.base import (
    BaseClientAdapter,
    ConfigScope,
    ServerConfigEntry,
)
from core.client_config.claude import ClaudeDesktopAdapter
from core.client_config.cursor import CursorAdapter
from core.client_config.merger import (
    apply_json_merge,
    create_backup,
    load_json_safe,
    merge_server_entry,
)
from core.client_config.registry import (
    configure_client,
    get_adapter,
    list_adapters,
)

__all__ = [
    "AntigravityAdapter",
    "BaseClientAdapter",
    "ClaudeDesktopAdapter",
    "ConfigScope",
    "CursorAdapter",
    "ServerConfigEntry",
    "apply_json_merge",
    "configure_client",
    "create_backup",
    "get_adapter",
    "list_adapters",
    "load_json_safe",
    "merge_server_entry",
]
