"""Client adapter registry and configuration orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

from core.client_config.antigravity import AntigravityAdapter
from core.client_config.base import BaseClientAdapter, ConfigScope, ServerConfigEntry
from core.client_config.claude import ClaudeDesktopAdapter
from core.client_config.cursor import CursorAdapter
from core.client_config.merger import apply_json_merge
from core.config.loader import get_config, get_project_root
from core.config.schema import AppConfig, MCPTransport


ADAPTER_CLASSES: Dict[str, Type[BaseClientAdapter]] = {
    "antigravity": AntigravityAdapter,
    "cursor": CursorAdapter,
    "claude": ClaudeDesktopAdapter,
}


def get_adapter(
    client_id: str,
    config: Optional[AppConfig] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> BaseClientAdapter:
    """Instantiate and return adapter for the requested client ID."""
    clean_id = client_id.lower().strip()
    cls = ADAPTER_CLASSES.get(clean_id)
    if not cls:
        valid_options = list(ADAPTER_CLASSES.keys())
        raise ValueError(f"Unknown MCP client: '{client_id}'. Supported clients: {valid_options}")
    return cls(config=config, project_root=project_root)


def list_adapters(
    config: Optional[AppConfig] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> List[BaseClientAdapter]:
    """List all registered MCP client adapters."""
    return [
        cls(config=config, project_root=project_root)
        for cls in ADAPTER_CLASSES.values()
    ]


def configure_client(
    client_id: str,
    transport: Union[MCPTransport, str] = MCPTransport.STREAMABLE_HTTP,
    scope: Union[ConfigScope, str] = ConfigScope.WORKSPACE,
    server_name: str = "cv-mcp-gateway",
    custom_config_path: Optional[Union[str, Path]] = None,
    server_url: Optional[str] = None,
    python_path: Optional[Union[str, Path]] = None,
    dry_run: bool = True,
    config: Optional[AppConfig] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Optional[Path], str]:
    """Generate, preview, or apply MCP server configuration to a client.

    Args:
        client_id: Client identifier ('antigravity', 'cursor', 'claude').
        transport: Transport protocol ('streamable-http', 'stdio', 'sse').
        scope: Target scope ('workspace' or 'global').
        server_name: MCP server name key.
        custom_config_path: Optional explicit configuration file path.
        server_url: Optional custom server URL for remote transports.
        python_path: Optional explicit python executable for stdio transport.
        dry_run: If True, only generates a preview without writing.
        config: Application configuration.
        project_root: Project root directory.

    Returns:
        (target_config_path, backup_path_or_none, preview_string)
    """
    transport_enum = (
        transport if isinstance(transport, MCPTransport) else MCPTransport(str(transport).lower())
    )
    scope_enum = (
        scope if isinstance(scope, ConfigScope) else ConfigScope(str(scope).lower())
    )

    adapter = get_adapter(client_id, config=config, project_root=project_root)

    if not adapter.supports_transport(transport_enum):
        raise ValueError(
            f"Client '{adapter.display_name}' does not support transport '{transport_enum.value}'. "
            f"Supported transports: {[t.value for t in adapter.supported_transports]}"
        )

    if not adapter.supports_scope(scope_enum):
        raise ValueError(
            f"Client '{adapter.display_name}' does not support scope '{scope_enum.value}'. "
            f"Supported scopes: {[s.value for s in adapter.supported_scopes]}"
        )

    target_path = adapter.discover_config_path(scope=scope_enum, custom_path=custom_config_path)

    entry = adapter.generate_entry(
        transport=transport_enum,
        server_name=server_name,
        server_url=server_url,
        python_path=python_path,
    )

    payload = adapter.format_client_json(entry)

    merged, preview, backup_path = apply_json_merge(
        target_file=target_path,
        server_name=server_name,
        server_payload=payload,
        dry_run=dry_run,
    )

    return target_path, backup_path, preview
