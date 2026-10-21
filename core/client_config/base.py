"""Base abstractions and data structures for MCP client configuration management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.config.loader import get_config, get_project_root
from core.config.schema import AppConfig, MCPTransport


class ConfigScope(str, Enum):
    WORKSPACE = "workspace"
    GLOBAL = "global"


@dataclass
class ServerConfigEntry:
    """Standardized representation of a single MCP server entry."""
    name: str = "cv-mcp-gateway"
    transport: MCPTransport = MCPTransport.STDIO
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    server_url: Optional[str] = None

    def to_client_dict(self) -> Dict[str, Any]:
        """Convert entry into standard dictionary based on transport."""
        if self.transport == MCPTransport.STDIO:
            d: Dict[str, Any] = {
                "command": self.command,
                "args": self.args or [],
            }
            if self.env:
                d["env"] = self.env
            return d
        else:
            # Streamable HTTP or SSE
            return {"serverUrl": self.server_url}


class BaseClientAdapter(ABC):
    """Abstract adapter representing a supported MCP client application."""

    client_id: str
    display_name: str
    supported_transports: List[MCPTransport]
    supported_scopes: List[ConfigScope]

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        project_root: Optional[Union[str, Path]] = None,
    ):
        self.config = config or get_config()
        self.project_root = Path(project_root).resolve() if project_root else get_project_root()

    def supports_transport(self, transport: MCPTransport) -> bool:
        return transport in self.supported_transports

    def supports_scope(self, scope: ConfigScope) -> bool:
        return scope in self.supported_scopes

    @abstractmethod
    def discover_config_path(
        self,
        scope: ConfigScope = ConfigScope.WORKSPACE,
        custom_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Discover the configuration file path for this client and scope."""
        pass

    @abstractmethod
    def generate_entry(
        self,
        transport: MCPTransport,
        server_name: str = "cv-mcp-gateway",
        server_url: Optional[str] = None,
        python_path: Optional[Union[str, Path]] = None,
    ) -> ServerConfigEntry:
        """Generate a client-specific ServerConfigEntry for the selected transport."""
        pass

    @abstractmethod
    def format_client_json(self, entry: ServerConfigEntry) -> Dict[str, Any]:
        """Format the server entry according to this client's specific JSON schema."""
        pass

    @abstractmethod
    def get_manual_guidance(self, transport: MCPTransport) -> str:
        """Provide human-readable setup and troubleshooting instructions."""
        pass
