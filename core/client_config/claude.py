"""Claude Desktop MCP configuration adapter."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.client_config.base import BaseClientAdapter, ConfigScope, ServerConfigEntry
from core.config.loader import get_config, get_project_root
from core.config.schema import AppConfig, MCPTransport


class ClaudeDesktopAdapter(BaseClientAdapter):
    """Adapter for Anthropic Claude Desktop."""

    client_id = "claude"
    display_name = "Claude Desktop"
    # Claude Desktop's claude_desktop_config.json ONLY supports local stdio processes.
    supported_transports = [MCPTransport.STDIO]
    supported_scopes = [ConfigScope.GLOBAL]

    def discover_config_path(
        self,
        scope: ConfigScope = ConfigScope.GLOBAL,
        custom_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Discover Claude Desktop configuration file location."""
        if custom_path:
            return Path(custom_path).resolve()

        if sys.platform == "win32":
            appdata = os.environ.get("APPDATA")
            base = Path(appdata) if appdata else (Path.home() / "AppData" / "Roaming")
            return (base / "Claude" / "claude_desktop_config.json").resolve()
        elif sys.platform == "darwin":
            return (
                Path.home()
                / "Library"
                / "Application Support"
                / "Claude"
                / "claude_desktop_config.json"
            ).resolve()
        else:
            return (Path.home() / ".config" / "Claude" / "claude_desktop_config.json").resolve()

    def generate_entry(
        self,
        transport: MCPTransport,
        server_name: str = "cv-mcp-gateway",
        server_url: Optional[str] = None,
        python_path: Optional[Union[str, Path]] = None,
    ) -> ServerConfigEntry:
        """Generate Claude Desktop server entry."""
        if transport != MCPTransport.STDIO:
            raise ValueError(
                f"Claude Desktop does not support remote '{transport}' in claude_desktop_config.json. "
                "Claude Desktop only accepts local stdio processes in this configuration file. "
                "To connect to a remote HTTP/SSE gateway, please configure it through "
                "Claude Desktop Settings -> Connectors."
            )

        py_exe = self._resolve_python(python_path)
        main_py = (self.project_root / "mcp_gateway" / "main.py").resolve()

        env_vars = {
            "DETECTION_SERVICE_URL": self.config.service.detection_service_url,
            "PROCESSING_SERVICE_URL": self.config.service.processing_service_url,
            "OCR_SERVICE_URL": self.config.service.ocr_service_url,
            "DATA_DIR": str(self.config.storage.data_dir),
        }

        return ServerConfigEntry(
            name=server_name,
            transport=MCPTransport.STDIO,
            command=str(py_exe),
            args=[str(main_py), "--transport", "stdio"],
            env=env_vars,
        )

    def _resolve_python(self, custom: Optional[Union[str, Path]] = None) -> Path:
        if custom and Path(custom).is_file():
            return Path(custom).resolve()
        if sys.platform == "win32":
            candidate = self.project_root / ".venv" / "Scripts" / "python.exe"
        else:
            candidate = self.project_root / ".venv" / "bin" / "python"
        if candidate.is_file():
            return candidate.resolve()
        return Path(sys.executable).resolve()

    def format_client_json(self, entry: ServerConfigEntry) -> Dict[str, Any]:
        """Format server entry into Claude Desktop JSON schema."""
        if entry.transport != MCPTransport.STDIO:
            raise ValueError("Claude Desktop config file only supports stdio transport.")
        return {
            "command": entry.command,
            "args": entry.args or [],
            "env": entry.env or {},
        }

    def get_manual_guidance(self, transport: MCPTransport) -> str:
        config_path = self.discover_config_path(ConfigScope.GLOBAL)
        if transport == MCPTransport.STDIO:
            return (
                f"Claude Desktop Configuration (Stdio):\n"
                f"- Configuration file: {config_path}\n"
                f"- Restart Claude Desktop completely after modifying this file to register new tools."
            )
        else:
            return (
                f"Claude Desktop Remote Connection Guidance:\n"
                f"- Remote '{transport}' cannot be added via claude_desktop_config.json.\n"
                f"- In Claude Desktop, navigate to: Settings -> Developer / Connectors to add remote servers.\n"
                f"- Use server URL: {self.config.service.detection_service_url.rsplit(':', 1)[0]}:8000/mcp"
            )
