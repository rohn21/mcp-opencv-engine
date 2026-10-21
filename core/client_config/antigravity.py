"""Antigravity IDE MCP client configuration adapter."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.client_config.base import BaseClientAdapter, ConfigScope, ServerConfigEntry
from core.config.loader import get_config, get_project_root
from core.config.schema import AppConfig, MCPTransport


class AntigravityAdapter(BaseClientAdapter):
    """Adapter for Google Antigravity IDE."""

    client_id = "antigravity"
    display_name = "Google Antigravity IDE"
    supported_transports = [MCPTransport.STREAMABLE_HTTP, MCPTransport.STDIO]
    supported_scopes = [ConfigScope.WORKSPACE, ConfigScope.GLOBAL]

    def discover_config_path(
        self,
        scope: ConfigScope = ConfigScope.WORKSPACE,
        custom_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Discover Antigravity config file location."""
        if custom_path:
            return Path(custom_path).resolve()

        if scope == ConfigScope.WORKSPACE:
            # Workspace: <project_root>/.agents/mcp_config.json
            return (self.project_root / ".agents" / "mcp_config.json").resolve()
        else:
            # Global: ~/.gemini/config/mcp_config.json
            return (Path.home() / ".gemini" / "config" / "mcp_config.json").resolve()

    def generate_entry(
        self,
        transport: MCPTransport,
        server_name: str = "cv-mcp-gateway",
        server_url: Optional[str] = None,
        python_path: Optional[Union[str, Path]] = None,
    ) -> ServerConfigEntry:
        """Generate Antigravity-specific server configuration entry."""
        if transport not in self.supported_transports:
            raise ValueError(f"Antigravity adapter does not support transport: '{transport}'")

        if transport == MCPTransport.STREAMABLE_HTTP:
            target_url = server_url or f"{self.config.service.detection_service_url.rsplit(':', 1)[0]}:8000/mcp"
            # Normalize to 127.0.0.1 for local gateway
            if "localhost" in target_url:
                target_url = target_url.replace("localhost", "127.0.0.1")
            return ServerConfigEntry(
                name=server_name,
                transport=MCPTransport.STREAMABLE_HTTP,
                server_url=target_url,
            )

        # Stdio transport
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
        """Format server entry into Antigravity JSON schema."""
        if entry.transport == MCPTransport.STREAMABLE_HTTP:
            return {"serverUrl": entry.server_url}
        return {
            "command": entry.command,
            "args": entry.args or [],
            "env": entry.env or {},
        }

    def get_manual_guidance(self, transport: MCPTransport) -> str:
        workspace_path = self.discover_config_path(ConfigScope.WORKSPACE)
        global_path = self.discover_config_path(ConfigScope.GLOBAL)
        return (
            f"Antigravity IDE Configuration:\n"
            f"- Workspace config: {workspace_path}\n"
            f"- Global config: {global_path}\n"
            f"- Supported Transports: Streamable HTTP ('serverUrl') and local Stdio ('command'/'args').\n"
            f"- Once configured, Antigravity discovers registered tools automatically on start."
        )
