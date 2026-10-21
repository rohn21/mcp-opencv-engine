"""Cursor AI Code Editor MCP configuration adapter."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.client_config.base import BaseClientAdapter, ConfigScope, ServerConfigEntry
from core.config.loader import get_config, get_project_root
from core.config.schema import AppConfig, MCPTransport


class CursorAdapter(BaseClientAdapter):
    """Adapter for Cursor IDE."""

    client_id = "cursor"
    display_name = "Cursor AI Code Editor"
    supported_transports = [MCPTransport.STREAMABLE_HTTP, MCPTransport.SSE, MCPTransport.STDIO]
    supported_scopes = [ConfigScope.WORKSPACE, ConfigScope.GLOBAL]

    def discover_config_path(
        self,
        scope: ConfigScope = ConfigScope.WORKSPACE,
        custom_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Discover Cursor MCP configuration path."""
        if custom_path:
            return Path(custom_path).resolve()

        if scope == ConfigScope.WORKSPACE:
            # Workspace: <project_root>/.cursor/mcp.json
            return (self.project_root / ".cursor" / "mcp.json").resolve()
        else:
            # Global: ~/.cursor/mcp.json
            return (Path.home() / ".cursor" / "mcp.json").resolve()

    def generate_entry(
        self,
        transport: MCPTransport,
        server_name: str = "cv-mcp-gateway",
        server_url: Optional[str] = None,
        python_path: Optional[Union[str, Path]] = None,
    ) -> ServerConfigEntry:
        """Generate Cursor-specific server entry."""
        if transport not in self.supported_transports:
            raise ValueError(f"Cursor adapter does not support transport: '{transport}'")

        if transport in (MCPTransport.STREAMABLE_HTTP, MCPTransport.SSE):
            default_endpoint = "/mcp" if transport == MCPTransport.STREAMABLE_HTTP else "/sse"
            target_url = server_url or f"http://127.0.0.1:8000{default_endpoint}"
            if "localhost" in target_url:
                target_url = target_url.replace("localhost", "127.0.0.1")
            return ServerConfigEntry(
                name=server_name,
                transport=transport,
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
        """Format server entry into Cursor JSON schema."""
        if entry.transport in (MCPTransport.STREAMABLE_HTTP, MCPTransport.SSE):
            # Cursor expects "url" for remote endpoints
            return {"url": entry.server_url}
        return {
            "command": entry.command,
            "args": entry.args or [],
            "env": entry.env or {},
        }

    def get_manual_guidance(self, transport: MCPTransport) -> str:
        workspace_path = self.discover_config_path(ConfigScope.WORKSPACE)
        global_path = self.discover_config_path(ConfigScope.GLOBAL)
        return (
            f"Cursor Configuration:\n"
            f"- Project config: {workspace_path}\n"
            f"- Global config: {global_path}\n"
            f"- Cursor UI: Open Settings -> Features -> MCP to view active servers."
        )
