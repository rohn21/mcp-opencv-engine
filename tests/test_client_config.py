"""Comprehensive unit tests for core.client_config module.

Tests cover:
- Client adapter registration and capability checks
- Transport and scope compatibility
- Antigravity, Cursor, and Claude Desktop adapters
- Stdio and Streamable HTTP JSON schema formatting
- Safe non-destructive JSON merges preserving unrelated keys/servers
- Timestamped backup creation
- Dry-run preview vs atomic apply
- Malformed JSON handling
- Claude Desktop remote-configuration refusal
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.client_config.antigravity import AntigravityAdapter
from core.client_config.base import BaseClientAdapter, ConfigScope, ServerConfigEntry
from core.client_config.claude import ClaudeDesktopAdapter
from core.client_config.cursor import CursorAdapter
from core.client_config.merger import (
    apply_json_merge,
    create_backup,
    generate_diff_preview,
    load_json_safe,
    merge_server_entry,
)
from core.client_config.registry import (
    configure_client,
    get_adapter,
    list_adapters,
)
from core.config.schema import MCPTransport


# ---------------------------------------------------------------------------
# Test Registry & Capabilities
# ---------------------------------------------------------------------------

class TestClientRegistry:
    def test_list_adapters(self, tmp_path):
        adapters = list_adapters(project_root=tmp_path)
        ids = [a.client_id for a in adapters]
        assert "antigravity" in ids
        assert "cursor" in ids
        assert "claude" in ids

    def test_get_adapter_valid(self, tmp_path):
        adapter = get_adapter("antigravity", project_root=tmp_path)
        assert isinstance(adapter, AntigravityAdapter)
        assert adapter.client_id == "antigravity"

    def test_get_adapter_unknown_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown MCP client: 'unknown_editor'"):
            get_adapter("unknown_editor", project_root=tmp_path)


# ---------------------------------------------------------------------------
# Test Antigravity Adapter
# ---------------------------------------------------------------------------

class TestAntigravityAdapter:
    def test_transports_and_scopes(self, tmp_path):
        adapter = AntigravityAdapter(project_root=tmp_path)
        assert adapter.supports_transport(MCPTransport.STREAMABLE_HTTP) is True
        assert adapter.supports_transport(MCPTransport.STDIO) is True
        assert adapter.supports_scope(ConfigScope.WORKSPACE) is True
        assert adapter.supports_scope(ConfigScope.GLOBAL) is True

    def test_discover_config_path(self, tmp_path):
        adapter = AntigravityAdapter(project_root=tmp_path)
        ws_path = adapter.discover_config_path(ConfigScope.WORKSPACE)
        assert ws_path == (tmp_path / ".agents" / "mcp_config.json").resolve()

        custom = tmp_path / "custom" / "mcp.json"
        assert adapter.discover_config_path(custom_path=custom) == custom.resolve()

    def test_generate_streamable_http_entry(self, tmp_path):
        adapter = AntigravityAdapter(project_root=tmp_path)
        entry = adapter.generate_entry(MCPTransport.STREAMABLE_HTTP, server_url="http://127.0.0.1:8000/mcp")
        assert entry.transport == MCPTransport.STREAMABLE_HTTP
        payload = adapter.format_client_json(entry)
        assert payload == {"serverUrl": "http://127.0.0.1:8000/mcp"}

    def test_generate_stdio_entry(self, tmp_path):
        adapter = AntigravityAdapter(project_root=tmp_path)
        entry = adapter.generate_entry(MCPTransport.STDIO)
        assert entry.transport == MCPTransport.STDIO
        assert entry.command is not None
        assert "--transport" in entry.args
        assert "stdio" in entry.args
        assert "DETECTION_SERVICE_URL" in entry.env


# ---------------------------------------------------------------------------
# Test Cursor Adapter
# ---------------------------------------------------------------------------

class TestCursorAdapter:
    def test_discover_config_path(self, tmp_path):
        adapter = CursorAdapter(project_root=tmp_path)
        ws_path = adapter.discover_config_path(ConfigScope.WORKSPACE)
        assert ws_path == (tmp_path / ".cursor" / "mcp.json").resolve()

    def test_generate_remote_url_entry(self, tmp_path):
        adapter = CursorAdapter(project_root=tmp_path)
        entry = adapter.generate_entry(MCPTransport.SSE, server_url="http://127.0.0.1:8000/sse")
        payload = adapter.format_client_json(entry)
        assert payload == {"url": "http://127.0.0.1:8000/sse"}

    def test_generate_streamable_http_default_url(self, tmp_path):
        adapter = CursorAdapter(project_root=tmp_path)
        entry = adapter.generate_entry(MCPTransport.STREAMABLE_HTTP)
        assert entry.server_url == "http://127.0.0.1:8000/mcp"

    def test_generate_sse_default_url(self, tmp_path):
        adapter = CursorAdapter(project_root=tmp_path)
        entry = adapter.generate_entry(MCPTransport.SSE)
        assert entry.server_url == "http://127.0.0.1:8000/sse"


# ---------------------------------------------------------------------------
# Test Claude Desktop Adapter
# ---------------------------------------------------------------------------

class TestClaudeDesktopAdapter:
    def test_supports_only_stdio(self, tmp_path):
        adapter = ClaudeDesktopAdapter(project_root=tmp_path)
        assert adapter.supports_transport(MCPTransport.STDIO) is True
        assert adapter.supports_transport(MCPTransport.STREAMABLE_HTTP) is False
        assert adapter.supports_transport(MCPTransport.SSE) is False

    def test_remote_streamable_http_refusal(self, tmp_path):
        adapter = ClaudeDesktopAdapter(project_root=tmp_path)
        with pytest.raises(ValueError, match="Claude Desktop does not support remote"):
            adapter.generate_entry(MCPTransport.STREAMABLE_HTTP)

    def test_stdio_entry_generation(self, tmp_path):
        adapter = ClaudeDesktopAdapter(project_root=tmp_path)
        entry = adapter.generate_entry(MCPTransport.STDIO)
        payload = adapter.format_client_json(entry)
        assert "command" in payload
        assert "args" in payload
        assert "env" in payload


# ---------------------------------------------------------------------------
# Test Non-Destructive JSON Merge & Atomic Writes
# ---------------------------------------------------------------------------

class TestMerger:
    def test_merge_into_nonexistent_file(self, tmp_path):
        target = tmp_path / "mcp_config.json"
        payload = {"serverUrl": "http://127.0.0.1:8000/mcp"}

        merged, preview, backup = apply_json_merge(
            target_file=target,
            server_name="cv-mcp-gateway",
            server_payload=payload,
            dry_run=False,
        )

        assert target.is_file()
        assert backup is None  # No prior file to backup
        saved = json.loads(target.read_text(encoding="utf-8"))
        assert saved["mcpServers"]["cv-mcp-gateway"] == payload

    def test_merge_preserves_unrelated_servers_and_keys(self, tmp_path):
        target = tmp_path / "mcp_config.json"
        initial_data = {
            "version": "1.0",
            "globalTheme": "dark",
            "mcpServers": {
                "sqlite-server": {"command": "uvx", "args": ["mcp-server-sqlite"]},
                "filesystem-server": {"command": "npx", "args": ["@modelcontextprotocol/server-filesystem"]},
            },
        }
        target.write_text(json.dumps(initial_data, indent=2), encoding="utf-8")

        new_payload = {"serverUrl": "http://127.0.0.1:8000/mcp"}

        merged, preview, backup = apply_json_merge(
            target_file=target,
            server_name="cv-mcp-gateway",
            server_payload=new_payload,
            dry_run=False,
        )

        assert backup is not None
        assert backup.is_file()

        saved = json.loads(target.read_text(encoding="utf-8"))
        # Verify unrelated keys preserved
        assert saved["version"] == "1.0"
        assert saved["globalTheme"] == "dark"
        # Verify unrelated servers preserved
        assert "sqlite-server" in saved["mcpServers"]
        assert "filesystem-server" in saved["mcpServers"]
        # Verify new server inserted
        assert saved["mcpServers"]["cv-mcp-gateway"] == new_payload

    def test_update_existing_entry_replaces_without_duplicates(self, tmp_path):
        target = tmp_path / "mcp_config.json"
        initial_data = {
            "mcpServers": {
                "cv-mcp-gateway": {"serverUrl": "http://old-url:8000"},
            }
        }
        target.write_text(json.dumps(initial_data), encoding="utf-8")

        updated_payload = {"serverUrl": "http://127.0.0.1:8000/mcp"}

        merged, preview, backup = apply_json_merge(
            target_file=target,
            server_name="cv-mcp-gateway",
            server_payload=updated_payload,
            dry_run=False,
        )

        saved = json.loads(target.read_text(encoding="utf-8"))
        assert saved["mcpServers"]["cv-mcp-gateway"]["serverUrl"] == "http://127.0.0.1:8000/mcp"
        assert len(saved["mcpServers"]) == 1

    def test_malformed_json_raises_descriptive_error(self, tmp_path):
        target = tmp_path / "broken_config.json"
        target.write_text("{ malformed: json, [unclosed", encoding="utf-8")

        with pytest.raises(ValueError, match="contains malformed JSON"):
            apply_json_merge(
                target_file=target,
                server_name="cv-mcp-gateway",
                server_payload={"serverUrl": "http://127.0.0.1:8000/mcp"},
                dry_run=False,
            )

    def test_dry_run_does_not_modify_file_or_create_backup(self, tmp_path):
        target = tmp_path / "config.json"
        target.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")

        merged, preview, backup = apply_json_merge(
            target_file=target,
            server_name="cv-mcp-gateway",
            server_payload={"serverUrl": "http://127.0.0.1:8000/mcp"},
            dry_run=True,
        )

        assert backup is None
        # File contents should still be original
        saved = json.loads(target.read_text(encoding="utf-8"))
        assert "cv-mcp-gateway" not in saved["mcpServers"]
        # Preview contains proposed diff
        assert "Proposed MCP Client Config Change" in preview


# ---------------------------------------------------------------------------
# Test High-Level configure_client Orchestration
# ---------------------------------------------------------------------------

class TestConfigureClientOrchestrator:
    def test_configure_antigravity_workspace(self, tmp_path):
        cfg_path, backup, preview = configure_client(
            client_id="antigravity",
            transport="streamable-http",
            scope="workspace",
            dry_run=False,
            project_root=tmp_path,
        )
        assert cfg_path == (tmp_path / ".agents" / "mcp_config.json").resolve()
        assert cfg_path.is_file()

        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert "cv-mcp-gateway" in data["mcpServers"]
        assert "serverUrl" in data["mcpServers"]["cv-mcp-gateway"]

    def test_configure_cursor_project(self, tmp_path):
        cfg_path, backup, preview = configure_client(
            client_id="cursor",
            transport="stdio",
            scope="workspace",
            dry_run=False,
            project_root=tmp_path,
        )
        assert cfg_path == (tmp_path / ".cursor" / "mcp.json").resolve()
        assert cfg_path.is_file()

        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert "cv-mcp-gateway" in data["mcpServers"]
        assert "command" in data["mcpServers"]["cv-mcp-gateway"]

    def test_configure_claude_unsupported_transport_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not support transport 'streamable-http'"):
            configure_client(
                client_id="claude",
                transport="streamable-http",
                scope="global",
                project_root=tmp_path,
            )
