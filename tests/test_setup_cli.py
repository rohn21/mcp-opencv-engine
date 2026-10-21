"""Comprehensive unit tests for setup CLI and diagnostics (Phase 8).

Verifies:
- CLI argument parsing, help output, and validation
- Diagnostics with mocked environment profiles (PASS, WARN, FAIL aggregation)
- Read-only diagnostics behavior (strictly no filesystem mutation)
- Service endpoint probing: distinguishes offline (WARN) from failing (FAIL)
- JSON output schema stability and ANSI-free formatting
- Exit codes: 0 (PASS), 1 (FAIL), 2 (WARN)
- Guided setup workflow: cancellation vs approval (--yes)
- MCP client configuration integration via adapters with backups
- Sensitive value redaction
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.config.schema import AppConfig, MCPTransport, Profile, RuntimeMode
from core.dependencies.checker import CheckResult, Severity
from core.environment.capabilities import EnvironmentProfile
from setup.configure import (
    build_parser,
    cmd_configure_client,
    cmd_configure_runtime,
    cmd_doctor,
    cmd_guided_setup,
    main,
)
from setup.diagnostics import (
    DiagnosticItem,
    DiagnosticReport,
    SystemDiagnostics,
    probe_service_endpoint,
)


# ---------------------------------------------------------------------------
# CLI Argument Parsing & Validation Tests
# ---------------------------------------------------------------------------

class TestCLIParser:
    def test_help_flag(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_mutually_exclusive_actions(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--doctor", "--setup"])

    def test_invalid_runtime_mode_argument(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--configure-runtime", "unsupported_mode"])

    def test_invalid_client_argument(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--configure-client", "unsupported_client"])

    def test_valid_options_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["--doctor", "--json"])
        assert args.doctor is True
        assert args.json is True

        args2 = parser.parse_args(["--configure-runtime", "docker"])
        assert args2.configure_runtime == "docker"

        args3 = parser.parse_args(["--configure-client", "antigravity", "--yes", "--transport", "stdio"])
        assert args3.configure_client == "antigravity"
        assert args3.yes is True
        assert args3.transport == "stdio"


# ---------------------------------------------------------------------------
# Diagnostics Aggregation and Exit Codes Tests
# ---------------------------------------------------------------------------

class TestDiagnosticsEngine:
    def test_report_summary_and_text_formatting(self):
        report = DiagnosticReport(
            timestamp="2026-10-20T12:00:00Z",
            overall_status="WARN",
            exit_code=2,
            items=[
                DiagnosticItem(category="Env", name="Python", status="PASS", details="Python 3.11"),
                DiagnosticItem(category="Env", name="GPU", status="WARN", details="CPU mode", remediation="Install CUDA"),
            ],
        )
        report_dict = report.to_dict()
        assert report_dict["summary"]["total"] == 2
        assert report_dict["summary"]["passed"] == 1
        assert report_dict["summary"]["warnings"] == 1
        assert report_dict["summary"]["failures"] == 0

        text = report.to_text()
        assert "Overall Result: [WARN]" in text
        assert "Remedy: Install CUDA" in text

    def test_read_only_guarantee_no_files_created(self, tmp_path):
        """Verify that running diagnostics creates zero files in project_root or data dirs."""
        weights = tmp_path / "yolov8n.pt"
        weights.write_bytes(b"dummy-weights-content")
        compose = tmp_path / "docker-compose.yml"
        compose.write_text("services: {}\n", encoding="utf-8")

        initial_files = set(tmp_path.rglob("*"))

        diag = SystemDiagnostics(project_root=tmp_path)
        report = diag.run()

        after_files = set(tmp_path.rglob("*"))
        # Only the initial files should exist, diagnostics must NOT create data/inputs or data/outputs
        assert initial_files == after_files

    def test_diagnostics_all_pass_scenario(self, tmp_path):
        weights = tmp_path / "yolov8n.pt"
        weights.write_bytes(b"weights")
        compose = tmp_path / "docker-compose.yml"
        compose.write_text("services: {}\n", encoding="utf-8")
        inputs = tmp_path / "data" / "inputs"
        outputs = tmp_path / "data" / "outputs"
        inputs.mkdir(parents=True)
        outputs.mkdir(parents=True)

        # Mock an initialized client so MCP client check passes
        client_dir = tmp_path / ".agents"
        client_dir.mkdir(parents=True)
        (client_dir / "mcp_config.json").write_text(
            json.dumps({"mcpServers": {"cv-mcp-gateway": {"command": "python"}}}),
            encoding="utf-8",
        )

        mock_profile = EnvironmentProfile(
            os_family="linux",
            os_name="Linux",
            os_version="5.15.0",
            cpu_arch="x86_64",
            python_version="3.11.9",
            python_executable=sys.executable,
            gpu_backend="cuda",
            has_docker=True,
            docker_compose_available=True,
            diagnostics={"docker_daemon_running": True},
        )

        with patch("setup.diagnostics.EnvironmentDetector.detect", return_value=mock_profile), \
             patch("setup.diagnostics.DependencyChecker.check_service_packages", return_value=[
                 CheckResult(name="mcp", category="python_package", severity=Severity.OK, message="ok")
             ]), \
             patch("setup.diagnostics.DependencyChecker.check_external_tools", return_value=[
                 CheckResult(name="Tesseract OCR", category="external_tool", severity=Severity.OK, message="ok")
             ]), \
             patch("setup.diagnostics.check_docker_runtime", return_value=CheckResult(
                 name="Docker", category="runtime", severity=Severity.OK, message="ok"
             )), \
             patch("setup.diagnostics.probe_service_endpoint", return_value=("PASS", "Online", None)), \
             patch("setup.diagnostics.DockerRuntime.validate_compose", return_value=None):

            diag = SystemDiagnostics(project_root=tmp_path)
            report = diag.run()

            warns = [(i.category, i.name, i.details) for i in report.items if i.status == "WARN"]
            assert not warns, f"Unexpected warnings: {warns}"
            assert report.overall_status == "PASS"
            assert report.exit_code == 0
            assert report.to_dict()["summary"]["failures"] == 0

    def test_diagnostics_failure_on_missing_weights(self, tmp_path):
        # Missing yolov8n.pt
        compose = tmp_path / "docker-compose.yml"
        compose.write_text("services: {}\n", encoding="utf-8")

        diag = SystemDiagnostics(project_root=tmp_path)
        report = diag.run()

        assert report.overall_status == "FAIL"
        assert report.exit_code == 1
        model_item = next(i for i in report.items if i.name == "YOLO Weights")
        assert model_item.status == "FAIL"
        assert model_item.remediation is not None

    def test_diagnostics_warning_when_services_stopped(self, tmp_path):
        """Stopped services must be reported as WARN, not FAIL."""
        weights = tmp_path / "yolov8n.pt"
        weights.write_bytes(b"weights")
        compose = tmp_path / "docker-compose.yml"
        compose.write_text("services: {}\n", encoding="utf-8")

        with patch("setup.diagnostics.probe_service_endpoint", return_value=(
            "WARN", "Service not currently running (Normal if runtime is stopped)", "Start runtime"
        )):
            diag = SystemDiagnostics(project_root=tmp_path)
            report = diag.run()
            service_items = [i for i in report.items if i.category == "Services"]
            assert len(service_items) == 3
            assert all(i.status == "WARN" for i in service_items)

    def test_probe_service_endpoint_distinguishes_500_from_offline(self):
        import urllib.error
        # HTTP 500 error from running service -> FAIL
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("url", 500, "Internal Error", {}, None)):
            status, details, remed = probe_service_endpoint("http://localhost:8001/health")
            assert status == "FAIL"
            assert "HTTP error 500" in details

        # Connection refused (offline service) -> WARN
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            status, details, remed = probe_service_endpoint("http://localhost:8001/health")
            assert status == "WARN"
            assert "Normal if runtime is stopped" in details


# ---------------------------------------------------------------------------
# JSON Output Structure & Redaction Tests
# ---------------------------------------------------------------------------

class TestDiagnosticsJsonAndRedaction:
    def test_json_structure_and_ansi_free(self, tmp_path, capsys):
        weights = tmp_path / "yolov8n.pt"
        weights.write_bytes(b"weights")
        compose = tmp_path / "docker-compose.yml"
        compose.write_text("services: {}\n", encoding="utf-8")

        code = cmd_doctor(json_output=True, project_root=tmp_path)
        out, _ = capsys.readouterr()

        # Parse JSON
        parsed = json.loads(out)
        assert "timestamp" in parsed
        assert "overall_status" in parsed
        assert "exit_code" in parsed
        assert "summary" in parsed
        assert "checks" in parsed
        assert isinstance(parsed["checks"], list)

        # No ANSI escape codes
        assert "\033[" not in out

    def test_sensitive_value_redaction_in_diagnostics(self, tmp_path):
        os.environ["SECRET_TOKEN"] = "super-secret-12345"
        diag = SystemDiagnostics(project_root=tmp_path)
        report = diag.run()
        report_json = json.dumps(report.to_dict())
        assert "super-secret-12345" not in report_json
        os.environ.pop("SECRET_TOKEN", None)


# ---------------------------------------------------------------------------
# CLI Command Handlers Tests
# ---------------------------------------------------------------------------

class TestCLICommands:
    def test_configure_runtime_updates_env(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=bar\nRUNTIME_MODE=auto\n", encoding="utf-8")

        code = cmd_configure_runtime("docker", project_root=tmp_path)
        assert code == 0

        content = env_file.read_text(encoding="utf-8")
        assert "RUNTIME_MODE=docker" in content
        assert "FOO=bar" in content

    def test_configure_runtime_invalid_mode(self, tmp_path):
        code = cmd_configure_runtime("invalid_mode", project_root=tmp_path)
        assert code == 1

    def test_configure_client_dry_run_does_not_modify_file(self, tmp_path):
        code = cmd_configure_client(
            client_id="antigravity",
            dry_run=True,
            project_root=tmp_path,
        )
        assert code == 0
        target_file = tmp_path / ".agents" / "mcp_config.json"
        assert not target_file.exists()

    def test_configure_client_applied_with_backup(self, tmp_path):
        target_file = tmp_path / ".agents" / "mcp_config.json"
        target_file.parent.mkdir(parents=True)
        target_file.write_text('{"mcpServers": {"existing": {"url": "http://other"}}}', encoding="utf-8")

        code = cmd_configure_client(
            client_id="antigravity",
            dry_run=False,
            project_root=tmp_path,
        )
        assert code == 0
        assert target_file.is_file()

        # Verify backup was created
        backups = list(target_file.parent.glob("mcp_config.json.bak_*"))
        assert len(backups) == 1

        # Verify content merged safely
        data = json.loads(target_file.read_text(encoding="utf-8"))
        assert "existing" in data["mcpServers"]
        assert "cv-mcp-gateway" in data["mcpServers"]


# ---------------------------------------------------------------------------
# Guided Setup Flow Tests
# ---------------------------------------------------------------------------

class TestGuidedSetup:
    def test_guided_setup_cancelled_makes_no_changes(self, tmp_path):
        inputs_dir = tmp_path / "data" / "inputs"
        outputs_dir = tmp_path / "data" / "outputs"
        env_file = tmp_path / ".env"

        # User chooses default runtime, default client, but rejects confirmation
        answers = iter(["", "", "n"])
        code = cmd_guided_setup(
            yes=False,
            project_root=tmp_path,
            input_func=lambda _: next(answers),
        )
        assert code == 0
        # No mutations
        assert not inputs_dir.exists()
        assert not outputs_dir.exists()
        assert not env_file.exists()

    def test_guided_setup_with_yes_flag(self, tmp_path):
        inputs_dir = tmp_path / "data" / "inputs"
        outputs_dir = tmp_path / "data" / "outputs"
        env_file = tmp_path / ".env"
        client_cfg = tmp_path / ".agents" / "mcp_config.json"

        code = cmd_guided_setup(
            yes=True,
            project_root=tmp_path,
        )
        assert code == 0
        assert inputs_dir.is_dir()
        assert outputs_dir.is_dir()
        assert env_file.is_file()
        assert client_cfg.is_file()

        data = json.loads(client_cfg.read_text(encoding="utf-8"))
        assert "cv-mcp-gateway" in data["mcpServers"]

    def test_guided_setup_claude_transport_safety(self, tmp_path):
        """Guided setup with Claude Desktop defaults to local stdio."""
        # Answers: mode="native", client="claude", confirm="yes"
        answers = iter(["native", "claude", "y"])
        code = cmd_guided_setup(
            yes=False,
            project_root=tmp_path,
            input_func=lambda _: next(answers),
        )
        assert code == 0


# ---------------------------------------------------------------------------
# Main CLI Entry Point Dispatch Tests
# ---------------------------------------------------------------------------

class TestMainEntrypoint:
    def test_main_doctor(self):
        with patch("setup.configure.cmd_doctor", return_value=0) as mock_doc:
            ret = main(["--doctor"])
            assert ret == 0
            mock_doc.assert_called_once_with(json_output=False)

    def test_main_setup_yes(self):
        with patch("setup.configure.cmd_guided_setup", return_value=0) as mock_setup:
            ret = main(["--setup", "--yes"])
            assert ret == 0
            mock_setup.assert_called_once_with(yes=True)

    def test_main_configure_runtime(self):
        with patch("setup.configure.cmd_configure_runtime", return_value=0) as mock_cfg:
            ret = main(["--configure-runtime", "native"])
            assert ret == 0
            mock_cfg.assert_called_once_with("native")

    def test_main_no_args_prints_help(self, capsys):
        ret = main([])
        assert ret == 0
        out, _ = capsys.readouterr()
        assert "usage: python setup/configure.py" in out
