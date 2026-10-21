"""Comprehensive unit tests for core.runtime module.

Tests cover:
- Runtime mode selection and auto-detection
- Native process startup, logs, duplicate avoidance, and graceful shutdown (mocked)
- Docker CLI invocation, prerequisite verification, config validation, and clean stopping (mocked)
- Dynamic Compose override generation with :ro input and :rw output mounts
- Broad-directory and protected-directory mount rejection
- Health endpoint polling and bounded timeout handling
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch
import pytest

from core.config.loader import load_config
from core.config.schema import AppConfig, RuntimeMode
from core.filesystem.permissions import AccessDeniedError
from core.runtime.base import (
    BaseRuntime,
    RuntimeState,
    RuntimeStatus,
    ServiceStatus,
    check_http_endpoint,
)
from core.runtime.docker import DockerRuntime
from core.runtime.manager import RuntimeManager
from core.runtime.mounts import (
    generate_compose_override,
    remove_compose_override,
    validate_mount_path,
)
from core.runtime.native import NativeRuntime


# ---------------------------------------------------------------------------
# Test Runtime Mode Selection & Manager
# ---------------------------------------------------------------------------

class TestRuntimeManager:
    def test_explicit_native_mode(self, tmp_path):
        mgr = RuntimeManager(runtime_mode="native", project_root=tmp_path)
        assert isinstance(mgr.runtime, NativeRuntime)
        assert mgr.mode == "native"

    def test_explicit_docker_mode(self, tmp_path):
        mgr = RuntimeManager(runtime_mode="docker", project_root=tmp_path)
        assert isinstance(mgr.runtime, DockerRuntime)
        assert mgr.mode == "docker"

    def test_auto_mode_selects_docker_when_available(self, tmp_path):
        mock_profile = MagicMock(has_docker=True, docker_compose_available=True)
        with patch("core.runtime.manager.EnvironmentDetector.detect", return_value=mock_profile):
            mgr = RuntimeManager(runtime_mode="auto", project_root=tmp_path)
            assert isinstance(mgr.runtime, DockerRuntime)
            assert mgr.mode == "docker"

    def test_auto_mode_selects_native_when_docker_missing(self, tmp_path):
        mock_profile = MagicMock(has_docker=False, docker_compose_available=False)
        with patch("core.runtime.manager.EnvironmentDetector.detect", return_value=mock_profile):
            mgr = RuntimeManager(runtime_mode="auto", project_root=tmp_path)
            assert isinstance(mgr.runtime, NativeRuntime)
            assert mgr.mode == "native"

    def test_invalid_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="is not a valid RuntimeMode"):
            RuntimeManager(runtime_mode="kubernetes", project_root=tmp_path)


# ---------------------------------------------------------------------------
# Test Native Runtime
# ---------------------------------------------------------------------------

class TestNativeRuntime:
    @pytest.fixture
    def native_runtime(self, tmp_path):
        return NativeRuntime(project_root=tmp_path)

    def test_python_executable_discovery(self, native_runtime):
        assert native_runtime.python_executable.is_file()

    @patch("subprocess.Popen")
    @patch("core.runtime.native.check_http_endpoint", return_value=False)
    def test_start_service_creates_subprocess_and_logs(self, mock_check, mock_popen, native_runtime, tmp_path):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        status = native_runtime.start_service("object-detection")
        assert status.state == RuntimeState.STARTING
        assert status.pid == 12345
        assert status.port == 8001

        # Verify command arguments list: no shell=True
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        cmd = args[0]
        assert cmd[1:4] == ["-m", "uvicorn", "opencv_services.object_detection.api:app"]
        assert kwargs["shell"] is False
        assert kwargs["cwd"] == str(tmp_path)

        # Verify log file was opened in .runtime/logs
        log_file = tmp_path / ".runtime" / "logs" / "object-detection.log"
        assert log_file.exists()

    @patch("subprocess.Popen")
    @patch("core.runtime.native.check_http_endpoint", return_value=False)
    def test_no_duplicate_launches(self, mock_check, mock_popen, native_runtime):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        # First start
        s1 = native_runtime.start_service("object-detection")
        assert mock_popen.call_count == 1

        # Second start with same process still running
        s2 = native_runtime.start_service("object-detection")
        assert mock_popen.call_count == 1
        assert s2.pid == 12345

    def test_start_already_responding_service_skips_launch(self, native_runtime):
        with patch("core.runtime.native.check_http_endpoint", return_value=True):
            with patch("subprocess.Popen") as mock_popen:
                status = native_runtime.start_service("object-detection")
                assert status.state == RuntimeState.RUNNING
                mock_popen.assert_not_called()

    def test_stop_service_terminates_gracefully(self, native_runtime):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        native_runtime._processes["object-detection"] = mock_proc

        success = native_runtime.stop_service("object-detection", timeout=2.0)
        assert success is True
        mock_proc.terminate.assert_called_once()
        assert "object-detection" not in native_runtime._processes

    def test_status_reporting(self, native_runtime):
        with patch("core.runtime.native.check_http_endpoint") as mock_http:
            mock_http.side_effect = lambda url: "8001" in url  # Only object-detection healthy
            status = native_runtime.get_status()
            assert status.mode == "native"
            assert status.overall_state == RuntimeState.PARTIAL
            assert status.services["object-detection"].state == RuntimeState.RUNNING
            assert status.services["image-processing"].state == RuntimeState.STOPPED


# ---------------------------------------------------------------------------
# Test Docker Runtime
# ---------------------------------------------------------------------------

class TestDockerRuntime:
    def test_prerequisite_check_docker_missing(self, tmp_path):
        def runner(cmd):
            if cmd == ["docker", "--version"]:
                return 127, "", "docker not found"
            return 0, "", ""

        runtime = DockerRuntime(project_root=tmp_path, command_runner=runner)
        with pytest.raises(RuntimeError, match="Docker CLI is not installed"):
            runtime.check_prerequisites()

    def test_prerequisite_check_daemon_down(self, tmp_path):
        def runner(cmd):
            if cmd == ["docker", "--version"]:
                return 0, "Docker version 27.0.0", ""
            if cmd == ["docker", "info"]:
                return 1, "", "Cannot connect to the Docker daemon"
            return 0, "", ""

        runtime = DockerRuntime(project_root=tmp_path, command_runner=runner)
        with pytest.raises(RuntimeError, match="Docker daemon is not running"):
            runtime.check_prerequisites()

    def test_validate_compose_missing_file_raises(self, tmp_path):
        runtime = DockerRuntime(project_root=tmp_path, compose_file=tmp_path / "nonexistent.yml")
        with pytest.raises(FileNotFoundError, match="Docker Compose file not found"):
            runtime.validate_compose()

    def test_validate_compose_invalid_config_raises(self, tmp_path):
        dummy_compose = tmp_path / "docker-compose.yml"
        dummy_compose.write_text("invalid: [yaml: broken")

        def runner(cmd):
            return 1, "", "yaml error on line 1"

        runtime = DockerRuntime(project_root=tmp_path, compose_file=dummy_compose, command_runner=runner)
        with pytest.raises(RuntimeError, match="Docker Compose configuration is invalid"):
            runtime.validate_compose()

    def test_start_uses_explicit_project_and_no_build(self, tmp_path):
        dummy_compose = tmp_path / "docker-compose.yml"
        dummy_compose.write_text("version: '3.9'")

        recorded_commands = []

        def runner(cmd):
            recorded_commands.append(cmd)
            return 0, "", ""

        runtime = DockerRuntime(
            project_root=tmp_path,
            compose_file=dummy_compose,
            project_name="test-project",
            command_runner=runner,
        )

        with patch.object(runtime, "wait_for_ready", return_value=True):
            success = runtime.start(build=False)
            assert success is True

        # Check recorded up command
        up_cmds = [c for c in recorded_commands if "up" in c]
        assert len(up_cmds) == 1
        cmd = up_cmds[0]
        assert "-p" in cmd and "test-project" in cmd
        assert "--no-build" in cmd
        assert "-d" in cmd

    def test_stop_does_not_remove_volumes(self, tmp_path):
        dummy_compose = tmp_path / "docker-compose.yml"
        dummy_compose.write_text("version: '3.9'")

        recorded_commands = []

        def runner(cmd):
            recorded_commands.append(cmd)
            return 0, "", ""

        runtime = DockerRuntime(
            project_root=tmp_path,
            compose_file=dummy_compose,
            project_name="my-project",
            command_runner=runner,
        )

        success = runtime.stop(timeout=10.0)
        assert success is True
        assert len(recorded_commands) == 1
        cmd = recorded_commands[0]
        assert "stop" in cmd
        assert "-v" not in cmd  # Must NEVER pass -v to stop or remove volumes

    def test_status_parsing_from_compose_ps(self, tmp_path):
        dummy_compose = tmp_path / "docker-compose.yml"
        dummy_compose.write_text("version: '3.9'")

        ps_json = json.dumps([
            {"Service": "object-detection", "State": "running"},
            {"Service": "image-processing", "State": "running"},
            {"Service": "ocr-service", "State": "exited"},
        ])

        def runner(cmd):
            if "ps" in cmd:
                return 0, ps_json, ""
            return 0, "", ""

        runtime = DockerRuntime(
            project_root=tmp_path,
            compose_file=dummy_compose,
            command_runner=runner,
        )

        with patch("core.runtime.docker.check_http_endpoint", return_value=False):
            status = runtime.get_status()
            assert status.mode == "docker"
            assert status.services["object-detection"].state == RuntimeState.STARTING
            assert status.services["ocr-service"].state == RuntimeState.ERROR


# ---------------------------------------------------------------------------
# Test Dynamic Mounts Generation & Permissions
# ---------------------------------------------------------------------------

class TestDynamicMounts:
    def test_validate_mount_path_success(self, tmp_path):
        valid_dir = tmp_path / "my_inputs"
        valid_dir.mkdir()
        resolved = validate_mount_path(valid_dir, is_output=False, project_root=tmp_path)
        assert resolved == valid_dir.resolve()

    def test_validate_mount_nonexistent_raises(self, tmp_path):
        missing = tmp_path / "nonexistent_dir"
        with pytest.raises(FileNotFoundError, match="Mount path does not exist"):
            validate_mount_path(missing, project_root=tmp_path)

    def test_validate_mount_file_raises(self, tmp_path):
        file_path = tmp_path / "file.txt"
        file_path.write_text("data")
        with pytest.raises(NotADirectoryError, match="must be a directory"):
            validate_mount_path(file_path, project_root=tmp_path)

    def test_validate_mount_project_root_forbidden(self, tmp_path):
        with pytest.raises(AccessDeniedError, match="Mounting the entire project root"):
            validate_mount_path(tmp_path, project_root=tmp_path)

    def test_validate_mount_broad_system_directory_forbidden(self, tmp_path):
        with pytest.raises(AccessDeniedError, match="broad system root or home directory"):
            validate_mount_path(Path.home(), project_root=tmp_path)

    def test_generate_compose_override_ro_rw_specifications(self, tmp_path):
        inputs = tmp_path / "custom_inputs"
        inputs.mkdir()
        outputs = tmp_path / "custom_outputs"
        outputs.mkdir()

        override_file = tmp_path / "docker-compose.override.yml"

        res_path = generate_compose_override(
            input_dirs=[inputs],
            output_dir=outputs,
            project_root=tmp_path,
            target_override_file=override_file,
            services=["object-detection", "image-processing"],
        )
        assert res_path.is_file()

        content = override_file.read_text(encoding="utf-8")

        # Verify read-only flag for inputs and read-write flag for outputs
        assert ":/app/data/inputs:ro" in content
        assert ":/app/data/outputs:rw" in content

        # Verify services declared in override
        assert "object-detection:" in content
        assert "image-processing:" in content

        # Verify cleanup function
        assert remove_compose_override(project_root=tmp_path) is False  # not in default .runtime
        # Move override to default location to test remove
        default_override = tmp_path / ".runtime" / "docker-compose.override.yml"
        default_override.parent.mkdir(parents=True, exist_ok=True)
        default_override.write_text("override")
        assert remove_compose_override(project_root=tmp_path) is True
        assert not default_override.exists()


# ---------------------------------------------------------------------------
# Test Health Endpoint Utility
# ---------------------------------------------------------------------------

class TestHealthEndpointCheck:
    def test_endpoint_success(self):
        mock_resp = MagicMock(status=200, code=200)
        mock_resp.__enter__.return_value = mock_resp
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert check_http_endpoint("http://localhost:8001/health") is True

    def test_endpoint_error_status(self):
        mock_resp = MagicMock(status=503, code=503)
        mock_resp.__enter__.return_value = mock_resp
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert check_http_endpoint("http://localhost:8001/health") is False

    def test_endpoint_connection_refused(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            assert check_http_endpoint("http://localhost:8001/health") is False
