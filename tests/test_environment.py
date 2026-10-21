"""Unit tests for core.environment module."""

from dataclasses import FrozenInstanceError
import json
from unittest.mock import MagicMock, patch
import pytest

from core.environment.capabilities import EnvironmentProfile
from core.environment.detector import (
    EnvironmentDetector,
    _detect_docker,
    _detect_gpu,
    _detect_tesseract,
    _get_os_family,
    detect_environment,
)


def test_get_os_family():
    assert _get_os_family("win32") == "windows"
    assert _get_os_family("windows") == "windows"
    assert _get_os_family("darwin") == "darwin"
    assert _get_os_family("linux") == "linux"
    assert _get_os_family("linux2") == "linux"
    assert _get_os_family("freebsd") == "linux"


def test_real_detect_environment():
    profile = detect_environment()
    assert isinstance(profile, EnvironmentProfile)
    assert profile.os_family in ("windows", "linux", "darwin")
    assert profile.python_version != ""
    assert profile.python_executable != ""
    assert profile.gpu_backend in ("cuda", "mps", "cpu")
    assert profile.recommended_runtime in ("docker", "native")

    # Verify properties
    if profile.os_family == "windows":
        assert profile.is_windows is True
        assert profile.is_linux is False
        assert profile.is_darwin is False

    # Verify serialization
    p_dict = profile.to_dict()
    assert p_dict["os_family"] == profile.os_family
    p_json = profile.to_json()
    parsed = json.loads(p_json)
    assert parsed["os_family"] == profile.os_family

    # Verify summary string
    summary = profile.summary()
    assert "MCP OpenCV Engine Environment Profile" in summary
    assert profile.os_family in summary.lower()


def test_environment_profile_immutability():
    profile = detect_environment()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        profile.os_family = "linux"  # type: ignore


def test_mock_detector_docker_active():
    def mock_runner(cmd, timeout):
        cmd_str = " ".join(cmd)
        if "docker --version" in cmd_str:
            return (True, "Docker version 24.0.7, build afdd53b")
        if "docker compose version" in cmd_str:
            return (True, "Docker Compose version v2.23.3")
        if "docker info" in cmd_str:
            return (True, "Server Version: 24.0.7")
        if "tesseract" in cmd_str:
            return (True, "tesseract 5.3.3")
        return (False, "")

    detector = EnvironmentDetector(
        cmd_runner=mock_runner,
        os_family_override="linux",
        cpu_arch_override="x86_64",
        python_version_override="3.11.5",
        memory_gb_override=16.0,
    )
    with patch("shutil.which", return_value="/usr/bin/tesseract"):
        profile = detector.detect(force_cpu=True)

    assert profile.os_family == "linux"
    assert profile.is_linux is True
    assert profile.cpu_arch == "x86_64"
    assert profile.has_docker is True
    assert profile.docker_version == "24.0.7"
    assert profile.docker_compose_available is True
    assert profile.docker_compose_version == "2.23.3"
    assert profile.recommended_runtime == "docker"
    assert profile.tesseract_installed is True
    assert profile.tesseract_version == "5.3.3"
    assert profile.gpu_backend == "cpu"
    assert profile.total_memory_gb == 16.0


def test_mock_detector_no_docker():
    def mock_runner(cmd, timeout):
        return (False, "")

    detector = EnvironmentDetector(
        cmd_runner=mock_runner,
        os_family_override="windows",
        cpu_arch_override="AMD64",
        python_version_override="3.11.9",
        memory_gb_override=8.0,
    )
    with patch("shutil.which", return_value=None):
        profile = detector.detect()

    assert profile.os_family == "windows"
    assert profile.is_windows is True
    assert profile.has_docker is False
    assert profile.docker_compose_available is False
    assert profile.recommended_runtime == "native"


def test_mock_detector_docker_installed_but_daemon_down():
    def mock_runner(cmd, timeout):
        cmd_str = " ".join(cmd)
        if "docker --version" in cmd_str:
            return (True, "Docker version 25.0.0")
        if "docker compose version" in cmd_str:
            return (True, "Docker Compose version v2.24.0")
        if "docker info" in cmd_str:
            return (False, "Cannot connect to the Docker daemon")
        return (False, "")

    detector = EnvironmentDetector(
        cmd_runner=mock_runner,
        os_family_override="darwin",
        cpu_arch_override="arm64",
        python_version_override="3.12.0",
    )
    with patch("shutil.which", return_value="/usr/local/bin/docker"):
        profile = detector.detect(force_cpu=True)

    assert profile.os_family == "darwin"
    assert profile.is_darwin is True
    assert profile.has_docker is True
    # Daemon down means recommended runtime falls back to native
    assert profile.recommended_runtime == "native"
    assert profile.diagnostics["docker_daemon_running"] is False


def test_detect_gpu_cuda():
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.4.1"
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 2
    mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4090"
    mock_torch.version.cuda = "12.1"

    with patch.dict("sys.modules", {"torch": mock_torch}):
        backend, name, count, diag = _detect_gpu(force_cpu=False)
        assert backend == "cuda"
        assert name == "NVIDIA GeForce RTX 4090"
        assert count == 2
        assert diag["cuda_version"] == "12.1"


def test_detect_gpu_mps():
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.4.1"
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True

    with patch.dict("sys.modules", {"torch": mock_torch}):
        backend, name, count, diag = _detect_gpu(force_cpu=False)
        assert backend == "mps"
        assert name == "Apple Silicon MPS"
        assert count == 1


def test_detect_gpu_forced_cpu():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    with patch.dict("sys.modules", {"torch": mock_torch}):
        backend, name, count, diag = _detect_gpu(force_cpu=True)
        assert backend == "cpu"
        assert count == 0
        assert diag["reason"] == "forced_cpu"
