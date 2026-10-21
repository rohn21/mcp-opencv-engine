"""Unit tests for core.dependencies module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from core.dependencies.checker import (
    CheckResult,
    DependencyChecker,
    PreflightReport,
    Severity,
    _parse_version_tuple,
    _version_satisfies,
    check_docker_runtime,
    check_external_tool,
    check_gpu_backend,
    check_python_package,
    run_preflight,
)
from core.environment.capabilities import EnvironmentProfile


# ---------------------------------------------------------------------------
# Version parsing and comparison tests
# ---------------------------------------------------------------------------

class TestVersionParsing:
    def test_simple_version(self):
        assert _parse_version_tuple("2.4.1") == (2, 4, 1)

    def test_version_with_suffix(self):
        assert _parse_version_tuple("1.7.1rc1") == (1, 7, 1)

    def test_single_digit(self):
        assert _parse_version_tuple("3") == (3,)

    def test_version_satisfies_ge(self):
        assert _version_satisfies("2.4.1", ">=2.4.0") is True
        assert _version_satisfies("2.4.0", ">=2.4.1") is False
        assert _version_satisfies("2.4.1", ">=2.4.1") is True

    def test_version_satisfies_eq(self):
        assert _version_satisfies("1.7.1", "==1.7.1") is True
        assert _version_satisfies("1.7.2", "==1.7.1") is False

    def test_version_satisfies_ne(self):
        assert _version_satisfies("2.4.0", "!=2.4.0") is False
        assert _version_satisfies("2.4.1", "!=2.4.0") is True

    def test_version_satisfies_compound(self):
        assert _version_satisfies("2.4.1", ">=2.4.1,!=2.4.0") is True
        assert _version_satisfies("2.4.0", ">=2.4.1,!=2.4.0") is False

    def test_version_satisfies_lt(self):
        assert _version_satisfies("2.3.0", "<3.0.0") is True
        assert _version_satisfies("3.0.0", "<3.0.0") is False

    def test_version_satisfies_empty_spec(self):
        assert _version_satisfies("1.0.0", "") is True


# ---------------------------------------------------------------------------
# Python package check tests
# ---------------------------------------------------------------------------

class TestCheckPythonPackage:
    def test_installed_package(self):
        result = check_python_package("pydantic", "pydantic", ">=2.0.0")
        assert result.severity == Severity.OK
        assert result.installed_version is not None

    def test_missing_package(self):
        result = check_python_package(
            "totally_nonexistent_pkg_12345",
            required=True,
        )
        assert result.severity == Severity.ERROR
        assert "not installed" in result.message

    def test_missing_optional_package(self):
        result = check_python_package(
            "totally_nonexistent_pkg_12345",
            required=False,
        )
        assert result.severity == Severity.WARNING
        assert result.passed is False

    def test_version_mismatch(self):
        # pydantic is installed but require an impossibly high version
        result = check_python_package("pydantic", "pydantic", ">=999.0.0")
        assert result.severity == Severity.ERROR
        assert "does not satisfy" in result.message

    def test_install_hint_present(self):
        result = check_python_package(
            "missing_pkg_xyz", required=True,
            install_hint="pip install missing_pkg_xyz==1.0",
        )
        assert result.install_hint == "pip install missing_pkg_xyz==1.0"


# ---------------------------------------------------------------------------
# External tool check tests
# ---------------------------------------------------------------------------

class TestCheckExternalTool:
    def test_python_on_path(self):
        result = check_external_tool(
            "Python", binary_name="python",
            version_cmd=["python", "--version"],
            version_pattern=r"Python\s+([0-9\.]+)",
        )
        assert result.severity == Severity.OK
        assert "Found" in result.message

    def test_missing_tool_required(self):
        result = check_external_tool(
            "nonexistent_binary_xyz", required=True,
            install_hint="Install xyz from https://example.com",
        )
        assert result.severity == Severity.ERROR

    def test_missing_tool_optional(self):
        result = check_external_tool(
            "nonexistent_binary_xyz", required=False,
        )
        assert result.severity == Severity.WARNING

    def test_tool_with_mocked_runner(self):
        def mock_runner(cmd, timeout):
            return (True, "tesseract 5.3.3")

        with patch("shutil.which", return_value="/usr/bin/tesseract"):
            result = check_external_tool(
                "Tesseract OCR", binary_name="tesseract",
                version_cmd=["tesseract", "--version"],
                version_pattern=r"tesseract\s+([0-9\.]+)",
                cmd_runner=mock_runner,
            )
        assert result.severity == Severity.OK
        assert result.installed_version == "5.3.3"


# ---------------------------------------------------------------------------
# GPU / hardware check tests
# ---------------------------------------------------------------------------

class TestCheckGpuBackend:
    def _make_profile(self, backend: str = "cpu", name=None, count=0) -> EnvironmentProfile:
        return EnvironmentProfile(
            os_family="linux",
            os_name="Linux",
            os_version="5.15",
            cpu_arch="x86_64",
            python_version="3.11.9",
            python_executable="/usr/bin/python3",
            has_docker=False,
            gpu_backend=backend,
            gpu_name=name,
            gpu_count=count,
        )

    def test_cpu_only(self):
        profile = self._make_profile("cpu")
        result = check_gpu_backend(env_profile=profile)
        assert result.severity == Severity.WARNING
        assert "CPU mode" in result.message

    def test_cuda_available(self):
        profile = self._make_profile("cuda", "NVIDIA RTX 4090", 1)
        result = check_gpu_backend(env_profile=profile)
        assert result.severity == Severity.OK
        assert "CUDA" in result.message

    def test_mps_available(self):
        profile = self._make_profile("mps", "Apple Silicon MPS", 1)
        result = check_gpu_backend(env_profile=profile)
        assert result.severity == Severity.OK

    def test_requested_cuda_but_got_cpu(self):
        profile = self._make_profile("cpu")
        result = check_gpu_backend(env_profile=profile, required_backend="cuda")
        assert result.severity == Severity.WARNING
        assert "fall back" in result.message.lower()


# ---------------------------------------------------------------------------
# Docker runtime check tests
# ---------------------------------------------------------------------------

class TestCheckDockerRuntime:
    def _make_profile(self, has_docker, daemon_running, has_compose, docker_ver=None, compose_ver=None):
        return EnvironmentProfile(
            os_family="linux",
            os_name="Linux",
            os_version="5.15",
            cpu_arch="x86_64",
            python_version="3.11.9",
            python_executable="/usr/bin/python3",
            has_docker=has_docker,
            docker_version=docker_ver,
            docker_compose_available=has_compose,
            docker_compose_version=compose_ver,
            gpu_backend="cpu",
            diagnostics={"docker_daemon_running": daemon_running},
        )

    def test_docker_fully_available(self):
        profile = self._make_profile(True, True, True, "24.0.7", "2.23.3")
        result = check_docker_runtime(env_profile=profile, required=False)
        assert result.severity == Severity.OK

    def test_docker_not_installed_optional(self):
        profile = self._make_profile(False, False, False)
        result = check_docker_runtime(env_profile=profile, required=False)
        assert result.severity == Severity.WARNING

    def test_docker_not_installed_required(self):
        profile = self._make_profile(False, False, False)
        result = check_docker_runtime(env_profile=profile, required=True)
        assert result.severity == Severity.ERROR

    def test_docker_daemon_down(self):
        profile = self._make_profile(True, False, True, "24.0.7", "2.23.3")
        result = check_docker_runtime(env_profile=profile, required=False)
        assert result.severity == Severity.WARNING
        assert "daemon" in result.message.lower()


# ---------------------------------------------------------------------------
# PreflightReport tests
# ---------------------------------------------------------------------------

class TestPreflightReport:
    def test_all_passed(self):
        report = PreflightReport(results=[
            CheckResult("pkg_a", "python_package", Severity.OK, "OK", required=True),
            CheckResult("pkg_b", "python_package", Severity.OK, "OK", required=True),
        ])
        assert report.all_passed is True
        assert len(report.errors) == 0

    def test_required_failure(self):
        report = PreflightReport(results=[
            CheckResult("pkg_a", "python_package", Severity.OK, "OK", required=True),
            CheckResult("pkg_b", "python_package", Severity.ERROR, "Missing", required=True),
        ])
        assert report.all_passed is False
        assert len(report.errors) == 1

    def test_optional_warning_still_passes(self):
        report = PreflightReport(results=[
            CheckResult("pkg_a", "python_package", Severity.OK, "OK", required=True),
            CheckResult("gpu", "hardware", Severity.WARNING, "No GPU", required=False),
        ])
        assert report.all_passed is True
        assert len(report.warnings) == 1

    def test_summary_output(self):
        report = PreflightReport(results=[
            CheckResult("pkg_a", "python_package", Severity.OK, "OK", required=True),
            CheckResult("docker", "runtime", Severity.WARNING, "Not installed", required=False),
        ])
        summary = report.summary()
        assert "Preflight Report" in summary
        assert "Warnings" in summary or "WARNINGS" in summary


# ---------------------------------------------------------------------------
# DependencyChecker scoped preflight tests
# ---------------------------------------------------------------------------

class TestDependencyChecker:
    def _make_profile(self):
        return EnvironmentProfile(
            os_family="windows",
            os_name="Windows 10",
            os_version="10.0",
            cpu_arch="AMD64",
            python_version="3.11.9",
            python_executable="C:\\Python\\python.exe",
            has_docker=True,
            docker_version="24.0.7",
            docker_compose_available=True,
            docker_compose_version="2.23.3",
            gpu_backend="cpu",
            diagnostics={"docker_daemon_running": True},
        )

    def test_gateway_scope(self):
        checker = DependencyChecker(env_profile=self._make_profile())
        results = checker.check_service_packages("gateway")
        pkg_names = [r.name for r in results]
        assert "mcp" in pkg_names
        assert "fastapi" in pkg_names
        assert "httpx" in pkg_names
        # Gateway must NOT pull in heavy CV packages
        assert "ultralytics" not in pkg_names
        assert "torch" not in pkg_names
        assert "easyocr" not in pkg_names

    def test_object_detection_scope(self):
        checker = DependencyChecker(env_profile=self._make_profile())
        results = checker.check_service_packages("object-detection")
        pkg_names = [r.name for r in results]
        assert "ultralytics" in pkg_names
        assert "torch" in pkg_names
        # Object detection should NOT pull in OCR packages
        assert "easyocr" not in pkg_names
        assert "pytesseract" not in pkg_names

    def test_ocr_scope(self):
        checker = DependencyChecker(env_profile=self._make_profile())
        results = checker.check_service_packages("ocr")
        pkg_names = [r.name for r in results]
        assert "easyocr" in pkg_names
        assert "pytesseract" in pkg_names
        # OCR should NOT pull in detection packages
        assert "ultralytics" not in pkg_names
        assert "torch" not in pkg_names

    def test_image_processing_scope(self):
        checker = DependencyChecker(env_profile=self._make_profile())
        results = checker.check_service_packages("image-processing")
        pkg_names = [r.name for r in results]
        assert "opencv-python-headless" in pkg_names
        assert "numpy" in pkg_names
        assert "torch" not in pkg_names
        assert "easyocr" not in pkg_names

    def test_all_scope_no_duplicates(self):
        checker = DependencyChecker(env_profile=self._make_profile())
        results = checker.check_service_packages("all")
        pkg_names = [r.name for r in results]
        # Ensure no duplicates
        assert len(pkg_names) == len(set(pkg_names))

    def test_full_preflight_report(self):
        checker = DependencyChecker(env_profile=self._make_profile())
        report = checker.run_preflight(scope="gateway", check_tools=False, check_hw=False)
        assert isinstance(report, PreflightReport)
        assert report.environment is not None
        assert len(report.results) > 0


# ---------------------------------------------------------------------------
# Convenience function test
# ---------------------------------------------------------------------------

class TestRunPreflight:
    def test_run_preflight_gateway(self):
        profile = EnvironmentProfile(
            os_family="linux",
            os_name="Linux",
            os_version="5.15",
            cpu_arch="x86_64",
            python_version="3.11.9",
            python_executable="/usr/bin/python3",
            has_docker=False,
            gpu_backend="cpu",
            diagnostics={"docker_daemon_running": False},
        )
        report = run_preflight(scope="gateway", env_profile=profile)
        assert isinstance(report, PreflightReport)
        assert len(report.results) > 0
        summary = report.summary()
        assert "Preflight Report" in summary
