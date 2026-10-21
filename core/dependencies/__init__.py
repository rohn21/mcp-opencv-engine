"""Dependency management and preflight checking module."""

from core.dependencies.checker import (
    CheckResult,
    DependencyChecker,
    PreflightReport,
    Severity,
    check_docker_runtime,
    check_external_tool,
    check_gpu_backend,
    check_python_package,
    run_preflight,
)

__all__ = [
    "CheckResult",
    "DependencyChecker",
    "PreflightReport",
    "Severity",
    "check_docker_runtime",
    "check_external_tool",
    "check_gpu_backend",
    "check_python_package",
    "run_preflight",
]
