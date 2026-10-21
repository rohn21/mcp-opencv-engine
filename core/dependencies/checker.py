"""Dependency verification and environment preflight checker for MCP OpenCV Engine.

Diagnoses missing Python packages, incompatible versions, absent external tools,
and hardware/runtime prerequisites — without installing or modifying anything.
"""

from __future__ import annotations

import importlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from core.environment.capabilities import EnvironmentProfile, GPUBackend


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Severity level for a dependency check result."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """Result of a single dependency or prerequisite check."""
    name: str
    category: Literal["python_package", "external_tool", "hardware", "runtime"]
    severity: Severity
    message: str
    required: bool = True
    installed_version: Optional[str] = None
    required_version: Optional[str] = None
    install_hint: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.severity in (Severity.OK, Severity.SKIPPED)


@dataclass
class PreflightReport:
    """Aggregated results from a full dependency preflight scan."""
    results: List[CheckResult] = field(default_factory=list)
    environment: Optional[EnvironmentProfile] = None

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results if r.required)

    @property
    def errors(self) -> List[CheckResult]:
        return [r for r in self.results if r.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[CheckResult]:
        return [r for r in self.results if r.severity == Severity.WARNING]

    def summary(self) -> str:
        ok = sum(1 for r in self.results if r.severity == Severity.OK)
        warn = len(self.warnings)
        err = len(self.errors)
        skip = sum(1 for r in self.results if r.severity == Severity.SKIPPED)
        total = len(self.results)

        lines = [
            "=== MCP OpenCV Engine Preflight Report ===",
            f"Total checks: {total}  |  OK: {ok}  |  Warnings: {warn}  |  Errors: {err}  |  Skipped: {skip}",
        ]

        if err > 0:
            lines.append("")
            lines.append("ERRORS (must fix):")
            for r in self.errors:
                hint = f"  → {r.install_hint}" if r.install_hint else ""
                lines.append(f"  ✗ [{r.category}] {r.name}: {r.message}{hint}")

        if warn > 0:
            lines.append("")
            lines.append("WARNINGS (optional, may degrade functionality):")
            for r in self.warnings:
                hint = f"  → {r.install_hint}" if r.install_hint else ""
                lines.append(f"  ⚠ [{r.category}] {r.name}: {r.message}{hint}")

        if err == 0 and warn == 0:
            lines.append("All checks passed.")

        lines.append("==========================================")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Version parsing helpers
# ---------------------------------------------------------------------------

def _parse_version_tuple(version_str: str) -> Tuple[int, ...]:
    """Parse a version string like '2.4.1' into a comparable tuple (2, 4, 1)."""
    parts = re.split(r"[.\-+]", version_str)
    numeric = []
    for p in parts:
        digits = re.match(r"(\d+)", p)
        if digits:
            numeric.append(int(digits.group(1)))
        else:
            break
    return tuple(numeric) if numeric else (0,)


def _version_satisfies(installed: str, spec: str) -> bool:
    """Check if an installed version satisfies a requirement specifier like '>=2.4.1' or '==1.7.1'."""
    spec = spec.strip()
    if not spec:
        return True

    inst_tuple = _parse_version_tuple(installed)

    # Handle compound specs: >=2.4.1,!=2.4.0,<3.0.0
    constraints = [c.strip() for c in spec.split(",") if c.strip()]
    for constraint in constraints:
        match = re.match(r"(>=|<=|==|!=|>|<|~=)\s*([\d\.]+)", constraint)
        if not match:
            continue
        op, ver_str = match.group(1), match.group(2)
        ver_tuple = _parse_version_tuple(ver_str)

        if op == "==" and inst_tuple != ver_tuple:
            return False
        elif op == "!=" and inst_tuple == ver_tuple:
            return False
        elif op == ">=" and inst_tuple < ver_tuple:
            return False
        elif op == "<=" and inst_tuple > ver_tuple:
            return False
        elif op == ">" and inst_tuple <= ver_tuple:
            return False
        elif op == "<" and inst_tuple >= ver_tuple:
            return False
        elif op == "~=":
            if inst_tuple < ver_tuple:
                return False
            if len(ver_tuple) >= 2 and inst_tuple[0] != ver_tuple[0]:
                return False

    return True


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_python_package(
    package_name: str,
    import_name: Optional[str] = None,
    version_spec: Optional[str] = None,
    required: bool = True,
    install_hint: Optional[str] = None,
) -> CheckResult:
    """Check if a Python package is importable and optionally satisfies a version constraint."""
    mod_name = import_name or package_name

    try:
        mod = importlib.import_module(mod_name)
    except ImportError:
        sev = Severity.ERROR if required else Severity.WARNING
        return CheckResult(
            name=package_name,
            category="python_package",
            severity=sev,
            message=f"Package '{package_name}' is not installed.",
            required=required,
            required_version=version_spec,
            install_hint=install_hint or f"pip install {package_name}",
        )

    installed_ver = getattr(mod, "__version__", None)
    if installed_ver is None:
        # Try importlib.metadata as fallback
        try:
            from importlib.metadata import version as meta_version
            installed_ver = meta_version(package_name)
        except Exception:
            pass

    if version_spec and installed_ver:
        if not _version_satisfies(installed_ver, version_spec):
            return CheckResult(
                name=package_name,
                category="python_package",
                severity=Severity.ERROR if required else Severity.WARNING,
                message=f"Version {installed_ver} does not satisfy {version_spec}.",
                required=required,
                installed_version=installed_ver,
                required_version=version_spec,
                install_hint=install_hint or f"pip install '{package_name}{version_spec}'",
            )

    return CheckResult(
        name=package_name,
        category="python_package",
        severity=Severity.OK,
        message=f"Installed ({installed_ver or 'version unknown'}).",
        required=required,
        installed_version=installed_ver,
        required_version=version_spec,
    )


def check_external_tool(
    tool_name: str,
    binary_name: Optional[str] = None,
    version_cmd: Optional[List[str]] = None,
    version_pattern: Optional[str] = None,
    required: bool = True,
    install_hint: Optional[str] = None,
    extra_search_paths: Optional[List[str]] = None,
    cmd_runner: Optional[Callable[[List[str], float], Tuple[bool, str]]] = None,
) -> CheckResult:
    """Check if an external system tool (e.g. tesseract, docker) is available."""
    bin_name = binary_name or tool_name
    found_path = shutil.which(bin_name)

    if not found_path and extra_search_paths:
        import os
        for search_dir in extra_search_paths:
            candidate = Path(search_dir) / bin_name
            if candidate.is_file():
                found_path = str(candidate)
                break
            # Try with .exe on Windows
            candidate_exe = Path(search_dir) / (bin_name + ".exe")
            if candidate_exe.is_file():
                found_path = str(candidate_exe)
                break

    if not found_path:
        return CheckResult(
            name=tool_name,
            category="external_tool",
            severity=Severity.ERROR if required else Severity.WARNING,
            message=f"'{bin_name}' not found on PATH.",
            required=required,
            install_hint=install_hint,
        )

    # Optionally check version
    installed_version = None
    if version_cmd:
        runner = cmd_runner or _default_cmd_runner
        ok, output = runner(version_cmd, 3.0)
        if ok and output and version_pattern:
            match = re.search(version_pattern, output, re.IGNORECASE)
            if match:
                installed_version = match.group(1)

    return CheckResult(
        name=tool_name,
        category="external_tool",
        severity=Severity.OK,
        message=f"Found at {found_path}" + (f" (v{installed_version})" if installed_version else ""),
        required=required,
        installed_version=installed_version,
    )


def check_gpu_backend(
    env_profile: Optional[EnvironmentProfile] = None,
    required_backend: Optional[GPUBackend] = None,
) -> CheckResult:
    """Check GPU/hardware acceleration availability."""
    if env_profile is None:
        from core.environment.detector import detect_environment
        env_profile = detect_environment()

    actual = env_profile.gpu_backend
    gpu_name = env_profile.gpu_name

    if required_backend and required_backend != "cpu" and actual != required_backend:
        return CheckResult(
            name=f"GPU ({required_backend})",
            category="hardware",
            severity=Severity.WARNING,
            message=f"Requested {required_backend.upper()} but detected {actual.upper()}. Will fall back to CPU.",
            required=False,
        )

    if actual == "cpu":
        return CheckResult(
            name="GPU Acceleration",
            category="hardware",
            severity=Severity.WARNING,
            message="No GPU acceleration detected. CPU mode will be used (slower inference).",
            required=False,
        )

    return CheckResult(
        name=f"GPU ({actual})",
        category="hardware",
        severity=Severity.OK,
        message=f"{actual.upper()} available" + (f" ({gpu_name})" if gpu_name else ""),
        required=False,
    )


def check_docker_runtime(
    env_profile: Optional[EnvironmentProfile] = None,
    required: bool = False,
) -> CheckResult:
    """Check Docker and Docker Compose availability and daemon responsiveness."""
    if env_profile is None:
        from core.environment.detector import detect_environment
        env_profile = detect_environment()

    if not env_profile.has_docker:
        return CheckResult(
            name="Docker",
            category="runtime",
            severity=Severity.ERROR if required else Severity.WARNING,
            message="Docker is not installed.",
            required=required,
            install_hint="Install Docker Desktop from https://www.docker.com/products/docker-desktop/",
        )

    daemon_ok = env_profile.diagnostics.get("docker_daemon_running", False)
    compose_ok = env_profile.docker_compose_available

    issues = []
    if not daemon_ok:
        issues.append("Docker daemon is not running")
    if not compose_ok:
        issues.append("Docker Compose is not available")

    if issues:
        return CheckResult(
            name="Docker",
            category="runtime",
            severity=Severity.ERROR if required else Severity.WARNING,
            message="; ".join(issues) + ".",
            required=required,
            installed_version=env_profile.docker_version,
            install_hint="Start Docker Desktop or install Docker Compose plugin.",
        )

    return CheckResult(
        name="Docker",
        category="runtime",
        severity=Severity.OK,
        message=f"Docker {env_profile.docker_version or ''}, Compose {env_profile.docker_compose_version or ''}",
        required=required,
        installed_version=env_profile.docker_version,
    )


def _default_cmd_runner(cmd: List[str], timeout: float) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout, check=False,
        )
        return (proc.returncode == 0, proc.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return (False, "")


# ---------------------------------------------------------------------------
# Service-scoped preflight profiles
# ---------------------------------------------------------------------------

# Canonical dependency definitions per service scope.
# Each entry: (package_name, import_name, version_spec, required)
GATEWAY_PACKAGES: List[Tuple[str, Optional[str], Optional[str], bool]] = [
    ("mcp", "mcp", ">=1.23.0", True),
    ("fastapi", "fastapi", "==0.115.0", True),
    ("uvicorn", "uvicorn", ">=0.30.0", True),
    ("httpx", "httpx", ">=0.27.0", True),
    ("pydantic", "pydantic", ">=2.10.0", True),
]

OBJECT_DETECTION_PACKAGES: List[Tuple[str, Optional[str], Optional[str], bool]] = [
    ("ultralytics", "ultralytics", "==8.3.0", True),
    ("torch", "torch", ">=2.4.1", True),
    ("torchvision", "torchvision", ">=0.19.1", True),
    ("opencv-python-headless", "cv2", ">=4.10.0.84", True),
    ("numpy", "numpy", "==1.26.4", True),
]

IMAGE_PROCESSING_PACKAGES: List[Tuple[str, Optional[str], Optional[str], bool]] = [
    ("opencv-python-headless", "cv2", ">=4.10.0.84", True),
    ("numpy", "numpy", "==1.26.4", True),
]

OCR_PACKAGES: List[Tuple[str, Optional[str], Optional[str], bool]] = [
    ("easyocr", "easyocr", "==1.7.1", True),
    ("pytesseract", "pytesseract", "==0.3.13", True),
    ("opencv-python-headless", "cv2", ">=4.10.0.84", True),
    ("pillow", "PIL", ">=10.0.0", True),
]


def _get_tesseract_install_hint(os_family: str) -> str:
    """Return OS-appropriate install instructions for Tesseract."""
    hints = {
        "windows": "winget install UB-Mannheim.TesseractOCR  (or choco install tesseract)",
        "darwin": "brew install tesseract",
        "linux": "sudo apt install -y tesseract-ocr  (or equivalent for your distro)",
    }
    return hints.get(os_family, "Install Tesseract OCR from https://github.com/tesseract-ocr/tesseract")


def _get_tesseract_extra_paths(os_family: str) -> List[str]:
    """Return common Tesseract installation directories for the given OS."""
    import os
    if os_family == "windows":
        return [
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Tesseract-OCR"),
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Tesseract-OCR"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR"),
        ]
    return []


# ---------------------------------------------------------------------------
# Preflight runner
# ---------------------------------------------------------------------------

class DependencyChecker:
    """Configurable preflight checker that diagnoses without modifying the host."""

    def __init__(
        self,
        env_profile: Optional[EnvironmentProfile] = None,
        cmd_runner: Optional[Callable[[List[str], float], Tuple[bool, str]]] = None,
    ):
        self.env_profile = env_profile
        self.cmd_runner = cmd_runner

    def _get_profile(self) -> EnvironmentProfile:
        if self.env_profile is None:
            from core.environment.detector import detect_environment
            self.env_profile = detect_environment()
        return self.env_profile

    def check_service_packages(
        self,
        scope: Literal["gateway", "object-detection", "image-processing", "ocr", "all"],
    ) -> List[CheckResult]:
        """Check Python packages required for a specific service scope."""
        packages_map: Dict[str, List[Tuple[str, Optional[str], Optional[str], bool]]] = {
            "gateway": GATEWAY_PACKAGES,
            "object-detection": OBJECT_DETECTION_PACKAGES,
            "image-processing": IMAGE_PROCESSING_PACKAGES,
            "ocr": OCR_PACKAGES,
        }

        if scope == "all":
            seen = set()
            combined: List[Tuple[str, Optional[str], Optional[str], bool]] = []
            for pkgs in packages_map.values():
                for entry in pkgs:
                    if entry[0] not in seen:
                        seen.add(entry[0])
                        combined.append(entry)
            target = combined
        else:
            target = packages_map.get(scope, [])

        results = []
        for pkg_name, import_name, version_spec, required in target:
            results.append(
                check_python_package(pkg_name, import_name, version_spec, required)
            )
        return results

    def check_external_tools(self) -> List[CheckResult]:
        """Check external system tools (Tesseract, Docker)."""
        profile = self._get_profile()
        results = []

        # Tesseract OCR
        results.append(check_external_tool(
            tool_name="Tesseract OCR",
            binary_name="tesseract",
            version_cmd=["tesseract", "--version"],
            version_pattern=r"tesseract\s+([0-9\.\w\-]+)",
            required=False,
            install_hint=_get_tesseract_install_hint(profile.os_family),
            extra_search_paths=_get_tesseract_extra_paths(profile.os_family),
            cmd_runner=self.cmd_runner,
        ))

        # Docker
        results.append(check_docker_runtime(env_profile=profile, required=False))

        return results

    def check_hardware(
        self,
        required_backend: Optional[GPUBackend] = None,
    ) -> List[CheckResult]:
        """Check GPU/hardware acceleration."""
        profile = self._get_profile()
        return [check_gpu_backend(env_profile=profile, required_backend=required_backend)]

    def run_preflight(
        self,
        scope: Literal["gateway", "object-detection", "image-processing", "ocr", "all"] = "all",
        check_tools: bool = True,
        check_hw: bool = True,
        required_gpu: Optional[GPUBackend] = None,
    ) -> PreflightReport:
        """Run a full preflight scan for the specified service scope.
        
        This method diagnoses and reports. It does not install anything.
        """
        profile = self._get_profile()
        report = PreflightReport(environment=profile)

        # Python packages for the specified scope
        report.results.extend(self.check_service_packages(scope))

        # External tools
        if check_tools:
            report.results.extend(self.check_external_tools())

        # Hardware acceleration
        if check_hw:
            report.results.extend(self.check_hardware(required_backend=required_gpu))

        return report


def run_preflight(
    scope: Literal["gateway", "object-detection", "image-processing", "ocr", "all"] = "all",
    env_profile: Optional[EnvironmentProfile] = None,
) -> PreflightReport:
    """Convenience function: run a full preflight scan and return the report."""
    checker = DependencyChecker(env_profile=env_profile)
    return checker.run_preflight(scope=scope)
