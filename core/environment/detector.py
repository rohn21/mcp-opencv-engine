"""System and capability detection logic for MCP OpenCV Engine."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.environment.capabilities import EnvironmentProfile, GPUBackend, OSFamily, RuntimeMode


def _get_os_family(sys_platform: Optional[str] = None) -> OSFamily:
    plat = (sys_platform or sys.platform).lower()
    if plat.startswith("win"):
        return "windows"
    if plat.startswith("darwin"):
        return "darwin"
    return "linux"


def _run_command(cmd: List[str], timeout: float = 2.0) -> Tuple[bool, str]:
    """Safely execute a command with a short timeout, returning (success, stdout)."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return (proc.returncode == 0, proc.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return (False, "")


def _detect_total_memory_gb() -> Optional[float]:
    """Detect total system physical memory in Gigabytes."""
    # Try psutil first if installed
    try:
        import psutil  # type: ignore
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        pass

    # Fallback for Windows
    if sys.platform.startswith("win"):
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullTotalPhys / (1024 ** 3), 2)
        except Exception:
            pass

    # Fallback for Linux / Unix
    try:
        if hasattr(os, "sysconf"):
            if "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
                pagesize = os.sysconf("SC_PAGE_SIZE")
                total_pages = os.sysconf("SC_PHYS_PAGES")
                return round((pagesize * total_pages) / (1024 ** 3), 2)
    except Exception:
        pass

    return None


def _detect_docker(
    cmd_runner: Callable[[List[str], float], Tuple[bool, str]] = _run_command
) -> Tuple[bool, Optional[str], bool, Optional[str], bool]:
    """Detect presence and version of docker and docker compose.
    
    Returns:
        (has_docker, docker_version, has_compose, compose_version, daemon_running)
    """
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return (False, None, False, None, False)

    # Check docker CLI version
    ok, out = cmd_runner(["docker", "--version"], 2.0)
    if not ok:
        return (False, None, False, None, False)

    docker_version = None
    ver_match = re.search(r"Docker version\s+([0-9\.]+)", out, re.IGNORECASE)
    if ver_match:
        docker_version = ver_match.group(1)
    else:
        docker_version = out.split("\n")[0] if out else "installed"

    # Check docker compose (plugin 'docker compose' or standalone 'docker-compose')
    has_compose = False
    compose_version = None
    ok_comp, out_comp = cmd_runner(["docker", "compose", "version"], 2.0)
    if ok_comp:
        has_compose = True
        cver_match = re.search(r"version\s+v?([0-9\.]+)", out_comp, re.IGNORECASE)
        compose_version = cver_match.group(1) if cver_match else out_comp.split("\n")[0]
    else:
        if shutil.which("docker-compose"):
            ok_legacy, out_legacy = cmd_runner(["docker-compose", "--version"], 2.0)
            if ok_legacy:
                has_compose = True
                cver_match = re.search(r"version\s+v?([0-9\.]+)", out_legacy, re.IGNORECASE)
                compose_version = cver_match.group(1) if cver_match else out_legacy.split("\n")[0]

    # Verify if Docker daemon is responsive
    daemon_running, _ = cmd_runner(["docker", "info"], 2.5)

    return (True, docker_version, has_compose, compose_version, daemon_running)


def _detect_tesseract(
    os_family: OSFamily,
    cmd_runner: Callable[[List[str], float], Tuple[bool, str]] = _run_command
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Detect Tesseract OCR binary and version.
    
    Returns:
        (tesseract_installed, tesseract_version, tesseract_path)
    """
    tess_path = shutil.which("tesseract")

    if not tess_path and os_family == "windows":
        # Check standard Windows installation directories
        candidates = [
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Tesseract-OCR", "tesseract.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Tesseract-OCR", "tesseract.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
        ]
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                tess_path = candidate
                break

    if not tess_path:
        return (False, None, None)

    ok, out = cmd_runner([tess_path, "--version"], 2.0)
    version = None
    if ok and out:
        ver_match = re.search(r"tesseract\s+([0-9\.\w\-]+)", out, re.IGNORECASE)
        if ver_match:
            version = ver_match.group(1)
        else:
            version = out.split("\n")[0]

    return (True, version, tess_path)


def _detect_gpu(force_cpu: bool = False) -> Tuple[GPUBackend, Optional[str], int, Dict[str, Any]]:
    """Detect GPU hardware acceleration backend (CUDA, Apple MPS, or CPU)."""
    diagnostics: Dict[str, Any] = {}
    if force_cpu:
        return ("cpu", None, 0, {"reason": "forced_cpu"})

    try:
        import torch  # type: ignore
        diagnostics["torch_version"] = getattr(torch, "__version__", "unknown")

        # Check CUDA (NVIDIA)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0) if count > 0 else "NVIDIA GPU"
            cuda_ver = getattr(torch.version, "cuda", None)
            diagnostics["cuda_version"] = cuda_ver
            return ("cuda", name, count, diagnostics)

        # Check MPS (Apple Silicon Metal Performance Shaders)
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available():
                return ("mps", "Apple Silicon MPS", 1, diagnostics)
            else:
                diagnostics["mps_available"] = False

        # Fallback to CPU
        return ("cpu", None, 0, diagnostics)

    except ImportError:
        diagnostics["torch_installed"] = False
        return ("cpu", None, 0, diagnostics)
    except Exception as exc:
        diagnostics["gpu_error"] = str(exc)
        return ("cpu", None, 0, diagnostics)


class EnvironmentDetector:
    """Configurable environment detector supporting dependency injection for testing."""

    def __init__(
        self,
        cmd_runner: Callable[[List[str], float], Tuple[bool, str]] = _run_command,
        os_family_override: Optional[OSFamily] = None,
        cpu_arch_override: Optional[str] = None,
        python_version_override: Optional[str] = None,
        memory_gb_override: Optional[float] = None,
    ):
        self.cmd_runner = cmd_runner
        self.os_family_override = os_family_override
        self.cpu_arch_override = cpu_arch_override
        self.python_version_override = python_version_override
        self.memory_gb_override = memory_gb_override

    def detect(
        self,
        force_cpu: bool = False,
        command_runner: Optional[Callable[[List[str], float], Tuple[bool, str]]] = None,
    ) -> EnvironmentProfile:
        """Detect and return an immutable EnvironmentProfile."""
        runner = command_runner or self.cmd_runner
        os_fam = self.os_family_override or _get_os_family()
        cpu_arch = self.cpu_arch_override or platform.machine() or "unknown"
        py_ver = self.python_version_override or platform.python_version()
        py_exe = sys.executable
        total_mem = self.memory_gb_override if self.memory_gb_override is not None else _detect_total_memory_gb()

        os_name = f"{platform.system()} {platform.release()}".strip()
        os_version = platform.version()

        has_docker, docker_ver, has_compose, compose_ver, daemon_running = _detect_docker(runner)
        has_tess, tess_ver, tess_path = _detect_tesseract(os_fam, runner)
        gpu_backend, gpu_name, gpu_count, gpu_diag = _detect_gpu(force_cpu)

        # Determine recommended runtime
        # If Docker and Compose are installed AND the daemon is active, Docker mode provides
        # the cleanest microservice isolation across platforms. Otherwise native mode is recommended.
        if has_docker and has_compose and daemon_running:
            recommended_runtime: RuntimeMode = "docker"
        else:
            recommended_runtime = "native"

        diagnostics: Dict[str, Any] = {
            "docker_daemon_running": daemon_running,
            "gpu_diagnostics": gpu_diag,
            "tesseract_path": tess_path,
        }

        return EnvironmentProfile(
            os_family=os_fam,
            os_name=os_name,
            os_version=os_version,
            cpu_arch=cpu_arch,
            python_version=py_ver,
            python_executable=py_exe,
            has_docker=has_docker,
            docker_version=docker_ver,
            docker_compose_available=has_compose,
            docker_compose_version=compose_ver,
            gpu_backend=gpu_backend,
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            tesseract_installed=has_tess,
            tesseract_version=tess_ver,
            tesseract_path=tess_path,
            total_memory_gb=total_mem,
            recommended_runtime=recommended_runtime,
            diagnostics=diagnostics,
        )


def detect_environment(force_cpu: bool = False) -> EnvironmentProfile:
    """Convenience function to detect system environment and capabilities."""
    return EnvironmentDetector().detect(force_cpu=force_cpu)
