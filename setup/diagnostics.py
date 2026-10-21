"""Read-only system, configuration, runtime, and MCP client diagnostics for MCP OpenCV Engine."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.client_config.base import ConfigScope
from core.client_config.merger import load_json_safe
from core.client_config.registry import list_adapters
from core.config.loader import get_config, get_project_root, load_config
from core.config.schema import AppConfig, MCPTransport, RuntimeMode
from core.dependencies.checker import DependencyChecker, Severity, check_docker_runtime
from core.environment.detector import EnvironmentDetector
from core.filesystem.path_resolver import PathResolver
from core.runtime.docker import DockerRuntime


def probe_service_endpoint(url: str, timeout: float = 0.8) -> Tuple[str, str, Optional[str]]:
    """Probe an endpoint in a strictly read-only manner.

    Distinguishes:
    - PASS: Online and healthy (HTTP 200)
    - FAIL: Active endpoint responded with error (e.g. HTTP 5xx)
    - WARN: Service not running (normal if runtime is intentionally stopped)
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MCPOpenCVDiagnostics/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", None) or getattr(resp, "code", None)
            if code == 200:
                return "PASS", f"Online and healthy (HTTP {code})", None
            else:
                return "FAIL", f"Service returned unexpected HTTP {code}", "Check service logs."
    except urllib.error.HTTPError as exc:
        return "FAIL", f"Service health endpoint returned HTTP error {exc.code}: {exc.reason}", "Check service logs."
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return (
            "WARN",
            "Service not currently running (Normal if runtime is stopped; no active listener detected)",
            "Start the runtime via 'python setup/configure.py --setup' or 'docker compose up -d'.",
        )


@dataclass
class DiagnosticItem:
    category: str
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    details: str
    remediation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "name": self.name,
            "status": self.status,
            "details": self.details,
            "remediation": self.remediation,
        }


@dataclass
class DiagnosticReport:
    timestamp: str
    overall_status: str  # "PASS", "WARN", "FAIL"
    exit_code: int
    items: List[DiagnosticItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "exit_code": self.exit_code,
            "summary": {
                "total": len(self.items),
                "passed": sum(1 for i in self.items if i.status == "PASS"),
                "warnings": sum(1 for i in self.items if i.status == "WARN"),
                "failures": sum(1 for i in self.items if i.status == "FAIL"),
            },
            "checks": [i.to_dict() for i in self.items],
        }

    def to_text(self) -> str:
        lines = [
            f"=== MCP OpenCV Engine Diagnostics ({self.timestamp}) ===",
            f"Overall Result: [{self.overall_status}] (exit code {self.exit_code})\n",
        ]

        # Group by category
        categories: Dict[str, List[DiagnosticItem]] = {}
        for item in self.items:
            categories.setdefault(item.category, []).append(item)

        for cat, cat_items in categories.items():
            lines.append(f"--- {cat} ---")
            for i in cat_items:
                marker = f"[{i.status}]"
                lines.append(f"  {marker:<8} {i.name}: {i.details}")
                if i.remediation and i.status in ("WARN", "FAIL"):
                    lines.append(f"           Remedy: {i.remediation}")
            lines.append("")

        summary = self.to_dict()["summary"]
        lines.append(
            f"Summary: {summary['passed']} passed, {summary['warnings']} warnings, {summary['failures']} failures."
        )
        return "\n".join(lines)


class SystemDiagnostics:
    """Coordinates read-only health and configuration diagnostics.

    Guarantees:
    - Never mutates filesystem or configuration.
    - Never starts, stops, or restarts processes or containers.
    - Never installs packages or downloads models.
    - Never exposes secrets or environment variable values.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        project_root: Optional[Union[str, Path]] = None,
        command_runner: Optional[Callable[..., tuple[int, str, str]]] = None,
    ):
        self.project_root = Path(project_root).resolve() if project_root else get_project_root()
        self.command_runner = command_runner
        self._config_error: Optional[str] = None

        if config is not None:
            self.config: Optional[AppConfig] = config
        else:
            try:
                if project_root is not None:
                    self.config = load_config(env_file=self.project_root / ".env")
                else:
                    self.config = get_config(reload=True)
            except Exception as exc:
                self.config = None
                self._config_error = str(exc)

    def run(self) -> DiagnosticReport:
        items: List[DiagnosticItem] = []

        # 1. Environment & Platform
        profile = self._check_environment(items)

        # 2. Dependencies & Tools
        self._check_dependencies(items)

        # 3. Docker Environment & Compose Validity
        self._check_docker(items, profile)

        # 4. Configuration & Storage Boundaries
        self._check_config_and_storage(items)

        # 5. Model Weights
        self._check_model_weights(items)

        # 6. Service Health (read-only probing)
        self._check_service_health(items)

        # 7. MCP Clients
        self._check_mcp_clients(items)

        # Calculate overall status and exit code
        fail_count = sum(1 for i in items if i.status == "FAIL")
        warn_count = sum(1 for i in items if i.status == "WARN")

        if fail_count > 0:
            overall = "FAIL"
            exit_code = 1
        elif warn_count > 0:
            overall = "WARN"
            exit_code = 2
        else:
            overall = "PASS"
            exit_code = 0

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return DiagnosticReport(
            timestamp=timestamp,
            overall_status=overall,
            exit_code=exit_code,
            items=items,
        )

    def _check_environment(self, items: List[DiagnosticItem]) -> Any:
        detector = (
            EnvironmentDetector(cmd_runner=self.command_runner)
            if self.command_runner
            else EnvironmentDetector()
        )
        profile = detector.detect()

        items.append(
            DiagnosticItem(
                category="Environment",
                name="OS & Architecture",
                status="PASS",
                details=f"{profile.os_family} ({profile.cpu_arch})",
            )
        )
        items.append(
            DiagnosticItem(
                category="Environment",
                name="Python Executable",
                status="PASS",
                details=f"Python {profile.python_version} ({sys.executable})",
            )
        )

        # Hardware acceleration
        gpu_status = "PASS" if profile.gpu_backend in ("cuda", "mps") else "WARN"
        gpu_msg = (
            f"Hardware acceleration: {profile.gpu_backend.upper()}"
            if profile.gpu_backend != "cpu"
            else "Running on CPU (No CUDA/MPS GPU detected; vision tasks will use CPU)"
        )
        items.append(
            DiagnosticItem(
                category="Environment",
                name="Compute Device",
                status=gpu_status,
                details=gpu_msg,
                remediation="Install NVIDIA CUDA drivers or run on Apple Silicon if hardware acceleration is desired."
                if profile.gpu_backend == "cpu"
                else None,
            )
        )
        return profile

    def _check_dependencies(self, items: List[DiagnosticItem]) -> None:
        checker = DependencyChecker(cmd_runner=self.command_runner)

        # Preflight packages across all service scopes
        seen_packages = set()
        for scope in ["gateway", "object-detection", "image-processing", "ocr"]:
            pkg_results = checker.check_service_packages(scope)
            for res in pkg_results:
                if res.name in seen_packages:
                    continue
                seen_packages.add(res.name)
                st = "PASS" if res.passed else ("WARN" if res.severity == Severity.WARNING else "FAIL")
                items.append(
                    DiagnosticItem(
                        category="Dependencies",
                        name=f"Package: {res.name}",
                        status=st,
                        details=res.message,
                        remediation=res.install_hint,
                    )
                )

        # External tools: Tesseract
        tess_res = checker.check_external_tools()
        for res in tess_res:
            if res.name == "Docker":
                continue  # Handled separately in _check_docker
            st = "PASS" if res.passed else ("WARN" if res.severity == Severity.WARNING else "FAIL")
            items.append(
                DiagnosticItem(
                    category="Dependencies",
                    name=f"External Tool: {res.name}",
                    status=st,
                    details=res.message,
                    remediation=res.install_hint,
                )
            )

    def _check_docker(self, items: List[DiagnosticItem], profile: Any) -> None:
        docker_res = check_docker_runtime(env_profile=profile, required=False)
        docker_st = "PASS" if docker_res.passed else ("WARN" if docker_res.severity == Severity.WARNING else "FAIL")
        items.append(
            DiagnosticItem(
                category="Docker",
                name="Docker CLI & Daemon",
                status=docker_st,
                details=docker_res.message,
                remediation=docker_res.install_hint,
            )
        )

        compose_file = self.project_root / "docker-compose.yml"
        if not compose_file.is_file():
            items.append(
                DiagnosticItem(
                    category="Docker",
                    name="docker-compose.yml",
                    status="FAIL",
                    details=f"Base docker-compose.yml not found at: {compose_file}",
                    remediation="Ensure docker-compose.yml is present in the project root.",
                )
            )
            return

        items.append(
            DiagnosticItem(
                category="Docker",
                name="docker-compose.yml",
                status="PASS",
                details=f"Present ({compose_file.name})",
            )
        )

        # Check Compose configuration validity when Docker is selected or daemon is active
        is_docker_mode = (
            self.config is not None and self.config.runtime_mode == RuntimeMode.DOCKER
        )
        daemon_active = getattr(profile, "diagnostics", {}).get("docker_daemon_running", False)

        if daemon_active:
            try:
                runtime = DockerRuntime(
                    config=self.config,
                    project_root=self.project_root,
                    command_runner=self.command_runner,
                )
                runtime.validate_compose()
                items.append(
                    DiagnosticItem(
                        category="Docker",
                        name="Docker Compose Configuration",
                        status="PASS",
                        details="Compose configuration syntax and service graph are valid",
                    )
                )
            except Exception as exc:
                items.append(
                    DiagnosticItem(
                        category="Docker",
                        name="Docker Compose Configuration",
                        status="FAIL",
                        details=f"Compose configuration error: {exc}",
                        remediation="Check docker-compose.yml service definitions and environment variables.",
                    )
                )
        elif is_docker_mode:
            items.append(
                DiagnosticItem(
                    category="Docker",
                    name="Docker Compose Configuration",
                    status="WARN",
                    details="Compose configuration validation skipped (Docker daemon is not running)",
                    remediation="Start Docker Desktop or Docker service to validate container configuration.",
                )
            )

    def _check_config_and_storage(self, items: List[DiagnosticItem]) -> None:
        if self._config_error or self.config is None:
            items.append(
                DiagnosticItem(
                    category="Configuration",
                    name="Configuration Loading",
                    status="FAIL",
                    details=f"Configuration failed to load: {self._config_error}",
                    remediation="Review .env file for invalid values or syntax errors.",
                )
            )
            return

        items.append(
            DiagnosticItem(
                category="Configuration",
                name="Configuration Loading",
                status="PASS",
                details=f"Profile: {self.config.profile.value}, Runtime Mode: {self.config.runtime_mode.value}, Gateway Port: {self.config.service.gateway_port}",
            )
        )

        resolver = PathResolver(self.config, project_root=self.project_root)

        # Inputs directory
        inputs_dir = resolver.inputs_dir
        if inputs_dir.is_dir():
            can_read = os.access(inputs_dir, os.R_OK)
            items.append(
                DiagnosticItem(
                    category="Filesystem",
                    name="Inputs Directory",
                    status="PASS" if can_read else "FAIL",
                    details=f"Accessible (Read: {'Yes' if can_read else 'No'}): {inputs_dir}",
                    remediation=None if can_read else "Grant read permissions to data/inputs directory.",
                )
            )
        else:
            items.append(
                DiagnosticItem(
                    category="Filesystem",
                    name="Inputs Directory",
                    status="WARN",
                    details=f"Directory does not exist yet: {inputs_dir}",
                    remediation="Create directory data/inputs to place input images.",
                )
            )

        # Outputs directory
        outputs_dir = resolver.outputs_dir
        if outputs_dir.is_dir():
            can_write = os.access(outputs_dir, os.W_OK)
            items.append(
                DiagnosticItem(
                    category="Filesystem",
                    name="Outputs Directory",
                    status="PASS" if can_write else "FAIL",
                    details=f"Accessible (Write: {'Yes' if can_write else 'No'}): {outputs_dir}",
                    remediation=None if can_write else "Grant write permissions to data/outputs directory.",
                )
            )
        else:
            items.append(
                DiagnosticItem(
                    category="Filesystem",
                    name="Outputs Directory",
                    status="WARN",
                    details=f"Directory does not exist yet: {outputs_dir}",
                    remediation="Directory will be auto-created upon first image processing operation.",
                )
            )

        # Runtime path mappings
        if self.config.runtime_mode == RuntimeMode.DOCKER:
            for host_p, cont_p in resolver.mount_pairs:
                items.append(
                    DiagnosticItem(
                        category="Filesystem",
                        name="Path Mapping (Docker)",
                        status="PASS",
                        details=f"Host '{host_p}' -> Container '{cont_p}'",
                    )
                )
        else:
            items.append(
                DiagnosticItem(
                    category="Filesystem",
                    name="Path Mapping (Native)",
                    status="PASS",
                    details=f"Direct host paths: data_dir='{resolver.data_dir}'",
                )
            )

    def _check_model_weights(self, items: List[DiagnosticItem]) -> None:
        if self.config is None:
            return

        weights_name = self.config.model.yolo_model_name
        weights_dir = self.config.model.yolo_weights_dir or self.project_root
        target = Path(weights_dir) / weights_name

        if target.is_file():
            size_mb = target.stat().st_size / (1024 * 1024)
            items.append(
                DiagnosticItem(
                    category="Models",
                    name="YOLO Weights",
                    status="PASS",
                    details=f"Available: '{weights_name}' ({size_mb:.1f} MB)",
                )
            )
        else:
            items.append(
                DiagnosticItem(
                    category="Models",
                    name="YOLO Weights",
                    status="FAIL",
                    details=f"Model file '{weights_name}' not found at '{target}'.",
                    remediation=f"Place {weights_name} in project root or configure YOLO_WEIGHTS_DIR.",
                )
            )

    def _check_service_health(self, items: List[DiagnosticItem]) -> None:
        """Probe endpoints in a strictly read-only manner, distinguishing stopped from failed."""
        endpoints = {
            "Object Detection": ("http://127.0.0.1:8001/health", 8001),
            "Image Processing": ("http://127.0.0.1:8002/health", 8002),
            "OCR Service": ("http://127.0.0.1:8003/health", 8003),
        }

        for svc, (url, port) in endpoints.items():
            status, details, remediation = probe_service_endpoint(url, timeout=0.8)
            items.append(
                DiagnosticItem(
                    category="Services",
                    name=svc,
                    status=status,
                    details=f"{details} (Port {port})",
                    remediation=remediation,
                )
            )

    def _check_mcp_clients(self, items: List[DiagnosticItem]) -> None:
        if self.config is None:
            return

        adapters = list_adapters(config=self.config, project_root=self.project_root)

        # First scan to see if at least one client has registered the gateway
        any_registered = False
        client_statuses = []
        for adapter in adapters:
            for scope in adapter.supported_scopes:
                path = adapter.discover_config_path(scope)
                scope_label = "Workspace" if scope == ConfigScope.WORKSPACE else "Global"
                if path.is_file():
                    try:
                        data = load_json_safe(path)
                        has_gateway = "cv-mcp-gateway" in data.get("mcpServers", {})
                        if has_gateway:
                            any_registered = True
                        client_statuses.append((adapter, scope_label, path, True, has_gateway, None))
                    except Exception as exc:
                        client_statuses.append((adapter, scope_label, path, True, False, str(exc)))
                else:
                    client_statuses.append((adapter, scope_label, path, False, False, None))

        for adapter, scope_label, path, exists, has_gateway, err in client_statuses:
            if err:
                items.append(
                    DiagnosticItem(
                        category="MCP Clients",
                        name=f"{adapter.display_name} ({scope_label})",
                        status="FAIL",
                        details=f"Config file at {path} contains errors: {err}",
                        remediation="Fix JSON syntax error in configuration file.",
                    )
                )
            elif exists and has_gateway:
                items.append(
                    DiagnosticItem(
                        category="MCP Clients",
                        name=f"{adapter.display_name} ({scope_label})",
                        status="PASS",
                        details=f"Config found: {path} (Registered: 'cv-mcp-gateway' is PRESENT)",
                    )
                )
            elif exists and not has_gateway:
                items.append(
                    DiagnosticItem(
                        category="MCP Clients",
                        name=f"{adapter.display_name} ({scope_label})",
                        status="WARN",
                        details=f"Config found: {path} ('cv-mcp-gateway' not registered)",
                        remediation=f"Run 'python setup/configure.py --configure-client {adapter.client_id}'",
                    )
                )
            else:
                # Config does not exist on disk
                status = "PASS" if any_registered else "WARN"
                details = f"Not configured at: {path} (Optional IDE)" if any_registered else f"Not configured at: {path}"
                remediation = None if any_registered else f"Run 'python setup/configure.py --configure-client {adapter.client_id}' to initialize."
                items.append(
                    DiagnosticItem(
                        category="MCP Clients",
                        name=f"{adapter.display_name} ({scope_label})",
                        status=status,
                        details=details,
                        remediation=remediation,
                    )
                )
