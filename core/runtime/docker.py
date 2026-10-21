"""Docker Compose runtime lifecycle management for MCP OpenCV services."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from core.config.loader import get_config, get_project_root
from core.config.schema import AppConfig
from core.runtime.base import (
    BaseRuntime,
    RuntimeState,
    RuntimeStatus,
    ServiceStatus,
    check_http_endpoint,
)


class DockerRuntime(BaseRuntime):
    """Manages Docker Compose container lifecycle for the MCP OpenCV Engine."""

    DEFAULT_PROJECT_NAME = "mcp-opencv-engine"

    SERVICE_SPECS = {
        "object-detection": {"port_attr": "detection_port", "default_port": 8001},
        "image-processing": {"port_attr": "processing_port", "default_port": 8002},
        "ocr-service": {"port_attr": "ocr_port", "default_port": 8003},
        "mcp-gateway": {"port_attr": "gateway_port", "default_port": 8000},
    }

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        project_root: Optional[Union[str, Path]] = None,
        command_runner: Optional[Callable[..., tuple[int, str, str]]] = None,
        project_name: Optional[str] = None,
        compose_file: Optional[Union[str, Path]] = None,
        override_file: Optional[Union[str, Path]] = None,
    ):
        super().__init__(config=config, project_root=project_root, command_runner=command_runner)
        self.project_name = project_name or os.getenv("COMPOSE_PROJECT_NAME", self.DEFAULT_PROJECT_NAME)
        self.compose_file = (
            Path(compose_file).resolve()
            if compose_file
            else (self.project_root / "docker-compose.yml").resolve()
        )
        self.override_file = (
            Path(override_file).resolve()
            if override_file
            else (self.project_root / ".runtime" / "docker-compose.override.yml").resolve()
        )

    def _default_runner(self, cmd: List[str]) -> tuple[int, str, str]:
        """Execute a CLI command using subprocess without shell=True."""
        try:
            res = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                shell=False,
                timeout=60,
            )
            return res.returncode, res.stdout, res.stderr
        except FileNotFoundError:
            return 127, "", f"Executable not found: {cmd[0]}"
        except subprocess.TimeoutExpired:
            return 124, "", f"Command timed out: {' '.join(cmd)}"
        except Exception as exc:
            return 1, "", str(exc)

    def _execute(self, cmd: List[str]) -> tuple[int, str, str]:
        runner = self.command_runner or self._default_runner
        return runner(cmd)

    def _build_compose_base(self) -> List[str]:
        base = ["docker", "compose", "-p", self.project_name, "-f", str(self.compose_file)]
        if self.override_file.is_file():
            base.extend(["-f", str(self.override_file)])
        return base

    def check_prerequisites(self) -> None:
        """Verify that Docker CLI is available and the daemon is actively running."""
        code, out, err = self._execute(["docker", "--version"])
        if code != 0:
            raise RuntimeError(
                f"Docker CLI is not installed or not in system PATH. ({err.strip()})"
            )

        code, out, err = self._execute(["docker", "info"])
        if code != 0:
            raise RuntimeError(
                "Docker daemon is not running. Please start Docker Desktop or system Docker service."
            )

    def validate_compose(self) -> None:
        """Validate the combined Docker Compose configuration."""
        if not self.compose_file.is_file():
            raise FileNotFoundError(f"Docker Compose file not found at: '{self.compose_file}'")

        cmd = self._build_compose_base() + ["config"]
        code, out, err = self._execute(cmd)
        if code != 0:
            raise RuntimeError(f"Docker Compose configuration is invalid: {err.strip() or out.strip()}")

    def _get_service_port(self, service_name: str) -> int:
        spec = self.SERVICE_SPECS.get(service_name)
        if not spec:
            raise ValueError(f"Unknown service: {service_name}")
        port_attr = spec["port_attr"]
        return getattr(self.config.service, port_attr, spec["default_port"])

    def start(
        self,
        services: Optional[List[str]] = None,
        timeout: float = 45.0,
        build: bool = False,
    ) -> bool:
        """Start the Docker Compose services and wait for readiness.

        Does not rebuild images unless explicitly requested with build=True.
        """
        self.check_prerequisites()
        self.validate_compose()

        cmd = self._build_compose_base() + ["up", "-d"]
        if build:
            cmd.append("--build")
        else:
            cmd.append("--no-build")

        if services:
            cmd.extend(services)

        code, out, err = self._execute(cmd)
        if code != 0:
            # If failed because images do not exist, retry once with standard up -d (allow pulling/building)
            if "no such image" in err.lower() or "not found" in err.lower():
                retry_cmd = self._build_compose_base() + ["up", "-d"]
                if services:
                    retry_cmd.extend(services)
                code, out, err = self._execute(retry_cmd)

            if code != 0:
                raise RuntimeError(f"Failed to start Docker Compose services: {err.strip() or out.strip()}")

        return self.wait_for_ready(timeout=timeout)

    def stop(self, timeout: float = 10.0) -> bool:
        """Stop only containers managed by this Compose project without removing volumes."""
        cmd = self._build_compose_base() + ["stop", "-t", str(int(timeout))]
        code, out, err = self._execute(cmd)
        return code == 0

    def down(self) -> bool:
        """Tear down containers and network for this project without deleting volumes."""
        cmd = self._build_compose_base() + ["down"]
        code, out, err = self._execute(cmd)
        return code == 0

    def get_status(self) -> RuntimeStatus:
        """Inspect container states and verify HTTP health endpoints."""
        statuses: Dict[str, ServiceStatus] = {}
        running_count = 0
        total_services = 0

        # Run docker compose ps to get container status
        cmd = self._build_compose_base() + ["ps", "--format", "json"]
        code, out, err = self._execute(cmd)

        container_states: Dict[str, str] = {}
        if code == 0 and out.strip():
            # Docker compose ps can return JSON lines or a JSON array depending on version
            lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
            for line in lines:
                try:
                    cdata = json.loads(line)
                    if isinstance(cdata, list):
                        for item in cdata:
                            c_svc = item.get("Service") or item.get("Name", "")
                            container_states[c_svc] = item.get("State", "").lower()
                    elif isinstance(cdata, dict):
                        c_svc = cdata.get("Service") or cdata.get("Name", "")
                        container_states[c_svc] = cdata.get("State", "").lower()
                except Exception:
                    pass

        # Check endpoints for each defined service
        for svc_name, spec in self.SERVICE_SPECS.items():
            if svc_name == "mcp-gateway":
                # Only check gateway if configured to run in Docker
                continue

            total_services += 1
            port = self._get_service_port(svc_name)
            health_url = f"http://127.0.0.1:{port}/health"

            c_state = container_states.get(svc_name, "")
            is_healthy = check_http_endpoint(health_url)

            if is_healthy:
                st = RuntimeState.RUNNING
                running_count += 1
            elif c_state == "running":
                st = RuntimeState.STARTING
            elif c_state in ("exited", "dead"):
                st = RuntimeState.ERROR
            else:
                st = RuntimeState.STOPPED

            statuses[svc_name] = ServiceStatus(
                name=svc_name,
                state=st,
                port=port,
                url=f"http://127.0.0.1:{port}",
                health_url=health_url,
            )

        if running_count == total_services and total_services > 0:
            overall = RuntimeState.RUNNING
        elif running_count > 0:
            overall = RuntimeState.PARTIAL
        else:
            has_error = any(s.state == RuntimeState.ERROR for s in statuses.values())
            overall = RuntimeState.ERROR if has_error else RuntimeState.STOPPED

        return RuntimeStatus(
            mode="docker",
            overall_state=overall,
            services=statuses,
        )
