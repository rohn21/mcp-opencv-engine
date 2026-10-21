"""Native process lifecycle management for MCP OpenCV services."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from core.config.loader import get_config, get_project_root
from core.config.schema import AppConfig
from core.runtime.base import (
    BaseRuntime,
    RuntimeState,
    RuntimeStatus,
    ServiceStatus,
    check_http_endpoint,
)


class NativeRuntime(BaseRuntime):
    """Manages Python microservices as native host subprocesses."""

    SERVICE_SPECS = {
        "object-detection": {
            "app_target": "opencv_services.object_detection.api:app",
            "port_attr": "detection_port",
            "default_port": 8001,
        },
        "image-processing": {
            "app_target": "opencv_services.image_processing.api:app",
            "port_attr": "processing_port",
            "default_port": 8002,
        },
        "ocr-service": {
            "app_target": "opencv_services.ocr_service.api:app",
            "port_attr": "ocr_port",
            "default_port": 8003,
        },
    }

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        project_root: Optional[Union[str, Path]] = None,
        python_executable: Optional[Union[str, Path]] = None,
    ):
        super().__init__(config=config, project_root=project_root)
        self.python_executable = self._resolve_python(python_executable)
        self._processes: Dict[str, subprocess.Popen] = {}
        self._log_files: Dict[str, any] = {}
        self._log_dir = self.project_root / ".runtime" / "logs"

    def _resolve_python(self, custom_path: Optional[Union[str, Path]] = None) -> Path:
        """Find and validate the Python executable (preferring active virtualenv)."""
        if custom_path:
            p = Path(custom_path).resolve()
            if p.is_file():
                return p

        # Check virtualenv in project root
        if sys.platform == "win32":
            venv_python = self.project_root / ".venv" / "Scripts" / "python.exe"
        else:
            venv_python = self.project_root / ".venv" / "bin" / "python"

        if venv_python.is_file():
            return venv_python.resolve()

        return Path(sys.executable).resolve()

    def _get_service_port(self, service_name: str) -> int:
        spec = self.SERVICE_SPECS.get(service_name)
        if not spec:
            raise ValueError(f"Unknown service: {service_name}")
        port_attr = spec["port_attr"]
        return getattr(self.config.service, port_attr, spec["default_port"])

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        root_str = str(self.project_root)
        current_pp = env.get("PYTHONPATH", "")
        paths = [root_str, str(self.project_root / "mcp_gateway")]
        if current_pp:
            paths.append(current_pp)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env["DATA_DIR"] = str(self.config.storage.data_dir)
        return env

    def start_service(self, service_name: str) -> ServiceStatus:
        """Start a single service as a subprocess if not already running."""
        spec = self.SERVICE_SPECS.get(service_name)
        if not spec:
            return ServiceStatus(name=service_name, state=RuntimeState.ERROR, error=f"Unknown service: {service_name}")

        port = self._get_service_port(service_name)
        health_url = f"http://127.0.0.1:{port}/health"

        # Check if already running in managed state
        proc = self._processes.get(service_name)
        if proc is not None and proc.poll() is None:
            # Still running
            return ServiceStatus(
                name=service_name,
                state=RuntimeState.RUNNING if check_http_endpoint(health_url) else RuntimeState.STARTING,
                port=port,
                url=f"http://127.0.0.1:{port}",
                health_url=health_url,
                pid=proc.pid,
            )

        # Check if port is already active externally (e.g. developer started it manually)
        if check_http_endpoint(health_url):
            return ServiceStatus(
                name=service_name,
                state=RuntimeState.RUNNING,
                port=port,
                url=f"http://127.0.0.1:{port}",
                health_url=health_url,
            )

        # Prepare log directory and file
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"{service_name}.log"
        log_file = open(log_path, "a", encoding="utf-8")
        self._log_files[service_name] = log_file

        cmd = [
            str(self.python_executable),
            "-m",
            "uvicorn",
            spec["app_target"],
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]

        try:
            p = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                env=self._build_env(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                # Explicit argument list, no shell=True
                shell=False,
            )
            self._processes[service_name] = p
            return ServiceStatus(
                name=service_name,
                state=RuntimeState.STARTING,
                port=port,
                url=f"http://127.0.0.1:{port}",
                health_url=health_url,
                pid=p.pid,
            )
        except Exception as exc:
            return ServiceStatus(
                name=service_name,
                state=RuntimeState.ERROR,
                port=port,
                error=f"Failed to launch service: {exc}",
            )

    def start(
        self,
        services: Optional[List[str]] = None,
        timeout: float = 30.0,
    ) -> bool:
        """Start all or selected microservices natively and wait for readiness."""
        target_services = services or list(self.SERVICE_SPECS.keys())

        for svc in target_services:
            self.start_service(svc)

        return self.wait_for_ready(timeout=timeout)

    def stop_service(self, service_name: str, timeout: float = 5.0) -> bool:
        """Gracefully terminate a single managed service subprocess."""
        proc = self._processes.get(service_name)
        if proc is None:
            return True

        if proc.poll() is None:
            # Terminate gracefully
            try:
                proc.terminate()
                proc.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except OSError:
                    pass

        # Close log file
        log_file = self._log_files.pop(service_name, None)
        if log_file and not log_file.closed:
            try:
                log_file.close()
            except Exception:
                pass

        self._processes.pop(service_name, None)
        return True

    def stop(self, timeout: float = 10.0) -> bool:
        """Stop all subprocesses started by this runtime manager."""
        per_service_timeout = max(2.0, timeout / max(1, len(self._processes)))
        success = True
        for svc_name in list(self._processes.keys()):
            if not self.stop_service(svc_name, timeout=per_service_timeout):
                success = False
        return success

    def get_status(self) -> RuntimeStatus:
        """Query state and health endpoints of all native services."""
        statuses: Dict[str, ServiceStatus] = {}
        running_count = 0
        total_services = len(self.SERVICE_SPECS)

        for svc_name, spec in self.SERVICE_SPECS.items():
            port = self._get_service_port(svc_name)
            health_url = f"http://127.0.0.1:{port}/health"
            proc = self._processes.get(svc_name)

            pid = proc.pid if proc else None
            is_alive = proc is not None and proc.poll() is None
            is_healthy = check_http_endpoint(health_url)

            if is_healthy:
                st = RuntimeState.RUNNING
                running_count += 1
            elif is_alive:
                st = RuntimeState.STARTING
            elif proc is not None and proc.poll() is not None:
                st = RuntimeState.ERROR
            else:
                st = RuntimeState.STOPPED

            statuses[svc_name] = ServiceStatus(
                name=svc_name,
                state=st,
                port=port,
                url=f"http://127.0.0.1:{port}",
                health_url=health_url,
                pid=pid,
                error=f"Process exited with code {proc.poll()}" if (proc and proc.poll() not in (None, 0)) else None,
            )

        if running_count == total_services:
            overall = RuntimeState.RUNNING
        elif running_count > 0:
            overall = RuntimeState.PARTIAL
        else:
            has_error = any(s.state == RuntimeState.ERROR for s in statuses.values())
            overall = RuntimeState.ERROR if has_error else RuntimeState.STOPPED

        return RuntimeStatus(
            mode="native",
            overall_state=overall,
            services=statuses,
        )
