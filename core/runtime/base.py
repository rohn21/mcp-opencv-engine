"""Abstract base interface and status schemas for MCP OpenCV Engine runtime management."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from core.config.loader import get_config, get_project_root
from core.config.schema import AppConfig


class RuntimeState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass
class ServiceStatus:
    name: str
    state: RuntimeState = RuntimeState.STOPPED
    port: Optional[int] = None
    url: Optional[str] = None
    health_url: Optional[str] = None
    pid: Optional[int] = None
    error: Optional[str] = None


@dataclass
class RuntimeStatus:
    mode: str
    overall_state: RuntimeState
    services: Dict[str, ServiceStatus] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self.overall_state == RuntimeState.RUNNING


def check_http_endpoint(url: str, timeout: float = 2.0) -> bool:
    """Helper to check if an HTTP health endpoint responds with 200 OK.

    Uses urllib standard library to remain lightweight and independent.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MCPOpenCVRuntime/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", None) or getattr(resp, "code", None)
            return code == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


class BaseRuntime(ABC):
    """Abstract runtime lifecycle manager for native processes or Docker containers."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        project_root: Optional[Union[str, Path]] = None,
        command_runner: Optional[Callable[..., tuple[int, str, str]]] = None,
    ):
        self.config = config or get_config()
        self.project_root = Path(project_root).resolve() if project_root else get_project_root()
        self.command_runner = command_runner

    @abstractmethod
    def start(
        self,
        services: Optional[List[str]] = None,
        timeout: float = 30.0,
    ) -> bool:
        """Start the runtime services."""
        pass

    @abstractmethod
    def stop(self, timeout: float = 10.0) -> bool:
        """Stop only the services managed by this runtime instance."""
        pass

    @abstractmethod
    def get_status(self) -> RuntimeStatus:
        """Query and return the current status of all managed services."""
        pass

    def wait_for_ready(
        self,
        timeout: float = 30.0,
        poll_interval: float = 0.5,
    ) -> bool:
        """Wait until all required service endpoints respond successfully or timeout."""
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            status = self.get_status()
            if status.overall_state == RuntimeState.RUNNING:
                return True
            time.sleep(poll_interval)
        return False

    def is_healthy(self) -> bool:
        """Check if all services are currently healthy and responding."""
        status = self.get_status()
        return status.overall_state == RuntimeState.RUNNING
