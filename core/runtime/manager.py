"""Unified Runtime Manager orchestrating native and Docker execution modes."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Union

from core.config.loader import get_config, get_project_root
from core.config.schema import AppConfig, RuntimeMode
from core.environment.detector import EnvironmentDetector
from core.runtime.base import BaseRuntime, RuntimeStatus
from core.runtime.docker import DockerRuntime
from core.runtime.native import NativeRuntime


class RuntimeManager:
    """Orchestrates runtime selection, lifecycle, and health verification."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        runtime_mode: Optional[Union[RuntimeMode, str]] = None,
        project_root: Optional[Union[str, Path]] = None,
        command_runner: Optional[Callable[..., tuple[int, str, str]]] = None,
    ):
        self.config = config or get_config()
        self.project_root = Path(project_root).resolve() if project_root else get_project_root()
        self.command_runner = command_runner

        # Determine target mode
        if runtime_mode is not None:
            if isinstance(runtime_mode, RuntimeMode):
                self._mode = runtime_mode
            else:
                self._mode = RuntimeMode(str(runtime_mode).lower().strip())
        else:
            self._mode = self.config.runtime_mode

        self.runtime = self._instantiate_runtime()

    def _instantiate_runtime(self) -> BaseRuntime:
        """Instantiate the concrete runtime adapter according to mode selection."""
        target_mode = self._mode

        if target_mode == RuntimeMode.AUTO:
            # Auto-detect: if Docker is available and active, choose Docker; otherwise native
            detector = EnvironmentDetector()
            env_profile = detector.detect(command_runner=self.command_runner)
            if env_profile.has_docker and env_profile.docker_compose_available:
                target_mode = RuntimeMode.DOCKER
            else:
                target_mode = RuntimeMode.NATIVE

        if target_mode == RuntimeMode.DOCKER:
            return DockerRuntime(
                config=self.config,
                project_root=self.project_root,
                command_runner=self.command_runner,
            )
        elif target_mode == RuntimeMode.NATIVE:
            return NativeRuntime(
                config=self.config,
                project_root=self.project_root,
            )
        else:
            raise ValueError(f"Unsupported runtime mode: '{target_mode}'")

    @property
    def mode(self) -> str:
        """Return the active runtime mode name ('native' or 'docker')."""
        return "docker" if isinstance(self.runtime, DockerRuntime) else "native"

    def start(
        self,
        services: Optional[List[str]] = None,
        timeout: float = 30.0,
        **kwargs,
    ) -> bool:
        """Start the configured runtime."""
        return self.runtime.start(services=services, timeout=timeout, **kwargs)

    def stop(self, timeout: float = 10.0) -> bool:
        """Stop all services managed by this runtime."""
        return self.runtime.stop(timeout=timeout)

    def get_status(self) -> RuntimeStatus:
        """Query status of all managed services."""
        return self.runtime.get_status()

    def wait_for_ready(self, timeout: float = 30.0, poll_interval: float = 0.5) -> bool:
        """Wait until all services are healthy and responding."""
        return self.runtime.wait_for_ready(timeout=timeout, poll_interval=poll_interval)

    def is_healthy(self) -> bool:
        """Check if all services are responding."""
        return self.runtime.is_healthy()
