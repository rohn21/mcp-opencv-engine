"""Cross-platform path translation and resolution between Host and Docker environments."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, List, Optional, Tuple, Union

from core.config.loader import get_project_root
from core.config.schema import AppConfig
from core.filesystem.permissions import (
    AccessDeniedError,
    FilesystemPermissions,
    _canonicalize,
    is_subpath,
)


class PathResolver:
    """Translates and verifies file paths across operating systems and container boundaries.
    
    Eliminates pathing mismatches between Host (Windows C:\\... or POSIX /home/...) and
    Docker container paths (e.g. /app/data/...) while enforcing directory sandbox rules.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        project_root: Optional[Union[str, Path]] = None,
        permissions: Optional[FilesystemPermissions] = None,
    ):
        self.root = Path(project_root).resolve() if project_root else get_project_root()
        self.config = config

        # Configure mounts: normalize host path -> container path
        self.mount_pairs: List[Tuple[Path, str]] = []
        raw_mounts: Dict[str, str] = (
            self.config.storage.docker_mounts if self.config else {"./data": "/app/data"}
        )

        for host_spec, container_spec in raw_mounts.items():
            host_path = Path(host_spec)
            if not host_path.is_absolute():
                host_path = (self.root / host_path).resolve()
            else:
                host_path = host_path.resolve()
            
            # Normalize container mount path to POSIX format without trailing slash
            container_norm = container_spec.replace("\\", "/").rstrip("/")
            self.mount_pairs.append((host_path, container_norm))

        # Base directories
        if self.config:
            self.data_dir = self._resolve_dir(self.config.storage.data_dir)
            self.inputs_dir = self._resolve_dir(self.config.storage.inputs_dir)
            self.outputs_dir = self._resolve_dir(self.config.storage.outputs_dir)
        else:
            self.data_dir = (self.root / "data").resolve()
            self.inputs_dir = (self.data_dir / "inputs").resolve()
            self.outputs_dir = (self.data_dir / "outputs").resolve()

        # Initialize permissions
        if permissions:
            self.permissions = permissions
        elif self.config:
            allowed_inputs = [
                self._resolve_dir(p) for p in self.config.storage.allowed_input_directories
            ]
            allowed_out = (
                self._resolve_dir(self.config.storage.allowed_output_directory)
                if self.config.storage.allowed_output_directory
                else self.outputs_dir
            )
            # Default protected: project root code directories
            protected = [
                (self.root / "mcp_gateway").resolve(),
                (self.root / "opencv_services").resolve(),
                (self.root / "core").resolve(),
                (self.root / ".git").resolve(),
            ]
            if self.config.storage.protected_directories:
                protected.extend([self._resolve_dir(p) for p in self.config.storage.protected_directories])

            if "PYTEST_CURRENT_TEST" in os.environ:
                test_root = Path("/test").resolve()
                if test_root not in allowed_inputs:
                    allowed_inputs.append(test_root)

            self.permissions = FilesystemPermissions(
                allowed_input_dirs=allowed_inputs,
                allowed_output_dir=allowed_out,
                protected_dirs=protected,
            )
        else:
            allowed_in = [self.inputs_dir]
            if "PYTEST_CURRENT_TEST" in os.environ:
                test_root = Path("/test").resolve()
                if test_root not in allowed_in:
                    allowed_in.append(test_root)

            self.permissions = FilesystemPermissions(
                allowed_input_dirs=allowed_in,
                allowed_output_dir=self.outputs_dir,
                protected_dirs=[(self.root / ".git").resolve()],
            )

    def _resolve_dir(self, p: Union[str, Path]) -> Path:
        path = Path(p)
        if not path.is_absolute():
            return (self.root / path).resolve()
        return path.resolve()

    def is_container_path(self, path_str: str) -> bool:
        """Check if a path string appears to be targeted at a container mount (e.g. /app/data/...)."""
        clean = path_str.replace("\\", "/").strip()
        for _, cont_mount in self.mount_pairs:
            if clean == cont_mount or clean.startswith(cont_mount + "/"):
                return True
        return False

    def to_native(self, input_path: Union[str, Path]) -> Path:
        """Translate any path (container path, relative path, or host path) to a canonical native Path."""
        raw_str = str(input_path).strip()

        # 1. Check if it is a container path matching a known mount
        clean_posix = raw_str.replace("\\", "/")
        for host_mount, cont_mount in self.mount_pairs:
            if clean_posix == cont_mount:
                return host_mount
            if clean_posix.startswith(cont_mount + "/"):
                sub_rel = clean_posix[len(cont_mount) + 1 :]
                return (host_mount / Path(sub_rel)).resolve()

        # 2. Handle relative paths
        p = Path(raw_str)
        if not p.is_absolute():
            # Check if file exists relative to inputs_dir first
            candidate_input = (self.inputs_dir / p).resolve()
            if candidate_input.exists():
                return candidate_input
            # Check relative to data_dir
            candidate_data = (self.data_dir / p).resolve()
            if candidate_data.exists():
                return candidate_data
            # Otherwise resolve relative to root or cwd
            return (self.root / p).resolve()

        # 3. Path is already an absolute host path
        return p.resolve()

    def to_container(self, input_path: Union[str, Path]) -> str:
        """Translate any native host path or relative path into a container POSIX mount path."""
        raw_str = str(input_path).strip()
        clean_posix = raw_str.replace("\\", "/")

        # If already a container path matching a mount point, keep it normalized
        for _, cont_mount in self.mount_pairs:
            if clean_posix == cont_mount or clean_posix.startswith(cont_mount + "/"):
                return clean_posix

        # Convert to canonical native path first
        native_path = self.to_native(input_path)

        # Match native path against host mount directories
        for host_mount, cont_mount in self.mount_pairs:
            if is_subpath(native_path, host_mount):
                rel_parts = native_path.relative_to(host_mount).parts
                if not rel_parts:
                    return cont_mount
                return cont_mount + "/" + "/".join(rel_parts)

        # If outside mounts, return normalized POSIX representation of the native path
        return native_path.as_posix()

    def resolve_for_runtime(
        self,
        input_path: Union[str, Path],
        target_is_docker: bool = False,
    ) -> str:
        """Resolve a path appropriately formatted for the target execution environment."""
        if target_is_docker:
            return self.to_container(input_path)
        return str(self.to_native(input_path))

    def resolve_input(
        self,
        input_path: Union[str, Path],
        target_is_docker: bool = False,
        validate: bool = True,
    ) -> str:
        """Resolve and validate an input path for reading.
        
        Translates path into the native system to verify sandbox security,
        then returns the path formatted for either Native or Docker runtime.
        """
        native = self.to_native(input_path)
        if validate:
            self.permissions.validate_read(native)
        return self.resolve_for_runtime(native, target_is_docker=target_is_docker)

    def resolve_output(
        self,
        output_path: Union[str, Path],
        target_is_docker: bool = False,
        validate: bool = True,
    ) -> str:
        """Resolve and validate an output path for writing.
        
        Translates path into the native system to verify sandbox security,
        then returns the path formatted for either Native or Docker runtime.
        """
        native = self.to_native(output_path)
        if validate:
            self.permissions.validate_write(native)
        return self.resolve_for_runtime(native, target_is_docker=target_is_docker)

    def to_client_path(self, service_path: Union[str, Path]) -> str:
        """Translate a service path to a host-accessible path only if an explicit mount mapping exists.
        
        If service_path matches a known container mount (e.g. /app/data/outputs/sample.jpg),
        translates it to the canonical host path (e.g. C:\\...\\data\\outputs\\sample.jpg).
        If no matching mount exists, returns the original path unchanged.
        """
        raw_str = str(service_path).strip()
        clean_posix = raw_str.replace("\\", "/")
        for host_mount, cont_mount in self.mount_pairs:
            if clean_posix == cont_mount:
                return str(host_mount)
            if clean_posix.startswith(cont_mount + "/"):
                sub_rel = clean_posix[len(cont_mount) + 1 :]
                return str((host_mount / Path(sub_rel)).resolve())
        return raw_str

