"""Dynamic, secure mount override generator for Docker Compose."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from core.config.loader import get_config, get_project_root
from core.config.schema import AppConfig
from core.filesystem.permissions import (
    AccessDeniedError,
    FilesystemPermissions,
    _canonicalize,
    is_subpath,
)


BROAD_SYSTEM_DIRECTORIES = [
    Path("/"),
    Path.home(),
]
if os.name == "nt":
    for drive in ("C", "D", "E", "F"):
        BROAD_SYSTEM_DIRECTORIES.append(Path(f"{drive}:\\"))
        BROAD_SYSTEM_DIRECTORIES.append(Path(f"{drive}:/"))
    win_dir = os.environ.get("SystemRoot") or "C:\\Windows"
    BROAD_SYSTEM_DIRECTORIES.append(Path(win_dir))
else:
    for posix_sys in ("/bin", "/sbin", "/usr", "/etc", "/var", "/dev", "/proc", "/sys"):
        BROAD_SYSTEM_DIRECTORIES.append(Path(posix_sys))


def validate_mount_path(
    path_input: Union[str, Path],
    is_output: bool = False,
    project_root: Optional[Path] = None,
    permissions: Optional[FilesystemPermissions] = None,
) -> Path:
    """Validate that a host mount path exists, is a directory, and is not a protected system root.

    Args:
        path_input: The host directory path to mount into the container.
        is_output: Whether this directory is intended for output writes.
        project_root: Canonical project root to prevent whole-project mounts.
        permissions: Optional filesystem permissions instance.

    Returns:
        Canonical Path if valid.

    Raises:
        FileNotFoundError: If path does not exist on host.
        NotADirectoryError: If path is not a directory.
        AccessDeniedError: If path attempts to mount root, home, or protected directories.
    """
    raw_str = str(path_input).strip()
    if not raw_str:
        raise ValueError("Mount path cannot be empty.")

    p = Path(raw_str).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Mount path does not exist on host: '{p}'")

    if not p.is_dir():
        raise NotADirectoryError(f"Mount path must be a directory, not a file: '{p}'")

    # Check broad system roots and home directory
    for broad in BROAD_SYSTEM_DIRECTORIES:
        try:
            if p == broad.resolve():
                raise AccessDeniedError(
                    f"Mounting broad system root or home directory '{p}' into Docker is forbidden for security."
                )
        except OSError:
            pass

    # Check project root: do not mount whole repo root
    root = (project_root or get_project_root()).resolve()
    if p == root:
        raise AccessDeniedError(
            f"Mounting the entire project root '{p}' is forbidden. Mount dedicated input/output subdirectories."
        )

    # Protected source directories
    protected_subdirs = [root / "mcp_gateway", root / "opencv_services", root / "core", root / ".git"]
    for prot in protected_subdirs:
        if is_subpath(p, prot.resolve()):
            raise AccessDeniedError(f"Mount path is in a protected source directory: '{p}'")

    if permissions is not None:
        if is_output:
            permissions.validate_write(p)
        else:
            permissions.validate_read(p)

    return p


def generate_compose_override(
    input_dirs: Optional[List[Union[str, Path]]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
    target_override_file: Optional[Union[str, Path]] = None,
    services: Optional[List[str]] = None,
) -> Path:
    """Generate a separate, secure docker-compose.override.yml without modifying the base file.

    - Input directories are mounted strictly Read-Only (:ro).
    - Output directory is mounted Read-Write (:rw).
    - Generated configuration is written into .runtime/ and kept deterministic.
    """
    root = Path(project_root).resolve() if project_root else get_project_root()
    override_path = (
        Path(target_override_file).resolve()
        if target_override_file
        else (root / ".runtime" / "docker-compose.override.yml").resolve()
    )

    override_path.parent.mkdir(parents=True, exist_ok=True)

    target_services = services or ["object-detection", "image-processing", "ocr-service"]

    # Validate inputs
    valid_inputs: List[Path] = []
    if input_dirs:
        for inp in input_dirs:
            valid_inputs.append(validate_mount_path(inp, is_output=False, project_root=root))
    else:
        # Default input dir from data/inputs if it exists
        default_in = root / "data" / "inputs"
        if default_in.is_dir():
            valid_inputs.append(default_in)

    # Validate output
    valid_output: Optional[Path] = None
    if output_dir:
        valid_output = validate_mount_path(output_dir, is_output=True, project_root=root)
    else:
        default_out = root / "data" / "outputs"
        if default_out.is_dir():
            valid_output = default_out

    # Format volume mappings as standard POSIX paths (Docker supports / on Windows)
    volume_lines: List[str] = []

    # Mount inputs read-only: first input mounts to /app/data/inputs:ro
    if valid_inputs:
        primary_input = valid_inputs[0].as_posix()
        volume_lines.append(f'      - "{primary_input}:/app/data/inputs:ro"')
        for idx, extra_in in enumerate(valid_inputs[1:], start=2):
            extra_posix = extra_in.as_posix()
            volume_lines.append(f'      - "{extra_posix}:/app/data/inputs_{idx}:ro"')

    if valid_output:
        out_posix = valid_output.as_posix()
        volume_lines.append(f'      - "{out_posix}:/app/data/outputs:rw"')

    # Build deterministic YAML
    lines = [
        'version: "3.9"',
        "services:",
    ]
    for svc in sorted(target_services):
        lines.append(f"  {svc}:")
        if volume_lines:
            lines.append("    volumes:")
            lines.extend(volume_lines)

    lines.append("")  # trailing newline
    content = "\n".join(lines)

    with open(override_path, "w", encoding="utf-8") as f:
        f.write(content)

    return override_path


def remove_compose_override(project_root: Optional[Union[str, Path]] = None) -> bool:
    """Remove generated override file if present."""
    root = Path(project_root).resolve() if project_root else get_project_root()
    override_path = root / ".runtime" / "docker-compose.override.yml"
    if override_path.is_file():
        try:
            override_path.unlink()
            return True
        except OSError:
            pass
    return False
