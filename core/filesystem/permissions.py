"""Filesystem security, sandboxing, and traversal prevention."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union


class FilesystemSecurityError(Exception):
    """Base exception for filesystem security violations."""
    pass


class PathTraversalError(FilesystemSecurityError):
    """Raised when path traversal (e.g. '../', null bytes) is detected."""
    pass


class AccessDeniedError(FilesystemSecurityError):
    """Raised when a path is outside the allowed directory sandbox or in a protected zone."""
    pass


def _canonicalize(path_input: Union[str, Path]) -> Path:
    """Validate path format and return canonical Path without resolving non-existent files."""
    if isinstance(path_input, str):
        if "\0" in path_input:
            raise PathTraversalError("Path contains illegal null bytes.")
        p = Path(path_input.strip())
    else:
        p = path_input

    # Normalize parts to eliminate redundant ./ and ../
    # Note: Path.resolve() resolves symlinks and relative paths.
    try:
        return p.resolve()
    except (OSError, RuntimeError) as exc:
        raise PathTraversalError(f"Failed to canonicalize path '{path_input}': {exc}") from exc


def is_subpath(child: Path, parent: Path) -> bool:
    """Check if child path is equal to or strictly inside parent path."""
    try:
        child_res = child.resolve()
        parent_res = parent.resolve()
        # Path.relative_to succeeds if child is equal to or inside parent
        child_res.relative_to(parent_res)
        return True
    except (ValueError, OSError):
        # On Windows, relative_to across drives raises ValueError
        return False


class FilesystemPermissions:
    """Enforces directory sandbox boundaries, symlink policies, and traversal protection."""

    def __init__(
        self,
        allowed_input_dirs: Optional[List[Union[str, Path]]] = None,
        allowed_output_dir: Optional[Union[str, Path]] = None,
        protected_dirs: Optional[List[Union[str, Path]]] = None,
        allow_symlinks_outside_sandbox: bool = False,
    ):
        self.allowed_input_dirs = [
            _canonicalize(d) for d in (allowed_input_dirs or [])
        ]
        self.allowed_output_dir = (
            _canonicalize(allowed_output_dir) if allowed_output_dir else None
        )
        self.protected_dirs = [
            _canonicalize(d) for d in (protected_dirs or [])
        ]
        self.allow_symlinks_outside_sandbox = allow_symlinks_outside_sandbox

    def is_protected(self, path: Path) -> bool:
        """Check if the given path falls inside any protected directory."""
        canonical = _canonicalize(path)
        for p_dir in self.protected_dirs:
            if is_subpath(canonical, p_dir):
                return True
        return False

    def validate_symlink(self, path: Path, allowed_root: Path) -> None:
        """Verify that symlinks do not escape the sandbox to unauthorized locations."""
        if self.allow_symlinks_outside_sandbox:
            return

        if path.is_symlink():
            try:
                # realpath resolves all symlinks
                real_target = Path(os.path.realpath(str(path)))
                if not is_subpath(real_target, allowed_root):
                    raise AccessDeniedError(
                        f"Symlink target '{real_target}' points outside allowed root '{allowed_root}'."
                    )
            except OSError as exc:
                raise AccessDeniedError(f"Unable to verify symlink '{path}': {exc}") from exc

    def validate_read(self, path: Union[str, Path]) -> Path:
        """Verify that a path is permitted for read access.
        
        Returns:
            Canonical Path if authorized.
        Raises:
            PathTraversalError: If traversal attack or null bytes detected.
            AccessDeniedError: If path is not within allowed input directories or in protected zone.
        """
        canonical = _canonicalize(path)

        # Check protected directories first
        if self.is_protected(canonical):
            raise AccessDeniedError(
                f"Security Error: Access to protected path is forbidden: {canonical}"
            )

        # Check within at least one allowed input directory
        allowed = False
        matching_root: Optional[Path] = None
        for root in self.allowed_input_dirs:
            if is_subpath(canonical, root):
                allowed = True
                matching_root = root
                break

        if not allowed:
            raise AccessDeniedError(
                f"Security Error: Read access denied for path outside allowed input directories: {canonical}"
            )

        if matching_root:
            self.validate_symlink(canonical, matching_root)

        return canonical

    def validate_write(self, path: Union[str, Path]) -> Path:
        """Verify that a path is permitted for write access.
        
        Returns:
            Canonical Path if authorized.
        Raises:
            PathTraversalError: If traversal attack or null bytes detected.
            AccessDeniedError: If path is not within allowed output directory or in protected zone.
        """
        canonical = _canonicalize(path)

        # Check protected directories
        if self.is_protected(canonical):
            raise AccessDeniedError(
                f"Security Error: Write access to protected path is forbidden: {canonical}"
            )

        # Check within allowed output directory
        if not self.allowed_output_dir or not is_subpath(canonical, self.allowed_output_dir):
            raise AccessDeniedError(
                f"Security Error: Write access denied. Path is outside allowed output directory: {canonical}"
            )

        self.validate_symlink(canonical, self.allowed_output_dir)
        return canonical

    def is_safe_path(
        self,
        path: Union[str, Path],
        mode: Literal["read", "write"] = "read"
    ) -> Tuple[bool, Optional[str]]:
        """Non-throwing check returning (is_safe, error_message)."""
        try:
            if mode == "read":
                self.validate_read(path)
            elif mode == "write":
                self.validate_write(path)
            else:
                return False, f"Unknown mode: {mode}"
            return True, None
        except FilesystemSecurityError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"Invalid path: {exc}"
