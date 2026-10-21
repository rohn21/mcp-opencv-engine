from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2

# Ensure project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from core.config.loader import get_config
    from core.filesystem.permissions import AccessDeniedError, FilesystemPermissions, is_subpath
    from core.filesystem.path_resolver import PathResolver
except ImportError:  # pragma: no cover
    get_config = None
    AccessDeniedError = PermissionError
    FilesystemPermissions = None
    PathResolver = None
    is_subpath = None

MAX_FILE_SIZE_MB = 20
MAX_DIMENSION_PX = 4000


def validate_image(
    image_path: Union[str, Path],
    allowed_base_dir: Optional[Union[str, Path]] = None,
    permissions: Optional[FilesystemPermissions] = None,
) -> Tuple[bool, str]:
    """Validate image file existence, size, readability, and security boundaries.

    Enforces filesystem sandbox containment before microservice access.

    Args:
        image_path: Target path to the image file.
        allowed_base_dir: Optional base directory to restrict access to.
        permissions: Optional preconfigured FilesystemPermissions instance.

    Returns:
        (is_valid, message) tuple.
    """
    raw_str = str(image_path).strip()
    if not raw_str:
        return False, "Path cannot be empty."

    path = Path(raw_str).resolve()

    # 1. Filesystem Security Validation
    if permissions is not None:
        try:
            permissions.validate_read(path)
        except AccessDeniedError as e:
            return False, f"Security Error: {e}"
    elif allowed_base_dir is not None and str(allowed_base_dir).strip():
        base = Path(allowed_base_dir).resolve()
        if is_subpath is not None:
            if not is_subpath(path, base):
                return False, f"Security Error: Access denied to path outside allowed directory: '{path}'"
        else:
            try:
                path.relative_to(base)
            except ValueError:
                if not str(path).lower().startswith(str(base).lower()):
                    return False, f"Security Error: Access denied to path outside allowed directory: '{path}'"
    elif get_config is not None:
        # If no explicit base dir passed, check with default config permissions if available
        try:
            cfg = get_config()
            perm = FilesystemPermissions(
                allowed_input_dirs=cfg.storage.allowed_input_directories,
                allowed_output_dir=cfg.storage.allowed_output_directory,
                protected_dirs=cfg.storage.protected_directories,
            )
            perm.validate_read(path)
        except AccessDeniedError as e:
            return False, f"Security Error: {e}"
        except Exception:
            pass

    # 2. File existence check
    if not path.is_file():
        return False, f"File not found: {image_path}"

    # 3. Empty file check
    try:
        file_size = path.stat().st_size
    except OSError as e:
        return False, f"Cannot read file attributes: {e}"

    if file_size == 0:
        return False, "File is empty."

    # 4. File size check
    file_size_mb = file_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({file_size_mb:.1f}MB). Max allowed is {MAX_FILE_SIZE_MB}MB."

    return True, "OK"


def get_dimensions(image_path: Union[str, Path]) -> Dict[str, int]:
    """Decode and extract dimensions from an image.

    Raises ValueError if decoding fails.
    """
    path_str = str(Path(image_path).resolve())
    img = cv2.imread(path_str)
    if img is None:
        raise ValueError(f"Could not decode image at: {path_str}")

    if len(img.shape) == 2:
        h, w = img.shape
        return {"width": int(w), "height": int(h), "channels": 1}

    h, w, c = img.shape
    return {"width": int(w), "height": int(h), "channels": int(c)}
