from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from core.config.loader import get_config
    from core.filesystem.permissions import FilesystemPermissions
except ImportError:
    get_config = None
    FilesystemPermissions = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def list_available_images() -> List[str]:
    """Returns absolute paths of all readable images in configured allowed input directories."""
    images: List[str] = []
    seen: set[str] = set()

    scan_dirs: List[Path] = []
    permissions: Optional[FilesystemPermissions] = None

    if get_config is not None:
        try:
            cfg = get_config()
            permissions = FilesystemPermissions(
                allowed_input_dirs=cfg.storage.allowed_input_directories,
                allowed_output_dir=cfg.storage.allowed_output_directory,
                protected_dirs=cfg.storage.protected_directories,
            )
            for p in cfg.storage.allowed_input_directories:
                resolved_p = p.resolve() if p.is_absolute() else (_project_root / p).resolve()
                if resolved_p.exists() and resolved_p not in scan_dirs:
                    scan_dirs.append(resolved_p)
        except Exception:
            pass

    if not scan_dirs:
        data_dir = Path(os.getenv("DATA_DIR", str(_project_root / "data"))).resolve()
        inputs_dir = data_dir / "inputs"
        if inputs_dir.exists():
            scan_dirs.append(inputs_dir)

    for directory in scan_dirs:
        if not directory.is_dir():
            continue
        try:
            for item in directory.iterdir():
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
                    resolved_item = str(item.resolve())
                    if permissions is not None:
                        try:
                            permissions.validate_read(item)
                        except Exception:
                            continue
                    if resolved_item not in seen:
                        seen.add(resolved_item)
                        images.append(resolved_item)
        except (OSError, PermissionError):
            continue

    return sorted(images)
