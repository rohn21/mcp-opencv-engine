from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

import torch
from ultralytics import YOLO

# Ensure project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from core.config.loader import get_config, get_project_root
    from core.config.schema import DeviceTarget
except ImportError:  # pragma: no cover
    get_config = None
    get_project_root = lambda: _project_root
    DeviceTarget = None

_model: Optional[YOLO] = None
_loaded_device: Optional[str] = None
_loaded_weights_path: Optional[Path] = None


def resolve_model_path(
    model_name: Optional[str] = None,
    weights_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Deterministically resolve the YOLO weights file path.

    Checks configured model directory, explicit argument, or project root.
    Raises FileNotFoundError if the file does not exist on disk.
    """
    root = get_project_root() if callable(get_project_root) else _project_root

    target_name = model_name
    target_dir = Path(weights_dir).resolve() if weights_dir else None

    if get_config is not None:
        try:
            cfg = get_config()
            if target_name is None:
                target_name = cfg.model.yolo_model_name
            if target_dir is None and cfg.model.yolo_weights_dir is not None:
                target_dir = Path(cfg.model.yolo_weights_dir).resolve()
        except Exception:
            pass

    if target_name is None:
        target_name = "yolov8n.pt"

    if target_dir is None:
        # Check if weights exist in project root or current working directory
        if (root / target_name).exists():
            target_dir = root
        elif (Path.cwd() / target_name).exists():
            target_dir = Path.cwd()
        else:
            target_dir = root

    weights_path = (target_dir / target_name).resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"YOLO weights file not found at: '{weights_path}'. "
            "Please ensure the model file is present or configure yolo_weights_dir in settings."
        )

    return weights_path


def resolve_device(requested_device: Optional[str] = None) -> str:
    """Resolve compute device target with validation and honest degradation.

    - "cuda": Requires torch.cuda.is_available() == True, else raises RuntimeError.
    - "mps": Requires torch.backends.mps.is_available() == True, else raises RuntimeError.
    - "cpu": Always valid.
    - "auto" / None: Uses CUDA if available, then MPS, falling back to CPU.
    """
    dev = requested_device

    if dev is None and get_config is not None:
        try:
            cfg = get_config()
            if cfg.device:
                dev = cfg.device.value if hasattr(cfg.device, "value") else str(cfg.device)
        except Exception:
            pass

    if dev is None:
        dev = "auto"

    dev_clean = dev.lower().strip()

    if dev_clean == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device was requested, but torch.cuda.is_available() is False. "
                "Ensure NVIDIA drivers and PyTorch CUDA build are installed, or switch to CPU/auto."
            )
        return "cuda"

    if dev_clean == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            raise RuntimeError(
                "Apple Silicon MPS device was requested, but torch.backends.mps.is_available() is False."
            )
        return "mps"

    if dev_clean == "cpu":
        return "cpu"

    if dev_clean == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    raise ValueError(f"Unknown device target: '{requested_device}'. Must be one of: 'auto', 'cpu', 'cuda', 'mps'.")


def get_model(
    weights_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    reload: bool = False,
) -> YOLO:
    """Singleton loader for YOLOv8 model with deterministic location and device.

    Args:
        weights_path: Optional explicit path to model weights file.
        device: Optional explicit compute device ('cpu', 'cuda', 'mps', 'auto').
        reload: If True, forces reloading even if already cached.

    Returns:
        Loaded and device-transferred YOLO model instance.
    """
    global _model, _loaded_device, _loaded_weights_path

    if _model is not None and not reload:
        return _model

    resolved_path = (
        Path(weights_path).resolve()
        if weights_path
        else resolve_model_path()
    )
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Model file not found at: '{resolved_path}'")

    target_device = resolve_device(device)

    model = YOLO(str(resolved_path))
    model.to(target_device)

    _model = model
    _loaded_device = target_device
    _loaded_weights_path = resolved_path

    return _model
