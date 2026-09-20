from ultralytics import YOLO
import torch

_model = None

def get_model():
    """Singleton pattern: loads YOLO model only once."""
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(device)
    return _model
