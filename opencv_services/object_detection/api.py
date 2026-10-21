from __future__ import annotations

import asyncio
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import torch
from fastapi import FastAPI, HTTPException

# Ensure project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.schemas import BoundingBox, DetectedObject, DetectionRequest, DetectionResponse
from shared.image_utils import validate_image
from opencv_services.object_detection.model_manager import get_model, resolve_device

try:
    from core.config.loader import get_config
    cfg = get_config()
    DATA_DIR = cfg.storage.data_dir.resolve()
    OUTPUT_DIR = cfg.storage.outputs_dir.resolve()
except Exception:
    DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(DATA_DIR / "outputs"))).resolve()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loaded
    get_model()
    model_loaded = True
    yield


app = FastAPI(title="Object Detection Service (YOLOv8)", lifespan=lifespan)


@app.get("/health")
def health_check():
    """Returns 200 only when the YOLO model is fully loaded."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading...")
    active_device = resolve_device()
    return {"status": "healthy", "model": "yolov8n", "gpu": active_device}


@app.post("/predict", response_model=DetectionResponse)
async def predict(req: DetectionRequest):
    is_valid, msg = validate_image(req.image_path, str(DATA_DIR))
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    model = get_model()

    results = await asyncio.to_thread(
        model,
        req.image_path,
        conf=req.confidence_threshold,
        verbose=False,
    )
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        detections.append(
            DetectedObject(
                class_name=model.names[cls_id],
                confidence=round(float(box.conf[0].item()), 3),
                bbox=BoundingBox(
                    x1=round(float(box.xyxy[0][0]), 1),
                    y1=round(float(box.xyxy[0][1]), 1),
                    x2=round(float(box.xyxy[0][2]), 1),
                    y2=round(float(box.xyxy[0][3]), 1),
                ),
            )
        )

    annotated_path = None
    if req.draw_boxes:
        annotated_img = result.plot()
        output_filename = f"detections_{uuid.uuid4().hex[:8]}.jpg"
        annotated_path = str(OUTPUT_DIR / output_filename)
        await asyncio.to_thread(cv2.imwrite, annotated_path, annotated_img)

    return DetectionResponse(
        status="success",
        total_objects=len(detections),
        annotated_image_path=annotated_path,
        detections=detections,
    )