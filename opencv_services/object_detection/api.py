import cv2
import uuid
import asyncio
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
import sys, os
from contextlib import asynccontextmanager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.schemas import DetectionRequest, DetectionResponse, DetectedObject, BoundingBox
from shared.image_utils import validate_image
from opencv_services.object_detection.model_manager import get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loaded
    get_model() 
    model_loaded = True
    yield
    
app = FastAPI(title="Object Detection Service (YOLOv8)", lifespan=lifespan)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model_loaded = False

@app.get("/health")
def health_check():
    """Returns 200 only when the YOLO model is fully loaded."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading...")
    return {"status": "healthy", "model": "yolov8n", "gpu": "cuda" if torch.cuda.is_available() else "cpu"}

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
        verbose=False
    )
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        detections.append(DetectedObject(
            class_name=model.names[cls_id],
            confidence=round(float(box.conf[0].item()), 3),
            bbox=BoundingBox(
                x1=round(float(box.xyxy[0][0]), 1),
                y1=round(float(box.xyxy[0][1]), 1),
                x2=round(float(box.xyxy[0][2]), 1),
                y2=round(float(box.xyxy[0][3]), 1),
            )
        ))

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