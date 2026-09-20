import cv2
import uuid
import asyncio
import hashlib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.image_utils import validate_image
from shared.schemas import ImageFilterRequest, FilterResponse, FaceBlurRequest, FaceBlurResponse
from opencv_services.image_processing.opencv_ops import FILTER_REGISTRY

app = FastAPI(title="Image Processing Service (OpenCV)")

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/health")
def health_check():
    """OpenCV loads instantly, so this service is always healthy once running."""
    return {
        "status": "healthy", 
        "service": "image_processing",
        "available_filters": list(FILTER_REGISTRY.keys())
    }



def _process_face_blur(img: np.ndarray, blur_intensity: int) -> tuple[np.ndarray, int]:
    """
    Synchronous helper to handle CPU-bound face detection and blurring.
    Runs entirely in a background thread to avoid event loop blocking.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20)
    )


    if blur_intensity % 2 == 0:
        blur_intensity += 1


    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi, (blur_intensity, blur_intensity), 30)
        img[y:y+h, x:x+w] = blurred_roi

    return img, len(faces)



@app.post("/filter", response_model=FilterResponse)
async def apply_filter(req: ImageFilterRequest):
    is_valid, msg = validate_image(req.image_path, str(DATA_DIR))
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)


    if req.filter_type not in FILTER_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown filter. Choose from: {list(FILTER_REGISTRY.keys())}"
        )

    hash_input = f"{req.image_path}_{req.filter_type}"
    file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:10]
    output_filename = f"filtered_{req.filter_type}_{file_hash}.jpg"
    output_path = OUTPUT_DIR / output_filename


    if output_path.exists():
        orig_img = await asyncio.to_thread(cv2.imread, req.image_path)
        proc_img = await asyncio.to_thread(cv2.imread, str(output_path))
        

        if len(orig_img.shape) == 2:
            h, w = orig_img.shape
            original_dims = {"width": w, "height": h, "channels": 1}
        else:
            h, w, c = orig_img.shape
            original_dims = {"width": w, "height": h, "channels": c}
            

        if len(proc_img.shape) == 2:
            h, w = proc_img.shape
            processed_dims = {"width": w, "height": h, "channels": 1}
        else:
            h, w, c = proc_img.shape
            processed_dims = {"width": w, "height": h, "channels": c}

        return FilterResponse(
            status="success",
            filter_applied=req.filter_type,
            output_image_path=str(output_path),
            original_dimensions=original_dims,
            processed_dimensions=processed_dims,
        )

    img = await asyncio.to_thread(cv2.imread, req.image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")


    if len(img.shape) == 2:
        h, w = img.shape
        original_dims = {"width": w, "height": h, "channels": 1}
    else:
        h, w, c = img.shape
        original_dims = {"width": w, "height": h, "channels": c}


    processed = await asyncio.to_thread(FILTER_REGISTRY[req.filter_type], img)


    output_filename = f"filtered_{req.filter_type}_{uuid.uuid4().hex[:8]}.jpg"
    output_path = OUTPUT_DIR / output_filename
    await asyncio.to_thread(cv2.imwrite, str(output_path), processed)


    if len(processed.shape) == 2:
        h, w = processed.shape
        processed_dims = {"width": w, "height": h, "channels": 1}
    else:
        processed_dims = {"width": processed.shape[1], "height": processed.shape[0], "channels": processed.shape[2]}

    return FilterResponse(
        status="success",
        filter_applied=req.filter_type,
        output_image_path=str(output_path),
        original_dimensions=original_dims,
        processed_dimensions=processed_dims,
    )



@app.post("/blur_faces", response_model=FaceBlurResponse)
async def blur_faces(req: FaceBlurRequest):
    """Detects faces and applies a heavy blur to protect privacy."""
    

    hash_input = f"{req.image_path}_{req.blur_intensity}"
    file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:10]
    output_filename = f"blurred_faces_{file_hash}.jpg"
    output_path = OUTPUT_DIR / output_filename

    if output_path.exists():
        img = await asyncio.to_thread(cv2.imread, req.image_path)
        gray = await asyncio.to_thread(cv2.cvtColor, img, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = await asyncio.to_thread(
            face_cascade.detectMultiScale, gray, 1.05, 3, (20, 20)
        )
        
        return FaceBlurResponse(
            status="success",
            faces_detected=len(faces),
            output_image_path=str(output_path)
        )
    

    is_valid, msg = validate_image(req.image_path, str(DATA_DIR))
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)


    img = await asyncio.to_thread(cv2.imread, req.image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")


    processed_img, faces_count = await asyncio.to_thread(
        _process_face_blur, 
        img, 
        req.blur_intensity
    )


    output_filename = f"blurred_faces_{uuid.uuid4().hex[:8]}.jpg"
    output_path = OUTPUT_DIR / output_filename
    await asyncio.to_thread(cv2.imwrite, str(output_path), processed_img)

    return FaceBlurResponse(
        status="success",
        faces_detected=faces_count,
        output_image_path=str(output_path)
    )