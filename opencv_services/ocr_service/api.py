import os
import easyocr
import asyncio
import warnings
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import sys

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")



sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.schemas import OCRRequest, OCRResponse
from shared.image_utils import validate_image


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_reader, ocr_loaded
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    ocr_loaded = True
    yield
    
app = FastAPI(title="OCR Service (Dual-Engine: EasyOCR + Tesseract)", lifespan=lifespan)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREFERRED_ENGINE = os.getenv("OCR_ENGINE", "easyocr").lower()

ocr_reader = None
ocr_loaded = False
active_engine = "tesseract"

if PREFERRED_ENGINE == "easyocr":
    try:
        import easyocr
        ocr_reader = easyocr.Reader(['en'], gpu=False)
        active_engine = "easyocr"
    except Exception as e:
        active_engine = "tesseract"
        

@app.get("/health")
def health_check():
    """Returns 200 only when the EasyOCR model is fully loaded."""
    if not ocr_loaded or ocr_reader is None:
        raise HTTPException(status_code=503, detail="OCR model is still loading...")
    return {"status": "healthy", "active_engine": active_engine, "note": "If active_engine is 'tesseract', EasyOCR failed to load or was disabled."
    }


def _run_tesseract_ocr(image_path: str) -> dict:
    import pytesseract
    import cv2
    from PIL import Image

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not decode image for OCR")
        
    h, w = img.shape[:2]
    if w > 1000:
        scale = 1000 / w
        img = cv2.resize(img, (1000, int(h * scale)), interpolation=cv2.INTER_AREA)
        
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    custom_config = r'--oem 3 --psm 11'
    
    ocr_data = pytesseract.image_to_data(
        pil_img, 
        output_type=pytesseract.Output.DICT,
        lang='eng',
        config=custom_config
    )
    
    text_blocks = []
    full_text_parts = []

    for i in range(len(ocr_data['text'])):
        if ocr_data['level'][i] == 5:
            conf = int(ocr_data['conf'][i])
            text = ocr_data['text'][i].strip()
            
            if conf > 30 and text:
                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                
                full_text_parts.append(text)
                text_blocks.append({
                    "text": text,
                    "confidence": round(conf / 100.0, 3),
                    "bbox": {
                        "x1": float(x),
                        "y1": float(y),
                        "x2": float(x + w),
                        "y2": float(y + h)
                    }
                })
                
    return {
        "extracted_text": " ".join(full_text_parts),
        "text_blocks": text_blocks
    }


@app.post("/extract_text", response_model=OCRResponse)
async def extract_text(req: OCRRequest):
    """Extracts all visible text from an image using OCR."""
    global active_engine, ocr_reader

    is_valid, msg = validate_image(req.image_path, str(DATA_DIR))
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    if active_engine == "easyocr" and ocr_reader is not None:
        try:
            results = await asyncio.to_thread(ocr_reader.readtext, req.image_path)

            text_blocks = []
            full_text = ""

            for (bbox, text, conf) in results:
                full_text += text + "\n"
                text_blocks.append({
                    "text": text,
                    "confidence": round(float(conf), 3),
                    "bbox": {
                        "x1": round(float(bbox[0][0]), 1), "y1": round(float(bbox[0][1]), 1),
                        "x2": round(float(bbox[2][0]), 1), "y2": round(float(bbox[2][1]), 1)
                    }
                })

            return OCRResponse(
                status="success",
                text_found=len(results) > 0,
                extracted_text=full_text.strip(),
                text_blocks=text_blocks
            )

        except Exception as e:
            active_engine = "tesseract"

    tesseract_result = await asyncio.to_thread(_run_tesseract_ocr, req.image_path)
    
    return OCRResponse(
        status="success",
        text_found=len(tesseract_result["text_blocks"]) > 0,
        extracted_text=tesseract_result["extracted_text"],
        text_blocks=tesseract_result["text_blocks"]
    )