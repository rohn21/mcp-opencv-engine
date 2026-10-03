FROM python:3.11-slim
WORKDIR /app
# Install necessary system dependencies including tesseract for OCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY opencv_services/ocr_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /app/shared/
COPY opencv_services/ocr_service/ /app/

EXPOSE 8003
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8003"]
