FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgomp1 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY opencv_services/object_detection/requirements.txt .
RUN pip install \
    --no-cache-dir \
    --default-timeout=300 \
    --retries=10 \
    -r requirements.txt
COPY shared/ /app/shared/
COPY opencv_services/ /app/opencv_services/
EXPOSE 8001
CMD ["uvicorn", "opencv_services.object_detection.api:app", "--host", "0.0.0.0", "--port", "8001"]
