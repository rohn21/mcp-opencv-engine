FROM python:3.11-slim
WORKDIR /app
COPY opencv_services/image_processing/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ /app/shared/
COPY opencv_services/image_processing/ /app/
EXPOSE 8002
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8002"]
