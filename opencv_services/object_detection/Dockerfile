FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgomp1 libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY cv_services/object_detection/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ /app/shared/
COPY cv_services/object_detection/ /app/
EXPOSE 8001
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
