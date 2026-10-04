FROM python:3.11-slim
WORKDIR /app
COPY mcp_gateway/requirements.txt .
RUN pip install \
    --no-cache-dir \
    --default-timeout=300 \
    --retries=10 \
    -r requirements.txt
COPY shared/ /app/shared/
COPY mcp_gateway/ /app/
EXPOSE 8000
CMD ["python", "main.py"]
