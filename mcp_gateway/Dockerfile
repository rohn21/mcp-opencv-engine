FROM python:3.11-slim
WORKDIR /app
COPY mcp_gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ /app/shared/
COPY mcp_gateway/ /app/
EXPOSE 8000
CMD ["python", "main.py"]
