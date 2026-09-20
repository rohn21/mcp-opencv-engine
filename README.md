# MCP OpenCV Engine

A microservice-based Computer Vision backend exposing AI tools directly to LLMs via the **Model Context Protocol (MCP)**. The system allows an AI assistant to apply image filters, detect objects, blur faces, and extract text from images through a unified MCP gateway.

---

## Architecture Overview

```
LLM / AI Client
      |
      | (MCP via STDIO)
      v
  mcp-gateway          (Port 8000)
      |
      |--- /filter, /blur_faces  ---->  image-processing   (Port 8002)
      |--- /predict              ---->  object-detection   (Port 8001)
      |--- /extract_text         ---->  ocr-service        (Port 8003)
```

The **MCP Gateway** is a lightweight FastMCP server that receives tool calls from an LLM and routes them to the appropriate downstream microservice. Each microservice is independently deployed and handles only its focused domain.

---

## Services

### mcp-gateway
- Communicates with the AI client via **STDIO** (Model Context Protocol)
- Registers and exposes the following tools to the LLM:
  - `apply_image_filter` — Applies grayscale, blur, edge, or threshold filters
  - `blur_faces_in_image` — Detects and anonymizes faces using Haar Cascades
  - `detect_objects` — Runs YOLOv8 object detection
  - `extract_text_from_image` — Runs OCR to extract readable text
- Also exposes an MCP **Resource** (`images://available`) that lists all available input images
- Exposes an MCP **Prompt** (`full_image_analysis`) for comprehensive scene analysis workflows

### image-processing (Port 8002)
- Built with **FastAPI + OpenCV** (CPU only)
- Handles filter application and face blurring
- Implements **deterministic output caching** using MD5 hashing — repeated requests skip heavy OpenCV processing and return instantly from disk

### object-detection (Port 8001)
- Built with **FastAPI + Ultralytics YOLOv8**
- Loads `yolov8n.pt` using a singleton pattern (loads once, reused across requests)
- Auto-detects and uses CUDA GPU if available, falls back to CPU
- Returns annotated images with bounding boxes and confidence scores

### ocr-service (Port 8003)
- Built with **FastAPI + EasyOCR + Tesseract**
- Dual-engine design: tries **EasyOCR** first for accuracy
- On EasyOCR failure (e.g., OOM on low-RAM machines), automatically falls back to **Tesseract**
- Tesseract downscales large images (max 1000px width) before processing to manage memory

---

## Request Flow — Example (OCR with Preprocessing)

1. LLM calls `apply_image_filter(image_path, filter_type='grayscale')`
2. Gateway routes to `image-processing` → returns grayscale output path
3. LLM calls `apply_image_filter(grayscale_path, filter_type='threshold')`
4. Gateway routes to `image-processing` → returns high-contrast binary image path
5. LLM calls `extract_text_from_image(threshold_path)`
6. Gateway routes to `ocr-service` → EasyOCR or Tesseract runs → returns extracted text blocks with confidence scores

---

## Minimum System Requirements

| Component        | Minimum Specification                     |
|------------------|-------------------------------------------|
| CPU              | Intel Core i5 11th Gen or equivalent      |
| RAM              | 8 GB (16 GB recommended)                  |
| Storage          | SSD strongly recommended                  |
| GPU              | Optional — NVIDIA GPU enables EasyOCR acceleration |
| OS               | Linux (Ubuntu 20.04+) or Windows 10+      |
| Python           | 3.10+                                     |
| Docker           | 24.0+                                     |

---

## Running Locally (Without Docker)

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r mcp_gateway/requirements.txt

# 3. Start each microservice in a separate terminal
uvicorn opencv_services.object_detection.api:app --host 0.0.0.0 --port 8001
uvicorn opencv_services.image_processing.api:app --host 0.0.0.0 --port 8002
uvicorn opencv_services.ocr_service.api:app --host 0.0.0.0 --port 8003

# 4. Start the MCP Gateway
python mcp_gateway/main.py
```

## Running with Docker

```bash
docker compose up --build
```

> Ensure the `.env` file has `DETECTION_SERVICE_URL`, `PROCESSING_SERVICE_URL`, and `DATA_DIR` set if you want to override defaults.

---

## Project Structure

```
computer_vision_mcp/
├── mcp_gateway/
│   ├── main.py                  # MCP server entry point
│   ├── tools/
│   │   ├── detection_tools.py   # detect_objects, extract_text_from_image tools
│   │   └── processing_tools.py  # apply_image_filter, blur_faces_in_image tools
│   └── utils/
│       └── assets_manager.py    # Lists available input images
│
├── opencv_services/
│   ├── image_processing/
│   │   ├── api.py               # /filter and /blur_faces endpoints
│   │   └── opencv_ops.py        # Filter registry (grayscale, blur, edges, threshold)
│   ├── object_detection/
│   │   ├── api.py               # /predict endpoint
│   │   └── model_manager.py     # YOLOv8 singleton loader
│   └── ocr_service/
│       └── api.py               # /extract_text endpoint (EasyOCR + Tesseract)
│
├── shared/
│   ├── schemas.py               # Pydantic request/response models
│   └── image_utils.py           # Image path validation utility
│
├── data/
│   ├── inputs/                  # Place input images here
│   └── outputs/                 # Processed images saved here
│
├── docker-compose.yml
├── .gitignore
└── README.md
```

---
## Using as an MCP Server in AI IDEs

This project exposes itself as an MCP server over **STDIO**. Before registering it in any IDE, ensure all three microservices are already running locally (or via Docker), then point the IDE's MCP config to `mcp_gateway/main.py`.

**Pre-requisite — start the backend services first:**
```bash
source .venv/bin/activate
uvicorn opencv_services.object_detection.api:app --host 0.0.0.0 --port 8001 &
uvicorn opencv_services.image_processing.api:app --host 0.0.0.0 --port 8002 &
uvicorn opencv_services.ocr_service.api:app --host 0.0.0.0 --port 8003 &
```

---

### Claude Desktop

Edit `claude_desktop_config.json` (usually at `~/.claude/claude_desktop_config.json` on Linux/macOS or `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "cv-mcp-gateway": {
      "command": "/absolute/path/to/computer_vision_mcp/.venv/bin/python3",
      "args": ["/absolute/path/to/computer_vision_mcp/mcp_gateway/main.py"],
      "env": {
        "DETECTION_SERVICE_URL": "http://localhost:8001",
        "PROCESSING_SERVICE_URL": "http://localhost:8002",
        "OCR_SERVICE_URL": "http://localhost:8003"
      }
    }
  }
}
```

---

### Cursor

Open **Cursor Settings → Features → MCP Servers** or directly edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "cv-mcp-gateway": {
      "command": "/absolute/path/to/computer_vision_mcp/.venv/bin/python3",
      "args": ["/absolute/path/to/computer_vision_mcp/mcp_gateway/main.py"],
      "env": {
        "DETECTION_SERVICE_URL": "http://localhost:8001",
        "PROCESSING_SERVICE_URL": "http://localhost:8002",
        "OCR_SERVICE_URL": "http://localhost:8003"
      }
    }
  }
}
```

Then restart Cursor. The tools (`apply_image_filter`, `detect_objects`, `blur_faces_in_image`, `extract_text_from_image`) will appear in the Composer tool list.

---

### Antigravity (Google DeepMind)

Edit the MCP settings file at `~/.gemini/antigravity/mcp_settings.json`:

```json
{
  "mcpServers": {
    "cv-mcp-gateway": {
      "command": "/absolute/path/to/computer_vision_mcp/.venv/bin/python3",
      "args": ["/absolute/path/to/computer_vision_mcp/mcp_gateway/main.py"],
      "env": {
        "DETECTION_SERVICE_URL": "http://localhost:8001",
        "PROCESSING_SERVICE_URL": "http://localhost:8002",
        "OCR_SERVICE_URL": "http://localhost:8003"
      }
    }
  }
}
```

---

### Windsurf (Codeium)

Edit `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "cv-mcp-gateway": {
      "command": "/absolute/path/to/computer_vision_mcp/.venv/bin/python3",
      "args": ["/absolute/path/to/computer_vision_mcp/mcp_gateway/main.py"],
      "env": {
        "DETECTION_SERVICE_URL": "http://localhost:8001",
        "PROCESSING_SERVICE_URL": "http://localhost:8002",
        "OCR_SERVICE_URL": "http://localhost:8003"
      }
    }
  }
}
```

---

### Any MCP-Compatible Client (Generic)

Any IDE or tool that supports the MCP standard can integrate this server using the universal pattern below:

| Field     | Value |
|-----------|-------|
| Transport | `stdio` |
| Command   | `/path/to/.venv/bin/python3` |
| Args      | `["/path/to/mcp_gateway/main.py"]` |
| Env vars  | `DETECTION_SERVICE_URL`, `PROCESSING_SERVICE_URL`, `OCR_SERVICE_URL` |

> **Tip:** Replace all `/absolute/path/to/computer_vision_mcp` with the actual path where you cloned this repo. On Linux you can find it quickly by running `pwd` from inside the project folder.
