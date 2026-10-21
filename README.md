# MCP OpenCV Engine

A production-ready, cross-platform Computer Vision backend exposing AI vision tools directly to LLMs via the **Model Context Protocol (MCP)**. The system enables AI assistants and IDEs (including Google Antigravity, Cursor, and Claude Desktop) to run object detection, image processing, face anonymization, and OCR text extraction through a unified gateway.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       LLM / MCP Client Application    |
                      |  (Google Antigravity, Cursor, Claude) |
                      +---------------------------------------+
                                          |
                        MCP Protocol (Streamable HTTP / STDIO)
                                          v
                      +---------------------------------------+
                      |         cv-mcp-gateway (Port 8000)    |
                      |          FastMCP Router & Bridge      |
                      +---------------------------------------+
                                     /    |    \
          +-------------------------+     |     +-------------------------+
          |                               |                               |
          v                               v                               v
+-------------------+           +-------------------+           +-------------------+
| object-detection  |           |  image-processing |           |    ocr-service    |
|    (Port 8001)    |           |    (Port 8002)    |           |    (Port 8003)    |
|  YOLOv8 Singleton |           |  OpenCV Caching   |           | EasyOCR/Tesseract |
+-------------------+           +-------------------+           +-------------------+
```

### Core Subsystems (`core/`)

1. **`core/environment/`** — Non-destructive detection of OS family, CPU architecture, Python runtime, GPU acceleration backends (CUDA, Apple Silicon MPS, or CPU fallback), and Docker/Compose availability.
2. **`core/config/`** — Pydantic v2 schemas and deterministic hierarchical configuration loader (`Profile`, `RuntimeMode`, `DeviceTarget`, ports, paths, and timeouts) with `.env` precedence.
3. **`core/filesystem/`** — Filesystem sandboxing, strict input/output directory boundary enforcement, path-traversal prevention, symlink escape checks, and cross-platform Host <-> Container path translation (`PathResolver`).
4. **`core/dependencies/`** — Preflight dependency checker scanning Python packages, version constraints, external binaries (Tesseract OCR), and container runtimes without mutating system state.
5. **`core/runtime/`** — Unified execution lifecycle orchestrator supporting both native Python processes (`NativeRuntime`) and isolated containers (`DockerRuntime`) with dynamic volume mount override generation (`.runtime/docker-compose.override.yml`).
6. **`core/client_config/`** — Non-destructive MCP client configuration registry with safe atomic merging and timestamped backups (`.bak_<timestamp>`), supporting Google Antigravity, Cursor, and Claude Desktop.
7. **`setup/`** — Unified CLI entry point (`setup/configure.py`) and read-only diagnostics engine (`setup/diagnostics.py`).

---

## Supported Platforms & Runtime Matrix

| Platform / OS | Hardware Backend | Native Mode | Docker Mode | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Windows 10/11 x64** | CUDA / CPU | Supported | Supported | **Tested & Verified** (Active development baseline) |
| **Linux (Ubuntu/Debian) x64** | CUDA / CPU | Supported | Supported | **Verified via Mock Matrix** (Container paths & POSIX translation validated) |
| **macOS (Darwin / Apple Silicon)** | MPS / CPU | Supported | Supported | **Untested on physical hardware** (Architecture and detection logic verified in suite) |

> [!NOTE]
> Platforms marked as **Untested** are architecturally supported using standard POSIX/`pathlib` APIs and capability detectors, but have not undergone end-to-end regression runs on physical hardware.

---

## Quickstart & Setup Workflow

The repository includes a unified, cross-platform CLI tool: `setup/configure.py`.

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/rohn21/mcp-opencv-engine.git
cd mcp-opencv-engine

# Create and activate virtual environment
# Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Dependency Profile Installation

Dependencies are modularized under `requirements/` to avoid unnecessary bloat (or install the unified `requirements.txt`):

```bash
# Gateway only (lightweight client runner):
pip install -r requirements/gateway.txt

# Full native development and testing:
pip install -r requirements/dev-test.txt

# Or unified complete environment:
pip install -r requirements.txt

# Individual native microservices:
pip install -r requirements/image-processing.txt
pip install -r requirements/object-detection-cpu.txt   # or object-detection-gpu.txt
pip install -r requirements/ocr.txt
```

### 3. Run Guided Setup

Run the interactive configuration wizard:

```bash
python setup/configure.py --setup
```

Or run non-interactively with defaults:

```bash
python setup/configure.py --setup --yes
```

The guided setup workflow:
1. Detects OS, CPU, Python version, hardware acceleration, and Docker status.
2. Prompts you to choose between **Auto**, **Docker**, or **Native** mode.
3. Preflights prerequisites and warns if any dependencies or tools are missing.
4. Verifies and ensures the sandboxed directories (`data/inputs` and `data/outputs`).
5. Previews MCP client configuration changes as a structured diff.
6. Asks for explicit confirmation before applying changes (creating timestamped backups of existing configs).
7. Provides concrete commands to start the services.

---

## Read-Only Diagnostics (`--doctor`)

The `--doctor` command runs a non-destructive audit of the entire system, dependencies, models, directory permissions, and service health without modifying files, starting processes, or altering configurations.

### Running Diagnostics

```bash
# Human-readable report:
python setup/configure.py --doctor

# Machine-readable JSON output:
python setup/configure.py --doctor --json
```

### Diagnostics Exit Codes

The CLI returns predictable exit codes suitable for CI/CD and automated scripting:

| Exit Code | Status | Meaning |
| :---: | :---: | :--- |
| **`0`** | `PASS` | All required dependencies, models, directories, and configurations are valid. |
| **`1`** | `FAIL` | One or more critical checks failed (e.g. missing YOLO weights, invalid JSON, or broken dependency). |
| **`2`** | `WARN` | System is operational, but optional components are absent (e.g. CPU fallback, services stopped, or missing optional tools). |

### Interpreting Service Health

The diagnostics engine accurately distinguishes between an **offline** service and a **failing** service:
- If a service endpoint returns `ConnectionRefused` or has no active listener, `--doctor` reports `[WARN] Service not currently running (Normal if runtime is stopped)`.
- If a service is actively listening on its port but returns an HTTP 5xx error, `--doctor` reports `[FAIL] Service health endpoint returned HTTP error 500`.

---

## Execution Modes: Docker vs Native

### Option A: Docker Compose Execution (Recommended for Microservice Isolation)

Ensure Docker Desktop or the Docker daemon is running:

```bash
# Set mode to Docker
python setup/configure.py --configure-runtime docker

# Start all microservices and gateway with standard default mounts (./data:/app/data):
docker compose up -d

# Or when custom directory mounts are configured, include the generated override:
docker compose -f docker-compose.yml -f .runtime/docker-compose.override.yml up -d

# Verify health
python setup/configure.py --doctor
```

Dynamic directory mounts are automatically translated: host inputs are mounted read-only (`:ro`) and host outputs are mounted read-write (`:rw`) to prevent container breakout. (Note: `core/runtime/docker.py` passes the override file automatically during programmatic lifecycle management).

### Option B: Native Microservices Execution

Ensure all service requirements are installed (`requirements/dev-test.txt`):

```bash
# Set mode to Native
python setup/configure.py --configure-runtime native

# Start individual microservices in separate background processes or terminals:
python -m uvicorn opencv_services.object_detection.api:app --host 0.0.0.0 --port 8001 &
python -m uvicorn opencv_services.image_processing.api:app --host 0.0.0.0 --port 8002 &
python -m uvicorn opencv_services.ocr_service.api:app --host 0.0.0.0 --port 8003 &

# Start the MCP Gateway:
python -m mcp_gateway.main
```

---

## MCP Client Configuration

Register the gateway with your preferred MCP client using the setup CLI. All configuration updates are performed using atomic writes with timestamped backups (`.bak_<timestamp>`).

### 1. Google Antigravity IDE

```bash
# Register workspace configuration (default):
python setup/configure.py --configure-client antigravity --yes

# Or preview changes without writing:
python setup/configure.py --configure-client antigravity --dry-run
```

Target location: `.agents/mcp_config.json` (Workspace) or `~/.gemini/config/mcp_config.json` (Global).

Config schema applied:
```json
{
  "mcpServers": {
    "cv-mcp-gateway": {
      "serverUrl": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

### 2. Cursor AI Code Editor

```bash
# Register with Cursor:
python setup/configure.py --configure-client cursor --yes
```

Target location: `.cursor/mcp.json` (Workspace) or `~/.cursor/mcp.json` (Global).

Config schema applied (Streamable HTTP / SSE):
```json
{
  "mcpServers": {
    "cv-mcp-gateway": {
      "url": "http://127.0.0.1:8000/sse"
    }
  }
}
```

### 3. Claude Desktop

Claude Desktop strictly supports local `stdio` transport in its configuration file.

The recommended and error-free way to configure Claude Desktop is using the setup CLI, which automatically detects your active virtual environment, canonical project paths, and service endpoints:

```bash
# Preview Claude Desktop registration without touching disk:
python setup/configure.py --configure-client claude --transport stdio --scope global --dry-run

# Apply registration into Claude Desktop configuration:
python setup/configure.py --configure-client claude --transport stdio --scope global --yes
```

Target location: `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS).

Generated configuration schema (with portable placeholders):
```json
{
  "mcpServers": {
    "cv-mcp-gateway": {
      "command": "<path-to-repository>/.venv/Scripts/python.exe",
      "args": [
        "<path-to-repository>/mcp_gateway/main.py",
        "--transport",
        "stdio"
      ],
      "env": {
        "DETECTION_SERVICE_URL": "http://localhost:8001",
        "PROCESSING_SERVICE_URL": "http://localhost:8002",
        "OCR_SERVICE_URL": "http://localhost:8003",
        "DATA_DIR": "data"
      }
    }
  }
}
```

> [!NOTE]
> **Subprocess Path Resolution:** When Claude Desktop spawns the STDIO child process from arbitrary working directories, `core.config.loader.get_project_root()` and `core.filesystem.PathResolver` automatically anchor relative paths (`data/inputs`, `data/outputs`) to the canonical project root discovered from repository markers.

> [!IMPORTANT]
> Claude Desktop does not support remote HTTP URLs in its configuration file. If remote transport is requested, the adapter cleanly rejects it with explicit instructions to use Claude Desktop Settings → Developer / Connectors.

---

## Available MCP Tools

Once connected, the LLM assistant gains access to the following four tools:

1. **`detect_objects`**
   - Detects visual objects using YOLOv8.
   - Arguments:
     - `image_path` (str, required): Absolute path or relative filename of the input image.
     - `confidence` (float, optional, default: `0.25`): Minimum confidence threshold (0.0 to 1.0). (Also accepts `confidence_threshold` as an alias).
   - Returns: JSON string with detected objects, bounding boxes (`x1`, `y1`, `x2`, `y2`), class names, confidence scores, and `annotated_image_path`.

2. **`apply_image_filter`**
   - Applies image processing filters via OpenCV.
   - Arguments:
     - `image_path` (str, required): Path to the image.
     - `filter_type` (str, required): Strictly one of `grayscale`, `blur`, `edges`, or `threshold`.
   - Returns: JSON string with `status`, `filter_applied`, `output_image_path`, `original_dimensions`, and `processed_dimensions`.

3. **`blur_faces_in_image`**
   - Anonymizes detected faces using OpenCV Haar Cascades.
   - Arguments:
     - `image_path` (str, required): Path to the image.
     - `blur_intensity` (int, optional, default: `99`): Blur kernel size (must be an odd integer).
   - Returns: JSON string with `status`, `faces_detected`, and `output_image_path`.

4. **`extract_text_from_image`**
   - Extracts printed and handwritten text using EasyOCR with automatic Tesseract fallback.
   - Arguments:
     - `image_path` (str, required): Path to the image.
   - Note: Target OCR languages are configured globally via `OCR_LANGUAGES` in `.env` (e.g. `OCR_LANGUAGES="en,fr"`).
   - Returns: JSON string with `status`, `text_found`, `extracted_text`, and `text_blocks` (with coordinates and confidence scores).

### MCP Resources & Prompts

- **Resource `images://available`** — Lists all available images currently present in the inputs directory (`data/inputs`).
- **Prompt `full_image_analysis(image_path: str)`** — Standardized template guiding the AI assistant through structured, evidence-based computer vision analysis.

---

## Filesystem Security & Sandbox Model

All file operations are strictly guarded by `core/filesystem/permissions.py`:
- **Allowed Input Directories**: Operations may only read from authorized locations (defaults to `data/inputs`).
- **Allowed Output Directory**: Output images may only be written to authorized destinations (defaults to `data/outputs`).
- **Path Traversal Protection**: Traversal sequences (`../`, `..\\`) and null bytes (`\0`) are rejected immediately with `PathTraversalError`.
- **Symlink Jail Prevention**: Symlinks pointing outside the allowed directory sandbox are rejected with `AccessDeniedError`.
- **Protected Zones**: System roots, OS directories, source directories (`core`, `mcp_gateway`), and `.git` are strictly blacklisted from volume mounting and file access.

---

## DevFest Baroda 2026 Live Demo Guide

To reproduce the live demonstration from a cold start on stage or during an evaluation:

### Step 1: Cold-Start Preflight Verification
Verify prerequisites and system status before starting services:
```bash
python setup/configure.py --doctor
```
*(Diagnostics should report 0 failures. Warnings for CPU compute device or offline Docker are expected).*

### Step 2: Start Native Microservices
In separate terminals (or background jobs), start the three computer-vision microservices:
```bash
# Terminal 1: Object Detection (YOLOv8)
python -m uvicorn opencv_services.object_detection.api:app --host 127.0.0.1 --port 8001

# Terminal 2: Image Processing & Face Blurring (OpenCV)
python -m uvicorn opencv_services.image_processing.api:app --host 127.0.0.1 --port 8002

# Terminal 3: Optical Character Recognition (EasyOCR / Tesseract)
python -m uvicorn opencv_services.ocr_service.api:app --host 127.0.0.1 --port 8003
```

### Step 3: Confirm Microservice Readiness
Re-run diagnostics to verify all three services report `Online and healthy (HTTP 200)`:
```bash
python setup/configure.py --doctor
```

### Step 4 & 5: Choose Client Integration Workflow

Depending on your client, follow either **Workflow A (Streamable HTTP)** or **Workflow B (Local STDIO)**:

#### Workflow A: Streamable HTTP Clients (Google Antigravity IDE, Cursor)

For clients configured to connect over the network to the running FastMCP HTTP gateway:

1. **Start the MCP Gateway:**
   ```bash
   # Terminal 4: FastMCP Gateway (Streamable HTTP)
   python -m mcp_gateway.main --transport streamable-http
   ```
   *The gateway binds to `http://127.0.0.1:8000` (or `0.0.0.0:8000`) and exposes the `/mcp` endpoint.*

2. **Verify the `/mcp` endpoint is listening:**
   ```bash
   curl -I http://127.0.0.1:8000/mcp
   ```

3. **Register your HTTP client:**
   ```bash
   # For Google Antigravity IDE (Workspace scope: .agents/mcp_config.json):
   python setup/configure.py --configure-client antigravity --transport streamable-http --scope workspace --yes

   # For Cursor AI Code Editor (Workspace scope: .cursor/mcp.json):
   python setup/configure.py --configure-client cursor --transport streamable-http --scope workspace --yes
   ```
   Both clients connect directly to the running HTTP gateway at `http://127.0.0.1:8000/mcp`.

---

#### Workflow B: Local STDIO Subprocess (Claude Desktop)

Claude Desktop manages its own subprocesses and does **not** connect to the TCP port 8000 server. Instead, Claude Desktop launches its own internal gateway process communicating over standard input/output (`stdio`).

> [!IMPORTANT]
> The three downstream microservices (Terminals 1, 2, 3 on ports 8001, 8002, 8003) **must still be running**. Claude Desktop's spawned gateway process makes direct HTTP calls to `http://localhost:8001`, `8002`, and `8003`.
> You do **not** run Terminal 4 (port 8000) for Claude Desktop STDIO mode.

1. **Preview and Apply Claude Desktop configuration:**
   ```bash
   # Preview configuration diff without modifying files:
   python setup/configure.py --configure-client claude --transport stdio --scope global --dry-run

   # Apply configuration:
   python setup/configure.py --configure-client claude --transport stdio --scope global --yes
   ```

2. **Verify configuration file (`%APPDATA%\Claude\claude_desktop_config.json`):**
   - **`command`**: Points to the project virtual environment (`.venv\Scripts\python.exe`).
   - **`args`**: `["<absolute-path>\\mcp_gateway\\main.py", "--transport", "stdio"]`.
   - **`env`**: Explicitly defines downstream service endpoints:
     - `DETECTION_SERVICE_URL: http://localhost:8001`
     - `PROCESSING_SERVICE_URL: http://localhost:8002`
     - `OCR_SERVICE_URL: http://localhost:8003`
     - `DATA_DIR: data`

3. **Restart Claude Desktop completely** to spawn the gateway subprocess and access tools.

### Step 6: Run Live AI Vision Inferences via LLM
Ask the LLM in your IDE:
- *"Detect all objects in `data/inputs/obj_test.jpg` with confidence 0.25 and show the annotated image."*
  - **Result**: Returns detected object classes, bounding boxes, and saves annotated image in `data/outputs/`.
- *"Extract all readable text from `data/inputs/testocr.jpg`."*
  - **Result**: Returns extracted text paragraph with bounding boxes and confidence metrics.
- *"Apply a privacy face-blur filter to `data/inputs/face_test.jpg`."*
  - **Result**: Anonymizes detected faces and saves `blurred_faces_*.jpg` in `data/outputs/`.
- *"Apply Canny edge detection to `data/inputs/obj_test.jpg`."*
  - **Result**: Applies OpenCV Canny edges and saves output in `data/outputs/`.

### Step 7: Demonstrate Sandbox Defense & Safe Shutdown
1. **Security Sandbox Defense**:
   - Ask the LLM: *"Inspect `../../secret.txt`."*
   - **Observed Result**: The gateway immediately blocks the call before making downstream requests, returning:
     `Security Error: Read access denied for path outside allowed input directories`.
2. **Safe Shutdown**:
   - Stop Terminal 4 (Gateway) with `Ctrl+C`.
   - Stop Terminals 1, 2, 3 (Microservices) with `Ctrl+C`.
   - Temporary outputs in `data/outputs/` can be safely cleaned up without affecting git state.

---

## Troubleshooting Guide

### 1. Missing YOLO Weights (`yolov8n.pt`)
- **Symptom**: `--doctor` reports `[FAIL] YOLO Weights: Model file 'yolov8n.pt' not found`.
- **Remedy**: Place the model file in the repository root or set `YOLO_WEIGHTS_DIR` in `.env`.
  ```bash
  python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
  ```

### 2. Tesseract OCR Not Found
- **Symptom**: `--doctor` warns `External Tool: Tesseract OCR: 'tesseract' not found on PATH`.
- **Remedy**: EasyOCR is used as the primary engine. To enable Tesseract fallback:
  - Windows: `winget install UB-Mannheim.TesseractOCR`
  - Linux: `sudo apt install -y tesseract-ocr`
  - macOS: `brew install tesseract`

### 3. Docker Daemon Not Running
- **Symptom**: `--doctor` warns `Docker daemon is not running`.
- **Remedy**: Start Docker Desktop (Windows/macOS) or `sudo systemctl start docker` (Linux).

### 4. Client Config File Syntax Error
- **Symptom**: `--doctor` reports `[FAIL] Config file contains errors`.
- **Remedy**: Restore from the latest `.bak_<timestamp>` backup created in the client configuration directory.

---

## Testing & Quality Assurance

Run the comprehensive unit and integration test suite:

```bash
# Run full regression suite (175 tests):
pytest tests/ -v

# Run real network TCP Streamable HTTP integration test:
pytest tests/test_network_mcp.py -v

# Run setup CLI and diagnostics tests only:
pytest tests/test_setup_cli.py -v

# Run with summary:
pytest tests/ -q
```

### Verified Test Baseline

- **174 Passed**, **1 Skipped** (Windows symlink privilege test skipped as expected on standard user accounts without developer elevation).
- **0 Failures**.

---

## Known Limitations & Roadmap

1. **Model Weight Distribution**: Pre-trained YOLO weights (`yolov8n.pt`) are cached locally in the repository root upon first inference.
2. **Claude Desktop Remote Connectors**: Claude Desktop requires manual UI configuration for remote HTTP SSE servers; file-based JSON configuration is limited to local `stdio`.
3. **Apple Silicon Hardware Verification**: MPS acceleration paths are covered by mock tests, but physical macOS verification remains on the post-Phase 8 roadmap.
4. **Docker Daemon Offline in CI**: Docker Compose orchestration is validated via Compose file validation and unit test mocks; full containerized execution requires a running Docker daemon.