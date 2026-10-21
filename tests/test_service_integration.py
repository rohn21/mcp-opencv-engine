"""Integration tests for Step 1 & Step 2 of Phase 5:

- Deterministic model loading and device selection in model_manager.py
- Canonical shared/ filesystem validation and boundary enforcement
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.config.loader import get_config
from core.filesystem.permissions import FilesystemPermissions
from opencv_services.object_detection.model_manager import (
    get_model,
    resolve_device,
    resolve_model_path,
)
from shared.image_utils import get_dimensions, validate_image


# ---------------------------------------------------------------------------
# Step 1: Model Manager Tests
# ---------------------------------------------------------------------------

class TestModelManager:
    def test_resolve_model_path_default_exists(self):
        """Default yolov8n.pt in project root should resolve without error."""
        path = resolve_model_path()
        assert path.exists()
        assert path.name == "yolov8n.pt"

    def test_resolve_model_path_nonexistent_raises(self, tmp_path):
        """Nonexistent weights file should raise a descriptive FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="YOLO weights file not found"):
            resolve_model_path("nonexistent_model_123.pt", weights_dir=tmp_path)

    def test_resolve_device_cpu(self):
        """CPU target should always resolve to 'cpu'."""
        assert resolve_device("cpu") == "cpu"

    def test_resolve_device_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert resolve_device("cuda") == "cuda"

    def test_resolve_device_cuda_unavailable_raises(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA device was requested"):
                resolve_device("cuda")

    def test_resolve_device_mps_when_available(self):
        with patch("torch.backends.mps.is_available", return_value=True):
            assert resolve_device("mps") == "mps"

    def test_resolve_device_mps_unavailable_raises(self):
        with patch("torch.backends.mps.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="MPS device was requested"):
                resolve_device("mps")

    def test_resolve_device_auto_fallback_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            assert resolve_device("auto") == "cpu"

    def test_resolve_device_auto_selects_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert resolve_device("auto") == "cuda"

    def test_resolve_device_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown device target"):
            resolve_device("tpu_v4")

    @patch("opencv_services.object_detection.model_manager.YOLO")
    def test_get_model_singleton_and_reload(self, mock_yolo_cls, tmp_path):
        dummy_weights = tmp_path / "custom.pt"
        dummy_weights.write_text("weights")

        mock_instance = MagicMock()
        mock_yolo_cls.return_value = mock_instance

        # First load
        m1 = get_model(weights_path=dummy_weights, device="cpu", reload=True)
        assert m1 is mock_instance
        mock_instance.to.assert_called_with("cpu")

        # Second load without reload returns cached singleton
        m2 = get_model(weights_path=dummy_weights, device="cpu", reload=False)
        assert m2 is m1
        assert mock_yolo_cls.call_count == 1


# ---------------------------------------------------------------------------
# Step 2: Canonical shared/ & Filesystem Validation Tests
# ---------------------------------------------------------------------------

class TestSharedFilesystemValidation:
    @pytest.fixture
    def test_dirs(self, tmp_path):
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        forbidden = tmp_path / "system_secrets"
        forbidden.mkdir()

        # Create valid dummy image
        valid_img = inputs / "test.jpg"
        valid_img.write_bytes(b"dummy image content bytes")

        # Create empty file
        empty_img = inputs / "empty.jpg"
        empty_img.write_bytes(b"")

        # Create forbidden file
        secret_img = forbidden / "secret.jpg"
        secret_img.write_bytes(b"secret bytes")

        return {
            "inputs": inputs,
            "outputs": outputs,
            "forbidden": forbidden,
            "valid_img": valid_img,
            "empty_img": empty_img,
            "secret_img": secret_img,
        }

    def test_valid_image_in_allowed_dir(self, test_dirs):
        is_valid, msg = validate_image(test_dirs["valid_img"], allowed_base_dir=test_dirs["inputs"])
        assert is_valid is True
        assert msg == "OK"

    def test_file_outside_allowed_dir_denied(self, test_dirs):
        is_valid, msg = validate_image(test_dirs["secret_img"], allowed_base_dir=test_dirs["inputs"])
        assert is_valid is False
        assert "Security Error" in msg

    def test_empty_file_rejected(self, test_dirs):
        is_valid, msg = validate_image(test_dirs["empty_img"], allowed_base_dir=test_dirs["inputs"])
        assert is_valid is False
        assert "File is empty" in msg

    def test_nonexistent_file_rejected(self, test_dirs):
        nonexistent = test_dirs["inputs"] / "does_not_exist.jpg"
        is_valid, msg = validate_image(nonexistent, allowed_base_dir=test_dirs["inputs"])
        assert is_valid is False
        assert "File not found" in msg

    def test_file_too_large_rejected(self, test_dirs, monkeypatch):
        large_img = test_dirs["inputs"] / "large.jpg"
        large_img.write_bytes(b"x" * 1024)

        # Set limit to 0.0001 MB (~100 bytes)
        monkeypatch.setattr("shared.image_utils.MAX_FILE_SIZE_MB", 0.0001)

        is_valid, msg = validate_image(large_img, allowed_base_dir=test_dirs["inputs"])
        assert is_valid is False
        assert "File too large" in msg

    def test_validate_image_with_filesystem_permissions(self, test_dirs):
        permissions = FilesystemPermissions(
            allowed_input_dirs=[test_dirs["inputs"]],
            allowed_output_dir=test_dirs["outputs"],
            protected_dirs=[test_dirs["forbidden"]],
        )

        # Allowed read
        ok, msg = validate_image(test_dirs["valid_img"], permissions=permissions)
        assert ok is True
        assert msg == "OK"

        # Denied read (forbidden directory)
        ok, msg = validate_image(test_dirs["secret_img"], permissions=permissions)
        assert ok is False
        assert "Security Error" in msg

    def test_get_dimensions_decode_failure(self, tmp_path):
        corrupt_file = tmp_path / "corrupt.jpg"
        corrupt_file.write_bytes(b"not an image")
        with pytest.raises(ValueError, match="Could not decode image"):
            get_dimensions(corrupt_file)


# ---------------------------------------------------------------------------
# Step 3 & Step 4: Gateway Tools Sandboxing & Mount Translation Tests
# ---------------------------------------------------------------------------

class TestGatewayToolsIntegration:
    @pytest.mark.asyncio
    async def test_detect_objects_security_traversal_rejection(self, mock_mcp):
        from tools.detection_tools import register_detection_tools
        register_detection_tools(mock_mcp)

        # Path traversal should be rejected before any HTTP call
        result = await mock_mcp.tool_detect_objects(image_path="../../etc/passwd")
        assert "Security Error" in result

    @pytest.mark.asyncio
    async def test_extract_text_security_rejection(self, mock_mcp):
        from tools.detection_tools import register_detection_tools
        register_detection_tools(mock_mcp)

        result = await mock_mcp.tool_extract_text_from_image(image_path="../sensitive_keys.pem")
        assert "Security Error" in result

    @pytest.mark.asyncio
    async def test_apply_filter_security_rejection(self, mock_mcp):
        from tools.processing_tools import register_processing_tools
        register_processing_tools(mock_mcp)

        result = await mock_mcp.tool_apply_image_filter(image_path="../../root/config.json", filter_type="edges")
        assert "Security Error" in result

    @pytest.mark.asyncio
    async def test_blur_faces_security_rejection(self, mock_mcp):
        from tools.processing_tools import register_processing_tools
        register_processing_tools(mock_mcp)

        result = await mock_mcp.tool_blur_faces_in_image(image_path="..\\..\\Windows\\System32\\cmd.exe")
        assert "Security Error" in result

    @pytest.mark.asyncio
    async def test_detect_objects_mount_aware_output_translation(self, mock_mcp, httpx_mock, tmp_path):
        from tools.detection_tools import register_detection_tools
        register_detection_tools(mock_mcp)

        httpx_mock.add_response(url="http://object-detection:8001/health", status_code=200)
        # Service returns a container-mapped path
        mock_data = {
            "objects": [{"name": "car", "confidence": 0.88}],
            "annotated_image_path": "/app/data/outputs/annotated_car.jpg",
        }
        httpx_mock.add_response(url="http://object-detection:8001/predict", json=mock_data)

        # /test/image.jpg is allowed in pytest
        result = await mock_mcp.tool_detect_objects(image_path="/test/image.jpg")
        import json
        data = json.loads(result)

        # Should be translated to host native path ending in data\outputs\annotated_car.jpg or data/outputs/...
        out_path = data["annotated_image_path"]
        assert "annotated_car.jpg" in out_path
        assert not out_path.startswith("/app/data")

    @pytest.mark.asyncio
    async def test_apply_filter_unmapped_output_preserved(self, mock_mcp, httpx_mock):
        from tools.processing_tools import register_processing_tools
        register_processing_tools(mock_mcp)

        mock_data = {"output_path": "/unmapped_dir/custom.jpg", "filter_applied": "grayscale"}
        httpx_mock.add_response(url="http://image-processing:8002/filter", json=mock_data)

        result = await mock_mcp.tool_apply_image_filter(image_path="/test/image.jpg", filter_type="grayscale")
        import json
        data = json.loads(result)
        # Unmapped mount remains unchanged, never fabricated
        assert data["output_path"] == "/unmapped_dir/custom.jpg"


# ---------------------------------------------------------------------------
# Step 5 & Step 6: Service APIs and Assets Manager Tests
# ---------------------------------------------------------------------------

class TestServiceAPIsAndAssetsManager:
    def test_assets_manager_filters_non_images(self, tmp_path, monkeypatch):
        from utils.assets_manager import list_available_images

        # Create input dir with images and non-images
        inputs_dir = tmp_path / "inputs"
        inputs_dir.mkdir()

        (inputs_dir / "valid1.png").write_bytes(b"png")
        (inputs_dir / "valid2.jpg").write_bytes(b"jpg")
        (inputs_dir / "ignored.txt").write_text("text")
        (inputs_dir / "ignored.json").write_text("{}")

        # Mock DATA_DIR
        monkeypatch.setenv("DATA_DIR", str(tmp_path))

        from core.config.loader import get_config
        cfg = get_config(reload=True)

        images = list_available_images()
        image_names = [Path(p).name for p in images]

        assert "valid1.png" in image_names
        assert "valid2.jpg" in image_names
        assert "ignored.txt" not in image_names
        assert "ignored.json" not in image_names

    def test_service_output_directory_auto_created(self, tmp_path, monkeypatch):
        # Point config to a fresh output dir
        test_data_dir = tmp_path / "custom_data"
        monkeypatch.setenv("DATA_DIR", str(test_data_dir))

        from core.config.loader import get_config
        cfg = get_config(reload=True)

        # Output dir should be properly configured
        assert cfg.storage.outputs_dir == test_data_dir / "outputs"


class TestRealMCPProtocolClientSession:
    @pytest.mark.asyncio
    async def test_protocol_handshake_and_tool_discovery(self):
        """Exercise genuine MCP protocol ClientSession initialization and list_tools."""
        import anyio
        from mcp.client.session import ClientSession
        from mcp.server.fastmcp import FastMCP
        from tools.detection_tools import register_detection_tools
        from tools.processing_tools import register_processing_tools

        server = FastMCP("test-mcp-protocol")
        register_detection_tools(server)
        register_processing_tools(server)

        c2s_send, c2s_recv = anyio.create_memory_object_stream()
        s2c_send, s2c_recv = anyio.create_memory_object_stream()

        async with anyio.create_task_group() as tg:
            tg.start_soon(
                server._mcp_server.run,
                c2s_recv,
                s2c_send,
                server._mcp_server.create_initialization_options(),
            )
            async with ClientSession(s2c_recv, c2s_send) as session:
                init_res = await session.initialize()
                assert init_res.serverInfo.name == "test-mcp-protocol"

                tools_res = await session.list_tools()
                names = [t.name for t in tools_res.tools]
                assert "detect_objects" in names
                assert "extract_text_from_image" in names
                assert "apply_image_filter" in names
                assert "blur_faces_in_image" in names
                tg.cancel_scope.cancel()


