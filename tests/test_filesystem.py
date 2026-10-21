"""Unit tests for core.filesystem module (permissions, security, and path resolver)."""

import os
from pathlib import Path
import pytest

from core.config.loader import load_config
from core.filesystem.path_resolver import PathResolver
from core.filesystem.permissions import (
    AccessDeniedError,
    FilesystemPermissions,
    PathTraversalError,
    is_subpath,
)


def test_is_subpath(tmp_path: Path):
    parent = tmp_path / "sandbox"
    parent.mkdir()
    child = parent / "sub" / "file.txt"

    assert is_subpath(child, parent) is True
    assert is_subpath(parent, parent) is True
    assert is_subpath(tmp_path, parent) is False


def test_permissions_read_and_write_boundaries(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    other_dir = tmp_path / "other"
    input_dir.mkdir()
    output_dir.mkdir()
    other_dir.mkdir()

    valid_input = input_dir / "photo.jpg"
    valid_input.write_text("dummy", encoding="utf-8")
    invalid_input = other_dir / "secret.txt"
    invalid_input.write_text("secret", encoding="utf-8")

    perms = FilesystemPermissions(
        allowed_input_dirs=[input_dir],
        allowed_output_dir=output_dir,
    )

    # Valid read
    assert perms.validate_read(valid_input) == valid_input.resolve()

    # Unauthorized read
    with pytest.raises(AccessDeniedError):
        perms.validate_read(invalid_input)

    # Traversal attack
    traversal_path = str(input_dir / ".." / "other" / "secret.txt")
    with pytest.raises(AccessDeniedError):
        perms.validate_read(traversal_path)

    # Null byte attack
    with pytest.raises(PathTraversalError):
        perms.validate_read(str(valid_input) + "\0.hack")

    # Valid write
    valid_out = output_dir / "result.jpg"
    assert perms.validate_write(valid_out) == valid_out.resolve()

    # Unauthorized write (trying to write into input dir)
    with pytest.raises(AccessDeniedError):
        perms.validate_write(input_dir / "overwrite.jpg")


def test_permissions_protected_directories(tmp_path: Path):
    allowed_dir = tmp_path / "allowed"
    protected_dir = allowed_dir / ".git"
    allowed_dir.mkdir()
    protected_dir.mkdir()

    secret_file = protected_dir / "config"
    secret_file.write_text("keys", encoding="utf-8")

    perms = FilesystemPermissions(
        allowed_input_dirs=[allowed_dir],
        allowed_output_dir=allowed_dir,
        protected_dirs=[protected_dir],
    )

    with pytest.raises(AccessDeniedError):
        perms.validate_read(secret_file)

    with pytest.raises(AccessDeniedError):
        perms.validate_write(secret_file)


def test_permissions_is_safe_path(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    f = input_dir / "test.png"
    f.touch()

    perms = FilesystemPermissions(allowed_input_dirs=[input_dir])

    ok, err = perms.is_safe_path(f, mode="read")
    assert ok is True
    assert err is None

    ok_bad, err_bad = perms.is_safe_path("C:/Windows/System32/cmd.exe", mode="read")
    assert ok_bad is False
    assert err_bad is not None


def test_permissions_symlink_escape(tmp_path: Path):
    sandbox = tmp_path / "sandbox"
    outside = tmp_path / "outside"
    sandbox.mkdir()
    outside.mkdir()

    target = outside / "secret.png"
    target.write_text("data", encoding="utf-8")
    link = sandbox / "link.png"

    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported on this OS without administrative privileges.")

    perms = FilesystemPermissions(allowed_input_dirs=[sandbox])
    with pytest.raises(AccessDeniedError):
        perms.validate_read(link)


def test_path_resolver_container_to_native(tmp_path: Path):
    # Setup test workspace
    workspace = tmp_path / "project"
    data_dir = workspace / "data"
    inputs_dir = data_dir / "inputs"
    outputs_dir = data_dir / "outputs"
    inputs_dir.mkdir(parents=True)
    outputs_dir.mkdir(parents=True)

    test_img = inputs_dir / "sample.jpg"
    test_img.write_text("img", encoding="utf-8")

    config = load_config(
        use_system_env=False,
        storage={
            "data_dir": data_dir,
            "inputs_dir": inputs_dir,
            "outputs_dir": outputs_dir,
            "allowed_input_directories": [inputs_dir],
            "allowed_output_directory": outputs_dir,
            "docker_mounts": {str(data_dir): "/app/data"},
        },
    )

    resolver = PathResolver(config=config, project_root=workspace)

    # 1. Container path translation
    container_input = "/app/data/inputs/sample.jpg"
    native_res = resolver.to_native(container_input)
    assert native_res == test_img.resolve()

    # 2. Host path to container
    container_res = resolver.to_container(test_img)
    assert container_res == "/app/data/inputs/sample.jpg"

    # 3. Relative path translation
    rel_native = resolver.to_native("sample.jpg")
    assert rel_native == test_img.resolve()

    rel_cont = resolver.to_container("sample.jpg")
    assert rel_cont == "/app/data/inputs/sample.jpg"


def test_path_resolver_resolve_input_and_output(tmp_path: Path):
    workspace = tmp_path / "project"
    data_dir = workspace / "data"
    inputs_dir = data_dir / "inputs"
    outputs_dir = data_dir / "outputs"
    inputs_dir.mkdir(parents=True)
    outputs_dir.mkdir(parents=True)

    test_img = inputs_dir / "obj_test.jpg"
    test_img.write_text("pixels", encoding="utf-8")

    config = load_config(
        use_system_env=False,
        storage={
            "data_dir": data_dir,
            "inputs_dir": inputs_dir,
            "outputs_dir": outputs_dir,
            "allowed_input_directories": [inputs_dir],
            "allowed_output_directory": outputs_dir,
            "docker_mounts": {str(data_dir): "/app/data"},
        },
    )

    resolver = PathResolver(config=config, project_root=workspace)

    # Resolve input for Docker service
    resolved_docker = resolver.resolve_input("/app/data/inputs/obj_test.jpg", target_is_docker=True)
    assert resolved_docker == "/app/data/inputs/obj_test.jpg"

    # Resolve input for Native service
    resolved_native = resolver.resolve_input("/app/data/inputs/obj_test.jpg", target_is_docker=False)
    assert resolved_native == str(test_img.resolve())

    # Resolve output for Docker service
    out_docker = resolver.resolve_output(outputs_dir / "result.jpg", target_is_docker=True)
    assert out_docker == "/app/data/outputs/result.jpg"

    # Resolve output for Native service
    out_native = resolver.resolve_output("/app/data/outputs/result.jpg", target_is_docker=False)
    assert out_native == str((outputs_dir / "result.jpg").resolve())

    # Unauthorized input resolution fails security check
    outside_file = tmp_path / "outside.jpg"
    outside_file.touch()
    with pytest.raises(AccessDeniedError):
        resolver.resolve_input(outside_file)
