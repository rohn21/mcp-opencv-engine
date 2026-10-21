"""Non-destructive JSON merger, timestamped backup creator, and atomic writer for MCP configs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


def create_backup(target_file: Path) -> Path:
    """Create a timestamped backup of the existing configuration file."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = target_file.parent / f"{target_file.name}.bak_{timestamp}"

    # If already exists in same second, append counter
    counter = 1
    while backup_path.exists():
        backup_path = target_file.parent / f"{target_file.name}.bak_{timestamp}_{counter}"
        counter += 1

    shutil.copy2(target_file, backup_path)
    return backup_path


def load_json_safe(file_path: Path) -> Dict[str, Any]:
    """Load JSON from file.

    Raises ValueError if file exists but contains invalid/malformed JSON.
    Returns empty dict if file does not exist.
    """
    if not file_path.is_file():
        return {}

    try:
        content = file_path.read_text(encoding="utf-8")
        if not content.strip():
            return {}
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError(
                f"Configuration at '{file_path}' must be a JSON object, found {type(data).__name__}."
            )
        return data
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Cannot merge configuration: existing file at '{file_path}' contains malformed JSON: {exc}."
        ) from exc
    except PermissionError as exc:
        raise PermissionError(f"Permission denied reading configuration file at '{file_path}': {exc}") from exc


def merge_server_entry(
    existing_config: Dict[str, Any],
    server_name: str,
    server_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Perform a deep merge preserving all other server entries and root keys."""
    # Clone dict so we do not mutate the input
    merged = json.loads(json.dumps(existing_config))

    if "mcpServers" not in merged or not isinstance(merged["mcpServers"], dict):
        merged["mcpServers"] = {}

    # Merge or insert new server
    merged["mcpServers"][server_name] = server_payload
    return merged


def generate_diff_preview(
    existing_config: Dict[str, Any],
    merged_config: Dict[str, Any],
    server_name: str,
) -> str:
    """Produce a human-readable preview of the proposed changes."""
    prev_server = existing_config.get("mcpServers", {}).get(server_name)
    new_server = merged_config.get("mcpServers", {}).get(server_name)

    action = "Updating existing" if prev_server is not None else "Adding new"
    preview_lines = [
        f"=== Proposed MCP Client Config Change ({action} '{server_name}') ===",
        f"Server Name: {server_name}",
        "Payload:",
        json.dumps({server_name: new_server}, indent=2),
        "",
        f"Total Servers in Target Config: {len(merged_config.get('mcpServers', {}))}",
        f"Other Preserved Servers: {[k for k in merged_config.get('mcpServers', {}).keys() if k != server_name]}",
        "=================================================================",
    ]
    return "\n".join(preview_lines)


def apply_json_merge(
    target_file: Path,
    server_name: str,
    server_payload: Dict[str, Any],
    dry_run: bool = False,
) -> Tuple[Dict[str, Any], str, Optional[Path]]:
    """Merge server entry into target config file with validation, preview, backup, and atomic write.

    Args:
        target_file: Destination config file.
        server_name: Key name for the MCP server.
        server_payload: Client-specific server configuration dictionary.
        dry_run: If True, returns preview without modifying filesystem.

    Returns:
        (merged_dict, preview_text, backup_path_or_none)

    Raises:
        ValueError: If existing config contains malformed JSON.
        PermissionError: If unable to read/write target file or directory.
    """
    target = target_file.resolve()
    existing = load_json_safe(target)
    merged = merge_server_entry(existing, server_name, server_payload)
    preview = generate_diff_preview(existing, merged, server_name)

    if dry_run:
        return merged, preview, None

    # Apply changes: create backup if file exists
    backup_path: Optional[Path] = None
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_file():
        backup_path = create_backup(target)

    # Atomic write: write to temp file in the same directory, then rename
    formatted_json = json.dumps(merged, indent=2) + "\n"

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            delete=False,
            suffix=".tmp",
        ) as tmp:
            tmp.write(formatted_json)
            tmp_path = Path(tmp.name)

        # Atomic replacement
        tmp_path.replace(target)
    except PermissionError as exc:
        raise PermissionError(
            f"Permission denied writing updated configuration to '{target}': {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to write configuration atomically to '{target}': {exc}") from exc

    return merged, preview, backup_path
