"""Unified Setup and Diagnostics CLI for MCP OpenCV Engine.

Provides:
  --help                         Show this help message and exit
  --doctor [--json]              Run read-only system and configuration diagnostics
  --configure-runtime <mode>     Configure runtime execution mode (auto, native, docker)
  --configure-client <client>    Register gateway with an MCP client (antigravity, cursor, claude)
  --setup [--yes]                Guided setup workflow (runtime, directories, client)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

# Ensure project root is in sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.client_config.base import ConfigScope
from core.client_config.registry import configure_client, get_adapter, list_adapters
from core.config.loader import get_config, get_project_root
from core.config.schema import AppConfig, MCPTransport, RuntimeMode
from core.dependencies.checker import DependencyChecker, Severity, check_docker_runtime
from core.environment.detector import EnvironmentDetector
from setup.diagnostics import SystemDiagnostics


def cmd_doctor(
    json_output: bool = False,
    project_root: Optional[Union[str, Path]] = None,
    config: Optional[AppConfig] = None,
    command_runner: Optional[Callable[..., tuple[int, str, str]]] = None,
) -> int:
    """Run read-only diagnostics and report findings with stable exit code."""
    diag = SystemDiagnostics(
        config=config,
        project_root=project_root,
        command_runner=command_runner,
    )
    report = diag.run()

    if json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.to_text())

    return report.exit_code


def cmd_configure_runtime(
    runtime_mode_str: str,
    project_root: Optional[Union[str, Path]] = None,
) -> int:
    """Validate and configure runtime mode in .env."""
    root = Path(project_root).resolve() if project_root else get_project_root()
    clean_mode = runtime_mode_str.lower().strip()

    try:
        mode_enum = RuntimeMode(clean_mode)
    except ValueError:
        valid_modes = [m.value for m in RuntimeMode]
        print(
            f"Error: Invalid runtime mode '{runtime_mode_str}'. Must be one of: {valid_modes}",
            file=sys.stderr,
        )
        return 1

    env_path = root / ".env"
    existing_lines: List[str] = []
    if env_path.is_file():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    updated = False
    new_lines: List[str] = []
    for line in existing_lines:
        if line.strip().startswith("RUNTIME_MODE="):
            new_lines.append(f"RUNTIME_MODE={mode_enum.value}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"RUNTIME_MODE={mode_enum.value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"Successfully configured runtime mode: {mode_enum.value} in {env_path}")
    return 0


def cmd_configure_client(
    client_id: str,
    transport: str = "streamable-http",
    scope: str = "workspace",
    dry_run: bool = True,
    project_root: Optional[Union[str, Path]] = None,
    config: Optional[AppConfig] = None,
) -> int:
    """Register the engine with a supported MCP client."""
    root = Path(project_root).resolve() if project_root else get_project_root()
    try:
        cfg_path, backup_path, preview = configure_client(
            client_id=client_id,
            transport=transport,
            scope=scope,
            dry_run=dry_run,
            project_root=root,
            config=config,
        )
    except Exception as exc:
        print(f"Configuration failed: {exc}", file=sys.stderr)
        return 1

    print(preview)
    if dry_run:
        print("\n[PREVIEW MODE] No files were modified. Run with --yes to apply changes.")
    else:
        print(f"\n[APPLIED] Successfully wrote configuration to: {cfg_path}")
        if backup_path:
            print(f"[BACKUP] Backup created at: {backup_path}")

    return 0


def cmd_guided_setup(
    yes: bool = False,
    project_root: Optional[Union[str, Path]] = None,
    input_func: Optional[Callable[[str], str]] = None,
    config: Optional[AppConfig] = None,
    command_runner: Optional[Callable[..., tuple[int, str, str]]] = None,
) -> int:
    """Guided, non-destructive configuration workflow with explicit confirmation."""
    root = Path(project_root).resolve() if project_root else get_project_root()
    prompt = input_func or input

    print("=========================================================")
    print("      MCP OpenCV Engine — Guided Setup & Configuration   ")
    print("=========================================================\n")

    # Step 1: Detect environment and capabilities
    print("[1/5] Detecting environment and capabilities...")
    detector = (
        EnvironmentDetector(cmd_runner=command_runner)
        if command_runner
        else EnvironmentDetector()
    )
    profile = detector.detect()
    print(f"  • Operating System: {profile.os_family} ({profile.cpu_arch})")
    print(f"  • Python Version:   {profile.python_version}")
    print(f"  • Hardware Backend: {profile.gpu_backend.upper()}")
    docker_ready = profile.has_docker and profile.docker_compose_available and profile.diagnostics.get("docker_daemon_running", False)
    print(f"  • Docker Active:    {'Yes' if docker_ready else 'No'}")
    print(f"  • Recommended Mode: {profile.recommended_runtime}\n")

    # Step 2: Runtime mode selection
    selected_mode = profile.recommended_runtime
    if not yes:
        user_choice = prompt(
            f"Choose runtime mode [auto/docker/native] (default: {selected_mode}): "
        ).strip().lower()
        if user_choice in ("auto", "docker", "native"):
            selected_mode = user_choice
        elif user_choice:
            print(f"Invalid mode '{user_choice}'. Keeping default: {selected_mode}")

    print(f"Selected Runtime Mode: {selected_mode.upper()}")

    # Step 3: Explain prerequisites and missing dependencies
    print("\n[2/5] Checking prerequisites for selected mode...")
    checker = DependencyChecker(env_profile=profile, cmd_runner=command_runner)

    if selected_mode == "docker":
        docker_res = check_docker_runtime(env_profile=profile, required=True)
        if not docker_res.passed:
            print(f"  ⚠ Docker prerequisite warning: {docker_res.message}")
            if docker_res.install_hint:
                print(f"    Remediation: {docker_res.install_hint}")
        else:
            print("  ✓ Docker and Docker Compose prerequisites met.")
    else:
        # Native or Auto mode: check Python packages
        gw_pkgs = checker.check_service_packages("gateway")
        missing_gw = [p for p in gw_pkgs if not p.passed and p.required]
        if missing_gw:
            print(f"  ⚠ Missing gateway dependencies: {', '.join(p.name for p in missing_gw)}")
            print("    Remediation: pip install -r requirements/base-gateway.txt")
        else:
            print("  ✓ Gateway dependencies verified.")

    # Step 4: Storage directories
    print("\n[3/5] Storage boundaries...")
    inputs_dir = root / "data" / "inputs"
    outputs_dir = root / "data" / "outputs"
    print(f"  • Input directory:  {inputs_dir}")
    print(f"  • Output directory: {outputs_dir}")

    # Step 5: MCP Client selection & preview
    print("\n[4/5] MCP Client Registration Preview...")
    client_choice = "antigravity"
    if not yes:
        avail_clients = [a.client_id for a in list_adapters(project_root=root, config=config)]
        print(f"Available clients: {', '.join(avail_clients)}")
        c_input = prompt(f"Select client to configure (default: {client_choice}): ").strip().lower()
        if c_input in avail_clients:
            client_choice = c_input

    transport = "stdio" if client_choice == "claude" else ("streamable-http" if selected_mode == "docker" else "stdio")

    # Generate preview without writing
    try:
        cfg_path, _, preview = configure_client(
            client_id=client_choice,
            transport=transport,
            scope="workspace" if client_choice != "claude" else "global",
            dry_run=True,
            project_root=root,
            config=config,
        )
        print(f"Target file: {cfg_path}")
        print("Proposed MCP Configuration Diff:")
        print(preview)
    except Exception as exc:
        print(f"Preview error: {exc}", file=sys.stderr)
        return 1

    # Step 6: Explicit approval before mutation
    print("\n[5/5] Confirmation")
    print("Summary of proposed changes:")
    print(f"  1. Set RUNTIME_MODE={selected_mode} in {root / '.env'}")
    print(f"  2. Create storage directories if absent: {inputs_dir}, {outputs_dir}")
    print(f"  3. Register MCP gateway into {client_choice} ({cfg_path})")

    if not yes:
        confirm = prompt("\nApply these changes now? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("\n[CANCELLED] Setup aborted by user. No files or directories were modified.")
            return 0

    # Apply changes
    cmd_configure_runtime(selected_mode, project_root=root)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    applied_cfg, backup_path, _ = configure_client(
        client_id=client_choice,
        transport=transport,
        scope="workspace" if client_choice != "claude" else "global",
        dry_run=False,
        project_root=root,
        config=config,
    )
    print(f"\n[APPLIED] Configuration successfully saved to: {applied_cfg}")
    if backup_path:
        print(f"[BACKUP] Existing configuration backed up to: {backup_path}")

    # Instructions and next steps
    print("\nSetup Complete!")
    print("---------------------------------------------------------")
    print("Next steps:")
    if selected_mode == "docker":
        print("  1. Start services with Docker:  docker compose up -d")
    else:
        print("  1. Start native microservices:")
        print("     python -m uvicorn opencv_services.object_detection.api:app --port 8001 &")
        print("     python -m uvicorn opencv_services.image_processing.api:app --port 8002 &")
        print("     python -m uvicorn opencv_services.ocr_service.api:app --port 8003 &")
        print("     python -m mcp_gateway.main")
    print("  2. Verify system health anytime: python setup/configure.py --doctor")
    print("  3. Restart or reload your MCP client to access the OpenCV tools.")
    print("=========================================================\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python setup/configure.py",
        description="Unified Setup & Diagnostics CLI for MCP OpenCV Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup/configure.py --doctor
  python setup/configure.py --doctor --json
  python setup/configure.py --setup --yes
  python setup/configure.py --configure-runtime native
  python setup/configure.py --configure-client antigravity --yes
  python setup/configure.py --configure-client cursor --transport stdio --yes
  python setup/configure.py --configure-client claude --dry-run
        """,
    )

    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--doctor",
        action="store_true",
        help="Run read-only system, dependency, and configuration diagnostics",
    )
    action_group.add_argument(
        "--setup",
        action="store_true",
        help="Run the guided setup flow (runtime, storage, client config)",
    )
    action_group.add_argument(
        "--configure-runtime",
        metavar="MODE",
        choices=["auto", "docker", "native"],
        help="Configure runtime execution mode (auto, docker, native)",
    )
    action_group.add_argument(
        "--configure-client",
        metavar="CLIENT",
        choices=["antigravity", "cursor", "claude"],
        help="Register gateway with an MCP client (antigravity, cursor, claude)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output diagnostics in machine-readable JSON format",
    )
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "stdio", "sse"],
        default="streamable-http",
        help="Transport type for client configuration (default: streamable-http)",
    )
    parser.add_argument(
        "--scope",
        choices=["workspace", "global"],
        default="workspace",
        help="Configuration scope for client registration (default: workspace)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview configuration changes without writing to disk",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Apply changes automatically without interactive confirmation prompts",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.doctor:
        return cmd_doctor(json_output=args.json)

    if args.configure_runtime:
        return cmd_configure_runtime(args.configure_runtime)

    if args.configure_client:
        # Dry run unless --yes is explicitly given, or if --dry-run is flagged
        dry_run = args.dry_run or (not args.yes)
        return cmd_configure_client(
            client_id=args.configure_client,
            transport=args.transport,
            scope=args.scope,
            dry_run=dry_run,
        )

    if args.setup:
        return cmd_guided_setup(yes=args.yes)

    # Default action if no arguments provided: print help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
