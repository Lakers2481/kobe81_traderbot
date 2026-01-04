#!/usr/bin/env python3
"""
version.py - Show Kobe trading system version information.

Usage:
    python version.py                  # Show current version info
    python version.py --changelog      # Show recent changes
    python version.py --check-updates  # Check for updates (placeholder)
    python version.py --dotenv PATH    # Load environment from .env file
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env

VERSION_FILE = ROOT / "VERSION.json"

# Default version info if VERSION.json doesn't exist
DEFAULT_VERSION = {
    "version": "1.0.0",
    "name": "Kobe Trading System",
    "codename": "Mamba",
    "release_date": "2024-01-01",
    "components": {
        "backtest_engine": "1.0.0",
        "walk_forward": "1.0.0",
        "polygon_eod": "1.0.0",
        "broker_alpaca": "1.0.0",
        "risk_policy": "1.0.0",
        "hash_chain": "1.0.0",
    },
    "changelog": [
        {"version": "1.0.0", "date": "2024-01-01", "changes": ["Initial release"]},
    ],
}


def load_version_info() -> Dict[str, Any]:
    """Load version info from VERSION.json or generate default."""
    if VERSION_FILE.exists():
        try:
            return json.loads(VERSION_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARN] Failed to load VERSION.json: {e}")
            return DEFAULT_VERSION
    else:
        # Generate version info based on file timestamps
        version_info = DEFAULT_VERSION.copy()
        version_info["generated"] = True
        version_info["generated_at"] = datetime.utcnow().isoformat()
        return version_info


def get_python_version() -> str:
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_dependency_versions() -> Dict[str, str]:
    """Get versions of key dependencies."""
    deps: Dict[str, str] = {}

    # Check common dependencies
    packages = [
        "pandas",
        "numpy",
        "requests",
        "alpaca-trade-api",
        "polygon",
    ]

    for pkg in packages:
        try:
            # Try importlib.metadata first (Python 3.8+)
            from importlib.metadata import version as get_version
            deps[pkg] = get_version(pkg)
        except Exception:
            try:
                # Fallback: try importing and checking __version__
                mod = __import__(pkg.replace("-", "_").replace(".", "_"))
                deps[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                deps[pkg] = "not installed"

    return deps


def get_last_update_timestamp() -> Optional[str]:
    """Get the most recent file modification time in the project."""
    latest_time: Optional[float] = None
    latest_file: Optional[Path] = None

    # Check key directories for recent modifications
    dirs_to_check = [
        ROOT / "scripts",
        ROOT / "core",
        ROOT / "strategies",
        ROOT / "execution",
        ROOT / "backtest",
    ]

    for check_dir in dirs_to_check:
        if not check_dir.exists():
            continue
        for py_file in check_dir.glob("*.py"):
            try:
                mtime = py_file.stat().st_mtime
                if latest_time is None or mtime > latest_time:
                    latest_time = mtime
                    latest_file = py_file
            except OSError:
                continue

    if latest_time is not None:
        ts = datetime.fromtimestamp(latest_time)
        file_name = latest_file.name if latest_file else "unknown"
        return f"{ts.isoformat()} ({file_name})"
    return None


def show_version(version_info: Dict[str, Any]) -> None:
    """Display version information."""
    print("=" * 60)
    print(f"  {version_info.get('name', 'Kobe Trading System')}")
    print(f"  Version: {version_info.get('version', 'unknown')}")
    if version_info.get("codename"):
        print(f"  Codename: {version_info['codename']}")
    if version_info.get("release_date"):
        print(f"  Release Date: {version_info['release_date']}")
    print("=" * 60)

    if version_info.get("generated"):
        print("\n[INFO] VERSION.json not found - using generated info")

    # Component versions
    print("\n--- Component Versions ---")
    components = version_info.get("components", {})
    for comp, ver in components.items():
        print(f"  {comp:20s} : {ver}")

    # Python version
    print("\n--- Runtime ---")
    print(f"  Python Version     : {get_python_version()}")
    print(f"  Platform           : {platform.system()} {platform.release()}")
    print(f"  Architecture       : {platform.machine()}")

    # Dependency versions
    print("\n--- Dependencies ---")
    deps = get_dependency_versions()
    for pkg, ver in deps.items():
        status = ver if ver != "not installed" else "[NOT INSTALLED]"
        print(f"  {pkg:20s} : {status}")

    # Last update
    last_update = get_last_update_timestamp()
    if last_update:
        print("\n--- Last Updated ---")
        print(f"  {last_update}")


def show_changelog(version_info: Dict[str, Any], limit: int = 10) -> None:
    """Display changelog."""
    print("=" * 60)
    print("  CHANGELOG")
    print("=" * 60)

    changelog = version_info.get("changelog", [])
    if not changelog:
        print("\n  No changelog entries found.")
        return

    for i, entry in enumerate(changelog[:limit]):
        if i > 0:
            print("-" * 40)
        print(f"\n  Version: {entry.get('version', 'unknown')}")
        print(f"  Date:    {entry.get('date', 'unknown')}")
        print("  Changes:")
        for change in entry.get("changes", []):
            print(f"    - {change}")

    if len(changelog) > limit:
        print(f"\n  ... and {len(changelog) - limit} more entries")


def check_updates(version_info: Dict[str, Any]) -> None:
    """Check for available updates (placeholder for future implementation)."""
    print("=" * 60)
    print("  UPDATE CHECK")
    print("=" * 60)

    current_version = version_info.get("version", "unknown")
    print(f"\n  Current Version: {current_version}")
    print("\n  [INFO] Update checking is not yet implemented.")
    print("  [INFO] Please check the repository manually for updates.")
    print("\n  Suggested actions:")
    print("    1. Check your git remote for new commits")
    print("    2. Review release notes in the repository")
    print("    3. Run 'pip list --outdated' to check dependencies")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Kobe Trading System - Version Information"
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file",
    )
    ap.add_argument(
        "--changelog",
        action="store_true",
        help="Show changelog/release notes",
    )
    ap.add_argument(
        "--check-updates",
        action="store_true",
        help="Check for available updates",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output version info as JSON",
    )
    args = ap.parse_args()

    # Load environment if specified
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # Load version information
    version_info = load_version_info()

    if args.json:
        # Add runtime info for JSON output
        version_info["runtime"] = {
            "python_version": get_python_version(),
            "platform": platform.system(),
            "architecture": platform.machine(),
            "dependencies": get_dependency_versions(),
            "last_update": get_last_update_timestamp(),
        }
        print(json.dumps(version_info, indent=2))
    elif args.changelog:
        show_changelog(version_info)
    elif args.check_updates:
        check_updates(version_info)
    else:
        show_version(version_info)


if __name__ == "__main__":
    main()
