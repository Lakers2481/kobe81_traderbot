#!/usr/bin/env python3
"""
cleanup.py - Cleanup old files and free disk space in the Kobe trading system.

Usage:
    python cleanup.py                  # Show what would be cleaned (dry run)
    python cleanup.py --force          # Actually delete files
    python cleanup.py --area logs      # Only clean logs
    python cleanup.py --area cache     # Only clean cache
    python cleanup.py --area pycache   # Only clean __pycache__
    python cleanup.py --dry-run        # Preview without deleting (default)
    python cleanup.py --dotenv PATH    # Load environment from .env file
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.env_loader import load_env

# Cleanup configuration
CLEANUP_CONFIG = {
    "logs": {
        "path": ROOT / "logs",
        "max_age_days": 30,
        "patterns": ["*.log", "*.jsonl"],
    },
    "cache": {
        "path": ROOT / "data" / "cache",
        "max_age_days": 30,
        "patterns": ["*"],
    },
    "pycache": {
        "paths": [
            ROOT,
            ROOT / "scripts",
            ROOT / "core",
            ROOT / "strategies",
            ROOT / "backtest",
            ROOT / "execution",
            ROOT / "data",
            ROOT / "configs",
            ROOT / "oms",
            ROOT / "risk",
            ROOT / "monitor",
        ],
        "dir_name": "__pycache__",
    },
    "temp": {
        "path": ROOT / "temp",
        "max_age_days": 7,
        "patterns": ["*"],
    },
}


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_file_age_days(path: Path) -> float:
    """Get file age in days."""
    try:
        mtime = path.stat().st_mtime
        age = datetime.now() - datetime.fromtimestamp(mtime)
        return age.total_seconds() / (24 * 3600)
    except OSError:
        return 0


def get_file_size(path: Path) -> int:
    """Get file size in bytes."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def get_dir_size(path: Path) -> int:
    """Get total directory size in bytes."""
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += get_file_size(item)
    except (OSError, PermissionError):
        pass
    return total


def find_old_files(
    directory: Path, patterns: List[str], max_age_days: int
) -> List[Tuple[Path, int, float]]:
    """Find files older than max_age_days matching patterns."""
    old_files: List[Tuple[Path, int, float]] = []

    if not directory.exists():
        return old_files

    for pattern in patterns:
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                age = get_file_age_days(file_path)
                if age > max_age_days:
                    size = get_file_size(file_path)
                    old_files.append((file_path, size, age))

    return old_files


def find_pycache_dirs(paths: List[Path]) -> List[Tuple[Path, int]]:
    """Find all __pycache__ directories."""
    pycache_dirs: List[Tuple[Path, int]] = []

    for base_path in paths:
        if not base_path.exists():
            continue
        # Direct __pycache__ in this directory
        direct = base_path / "__pycache__"
        if direct.exists() and direct.is_dir():
            size = get_dir_size(direct)
            pycache_dirs.append((direct, size))
        # Also search subdirectories
        for pycache in base_path.rglob("__pycache__"):
            if pycache.is_dir() and pycache not in [p[0] for p in pycache_dirs]:
                size = get_dir_size(pycache)
                pycache_dirs.append((pycache, size))

    return pycache_dirs


def cleanup_logs(dry_run: bool = True) -> Tuple[int, int]:
    """Clean up old log files."""
    config = CLEANUP_CONFIG["logs"]
    log_dir = config["path"]
    max_age = config["max_age_days"]
    patterns = config["patterns"]

    print(f"\n--- Cleaning Logs (>{max_age} days old) ---")
    print(f"Directory: {log_dir}")

    if not log_dir.exists():
        print("  [SKIP] Log directory does not exist")
        return 0, 0

    old_files = find_old_files(log_dir, patterns, max_age)

    if not old_files:
        print("  [OK] No old log files found")
        return 0, 0

    total_size = 0
    deleted_count = 0

    for file_path, size, age in old_files:
        rel_path = file_path.relative_to(ROOT) if file_path.is_relative_to(ROOT) else file_path
        if dry_run:
            print(f"  [DRY-RUN] Would delete: {rel_path} ({format_size(size)}, {age:.1f} days old)")
        else:
            try:
                file_path.unlink()
                print(f"  [DELETED] {rel_path} ({format_size(size)})")
                deleted_count += 1
            except OSError as e:
                print(f"  [ERROR] Failed to delete {rel_path}: {e}")
                continue
        total_size += size

    return deleted_count if not dry_run else len(old_files), total_size


def cleanup_cache(dry_run: bool = True) -> Tuple[int, int]:
    """Clean up stale cache files."""
    config = CLEANUP_CONFIG["cache"]
    cache_dir = config["path"]
    max_age = config["max_age_days"]
    patterns = config["patterns"]

    print(f"\n--- Cleaning Cache (>{max_age} days unused) ---")
    print(f"Directory: {cache_dir}")

    if not cache_dir.exists():
        print("  [SKIP] Cache directory does not exist")
        return 0, 0

    old_files = find_old_files(cache_dir, patterns, max_age)

    if not old_files:
        print("  [OK] No stale cache files found")
        return 0, 0

    total_size = 0
    deleted_count = 0

    for file_path, size, age in old_files:
        rel_path = file_path.relative_to(ROOT) if file_path.is_relative_to(ROOT) else file_path
        if dry_run:
            print(f"  [DRY-RUN] Would delete: {rel_path} ({format_size(size)}, {age:.1f} days old)")
        else:
            try:
                file_path.unlink()
                print(f"  [DELETED] {rel_path} ({format_size(size)})")
                deleted_count += 1
            except OSError as e:
                print(f"  [ERROR] Failed to delete {rel_path}: {e}")
                continue
        total_size += size

    return deleted_count if not dry_run else len(old_files), total_size


def cleanup_pycache(dry_run: bool = True) -> Tuple[int, int]:
    """Clean up __pycache__ directories."""
    config = CLEANUP_CONFIG["pycache"]
    paths = config["paths"]

    print("\n--- Cleaning __pycache__ Directories ---")

    pycache_dirs = find_pycache_dirs(paths)

    if not pycache_dirs:
        print("  [OK] No __pycache__ directories found")
        return 0, 0

    total_size = 0
    deleted_count = 0

    for dir_path, size in pycache_dirs:
        rel_path = dir_path.relative_to(ROOT) if dir_path.is_relative_to(ROOT) else dir_path
        if dry_run:
            print(f"  [DRY-RUN] Would remove: {rel_path} ({format_size(size)})")
        else:
            try:
                shutil.rmtree(dir_path)
                print(f"  [REMOVED] {rel_path} ({format_size(size)})")
                deleted_count += 1
            except OSError as e:
                print(f"  [ERROR] Failed to remove {rel_path}: {e}")
                continue
        total_size += size

    return deleted_count if not dry_run else len(pycache_dirs), total_size


def cleanup_temp(dry_run: bool = True) -> Tuple[int, int]:
    """Clean up old temp files."""
    config = CLEANUP_CONFIG["temp"]
    temp_dir = config["path"]
    max_age = config["max_age_days"]
    patterns = config["patterns"]

    print(f"\n--- Cleaning Temp Files (>{max_age} days old) ---")
    print(f"Directory: {temp_dir}")

    if not temp_dir.exists():
        print("  [SKIP] Temp directory does not exist")
        return 0, 0

    old_files = find_old_files(temp_dir, patterns, max_age)

    if not old_files:
        print("  [OK] No old temp files found")
        return 0, 0

    total_size = 0
    deleted_count = 0

    for file_path, size, age in old_files:
        rel_path = file_path.relative_to(ROOT) if file_path.is_relative_to(ROOT) else file_path
        if dry_run:
            print(f"  [DRY-RUN] Would delete: {rel_path} ({format_size(size)}, {age:.1f} days old)")
        else:
            try:
                file_path.unlink()
                print(f"  [DELETED] {rel_path} ({format_size(size)})")
                deleted_count += 1
            except OSError as e:
                print(f"  [ERROR] Failed to delete {rel_path}: {e}")
                continue
        total_size += size

    return deleted_count if not dry_run else len(old_files), total_size


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Kobe Trading System - Cleanup Utility"
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview cleanup without deleting (default)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Actually delete files (disables dry-run)",
    )
    ap.add_argument(
        "--area",
        type=str,
        choices=["logs", "cache", "pycache", "temp", "all"],
        default="all",
        help="Specific area to clean (default: all)",
    )
    ap.add_argument(
        "--max-age",
        type=int,
        default=None,
        help="Override max age in days for cleanup",
    )
    args = ap.parse_args()

    # Load environment if specified
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # Determine if this is a dry run
    dry_run = not args.force

    # Override max age if specified
    if args.max_age is not None:
        for config in CLEANUP_CONFIG.values():
            if "max_age_days" in config:
                config["max_age_days"] = args.max_age

    print("=" * 60)
    print("  KOBE TRADING SYSTEM - CLEANUP UTILITY")
    print("=" * 60)
    print(f"Mode: {'DRY RUN (preview only)' if dry_run else 'FORCE (deleting files)'}")
    print(f"Area: {args.area}")

    if not dry_run:
        print("\n[WARNING] This will permanently delete files!")
        try:
            response = input("Type 'yes' to confirm: ")
            if response.lower() != "yes":
                print("Aborted.")
                return
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return

    total_items = 0
    total_size = 0

    # Run cleanup based on area selection
    if args.area in ("logs", "all"):
        count, size = cleanup_logs(dry_run)
        total_items += count
        total_size += size

    if args.area in ("cache", "all"):
        count, size = cleanup_cache(dry_run)
        total_items += count
        total_size += size

    if args.area in ("pycache", "all"):
        count, size = cleanup_pycache(dry_run)
        total_items += count
        total_size += size

    if args.area in ("temp", "all"):
        count, size = cleanup_temp(dry_run)
        total_items += count
        total_size += size

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    action = "Would free" if dry_run else "Freed"
    print(f"  Items: {total_items}")
    print(f"  Space {action}: {format_size(total_size)}")

    if dry_run and total_items > 0:
        print("\n  [INFO] Run with --force to actually delete files")


if __name__ == "__main__":
    main()
