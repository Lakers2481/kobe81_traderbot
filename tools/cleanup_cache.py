#!/usr/bin/env python3
"""
cleanup_cache.py - Clean Expired Data Cache Files

Removes cache files older than the specified TTL to prevent infinite cache growth.
Default: 24 hours for CSV/parquet files.

Usage:
    python tools/cleanup_cache.py              # Dry run (shows what would be removed)
    python tools/cleanup_cache.py --execute    # Actually remove files
    python tools/cleanup_cache.py --ttl 48     # Custom TTL in hours
    python tools/cleanup_cache.py --verbose    # Show all files (not just expired)
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default cache directories to clean
CACHE_DIRS = [
    PROJECT_ROOT / "data" / "cache",
    PROJECT_ROOT / "cache",
]

# File extensions to clean
CLEAN_EXTENSIONS = {".csv", ".parquet", ".json"}

# Default TTL in hours
DEFAULT_TTL_HOURS = 24


def get_file_age_hours(file_path: Path) -> float:
    """Get age of file in hours."""
    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
    age = datetime.now() - mtime
    return age.total_seconds() / 3600


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def find_expired_files(
    cache_dirs: List[Path],
    ttl_hours: float,
    extensions: set = CLEAN_EXTENSIONS,
) -> List[Tuple[Path, float, int]]:
    """
    Find all expired cache files.

    Returns:
        List of (path, age_hours, size_bytes) tuples
    """
    expired = []

    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue

        for ext in extensions:
            for f in cache_dir.rglob(f"*{ext}"):
                try:
                    age = get_file_age_hours(f)
                    if age > ttl_hours:
                        size = f.stat().st_size
                        expired.append((f, age, size))
                except (OSError, PermissionError):
                    pass

    return expired


def cleanup_cache(
    ttl_hours: float = DEFAULT_TTL_HOURS,
    dry_run: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Clean expired cache files.

    Args:
        ttl_hours: Time-to-live in hours (files older than this are removed)
        dry_run: If True, only report what would be removed
        verbose: If True, show all files, not just expired

    Returns:
        Summary dict with counts and sizes
    """
    expired_files = find_expired_files(CACHE_DIRS, ttl_hours)

    total_size = sum(size for _, _, size in expired_files)
    removed_count = 0
    removed_size = 0
    errors = []

    print("=" * 60)
    print("KOBE CACHE CLEANUP")
    print("=" * 60)
    print(f"TTL: {ttl_hours} hours")
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print(f"Cache dirs: {len(CACHE_DIRS)}")
    print(f"Expired files found: {len(expired_files)}")
    print(f"Total size: {format_size(total_size)}")
    print()

    if not expired_files:
        print("No expired cache files found.")
        return {
            "expired_count": 0,
            "expired_size": 0,
            "removed_count": 0,
            "removed_size": 0,
            "errors": [],
        }

    print("Expired files:")
    for path, age, size in sorted(expired_files, key=lambda x: -x[1]):
        rel_path = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
        action = "Would remove" if dry_run else "Removing"
        print(f"  [{format_size(size):>10}] [{age:>6.1f}h] {rel_path}")

        if not dry_run:
            try:
                path.unlink()
                removed_count += 1
                removed_size += size
            except (OSError, PermissionError) as e:
                errors.append(f"{rel_path}: {e}")

    print()
    if dry_run:
        print(f"DRY RUN: Would remove {len(expired_files)} files ({format_size(total_size)})")
        print("Run with --execute to actually remove files.")
    else:
        print(f"Removed: {removed_count} files ({format_size(removed_size)})")
        if errors:
            print(f"Errors: {len(errors)}")
            for err in errors[:5]:
                print(f"  - {err}")

    return {
        "expired_count": len(expired_files),
        "expired_size": total_size,
        "removed_count": removed_count,
        "removed_size": removed_size,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Clean expired data cache files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/cleanup_cache.py              # Dry run
  python tools/cleanup_cache.py --execute    # Actually remove
  python tools/cleanup_cache.py --ttl 48     # 48-hour TTL
        """,
    )
    parser.add_argument(
        "--ttl",
        type=float,
        default=DEFAULT_TTL_HOURS,
        help=f"Time-to-live in hours (default: {DEFAULT_TTL_HOURS})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually remove files (default is dry run)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    result = cleanup_cache(
        ttl_hours=args.ttl,
        dry_run=not args.execute,
        verbose=args.verbose,
    )

    # Exit with error if there were issues
    if result["errors"]:
        exit(1)
    exit(0)


if __name__ == "__main__":
    main()
