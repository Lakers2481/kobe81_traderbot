#!/usr/bin/env python3
"""
snapshot.py - Create and manage state snapshots for the Kobe trading system.

Usage:
    python snapshot.py --create                    # Create snapshot with auto-generated name
    python snapshot.py --create --name my_backup   # Create snapshot with custom name
    python snapshot.py --list                      # List existing snapshots
    python snapshot.py --restore snapshot_name     # Restore from snapshot
    python snapshot.py --delete snapshot_name      # Delete a snapshot
    python snapshot.py --dotenv PATH               # Load environment from .env file
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.env_loader import load_env

SNAPSHOTS_DIR = ROOT / "snapshots"

# Directories and files to include in snapshots
SNAPSHOT_INCLUDES = {
    "state": {
        "path": ROOT / "state",
        "description": "Runtime state files (positions, orders, hash chain)",
    },
    "configs": {
        "path": ROOT / "configs",
        "description": "Configuration files",
        "exclude_patterns": ["__pycache__", "*.pyc"],
    },
    "universe": {
        "path": ROOT / "data" / "universe",
        "description": "Universe definition files",
        "exclude_patterns": ["__pycache__", "*.pyc"],
    },
}

# Individual files to always include
SNAPSHOT_FILES = [
    ROOT / "config" / "settings.json",
    ROOT / "VERSION.json",
]


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


def generate_snapshot_name() -> str:
    """Generate a timestamped snapshot name."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"snapshot_{ts}"


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except (IOError, OSError):
        return ""


def should_exclude(path: Path, exclude_patterns: List[str]) -> bool:
    """Check if path should be excluded based on patterns."""
    name = path.name
    for pattern in exclude_patterns:
        if pattern.startswith("*"):
            if name.endswith(pattern[1:]):
                return True
        elif pattern in str(path):
            return True
    return False


def create_snapshot(name: Optional[str] = None) -> Optional[Path]:
    """Create a full state snapshot."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_name = name or generate_snapshot_name()
    snapshot_path = SNAPSHOTS_DIR / f"{snapshot_name}.zip"

    if snapshot_path.exists():
        print(f"[ERROR] Snapshot already exists: {snapshot_path.name}")
        return None

    print("=" * 60)
    print("  CREATING SNAPSHOT")
    print("=" * 60)
    print(f"Name: {snapshot_name}")
    print(f"Path: {snapshot_path}")

    manifest: Dict[str, Any] = {
        "name": snapshot_name,
        "created_at": datetime.utcnow().isoformat(),
        "kobe_root": str(ROOT),
        "contents": {},
        "file_hashes": {},
    }

    files_added = 0
    total_size = 0

    try:
        with zipfile.ZipFile(snapshot_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add directories
            for section_name, config in SNAPSHOT_INCLUDES.items():
                section_path = config["path"]
                exclude_patterns = config.get("exclude_patterns", [])

                print(f"\n--- {section_name}: {config['description']} ---")

                if not section_path.exists():
                    print(f"  [SKIP] Directory not found: {section_path}")
                    manifest["contents"][section_name] = {"status": "not_found"}
                    continue

                section_files = []
                for file_path in section_path.rglob("*"):
                    if file_path.is_file():
                        if should_exclude(file_path, exclude_patterns):
                            continue

                        rel_path = file_path.relative_to(ROOT)
                        arc_name = str(rel_path).replace("\\", "/")

                        try:
                            zf.write(file_path, arc_name)
                            file_hash = compute_file_hash(file_path)
                            file_size = file_path.stat().st_size

                            section_files.append(arc_name)
                            manifest["file_hashes"][arc_name] = file_hash

                            print(f"  [ADD] {arc_name} ({format_size(file_size)})")
                            files_added += 1
                            total_size += file_size
                        except (IOError, OSError) as e:
                            print(f"  [ERROR] Failed to add {arc_name}: {e}")

                manifest["contents"][section_name] = {
                    "status": "included",
                    "files": section_files,
                }

            # Add individual files
            print("\n--- Individual Files ---")
            for file_path in SNAPSHOT_FILES:
                if file_path.exists():
                    rel_path = file_path.relative_to(ROOT)
                    arc_name = str(rel_path).replace("\\", "/")

                    try:
                        zf.write(file_path, arc_name)
                        file_hash = compute_file_hash(file_path)
                        file_size = file_path.stat().st_size

                        manifest["file_hashes"][arc_name] = file_hash

                        print(f"  [ADD] {arc_name} ({format_size(file_size)})")
                        files_added += 1
                        total_size += file_size
                    except (IOError, OSError) as e:
                        print(f"  [ERROR] Failed to add {arc_name}: {e}")

            # Add manifest
            manifest["total_files"] = files_added
            manifest["total_size"] = total_size
            manifest_json = json.dumps(manifest, indent=2)
            zf.writestr("MANIFEST.json", manifest_json)

        # Verify the archive
        snapshot_size = snapshot_path.stat().st_size
        print("\n" + "=" * 60)
        print("  SNAPSHOT CREATED")
        print("=" * 60)
        print(f"  Name: {snapshot_name}")
        print(f"  Files: {files_added}")
        print(f"  Original size: {format_size(total_size)}")
        print(f"  Archive size: {format_size(snapshot_size)}")
        print(f"  Compression: {(1 - snapshot_size/total_size)*100:.1f}%" if total_size > 0 else "  N/A")
        print(f"  Location: {snapshot_path}")

        return snapshot_path

    except Exception as e:
        print(f"\n[ERROR] Failed to create snapshot: {e}")
        if snapshot_path.exists():
            snapshot_path.unlink()
        return None


def list_snapshots() -> List[Dict[str, Any]]:
    """List all existing snapshots."""
    print("=" * 60)
    print("  AVAILABLE SNAPSHOTS")
    print("=" * 60)

    if not SNAPSHOTS_DIR.exists():
        print("\n  [INFO] No snapshots directory found")
        return []

    snapshots = []
    for zip_path in sorted(SNAPSHOTS_DIR.glob("*.zip"), reverse=True):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if "MANIFEST.json" in zf.namelist():
                    manifest = json.loads(zf.read("MANIFEST.json"))
                else:
                    manifest = {"name": zip_path.stem, "created_at": "unknown"}

            file_size = zip_path.stat().st_size
            snapshots.append({
                "name": zip_path.stem,
                "path": zip_path,
                "size": file_size,
                "created_at": manifest.get("created_at", "unknown"),
                "total_files": manifest.get("total_files", "?"),
            })
        except (zipfile.BadZipFile, json.JSONDecodeError, OSError) as e:
            print(f"  [WARN] Invalid snapshot: {zip_path.name} ({e})")

    if not snapshots:
        print("\n  [INFO] No snapshots found")
        return []

    print(f"\n  {'Name':<30} {'Created':<25} {'Files':>8} {'Size':>12}")
    print("  " + "-" * 78)

    for snap in snapshots:
        created = snap["created_at"]
        if created != "unknown":
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                created = dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        print(f"  {snap['name']:<30} {created:<25} {snap['total_files']:>8} {format_size(snap['size']):>12}")

    print(f"\n  Total: {len(snapshots)} snapshot(s)")
    return snapshots


def restore_snapshot(name: str, force: bool = False) -> bool:
    """Restore from a snapshot."""
    snapshot_path = SNAPSHOTS_DIR / f"{name}.zip"

    if not snapshot_path.exists():
        # Try without .zip extension
        snapshot_path = SNAPSHOTS_DIR / name
        if not snapshot_path.exists():
            print(f"[ERROR] Snapshot not found: {name}")
            return False

    print("=" * 60)
    print("  RESTORING SNAPSHOT")
    print("=" * 60)
    print(f"Snapshot: {snapshot_path.name}")

    try:
        with zipfile.ZipFile(snapshot_path, "r") as zf:
            # Read manifest
            if "MANIFEST.json" in zf.namelist():
                manifest = json.loads(zf.read("MANIFEST.json"))
                print(f"Created: {manifest.get('created_at', 'unknown')}")
                print(f"Files: {manifest.get('total_files', '?')}")
            else:
                manifest = {}

            # List what will be restored
            print("\n--- Files to Restore ---")
            for name_in_zip in zf.namelist():
                if name_in_zip == "MANIFEST.json":
                    continue
                print(f"  {name_in_zip}")

            if not force:
                print("\n[WARNING] This will overwrite existing files!")
                try:
                    response = input("Type 'yes' to confirm: ")
                    if response.lower() != "yes":
                        print("Aborted.")
                        return False
                except (EOFError, KeyboardInterrupt):
                    print("\nAborted.")
                    return False

            # Extract files
            print("\n--- Extracting ---")
            for name_in_zip in zf.namelist():
                if name_in_zip == "MANIFEST.json":
                    continue

                target_path = ROOT / name_in_zip
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract file
                with zf.open(name_in_zip) as src:
                    with open(target_path, "wb") as dst:
                        dst.write(src.read())

                print(f"  [RESTORED] {name_in_zip}")

            # Verify hashes if available
            if "file_hashes" in manifest:
                print("\n--- Verifying ---")
                errors = 0
                for file_name, expected_hash in manifest["file_hashes"].items():
                    file_path = ROOT / file_name
                    if file_path.exists():
                        actual_hash = compute_file_hash(file_path)
                        if actual_hash == expected_hash:
                            print(f"  [OK] {file_name}")
                        else:
                            print(f"  [MISMATCH] {file_name}")
                            errors += 1
                    else:
                        print(f"  [MISSING] {file_name}")
                        errors += 1

                if errors > 0:
                    print(f"\n[WARNING] {errors} verification error(s)")

        print("\n" + "=" * 60)
        print("  RESTORE COMPLETE")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to restore snapshot: {e}")
        return False


def delete_snapshot(name: str) -> bool:
    """Delete a snapshot."""
    snapshot_path = SNAPSHOTS_DIR / f"{name}.zip"

    if not snapshot_path.exists():
        # Try without .zip extension
        snapshot_path = SNAPSHOTS_DIR / name
        if not snapshot_path.exists():
            print(f"[ERROR] Snapshot not found: {name}")
            return False

    print(f"Deleting snapshot: {snapshot_path.name}")
    try:
        response = input("Type 'yes' to confirm: ")
        if response.lower() != "yes":
            print("Aborted.")
            return False
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return False

    try:
        snapshot_path.unlink()
        print(f"[DELETED] {snapshot_path.name}")
        return True
    except OSError as e:
        print(f"[ERROR] Failed to delete: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Kobe Trading System - State Snapshot Manager"
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file",
    )
    ap.add_argument(
        "--create",
        action="store_true",
        help="Create a new snapshot",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List existing snapshots",
    )
    ap.add_argument(
        "--restore",
        type=str,
        metavar="NAME",
        help="Restore from a snapshot",
    )
    ap.add_argument(
        "--delete",
        type=str,
        metavar="NAME",
        help="Delete a snapshot",
    )
    ap.add_argument(
        "--name",
        type=str,
        help="Custom name for snapshot (used with --create)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts",
    )
    args = ap.parse_args()

    # Load environment if specified
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # Execute requested action
    if args.create:
        create_snapshot(args.name)
    elif args.list:
        list_snapshots()
    elif args.restore:
        restore_snapshot(args.restore, args.force)
    elif args.delete:
        delete_snapshot(args.delete)
    else:
        # Default: list snapshots
        list_snapshots()
        print("\nUsage:")
        print("  --create [--name NAME]  Create a new snapshot")
        print("  --list                  List existing snapshots")
        print("  --restore NAME          Restore from a snapshot")
        print("  --delete NAME           Delete a snapshot")


if __name__ == "__main__":
    main()
