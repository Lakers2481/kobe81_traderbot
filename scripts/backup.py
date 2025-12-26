#!/usr/bin/env python3
"""
backup.py - Backup state, configs, and logs for the Kobe trading system.

Usage:
    python scripts/backup.py --create
    python scripts/backup.py --list
    python scripts/backup.py --restore backup_20240101_120000.zip
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from core.structured_log import jlog

STATE_DIR = ROOT / "state"
CONFIGS_DIR = ROOT / "configs"
LOGS_DIR = ROOT / "logs"
BACKUPS_DIR = ROOT / "backups"

# Directories to backup
BACKUP_DIRS = {
    "state": STATE_DIR,
    "configs": CONFIGS_DIR,
    "logs": LOGS_DIR,
}


def ensure_backups_dir() -> None:
    """Create backups directory if it doesn't exist."""
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)


def generate_backup_name() -> str:
    """Generate timestamped backup filename."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"backup_{ts}.zip"


def get_file_count(directory: Path) -> int:
    """Count files in a directory recursively."""
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.rglob("*") if _.is_file())


def create_backup(
    include_state: bool = True,
    include_configs: bool = True,
    include_logs: bool = True,
    comment: Optional[str] = None
) -> Optional[Path]:
    """Create a timestamped backup of state, configs, and logs."""
    ensure_backups_dir()

    backup_name = generate_backup_name()
    backup_path = BACKUPS_DIR / backup_name

    dirs_to_backup: Dict[str, Path] = {}
    if include_state and STATE_DIR.exists():
        dirs_to_backup["state"] = STATE_DIR
    if include_configs and CONFIGS_DIR.exists():
        dirs_to_backup["configs"] = CONFIGS_DIR
    if include_logs and LOGS_DIR.exists():
        dirs_to_backup["logs"] = LOGS_DIR

    if not dirs_to_backup:
        print("No directories to backup (none exist)")
        return None

    print(f"Creating backup: {backup_name}")
    print(f"Directories to backup: {', '.join(dirs_to_backup.keys())}")

    total_files = 0
    total_size = 0

    try:
        with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add metadata
            metadata = {
                "created": datetime.utcnow().isoformat(),
                "comment": comment,
                "directories": list(dirs_to_backup.keys()),
            }
            zf.writestr("backup_metadata.json", json.dumps(metadata, indent=2))

            for dir_name, dir_path in dirs_to_backup.items():
                print(f"  Adding {dir_name}/...")
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file():
                        # Skip __pycache__ and .pyc files
                        if "__pycache__" in str(file_path) or file_path.suffix == ".pyc":
                            continue

                        arcname = dir_name + "/" + str(file_path.relative_to(dir_path))
                        zf.write(file_path, arcname)
                        total_files += 1
                        total_size += file_path.stat().st_size

        # Get final compressed size
        compressed_size = backup_path.stat().st_size

        print(f"\nBackup created successfully:")
        print(f"  File: {backup_path}")
        print(f"  Files backed up: {total_files}")
        print(f"  Original size: {total_size / 1024:.1f} KB")
        print(f"  Compressed size: {compressed_size / 1024:.1f} KB")
        print(f"  Compression ratio: {(1 - compressed_size / max(total_size, 1)) * 100:.1f}%")

        jlog("backup_created", backup=backup_name, files=total_files, size_bytes=compressed_size)

        return backup_path

    except Exception as e:
        print(f"Error creating backup: {e}")
        jlog("backup_error", error=str(e), level="ERROR")
        if backup_path.exists():
            backup_path.unlink()
        return None


def list_backups() -> List[Dict[str, Any]]:
    """List all existing backups."""
    if not BACKUPS_DIR.exists():
        return []

    backups = []
    for f in sorted(BACKUPS_DIR.glob("backup_*.zip"), reverse=True):
        stat = f.stat()
        info = {
            "name": f.name,
            "path": str(f),
            "size_bytes": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

        # Try to read metadata
        try:
            with zipfile.ZipFile(f, "r") as zf:
                if "backup_metadata.json" in zf.namelist():
                    metadata = json.loads(zf.read("backup_metadata.json"))
                    info["metadata"] = metadata
                info["file_count"] = len([n for n in zf.namelist() if not n.endswith("/")])
        except Exception:
            pass

        backups.append(info)

    return backups


def show_backups() -> None:
    """Display list of existing backups."""
    print("\n=== EXISTING BACKUPS ===")
    print(f"Directory: {BACKUPS_DIR}")

    backups = list_backups()
    if not backups:
        print("No backups found")
        return

    print(f"Total backups: {len(backups)}\n")

    for i, b in enumerate(backups, 1):
        size_kb = b["size_bytes"] / 1024
        size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB"
        print(f"{i}. {b['name']}")
        print(f"   Size: {size_str}, Files: {b.get('file_count', '?')}")
        print(f"   Created: {b['created']}")
        if "metadata" in b and b["metadata"].get("comment"):
            print(f"   Comment: {b['metadata']['comment']}")
        print()


def restore_backup(backup_name: str, confirm: bool = False) -> bool:
    """Restore from a backup file."""
    # Find backup file
    if Path(backup_name).exists():
        backup_path = Path(backup_name)
    else:
        backup_path = BACKUPS_DIR / backup_name
        if not backup_path.exists():
            # Try adding .zip extension
            backup_path = BACKUPS_DIR / (backup_name + ".zip")

    if not backup_path.exists():
        print(f"Backup not found: {backup_name}")
        print(f"Searched: {BACKUPS_DIR}")
        return False

    print(f"\n=== RESTORE BACKUP ===")
    print(f"Backup: {backup_path.name}")

    # Show backup contents
    try:
        with zipfile.ZipFile(backup_path, "r") as zf:
            files = zf.namelist()
            print(f"Files in backup: {len(files)}")

            # Check what directories will be restored
            dirs_in_backup = set()
            for f in files:
                if "/" in f:
                    dirs_in_backup.add(f.split("/")[0])

            print(f"Directories: {', '.join(sorted(dirs_in_backup))}")

            # Show metadata
            if "backup_metadata.json" in files:
                metadata = json.loads(zf.read("backup_metadata.json"))
                print(f"Created: {metadata.get('created', '?')}")
                if metadata.get("comment"):
                    print(f"Comment: {metadata['comment']}")

    except Exception as e:
        print(f"Error reading backup: {e}")
        return False

    if not confirm:
        print("\n!!! WARNING !!!")
        print("This will OVERWRITE existing files in state/, configs/, and logs/")
        print("Run with --confirm to proceed with restore")
        return False

    # Confirm again
    print("\nRestoring backup...")

    try:
        with zipfile.ZipFile(backup_path, "r") as zf:
            restored = 0
            for name in zf.namelist():
                if name == "backup_metadata.json":
                    continue
                if name.endswith("/"):
                    continue

                # Determine target path
                parts = name.split("/", 1)
                if len(parts) != 2:
                    continue

                dir_name, rel_path = parts
                if dir_name == "state":
                    target = STATE_DIR / rel_path
                elif dir_name == "configs":
                    target = CONFIGS_DIR / rel_path
                elif dir_name == "logs":
                    target = LOGS_DIR / rel_path
                else:
                    continue

                # Create parent directory
                target.parent.mkdir(parents=True, exist_ok=True)

                # Extract file
                with zf.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                restored += 1

            print(f"Restored {restored} files")
            jlog("backup_restored", backup=backup_path.name, files=restored)
            return True

    except Exception as e:
        print(f"Error restoring backup: {e}")
        jlog("backup_restore_error", error=str(e), level="ERROR")
        return False


def delete_backup(backup_name: str, confirm: bool = False) -> bool:
    """Delete a backup file."""
    backup_path = BACKUPS_DIR / backup_name
    if not backup_path.exists():
        backup_path = BACKUPS_DIR / (backup_name + ".zip")

    if not backup_path.exists():
        print(f"Backup not found: {backup_name}")
        return False

    if not confirm:
        print(f"Would delete: {backup_path}")
        print("Run with --confirm to proceed")
        return False

    try:
        backup_path.unlink()
        print(f"Deleted: {backup_path.name}")
        jlog("backup_deleted", backup=backup_path.name)
        return True
    except Exception as e:
        print(f"Error deleting backup: {e}")
        return False


def cleanup_old_backups(keep: int = 10, confirm: bool = False) -> None:
    """Remove old backups, keeping only the most recent N."""
    backups = list_backups()
    if len(backups) <= keep:
        print(f"Only {len(backups)} backups exist, nothing to clean up (keeping {keep})")
        return

    to_delete = backups[keep:]
    print(f"Backups to delete: {len(to_delete)}")

    if not confirm:
        for b in to_delete:
            print(f"  Would delete: {b['name']}")
        print(f"\nRun with --confirm to delete these backups")
        return

    for b in to_delete:
        delete_backup(b["name"], confirm=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backup Kobe trading system state, configs, and logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/backup.py --create
    python scripts/backup.py --create --comment "Before major update"
    python scripts/backup.py --list
    python scripts/backup.py --restore backup_20240101_120000.zip --confirm
    python scripts/backup.py --cleanup --keep 5 --confirm
        """
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file"
    )
    ap.add_argument(
        "--create",
        action="store_true",
        help="Create a new backup"
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List existing backups"
    )
    ap.add_argument(
        "--restore",
        type=str,
        metavar="BACKUP",
        help="Restore from a backup file"
    )
    ap.add_argument(
        "--delete",
        type=str,
        metavar="BACKUP",
        help="Delete a backup file"
    )
    ap.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove old backups"
    )
    ap.add_argument(
        "--keep",
        type=int,
        default=10,
        help="Number of backups to keep during cleanup (default: 10)"
    )
    ap.add_argument(
        "--comment",
        type=str,
        help="Comment to add to backup metadata"
    )
    ap.add_argument(
        "--no-state",
        action="store_true",
        help="Exclude state/ from backup"
    )
    ap.add_argument(
        "--no-configs",
        action="store_true",
        help="Exclude configs/ from backup"
    )
    ap.add_argument(
        "--no-logs",
        action="store_true",
        help="Exclude logs/ from backup"
    )
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm destructive operations (restore, delete, cleanup)"
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )

    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    print("=" * 60)
    print("KOBE TRADING SYSTEM - BACKUP MANAGER")
    print(f"Time: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

    # Handle commands
    if args.create:
        create_backup(
            include_state=not args.no_state,
            include_configs=not args.no_configs,
            include_logs=not args.no_logs,
            comment=args.comment
        )

    elif args.list:
        if args.json:
            backups = list_backups()
            print(json.dumps(backups, indent=2))
        else:
            show_backups()

    elif args.restore:
        restore_backup(args.restore, confirm=args.confirm)

    elif args.delete:
        delete_backup(args.delete, confirm=args.confirm)

    elif args.cleanup:
        cleanup_old_backups(keep=args.keep, confirm=args.confirm)

    else:
        # Default: show list
        show_backups()
        print("\nUse --create to create a new backup")
        print("Use --restore <name> --confirm to restore from backup")


if __name__ == "__main__":
    main()
