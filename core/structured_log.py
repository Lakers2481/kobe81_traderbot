from __future__ import annotations

import json
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict
import logging


LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "events.jsonl"

# Log rotation settings (configurable via environment)
MAX_LOG_BYTES = int(os.getenv("KOBE_LOG_MAX_BYTES", 10 * 1024 * 1024))  # 10MB default
LOG_BACKUP_COUNT = int(os.getenv("KOBE_LOG_BACKUP_COUNT", 5))  # Keep 5 backups

# Set up rotating file handler
_file_handler: RotatingFileHandler | None = None


def _get_file_handler() -> RotatingFileHandler:
    """Get or create the rotating file handler."""
    global _file_handler
    if _file_handler is None:
        _file_handler = RotatingFileHandler(
            str(LOG_FILE),
            maxBytes=MAX_LOG_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
    return _file_handler


def jlog(event: str, level: str = "INFO", **fields: Any) -> None:
    """
    Write a structured JSON log entry with automatic rotation.

    Args:
        event: Event name/type
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        **fields: Additional fields to include in the log entry
    """
    rec: Dict[str, Any] = {
        "ts": datetime.utcnow().isoformat(),
        "level": level,
        "event": event,
        **fields,
    }
    line = json.dumps(rec, default=str)

    # Write to rotating file
    handler = _get_file_handler()
    try:
        # Use the handler's stream directly for atomic writes
        handler.stream.write(line + "\n")
        handler.stream.flush()

        # Check if rotation is needed
        if handler.shouldRollover(logging.LogRecord(
            name="kobe", level=logging.INFO, pathname="", lineno=0,
            msg=line, args=(), exc_info=None
        )):
            handler.doRollover()
    except Exception:
        # Fallback to direct file write if handler fails
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Also echo concise line to console
    print(f"[{rec['level']}] {rec['event']} | {fields}")


def get_log_stats() -> Dict[str, Any]:
    """Get statistics about current log files."""
    stats = {
        "main_log": str(LOG_FILE),
        "main_log_size_bytes": 0,
        "backup_files": [],
        "total_size_bytes": 0,
    }

    if LOG_FILE.exists():
        stats["main_log_size_bytes"] = LOG_FILE.stat().st_size
        stats["total_size_bytes"] = stats["main_log_size_bytes"]

    # Find backup files
    for i in range(1, LOG_BACKUP_COUNT + 1):
        backup = LOG_DIR / f"events.jsonl.{i}"
        if backup.exists():
            size = backup.stat().st_size
            stats["backup_files"].append({"file": str(backup), "size_bytes": size})
            stats["total_size_bytes"] += size

    return stats


def rotate_logs_now() -> bool:
    """Force immediate log rotation."""
    handler = _get_file_handler()
    try:
        handler.doRollover()
        return True
    except Exception:
        return False


def read_recent_logs(count: int = 100, level: str | None = None) -> list[Dict[str, Any]]:
    """
    Read the most recent log entries.

    Args:
        count: Maximum number of entries to return
        level: Optional filter by log level

    Returns:
        List of log entries (most recent last)
    """
    entries = []

    if not LOG_FILE.exists():
        return entries

    try:
        with LOG_FILE.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        # Read from end for efficiency
        for line in reversed(lines):
            if len(entries) >= count:
                break
            try:
                entry = json.loads(line.strip())
                if level is None or entry.get("level") == level:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue

        # Return in chronological order
        return list(reversed(entries))
    except Exception:
        return []
