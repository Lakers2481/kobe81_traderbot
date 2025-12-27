"""
Heartbeat tracking for the Kobe trading daemon.

Writes periodic heartbeat files to detect stale/crashed processes.
External monitors can check heartbeat freshness to determine if
the trading system is healthy.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Default heartbeat file location
DEFAULT_HEARTBEAT_PATH = Path("state/heartbeat.json")

# Default heartbeat interval in seconds
DEFAULT_HEARTBEAT_INTERVAL = 60

# Maximum heartbeat age before considered stale
DEFAULT_STALE_THRESHOLD = 300  # 5 minutes


class HeartbeatWriter:
    """
    Writes periodic heartbeat files for process monitoring.

    The heartbeat file contains:
    - timestamp: Last heartbeat time (ISO format)
    - pid: Process ID
    - mode: Operating mode (paper/live)
    - last_action: Description of last action taken
    - uptime_seconds: Process uptime

    Example:
        heartbeat = HeartbeatWriter("state/heartbeat.json", mode="paper")
        heartbeat.start()

        # In main loop:
        heartbeat.update("Scanned 50 stocks")

        # On shutdown:
        heartbeat.stop()
    """

    def __init__(
        self,
        heartbeat_path: str | Path = DEFAULT_HEARTBEAT_PATH,
        interval: int = DEFAULT_HEARTBEAT_INTERVAL,
        mode: str = "unknown",
    ):
        self.heartbeat_path = Path(heartbeat_path)
        self.interval = interval
        self.mode = mode
        self.start_time = time.time()
        self.last_action = "initialized"
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _write_heartbeat(self) -> None:
        """Write heartbeat to file."""
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            data = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "pid": os.getpid(),
                "mode": self.mode,
                "last_action": self.last_action,
                "uptime_seconds": int(time.time() - self.start_time),
            }

        try:
            # Write atomically (write to temp, then rename)
            temp_path = self.heartbeat_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.heartbeat_path)
            logger.debug(f"Heartbeat written: {data['timestamp']}")
        except Exception as e:
            logger.error(f"Failed to write heartbeat: {e}")

    def _heartbeat_loop(self) -> None:
        """Background thread that writes heartbeats."""
        while self._running:
            self._write_heartbeat()
            # Sleep in small intervals to allow quick shutdown
            for _ in range(self.interval):
                if not self._running:
                    break
                time.sleep(1)

    def start(self) -> None:
        """Start the background heartbeat writer."""
        if self._running:
            return

        self._running = True
        self._write_heartbeat()  # Write immediately
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        logger.info(f"Heartbeat writer started (interval={self.interval}s)")

    def stop(self) -> None:
        """Stop the background heartbeat writer."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

        # Write final heartbeat with "stopped" status
        self.update("stopped")
        self._write_heartbeat()
        logger.info("Heartbeat writer stopped")

    def update(self, action: str) -> None:
        """Update the last action (for next heartbeat write)."""
        with self._lock:
            self.last_action = action

    def write_now(self) -> None:
        """Force an immediate heartbeat write."""
        self._write_heartbeat()

    def is_running(self) -> bool:
        """Check if heartbeat writer is running."""
        return self._running

    def get_status(self) -> Dict[str, Any]:
        """Get current heartbeat status."""
        with self._lock:
            return {
                "running": self._running,
                "mode": self.mode,
                "last_action": self.last_action,
                "uptime_seconds": int(time.time() - self.start_time),
                "heartbeat_path": str(self.heartbeat_path),
            }


def read_heartbeat(
    heartbeat_path: str | Path = DEFAULT_HEARTBEAT_PATH,
) -> Optional[Dict[str, Any]]:
    """
    Read the current heartbeat file.

    Returns:
        Heartbeat data dict, or None if file doesn't exist or is invalid
    """
    path = Path(heartbeat_path)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read heartbeat: {e}")
        return None


def is_heartbeat_stale(
    heartbeat_path: str | Path = DEFAULT_HEARTBEAT_PATH,
    max_age_seconds: int = DEFAULT_STALE_THRESHOLD,
) -> bool:
    """
    Check if the heartbeat is stale (too old or missing).

    Args:
        heartbeat_path: Path to heartbeat file
        max_age_seconds: Maximum age in seconds before considered stale

    Returns:
        True if heartbeat is stale or missing, False if fresh
    """
    data = read_heartbeat(heartbeat_path)
    if data is None:
        return True

    try:
        timestamp_str = data.get("timestamp", "")
        if not timestamp_str:
            return True

        # Parse ISO timestamp
        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        age = (datetime.now(timestamp.tzinfo) - timestamp).total_seconds()

        if age > max_age_seconds:
            logger.warning(f"Heartbeat is stale: {age:.0f}s old (max: {max_age_seconds}s)")
            return True

        return False

    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse heartbeat timestamp: {e}")
        return True


def get_heartbeat_age(
    heartbeat_path: str | Path = DEFAULT_HEARTBEAT_PATH,
) -> Optional[float]:
    """
    Get the age of the heartbeat in seconds.

    Returns:
        Age in seconds, or None if heartbeat is missing/invalid
    """
    data = read_heartbeat(heartbeat_path)
    if data is None:
        return None

    try:
        timestamp_str = data.get("timestamp", "")
        if not timestamp_str:
            return None

        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        age = (datetime.now(timestamp.tzinfo) - timestamp).total_seconds()
        return age

    except (ValueError, TypeError):
        return None


# Global heartbeat instance (optional convenience)
_global_heartbeat: Optional[HeartbeatWriter] = None


def init_global_heartbeat(
    heartbeat_path: str | Path = DEFAULT_HEARTBEAT_PATH,
    interval: int = DEFAULT_HEARTBEAT_INTERVAL,
    mode: str = "unknown",
) -> HeartbeatWriter:
    """
    Initialize and start a global heartbeat writer.

    Useful when you want a single heartbeat instance across the application.
    """
    global _global_heartbeat
    if _global_heartbeat is not None:
        _global_heartbeat.stop()

    _global_heartbeat = HeartbeatWriter(heartbeat_path, interval, mode)
    _global_heartbeat.start()
    return _global_heartbeat


def get_global_heartbeat() -> Optional[HeartbeatWriter]:
    """Get the global heartbeat writer instance."""
    return _global_heartbeat


def update_global_heartbeat(action: str) -> None:
    """Update the global heartbeat's last action."""
    if _global_heartbeat:
        _global_heartbeat.update(action)


def stop_global_heartbeat() -> None:
    """Stop the global heartbeat writer."""
    global _global_heartbeat
    if _global_heartbeat:
        _global_heartbeat.stop()
        _global_heartbeat = None
