"""
Central State Manager
=====================

Provides atomic, lock-safe state management for all Kobe state files.
Replaces direct JSON writes across 22+ locations with a single, safe API.

Features:
- Cross-platform file locking (via filelock package)
- Atomic writes (temp file + rename)
- Consistent serialization
- Automatic backup before writes
- State validation

FIX (2026-01-05): Created to address race conditions in state writes.

Usage:
    from portfolio.state_manager import get_state_manager

    sm = get_state_manager()

    # Read state
    positions = sm.get_positions()

    # Write state (atomic, locked)
    sm.set_positions({"AAPL": {"qty": 100, "entry": 150.0}})

    # Transaction-like updates
    with sm.state_lock("positions"):
        positions = sm.get_positions()
        positions["MSFT"] = {"qty": 50, "entry": 300.0}
        sm.set_positions(positions)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Generator
import threading

logger = logging.getLogger(__name__)

# Try to import filelock for cross-platform locking
try:
    from filelock import FileLock, Timeout
    FILELOCK_AVAILABLE = True
except ImportError:
    FILELOCK_AVAILABLE = False
    logger.warning(
        "filelock not installed. State manager will use threading locks only. "
        "Install with: pip install filelock"
    )


# State file paths (relative to project root)
STATE_DIR = Path(__file__).resolve().parents[1] / "state"

STATE_FILES = {
    "positions": STATE_DIR / "position_state.json",
    "weekly_budget": STATE_DIR / "weekly_budget.json",
    "brain_state": STATE_DIR / "autonomous" / "brain_state.json",
    "reconciliation": STATE_DIR / "reconciliation" / "last_reconcile.json",
    "earnings_cache": STATE_DIR / "earnings_cache.json",
    "orders": STATE_DIR / "orders.json",
    "daily_pnl": STATE_DIR / "daily_pnl.json",
    "kill_switch": STATE_DIR / "KILL_SWITCH",
    "watchlist": STATE_DIR / "watchlist" / "today_validated.json",
}


class StateManager:
    """
    Central state manager with atomic writes and file locking.

    Thread-safe and process-safe (when filelock is installed).
    """

    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize the state manager.

        Args:
            state_dir: Override the default state directory
        """
        self.state_dir = state_dir or STATE_DIR
        self._thread_locks: Dict[str, threading.RLock] = {}
        self._file_locks: Dict[str, Any] = {}  # FileLock instances
        self._lock = threading.RLock()  # Lock for accessing lock dicts

        # Ensure state directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create required state directories."""
        directories = [
            self.state_dir,
            self.state_dir / "autonomous",
            self.state_dir / "reconciliation",
            self.state_dir / "watchlist",
            self.state_dir / "backups",
        ]
        for d in directories:
            d.mkdir(parents=True, exist_ok=True)

    def _get_thread_lock(self, name: str) -> threading.RLock:
        """Get or create a threading lock for a state name."""
        with self._lock:
            if name not in self._thread_locks:
                self._thread_locks[name] = threading.RLock()
            return self._thread_locks[name]

    def _get_file_lock(self, name: str) -> Optional[Any]:
        """Get or create a file lock for a state name."""
        if not FILELOCK_AVAILABLE:
            return None

        with self._lock:
            if name not in self._file_locks:
                lock_path = self._get_state_path(name).with_suffix(".lock")
                self._file_locks[name] = FileLock(lock_path, timeout=30)
            return self._file_locks[name]

    def _get_state_path(self, name: str) -> Path:
        """Get the file path for a named state."""
        if name in STATE_FILES:
            return STATE_FILES[name]
        # Default to state_dir/<name>.json
        return self.state_dir / f"{name}.json"

    @contextmanager
    def state_lock(self, name: str) -> Generator[None, None, None]:
        """
        Context manager for exclusive access to a state file.

        Uses both thread lock and file lock for maximum safety.

        Args:
            name: State name (e.g., "positions", "weekly_budget")

        Yields:
            None (use get/set methods inside the context)

        Example:
            with sm.state_lock("positions"):
                positions = sm.get_positions()
                positions["AAPL"] = {...}
                sm.set_positions(positions)
        """
        thread_lock = self._get_thread_lock(name)
        file_lock = self._get_file_lock(name)

        # Acquire thread lock first
        thread_lock.acquire()
        try:
            # Then acquire file lock if available
            if file_lock:
                try:
                    file_lock.acquire()
                except Timeout:
                    logger.error(f"Timeout acquiring file lock for {name}")
                    raise

            try:
                yield
            finally:
                if file_lock and file_lock.is_locked:
                    file_lock.release()
        finally:
            thread_lock.release()

    def _read_json(self, path: Path, default: Any = None) -> Any:
        """Read JSON from a file, returning default if not found."""
        if not path.exists():
            return default if default is not None else {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {path}: {e}")
            return default if default is not None else {}
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return default if default is not None else {}

    def _write_json(self, path: Path, data: Any, backup: bool = True) -> bool:
        """
        Write JSON to a file atomically.

        Uses temp file + rename pattern for atomicity.

        Args:
            path: Target file path
            data: Data to serialize as JSON
            backup: Whether to create a backup before writing

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup if file exists
            if backup and path.exists():
                backup_path = self.state_dir / "backups" / f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup_path)

            # Write to temp file first
            fd, temp_path = tempfile.mkstemp(suffix=".json", dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)

                # Atomic rename (works on POSIX, best-effort on Windows)
                temp = Path(temp_path)
                if os.name == "nt":
                    # Windows: need to remove target first
                    if path.exists():
                        path.unlink()
                temp.rename(path)
                return True
            except Exception:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

        except Exception as e:
            logger.error(f"Error writing {path}: {e}")
            return False

    # =========================================================================
    # Position State
    # =========================================================================

    def get_positions(self) -> Dict[str, Any]:
        """
        Get current position state.

        Returns:
            Dict mapping symbol -> position details
        """
        path = self._get_state_path("positions")
        return self._read_json(path, default={})

    def set_positions(self, positions: Dict[str, Any]) -> bool:
        """
        Set position state atomically.

        Args:
            positions: Dict mapping symbol -> position details

        Returns:
            True if successful
        """
        path = self._get_state_path("positions")
        with self.state_lock("positions"):
            return self._write_json(path, positions)

    def update_position(self, symbol: str, position: Dict[str, Any]) -> bool:
        """
        Update a single position atomically.

        Args:
            symbol: Stock symbol
            position: Position details

        Returns:
            True if successful
        """
        with self.state_lock("positions"):
            positions = self.get_positions()
            positions[symbol] = position
            return self.set_positions(positions)

    def remove_position(self, symbol: str) -> bool:
        """
        Remove a position atomically.

        Args:
            symbol: Stock symbol to remove

        Returns:
            True if successful
        """
        with self.state_lock("positions"):
            positions = self.get_positions()
            if symbol in positions:
                del positions[symbol]
                return self.set_positions(positions)
            return True

    # =========================================================================
    # Weekly Budget State
    # =========================================================================

    def get_weekly_budget(self) -> Dict[str, Any]:
        """Get weekly budget state."""
        path = self._get_state_path("weekly_budget")
        return self._read_json(path, default={
            "week_start": None,
            "notional_used": 0.0,
            "max_notional": 10000.0,
        })

    def set_weekly_budget(self, budget: Dict[str, Any]) -> bool:
        """Set weekly budget state atomically."""
        path = self._get_state_path("weekly_budget")
        with self.state_lock("weekly_budget"):
            return self._write_json(path, budget)

    def update_weekly_notional(self, amount: float) -> bool:
        """Add to weekly notional usage atomically."""
        with self.state_lock("weekly_budget"):
            budget = self.get_weekly_budget()
            budget["notional_used"] = budget.get("notional_used", 0.0) + amount
            return self.set_weekly_budget(budget)

    # =========================================================================
    # Brain State (Autonomous)
    # =========================================================================

    def get_brain_state(self) -> Dict[str, Any]:
        """Get autonomous brain state."""
        path = self._get_state_path("brain_state")
        return self._read_json(path, default={
            "status": "idle",
            "last_heartbeat": None,
            "current_task": None,
        })

    def set_brain_state(self, state: Dict[str, Any]) -> bool:
        """Set brain state atomically."""
        path = self._get_state_path("brain_state")
        with self.state_lock("brain_state"):
            return self._write_json(path, state)

    # =========================================================================
    # Reconciliation State
    # =========================================================================

    def get_reconciliation(self) -> Dict[str, Any]:
        """Get last reconciliation result."""
        path = self._get_state_path("reconciliation")
        return self._read_json(path, default={})

    def save_reconciliation(self, result: Dict[str, Any]) -> bool:
        """Save reconciliation result atomically."""
        path = self._get_state_path("reconciliation")
        result["timestamp"] = datetime.now().isoformat()
        with self.state_lock("reconciliation"):
            return self._write_json(path, result)

    # =========================================================================
    # Earnings Cache
    # =========================================================================

    def get_earnings_cache(self) -> Dict[str, Any]:
        """Get earnings cache."""
        path = self._get_state_path("earnings_cache")
        return self._read_json(path, default={})

    def set_earnings_cache(self, cache: Dict[str, Any]) -> bool:
        """Set earnings cache atomically."""
        path = self._get_state_path("earnings_cache")
        with self.state_lock("earnings_cache"):
            return self._write_json(path, cache)

    # =========================================================================
    # Orders State
    # =========================================================================

    def get_orders(self) -> Dict[str, Any]:
        """Get orders state."""
        path = self._get_state_path("orders")
        return self._read_json(path, default={"orders": []})

    def set_orders(self, orders: Dict[str, Any]) -> bool:
        """Set orders state atomically."""
        path = self._get_state_path("orders")
        with self.state_lock("orders"):
            return self._write_json(path, orders)

    def append_order(self, order: Dict[str, Any]) -> bool:
        """Append an order atomically."""
        with self.state_lock("orders"):
            orders = self.get_orders()
            if "orders" not in orders:
                orders["orders"] = []
            orders["orders"].append(order)
            return self.set_orders(orders)

    # =========================================================================
    # Watchlist State
    # =========================================================================

    def get_watchlist(self) -> Dict[str, Any]:
        """Get validated watchlist."""
        path = self._get_state_path("watchlist")
        return self._read_json(path, default={"symbols": [], "validated_at": None})

    def set_watchlist(self, watchlist: Dict[str, Any]) -> bool:
        """Set watchlist atomically."""
        path = self._get_state_path("watchlist")
        with self.state_lock("watchlist"):
            return self._write_json(path, watchlist)

    # =========================================================================
    # Generic State Access
    # =========================================================================

    def get_state(self, name: str, default: Any = None) -> Any:
        """
        Get any named state.

        Args:
            name: State name
            default: Default value if not found

        Returns:
            State data
        """
        path = self._get_state_path(name)
        return self._read_json(path, default=default)

    def set_state(self, name: str, data: Any, backup: bool = True) -> bool:
        """
        Set any named state atomically.

        Args:
            name: State name
            data: Data to save
            backup: Whether to create backup

        Returns:
            True if successful
        """
        path = self._get_state_path(name)
        with self.state_lock(name):
            return self._write_json(path, data, backup=backup)


# Global instance
_state_manager: Optional[StateManager] = None
_state_manager_lock = threading.Lock()


def get_state_manager() -> StateManager:
    """Get or create the global StateManager instance."""
    global _state_manager
    if _state_manager is None:
        with _state_manager_lock:
            if _state_manager is None:
                _state_manager = StateManager()
    return _state_manager


def reset_state_manager() -> None:
    """Reset the global StateManager (for testing)."""
    global _state_manager
    with _state_manager_lock:
        _state_manager = None
