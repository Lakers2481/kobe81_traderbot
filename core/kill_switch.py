"""
Kill Switch for Kobe Trading System.
Global mechanism to halt all order submissions.

Usage:
    from core.kill_switch import require_no_kill_switch, activate_kill_switch

    @require_no_kill_switch
    def place_order(...):
        ...

    # In emergency:
    activate_kill_switch("Manual halt - investigating issue")

FIX (2026-01-04): Added atomic write pattern for kill switch activation.
Uses temp file + rename to prevent corruption on crash.
"""
from __future__ import annotations

from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable, TypeVar, Optional
import json
import os
import tempfile

from core.structured_log import jlog


# Kill switch file location
KILL_SWITCH_PATH = Path("state/KILL_SWITCH")

T = TypeVar("T")


class KillSwitchActiveError(Exception):
    """Raised when operation is blocked by active kill switch."""
    pass


def is_kill_switch_active() -> bool:
    """Check if kill switch is currently active."""
    return KILL_SWITCH_PATH.exists()


def get_kill_switch_info() -> Optional[dict]:
    """Get kill switch details if active."""
    if not KILL_SWITCH_PATH.exists():
        return None
    try:
        content = KILL_SWITCH_PATH.read_text(encoding="utf-8").strip()
        if content.startswith("{"):
            return json.loads(content)
        return {"reason": content, "activated_at": "unknown"}
    except Exception:
        return {"reason": "Kill switch active (unreadable)", "activated_at": "unknown"}


def activate_kill_switch(reason: str = "Manual activation") -> None:
    """
    Activate the kill switch to halt all trading.

    FIX (2026-01-04): Now uses atomic write pattern (temp file + rename) to prevent
    file corruption if there's a crash during write. This is critical for a safety
    mechanism - a corrupted kill switch file could lead to undefined behavior.

    Args:
        reason: Human-readable reason for activation
    """
    KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "reason": reason,
        "activated_at": datetime.utcnow().isoformat(),
        "activated_by": "system",
    }

    # Atomic write: write to temp file, then rename
    # This prevents corruption if there's a crash during write
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=KILL_SWITCH_PATH.parent,
            suffix=".tmp",
            prefix="KILL_SWITCH_"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        # Atomic rename (on POSIX systems)
        # On Windows, this may not be fully atomic but is still safer
        Path(tmp_path).replace(KILL_SWITCH_PATH)
        tmp_path = None  # Successfully renamed, don't cleanup
    except Exception as e:
        jlog("kill_switch_write_error", level="ERROR", error=str(e))
        # Fallback: try direct write if atomic failed
        try:
            KILL_SWITCH_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
        except Exception:
            pass
        raise
    finally:
        # Clean up temp file if rename failed
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    jlog("kill_switch_activated", level="CRITICAL", reason=reason)


def deactivate_kill_switch() -> bool:
    """
    Deactivate the kill switch to resume trading.

    Returns:
        True if switch was deactivated, False if it wasn't active
    """
    if not KILL_SWITCH_PATH.exists():
        return False
    try:
        KILL_SWITCH_PATH.unlink()
        jlog("kill_switch_deactivated", level="WARNING")
        return True
    except Exception as e:
        jlog("kill_switch_deactivate_failed", level="ERROR", error=str(e))
        return False


def check_kill_switch() -> None:
    """
    Check kill switch and raise if active.

    Raises:
        KillSwitchActiveError: If kill switch is active
    """
    if is_kill_switch_active():
        info = get_kill_switch_info()
        reason = info.get("reason", "Unknown") if info else "Unknown"
        # SECURITY FIX (2026-01-04): Increment Prometheus counter
        try:
            from trade_logging.prometheus_metrics import KILL_SWITCH_BLOCKS
            KILL_SWITCH_BLOCKS.inc()
        except Exception:
            pass
        raise KillSwitchActiveError(f"Kill switch active: {reason}")


def require_no_kill_switch(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to block function execution when kill switch is active.

    Usage:
        @require_no_kill_switch
        def place_order(order: OrderRecord) -> OrderRecord:
            ...

    Raises:
        KillSwitchActiveError: If kill switch is active
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        check_kill_switch()
        return func(*args, **kwargs)
    return wrapper


def require_no_kill_switch_async(func: Callable[..., T]) -> Callable[..., T]:
    """
    Async decorator to block function execution when kill switch is active.

    Usage:
        @require_no_kill_switch_async
        async def place_order_async(order: OrderRecord) -> OrderRecord:
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        check_kill_switch()
        return await func(*args, **kwargs)
    return wrapper
