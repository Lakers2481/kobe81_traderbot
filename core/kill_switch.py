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

    # Auto-trigger from specific condition:
    auto_trigger_kill_switch(
        trigger=KillSwitchTrigger.MAX_DRAWDOWN,
        reason="Drawdown exceeded 15%",
        context={"drawdown_pct": 15.2}
    )

FIX (2026-01-04): Added atomic write pattern for kill switch activation.
Uses temp file + rename to prevent corruption on crash.

FIX (2026-01-10): Unified exception hierarchy - now uses core.exceptions.
Added KillSwitchTrigger enum for automatic trigger types (blueprint alignment).
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar
import json
import os
import tempfile

from core.structured_log import jlog

# Import from unified exception hierarchy
from core.exceptions import KillSwitchActiveError

# Re-export for backward compatibility (other modules import from here)
__all__ = [
    "KillSwitchActiveError",
    "KillSwitchTrigger",
    "is_kill_switch_active",
    "get_kill_switch_info",
    "activate_kill_switch",
    "auto_trigger_kill_switch",
    "deactivate_kill_switch",
    "check_kill_switch",
    "require_no_kill_switch",
    "require_no_kill_switch_async",
    "require_no_kill_switch_graceful",
    "require_no_kill_switch_graceful_async",
]


# =============================================================================
# KILL SWITCH TRIGGER TYPES (Blueprint Alignment)
# =============================================================================

class KillSwitchTrigger(Enum):
    """
    Automatic kill switch trigger types.

    These represent different conditions that can automatically
    activate the kill switch to halt trading.
    """
    MANUAL = "manual"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_DAILY_LOSS = "max_daily_loss"
    RECONCILIATION_FAILURE = "reconciliation_failure"
    DATA_FEED_FAILURE = "data_feed_failure"
    BROKER_DISCONNECT = "broker_disconnect"
    EQUITY_FETCH_FAILURE = "equity_fetch_failure"
    EXECUTION_ANOMALY = "execution_anomaly"
    POSITION_LIMIT_BREACH = "position_limit_breach"


# Kill switch file location
KILL_SWITCH_PATH = Path("state/KILL_SWITCH")

# FIX (2026-01-07): Grace period allows exit orders after kill switch activation
# This prevents orphaned positions when kill switch is triggered
GRACE_PERIOD_SECONDS = 60

T = TypeVar("T")


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


def auto_trigger_kill_switch(
    trigger: KillSwitchTrigger,
    reason: str,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Automatically trigger the kill switch from a specific condition.

    This function is called by various monitoring systems when they detect
    conditions that require immediate trading halt.

    Args:
        trigger: The type of trigger that caused activation
        reason: Human-readable explanation
        context: Additional context about the trigger condition

    Example:
        auto_trigger_kill_switch(
            trigger=KillSwitchTrigger.MAX_DRAWDOWN,
            reason="Drawdown exceeded 15%",
            context={"drawdown_pct": 15.2, "threshold": 15.0}
        )
    """
    KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)

    info = {
        "reason": reason,
        "trigger": trigger.value,
        "activated_at": datetime.utcnow().isoformat(),
        "activated_by": "auto_trigger",
        "context": context or {},
    }

    # Atomic write: write to temp file, then rename
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=KILL_SWITCH_PATH.parent,
            suffix=".tmp",
            prefix="KILL_SWITCH_"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        Path(tmp_path).replace(KILL_SWITCH_PATH)
        tmp_path = None
    except Exception as e:
        jlog("kill_switch_auto_trigger_error", level="ERROR", error=str(e))
        # Fallback: try direct write
        try:
            KILL_SWITCH_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
        except Exception:
            pass
        raise
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    jlog(
        "kill_switch_auto_triggered",
        level="CRITICAL",
        trigger=trigger.value,
        reason=reason,
        context=context,
    )


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


def check_kill_switch(allow_exit_orders: bool = False) -> None:
    """
    Check kill switch and raise if active.

    FIX (2026-01-07): Added grace period for exit orders.
    When kill switch activates, exit orders are allowed for GRACE_PERIOD_SECONDS
    to close positions gracefully before full halt.

    Args:
        allow_exit_orders: If True, allows operation during grace period for exits

    Raises:
        KillSwitchActiveError: If kill switch is active (and outside grace period for exits)
    """
    if not is_kill_switch_active():
        return

    info = get_kill_switch_info()
    reason = info.get("reason", "Unknown") if info else "Unknown"
    activated_at = info.get("activated_at") if info else None

    # FIX (2026-01-07): Grace period for exit orders
    if allow_exit_orders and activated_at:
        try:
            activation_time = datetime.fromisoformat(activated_at)
            elapsed = (datetime.now() - activation_time).total_seconds()

            if elapsed < GRACE_PERIOD_SECONDS:
                # Within grace period - allow exit orders only
                remaining = GRACE_PERIOD_SECONDS - elapsed
                jlog("kill_switch_grace_period",
                     level="WARNING",
                     remaining_seconds=int(remaining),
                     reason=reason)
                return  # Allow the operation
        except Exception as e:
            jlog("kill_switch_grace_parse_error", level="ERROR", error=str(e))
            # If parsing fails, enforce kill switch

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


# FIX (2026-01-07): Graceful variants allow exit orders during grace period
def require_no_kill_switch_graceful(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for EXIT functions that allows operation during grace period.

    Use this for functions that close positions. After kill switch activates,
    there's a 60-second window to gracefully close positions.

    Usage:
        @require_no_kill_switch_graceful
        def close_position(symbol: str) -> None:
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        check_kill_switch(allow_exit_orders=True)
        return func(*args, **kwargs)
    return wrapper


def require_no_kill_switch_graceful_async(func: Callable[..., T]) -> Callable[..., T]:
    """
    Async decorator for EXIT functions that allows operation during grace period.

    Use this for async functions that close positions.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        check_kill_switch(allow_exit_orders=True)
        return await func(*args, **kwargs)
    return wrapper
