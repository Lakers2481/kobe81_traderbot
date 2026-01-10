"""
PAPER MODE ENFORCEMENT - SINGLE GUARD FUNCTION
==============================================

All order placement MUST call ensure_paper_mode_or_die() before submitting.

This is the SINGLE SOURCE OF TRUTH for paper mode enforcement.
Live trading is IMPOSSIBLE unless this guard is bypassed.

DEFENSE IN DEPTH:
1. PAPER_ONLY_ENFORCED = True (hardcoded, cannot change at runtime)
2. Checks ALPACA_BASE_URL for live endpoint
3. Checks kill switch status
4. Logs ALL checks to audit trail

To enable live trading (NOT RECOMMENDED):
1. Change PAPER_ONLY_ENFORCED = False in this file
2. Change NO_LIVE_ORDERS = False in execution_choke.py
3. Set all 7 flags in evaluate_safety_gates()
4. Provide valid ACK token at runtime

This requires deliberate, manual code changes - not just environment variables.

FIX (2026-01-10): Unified exception hierarchy - now uses core.exceptions.
"""

import os
import logging
from pathlib import Path
from typing import Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORT FROM UNIFIED EXCEPTION HIERARCHY
# =============================================================================

from core.exceptions import LiveTradingBlockedError, KillSwitchActiveError

# Re-export for backward compatibility (other modules may import from here)
__all__ = [
    "LiveTradingBlockedError",
    "KillSwitchActiveError",
    "ensure_paper_mode_or_die",
    "is_paper_endpoint",
    "get_current_endpoint",
    "require_paper_mode",
    "verify_paper_mode_active",
    "print_paper_mode_status",
    "PAPER_ONLY_ENFORCED",
]

# =============================================================================
# HARDCODED SAFETY FLAGS - Cannot be changed at runtime
# =============================================================================

PAPER_ONLY_ENFORCED: bool = True  # NEVER change this programmatically
LIVE_ENDPOINT: str = "https://api.alpaca.markets"
PAPER_ENDPOINT: str = "https://paper-api.alpaca.markets"


# =============================================================================
# THE SINGLE GUARD FUNCTION
# =============================================================================


def ensure_paper_mode_or_die(context: str = "unknown") -> Tuple[bool, str]:
    """
    THE SINGLE GUARD FUNCTION FOR ALL ORDER PLACEMENT.

    This function MUST be called as the FIRST line in any function
    that places orders. No exceptions.

    Args:
        context: Description of where this is being called from
                 (e.g., "place_ioc_limit:AAPL")

    Returns:
        Tuple of (True, "PAPER MODE CONFIRMED") if paper mode is active

    Raises:
        LiveTradingBlockedError: If attempting to use live endpoint
        KillSwitchActiveError: If kill switch is active

    Example:
        >>> ensure_paper_mode_or_die(context="place_ioc_limit:AAPL")
        (True, "PAPER MODE CONFIRMED")

        >>> # With live endpoint:
        >>> # ALPACA_BASE_URL=https://api.alpaca.markets
        >>> ensure_paper_mode_or_die(context="test")
        LiveTradingBlockedError: BLOCKED: ALPACA_BASE_URL points to LIVE endpoint
    """
    timestamp = datetime.now().isoformat()

    # =========================================================================
    # CHECK 1: PAPER_ONLY_ENFORCED flag (hardcoded defense)
    # =========================================================================
    if not PAPER_ONLY_ENFORCED:
        # This should NEVER happen - code has been tampered with
        msg = (
            f"BLOCKED: PAPER_ONLY_ENFORCED is False. Context: {context}. "
            f"This should NEVER happen - code has been tampered with. "
            f"Timestamp: {timestamp}"
        )
        logger.critical(msg)
        _log_safety_violation("paper_only_enforced_false", context, msg)
        raise LiveTradingBlockedError(msg)

    # =========================================================================
    # CHECK 2: ALPACA_BASE_URL must be paper endpoint
    # =========================================================================
    base_url = os.getenv("ALPACA_BASE_URL", PAPER_ENDPOINT)

    # Check for live endpoint
    is_live_url = (
        base_url == LIVE_ENDPOINT or
        ("api.alpaca.markets" in base_url and "paper" not in base_url.lower())
    )

    if is_live_url:
        msg = (
            f"BLOCKED: ALPACA_BASE_URL points to LIVE endpoint ({base_url}). "
            f"Context: {context}. "
            f"This robot is PAPER-ONLY. Live trading is disabled. "
            f"Timestamp: {timestamp}"
        )
        logger.critical(msg)
        _log_safety_violation("live_endpoint_blocked", context, msg)
        raise LiveTradingBlockedError(msg)

    # =========================================================================
    # CHECK 3: Kill switch must NOT be active
    # =========================================================================
    kill_switch_file = Path("state/KILL_SWITCH")
    if kill_switch_file.exists():
        try:
            import json
            kill_data = json.loads(kill_switch_file.read_text())
            reason = kill_data.get("reason", "Unknown")
            activated_at = kill_data.get("timestamp", "Unknown")
        except Exception:
            reason = "Unknown"
            activated_at = "Unknown"

        msg = (
            f"BLOCKED: Kill switch is ACTIVE. Context: {context}. "
            f"Reason: {reason}. Activated: {activated_at}. "
            f"Timestamp: {timestamp}"
        )
        logger.warning(msg)
        _log_safety_violation("kill_switch_active", context, msg)
        raise KillSwitchActiveError(msg)

    # =========================================================================
    # ALL CHECKS PASSED - Paper mode confirmed
    # =========================================================================
    logger.debug(f"PAPER MODE CONFIRMED. Context: {context}. URL: {base_url}")

    return True, "PAPER MODE CONFIRMED"


def is_paper_endpoint(url: str) -> bool:
    """
    Check if a URL is the paper trading endpoint.

    Args:
        url: The URL to check

    Returns:
        True if paper endpoint, False otherwise
    """
    if not url:
        return True  # Default to paper if not specified

    url_lower = url.lower()
    return "paper-api" in url_lower or "paper" in url_lower


def get_current_endpoint() -> Tuple[str, bool]:
    """
    Get the current Alpaca endpoint and whether it's paper.

    Returns:
        Tuple of (endpoint_url, is_paper)
    """
    url = os.getenv("ALPACA_BASE_URL", PAPER_ENDPOINT)
    return url, is_paper_endpoint(url)


def _log_safety_violation(violation_type: str, context: str, message: str) -> None:
    """
    Log a safety violation to the audit trail.

    This creates a permanent record of any attempt to bypass paper mode.
    """
    try:
        from core.structured_log import jlog
        jlog(
            "safety_violation",
            violation_type=violation_type,
            context=context,
            message=message,
            timestamp=datetime.now().isoformat(),
            level="CRITICAL"
        )
    except ImportError:
        # If structured logging not available, use standard logging
        logger.critical(f"SAFETY_VIOLATION: {violation_type} | {context} | {message}")


# =============================================================================
# DECORATOR VERSION (for defense in depth)
# =============================================================================


def require_paper_mode(func):
    """
    Decorator that ensures paper mode before executing a function.

    Usage:
        @require_paper_mode
        def place_order(order):
            # This will only execute if paper mode is confirmed
            ...
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract context from function name and first arg if possible
        context = f"{func.__name__}"
        if args and hasattr(args[0], "symbol"):
            context = f"{func.__name__}:{args[0].symbol}"

        # Call the guard
        ensure_paper_mode_or_die(context=context)

        # If we get here, paper mode is confirmed
        return func(*args, **kwargs)

    return wrapper


# =============================================================================
# VERIFICATION FUNCTIONS (for smoke tests)
# =============================================================================


def verify_paper_mode_active() -> dict:
    """
    Verify that paper mode is properly configured.

    Returns:
        Dict with verification results
    """
    url, is_paper = get_current_endpoint()
    kill_switch_active = Path("state/KILL_SWITCH").exists()

    return {
        "paper_only_enforced": PAPER_ONLY_ENFORCED,
        "alpaca_url": url,
        "is_paper_endpoint": is_paper,
        "kill_switch_active": kill_switch_active,
        "paper_mode_confirmed": PAPER_ONLY_ENFORCED and is_paper and not kill_switch_active,
    }


def print_paper_mode_status() -> None:
    """Print current paper mode status to console."""
    status = verify_paper_mode_active()

    print("=" * 60)
    print("PAPER MODE STATUS")
    print("=" * 60)
    print(f"  PAPER_ONLY_ENFORCED: {status['paper_only_enforced']}")
    print(f"  ALPACA_BASE_URL:     {status['alpaca_url']}")
    print(f"  Is Paper Endpoint:   {status['is_paper_endpoint']}")
    print(f"  Kill Switch Active:  {status['kill_switch_active']}")
    print("-" * 60)

    if status["paper_mode_confirmed"]:
        print("  STATUS: PAPER MODE CONFIRMED")
    else:
        print("  STATUS: WARNING - Paper mode NOT fully confirmed!")

    print("=" * 60)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    print_paper_mode_status()

    # Test the guard
    try:
        ok, msg = ensure_paper_mode_or_die(context="self_test")
        print(f"\nGuard test: {msg}")
    except (LiveTradingBlockedError, KillSwitchActiveError) as e:
        print(f"\nGuard test BLOCKED: {e}")
