"""
Centralized Trading Mode Enforcement for Kobe Trading System.

This module is the SINGLE SOURCE OF TRUTH for trading mode.
All execution paths MUST check this module before placing orders.

CRITICAL SAFETY:
    - PAPER_ONLY = True by default (NEVER change programmatically)
    - LIVE_TRADING_ENABLED = False by default
    - To enable live trading: Set environment variable KOBE_LIVE_TRADING=true
    - Even then, kill switch takes precedence

Integration Points:
    - execution/broker_alpaca.py: Checks before order submission
    - scripts/run_paper_trade.py: Validates paper mode
    - scripts/run_live_trade_micro.py: Validates live mode allowed
    - autonomous/brain.py: Enforces mode throughout operation
    - agents/base_agent.py: ReAct agents check before actions

Author: Kobe Trading System
Version: 1.0.0
Last Updated: 2026-01-03
"""

import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class TradingMode(Enum):
    """Trading mode enumeration."""
    PAPER = "paper"
    LIVE = "live"
    DISABLED = "disabled"  # Kill switch active


class SafetyViolationError(Exception):
    """Raised when a safety constraint is violated."""
    pass


# =============================================================================
# CORE SAFETY CONSTANTS - DO NOT MODIFY PROGRAMMATICALLY
# =============================================================================

# Primary safety flag - hardcoded to True
# This can ONLY be overridden by explicit environment variable
PAPER_ONLY: bool = True

# Secondary flag - must be explicitly enabled for live trading
LIVE_TRADING_ENABLED: bool = False

# Environment variable to allow live trading
# Set KOBE_LIVE_TRADING=true to enable (requires conscious decision)
_ENV_LIVE_TRADING = os.getenv("KOBE_LIVE_TRADING", "false").lower() == "true"

# Kill switch file path
KILL_SWITCH_PATH = Path(__file__).parent.parent / "state" / "KILL_SWITCH"


def _check_kill_switch() -> bool:
    """Check if kill switch file exists."""
    return KILL_SWITCH_PATH.exists()


def get_trading_mode() -> dict:
    """
    Get current trading mode with detailed explanation.

    Returns:
        dict with keys:
            - mode: TradingMode enum value
            - mode_str: String representation
            - paper_only: bool
            - live_allowed: bool
            - kill_switch: bool
            - reason: Explanation string
            - timestamp: Current UTC time
            - env_override: Whether env var is set

    Example:
        >>> mode = get_trading_mode()
        >>> print(mode['mode_str'])
        'paper'
        >>> print(mode['reason'])
        'PAPER_ONLY=True, live trading disabled by default'
    """
    kill_switch = _check_kill_switch()

    # Determine mode
    if kill_switch:
        mode = TradingMode.DISABLED
        reason = "Kill switch file exists at state/KILL_SWITCH"
        live_allowed = False
    elif _ENV_LIVE_TRADING and not PAPER_ONLY:
        mode = TradingMode.LIVE
        reason = "Live trading enabled via KOBE_LIVE_TRADING=true environment variable"
        live_allowed = True
    else:
        mode = TradingMode.PAPER
        if not _ENV_LIVE_TRADING:
            reason = "PAPER_ONLY=True, live trading disabled by default"
        else:
            reason = "PAPER_ONLY=True despite env override (code constant takes precedence)"
        live_allowed = False

    return {
        "mode": mode,
        "mode_str": mode.value,
        "paper_only": PAPER_ONLY,
        "live_allowed": live_allowed,
        "kill_switch": kill_switch,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat(),
        "env_override": _ENV_LIVE_TRADING,
    }


def is_paper_mode() -> bool:
    """Check if system is in paper trading mode."""
    mode_info = get_trading_mode()
    return mode_info["mode"] == TradingMode.PAPER


def is_live_mode() -> bool:
    """Check if system is allowed to do live trading."""
    mode_info = get_trading_mode()
    return mode_info["mode"] == TradingMode.LIVE


def assert_paper_only(context: Optional[str] = None) -> None:
    """
    Assert that system is in paper-only mode.

    Raises SafetyViolationError if:
        - Kill switch is active (trading disabled)
        - System is somehow in live mode when it shouldn't be

    This is a SAFETY CHECK to be called before any order submission.

    Args:
        context: Optional context string for error messages

    Raises:
        SafetyViolationError: If safety constraints are violated

    Example:
        >>> from safety import assert_paper_only
        >>> assert_paper_only("placing order for AAPL")
        >>> # If kill switch active, raises:
        >>> # SafetyViolationError: Trading disabled by kill switch [placing order for AAPL]
    """
    mode_info = get_trading_mode()
    context_str = f" [{context}]" if context else ""

    if mode_info["kill_switch"]:
        raise SafetyViolationError(
            f"Trading disabled by kill switch{context_str}. "
            f"Remove state/KILL_SWITCH to resume."
        )

    if mode_info["mode"] == TradingMode.LIVE and PAPER_ONLY:
        raise SafetyViolationError(
            f"Live trading attempted but PAPER_ONLY=True{context_str}. "
            f"This should never happen - check code for bypass attempts."
        )


def validate_order_allowed(
    symbol: str,
    side: str,
    quantity: float,
    is_paper: bool = True,
) -> dict:
    """
    Validate if an order is allowed under current safety constraints.

    Args:
        symbol: Stock symbol
        side: 'buy' or 'sell'
        quantity: Number of shares
        is_paper: Whether this is a paper order

    Returns:
        dict with:
            - allowed: bool
            - reason: Explanation
            - mode: Current trading mode

    Example:
        >>> result = validate_order_allowed("AAPL", "buy", 100, is_paper=True)
        >>> if result['allowed']:
        ...     place_order(...)
    """
    mode_info = get_trading_mode()

    # Check kill switch first
    if mode_info["kill_switch"]:
        return {
            "allowed": False,
            "reason": "Kill switch active - all trading halted",
            "mode": mode_info["mode_str"],
        }

    # Paper orders always allowed (when not killed)
    if is_paper:
        return {
            "allowed": True,
            "reason": "Paper order allowed",
            "mode": mode_info["mode_str"],
        }

    # Live orders only if explicitly enabled
    if mode_info["live_allowed"]:
        return {
            "allowed": True,
            "reason": "Live trading enabled via environment",
            "mode": mode_info["mode_str"],
        }

    return {
        "allowed": False,
        "reason": f"Live trading not allowed: {mode_info['reason']}",
        "mode": mode_info["mode_str"],
    }


def log_safety_status() -> str:
    """
    Generate a safety status report for logging.

    Returns:
        Multi-line string with current safety status
    """
    mode_info = get_trading_mode()

    status_lines = [
        "=" * 50,
        "KOBE TRADING SYSTEM - SAFETY STATUS",
        "=" * 50,
        f"Timestamp: {mode_info['timestamp']}",
        f"Mode: {mode_info['mode_str'].upper()}",
        f"Paper Only: {mode_info['paper_only']}",
        f"Live Allowed: {mode_info['live_allowed']}",
        f"Kill Switch: {'ACTIVE' if mode_info['kill_switch'] else 'inactive'}",
        f"Env Override: {mode_info['env_override']}",
        f"Reason: {mode_info['reason']}",
        "=" * 50,
    ]

    return "\n".join(status_lines)


# =============================================================================
# Module initialization - log safety status on import
# =============================================================================

def _init_safety_module():
    """Initialize safety module and validate state."""
    # Ensure state directory exists for kill switch
    state_dir = Path(__file__).parent.parent / "state"
    state_dir.mkdir(exist_ok=True)

    # Log current status
    mode_info = get_trading_mode()

    # Return status for callers who want it
    return mode_info


# Run on import
_INIT_STATUS = _init_safety_module()
