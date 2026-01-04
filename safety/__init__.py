"""
Safety module for Kobe Trading System.

This module provides centralized safety enforcement across all components.
PAPER_ONLY mode is enforced globally - no live trading without explicit override.

Usage:
    from safety import PAPER_ONLY, assert_paper_only, get_trading_mode

    # Check mode
    if PAPER_ONLY:
        print("Running in paper mode")

    # Assert before any execution
    assert_paper_only()  # Raises if violated

    # Get detailed mode info
    mode = get_trading_mode()
    print(f"Mode: {mode['mode']}, Reason: {mode['reason']}")
"""

from safety.mode import (
    PAPER_ONLY,
    LIVE_TRADING_ENABLED,
    assert_paper_only,
    get_trading_mode,
    is_paper_mode,
    is_live_mode,
    SafetyViolationError,
    TradingMode,
    log_safety_status,
    validate_order_allowed,
)

__all__ = [
    "PAPER_ONLY",
    "LIVE_TRADING_ENABLED",
    "assert_paper_only",
    "get_trading_mode",
    "is_paper_mode",
    "is_live_mode",
    "SafetyViolationError",
    "TradingMode",
    "log_safety_status",
    "validate_order_allowed",
]
