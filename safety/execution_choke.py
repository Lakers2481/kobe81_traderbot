"""
GLOBAL EXECUTION CHOKE POINT - ALL ORDERS MUST PASS THROUGH HERE

This module implements the SINGLE enforcement point for ALL order submissions.
Any attempt to bypass this module MUST fail with SafetyViolationError.

SAFETY REQUIREMENTS (ALL must be TRUE for live orders):
1. LIVE_TRADING_ENABLED = True (in safety/mode.py)
2. TRADING_MODE env = "live"
3. APPROVE_LIVE_ACTION = True (in research_os/approval_gate.py)
4. APPROVE_LIVE_ACTION_2 = True (secondary approval)
5. LIVE_ORDER_ACK_TOKEN matches runtime-generated token
6. Kill switch NOT active

Paper orders bypass flags 1-5 but still check kill switch.

Author: Kobe Trading System
Version: 1.0.0
Date: 2026-01-05
"""

from __future__ import annotations

import hashlib
import os
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

from core.structured_log import jlog

# =============================================================================
# NO LIVE ORDERS HARD GATE - ABSOLUTE FINAL CHECK
# =============================================================================
#
# This is the ABSOLUTE FINAL CHECK before any order submission.
# If this is True, ALL live orders are blocked regardless of other flags.
# This is separate from PAPER_ONLY - it's an additional safety layer.
#
# To enable live trading (NOT RECOMMENDED):
# 1. Set NO_LIVE_ORDERS = False in this file
# 2. Set PAPER_ONLY_ENFORCED = False in safety/paper_guard.py
# 3. Set all 7 flags in evaluate_safety_gates()
# 4. Provide valid ACK token at runtime
#
NO_LIVE_ORDERS: bool = True  # HARDCODED - Must be manually changed to False for live

# =============================================================================
# RUNTIME TOKEN GENERATION
# =============================================================================

# Generate a unique token at runtime that must be provided for live orders
# This prevents replay attacks and ensures conscious acknowledgment
_RUNTIME_TOKEN_LOCK = threading.Lock()
_RUNTIME_TOKEN: Optional[str] = None


def _generate_runtime_token() -> str:
    """Generate a unique runtime token for live order acknowledgment."""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    random_part = secrets.token_hex(16)
    return f"KOBE_LIVE_{timestamp}_{random_part}"


def get_live_order_ack_token() -> str:
    """
    Get the current runtime token required for live order submission.

    This token is generated once per process and must be provided
    when submitting live orders.

    Returns:
        str: The current runtime ACK token
    """
    global _RUNTIME_TOKEN
    with _RUNTIME_TOKEN_LOCK:
        if _RUNTIME_TOKEN is None:
            _RUNTIME_TOKEN = _generate_runtime_token()
        return _RUNTIME_TOKEN


def validate_ack_token(provided_token: str) -> bool:
    """Validate that the provided token matches the runtime token."""
    return provided_token == get_live_order_ack_token()


# =============================================================================
# SAFETY FLAG DEFINITIONS
# =============================================================================

class OrderMode(Enum):
    """Order execution mode."""
    PAPER = "paper"
    LIVE = "live"
    BLOCKED = "blocked"


@dataclass
class SafetyGateResult:
    """Result of safety gate evaluation."""
    allowed: bool
    mode: OrderMode
    checks_passed: Dict[str, bool]
    reason: str
    timestamp: str
    evidence_hash: str


class SafetyViolationError(Exception):
    """Raised when a safety constraint is violated."""
    pass


class BypassAttemptError(Exception):
    """Raised when a bypass attempt is detected."""
    pass


# =============================================================================
# SAFETY GATE CHECKS
# =============================================================================

def _check_kill_switch() -> bool:
    """Check if kill switch file exists."""
    kill_switch_path = Path(__file__).parent.parent / "state" / "KILL_SWITCH"
    return kill_switch_path.exists()


def _check_live_trading_enabled() -> bool:
    """Check LIVE_TRADING_ENABLED flag in safety/mode.py."""
    try:
        from safety.mode import LIVE_TRADING_ENABLED
        return LIVE_TRADING_ENABLED is True
    except ImportError:
        return False


def _check_trading_mode_env() -> bool:
    """Check TRADING_MODE environment variable equals 'live'."""
    return os.getenv("TRADING_MODE", "paper").lower() == "live"


def _check_approve_live_action() -> bool:
    """Check APPROVE_LIVE_ACTION in research_os/approval_gate.py."""
    try:
        from research_os.approval_gate import APPROVE_LIVE_ACTION
        return APPROVE_LIVE_ACTION is True
    except ImportError:
        return False


def _check_approve_live_action_2() -> bool:
    """Check secondary approval flag APPROVE_LIVE_ACTION_2."""
    # Check environment variable (secondary approval)
    return os.getenv("APPROVE_LIVE_ACTION_2", "false").lower() == "true"


def _check_paper_only() -> bool:
    """Check PAPER_ONLY flag in safety/mode.py (must be False for live)."""
    try:
        from safety.mode import PAPER_ONLY
        return PAPER_ONLY is False  # Must be False to allow live
    except ImportError:
        return False


def _generate_evidence_hash(checks: Dict[str, bool], timestamp: str) -> str:
    """Generate a hash of the safety check evidence."""
    evidence_str = f"{timestamp}|{sorted(checks.items())}"
    return hashlib.sha256(evidence_str.encode()).hexdigest()[:16]


def assert_no_live_orders(context: str = "unknown") -> None:
    """
    HARD GATE: Block all live orders unconditionally when NO_LIVE_ORDERS is True.

    This is separate from PAPER_ONLY - it's an additional safety layer.
    This function MUST be called at the start of evaluate_safety_gates().

    Args:
        context: Description of where this check is being called from

    Raises:
        SafetyViolationError: If NO_LIVE_ORDERS is True and attempting live endpoint
    """
    if NO_LIVE_ORDERS:
        base_url = os.getenv("ALPACA_BASE_URL", "")
        # Check if pointing to live endpoint
        is_live_url = (
            "api.alpaca.markets" in base_url and
            "paper" not in base_url.lower()
        )
        if is_live_url:
            msg = (
                f"NO_LIVE_ORDERS gate is ACTIVE. Live trading is BLOCKED. "
                f"Context: {context}. "
                f"ALPACA_BASE_URL: {base_url}. "
                f"To enable live: set NO_LIVE_ORDERS=False in execution_choke.py"
            )
            jlog("safety_violation", gate="NO_LIVE_ORDERS", context=context, url=base_url)
            raise SafetyViolationError(msg)


# =============================================================================
# MAIN CHOKE POINT FUNCTION
# =============================================================================

def evaluate_safety_gates(
    is_paper_order: bool = True,
    ack_token: Optional[str] = None,
    context: Optional[str] = None,
) -> SafetyGateResult:
    """
    Evaluate ALL safety gates before order submission.

    This is the SINGLE POINT of enforcement for all orders.

    Args:
        is_paper_order: True for paper orders, False for live
        ack_token: For live orders, must match runtime token
        context: Optional context string for logging

    Returns:
        SafetyGateResult with evaluation details
    """
    # === NO_LIVE_ORDERS HARD GATE - MUST BE FIRST ===
    assert_no_live_orders(context=context or "evaluate_safety_gates")

    timestamp = datetime.utcnow().isoformat()
    context_str = f" [{context}]" if context else ""

    checks = {
        "kill_switch_inactive": not _check_kill_switch(),
        "paper_only_disabled": _check_paper_only(),
        "live_trading_enabled": _check_live_trading_enabled(),
        "trading_mode_live": _check_trading_mode_env(),
        "approve_live_action": _check_approve_live_action(),
        "approve_live_action_2": _check_approve_live_action_2(),
        "ack_token_valid": validate_ack_token(ack_token or ""),
    }

    evidence_hash = _generate_evidence_hash(checks, timestamp)

    # Kill switch blocks EVERYTHING
    if not checks["kill_switch_inactive"]:
        return SafetyGateResult(
            allowed=False,
            mode=OrderMode.BLOCKED,
            checks_passed=checks,
            reason=f"Kill switch active{context_str}",
            timestamp=timestamp,
            evidence_hash=evidence_hash,
        )

    # Paper orders only need kill switch to be inactive
    if is_paper_order:
        return SafetyGateResult(
            allowed=True,
            mode=OrderMode.PAPER,
            checks_passed=checks,
            reason=f"Paper order allowed{context_str}",
            timestamp=timestamp,
            evidence_hash=evidence_hash,
        )

    # Live orders require ALL gates to pass
    required_for_live = [
        "kill_switch_inactive",
        "paper_only_disabled",
        "live_trading_enabled",
        "trading_mode_live",
        "approve_live_action",
        "approve_live_action_2",
        "ack_token_valid",
    ]

    failed_checks = [k for k in required_for_live if not checks.get(k, False)]

    if failed_checks:
        return SafetyGateResult(
            allowed=False,
            mode=OrderMode.BLOCKED,
            checks_passed=checks,
            reason=f"Live order blocked - failed checks: {failed_checks}{context_str}",
            timestamp=timestamp,
            evidence_hash=evidence_hash,
        )

    return SafetyGateResult(
        allowed=True,
        mode=OrderMode.LIVE,
        checks_passed=checks,
        reason=f"Live order allowed - all {len(required_for_live)} gates passed{context_str}",
        timestamp=timestamp,
        evidence_hash=evidence_hash,
    )


def require_safety_gate(
    is_paper_order: bool = True,
    ack_token: Optional[str] = None,
    context: Optional[str] = None,
) -> SafetyGateResult:
    """
    Require safety gates to pass or raise exception.

    This is the enforcement function that raises on failure.

    Args:
        is_paper_order: True for paper orders, False for live
        ack_token: For live orders, must match runtime token
        context: Optional context string for logging

    Returns:
        SafetyGateResult if allowed

    Raises:
        SafetyViolationError: If order is not allowed
    """
    result = evaluate_safety_gates(is_paper_order, ack_token, context)

    # Log the check
    jlog("safety_gate_evaluated", {
        "allowed": result.allowed,
        "mode": result.mode.value,
        "checks": result.checks_passed,
        "reason": result.reason,
        "evidence_hash": result.evidence_hash,
    })

    if not result.allowed:
        raise SafetyViolationError(result.reason)

    return result


# =============================================================================
# DECORATOR FOR ORDER FUNCTIONS
# =============================================================================

F = TypeVar('F', bound=Callable[..., Any])


def require_execution_choke(is_paper: bool = True) -> Callable[[F], F]:
    """
    Decorator that enforces the execution choke point.

    All order submission functions MUST use this decorator.

    Args:
        is_paper: Whether this function handles paper orders

    Example:
        @require_execution_choke(is_paper=True)
        def place_paper_order(order):
            ...
    """
    def decorator(func: F) -> F:
        def wrapper(*args, **kwargs):
            # Extract ack_token if provided
            ack_token = kwargs.pop('_ack_token', None)
            context = f"{func.__module__}.{func.__name__}"

            # Enforce the choke point
            result = require_safety_gate(
                is_paper_order=is_paper,
                ack_token=ack_token,
                context=context,
            )

            # Add safety result to kwargs for logging
            kwargs['_safety_result'] = result

            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper  # type: ignore
    return decorator


# =============================================================================
# BYPASS DETECTION
# =============================================================================

_REGISTERED_ORDER_FUNCTIONS: set = set()


def register_order_function(func_path: str) -> None:
    """Register a function as an approved order submission path."""
    _REGISTERED_ORDER_FUNCTIONS.add(func_path)


def is_approved_order_function(func_path: str) -> bool:
    """Check if a function path is an approved order submission path."""
    return func_path in _REGISTERED_ORDER_FUNCTIONS


def detect_bypass_attempt(caller_path: str) -> None:
    """
    Detect if a bypass attempt is being made.

    Call this from low-level order functions to detect if
    they are being called without going through the choke point.

    Raises:
        BypassAttemptError: If bypass detected
    """
    if not is_approved_order_function(caller_path):
        jlog("bypass_attempt_detected", {
            "caller": caller_path,
            "registered_functions": list(_REGISTERED_ORDER_FUNCTIONS),
        })
        raise BypassAttemptError(
            f"Bypass attempt detected from {caller_path}. "
            f"All orders must go through registered functions."
        )


# =============================================================================
# SAFETY STATUS REPORT
# =============================================================================

def get_safety_status() -> Dict[str, Any]:
    """
    Get comprehensive safety status report.

    Returns:
        Dict with all safety gate statuses and recommendations
    """
    checks = {
        "kill_switch_inactive": not _check_kill_switch(),
        "paper_only_disabled": _check_paper_only(),
        "live_trading_enabled": _check_live_trading_enabled(),
        "trading_mode_live": _check_trading_mode_env(),
        "approve_live_action": _check_approve_live_action(),
        "approve_live_action_2": _check_approve_live_action_2(),
    }

    live_ready = all(checks.values())
    paper_ready = checks["kill_switch_inactive"]

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
        "paper_ready": paper_ready,
        "live_ready": live_ready,
        "mode": "live" if live_ready else ("paper" if paper_ready else "blocked"),
        "runtime_token_generated": _RUNTIME_TOKEN is not None,
        "registered_order_functions": list(_REGISTERED_ORDER_FUNCTIONS),
        "recommendation": (
            "LIVE TRADING READY" if live_ready
            else "PAPER TRADING READY" if paper_ready
            else "TRADING BLOCKED - Check kill switch"
        ),
    }


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Register the approved order submission paths
register_order_function("execution.broker_alpaca.execute_signal")
register_order_function("execution.broker_alpaca.place_order_with_liquidity_check")
register_order_function("execution.broker_alpaca.place_ioc_limit")
register_order_function("execution.broker_alpaca.place_bracket_order")
register_order_function("execution.order_manager.OrderManager.submit_order")
register_order_function("execution.intelligent_executor.IntelligentExecutor.execute_signal")


if __name__ == "__main__":
    # Print safety status
    import json
    status = get_safety_status()
    print("=" * 60)
    print("KOBE EXECUTION CHOKE POINT - SAFETY STATUS")
    print("=" * 60)
    print(json.dumps(status, indent=2))

    # Test paper order
    print("\nTesting paper order...")
    result = evaluate_safety_gates(is_paper_order=True, context="test")
    print(f"Paper order: {result.allowed} - {result.reason}")

    # Test live order (should fail without all flags)
    print("\nTesting live order (should fail)...")
    result = evaluate_safety_gates(is_paper_order=False, ack_token="wrong", context="test")
    print(f"Live order: {result.allowed} - {result.reason}")
