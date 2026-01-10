"""
Unified Exception Hierarchy for Kobe Trading System.

This module provides a structured exception hierarchy for the entire system.
All exceptions inherit from QuantSystemError, enabling consistent error handling.

Usage:
    from core.exceptions import QuantSystemError, SafetyError, ExecutionError

    try:
        execute_trade()
    except SafetyError as e:
        # Handle safety-critical errors (non-recoverable)
        halt_trading(e.error_code, e.context)
    except ExecutionError as e:
        # Handle execution errors (may be recoverable)
        log_and_retry(e)
    except QuantSystemError as e:
        # Catch-all for system errors
        log_error(e)

Blueprint Alignment:
    This module implements the unified exception hierarchy required by the
    production-grade trading system blueprint (Section 2.1).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


# =============================================================================
# BASE EXCEPTION
# =============================================================================

class QuantSystemError(Exception):
    """
    Base exception for all Kobe trading system errors.

    All custom exceptions in the system should inherit from this class,
    enabling consistent error handling and monitoring.

    Attributes:
        error_code: Unique identifier for this error type
        is_recoverable: Whether the system can attempt recovery
        context: Additional context about the error
        timestamp: When the error occurred
    """
    error_code: str = "SYSTEM_ERROR"
    is_recoverable: bool = True

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        self.message = message
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()
        super().__init__(message)

    def __str__(self) -> str:
        base = f"[{self.error_code}] {self.message}"
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base = f"{base} ({ctx_str})"
        return base

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "is_recoverable": self.is_recoverable,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
        }


# =============================================================================
# SAFETY ERRORS (Non-recoverable - Trading MUST stop)
# =============================================================================

class SafetyError(QuantSystemError):
    """
    Base class for safety-critical errors.

    When raised, trading operations MUST halt immediately.
    These errors indicate potential for financial loss or system compromise.
    """
    error_code = "SAFETY_ERROR"
    is_recoverable = False


class KillSwitchActiveError(SafetyError):
    """
    Raised when an operation is blocked by an active kill switch.

    The kill switch is the primary emergency stop mechanism.
    All order submissions MUST be blocked when active.
    """
    error_code = "KILL_SWITCH_ACTIVE"

    def __init__(
        self,
        reason: str = "Kill switch is active",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(reason, context)


class SafetyViolationError(SafetyError):
    """
    Raised when a safety constraint is violated.

    Examples:
    - Attempting live trading when PAPER_ONLY is set
    - Bypassing execution guards
    - Unauthorized mode changes
    """
    error_code = "SAFETY_VIOLATION"


class LiveTradingBlockedError(SafetyError):
    """
    Raised when live trading is attempted but blocked.

    This error is raised when:
    - APPROVE_LIVE_ACTION is False
    - Paper mode is enforced
    - Live broker endpoint detected without authorization
    """
    error_code = "LIVE_BLOCKED"


class BypassAttemptError(SafetyError):
    """
    Raised when an attempt to bypass safety mechanisms is detected.

    This is a CRITICAL security event that should be logged and investigated.
    """
    error_code = "BYPASS_ATTEMPT"


class KillZoneViolationError(SafetyError):
    """
    Raised when trading is attempted outside allowed time windows.

    Kill zones (e.g., 9:30-10:00 opening range) are enforced periods
    where new entries are blocked.
    """
    error_code = "KILL_ZONE_VIOLATION"


# =============================================================================
# EXECUTION ERRORS
# =============================================================================

class ExecutionError(QuantSystemError):
    """
    Base class for order execution errors.

    These errors occur during the order lifecycle and may or may not
    be recoverable depending on the specific situation.
    """
    error_code = "EXECUTION_ERROR"


class PolicyGateError(ExecutionError):
    """
    Raised when an order is rejected by the policy gate.

    The policy gate enforces:
    - Maximum notional per order ($75 default)
    - Daily notional budget ($1,000 default)
    - Price bounds
    - Position limits
    """
    error_code = "POLICY_GATE_REJECT"
    is_recoverable = False


class ComplianceError(ExecutionError):
    """
    Raised when an order violates compliance rules.

    Compliance rules include:
    - Prohibited securities list
    - Legislative restrictions
    - Corporate action exclusions
    """
    error_code = "COMPLIANCE_REJECT"
    is_recoverable = False


class PortfolioRiskError(ExecutionError):
    """
    Raised when an order would breach portfolio-level risk limits.

    Examples:
    - Exceeding maximum portfolio heat
    - Correlation limits breached
    - VaR limits exceeded
    """
    error_code = "PORTFOLIO_RISK_BREACH"
    is_recoverable = False


class CircuitBreakerError(ExecutionError):
    """
    Raised when a circuit breaker prevents execution.

    Circuit breakers monitor:
    - Drawdown (daily, weekly, max)
    - Volatility spikes
    - Consecutive losses (streak)
    - Execution quality (slippage)
    - Correlation concentration
    """
    error_code = "CIRCUIT_BREAKER_OPEN"
    is_recoverable = False


class InvalidTransitionError(ExecutionError):
    """
    Raised when an invalid order state transition is attempted.

    The order state machine enforces valid transitions:
    PENDING -> SUBMITTED -> FILLED (valid)
    FILLED -> PENDING (invalid - raises this error)
    """
    error_code = "INVALID_STATE_TRANSITION"


class LiquidityError(ExecutionError):
    """
    Raised when liquidity requirements are not met.

    Checks:
    - Minimum ADV (Average Daily Volume)
    - Maximum bid-ask spread
    - Order size relative to ADV
    """
    error_code = "LIQUIDITY_INSUFFICIENT"


class SlippageError(ExecutionError):
    """
    Raised when slippage exceeds acceptable thresholds.
    """
    error_code = "SLIPPAGE_EXCEEDED"


# =============================================================================
# DATA ERRORS
# =============================================================================

class DataError(QuantSystemError):
    """
    Base class for data-related errors.

    These errors occur during data fetching, validation, or processing.
    """
    error_code = "DATA_ERROR"


class DataFetchError(DataError):
    """
    Raised when data fetching fails.

    This can occur when:
    - API rate limits are exceeded
    - Network connectivity issues
    - Provider returns empty/invalid data
    - All fallback sources fail
    """
    error_code = "DATA_FETCH_FAIL"


class DataValidationError(DataError):
    """
    Raised when data fails validation checks.

    Validation includes:
    - OHLC relationships (high >= max(open, close))
    - Null/NaN checks
    - Timestamp uniqueness
    - Price range sanity checks
    """
    error_code = "DATA_VALIDATION_FAIL"


class FakeDataError(DataError):
    """
    Raised when fake or manipulated data is detected.

    Detection methods:
    - Impossible price patterns
    - Zero volume with price movement
    - Identical consecutive bars
    - Statistical anomalies

    This is NON-RECOVERABLE as it indicates data integrity compromise.
    """
    error_code = "FAKE_DATA_DETECTED"
    is_recoverable = False


class LookaheadBiasError(DataError):
    """
    Raised when lookahead bias is detected in data processing.

    Lookahead bias occurs when future data is used to make historical decisions.
    This is a critical error that invalidates backtest results.
    """
    error_code = "LOOKAHEAD_BIAS"
    is_recoverable = False


class SurvivorshipBiasError(DataError):
    """
    Raised when survivorship bias is detected.

    Survivorship bias occurs when only currently-existing securities
    are included in historical analysis.
    """
    error_code = "SURVIVORSHIP_BIAS"


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================

class ConfigurationError(QuantSystemError):
    """
    Base class for configuration-related errors.
    """
    error_code = "CONFIG_ERROR"


class SettingsValidationError(ConfigurationError):
    """
    Raised when settings fail schema validation.

    Uses Pydantic validation under the hood.
    """
    error_code = "SETTINGS_INVALID"


class MissingConfigError(ConfigurationError):
    """
    Raised when required configuration is missing.
    """
    error_code = "CONFIG_MISSING"


class FrozenParamsError(ConfigurationError):
    """
    Raised when attempting to modify frozen strategy parameters.
    """
    error_code = "FROZEN_PARAMS_LOCKED"
    is_recoverable = False


# =============================================================================
# SYSTEM ERRORS
# =============================================================================

class InfrastructureError(QuantSystemError):
    """
    Base class for general system/infrastructure errors.

    Note: Named InfrastructureError (not SystemError) to avoid
    shadowing Python's built-in SystemError exception.
    """
    error_code = "INFRASTRUCTURE_ERROR"


class UnsafePathError(InfrastructureError):
    """
    Raised when an unsafe file path is detected.

    Used by safe_pickle to prevent path traversal attacks.
    """
    error_code = "UNSAFE_PATH"
    is_recoverable = False


class LockError(InfrastructureError):
    """
    Raised when a lock cannot be acquired.
    """
    error_code = "LOCK_FAILED"


class ReconciliationError(InfrastructureError):
    """
    Raised when broker-local reconciliation fails.

    Indicates mismatch between:
    - Local position state vs broker positions
    - Local order state vs broker orders
    - Local cash vs broker cash
    """
    error_code = "RECONCILIATION_MISMATCH"


class StateCorruptionError(InfrastructureError):
    """
    Raised when system state is corrupted.
    """
    error_code = "STATE_CORRUPTED"
    is_recoverable = False


# =============================================================================
# RESEARCH/EXPERIMENT ERRORS
# =============================================================================

class ResearchError(QuantSystemError):
    """
    Base class for research and experiment errors.
    """
    error_code = "RESEARCH_ERROR"


class ApprovalGateError(ResearchError):
    """
    Raised when a research proposal is rejected by the approval gate.
    """
    error_code = "APPROVAL_GATE_REJECT"


class ExperimentError(ResearchError):
    """
    Raised when an experiment fails.
    """
    error_code = "EXPERIMENT_FAILED"


class ReproducibilityError(ResearchError):
    """
    Raised when results cannot be reproduced.
    """
    error_code = "NOT_REPRODUCIBLE"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_safety_critical(error: Exception) -> bool:
    """
    Check if an error is safety-critical (non-recoverable).

    Args:
        error: The exception to check

    Returns:
        True if the error is safety-critical and trading should halt
    """
    if isinstance(error, QuantSystemError):
        return not error.is_recoverable
    return False


def get_error_code(error: Exception) -> str:
    """
    Get the error code for an exception.

    Args:
        error: The exception to get the code for

    Returns:
        Error code string, or "UNKNOWN" for non-Kobe exceptions
    """
    if isinstance(error, QuantSystemError):
        return error.error_code
    return "UNKNOWN"


# =============================================================================
# EXCEPTION MAPPING (for backward compatibility)
# =============================================================================

# Map old exception names to new ones for gradual migration
_EXCEPTION_ALIASES = {
    "CircuitOpenError": CircuitBreakerError,
}
