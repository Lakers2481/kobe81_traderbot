"""
Tests for the unified exception hierarchy.

This module tests core/exceptions.py which provides the standard
exception hierarchy for the entire Kobe trading system.
"""

import pytest
from datetime import datetime

from core.exceptions import (
    # Base exception
    QuantSystemError,
    # Safety exceptions (non-recoverable)
    SafetyError,
    KillSwitchActiveError,
    SafetyViolationError,
    LiveTradingBlockedError,
    BypassAttemptError,
    KillZoneViolationError,
    # Execution exceptions
    ExecutionError,
    PolicyGateError,
    ComplianceError,
    PortfolioRiskError,
    CircuitBreakerError,
    InvalidTransitionError,
    LiquidityError,
    SlippageError,
    # Data exceptions
    DataError,
    DataFetchError,
    DataValidationError,
    FakeDataError,
    LookaheadBiasError,
    SurvivorshipBiasError,
    # Configuration exceptions
    ConfigurationError,
    SettingsValidationError,
    MissingConfigError,
    FrozenParamsError,
    # System exceptions
    SystemError,
    UnsafePathError,
    LockError,
    ReconciliationError,
    StateCorruptionError,
    # Research exceptions
    ResearchError,
    ApprovalGateError,
    ExperimentError,
    ReproducibilityError,
    # Helper functions
    is_safety_critical,
    get_error_code,
)


class TestQuantSystemError:
    """Tests for the base exception class."""

    def test_basic_creation(self):
        """Test basic exception creation."""
        error = QuantSystemError("Test error")
        assert str(error) == "[SYSTEM_ERROR] Test error"
        assert error.message == "Test error"
        assert error.error_code == "SYSTEM_ERROR"
        assert error.is_recoverable is True
        assert error.context == {}
        assert error.cause is None
        assert isinstance(error.timestamp, datetime)

    def test_with_context(self):
        """Test exception with context dictionary."""
        error = QuantSystemError(
            "Test error",
            context={"symbol": "AAPL", "qty": 100}
        )
        assert "symbol=AAPL" in str(error)
        assert "qty=100" in str(error)
        assert error.context["symbol"] == "AAPL"

    def test_with_cause(self):
        """Test exception with cause (chained exception)."""
        original = ValueError("Original error")
        error = QuantSystemError("Wrapped error", cause=original)
        assert error.cause is original
        assert str(error.cause) == "Original error"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        error = QuantSystemError(
            "Test error",
            context={"symbol": "AAPL"}
        )
        d = error.to_dict()
        assert d["error_code"] == "SYSTEM_ERROR"
        assert d["message"] == "Test error"
        assert d["is_recoverable"] is True
        assert d["context"]["symbol"] == "AAPL"
        assert "timestamp" in d
        assert d["cause"] is None


class TestSafetyErrors:
    """Tests for safety-critical exceptions."""

    def test_safety_error_is_not_recoverable(self):
        """SafetyError should always be non-recoverable."""
        error = SafetyError("Safety issue")
        assert error.is_recoverable is False
        assert error.error_code == "SAFETY_ERROR"

    def test_kill_switch_active_error(self):
        """Test KillSwitchActiveError."""
        error = KillSwitchActiveError("Kill switch activated due to drawdown")
        assert error.error_code == "KILL_SWITCH_ACTIVE"
        assert error.is_recoverable is False
        assert isinstance(error, SafetyError)
        assert isinstance(error, QuantSystemError)

    def test_live_trading_blocked_error(self):
        """Test LiveTradingBlockedError."""
        error = LiveTradingBlockedError("PAPER_ONLY is enforced")
        assert error.error_code == "LIVE_BLOCKED"
        assert error.is_recoverable is False

    def test_bypass_attempt_error(self):
        """Test BypassAttemptError for security events."""
        error = BypassAttemptError("Attempt to bypass policy gate")
        assert error.error_code == "BYPASS_ATTEMPT"
        assert error.is_recoverable is False

    def test_safety_inheritance(self):
        """All safety errors should inherit from SafetyError."""
        safety_classes = [
            KillSwitchActiveError,
            SafetyViolationError,
            LiveTradingBlockedError,
            BypassAttemptError,
            KillZoneViolationError,
        ]
        for cls in safety_classes:
            error = cls("Test")
            assert isinstance(error, SafetyError)
            assert error.is_recoverable is False


class TestExecutionErrors:
    """Tests for execution-related exceptions."""

    def test_execution_error_default_recoverable(self):
        """ExecutionError should be recoverable by default."""
        error = ExecutionError("Order failed")
        assert error.is_recoverable is True
        assert error.error_code == "EXECUTION_ERROR"

    def test_policy_gate_error_not_recoverable(self):
        """PolicyGateError is NOT recoverable (hard block)."""
        error = PolicyGateError("Notional exceeds $75 limit")
        assert error.error_code == "POLICY_GATE_REJECT"
        assert error.is_recoverable is False

    def test_circuit_breaker_error(self):
        """Test CircuitBreakerError."""
        error = CircuitBreakerError(
            "Drawdown circuit breaker triggered",
            context={"drawdown_pct": 8.5, "threshold": 5.0}
        )
        assert error.error_code == "CIRCUIT_BREAKER_OPEN"
        assert error.is_recoverable is False
        assert error.context["drawdown_pct"] == 8.5

    def test_invalid_transition_error(self):
        """Test InvalidTransitionError for order state machine."""
        error = InvalidTransitionError(
            "Cannot transition from FILLED to PENDING",
            context={"from_state": "FILLED", "to_state": "PENDING"}
        )
        assert error.error_code == "INVALID_STATE_TRANSITION"


class TestDataErrors:
    """Tests for data-related exceptions."""

    def test_data_fetch_error(self):
        """Test DataFetchError."""
        error = DataFetchError(
            "Polygon API rate limit exceeded",
            context={"provider": "polygon", "symbol": "AAPL"}
        )
        assert error.error_code == "DATA_FETCH_FAIL"
        assert error.is_recoverable is True

    def test_fake_data_error_not_recoverable(self):
        """FakeDataError indicates data integrity compromise."""
        error = FakeDataError("Zero volume with price movement detected")
        assert error.error_code == "FAKE_DATA_DETECTED"
        assert error.is_recoverable is False

    def test_lookahead_bias_error(self):
        """LookaheadBiasError is critical for backtest validity."""
        error = LookaheadBiasError("Future data used in historical decision")
        assert error.error_code == "LOOKAHEAD_BIAS"
        assert error.is_recoverable is False


class TestConfigurationErrors:
    """Tests for configuration-related exceptions."""

    def test_frozen_params_error(self):
        """Frozen parameters cannot be modified."""
        error = FrozenParamsError("Cannot modify frozen_strategy_params_v2.3.json")
        assert error.error_code == "FROZEN_PARAMS_LOCKED"
        assert error.is_recoverable is False

    def test_settings_validation_error(self):
        """Test SettingsValidationError for Pydantic failures."""
        error = SettingsValidationError(
            "Invalid settings",
            context={"field": "max_position_size", "error": "must be positive"}
        )
        assert error.error_code == "SETTINGS_INVALID"


class TestSystemErrors:
    """Tests for general system exceptions."""

    def test_reconciliation_error(self):
        """Test ReconciliationError for broker mismatch."""
        error = ReconciliationError(
            "Position mismatch: local=100, broker=90",
            context={"symbol": "AAPL", "local_qty": 100, "broker_qty": 90}
        )
        assert error.error_code == "RECONCILIATION_MISMATCH"

    def test_state_corruption_error(self):
        """StateCorruptionError is not recoverable."""
        error = StateCorruptionError("Hash chain verification failed")
        assert error.error_code == "STATE_CORRUPTED"
        assert error.is_recoverable is False


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_safety_critical_true(self):
        """Safety errors are safety-critical."""
        assert is_safety_critical(KillSwitchActiveError("test")) is True
        assert is_safety_critical(SafetyViolationError("test")) is True
        assert is_safety_critical(FakeDataError("test")) is True
        assert is_safety_critical(StateCorruptionError("test")) is True

    def test_is_safety_critical_false(self):
        """Recoverable errors are not safety-critical."""
        assert is_safety_critical(DataFetchError("test")) is False
        assert is_safety_critical(ExecutionError("test")) is False
        assert is_safety_critical(QuantSystemError("test")) is False

    def test_is_safety_critical_non_kobe_exception(self):
        """Non-Kobe exceptions are not safety-critical."""
        assert is_safety_critical(ValueError("test")) is False
        assert is_safety_critical(RuntimeError("test")) is False

    def test_get_error_code(self):
        """get_error_code returns correct codes."""
        assert get_error_code(KillSwitchActiveError("test")) == "KILL_SWITCH_ACTIVE"
        assert get_error_code(PolicyGateError("test")) == "POLICY_GATE_REJECT"
        assert get_error_code(DataFetchError("test")) == "DATA_FETCH_FAIL"

    def test_get_error_code_unknown(self):
        """Non-Kobe exceptions return UNKNOWN."""
        assert get_error_code(ValueError("test")) == "UNKNOWN"
        assert get_error_code(RuntimeError("test")) == "UNKNOWN"


class TestExceptionCatching:
    """Tests for exception catching behavior."""

    def test_catch_all_with_quant_system_error(self):
        """QuantSystemError should catch all Kobe exceptions."""
        exceptions_to_test = [
            KillSwitchActiveError("test"),
            PolicyGateError("test"),
            DataFetchError("test"),
            FrozenParamsError("test"),
            ReconciliationError("test"),
            ExperimentError("test"),
        ]
        for exc in exceptions_to_test:
            try:
                raise exc
            except QuantSystemError as e:
                assert True
            except Exception:
                pytest.fail(f"{type(exc).__name__} not caught by QuantSystemError")

    def test_catch_safety_error(self):
        """SafetyError catches all safety-critical exceptions."""
        safety_exceptions = [
            KillSwitchActiveError("test"),
            SafetyViolationError("test"),
            LiveTradingBlockedError("test"),
            BypassAttemptError("test"),
            KillZoneViolationError("test"),
        ]
        for exc in safety_exceptions:
            try:
                raise exc
            except SafetyError as e:
                assert e.is_recoverable is False
            except Exception:
                pytest.fail(f"{type(exc).__name__} not caught by SafetyError")

    def test_catch_execution_error(self):
        """ExecutionError catches all execution-related exceptions."""
        execution_exceptions = [
            PolicyGateError("test"),
            ComplianceError("test"),
            CircuitBreakerError("test"),
            InvalidTransitionError("test"),
            LiquidityError("test"),
            SlippageError("test"),
        ]
        for exc in execution_exceptions:
            try:
                raise exc
            except ExecutionError as e:
                pass
            except Exception:
                pytest.fail(f"{type(exc).__name__} not caught by ExecutionError")


class TestBackwardCompatibility:
    """Tests for backward compatibility with old imports."""

    def test_kill_switch_import_from_kill_switch(self):
        """KillSwitchActiveError can still be imported from kill_switch."""
        from core.kill_switch import KillSwitchActiveError as KS_Error
        error = KS_Error("test")
        assert isinstance(error, QuantSystemError)
        assert error.is_recoverable is False

    def test_paper_guard_import(self):
        """Exceptions can still be imported from paper_guard."""
        from safety.paper_guard import KillSwitchActiveError, LiveTradingBlockedError
        assert KillSwitchActiveError("test").error_code == "KILL_SWITCH_ACTIVE"
        assert LiveTradingBlockedError("test").error_code == "LIVE_BLOCKED"
