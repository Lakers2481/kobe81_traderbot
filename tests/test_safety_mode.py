"""
Tests for the safety module.

Verifies:
- PAPER_ONLY constant is True
- assert_paper_only() works correctly
- Kill switch detection works
- Trading mode detection works
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestSafetyMode:
    """Test safety/mode.py functionality."""

    def test_paper_only_is_true(self):
        """PAPER_ONLY must always be True."""
        from safety import PAPER_ONLY
        assert PAPER_ONLY is True, "PAPER_ONLY must be True"

    def test_get_trading_mode_returns_dict(self):
        """get_trading_mode() returns proper dict."""
        from safety import get_trading_mode

        mode = get_trading_mode()

        assert isinstance(mode, dict)
        assert "mode" in mode
        assert "mode_str" in mode
        assert "paper_only" in mode
        assert "kill_switch" in mode
        assert "timestamp" in mode

    def test_is_paper_mode_returns_bool(self):
        """is_paper_mode() returns boolean."""
        from safety import is_paper_mode

        result = is_paper_mode()
        assert isinstance(result, bool)

    def test_assert_paper_only_does_not_raise_normally(self):
        """assert_paper_only() should not raise when no kill switch."""
        from safety import assert_paper_only

        # Should not raise in normal operation
        # (unless kill switch is active)
        try:
            assert_paper_only()
        except Exception as e:
            # If kill switch is active, that's OK for this test
            if "kill switch" not in str(e).lower():
                raise

    def test_safety_module_exports(self):
        """Verify all expected exports from safety module."""
        from safety import (
            PAPER_ONLY,
            LIVE_TRADING_ENABLED,
            assert_paper_only,
            get_trading_mode,
            is_paper_mode,
            is_live_mode,
            SafetyViolationError,
            TradingMode,
        )

        # All should be importable
        assert PAPER_ONLY is not None
        assert LIVE_TRADING_ENABLED is not None
        assert callable(assert_paper_only)
        assert callable(get_trading_mode)
        assert callable(is_paper_mode)
        assert callable(is_live_mode)

    def test_trading_mode_enum_values(self):
        """TradingMode enum has expected values."""
        from safety import TradingMode

        assert TradingMode.PAPER.value == "paper"
        assert TradingMode.LIVE.value == "live"
        assert TradingMode.DISABLED.value == "disabled"


class TestKillSwitch:
    """Test kill switch functionality."""

    def test_kill_switch_detection(self):
        """Kill switch path is correct."""
        from safety.mode import KILL_SWITCH_PATH

        # Path should be in state directory
        assert "state" in str(KILL_SWITCH_PATH)
        assert "KILL_SWITCH" in str(KILL_SWITCH_PATH)

    def test_kill_switch_check_function(self):
        """_check_kill_switch function exists and works."""
        from safety.mode import _check_kill_switch

        result = _check_kill_switch()
        assert isinstance(result, bool)


class TestValidateOrderAllowed:
    """Test order validation function."""

    def test_validate_order_allowed_returns_dict(self):
        """validate_order_allowed returns proper dict."""
        from safety.mode import validate_order_allowed

        result = validate_order_allowed(
            symbol="AAPL",
            side="buy",
            quantity=100,
            is_paper=True
        )

        assert isinstance(result, dict)
        assert "allowed" in result
        assert "reason" in result
        assert "mode" in result

    def test_paper_orders_allowed(self):
        """Paper orders should be allowed (when no kill switch)."""
        from safety.mode import validate_order_allowed, _check_kill_switch

        result = validate_order_allowed(
            symbol="AAPL",
            side="buy",
            quantity=100,
            is_paper=True
        )

        # If kill switch is not active, paper should be allowed
        if not _check_kill_switch():
            assert result["allowed"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
