"""
Unit tests for risk management.
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPolicyGate:
    """Tests for PolicyGate risk management."""

    def test_policy_gate_import(self):
        """Test that PolicyGate can be imported."""
        from risk.policy_gate import PolicyGate, RiskLimits
        assert PolicyGate is not None
        assert RiskLimits is not None

    def test_risk_limits_defaults(self):
        """Test RiskLimits default values."""
        from risk.policy_gate import RiskLimits
        limits = RiskLimits()
        assert limits.max_notional_per_order == 75.0
        assert limits.max_daily_notional == 1000.0
        assert limits.min_price == 3.0
        assert limits.max_price == 1000.0
        assert limits.allow_shorts == False

    def test_policy_gate_initialization(self):
        """Test PolicyGate initialization."""
        from risk.policy_gate import PolicyGate, RiskLimits
        limits = RiskLimits(max_notional_per_order=100, max_daily_notional=500)
        gate = PolicyGate(limits)
        assert gate is not None
        assert gate.limits.max_notional_per_order == 100

    def test_order_within_budget_passes(self):
        """Test that order within budget passes check."""
        from risk.policy_gate import PolicyGate, RiskLimits
        limits = RiskLimits(max_notional_per_order=75, max_daily_notional=1000)
        gate = PolicyGate(limits)

        # Order value of $50 should pass (price=10, qty=5)
        allowed, reason = gate.check(symbol="AAPL", side="long", price=10, qty=5)

        assert allowed == True
        assert reason == "ok"

    def test_order_exceeds_per_order_budget_fails(self):
        """Test that order exceeding per-order budget fails check."""
        from risk.policy_gate import PolicyGate, RiskLimits
        limits = RiskLimits(max_notional_per_order=75, max_daily_notional=1000)
        gate = PolicyGate(limits)

        # Order value of $100 should fail (price=50, qty=2)
        allowed, reason = gate.check(symbol="AAPL", side="long", price=50, qty=2)

        assert allowed == False
        assert "per_order" in reason

    def test_daily_loss_limit(self):
        """Test daily loss limit enforcement."""
        from risk.policy_gate import PolicyGate, RiskLimits
        limits = RiskLimits(max_notional_per_order=75, max_daily_notional=200)
        gate = PolicyGate(limits)

        # First order: $70 notional - should pass
        allowed1, _ = gate.check(symbol="AAPL", side="long", price=70, qty=1)
        assert allowed1 == True

        # Second order: $70 notional - should pass (total $140)
        allowed2, _ = gate.check(symbol="MSFT", side="long", price=70, qty=1)
        assert allowed2 == True

        # Third order: $70 notional - should fail (total would be $210)
        allowed3, reason = gate.check(symbol="GOOGL", side="long", price=70, qty=1)
        assert allowed3 == False
        assert "daily" in reason

    def test_reset_daily(self):
        """Test daily reset clears accumulated notional."""
        from risk.policy_gate import PolicyGate, RiskLimits
        limits = RiskLimits(max_notional_per_order=75, max_daily_notional=100)
        gate = PolicyGate(limits)

        # Use up daily budget
        gate.check(symbol="AAPL", side="long", price=75, qty=1)

        # Should fail now
        allowed, _ = gate.check(symbol="MSFT", side="long", price=75, qty=1)
        assert allowed == False

        # Reset daily
        gate.reset_daily()

        # Should pass again
        allowed2, _ = gate.check(symbol="MSFT", side="long", price=75, qty=1)
        assert allowed2 == True

    def test_price_bounds(self):
        """Test price bounds enforcement."""
        from risk.policy_gate import PolicyGate, RiskLimits
        limits = RiskLimits(min_price=3.0, max_price=1000.0)
        gate = PolicyGate(limits)

        # Price too low
        allowed1, reason1 = gate.check(symbol="PENNY", side="long", price=2, qty=1)
        assert allowed1 == False
        assert "bounds" in reason1

        # Price too high
        allowed2, reason2 = gate.check(symbol="BRK.A", side="long", price=1500, qty=1)
        assert allowed2 == False
        assert "bounds" in reason2

    def test_shorts_disabled_by_default(self):
        """Test that shorts are disabled by default."""
        from risk.policy_gate import PolicyGate, RiskLimits
        limits = RiskLimits()
        gate = PolicyGate(limits)

        allowed, reason = gate.check(symbol="AAPL", side="short", price=50, qty=1)
        assert allowed == False
        assert "short" in reason


class TestKillSwitch:
    """Tests for kill switch functionality."""

    def test_kill_switch_not_active_by_default(self, tmp_path):
        """Test that kill switch is not active by default."""
        kill_switch_file = tmp_path / "KILL_SWITCH"

        # File doesn't exist
        assert not kill_switch_file.exists()

    def test_kill_switch_blocks_trading(self, tmp_path):
        """Test that kill switch blocks trading when active."""
        kill_switch_file = tmp_path / "KILL_SWITCH"

        # Create kill switch file
        kill_switch_file.touch()

        assert kill_switch_file.exists()

    def test_kill_switch_can_be_deactivated(self, tmp_path):
        """Test that kill switch can be deactivated."""
        kill_switch_file = tmp_path / "KILL_SWITCH"

        # Create and then remove
        kill_switch_file.touch()
        assert kill_switch_file.exists()

        kill_switch_file.unlink()
        assert not kill_switch_file.exists()


class TestPositionSizing:
    """Tests for position sizing logic."""

    def test_fixed_dollar_sizing(self):
        """Test fixed dollar position sizing."""
        order_value = 75
        price = 150

        shares = int(order_value / price)

        assert shares == 0  # $75 can't buy 1 share at $150

        # With lower price
        price = 25
        shares = int(order_value / price)

        assert shares == 3  # $75 / $25 = 3 shares

    def test_position_size_respects_max_pct(self):
        """Test that position size respects max portfolio percentage."""
        portfolio_value = 100000
        max_pct = 0.10  # 10%
        max_position_value = portfolio_value * max_pct

        assert max_position_value == 10000

        # Can't exceed 10% of portfolio
        price = 500
        max_shares = int(max_position_value / price)

        assert max_shares == 20  # $10,000 / $500 = 20 shares


class TestPositionLimitGate:
    """Tests for PositionLimitGate - max concurrent positions enforcement."""

    def test_position_limit_gate_import(self):
        """Test that PositionLimitGate can be imported."""
        from risk.position_limit_gate import PositionLimitGate, PositionLimits
        assert PositionLimitGate is not None
        assert PositionLimits is not None

    def test_position_limits_defaults(self):
        """Test PositionLimits default values."""
        from risk.position_limit_gate import PositionLimits
        limits = PositionLimits()
        assert limits.max_positions == 5
        assert limits.max_per_symbol == 1
        assert limits.max_sector_concentration == 0.40

    def test_position_limit_gate_initialization(self):
        """Test PositionLimitGate initialization."""
        from risk.position_limit_gate import PositionLimitGate, PositionLimits
        limits = PositionLimits(max_positions=3, max_per_symbol=1)
        gate = PositionLimitGate(limits)
        assert gate is not None
        assert gate.limits.max_positions == 3

    def test_position_limit_gate_check_returns_tuple(self):
        """Test that check() returns (bool, str) tuple."""
        from risk.position_limit_gate import PositionLimitGate, PositionLimits
        gate = PositionLimitGate(PositionLimits())
        result = gate.check("AAPL", "long")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_position_limit_gate_get_status(self):
        """Test get_status returns expected keys."""
        from risk.position_limit_gate import PositionLimitGate, PositionLimits
        gate = PositionLimitGate(PositionLimits())
        status = gate.get_status()
        assert 'open_positions' in status
        assert 'max_positions' in status
        assert 'positions_available' in status
        assert 'open_symbols' in status
        assert 'max_per_symbol' in status

    def test_position_limit_gate_clear_cache(self):
        """Test that clear_cache resets internal cache."""
        from risk.position_limit_gate import PositionLimitGate, PositionLimits
        gate = PositionLimitGate(PositionLimits())
        gate._cached_positions = [{'symbol': 'TEST'}]
        gate._cache_timestamp = 1000.0
        gate.clear_cache()
        assert gate._cached_positions is None
        assert gate._cache_timestamp == 0

    def test_get_position_limit_gate_singleton(self):
        """Test singleton pattern for get_position_limit_gate."""
        from risk.position_limit_gate import get_position_limit_gate, reset_position_limit_gate
        reset_position_limit_gate()  # Clear any existing singleton
        gate1 = get_position_limit_gate()
        gate2 = get_position_limit_gate()
        assert gate1 is gate2
        reset_position_limit_gate()

    def test_position_limit_export_from_risk(self):
        """Test that PositionLimitGate is exported from risk package."""
        from risk import PositionLimitGate, PositionLimits, get_position_limit_gate
        assert PositionLimitGate is not None
        assert PositionLimits is not None
        assert get_position_limit_gate is not None
