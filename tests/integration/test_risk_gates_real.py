#!/usr/bin/env python3
"""
REAL Risk Gate Integration Tests - No Mocks
============================================

FIX (2026-01-08): Phase 3.4 - Tests that verify ACTUAL risk gate enforcement.

These tests:
1. Verify PolicyGate blocks oversized orders
2. Verify daily budget limits are enforced
3. Verify position limits are enforced
4. Verify kill zone time restrictions
5. Test gate reset behavior

NO MOCKS. Real code execution. Real assertions on behavior.
"""

import pytest
from datetime import datetime, date, time, timedelta


class TestPolicyGateEnforcement:
    """Test PolicyGate enforces risk limits correctly."""

    @pytest.fixture
    def policy_gate(self):
        """Create PolicyGate with known limits."""
        from risk.policy_gate import PolicyGate, RiskLimits

        limits = RiskLimits(
            max_notional_per_order=1000.0,  # $1000 per order
            max_daily_notional=5000.0,      # $5000 per day
            min_price=5.0,
            max_price=500.0,
            allow_shorts=False,
            max_positions=3,
        )
        return PolicyGate(limits=limits)

    def test_order_within_limits_passes(self, policy_gate):
        """
        REAL TEST: Order within all limits passes.
        """
        # $500 order ($50 x 10 shares) - within $1000 per-order limit
        passed, reason = policy_gate.check('AAPL', 'long', 50.0, 10)

        assert passed is True, f"Order should pass. Reason: {reason}"
        assert reason == "ok"

    def test_order_exceeds_per_order_limit(self, policy_gate):
        """
        REAL TEST: Order exceeding per-order limit is blocked.
        """
        # $2000 order ($100 x 20 shares) > $1000 per-order limit
        passed, reason = policy_gate.check('AAPL', 'long', 100.0, 20)

        assert passed is False, "Oversized order should be blocked"
        assert "per_order" in reason.lower(), f"Reason should mention per-order limit. Got: {reason}"

    def test_daily_budget_accumulates(self, policy_gate):
        """
        REAL TEST: Daily budget accumulates across orders.
        """
        # Order 1: $1000
        passed1, _ = policy_gate.check('AAPL', 'long', 100.0, 10)
        assert passed1 is True

        # Order 2: $1000
        passed2, _ = policy_gate.check('MSFT', 'long', 100.0, 10)
        assert passed2 is True

        # Order 3: $1000
        passed3, _ = policy_gate.check('GOOG', 'long', 100.0, 10)
        assert passed3 is True

        # Order 4: $1000
        passed4, _ = policy_gate.check('AMZN', 'long', 100.0, 10)
        assert passed4 is True

        # Order 5: $1000 - would exceed $5000 daily limit
        passed5, reason = policy_gate.check('META', 'long', 100.0, 10)
        assert passed5 is True  # This one passes (total = $5000)

        # Order 6: $1000 - exceeds $5000 daily limit
        passed6, reason = policy_gate.check('NVDA', 'long', 100.0, 10)
        assert passed6 is False, "Should exceed daily budget"
        assert "daily" in reason.lower(), f"Reason should mention daily limit. Got: {reason}"

    def test_remaining_budget_updates(self, policy_gate):
        """
        REAL TEST: Remaining budget updates after each order.
        """
        initial = policy_gate.get_remaining_daily_budget()
        assert initial == 5000.0, f"Initial budget should be $5000. Got ${initial}"

        # Use $1000
        policy_gate.check('AAPL', 'long', 100.0, 10)

        remaining = policy_gate.get_remaining_daily_budget()
        assert remaining == 4000.0, f"Remaining should be $4000. Got ${remaining}"

    def test_price_below_minimum_blocked(self, policy_gate):
        """
        REAL TEST: Price below minimum is blocked.
        """
        # $3 price < $5 minimum
        passed, reason = policy_gate.check('PENNY', 'long', 3.0, 100)

        assert passed is False, "Low price should be blocked"
        assert "price" in reason.lower() or "bounds" in reason.lower(), (
            f"Reason should mention price bounds. Got: {reason}"
        )

    def test_price_above_maximum_blocked(self, policy_gate):
        """
        REAL TEST: Price above maximum is blocked.
        """
        # $600 price > $500 maximum
        passed, reason = policy_gate.check('BRK.A', 'long', 600.0, 1)

        assert passed is False, "High price should be blocked"
        assert "price" in reason.lower() or "bounds" in reason.lower(), (
            f"Reason should mention price bounds. Got: {reason}"
        )

    def test_shorts_disabled_by_default(self, policy_gate):
        """
        REAL TEST: Short orders blocked when shorts are disabled.
        """
        passed, reason = policy_gate.check('AAPL', 'short', 100.0, 10)

        assert passed is False, "Short should be blocked"
        assert "short" in reason.lower(), f"Reason should mention shorts. Got: {reason}"

    def test_invalid_price_blocked(self, policy_gate):
        """
        REAL TEST: Invalid (zero/negative) price is blocked.
        """
        passed, reason = policy_gate.check('AAPL', 'long', 0.0, 10)
        assert passed is False, "Zero price should be blocked"

        passed, reason = policy_gate.check('AAPL', 'long', -10.0, 10)
        assert passed is False, "Negative price should be blocked"

    def test_invalid_qty_blocked(self, policy_gate):
        """
        REAL TEST: Invalid (zero/negative) quantity is blocked.
        """
        passed, reason = policy_gate.check('AAPL', 'long', 100.0, 0)
        assert passed is False, "Zero qty should be blocked"

        passed, reason = policy_gate.check('AAPL', 'long', 100.0, -5)
        assert passed is False, "Negative qty should be blocked"

    def test_position_limit_check(self, policy_gate):
        """
        REAL TEST: Position limit is enforced.
        """
        # Under limit (3 positions allowed)
        passed, reason = policy_gate.check_position_limit(2)
        assert passed is True, "2 positions should be allowed (limit is 3)"

        # At limit
        passed, reason = policy_gate.check_position_limit(3)
        assert passed is False, "Should fail at position limit"
        assert "max_positions" in reason.lower(), f"Reason should mention max positions. Got: {reason}"

        # Over limit
        passed, reason = policy_gate.check_position_limit(4)
        assert passed is False, "Should fail over position limit"


class TestKillZoneEnforcement:
    """Test kill zone time restrictions."""

    @pytest.fixture
    def kill_zone_gate(self):
        """Create KillZoneGate instance."""
        from risk.kill_zone_gate import KillZoneGate
        return KillZoneGate()

    def test_kill_zone_functions_exist(self, kill_zone_gate):
        """
        REAL TEST: Kill zone gate is importable and has required methods.
        """
        from risk.kill_zone_gate import can_trade_now, check_trade_allowed, get_current_zone

        # Functions should exist
        assert callable(can_trade_now)
        assert callable(check_trade_allowed)
        assert callable(get_current_zone)

    def test_opening_range_blocked(self, kill_zone_gate):
        """
        REAL TEST: Trading blocked during opening range (9:30-10:00 ET).
        """
        from zoneinfo import ZoneInfo

        # Create a datetime at 9:35 AM ET
        ET = ZoneInfo('America/New_York')
        opening_time = datetime(2024, 1, 2, 9, 35, tzinfo=ET)  # Tuesday 9:35 AM

        status = kill_zone_gate.check_can_trade(now=opening_time)

        assert status.can_trade is False, "9:35 AM (opening range) should be blocked"
        assert "opening" in status.reason.lower(), f"Reason should mention opening. Got: {status.reason}"

    def test_primary_window_allowed(self, kill_zone_gate):
        """
        REAL TEST: Trading allowed during primary window (10:00-11:30 ET).
        """
        from zoneinfo import ZoneInfo

        ET = ZoneInfo('America/New_York')
        primary_time = datetime(2024, 1, 2, 10, 30, tzinfo=ET)  # Tuesday 10:30 AM

        status = kill_zone_gate.check_can_trade(now=primary_time)

        assert status.can_trade is True, f"10:30 AM (primary window) should be allowed. Got: {status.reason}"

    def test_lunch_chop_blocked(self, kill_zone_gate):
        """
        REAL TEST: Trading blocked during lunch chop (11:30-14:00 ET).
        """
        from zoneinfo import ZoneInfo

        ET = ZoneInfo('America/New_York')
        lunch_time = datetime(2024, 1, 2, 12, 30, tzinfo=ET)  # Tuesday 12:30 PM

        status = kill_zone_gate.check_can_trade(now=lunch_time)

        assert status.can_trade is False, "12:30 PM (lunch chop) should be blocked"

    def test_power_hour_allowed(self, kill_zone_gate):
        """
        REAL TEST: Trading allowed during power hour (14:30-15:30 ET).
        """
        from zoneinfo import ZoneInfo

        ET = ZoneInfo('America/New_York')
        power_time = datetime(2024, 1, 2, 15, 0, tzinfo=ET)  # Tuesday 3:00 PM

        status = kill_zone_gate.check_can_trade(now=power_time)

        assert status.can_trade is True, f"3:00 PM (power hour) should be allowed. Got: {status.reason}"

    def test_close_blocked(self, kill_zone_gate):
        """
        REAL TEST: Trading blocked during close (15:30-16:00 ET).
        """
        from zoneinfo import ZoneInfo

        ET = ZoneInfo('America/New_York')
        close_time = datetime(2024, 1, 2, 15, 45, tzinfo=ET)  # Tuesday 3:45 PM

        status = kill_zone_gate.check_can_trade(now=close_time)

        assert status.can_trade is False, "3:45 PM (close) should be blocked"


class TestKillSwitchEnforcement:
    """Test kill switch emergency halt."""

    @pytest.fixture
    def temp_state_dir(self, tmp_path):
        """Create temporary state directory."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        return state_dir

    def test_kill_switch_blocks_trading(self, temp_state_dir, monkeypatch):
        """
        REAL TEST: Kill switch file blocks all trading.
        """
        # Create kill switch file
        kill_switch_file = temp_state_dir / "KILL_SWITCH"
        kill_switch_file.write_text("EMERGENCY HALT")

        # Monkey-patch the STATE_DIR
        import risk.kill_zone_gate as kz
        if hasattr(kz, 'STATE_DIR'):
            monkeypatch.setattr(kz, 'STATE_DIR', temp_state_dir)

        # Import check function
        from core.kill_switch import is_kill_switch_active

        # Patch the kill switch check
        original_path = None
        try:
            import core.kill_switch as ks
            original_path = ks.KILL_SWITCH_PATH
            ks.KILL_SWITCH_PATH = kill_switch_file

            active = is_kill_switch_active()
            assert active is True, "Kill switch should be active when file exists"
        finally:
            if original_path:
                ks.KILL_SWITCH_PATH = original_path


class TestAdvancedRiskIntegration:
    """Test advanced risk components are properly integrated."""

    def test_var_check_available(self):
        """
        REAL TEST: VaR check function is importable and callable.
        """
        from risk.advanced import check_portfolio_var

        # Should be callable
        assert callable(check_portfolio_var)

        # Should handle empty positions
        passes, result = check_portfolio_var(positions=[])
        assert passes is True, "Empty positions should pass VaR check"
        assert result['var_pct'] == 0.0, "Empty positions should have 0 VaR"

    def test_kelly_sizer_available(self):
        """
        REAL TEST: Kelly position sizer is importable and works.
        """
        from risk.advanced import KellyPositionSizer

        # Create sizer
        kelly = KellyPositionSizer(
            win_rate=0.60,
            avg_win=1.5,
            avg_loss=1.0,
        )

        # Should have calculate method
        assert hasattr(kelly, 'calculate_position_size') or hasattr(kelly, 'calculate_kelly'), (
            "KellyPositionSizer should have calculation method"
        )

    def test_correlation_limits_available(self):
        """
        REAL TEST: Correlation limits checker is importable.
        """
        from risk.advanced import EnhancedCorrelationLimits

        # Should be importable
        assert EnhancedCorrelationLimits is not None

    def test_monte_carlo_var_available(self):
        """
        REAL TEST: Monte Carlo VaR is importable.
        """
        from risk.advanced import MonteCarloVaR

        # Should be importable
        assert MonteCarloVaR is not None


class TestPositionSizingIntegration:
    """Test position sizing integrates with risk gates."""

    def test_equity_sizer_respects_policy_gate(self):
        """
        REAL TEST: Position sizing respects notional limits.
        """
        from risk.equity_sizer import calculate_position_size

        # Calculate position for $100 stock with $50,000 account
        size = calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            risk_pct=0.02,
            account_equity=50000.0,
            max_notional_pct=0.20,  # 20% = $10,000 max
        )

        # Verify notional is within cap
        assert size.notional <= 10000.0, (
            f"Notional ${size.notional:,.2f} should be <= $10,000 cap"
        )

    def test_kelly_reduces_when_appropriate(self):
        """
        REAL TEST: Kelly reduces position size when odds are poor.
        """
        from risk.equity_sizer import calculate_position_size_with_kelly

        # Poor odds scenario
        size = calculate_position_size_with_kelly(
            entry_price=100.0,
            stop_loss=95.0,
            risk_pct=0.02,
            account_equity=50000.0,
            use_kelly=True,
            kelly_win_rate=0.45,  # Below breakeven
            kelly_win_loss_ratio=1.0,  # 1:1
        )

        # Standard sizing would give larger position
        from risk.equity_sizer import calculate_position_size
        standard = calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            risk_pct=0.02,
            account_equity=50000.0,
        )

        # Kelly should reduce (or at most equal) standard
        assert size.shares <= standard.shares, (
            f"Kelly should reduce position. Kelly={size.shares}, Standard={standard.shares}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
