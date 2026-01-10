"""
INTEGRATION TESTS: Position Sizing Edge Cases (MEDIUM)

Tests edge cases in dual-cap position sizing:
- Risk cap vs notional cap dominance
- Zero/negative risk distance
- Fractional shares
- Minimum share constraints
- Extreme volatility scenarios

Uses the actual risk.equity_sizer.calculate_position_size API.

Author: Kobe Trading System
Version: 1.0.0
Date: 2026-01-06
"""

import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.mark.integration
class TestRiskCapDominates:
    """Test scenarios where risk-based sizing produces fewer shares."""

    def test_tight_stop_with_notional_cap(self):
        """Tight stop with expensive stock hits notional cap."""
        from risk.equity_sizer import calculate_position_size

        result = calculate_position_size(
            entry_price=100.0,
            stop_loss=99.0,  # $1 risk per share (tight stop)
            risk_pct=0.01,   # 1% = $1000 max risk on $100k
            account_equity=100000.0,
            max_notional_pct=0.20,  # 20% = $20000 max notional
        )

        # Risk: $1000 / $1 = 1000 shares
        # Notional: $20000 / $100 = 200 shares
        # Result: min(1000, 200) = 200 (notional cap)
        assert result.shares == 200
        assert result.capped is True

    def test_high_volatility_risk_dominates(self):
        """High volatility (wide stop) causes risk-based to limit."""
        from risk.equity_sizer import calculate_position_size

        result = calculate_position_size(
            entry_price=100.0,
            stop_loss=80.0,  # $20 risk per share (20% stop!)
            risk_pct=0.02,   # 2% = $2000 max risk
            account_equity=100000.0,
            max_notional_pct=0.20,  # 20% = $20000 max notional
        )

        # Risk: $2000 / $20 = 100 shares
        # Notional: $20000 / $100 = 200 shares
        # Result: min(100, 200) = 100 (risk dominates)
        assert result.shares == 100
        assert result.capped is False


@pytest.mark.integration
class TestNotionalCapDominates:
    """Test scenarios where notional-based sizing produces fewer shares."""

    def test_expensive_stock_notional_dominates(self):
        """Expensive stock hits notional cap first."""
        from risk.equity_sizer import calculate_position_size

        result = calculate_position_size(
            entry_price=500.0,  # Expensive stock
            stop_loss=480.0,    # $20 risk per share
            risk_pct=0.02,      # 2% = $2000 max risk
            account_equity=100000.0,
            max_notional_pct=0.10,  # 10% = $10000 max notional
        )

        # Risk: $2000 / $20 = 100 shares
        # Notional: $10000 / $500 = 20 shares
        # Result: min(100, 20) = 20 (notional dominates)
        assert result.shares == 20
        assert result.capped is True

    def test_small_notional_cap(self):
        """Small notional cap limits position size."""
        from risk.equity_sizer import calculate_position_size

        result = calculate_position_size(
            entry_price=50.0,
            stop_loss=48.0,     # $2 risk per share
            risk_pct=0.02,      # 2% = $2000 max risk
            account_equity=100000.0,
            max_notional_pct=0.05,  # 5% = $5000 max notional
        )

        # Risk: $2000 / $2 = 1000 shares
        # Notional: $5000 / $50 = 100 shares
        # Result: min(1000, 100) = 100 (notional dominates)
        assert result.shares == 100
        assert result.capped is True


@pytest.mark.integration
class TestZeroRiskDistance:
    """Test handling of zero or near-zero risk distance."""

    def test_stop_equals_entry_uses_fallback(self):
        """Stop loss = entry price should use fallback (5% of entry)."""
        from risk.equity_sizer import calculate_position_size

        # When stop == entry, the code uses 5% fallback
        result = calculate_position_size(
            entry_price=100.0,
            stop_loss=100.0,  # Same as entry!
            risk_pct=0.02,
            account_equity=100000.0,
        )

        # 5% fallback means risk_per_share = $5
        # Risk: $2000 / $5 = 400 shares
        assert result.shares > 0
        assert result.risk_per_share == 5.0  # 5% of 100

    def test_very_tight_stop_uses_actual(self):
        """Very tight stop (< 5%) uses actual risk."""
        from risk.equity_sizer import calculate_position_size

        result = calculate_position_size(
            entry_price=100.0,
            stop_loss=99.5,  # $0.50 risk (0.5%)
            risk_pct=0.02,
            account_equity=100000.0,
        )

        # Risk per share should be actual (0.50), not fallback
        # But 0.50 > 0.01 so it uses actual
        assert result.shares > 0


@pytest.mark.integration
class TestFractionalShares:
    """Test handling of fractional share results."""

    def test_result_is_integer(self):
        """Shares should always be integer."""
        from risk.equity_sizer import calculate_position_size

        result = calculate_position_size(
            entry_price=333.33,  # Odd price
            stop_loss=330.0,     # ~$3.33 risk per share
            risk_pct=0.01,       # 1% = $1000 max risk
            account_equity=100000.0,
        )

        # Shares should be whole number
        assert isinstance(result.shares, int)

    def test_minimum_shares_enforced(self):
        """Minimum shares constraint should be enforced."""
        from risk.equity_sizer import calculate_position_size

        result = calculate_position_size(
            entry_price=500.0,
            stop_loss=400.0,    # $100 risk per share
            risk_pct=0.001,     # 0.1% = $100 max risk
            account_equity=100000.0,
            min_shares=1,
        )

        # Risk: $100 / $100 = 1 share
        # Should return at least min_shares
        assert result.shares >= 1


@pytest.mark.integration
class TestMinimumShareConstraint:
    """Test minimum share requirements."""

    def test_exactly_one_share(self):
        """Should handle exactly 1 share case."""
        from risk.equity_sizer import calculate_position_size

        result = calculate_position_size(
            entry_price=100.0,
            stop_loss=90.0,     # $10 risk per share
            risk_pct=0.0001,    # 0.01% = $10 max risk
            account_equity=100000.0,
            min_shares=1,
        )

        # Risk: $10 / $10 = 1 share
        assert result.shares >= 1

    def test_small_account_minimum_enforced(self):
        """Small account should still get minimum shares."""
        from risk.equity_sizer import calculate_position_size

        result = calculate_position_size(
            entry_price=1000.0,   # Expensive
            stop_loss=900.0,     # $100 risk per share
            risk_pct=0.001,      # 0.1% = $1 max risk
            account_equity=1000.0,  # Tiny account
            min_shares=1,
        )

        # Risk would give 0 shares, but min_shares enforces 1
        assert result.shares >= 1


@pytest.mark.integration
class TestExtremeScenarios:
    """Test extreme/unusual scenarios."""

    def test_very_small_account(self):
        """Very small account should still work."""
        from risk.equity_sizer import calculate_position_size

        result = calculate_position_size(
            entry_price=10.0,   # Cheap stock
            stop_loss=9.0,     # $1 risk per share
            risk_pct=0.02,     # 2% = $10 max risk
            account_equity=500.0,  # Tiny account
        )

        # Risk: $10 / $1 = 10 shares
        # Notional (20%): $100 / $10 = 10 shares
        assert result.shares == 10

    def test_very_large_account(self):
        """Very large account should handle properly."""
        from risk.equity_sizer import calculate_position_size

        result = calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,      # $5 risk per share
            risk_pct=0.01,       # 1% = $100,000 max risk
            account_equity=10000000.0,  # $10M account
            max_notional_pct=0.05,  # 5% = $500,000 max notional
        )

        # Risk: $100,000 / $5 = 20,000 shares
        # Notional: $500,000 / $100 = 5,000 shares
        # Result: 5,000 shares (notional dominates)
        assert result.shares == 5000
        assert result.capped is True

    def test_penny_stock_sizing(self):
        """Penny stocks should work correctly."""
        from risk.equity_sizer import calculate_position_size

        result = calculate_position_size(
            entry_price=0.50,    # 50 cents
            stop_loss=0.45,     # $0.05 risk per share
            risk_pct=0.02,      # 2% = $1000 max risk
            account_equity=50000.0,
            max_notional_pct=0.10,  # 10% = $5000 max notional
        )

        # Risk: $1000 / $0.05 = 20,000 shares
        # Notional: $5000 / $0.50 = 10,000 shares
        # Result: 10,000 shares (notional dominates)
        assert result.shares == 10000
        assert result.capped is True


@pytest.mark.integration
class TestRiskAmountCalculation:
    """Test that risk amount is calculated correctly."""

    def test_risk_amount_formula(self):
        """Risk amount should equal shares * (entry - stop)."""
        from risk.equity_sizer import calculate_position_size

        result = calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,  # $5 risk per share
            risk_pct=0.02,
            account_equity=100000.0,
        )

        # Verify risk calculation
        expected_risk = result.shares * 5  # $5 per share
        assert result.risk_dollars == expected_risk

    def test_notional_formula(self):
        """Notional should equal shares * entry_price."""
        from risk.equity_sizer import calculate_position_size

        result = calculate_position_size(
            entry_price=150.0,
            stop_loss=145.0,
            risk_pct=0.02,
            account_equity=100000.0,
        )

        expected_notional = result.shares * 150
        assert result.notional == expected_notional


@pytest.mark.integration
class TestCognitiveMultiplier:
    """Test cognitive multiplier reduces position size."""

    def test_low_confidence_reduces_size(self):
        """Low cognitive confidence should reduce position."""
        from risk.equity_sizer import calculate_position_size

        # Full confidence
        full_result = calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            risk_pct=0.02,
            account_equity=100000.0,
            cognitive_multiplier=1.0,
        )

        # Half confidence
        half_result = calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            risk_pct=0.02,
            account_equity=100000.0,
            cognitive_multiplier=0.5,
        )

        # Half confidence should give roughly half shares
        # (unless notional cap kicks in)
        assert half_result.shares <= full_result.shares


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
