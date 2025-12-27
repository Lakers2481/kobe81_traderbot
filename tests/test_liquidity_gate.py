"""
Tests for Liquidity Gate.

Tests ADV, spread, and order impact checks.
"""
import pytest

from risk.liquidity_gate import (
    LiquidityGate,
    LiquidityCheck,
    LiquidityIssue,
    check_liquidity,
    get_liquidity_gate,
)


class TestLiquidityGate:
    """Tests for LiquidityGate."""

    def test_initialization(self):
        """Should initialize with defaults."""
        gate = LiquidityGate()
        assert gate.min_adv_usd == 100_000
        assert gate.max_spread_pct == 0.50
        assert gate.max_pct_of_adv == 1.0

    def test_custom_thresholds(self):
        """Should accept custom thresholds."""
        gate = LiquidityGate(
            min_adv_usd=500_000,
            max_spread_pct=0.25,
            max_pct_of_adv=0.5,
        )
        assert gate.min_adv_usd == 500_000
        assert gate.max_spread_pct == 0.25
        assert gate.max_pct_of_adv == 0.5

    def test_pass_good_liquidity(self):
        """Should pass stocks with good liquidity."""
        gate = LiquidityGate(min_adv_usd=100_000, max_spread_pct=0.50)

        check = gate.check_liquidity(
            symbol='AAPL',
            price=150.0,
            shares=100,
            bid=149.98,
            ask=150.02,
            avg_volume=50_000_000,
        )

        assert check.passed
        assert len(check.issues) == 0
        assert "passed" in check.reason.lower()

    def test_fail_insufficient_adv(self):
        """Should fail stocks with insufficient ADV."""
        gate = LiquidityGate(min_adv_usd=100_000)

        check = gate.check_liquidity(
            symbol='LOWVOL',
            price=10.0,
            shares=100,
            bid=9.99,
            ask=10.01,
            avg_volume=5_000,  # Only $50k ADV
        )

        assert not check.passed
        assert LiquidityIssue.INSUFFICIENT_ADV in check.issues
        assert "ADV" in check.reason

    def test_fail_wide_spread(self):
        """Should fail stocks with wide spreads."""
        gate = LiquidityGate(max_spread_pct=0.50)

        check = gate.check_liquidity(
            symbol='WIDESPREAD',
            price=50.0,
            shares=100,
            bid=49.00,
            ask=51.00,  # 4% spread
            avg_volume=1_000_000,
        )

        assert not check.passed
        assert LiquidityIssue.WIDE_SPREAD in check.issues
        assert "Spread" in check.reason

    def test_fail_large_order_impact(self):
        """Should fail orders that are too large relative to ADV."""
        gate = LiquidityGate(max_pct_of_adv=1.0)

        check = gate.check_liquidity(
            symbol='SMALLCAP',
            price=50.0,
            shares=10_000,  # $500k order
            bid=49.98,
            ask=50.02,
            avg_volume=100_000,  # Only $5M ADV, order is 10% of ADV
        )

        assert not check.passed
        assert LiquidityIssue.LARGE_ORDER_IMPACT in check.issues
        assert "ADV" in check.reason

    def test_fail_missing_quote(self):
        """Should fail when quote data is missing."""
        gate = LiquidityGate()

        check = gate.check_liquidity(
            symbol='NOQUOTE',
            price=100.0,
            shares=100,
            bid=None,
            ask=None,
            avg_volume=None,
        )

        assert not check.passed
        assert LiquidityIssue.MISSING_QUOTE in check.issues

    def test_multiple_issues(self):
        """Should report all issues found."""
        gate = LiquidityGate(min_adv_usd=100_000, max_spread_pct=0.50)

        check = gate.check_liquidity(
            symbol='BADSTOCK',
            price=10.0,
            shares=100,
            bid=9.80,
            ask=10.20,  # 4% spread
            avg_volume=1_000,  # Only $10k ADV
        )

        assert not check.passed
        assert LiquidityIssue.INSUFFICIENT_ADV in check.issues
        assert LiquidityIssue.WIDE_SPREAD in check.issues
        assert len(check.issues) >= 2

    def test_non_strict_mode(self):
        """Should only fail on critical issues in non-strict mode."""
        gate = LiquidityGate(max_pct_of_adv=0.5)

        # Large order but good liquidity otherwise
        check = gate.check_liquidity(
            symbol='AAPL',
            price=150.0,
            shares=50_000,  # $7.5M order = 0.75% of ADV
            bid=149.98,
            ask=150.02,
            avg_volume=50_000_000,  # $7.5B ADV
            strict=False,
        )

        # Non-strict: order impact is not critical
        assert check.passed

    def test_spread_calculation(self):
        """Should calculate spread percentage correctly."""
        gate = LiquidityGate()

        check = gate.check_liquidity(
            symbol='TEST',
            price=100.0,
            shares=100,
            bid=99.90,
            ask=100.10,  # $0.20 spread on $100 = 0.2%
            avg_volume=1_000_000,
        )

        assert 0.19 <= check.spread_pct <= 0.21

    def test_adv_calculation(self):
        """Should calculate ADV in USD correctly."""
        gate = LiquidityGate()

        check = gate.check_liquidity(
            symbol='TEST',
            price=50.0,
            shares=100,
            bid=49.98,
            ask=50.02,
            avg_volume=100_000,  # 100k shares at $50 = $5M
        )

        assert check.adv_usd == 5_000_000

    def test_order_pct_calculation(self):
        """Should calculate order percentage of ADV correctly."""
        gate = LiquidityGate()

        check = gate.check_liquidity(
            symbol='TEST',
            price=100.0,
            shares=100,  # $10k order
            bid=99.98,
            ask=100.02,
            avg_volume=100_000,  # $10M ADV
        )

        # $10k / $10M = 0.1%
        assert 0.09 <= check.order_pct_of_adv <= 0.11

    def test_stats_tracking(self):
        """Should track check statistics."""
        gate = LiquidityGate(min_adv_usd=50_000)

        # Some passes, some fails
        gate.check_liquidity('GOOD1', 100.0, 100, 99.98, 100.02, 1_000_000)
        gate.check_liquidity('GOOD2', 100.0, 100, 99.98, 100.02, 1_000_000)
        gate.check_liquidity('BAD1', 10.0, 100, 9.98, 10.02, 1_000)  # Low ADV

        stats = gate.get_stats()

        assert stats['total_checks'] == 3
        assert stats['passed'] == 2
        assert stats['failed'] == 1
        assert stats['pass_rate'] == pytest.approx(66.67, rel=0.01)

    def test_reset_history(self):
        """Should clear history on reset."""
        gate = LiquidityGate()

        gate.check_liquidity('TEST', 100.0, 100, 99.98, 100.02, 1_000_000)
        assert gate.get_stats()['total_checks'] == 1

        gate.reset_history()
        assert gate.get_stats()['total_checks'] == 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_check_liquidity(self):
        """Should use default gate."""
        check = check_liquidity(
            symbol='AAPL',
            price=150.0,
            shares=100,
            bid=149.98,
            ask=150.02,
            avg_volume=50_000_000,
        )

        assert isinstance(check, LiquidityCheck)
        assert check.passed

    def test_get_liquidity_gate(self):
        """Should return singleton gate."""
        gate1 = get_liquidity_gate()
        gate2 = get_liquidity_gate()

        assert gate1 is gate2


class TestLiquidityCheck:
    """Tests for LiquidityCheck dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        check = LiquidityCheck(
            symbol='TEST',
            passed=True,
            reason='All checks passed',
            issues=[],
            adv_usd=1_000_000,
            spread_pct=0.05,
        )

        d = check.to_dict()

        assert d['symbol'] == 'TEST'
        assert d['passed'] == True
        assert d['adv_usd'] == 1_000_000
        assert 'checked_at' in d


# Run with: pytest tests/test_liquidity_gate.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
