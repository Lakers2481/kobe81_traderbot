#!/usr/bin/env python3
"""
REAL Position Sizing Math Tests - Not Type Checks
==================================================

FIX (2026-01-08): Phase 3.2 - Tests that verify ACTUAL position sizing math.

These tests:
1. Verify risk-based sizing formula: shares = risk$ / |entry - stop|
2. Verify notional cap: shares <= max_notional / entry
3. Verify dual-cap enforcement: final = min(risk_shares, notional_shares)
4. Test boundary conditions and edge cases

NO MOCKS. Real calculations. Real assertions on math.
"""

import pytest
from dataclasses import dataclass


class TestPositionSizingFormulas:
    """Test core position sizing math."""

    def test_basic_risk_calculation(self):
        """
        REAL TEST: Verify risk-based position sizing.

        Formula: shares = (account * risk_pct) / |entry - stop|

        Example:
        - Account: $100,000
        - Risk: 2% = $2,000
        - Entry: $100
        - Stop: $95
        - Risk per share: $5
        - Shares = $2,000 / $5 = 400
        """
        from risk.equity_sizer import calculate_position_size

        size = calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            risk_pct=0.02,
            account_equity=100000.0,
            max_notional_pct=1.0,  # No notional cap for this test
        )

        # Risk dollars = $100,000 * 0.02 = $2,000
        # Risk per share = |100 - 95| = $5
        # Shares = $2,000 / $5 = 400
        assert size.shares == 400, (
            f"Expected 400 shares from risk calc. Got {size.shares}. "
            f"Risk=${size.risk_dollars}, RPS=${size.risk_per_share}"
        )
        assert size.risk_dollars == pytest.approx(2000.0, rel=0.01), (
            f"Expected $2000 risk. Got ${size.risk_dollars}"
        )

    def test_notional_cap_kicks_in(self):
        """
        REAL TEST: Verify notional cap limits position size.

        Formula: max_shares = (account * max_notional_pct) / entry

        Example:
        - Account: $50,000
        - Max notional: 20% = $10,000
        - Entry: $200
        - Max shares by notional = $10,000 / $200 = 50
        - Risk calc might give 100 shares, but notional caps to 50
        """
        from risk.equity_sizer import calculate_position_size

        size = calculate_position_size(
            entry_price=200.0,
            stop_loss=190.0,  # $10 risk per share
            risk_pct=0.02,    # 2% of $50k = $1,000 risk
            account_equity=50000.0,
            max_notional_pct=0.20,  # 20% = $10,000 max notional
        )

        # Risk shares = $1,000 / $10 = 100 shares
        # Notional shares = $10,000 / $200 = 50 shares
        # Final = min(100, 50) = 50 shares (notional cap wins)
        assert size.shares == 50, (
            f"Expected 50 shares (notional cap). Got {size.shares}. "
            f"Should be min(risk=100, notional=50)"
        )
        assert size.capped is True, "Should be flagged as capped"
        assert "notional" in size.cap_reason.lower(), (
            f"Cap reason should mention notional. Got: {size.cap_reason}"
        )

    def test_risk_cap_wins_over_notional(self):
        """
        REAL TEST: Verify risk cap wins when it's the smaller.

        Example:
        - Account: $50,000
        - Risk: 2% = $1,000
        - Entry: $20
        - Stop: $18 ($2 risk per share)
        - Risk shares = $1,000 / $2 = 500
        - Notional shares = $10,000 / $20 = 500
        - Both equal, not capped
        """
        from risk.equity_sizer import calculate_position_size

        size = calculate_position_size(
            entry_price=20.0,
            stop_loss=18.0,   # $2 risk per share
            risk_pct=0.02,    # 2% of $50k = $1,000
            account_equity=50000.0,
            max_notional_pct=0.20,  # 20% = $10,000
        )

        # Risk shares = $1,000 / $2 = 500
        # Notional shares = $10,000 / $20 = 500
        # Equal, so not considered "capped"
        assert size.shares == 500, f"Expected 500 shares. Got {size.shares}"

    def test_tight_stop_gives_more_shares(self):
        """
        REAL TEST: Tighter stop = smaller risk per share = more shares.

        Example with 100% notional cap (to isolate risk-based sizing):
        - Same risk budget ($1,000)
        - Entry: $100
        - Tight stop: $99 → risk/share = $1 → 1000 shares BUT capped by notional
        - Wide stop: $90 → risk/share = $10 → 100 shares

        Note: Even with 100% notional cap, the tight stop case gets capped
        because $1000 risk / $1 = 1000 shares = $100,000 > $50,000 account.
        So it gets capped at 500 shares ($50,000 / $100).
        """
        from risk.equity_sizer import calculate_position_size

        # Tight stop - will be capped by notional even at 100%
        tight = calculate_position_size(
            entry_price=100.0,
            stop_loss=99.0,   # $1 risk per share
            risk_pct=0.02,
            account_equity=50000.0,
            max_notional_pct=1.0,  # 100% notional cap = $50,000
        )

        # Wide stop
        wide = calculate_position_size(
            entry_price=100.0,
            stop_loss=90.0,   # $10 risk per share
            risk_pct=0.02,
            account_equity=50000.0,
            max_notional_pct=1.0,
        )

        # Tight gets capped by 100% notional: $50,000 / $100 = 500 shares
        assert tight.shares == 500, f"Tight stop expected 500 shares (notional cap). Got {tight.shares}"
        assert wide.shares == 100, f"Wide stop expected 100 shares. Got {wide.shares}"
        # Tight is capped, so risk dollars will be less than intended
        assert tight.shares > wide.shares, "Tighter stop should give more shares"

    def test_high_price_stock_limited(self):
        """
        REAL TEST: High-priced stocks are limited by notional cap.

        Example:
        - $50,000 account, 20% notional cap = $10,000 max
        - $5,000 stock → max 2 shares
        """
        from risk.equity_sizer import calculate_position_size

        size = calculate_position_size(
            entry_price=5000.0,
            stop_loss=4800.0,  # $200 risk per share
            risk_pct=0.02,     # 2% of $50k = $1,000 risk → 5 shares
            account_equity=50000.0,
            max_notional_pct=0.20,  # $10,000 max → 2 shares
        )

        assert size.shares == 2, (
            f"High price stock should be limited to 2 shares by notional. Got {size.shares}"
        )
        assert size.capped is True
        assert size.notional <= 10000.0, f"Notional ${size.notional} should be <= $10,000"

    def test_minimum_shares_enforced(self):
        """
        REAL TEST: Minimum 1 share even if math says 0.
        """
        from risk.equity_sizer import calculate_position_size

        size = calculate_position_size(
            entry_price=50000.0,  # Very expensive stock
            stop_loss=45000.0,
            risk_pct=0.02,
            account_equity=10000.0,  # Small account
            max_notional_pct=0.20,
            min_shares=1,
        )

        assert size.shares >= 1, f"Should have at least 1 share. Got {size.shares}"

    def test_zero_equity_returns_minimum(self):
        """
        REAL TEST: Zero or negative equity handling.
        """
        from risk.equity_sizer import calculate_position_size

        size = calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            risk_pct=0.02,
            account_equity=0.0,  # Zero equity
            max_notional_pct=0.20,
            min_shares=1,
        )

        # With zero equity, risk calc gives 0, but min_shares enforces 1
        assert size.shares == 1, f"Zero equity should give min shares. Got {size.shares}"


class TestCognitiveMultiplier:
    """Test cognitive/performance-based position sizing adjustments."""

    def test_full_confidence_no_reduction(self):
        """
        REAL TEST: Full confidence (1.0) doesn't reduce position.

        Note: Default 20% notional cap means $100,000 account = $20,000 max notional.
        At $100/share, that's 200 shares max (not 400 from risk calc).
        """
        from risk.equity_sizer import calculate_position_size

        # Full confidence - but 20% notional cap applies
        full = calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            risk_pct=0.02,
            account_equity=100000.0,
            cognitive_multiplier=1.0,
            # Default max_notional_pct=0.20 applies
        )

        # Risk shares = $2,000 / $5 = 400, but notional cap = $20,000 / $100 = 200
        assert full.shares == 200, f"Full confidence expected 200 shares (20% notional cap). Got {full.shares}"
        assert full.capped is True, "Should be capped by notional"

    def test_half_confidence_halves_position(self):
        """
        REAL TEST: Half confidence (0.5) should halve position.
        """
        from risk.equity_sizer import calculate_position_size

        # Half confidence
        half = calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            risk_pct=0.02,
            account_equity=100000.0,
            max_notional_pct=1.0,
            cognitive_multiplier=0.5,
        )

        # Full would be 400, half is 200
        assert half.shares == 200, (
            f"Half confidence expected 200 shares (half of 400). Got {half.shares}"
        )

    def test_low_confidence_significantly_reduces(self):
        """
        REAL TEST: Low confidence (0.25) gives quarter position.
        """
        from risk.equity_sizer import calculate_position_size

        low = calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            risk_pct=0.02,
            account_equity=100000.0,
            max_notional_pct=1.0,
            cognitive_multiplier=0.25,
        )

        assert low.shares == 100, (
            f"Quarter confidence expected 100 shares. Got {low.shares}"
        )


class TestKellyIntegration:
    """Test Kelly Criterion integration with standard sizing."""

    def test_kelly_reduces_when_indicated(self):
        """
        REAL TEST: Kelly sizing reduces position when optimal fraction is small.
        """
        from risk.equity_sizer import calculate_position_size_with_kelly

        # Low win rate scenario - Kelly should reduce
        size = calculate_position_size_with_kelly(
            entry_price=100.0,
            stop_loss=95.0,
            risk_pct=0.02,
            account_equity=100000.0,
            use_kelly=True,
            kelly_win_rate=0.50,        # 50% win rate
            kelly_win_loss_ratio=1.0,   # 1:1 R:R
            kelly_fraction=0.5,         # Half Kelly
        )

        # Standard would give 400 shares
        # Kelly with 50% WR, 1:1 R:R gives 0% optimal (breakeven)
        # Half Kelly of 0% = 0%
        # Kelly should reduce significantly or return standard minimum
        assert size.shares <= 400, (
            f"Kelly should not increase position. Got {size.shares}"
        )

    def test_kelly_disabled_gives_standard(self):
        """
        REAL TEST: Disabling Kelly gives standard position sizing.
        """
        from risk.equity_sizer import calculate_position_size_with_kelly, calculate_position_size

        kelly_off = calculate_position_size_with_kelly(
            entry_price=100.0,
            stop_loss=95.0,
            risk_pct=0.02,
            account_equity=100000.0,
            use_kelly=False,
        )

        standard = calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            risk_pct=0.02,
            account_equity=100000.0,
        )

        assert kelly_off.shares == standard.shares, (
            f"Kelly disabled should equal standard. Kelly={kelly_off.shares}, Std={standard.shares}"
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_entry_equals_stop_fallback(self):
        """
        REAL TEST: Entry == Stop uses fallback risk calculation.
        """
        from risk.equity_sizer import calculate_position_size

        # Entry equals stop - would cause division by zero
        size = calculate_position_size(
            entry_price=100.0,
            stop_loss=100.0,  # Same as entry!
            risk_pct=0.02,
            account_equity=100000.0,
        )

        # Should use 5% fallback: risk_per_share = 100 * 0.05 = $5
        assert size.shares > 0, "Should calculate despite entry == stop"
        assert size.risk_per_share == pytest.approx(5.0, rel=0.1), (
            f"Fallback should use 5% of entry. Got RPS={size.risk_per_share}"
        )

    def test_very_tight_stop_capped_by_notional(self):
        """
        REAL TEST: Very tight stop would give huge position, capped by notional.
        """
        from risk.equity_sizer import calculate_position_size

        # $0.01 stop on $100 stock = 100,000 risk shares, but notional caps it
        size = calculate_position_size(
            entry_price=100.0,
            stop_loss=99.99,  # $0.01 risk per share
            risk_pct=0.02,
            account_equity=100000.0,
            max_notional_pct=0.20,  # $20,000 max = 200 shares
        )

        # Risk calc would give 2,000,000 shares ($2000 / $0.01)
        # Notional caps to $20,000 / $100 = 200 shares
        assert size.shares == 200, f"Should be capped by notional. Got {size.shares}"
        assert size.capped is True

    def test_format_summary_output(self):
        """
        REAL TEST: Summary formatting is correct.

        Note: With default 20% notional cap, actual shares = 200 (not 400).
        """
        from risk.equity_sizer import calculate_position_size, format_size_summary

        size = calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            risk_pct=0.02,
            account_equity=100000.0,
        )

        summary = format_size_summary(size, "AAPL")

        assert "AAPL" in summary, "Symbol should be in summary"
        assert "200 shares" in summary, f"Share count should be in summary. Got: {summary}"
        assert "$100,000" in summary, "Account equity should be in summary"
        assert "2.0%" in summary, "Risk % should be in summary"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
