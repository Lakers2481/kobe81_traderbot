"""
Unit tests for execution guard.
"""

import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from execution.execution_guard import (
    ExecutionGuard,
    GuardStatus,
    GuardCheckResult,
    QuoteData,
)

ET = ZoneInfo("America/New_York")


class TestQuoteFreshness:
    """Tests for quote freshness validation."""

    def test_fresh_quote(self):
        guard = ExecutionGuard(max_quote_age_seconds=5.0)
        now = datetime.now(ET)
        passed, reason = guard.check_quote_freshness(now - timedelta(seconds=2))
        assert passed
        assert "fresh" in reason.lower()

    def test_stale_quote(self):
        guard = ExecutionGuard(max_quote_age_seconds=5.0)
        now = datetime.now(ET)
        # Quote from 10 seconds ago
        old_time = now - timedelta(seconds=10)
        passed, reason = guard.check_quote_freshness(old_time)
        assert not passed
        assert "stale" in reason.lower()

    def test_future_quote(self):
        guard = ExecutionGuard()
        future = datetime.now(ET) + timedelta(seconds=10)
        passed, reason = guard.check_quote_freshness(future)
        assert not passed
        assert "future" in reason.lower()


class TestSpreadValidation:
    """Tests for spread validation."""

    def test_acceptable_spread(self):
        guard = ExecutionGuard(max_spread_pct=0.50)
        # 0.2% spread
        passed, spread_pct, reason = guard.check_spread(100.0, 100.2)
        assert passed
        assert spread_pct == pytest.approx(0.2, rel=0.01)

    def test_wide_spread(self):
        guard = ExecutionGuard(max_spread_pct=0.50)
        # 1% spread
        passed, spread_pct, reason = guard.check_spread(100.0, 101.0)
        assert not passed
        assert "exceeds" in reason.lower()

    def test_crossed_market(self):
        guard = ExecutionGuard()
        passed, spread_pct, reason = guard.check_spread(100.0, 99.0)  # bid > ask
        assert not passed
        assert "crossed" in reason.lower()

    def test_zero_prices(self):
        guard = ExecutionGuard()
        passed, _, reason = guard.check_spread(0.0, 100.0)
        assert not passed
        assert "invalid" in reason.lower()


class TestQuoteValidity:
    """Tests for quote validity."""

    def test_valid_quote(self):
        guard = ExecutionGuard()
        passed, reason = guard.check_quote_validity(100.0, 100.1)
        assert passed

    def test_missing_bid(self):
        guard = ExecutionGuard()
        passed, reason = guard.check_quote_validity(None, 100.0)
        assert not passed
        assert "bid" in reason.lower()

    def test_missing_ask(self):
        guard = ExecutionGuard()
        passed, reason = guard.check_quote_validity(100.0, None)
        assert not passed
        assert "ask" in reason.lower()

    def test_negative_prices(self):
        guard = ExecutionGuard()
        passed, reason = guard.check_quote_validity(-10.0, 100.0)
        assert not passed


class TestSizeLimits:
    """Tests for order size limits."""

    def test_valid_size(self):
        guard = ExecutionGuard()
        passed, reason = guard.check_size_limits(100, 50.0)
        assert passed

    def test_zero_quantity(self):
        guard = ExecutionGuard()
        passed, reason = guard.check_size_limits(0, 50.0)
        assert not passed

    def test_order_too_small(self):
        guard = ExecutionGuard()
        passed, reason = guard.check_size_limits(1, 0.10, min_notional=1.0)
        assert not passed
        assert "too small" in reason.lower()

    def test_order_too_large(self):
        guard = ExecutionGuard()
        passed, reason = guard.check_size_limits(1000, 200.0, max_notional=100000.0)
        assert not passed
        assert "too large" in reason.lower()


class TestFullCheck:
    """Tests for full guard check."""

    def test_disabled_guard(self):
        guard = ExecutionGuard(enabled=False)
        result = guard.full_check("AAPL", "buy", 100, 150.0)
        assert result.approved
        assert result.status == GuardStatus.PASSED

    def test_passed_check(self):
        guard = ExecutionGuard(stand_down_on_uncertainty=False)
        quote = QuoteData(
            symbol="AAPL",
            bid=149.95,
            ask=150.05,
            bid_size=100,
            ask_size=100,
            timestamp=datetime.now(ET),
        )
        result = guard.full_check("AAPL", "buy", 100, 150.0, quote=quote, skip_trading_status=True)
        assert result.approved
        assert result.status == GuardStatus.PASSED

    def test_rejected_wide_spread(self):
        guard = ExecutionGuard(max_spread_pct=0.10)
        quote = QuoteData(
            symbol="AAPL",
            bid=149.0,
            ask=151.0,  # 1.3% spread
            bid_size=100,
            ask_size=100,
            timestamp=datetime.now(ET),
        )
        result = guard.full_check("AAPL", "buy", 100, 150.0, quote=quote, skip_trading_status=True)
        assert not result.approved
        assert result.status == GuardStatus.REJECTED

    def test_stand_down_stale_quote(self):
        guard = ExecutionGuard(max_quote_age_seconds=5.0, stand_down_on_uncertainty=True)
        quote = QuoteData(
            symbol="AAPL",
            bid=149.95,
            ask=150.05,
            bid_size=100,
            ask_size=100,
            timestamp=datetime.now(ET) - timedelta(seconds=10),  # Stale
        )
        result = guard.full_check("AAPL", "buy", 100, 150.0, quote=quote, skip_trading_status=True)
        assert not result.approved
        assert result.status == GuardStatus.STAND_DOWN
        assert len(result.stand_down_reasons) > 0

    def test_stand_down_no_quote(self):
        guard = ExecutionGuard(stand_down_on_uncertainty=True)
        result = guard.full_check("AAPL", "buy", 100, 150.0, quote=None, skip_trading_status=True)
        assert not result.approved
        assert result.status == GuardStatus.STAND_DOWN
        assert "No quote" in result.stand_down_reasons[0]

    def test_price_sanity_check(self):
        guard = ExecutionGuard(stand_down_on_uncertainty=True)
        quote = QuoteData(
            symbol="AAPL",
            bid=149.95,
            ask=150.05,
            bid_size=100,
            ask_size=100,
            timestamp=datetime.now(ET),
        )
        # Price differs by more than 5% from mid
        result = guard.full_check("AAPL", "buy", 100, 170.0, quote=quote, skip_trading_status=True)
        assert not result.approved
        assert result.status == GuardStatus.STAND_DOWN

    def test_from_config(self):
        config = {
            "enabled": True,
            "max_quote_age_seconds": 10.0,
            "max_spread_pct": 1.0,
            "stand_down_on_uncertainty": False,
        }
        guard = ExecutionGuard.from_config(config)
        assert guard.enabled
        assert guard.max_quote_age_seconds == 10.0
        assert guard.max_spread_pct == 1.0
        assert not guard.stand_down_on_uncertainty


class TestGuardCheckResult:
    """Tests for guard check result."""

    def test_to_dict(self):
        result = GuardCheckResult(
            status=GuardStatus.PASSED,
            approved=True,
            symbol="AAPL",
            side="buy",
            qty=100,
            price=150.0,
        )
        d = result.to_dict()
        assert d["status"] == "PASSED"
        assert d["approved"] is True
        assert d["symbol"] == "AAPL"

    def test_timestamp_auto_generated(self):
        result = GuardCheckResult(
            status=GuardStatus.PASSED,
            approved=True,
            symbol="AAPL",
            side="buy",
            qty=100,
            price=150.0,
        )
        assert result.timestamp != ""
