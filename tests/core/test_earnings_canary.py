"""
Tests for earnings filter source tagging and canary functions.

FIX (2026-01-05): Added for data quality monitoring.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.earnings_filter import (
    fetch_earnings_dates,
    get_earnings_source,
    run_earnings_canary,
    check_earnings_source_health,
    clear_cache,
    KNOWN_EARNINGS_SYMBOLS,
)


@pytest.fixture(autouse=True)
def clear_cache_before_each():
    """Clear cache before each test."""
    clear_cache()
    yield
    clear_cache()


class TestSourceTagging:
    """Tests for earnings source tagging."""

    def test_polygon_source_tagging(self):
        """Polygon source is tagged correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": {
                "events": [
                    {"earnings_date": "2025-01-15"},
                    {"earnings_date": "2024-10-15"},
                ]
            }
        }

        with patch("os.getenv", return_value="test_api_key"):
            with patch("requests.get", return_value=mock_response):
                dates = fetch_earnings_dates("AAPL", force_refresh=True)
                source = get_earnings_source("AAPL")

        assert len(dates) == 2
        assert source == "polygon"

    def test_no_polygon_key_returns_empty(self):
        """Returns empty list when no Polygon API key is set.

        FIX (2026-01-06): yfinance fallback removed. Now returns empty
        when Polygon is unavailable, for faster/more reliable scans.
        """
        with patch("os.getenv", return_value=None):  # No Polygon key
            dates = fetch_earnings_dates("AAPL", force_refresh=True)
            source = get_earnings_source("AAPL")

        assert dates == []
        assert source == "none"

    def test_none_source_when_no_data(self):
        """Source is 'none' when no earnings found."""
        with patch("os.getenv", return_value=None):  # No Polygon key
            with patch("core.earnings_filter._fetch_from_yfinance", return_value=[]):
                dates = fetch_earnings_dates("UNKNOWN_SYMBOL_XYZ", force_refresh=True)
                source = get_earnings_source("UNKNOWN_SYMBOL_XYZ")

        assert dates == []
        assert source == "none"


class TestEarningsCanary:
    """Tests for earnings canary function."""

    def test_canary_passes_with_recent_earnings(self):
        """Canary passes when earnings are found within lookback period."""
        # Mock fetch to return recent earnings
        recent_date = datetime.now() - timedelta(days=30)

        with patch(
            "core.earnings_filter.fetch_earnings_dates",
            return_value=[recent_date],
        ):
            with patch("core.earnings_filter.get_earnings_source", return_value="polygon"):
                results = run_earnings_canary(symbols=["AAPL"])

        assert results["AAPL"]["passed"] is True
        assert results["AAPL"]["source"] == "polygon"
        assert results["AAPL"]["recent_dates"] == 1

    def test_canary_fails_with_no_recent_earnings(self):
        """Canary fails when no earnings within lookback period."""
        # Mock fetch to return only old earnings
        old_date = datetime.now() - timedelta(days=180)

        with patch(
            "core.earnings_filter.fetch_earnings_dates",
            return_value=[old_date],
        ):
            with patch("core.earnings_filter.get_earnings_source", return_value="polygon"):
                with patch(
                    "trade_logging.prometheus_metrics.EARNINGS_CANARY_FAILED"
                ) as mock_counter:
                    mock_counter.labels.return_value = MagicMock()
                    results = run_earnings_canary(symbols=["AAPL"])

        assert results["AAPL"]["passed"] is False
        assert "No earnings in last 90 days" in results["AAPL"]["reason"]
        mock_counter.labels.assert_called_with(symbol="AAPL", source="polygon")

    def test_canary_handles_fetch_errors(self):
        """Canary handles errors gracefully."""
        with patch(
            "core.earnings_filter.fetch_earnings_dates",
            side_effect=Exception("API Error"),
        ):
            with patch(
                "trade_logging.prometheus_metrics.EARNINGS_CANARY_FAILED"
            ) as mock_counter:
                mock_counter.labels.return_value = MagicMock()
                results = run_earnings_canary(symbols=["AAPL"])

        assert results["AAPL"]["passed"] is False
        assert results["AAPL"]["source"] == "error"
        assert "API Error" in results["AAPL"]["reason"]

    def test_canary_uses_known_symbols_by_default(self):
        """Canary uses KNOWN_EARNINGS_SYMBOLS when no symbols provided."""
        with patch(
            "core.earnings_filter.fetch_earnings_dates",
            return_value=[datetime.now()],
        ):
            with patch("core.earnings_filter.get_earnings_source", return_value="polygon"):
                results = run_earnings_canary()

        # Should have results for all known symbols
        for symbol in KNOWN_EARNINGS_SYMBOLS:
            assert symbol in results


class TestEarningsSourceHealth:
    """Tests for earnings source health check."""

    def test_healthy_when_majority_pass(self):
        """Reports healthy when majority of canaries pass."""
        recent = datetime.now() - timedelta(days=30)

        with patch(
            "core.earnings_filter.fetch_earnings_dates",
            return_value=[recent],
        ):
            with patch("core.earnings_filter.get_earnings_source", return_value="polygon"):
                health = check_earnings_source_health()

        assert health["healthy"] is True
        assert health["passed"] == health["total"]

    def test_unhealthy_when_majority_fail(self):
        """Reports unhealthy when majority of canaries fail."""
        # No earnings at all
        with patch("core.earnings_filter.fetch_earnings_dates", return_value=[]):
            with patch("core.earnings_filter.get_earnings_source", return_value="none"):
                with patch(
                    "trade_logging.prometheus_metrics.EARNINGS_CANARY_FAILED"
                ) as mock_counter:
                    mock_counter.labels.return_value = MagicMock()
                    health = check_earnings_source_health()

        assert health["healthy"] is False
        assert health["passed"] == 0

    def test_source_stats_aggregation(self):
        """Source statistics are correctly aggregated."""
        recent = datetime.now() - timedelta(days=30)

        with patch(
            "core.earnings_filter.fetch_earnings_dates",
            return_value=[recent],
        ):
            with patch("core.earnings_filter.get_earnings_source", return_value="polygon"):
                health = check_earnings_source_health()

        assert "polygon" in health["source_stats"]
        assert health["source_stats"]["polygon"] == len(KNOWN_EARNINGS_SYMBOLS)
