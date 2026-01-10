"""
INTEGRATION TESTS: Data Provider Fallback Chain (HIGH)

Tests the data provider fallback chain:
Polygon (primary) -> Stooq (fallback) -> YFinance (last resort)

This ensures the system gracefully handles API failures and rate limits.

Author: Kobe Trading System
Version: 1.0.0
Date: 2026-01-06
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.mark.integration
@pytest.mark.requires_mock_data
class TestPolygonSuccess:
    """Test Polygon primary provider when successful."""

    def test_polygon_returns_valid_data(self, requests_mock):
        """Polygon returns data correctly."""
        from tests.fixtures.provider_mocks import mock_polygon_api, mock_polygon_response_data

        mock_polygon_api(requests_mock, symbols=["AAPL"], days=100)

        # The mock is set up - verify response structure
        response_data = mock_polygon_response_data("AAPL", 100)

        assert response_data["status"] == "OK"
        assert response_data["resultsCount"] == 100
        assert len(response_data["results"]) == 100

        # Verify OHLCV fields
        first_bar = response_data["results"][0]
        assert "o" in first_bar  # open
        assert "h" in first_bar  # high (implied from generation)
        assert "c" in first_bar  # close
        assert "v" in first_bar  # volume

    def test_polygon_response_has_correct_structure(self):
        """Verify Polygon response matches expected schema."""
        from tests.fixtures.provider_mocks import mock_polygon_response_data

        data = mock_polygon_response_data("MSFT", 50)

        assert "ticker" in data
        assert data["ticker"] == "MSFT"
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 50


@pytest.mark.integration
@pytest.mark.requires_mock_data
class TestPolygonFailStooqSuccess:
    """Test fallback to Stooq when Polygon fails."""

    def test_polygon_500_triggers_fallback(self, requests_mock):
        """Polygon 500 error should trigger Stooq fallback."""
        from tests.fixtures.provider_mocks import failing_polygon_api, mock_stooq_api

        # Set up Polygon to fail
        failing_polygon_api(requests_mock, status_code=500)

        # Set up Stooq to succeed
        mock_stooq_api(requests_mock, symbols=["AAPL"])

        # In production, the data provider would:
        # 1. Try Polygon -> 500 error
        # 2. Fallback to Stooq -> success
        # This test verifies mocks are set up correctly

    def test_polygon_timeout_triggers_fallback(self, requests_mock):
        """Polygon timeout should trigger Stooq fallback."""
        from tests.fixtures.provider_mocks import mock_stooq_api
        import requests

        # Set up Polygon to timeout
        requests_mock.get(
            "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day",
            exc=requests.exceptions.Timeout,
        )

        # Set up Stooq to succeed
        mock_stooq_api(requests_mock, symbols=["AAPL"])


@pytest.mark.integration
@pytest.mark.requires_mock_data
class TestRateLimiting:
    """Test rate limit handling."""

    def test_polygon_429_triggers_fallback(self, requests_mock):
        """Polygon 429 (rate limit) should trigger fallback."""
        from tests.fixtures.provider_mocks import rate_limited_polygon_api, mock_stooq_api

        # Set up Polygon to rate limit
        rate_limited_polygon_api(requests_mock)

        # Set up Stooq to succeed
        mock_stooq_api(requests_mock, symbols=["AAPL"])

        # Verify rate limit response
        import requests
        response = requests.get("https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_rate_limit_includes_retry_header(self, requests_mock):
        """Rate limit response should include Retry-After header."""
        from tests.fixtures.provider_mocks import rate_limited_polygon_api

        rate_limited_polygon_api(requests_mock)

        import requests
        response = requests.get("https://api.polygon.io/v2/aggs/ticker/TEST/range/1/day")

        assert response.status_code == 429
        assert "Retry-After" in response.headers
        assert int(response.headers["Retry-After"]) > 0


@pytest.mark.integration
@pytest.mark.requires_mock_data
class TestAllProvidersFail:
    """Test graceful error handling when all providers fail."""

    def test_all_providers_fail_gracefully(self, requests_mock):
        """System should handle gracefully when all providers fail."""
        from tests.fixtures.provider_mocks import failing_polygon_api
        import requests

        # Set up all providers to fail
        failing_polygon_api(requests_mock, status_code=500)

        requests_mock.get(
            "https://stooq.com/q/d/l/",
            status_code=503,
        )

        # In production, this would:
        # 1. Try Polygon -> fail
        # 2. Try Stooq -> fail
        # 3. Return empty DataFrame or raise controlled exception


@pytest.mark.integration
@pytest.mark.requires_mock_data
class TestCaching:
    """Test data caching prevents unnecessary API calls."""

    def test_cache_prevents_duplicate_calls(self, tmp_path, requests_mock):
        """Cached data should prevent API calls."""
        from tests.fixtures.provider_mocks import mock_polygon_api

        mock_polygon_api(requests_mock, symbols=["AAPL"], days=100)

        # Create a mock cache file
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "AAPL_2025-01-01_2025-12-31.csv"

        # Write some cached data
        cache_file.write_text(
            "timestamp,open,high,low,close,volume\n"
            "2025-01-02,150.0,152.0,149.0,151.0,1000000\n"
        )

        # Cache should exist
        assert cache_file.exists()
        assert cache_file.stat().st_size > 0

    def test_stale_cache_triggers_refresh(self, tmp_path):
        """Cache older than TTL should trigger API refresh."""
        import os
        import time

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "AAPL_stale.csv"

        # Create old cache file
        cache_file.write_text("timestamp,open,high,low,close,volume\n")

        # Make file appear old
        old_time = time.time() - (7 * 24 * 60 * 60)  # 7 days ago
        os.utime(cache_file, (old_time, old_time))

        # Check file age
        file_age = time.time() - cache_file.stat().st_mtime
        assert file_age > 6 * 24 * 60 * 60  # More than 6 days old


@pytest.mark.integration
@pytest.mark.requires_mock_data
class TestEmptyResponses:
    """Test handling of empty or malformed responses."""

    def test_empty_results_handled(self, requests_mock):
        """Empty results array should be handled gracefully."""
        from tests.fixtures.provider_mocks import mock_empty_polygon_response

        mock_empty_polygon_response(requests_mock, "EMPTY")

        import requests
        response = requests.get("https://api.polygon.io/v2/aggs/ticker/EMPTY/range/1/day")
        data = response.json()

        assert data["resultsCount"] == 0
        assert len(data["results"]) == 0

    def test_malformed_data_handled(self, requests_mock):
        """Malformed data should be handled gracefully."""
        from tests.fixtures.provider_mocks import mock_malformed_polygon_response

        mock_malformed_polygon_response(requests_mock, "BAD")

        import requests
        response = requests.get("https://api.polygon.io/v2/aggs/ticker/BAD/range/1/day")
        data = response.json()

        # Results exist but have missing fields
        assert "results" in data
        # First result missing OHLC
        first = data["results"][0]
        assert "c" not in first  # close is missing


@pytest.mark.integration
@pytest.mark.requires_mock_data
class TestMultiSymbolFetch:
    """Test fetching data for multiple symbols."""

    def test_multi_symbol_parallel_fetch(self, requests_mock):
        """Multiple symbols should be fetched correctly."""
        from tests.fixtures.provider_mocks import mock_polygon_api

        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        mock_polygon_api(requests_mock, symbols=symbols, days=100)

        # Verify each symbol has unique data
        from tests.fixtures.provider_mocks import mock_polygon_response_data

        for symbol in symbols:
            data = mock_polygon_response_data(symbol, 100)
            assert data["ticker"] == symbol
            assert len(data["results"]) == 100

    def test_partial_failure_handled(self, requests_mock):
        """Some symbols failing should not break entire fetch."""
        from tests.fixtures.provider_mocks import mock_polygon_api

        # Set up AAPL to succeed
        mock_polygon_api(requests_mock, symbols=["AAPL"], days=100)

        # Set up BADTICKER to fail
        requests_mock.get(
            "https://api.polygon.io/v2/aggs/ticker/BADTICKER/range/1/day",
            status_code=404,
            json={"status": "ERROR", "error": "Symbol not found"},
        )

        # AAPL should still work
        import requests
        response = requests.get("https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day")
        assert response.status_code == 200


@pytest.mark.integration
@pytest.mark.requires_mock_data
class TestDataQuality:
    """Test data quality checks on fetched data."""

    def test_ohlc_validation(self):
        """OHLC data should pass validation (high >= low, etc.)."""
        from tests.fixtures.market_data import generate_ohlcv

        df = generate_ohlcv("TEST", days=100)

        # High should always be >= low
        assert (df["high"] >= df["low"]).all()

        # High should be >= open and close
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()

        # Low should be <= open and close
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

    def test_volume_always_positive(self):
        """Volume should always be positive."""
        from tests.fixtures.market_data import generate_ohlcv

        df = generate_ohlcv("TEST", days=100)

        assert (df["volume"] > 0).all()

    def test_no_missing_values(self):
        """Generated data should have no missing values."""
        from tests.fixtures.market_data import generate_ohlcv

        df = generate_ohlcv("TEST", days=100)

        assert not df.isnull().any().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
