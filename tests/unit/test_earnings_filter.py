"""
Tests for core/earnings_filter.py - Earnings proximity filtering.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.earnings_filter import (
    fetch_earnings_dates,
    is_near_earnings,
    filter_signals_by_earnings,
    get_blackout_dates,
    clear_cache,
    _get_cache_path,
)


class TestFetchEarningsDates:
    """Tests for fetch_earnings_dates function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        clear_cache()

    def test_returns_empty_without_api_key_and_yfinance(self):
        """Returns empty list when no API key is set and yfinance fails.

        Note: FIX (2026-01-04) added yfinance fallback, so we need to mock both.
        """
        with patch.dict('os.environ', {'POLYGON_API_KEY': ''}, clear=True):
            # Mock yfinance to fail (simulates yfinance not installed or failing)
            with patch('core.earnings_filter._fetch_from_yfinance', return_value=[]):
                result = fetch_earnings_dates('AAPL')
                assert result == []

    def test_falls_back_to_yfinance_without_api_key(self):
        """Falls back to yfinance when no Polygon API key is set.

        FIX (2026-01-04): Added yfinance fallback for users without Polygon API key.
        """
        mock_dates = [datetime(2024, 1, 15)]
        with patch.dict('os.environ', {'POLYGON_API_KEY': ''}, clear=True):
            with patch('core.earnings_filter._fetch_from_yfinance', return_value=mock_dates):
                result = fetch_earnings_dates('AAPL', force_refresh=True)
                assert result == mock_dates

    def test_uses_memory_cache(self):
        """Uses memory cache for repeated calls."""
        # Prime the cache with mock data
        from core import earnings_filter
        test_dates = [datetime(2024, 1, 15), datetime(2024, 4, 15)]
        earnings_filter._earnings_cache['AAPL'] = test_dates

        # Should return cached data without API call
        result = fetch_earnings_dates('AAPL')
        assert result == test_dates

        # Cleanup
        clear_cache()

    def test_normalizes_symbol_to_uppercase(self):
        """Symbol is normalized to uppercase."""
        from core import earnings_filter
        test_dates = [datetime(2024, 1, 15)]
        earnings_filter._earnings_cache['AAPL'] = test_dates

        # Lowercase should still find it
        result = fetch_earnings_dates('aapl')
        assert result == test_dates

        clear_cache()


class TestIsNearEarnings:
    """Tests for is_near_earnings function."""

    def setup_method(self):
        """Clear cache and set up mock data."""
        clear_cache()
        from core import earnings_filter
        # Set up mock earnings dates
        earnings_filter._earnings_cache['TEST'] = [
            datetime(2024, 1, 15),  # Earnings on Jan 15
            datetime(2024, 4, 20),  # Earnings on Apr 20
        ]

    def teardown_method(self):
        clear_cache()

    def test_returns_false_when_disabled(self):
        """Returns False when filter is disabled."""
        with patch('core.earnings_filter.get_earnings_filter_config') as mock_cfg:
            mock_cfg.return_value = {'enabled': False}
            result = is_near_earnings('TEST', datetime(2024, 1, 14))
            assert result is False

    def test_detects_day_before_earnings(self):
        """Detects date within days_before window."""
        with patch('core.earnings_filter.get_earnings_filter_config') as mock_cfg:
            mock_cfg.return_value = {'enabled': True, 'days_before': 2, 'days_after': 1}
            # Jan 14 is 1 day before Jan 15 earnings
            result = is_near_earnings('TEST', datetime(2024, 1, 14))
            assert result is True

    def test_detects_day_after_earnings(self):
        """Detects date within days_after window."""
        with patch('core.earnings_filter.get_earnings_filter_config') as mock_cfg:
            mock_cfg.return_value = {'enabled': True, 'days_before': 2, 'days_after': 1}
            # Jan 16 is 1 day after Jan 15 earnings
            result = is_near_earnings('TEST', datetime(2024, 1, 16))
            assert result is True

    def test_detects_earnings_day(self):
        """Detects the actual earnings day."""
        with patch('core.earnings_filter.get_earnings_filter_config') as mock_cfg:
            mock_cfg.return_value = {'enabled': True, 'days_before': 2, 'days_after': 1}
            result = is_near_earnings('TEST', datetime(2024, 1, 15))
            assert result is True

    def test_allows_date_outside_window(self):
        """Allows dates outside the exclusion window."""
        with patch('core.earnings_filter.get_earnings_filter_config') as mock_cfg:
            mock_cfg.return_value = {'enabled': True, 'days_before': 2, 'days_after': 1}
            # Jan 10 is 5 days before earnings - outside window
            result = is_near_earnings('TEST', datetime(2024, 1, 10))
            assert result is False

    def test_returns_false_for_unknown_symbol(self):
        """Returns False for symbols with no earnings data."""
        with patch('core.earnings_filter.get_earnings_filter_config') as mock_cfg:
            mock_cfg.return_value = {'enabled': True, 'days_before': 2, 'days_after': 1}
            result = is_near_earnings('UNKNOWN', datetime(2024, 1, 15))
            assert result is False


class TestFilterSignalsByEarnings:
    """Tests for filter_signals_by_earnings function."""

    def setup_method(self):
        clear_cache()
        from core import earnings_filter
        earnings_filter._earnings_cache['AAPL'] = [datetime(2024, 1, 15)]
        earnings_filter._earnings_cache['MSFT'] = [datetime(2024, 2, 20)]

    def teardown_method(self):
        clear_cache()

    def test_returns_all_when_disabled(self):
        """Returns all signals when filter is disabled."""
        with patch('core.earnings_filter.get_earnings_filter_config') as mock_cfg:
            mock_cfg.return_value = {'enabled': False}
            signals = [
                {'symbol': 'AAPL', 'timestamp': datetime(2024, 1, 14)},
                {'symbol': 'MSFT', 'timestamp': datetime(2024, 2, 19)},
            ]
            result = filter_signals_by_earnings(signals)
            assert len(result) == 2

    def test_filters_near_earnings_signals(self):
        """Filters out signals near earnings."""
        with patch('core.earnings_filter.get_earnings_filter_config') as mock_cfg:
            mock_cfg.return_value = {'enabled': True, 'days_before': 2, 'days_after': 1}
            signals = [
                {'symbol': 'AAPL', 'timestamp': datetime(2024, 1, 14)},  # Near earnings
                {'symbol': 'AAPL', 'timestamp': datetime(2024, 1, 5)},   # Not near
                {'symbol': 'MSFT', 'timestamp': datetime(2024, 2, 10)},  # Not near
            ]
            result = filter_signals_by_earnings(signals)
            assert len(result) == 2
            assert result[0]['timestamp'] == datetime(2024, 1, 5)
            assert result[1]['symbol'] == 'MSFT'

    def test_handles_string_timestamps(self):
        """Handles string timestamps in signals."""
        with patch('core.earnings_filter.get_earnings_filter_config') as mock_cfg:
            mock_cfg.return_value = {'enabled': True, 'days_before': 2, 'days_after': 1}
            signals = [
                {'symbol': 'AAPL', 'timestamp': '2024-01-05T10:00:00'},
            ]
            result = filter_signals_by_earnings(signals)
            assert len(result) == 1


class TestGetBlackoutDates:
    """Tests for get_blackout_dates function."""

    def setup_method(self):
        clear_cache()
        from core import earnings_filter
        earnings_filter._earnings_cache['AAPL'] = [datetime(2024, 1, 15)]

    def teardown_method(self):
        clear_cache()

    def test_returns_blackout_dates(self):
        """Returns all blackout dates in range."""
        with patch('core.earnings_filter.get_earnings_filter_config') as mock_cfg:
            mock_cfg.return_value = {'days_before': 2, 'days_after': 1}
            result = get_blackout_dates(
                'AAPL',
                datetime(2024, 1, 1),
                datetime(2024, 1, 31),
            )
            # Should include Jan 13, 14, 15, 16
            assert len(result) == 4

    def test_respects_date_range(self):
        """Only includes blackout dates within the specified range."""
        with patch('core.earnings_filter.get_earnings_filter_config') as mock_cfg:
            mock_cfg.return_value = {'days_before': 2, 'days_after': 1}
            result = get_blackout_dates(
                'AAPL',
                datetime(2024, 1, 15),  # Start at earnings day
                datetime(2024, 1, 16),  # End day after
            )
            # Should only include Jan 15 and 16
            assert len(result) == 2


class TestClearCache:
    """Tests for clear_cache function."""

    def test_clears_memory_cache(self):
        """Clears in-memory cache."""
        from core import earnings_filter
        earnings_filter._earnings_cache['TEST'] = [datetime(2024, 1, 1)]

        clear_cache()

        assert earnings_filter._earnings_cache == {}

    def test_removes_cache_file(self):
        """Removes disk cache file."""
        cache_path = _get_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text('{}')

        clear_cache()

        assert not cache_path.exists()
