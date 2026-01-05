"""
Tests for corporate actions canary (price discontinuity detection).

FIX (2026-01-05): Added for data quality monitoring.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.quality.corporate_actions_canary import (
    DiscontinuityEvent,
    CanaryResult,
    calculate_overnight_returns,
    classify_discontinuity,
    check_symbol_for_splits,
    run_price_discontinuity_check,
    get_canary_summary,
)


def make_ohlcv_df(prices: list, start_date: str = "2025-01-01") -> pd.DataFrame:
    """Helper to create OHLCV DataFrame from price list."""
    dates = pd.date_range(start=start_date, periods=len(prices), freq="D")
    return pd.DataFrame({
        "open": prices,
        "high": [p * 1.02 for p in prices],
        "low": [p * 0.98 for p in prices],
        "close": prices,
        "volume": [1000000] * len(prices),
    }, index=dates)


class TestOvernightReturns:
    """Tests for overnight return calculation."""

    def test_calculates_overnight_returns(self):
        """Calculates overnight returns correctly."""
        df = make_ohlcv_df([100, 102, 101, 105])
        overnight = calculate_overnight_returns(df)

        # First value is NaN (no previous close)
        assert pd.isna(overnight.iloc[0])
        # Second: (102 - 100) / 100 = 2%
        assert abs(overnight.iloc[1] - 2.0) < 0.01

    def test_empty_dataframe(self):
        """Handles empty DataFrame."""
        df = pd.DataFrame()
        overnight = calculate_overnight_returns(df)
        assert len(overnight) == 0


class TestClassifyDiscontinuity:
    """Tests for discontinuity classification."""

    def test_detects_2_to_1_split(self):
        """Classifies ~50% drop as split."""
        assert classify_discontinuity(-50) == "split"
        assert classify_discontinuity(-48) == "split"
        assert classify_discontinuity(-52) == "split"

    def test_detects_3_to_1_split(self):
        """Classifies ~66% drop as split."""
        assert classify_discontinuity(-66) == "split"
        assert classify_discontinuity(-65) == "split"

    def test_detects_reverse_split(self):
        """Classifies ~100% gain as reverse split."""
        assert classify_discontinuity(100) == "reverse_split"
        assert classify_discontinuity(95) == "reverse_split"

    def test_detects_dividend(self):
        """Classifies moderate drops as dividend."""
        assert classify_discontinuity(-10) == "dividend"
        assert classify_discontinuity(-15) == "dividend"

    def test_extreme_change_is_data_error(self):
        """Classifies extreme changes as data error."""
        assert classify_discontinuity(-90) == "data_error"
        assert classify_discontinuity(500) == "data_error"


class TestCheckSymbolForSplits:
    """Tests for single symbol split detection."""

    def test_passes_with_normal_data(self):
        """Passes when no discontinuities present."""
        # Normal price movement (1-2% daily)
        prices = [100, 101, 99, 102, 100, 101]
        df = make_ohlcv_df(prices)

        result = check_symbol_for_splits("AAPL", df)

        assert result.passed is True
        assert len(result.events) == 0
        assert result.symbol == "AAPL"

    def test_detects_split(self):
        """Detects a 2:1 stock split."""
        # Simulate 2:1 split: price drops 50%
        prices = [100, 102, 101, 50, 51, 52]  # Split on day 4
        df = make_ohlcv_df(prices)

        result = check_symbol_for_splits("AAPL", df, threshold_pct=40)

        assert result.passed is False
        assert len(result.events) == 1
        assert result.events[0].likely_cause == "split"
        assert result.events[0].pct_change < -40

    def test_detects_reverse_split(self):
        """Detects a reverse split (1:2)."""
        # Simulate 1:2 reverse split: price doubles
        prices = [50, 51, 52, 100, 102, 101]  # Reverse split on day 4
        df = make_ohlcv_df(prices)

        result = check_symbol_for_splits("XYZ", df, threshold_pct=50)

        assert result.passed is False
        assert len(result.events) == 1
        assert result.events[0].likely_cause == "reverse_split"

    def test_handles_empty_dataframe(self):
        """Handles empty DataFrame gracefully."""
        df = pd.DataFrame()
        result = check_symbol_for_splits("AAPL", df)

        assert result.passed is True
        assert len(result.events) == 0

    def test_handles_short_dataframe(self):
        """Handles DataFrame with only 1 row."""
        df = make_ohlcv_df([100])
        result = check_symbol_for_splits("AAPL", df)

        assert result.passed is True


class TestRunPriceDiscontinuityCheck:
    """Tests for multi-symbol discontinuity check."""

    def test_checks_multiple_symbols(self):
        """Runs checks on multiple symbols."""
        data = {
            "AAPL": make_ohlcv_df([100, 101, 102]),
            "MSFT": make_ohlcv_df([200, 201, 202]),
            "NVDA": make_ohlcv_df([300, 150, 151]),  # Split!
        }

        results = run_price_discontinuity_check(data, threshold_pct=40)

        assert "AAPL" in results
        assert "MSFT" in results
        assert "NVDA" in results
        assert results["AAPL"].passed is True
        assert results["MSFT"].passed is True
        assert results["NVDA"].passed is False

    def test_handles_empty_data(self):
        """Handles empty data dict."""
        results = run_price_discontinuity_check({})
        assert len(results) == 0


class TestGetCanarySummary:
    """Tests for summary generation."""

    def test_generates_summary(self):
        """Generates correct summary statistics."""
        data = {
            "AAPL": make_ohlcv_df([100, 101, 102]),
            "NVDA": make_ohlcv_df([300, 150, 151]),  # Split!
        }

        results = run_price_discontinuity_check(data, threshold_pct=40)
        summary = get_canary_summary(results)

        assert summary["total_symbols"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 0.5
        assert "NVDA" in summary["symbols_flagged"]
        assert summary["total_events"] >= 1

    def test_empty_results(self):
        """Handles empty results."""
        summary = get_canary_summary({})

        assert summary["total_symbols"] == 0
        assert summary["passed"] == 0
        assert summary["pass_rate"] == 0
