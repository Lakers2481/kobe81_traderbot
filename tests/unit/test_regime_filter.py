"""
Tests for core/regime_filter.py - Market regime filtering.
"""
from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.regime_filter import (
    compute_regime_mask,
    get_allowed_timestamps,
    filter_signals_by_regime,
    get_regime_filter_config,
    is_regime_filter_enabled,
)


def create_spy_bars(
    start_date: str = "2023-01-01",
    num_days: int = 300,
    base_price: float = 400.0,
    trend: float = 0.0005,  # Daily trend
    volatility: float = 0.01,
) -> pd.DataFrame:
    """Create synthetic SPY bars for testing."""
    np.random.seed(42)
    dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    returns = np.random.normal(trend, volatility, num_days)
    prices = base_price * np.cumprod(1 + returns)

    return pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'symbol': 'SPY',
    })


class TestComputeRegimeMask:
    """Tests for compute_regime_mask function."""

    def test_empty_dataframe_returns_empty(self):
        """Returns empty DataFrame for empty input."""
        df = pd.DataFrame(columns=['timestamp', 'close'])
        result = compute_regime_mask(df)
        assert result.empty

    def test_returns_correct_columns(self):
        """Returns DataFrame with timestamp and regime_ok columns."""
        df = create_spy_bars(num_days=250)
        result = compute_regime_mask(df, trend_slow=200)
        assert 'timestamp' in result.columns
        assert 'regime_ok' in result.columns

    def test_trend_filter_above_slow_sma(self):
        """Regime is OK when price is above slow SMA."""
        # Create strongly uptrending data
        df = create_spy_bars(num_days=250, trend=0.002)
        result = compute_regime_mask(df, trend_slow=200, require_above_slow=True)
        # Most recent days should be in OK regime
        recent = result.tail(10)
        assert recent['regime_ok'].sum() >= 5

    def test_trend_filter_below_slow_sma(self):
        """Regime is NOT OK when price is below slow SMA."""
        # Create downtrending data
        df = create_spy_bars(num_days=250, trend=-0.002)
        result = compute_regime_mask(df, trend_slow=200, require_above_slow=True)
        # Most recent days should NOT be in OK regime
        recent = result.tail(10)
        assert recent['regime_ok'].sum() <= 5

    def test_fast_sma_crossover(self):
        """Regime requires fast SMA > slow SMA when both enabled."""
        df = create_spy_bars(num_days=250)
        result = compute_regime_mask(df, trend_slow=200, trend_fast=20)
        assert len(result) > 0
        # Result should have regime_ok values
        assert result['regime_ok'].dtype == bool

    def test_volatility_filter(self):
        """Regime filters by maximum annualized volatility."""
        # Create high volatility data
        df = create_spy_bars(num_days=100, volatility=0.05)  # 5% daily vol
        result = compute_regime_mask(
            df,
            trend_slow=50,
            require_above_slow=False,
            vol_window=20,
            max_ann_vol=0.20,  # 20% annual vol limit
        )
        # High vol should fail the filter
        recent = result.tail(10)
        # With 5% daily vol, annualized is ~80%, so should fail
        assert recent['regime_ok'].sum() == 0

    def test_low_volatility_passes(self):
        """Low volatility passes the volatility filter."""
        # Create low volatility data
        df = create_spy_bars(num_days=100, volatility=0.005)  # 0.5% daily vol
        result = compute_regime_mask(
            df,
            trend_slow=50,
            require_above_slow=False,
            vol_window=20,
            max_ann_vol=0.30,  # 30% annual vol limit
        )
        # Low vol should pass
        recent = result.tail(10)
        assert recent['regime_ok'].sum() > 0


class TestGetAllowedTimestamps:
    """Tests for get_allowed_timestamps function."""

    def test_returns_empty_when_disabled(self):
        """Returns empty set when filter is disabled."""
        with patch('core.regime_filter.get_regime_filter_config') as mock_cfg:
            mock_cfg.return_value = {'enabled': False}
            df = create_spy_bars(num_days=50)
            result = get_allowed_timestamps(df)
            assert result == set()

    def test_returns_allowed_dates(self):
        """Returns set of allowed timestamps."""
        config = {
            'enabled': True,
            'trend_slow': 50,
            'trend_fast': 0,
            'require_above_slow': True,
            'vol_window': 20,
            'max_ann_vol': None,
        }
        # Create uptrending data
        df = create_spy_bars(num_days=100, trend=0.002)
        result = get_allowed_timestamps(df, config)
        # Should have some allowed timestamps
        assert len(result) > 0
        # All should be datetime/Timestamp objects
        for ts in result:
            assert isinstance(ts, (datetime, pd.Timestamp))


class TestFilterSignalsByRegime:
    """Tests for filter_signals_by_regime function."""

    def test_returns_all_when_disabled(self):
        """Returns all signals when filter is disabled."""
        with patch('core.regime_filter.get_regime_filter_config') as mock_cfg:
            mock_cfg.return_value = {'enabled': False}
            signals = pd.DataFrame({
                'timestamp': pd.date_range('2023-06-01', periods=5),
                'symbol': ['AAPL'] * 5,
                'side': ['long'] * 5,
            })
            spy_bars = create_spy_bars()
            result = filter_signals_by_regime(signals, spy_bars)
            assert len(result) == 5

    def test_returns_empty_for_empty_signals(self):
        """Returns empty DataFrame for empty signals."""
        config = {'enabled': True, 'trend_slow': 50}
        signals = pd.DataFrame(columns=['timestamp', 'symbol'])
        spy_bars = create_spy_bars()
        result = filter_signals_by_regime(signals, spy_bars, config)
        assert len(result) == 0

    def test_filters_by_regime(self):
        """Filters signals based on regime."""
        config = {
            'enabled': True,
            'trend_slow': 50,
            'trend_fast': 0,
            'require_above_slow': True,
            'vol_window': 20,
            'max_ann_vol': None,
        }
        # Create SPY bars with clear uptrend at end
        spy_bars = create_spy_bars(num_days=100, trend=0.002, start_date='2023-01-01')

        # Create signals
        signals = pd.DataFrame({
            'timestamp': pd.date_range('2023-03-01', periods=10, freq='D'),
            'symbol': ['AAPL'] * 10,
            'side': ['long'] * 10,
        })

        result = filter_signals_by_regime(signals, spy_bars, config)
        # Should have some signals (those on allowed dates)
        assert isinstance(result, pd.DataFrame)


class TestConfigFunctions:
    """Tests for configuration functions."""

    def test_get_regime_filter_config(self):
        """Config function returns expected structure."""
        config = get_regime_filter_config()
        assert 'enabled' in config
        assert 'trend_fast' in config
        assert 'trend_slow' in config
        assert 'require_above_slow' in config
        assert 'vol_window' in config

    def test_is_regime_filter_enabled(self):
        """is_regime_filter_enabled returns boolean."""
        result = is_regime_filter_enabled()
        assert isinstance(result, bool)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_nan_values(self):
        """Handles NaN values in price data."""
        df = create_spy_bars(num_days=100)
        df.loc[50, 'close'] = np.nan
        result = compute_regime_mask(df, trend_slow=50)
        # Should still return results (with NaNs dropped)
        assert len(result) > 0

    def test_handles_insufficient_data(self):
        """Handles data shorter than lookback period."""
        df = create_spy_bars(num_days=50)  # Less than 200-day SMA
        result = compute_regime_mask(df, trend_slow=200)
        # Should return results but all regime_ok should be False (no valid SMA)
        assert len(result) <= 50
        # All should be False since we can't compute 200-day SMA with 50 days
        assert result['regime_ok'].sum() == 0

    def test_handles_zero_volatility(self):
        """Handles zero volatility (constant prices)."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=50),
            'close': [100.0] * 50,
        })
        result = compute_regime_mask(df, trend_slow=20, vol_window=10)
        # Should handle gracefully
        assert isinstance(result, pd.DataFrame)
