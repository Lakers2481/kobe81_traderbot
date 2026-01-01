"""
Multi-Timeframe (MTF) Utilities for Kobe Trading System

Provides helpers for combining signals across multiple timeframes,
similar to backtesting.py's resample_apply functionality.

Key Features:
- resample_apply(): Apply indicators to resampled data, align back to original
- mtf_filter(): Higher timeframe filter for lower timeframe signals
- Common MTF patterns: weekly RSI filter, monthly trend filter, etc.

Usage:
    from backtest.multi_timeframe import resample_apply, mtf_filter

    # Weekly RSI on daily data
    weekly_rsi = resample_apply(daily_data, 'W', lambda df: ta.rsi(df['close'], 14))

    # Filter daily signals by weekly trend
    filtered = mtf_filter(daily_signals, weekly_data,
                          condition=lambda w: w['close'] > w['sma_50'])
"""

from __future__ import annotations

from typing import Callable, Optional, Union, List
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to title case."""
    col_map = {
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume',
        'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low',
        'CLOSE': 'Close', 'VOLUME': 'Volume',
    }
    return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})


def resample_ohlcv(
    df: pd.DataFrame,
    rule: str = 'W',
    label: str = 'right',
    closed: str = 'right'
) -> pd.DataFrame:
    """
    Resample OHLCV data to a lower frequency (e.g., daily -> weekly).

    Args:
        df: DataFrame with OHLCV columns and DatetimeIndex
        rule: Pandas resample rule ('W' for weekly, 'M' for monthly, etc.)
        label: Which edge to label ('right' or 'left')
        closed: Which edge is closed ('right' or 'left')

    Returns:
        Resampled OHLCV DataFrame
    """
    df = _normalize_columns(df.copy())

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        elif 'date' in df.columns:
            df = df.set_index('date')
        df.index = pd.to_datetime(df.index)

    agg_dict = {}
    if 'Open' in df.columns:
        agg_dict['Open'] = 'first'
    if 'High' in df.columns:
        agg_dict['High'] = 'max'
    if 'Low' in df.columns:
        agg_dict['Low'] = 'min'
    if 'Close' in df.columns:
        agg_dict['Close'] = 'last'
    if 'Volume' in df.columns:
        agg_dict['Volume'] = 'sum'

    resampled = df.resample(rule, label=label, closed=closed).agg(agg_dict)
    return resampled.dropna()


def resample_apply(
    df: pd.DataFrame,
    rule: str,
    func: Callable[[pd.DataFrame], Union[pd.Series, float]],
    column_name: Optional[str] = None,
    ffill: bool = True
) -> pd.Series:
    """
    Apply a function to resampled data and align results back to original timeframe.

    This is the core MTF helper - similar to backtesting.py's resample_apply.
    It resamples data to a higher timeframe, applies an indicator/function,
    then forward-fills the result back to the original timeframe.

    Args:
        df: DataFrame with OHLCV data and DatetimeIndex
        rule: Pandas resample rule ('W', 'M', '2W', etc.)
        func: Function to apply to resampled data. Should return Series or scalar.
              Receives the resampled OHLCV DataFrame.
        column_name: Name for the resulting Series (optional)
        ffill: Whether to forward-fill results to original timeframe (default True)

    Returns:
        Series aligned to original DataFrame's index with MTF indicator values

    Examples:
        # Weekly RSI(14)
        weekly_rsi = resample_apply(daily_df, 'W',
            lambda df: ta.rsi(df['Close'], 14))

        # Monthly SMA(20)
        monthly_sma = resample_apply(daily_df, 'M',
            lambda df: df['Close'].rolling(20).mean())

        # Weekly close
        weekly_close = resample_apply(daily_df, 'W',
            lambda df: df['Close'])
    """
    df = _normalize_columns(df.copy())

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        elif 'date' in df.columns:
            df = df.set_index('date')
        df.index = pd.to_datetime(df.index)

    original_index = df.index

    # Resample to higher timeframe
    resampled = resample_ohlcv(df, rule=rule)

    # Apply the function
    result = func(resampled)

    # Handle scalar result (apply to all rows)
    if isinstance(result, (int, float)):
        result = pd.Series(result, index=resampled.index)

    # Ensure result is a Series with proper index
    if not isinstance(result, pd.Series):
        result = pd.Series(result, index=resampled.index)

    # Align back to original timeframe
    aligned = result.reindex(original_index)

    if ffill:
        aligned = aligned.ffill()

    if column_name:
        aligned.name = column_name

    return aligned


def mtf_sma(
    df: pd.DataFrame,
    rule: str,
    period: int,
    column: str = 'Close'
) -> pd.Series:
    """
    Compute SMA on a higher timeframe, aligned to original data.

    Args:
        df: OHLCV DataFrame
        rule: Resample rule ('W', 'M', etc.)
        period: SMA period on the resampled timeframe
        column: Column to use (default 'Close')

    Returns:
        Series with MTF SMA values
    """
    return resample_apply(
        df, rule,
        lambda x: x[column].rolling(period).mean(),
        column_name=f'sma_{period}_{rule}'
    )


def mtf_rsi(
    df: pd.DataFrame,
    rule: str,
    period: int = 14
) -> pd.Series:
    """
    Compute RSI on a higher timeframe, aligned to original data.

    Args:
        df: OHLCV DataFrame
        rule: Resample rule ('W', 'M', etc.)
        period: RSI period (default 14)

    Returns:
        Series with MTF RSI values
    """
    def calc_rsi(data: pd.DataFrame) -> pd.Series:
        close = data['Close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    return resample_apply(df, rule, calc_rsi, column_name=f'rsi_{period}_{rule}')


def mtf_atr(
    df: pd.DataFrame,
    rule: str,
    period: int = 14
) -> pd.Series:
    """
    Compute ATR on a higher timeframe, aligned to original data.

    Args:
        df: OHLCV DataFrame
        rule: Resample rule ('W', 'M', etc.)
        period: ATR period (default 14)

    Returns:
        Series with MTF ATR values
    """
    def calc_atr(data: pd.DataFrame) -> pd.Series:
        high = data['High']
        low = data['Low']
        close = data['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    return resample_apply(df, rule, calc_atr, column_name=f'atr_{period}_{rule}')


def mtf_trend_filter(
    df: pd.DataFrame,
    rule: str = 'W',
    fast_period: int = 10,
    slow_period: int = 30
) -> pd.Series:
    """
    Higher timeframe trend filter: 1 if fast SMA > slow SMA, else 0.

    Args:
        df: OHLCV DataFrame
        rule: Resample rule ('W', 'M', etc.)
        fast_period: Fast SMA period
        slow_period: Slow SMA period

    Returns:
        Series with 1 (bullish) or 0 (bearish) trend values
    """
    def calc_trend(data: pd.DataFrame) -> pd.Series:
        close = data['Close']
        fast = close.rolling(fast_period).mean()
        slow = close.rolling(slow_period).mean()
        return (fast > slow).astype(int)

    return resample_apply(df, rule, calc_trend, column_name=f'trend_{rule}')


def mtf_filter(
    signals: pd.DataFrame,
    htf_data: pd.DataFrame,
    condition: Callable[[pd.DataFrame], pd.Series],
    rule: str = 'W'
) -> pd.DataFrame:
    """
    Filter signals based on a higher timeframe condition.

    Args:
        signals: DataFrame with signals (must have 'timestamp' column)
        htf_data: Higher timeframe OHLCV data for the condition
        condition: Function that takes HTF data and returns boolean Series
        rule: Resample rule if htf_data needs resampling

    Returns:
        Filtered signals DataFrame (only signals where HTF condition is True)

    Example:
        # Only take signals when weekly close > weekly SMA(20)
        filtered = mtf_filter(
            daily_signals,
            daily_data,  # Will be resampled to weekly
            condition=lambda w: w['Close'] > w['Close'].rolling(20).mean(),
            rule='W'
        )
    """
    if signals.empty:
        return signals

    # Get HTF condition
    htf_resampled = resample_ohlcv(htf_data, rule=rule)
    htf_condition = condition(htf_resampled)

    # Align condition to signal timestamps
    signals = signals.copy()
    if 'timestamp' in signals.columns:
        signal_dates = pd.to_datetime(signals['timestamp'])
    else:
        signal_dates = signals.index

    # For each signal, find the most recent HTF bar
    aligned_condition = htf_condition.reindex(signal_dates, method='ffill')

    # Filter signals
    mask = aligned_condition.values.astype(bool)
    filtered = signals[mask]

    logger.info(f"MTF filter: {len(signals)} signals -> {len(filtered)} after {rule} filter")

    return filtered


def bars_since(
    condition: pd.Series,
    max_lookback: int = 100
) -> pd.Series:
    """
    Count bars since condition was last True.

    Similar to backtesting.py's bars_since helper.

    Args:
        condition: Boolean Series
        max_lookback: Maximum lookback (returns this if condition never True)

    Returns:
        Series with count of bars since condition was True

    Example:
        # Bars since RSI crossed below 30
        bars = bars_since(rsi < 30)
    """
    condition = condition.astype(bool)
    result = pd.Series(index=condition.index, dtype=float)

    count = max_lookback
    for i, val in enumerate(condition):
        if val:
            count = 0
        else:
            count = min(count + 1, max_lookback)
        result.iloc[i] = count

    return result.astype(int)


def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Detect crossover: series1 crosses above series2.

    Args:
        series1: First series (e.g., fast MA)
        series2: Second series (e.g., slow MA)

    Returns:
        Boolean Series, True where crossover occurs
    """
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))


def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Detect crossunder: series1 crosses below series2.

    Args:
        series1: First series (e.g., fast MA)
        series2: Second series (e.g., slow MA)

    Returns:
        Boolean Series, True where crossunder occurs
    """
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))


# Convenience aliases
weekly_sma = lambda df, period: mtf_sma(df, 'W', period)
weekly_rsi = lambda df, period=14: mtf_rsi(df, 'W', period)
monthly_sma = lambda df, period: mtf_sma(df, 'M', period)
monthly_trend = lambda df: mtf_trend_filter(df, 'M')
