"""
Market data generators for testing.

Provides OHLCV data generation with various patterns:
- Random walk (default)
- Bullish trend
- Bearish trend
- Choppy/sideways
- Squeeze (low volatility)
- Gap data (overnight gaps)
- Strategy-triggering data (IBS+RSI, Turtle Soup)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Literal


PatternType = Literal["random", "bullish", "bearish", "choppy", "squeeze"]


def generate_ohlcv(
    symbol: str = "TEST",
    days: int = 250,
    pattern: PatternType = "random",
    start_date: Optional[str] = None,
    base_price: float = 100.0,
    volatility: float = 0.02,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate OHLCV data with specified pattern.

    Args:
        symbol: Stock symbol
        days: Number of trading days
        pattern: Price pattern type
        start_date: Start date (YYYY-MM-DD), defaults to 1 year ago
        base_price: Starting price
        volatility: Daily volatility (standard deviation of returns)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    """
    if seed is not None:
        np.random.seed(seed)

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    dates = pd.date_range(start=start_date, periods=days, freq="B")  # Business days

    # Generate returns based on pattern
    if pattern == "bullish":
        drift = 0.0008  # ~20% annual
        returns = drift + np.random.randn(days) * volatility
    elif pattern == "bearish":
        drift = -0.0008
        returns = drift + np.random.randn(days) * volatility
    elif pattern == "choppy":
        # Mean-reverting with higher volatility
        returns = np.random.randn(days) * volatility * 1.5
        # Add mean reversion
        for i in range(1, len(returns)):
            if returns[i-1] > 0:
                returns[i] -= 0.005
            else:
                returns[i] += 0.005
    elif pattern == "squeeze":
        # Very low volatility
        returns = np.random.randn(days) * volatility * 0.3
    else:  # random
        returns = np.random.randn(days) * volatility

    # Generate prices
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": symbol,
        "open": prices * (1 + np.random.randn(days) * 0.003),
        "high": prices * (1 + np.abs(np.random.randn(days) * 0.008)),
        "low": prices * (1 - np.abs(np.random.randn(days) * 0.008)),
        "close": prices,
        "volume": np.random.randint(500000, 20000000, days),
    })

    # Ensure OHLC constraints
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


def generate_multi_symbol_ohlcv(
    symbols: List[str],
    days: int = 250,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate OHLCV data for multiple symbols.

    Args:
        symbols: List of stock symbols
        days: Number of trading days
        seed: Random seed for reproducibility

    Returns:
        DataFrame with data for all symbols concatenated
    """
    dfs = []
    for i, symbol in enumerate(symbols):
        # Use different seed for each symbol but deterministic
        sym_seed = seed + i if seed is not None else None
        base_price = np.random.randint(50, 500) if seed is None else 100 + i * 50
        df = generate_ohlcv(
            symbol=symbol,
            days=days,
            seed=sym_seed,
            base_price=base_price,
        )
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def generate_signal_triggering_data(
    strategy: Literal["ibs_rsi", "turtle_soup", "dual"] = "dual",
    symbol: str = "TEST",
    days: int = 250,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate data specifically designed to trigger strategy signals.

    Args:
        strategy: Which strategy to trigger
        symbol: Stock symbol
        days: Number of trading days
        seed: Random seed

    Returns:
        DataFrame with data that will trigger signals
    """
    if strategy == "ibs_rsi":
        return generate_ibs_rsi_trigger_data(symbol, days, seed)
    elif strategy == "turtle_soup":
        return generate_turtle_soup_trigger_data(symbol, days, seed)
    else:  # dual
        # Combine patterns - some days trigger IBS+RSI, some trigger Turtle Soup
        df = generate_ohlcv(symbol, days, seed=seed)
        # Modify last few bars to potentially trigger signals
        df = _inject_ibs_rsi_setup(df)
        return df


def generate_ibs_rsi_trigger_data(
    symbol: str = "TEST",
    days: int = 250,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate data that triggers IBS+RSI strategy.

    IBS (Internal Bar Strength) = (Close - Low) / (High - Low)
    Signal triggers when: IBS < 0.08 AND RSI(2) < 5 AND Close > SMA(200)
    """
    if seed is not None:
        np.random.seed(seed)

    # Start with bullish trend (above SMA200)
    df = generate_ohlcv(symbol, days, pattern="bullish", seed=seed)

    # Inject IBS+RSI trigger in last bar
    df = _inject_ibs_rsi_setup(df)

    return df


def _inject_ibs_rsi_setup(df: pd.DataFrame) -> pd.DataFrame:
    """Modify last bars to create IBS+RSI trigger conditions."""
    df = df.copy()

    # Create a sequence of down closes to drive RSI(2) below 5
    # Then create low IBS on final bar
    for i in range(-5, 0):  # Last 5 bars
        idx = len(df) + i
        if idx >= 0:
            prev_close = df.loc[idx-1, "close"] if idx > 0 else df.loc[idx, "open"]
            # Down day
            df.loc[idx, "close"] = prev_close * 0.97  # 3% down
            df.loc[idx, "open"] = prev_close * 0.995
            df.loc[idx, "high"] = prev_close * 1.001
            df.loc[idx, "low"] = prev_close * 0.965

    # Final bar: very low IBS (close near low)
    last_idx = len(df) - 1
    prev_close = df.loc[last_idx-1, "close"]
    df.loc[last_idx, "open"] = prev_close * 0.99
    df.loc[last_idx, "high"] = prev_close * 0.995
    df.loc[last_idx, "low"] = prev_close * 0.96
    df.loc[last_idx, "close"] = prev_close * 0.962  # Close very near low -> IBS ~0.05

    return df


def generate_turtle_soup_trigger_data(
    symbol: str = "TEST",
    days: int = 250,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate data that triggers Turtle Soup strategy.

    Signal triggers when price sweeps below 20-day low by >= 0.3 ATR
    then closes back above.
    """
    if seed is not None:
        np.random.seed(seed)

    df = generate_ohlcv(symbol, days, seed=seed)

    # Calculate 20-day low
    df["low_20"] = df["low"].rolling(20).min()

    # Inject sweep setup in last bar
    last_idx = len(df) - 1
    low_20 = df.loc[last_idx-1, "low_20"]  # Use previous bar's 20-day low

    # Calculate approximate ATR
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            np.abs(df["high"] - df["close"].shift(1)),
            np.abs(df["low"] - df["close"].shift(1))
        )
    )
    atr = df["tr"].rolling(14).mean().iloc[-1]

    # Create sweep: low goes 0.4 ATR below 20-day low, close recovers
    sweep_depth = atr * 0.4
    df.loc[last_idx, "low"] = low_20 - sweep_depth
    df.loc[last_idx, "open"] = low_20 + atr * 0.1
    df.loc[last_idx, "high"] = low_20 + atr * 0.3
    df.loc[last_idx, "close"] = low_20 + atr * 0.2  # Close back above sweep level

    # Drop helper columns
    df = df.drop(columns=["low_20", "tr"], errors="ignore")

    return df


def generate_gap_data(
    symbol: str = "TEST",
    gap_percent: float = 5.0,
    gap_direction: Literal["up", "down"] = "up",
    days: int = 250,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate OHLCV data with an overnight gap.

    Args:
        symbol: Stock symbol
        gap_percent: Gap size as percentage
        gap_direction: Direction of gap
        days: Number of trading days
        seed: Random seed

    Returns:
        DataFrame with gap in last bar
    """
    df = generate_ohlcv(symbol, days, seed=seed)

    last_idx = len(df) - 1
    prev_close = df.loc[last_idx-1, "close"]

    if gap_direction == "up":
        gap_open = prev_close * (1 + gap_percent / 100)
    else:
        gap_open = prev_close * (1 - gap_percent / 100)

    # Adjust last bar for gap
    df.loc[last_idx, "open"] = gap_open
    df.loc[last_idx, "high"] = gap_open * 1.01
    df.loc[last_idx, "low"] = gap_open * 0.99
    df.loc[last_idx, "close"] = gap_open * 1.005

    return df
