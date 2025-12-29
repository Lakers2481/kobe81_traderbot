"""
Data loading utilities for ML Alpha Discovery.

Loads trade data from wf_outputs/ and price data from Polygon cache.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_wf_trades(
    wf_dir: str = "wf_outputs",
    strategy: Optional[str] = None,
    min_trades: int = 1,
) -> pd.DataFrame:
    """
    Load trade data from walk-forward output directories.

    Args:
        wf_dir: Path to walk-forward outputs directory
        strategy: Filter to specific strategy (e.g., 'ibs_rsi', 'turtle_soup')
        min_trades: Minimum trades per split to include

    Returns:
        DataFrame with columns: timestamp, symbol, side, entry_price, exit_price,
                                pnl, pnl_pct, won, strategy, split
    """
    wf_path = PROJECT_ROOT / wf_dir
    if not wf_path.exists():
        logger.warning(f"WF directory not found: {wf_path}")
        return pd.DataFrame()

    all_trades = []

    # Iterate through strategy directories
    for strategy_dir in wf_path.iterdir():
        if not strategy_dir.is_dir():
            continue
        if strategy and strategy_dir.name != strategy:
            continue

        # Iterate through split directories
        for split_dir in strategy_dir.iterdir():
            if not split_dir.is_dir() or not split_dir.name.startswith('split_'):
                continue

            trade_file = split_dir / 'trade_list.csv'
            if not trade_file.exists():
                continue

            try:
                df = pd.read_csv(trade_file)
                if len(df) < min_trades:
                    continue

                df['strategy'] = strategy_dir.name
                df['split'] = split_dir.name
                all_trades.append(df)
            except Exception as e:
                logger.warning(f"Error loading {trade_file}: {e}")
                continue

    if not all_trades:
        logger.warning("No trade data found in wf_outputs")
        return pd.DataFrame()

    result = pd.concat(all_trades, ignore_index=True)

    # Ensure required columns exist
    if 'timestamp' in result.columns:
        result['timestamp'] = pd.to_datetime(result['timestamp'])

    # Calculate won column if not present
    if 'won' not in result.columns and 'pnl' in result.columns:
        result['won'] = (result['pnl'] > 0).astype(int)

    logger.info(f"Loaded {len(result)} trades from {len(all_trades)} splits")
    return result


def load_price_data(
    symbols: List[str],
    cache_dir: str = "data/cache",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load price data from Polygon cache for given symbols.

    Args:
        symbols: List of stock symbols
        cache_dir: Path to cache directory
        start: Optional start date filter (YYYY-MM-DD)
        end: Optional end date filter (YYYY-MM-DD)

    Returns:
        Dict mapping symbol -> DataFrame with OHLCV data
    """
    cache_path = PROJECT_ROOT / cache_dir
    if not cache_path.exists():
        logger.warning(f"Cache directory not found: {cache_path}")
        return {}

    price_data = {}

    for symbol in symbols:
        # Find cached file for this symbol
        pattern = f"{symbol}_*.csv"
        matches = list(cache_path.glob(pattern))

        if not matches:
            logger.debug(f"No cached data for {symbol}")
            continue

        # Use most recent cache file
        cache_file = sorted(matches, key=lambda x: x.stat().st_mtime)[-1]

        try:
            df = pd.read_csv(cache_file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Apply date filters
                if start:
                    df = df[df['timestamp'] >= start]
                if end:
                    df = df[df['timestamp'] <= end]

            if len(df) > 0:
                price_data[symbol] = df
        except Exception as e:
            logger.warning(f"Error loading {cache_file}: {e}")
            continue

    logger.info(f"Loaded price data for {len(price_data)} symbols")
    return price_data


def load_cached_bars(
    symbol: str,
    cache_dir: str = "data/cache",
) -> Optional[pd.DataFrame]:
    """
    Load cached OHLCV bars for a single symbol.

    Args:
        symbol: Stock symbol
        cache_dir: Path to cache directory

    Returns:
        DataFrame with OHLCV data or None if not found
    """
    result = load_price_data([symbol], cache_dir)
    return result.get(symbol)


def get_trade_features(
    trade: pd.Series,
    price_data: pd.DataFrame,
    lookback: int = 50,
) -> Dict[str, float]:
    """
    Extract features for a trade from price data.

    Args:
        trade: Trade row with timestamp, symbol, entry_price
        price_data: OHLCV DataFrame for the symbol
        lookback: Number of bars to look back

    Returns:
        Dict of feature name -> value
    """
    features = {}

    if price_data is None or price_data.empty:
        return features

    # Get data up to trade date
    trade_date = pd.to_datetime(trade.get('timestamp'))
    if trade_date is None:
        return features

    historical = price_data[price_data['timestamp'] < trade_date].tail(lookback)

    if len(historical) < 10:
        return features

    close = historical['close'].values
    high = historical['high'].values
    low = historical['low'].values
    volume = historical['volume'].values if 'volume' in historical else None

    # Price features
    features['close'] = float(close[-1])
    features['returns_1d'] = float((close[-1] / close[-2] - 1) if len(close) > 1 else 0)
    features['returns_5d'] = float((close[-1] / close[-5] - 1) if len(close) > 5 else 0)
    features['returns_20d'] = float((close[-1] / close[-20] - 1) if len(close) > 20 else 0)

    # Volatility features
    if len(close) >= 14:
        returns = np.diff(close) / close[:-1]
        features['volatility_14d'] = float(np.std(returns[-14:]) * np.sqrt(252))

    # ATR
    if len(high) >= 14:
        tr = np.maximum(high[1:] - low[1:],
                        np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:] - close[:-1]))
        features['atr_14'] = float(np.mean(tr[-14:]))

    # RSI
    if len(close) >= 15:
        delta = np.diff(close)
        up = np.clip(delta, 0, None)
        down = -np.clip(delta, None, 0)
        avg_up = np.mean(up[-14:])
        avg_down = np.mean(down[-14:])
        rs = avg_up / (avg_down + 1e-10)
        features['rsi_14'] = float(100 - (100 / (1 + rs)))

    # IBS (Internal Bar Strength)
    if len(high) > 0:
        h, l, c = high[-1], low[-1], close[-1]
        features['ibs'] = float((c - l) / (h - l + 1e-10))

    # Moving averages
    if len(close) >= 20:
        features['sma_20'] = float(np.mean(close[-20:]))
        features['price_vs_sma20'] = float(close[-1] / features['sma_20'] - 1)
    if len(close) >= 50:
        features['sma_50'] = float(np.mean(close[-50:]))
        features['price_vs_sma50'] = float(close[-1] / features['sma_50'] - 1)

    # Volume features
    if volume is not None and len(volume) >= 20:
        features['volume_20d_avg'] = float(np.mean(volume[-20:]))
        features['volume_ratio'] = float(volume[-1] / (features['volume_20d_avg'] + 1))

    return features


def load_daily_picks(
    picks_file: str = "logs/daily_picks.csv",
) -> pd.DataFrame:
    """
    Load daily picks/signals from the log file.

    Args:
        picks_file: Path to daily picks CSV

    Returns:
        DataFrame with signal data
    """
    picks_path = PROJECT_ROOT / picks_file
    if not picks_path.exists():
        logger.warning(f"Daily picks file not found: {picks_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(picks_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        logger.warning(f"Error loading daily picks: {e}")
        return pd.DataFrame()


def load_totd_trades(
    totd_dir: str = "reports/totd_2025",
) -> pd.DataFrame:
    """
    Load Trade-of-the-Day results.

    Args:
        totd_dir: Path to TOTD reports directory

    Returns:
        DataFrame with TOTD trade results
    """
    totd_path = PROJECT_ROOT / totd_dir
    if not totd_path.exists():
        logger.warning(f"TOTD directory not found: {totd_path}")
        return pd.DataFrame()

    all_trades_file = totd_path / "all_trades.csv"
    if all_trades_file.exists():
        try:
            df = pd.read_csv(all_trades_file)
            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            logger.warning(f"Error loading TOTD trades: {e}")

    return pd.DataFrame()
