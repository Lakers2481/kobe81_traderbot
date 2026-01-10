"""
Regime Filter for Kobe81 Trading Bot.
Config-gated: only active when regime_filter.enabled = true.

Uses SPY daily bars to determine market regime:
- Trend: SPY close > SMA(slow) and optionally SMA(fast) > SMA(slow)
- Volatility: realized volatility <= max threshold
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Dict, Any

import numpy as np
import pandas as pd

from config.settings_loader import get_setting


def get_regime_filter_config() -> Dict[str, Any]:
    """Get regime filter configuration."""
    return {
        "enabled": bool(get_setting("regime_filter.enabled", False)),
        "trend_fast": int(get_setting("regime_filter.trend.fast", 20)),
        "trend_slow": int(get_setting("regime_filter.trend.slow", 200)),
        "require_above_slow": bool(get_setting("regime_filter.trend.require_above_slow", True)),
        "vol_window": int(get_setting("regime_filter.vol.window", 20)),
        "max_ann_vol": get_setting("regime_filter.vol.max_ann_vol", None),
    }


def is_regime_filter_enabled() -> bool:
    """Check if regime filter is enabled."""
    return bool(get_setting("regime_filter.enabled", False))


def compute_regime_mask(
    spy_bars: pd.DataFrame,
    trend_slow: int = 200,
    trend_fast: int = 20,
    require_above_slow: bool = True,
    vol_window: int = 20,
    max_ann_vol: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute regime mask from SPY daily bars.

    Args:
        spy_bars: DataFrame with columns [timestamp, close]
        trend_slow: Slow SMA period (e.g., 200)
        trend_fast: Fast SMA period (e.g., 20), 0 to disable
        require_above_slow: Require close > SMA(slow)
        vol_window: Window for realized volatility
        max_ann_vol: Max annualized volatility threshold (e.g., 0.25)

    Returns:
        DataFrame with [timestamp, regime_ok] where regime_ok=True allows trading
    """
    if spy_bars.empty:
        return pd.DataFrame(columns=["timestamp", "regime_ok"])

    df = spy_bars.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute SMAs
    df["sma_slow"] = df["close"].rolling(window=trend_slow, min_periods=trend_slow).mean()
    if trend_fast > 0:
        df["sma_fast"] = df["close"].rolling(window=trend_fast, min_periods=trend_fast).mean()
    else:
        df["sma_fast"] = np.nan

    # Compute realized volatility (annualized)
    df["returns"] = df["close"].pct_change(fill_method=None)
    df["realized_vol"] = df["returns"].rolling(window=vol_window, min_periods=vol_window).std() * np.sqrt(252)

    # Trend condition: close > SMA(slow)
    trend_ok = pd.Series(True, index=df.index)
    if require_above_slow:
        trend_ok = df["close"] > df["sma_slow"]

    # Optional: fast > slow
    if trend_fast > 0 and not df["sma_fast"].isna().all():
        trend_ok = trend_ok & (df["sma_fast"] > df["sma_slow"])

    # Volatility condition: realized vol <= threshold
    vol_ok = pd.Series(True, index=df.index)
    if max_ann_vol is not None:
        vol_ok = df["realized_vol"] <= max_ann_vol

    # Combined regime
    df["regime_ok"] = trend_ok & vol_ok

    return df[["timestamp", "regime_ok"]].dropna()


def get_allowed_timestamps(
    spy_bars: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Set[datetime]:
    """
    Get set of timestamps where trading is allowed based on regime.

    Args:
        spy_bars: SPY daily bars DataFrame
        config: Optional config dict (uses global config if None)

    Returns:
        Set of allowed timestamps (date only, normalized to midnight)
    """
    if config is None:
        config = get_regime_filter_config()

    if not config.get("enabled", False):
        # If disabled, all timestamps are allowed
        return set()  # Empty set means "no filter applied"

    mask_df = compute_regime_mask(
        spy_bars,
        trend_slow=config.get("trend_slow", 200),
        trend_fast=config.get("trend_fast", 20),
        require_above_slow=config.get("require_above_slow", True),
        vol_window=config.get("vol_window", 20),
        max_ann_vol=config.get("max_ann_vol"),
    )

    # Extract allowed dates
    allowed = mask_df[mask_df["regime_ok"]]["timestamp"]
    # Normalize to date only for matching
    allowed_dates = set(pd.to_datetime(allowed).dt.normalize())
    return allowed_dates


def filter_signals_by_regime(
    signals: pd.DataFrame,
    spy_bars: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Filter signals to only include those in allowed regime periods.
    Config-gated: returns all signals if filter is disabled.

    Args:
        signals: DataFrame with timestamp column
        spy_bars: SPY daily bars for regime calculation
        config: Optional config dict

    Returns:
        Filtered signals DataFrame
    """
    if config is None:
        config = get_regime_filter_config()

    if not config.get("enabled", False):
        return signals

    if signals.empty:
        return signals

    allowed = get_allowed_timestamps(spy_bars, config)
    if not allowed:
        return signals  # No filter applied

    # Normalize signal timestamps to date
    signals = signals.copy()
    signal_dates = pd.to_datetime(signals["timestamp"]).dt.normalize()

    # Filter to allowed dates
    mask = signal_dates.isin(allowed)
    return signals[mask].reset_index(drop=True)


def fetch_spy_bars(
    start: str,
    end: str,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch SPY daily bars with fallback to free sources for robustness.

    Uses the multi-source provider which prefers Polygon but will backfill
    or substitute with Yahoo/Stooq when Polygon coverage is missing.

    Args:
        start: Start date YYYY-MM-DD
        end: End date YYYY-MM-DD
        cache_dir: Optional cache directory

    Returns:
        DataFrame with SPY daily bars
    """
    try:
        from data.providers.multi_source import fetch_daily_bars_multi
        return fetch_daily_bars_multi("SPY", start, end, cache_dir=cache_dir)
    except ImportError:
        try:
            from data.providers.polygon_eod import fetch_daily_bars_polygon
            return fetch_daily_bars_polygon("SPY", start, end, cache_dir=cache_dir)
        except ImportError:
            return pd.DataFrame()
