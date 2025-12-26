"""
Polygon Crypto Data Provider for Kobe81 Trading Bot.
Fetches hourly OHLCV bars for crypto pairs (X:BTCUSD, etc).
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


RATE_SLEEP_SEC = 0.30  # Conservative rate limiting


def fetch_crypto_bars(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1h",
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch hourly crypto bars from Polygon.

    Args:
        symbol: Crypto ticker (e.g., "X:BTCUSD")
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        timeframe: Bar timeframe (default "1h")
        cache_dir: Optional cache directory

    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    """
    # Normalize symbol
    symbol_upper = symbol.upper()
    if not symbol_upper.startswith("X:"):
        symbol_upper = f"X:{symbol_upper}"

    # Check cache first
    if cache_dir:
        cache_path = _get_cache_path(cache_dir, symbol_upper, start, end, timeframe)
        if cache_path.exists():
            df = pd.read_csv(cache_path, parse_dates=["timestamp"])
            return df

    # Fetch from Polygon
    df = _fetch_from_polygon(symbol_upper, start, end, timeframe)

    # Cache if directory provided
    if cache_dir and not df.empty:
        cache_path = _get_cache_path(cache_dir, symbol_upper, start, end, timeframe)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df


def _get_cache_path(
    cache_dir: Path,
    symbol: str,
    start: str,
    end: str,
    timeframe: str,
) -> Path:
    """Generate cache file path for crypto bars."""
    # Sanitize symbol for filename (X:BTCUSD -> X_BTCUSD)
    safe_symbol = symbol.replace(":", "_")
    filename = f"{safe_symbol}_{start}_{end}_{timeframe}.csv"
    return cache_dir / "crypto" / filename


def _fetch_from_polygon(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1h",
) -> pd.DataFrame:
    """
    Fetch crypto bars from Polygon API.

    Uses: /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        return pd.DataFrame()

    # Parse timeframe (e.g., "1h" -> multiplier=1, timespan=hour)
    multiplier, timespan = _parse_timeframe(timeframe)

    # Convert dates to timestamps
    start_ts = datetime.strptime(start, "%Y-%m-%d")
    end_ts = datetime.strptime(end, "%Y-%m-%d")

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
        f"{multiplier}/{timespan}/{start}/{end}"
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    all_bars = []
    next_url = url

    while next_url:
        try:
            time.sleep(RATE_SLEEP_SEC)

            if "?" in next_url:
                resp = requests.get(next_url, timeout=30)
            else:
                resp = requests.get(next_url, params=params, timeout=30)

            if resp.status_code != 200:
                break

            data = resp.json()
            results = data.get("results", [])

            for bar in results:
                all_bars.append({
                    "timestamp": pd.to_datetime(bar["t"], unit="ms", utc=True),
                    "symbol": symbol,
                    "open": float(bar["o"]),
                    "high": float(bar["h"]),
                    "low": float(bar["l"]),
                    "close": float(bar["c"]),
                    "volume": float(bar.get("v", 0)),
                })

            # Check for pagination
            next_url = data.get("next_url")
            if next_url and api_key not in next_url:
                next_url = f"{next_url}&apiKey={api_key}"

        except Exception:
            break

    if not all_bars:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_bars)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _parse_timeframe(timeframe: str) -> tuple:
    """
    Parse timeframe string to Polygon API format.

    Args:
        timeframe: e.g., "1h", "4h", "1d"

    Returns:
        (multiplier, timespan) tuple
    """
    timeframe = timeframe.lower().strip()

    if timeframe.endswith("h"):
        return int(timeframe[:-1]), "hour"
    elif timeframe.endswith("d"):
        return int(timeframe[:-1]), "day"
    elif timeframe.endswith("m"):
        return int(timeframe[:-1]), "minute"
    else:
        # Default to 1 hour
        return 1, "hour"


def prefetch_crypto_universe(
    symbols: list,
    start: str,
    end: str,
    timeframe: str = "1h",
    cache_dir: Optional[Path] = None,
    concurrency: int = 1,
) -> dict:
    """
    Prefetch crypto bars for a list of symbols.

    Args:
        symbols: List of crypto tickers
        start: Start date
        end: End date
        timeframe: Bar timeframe
        cache_dir: Cache directory
        concurrency: Not used (sequential for rate limiting)

    Returns:
        Dict mapping symbol to row count
    """
    results = {}
    for sym in symbols:
        df = fetch_crypto_bars(sym, start, end, timeframe, cache_dir)
        results[sym] = len(df)
    return results
