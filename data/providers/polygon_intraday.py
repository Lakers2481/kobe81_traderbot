"""
Polygon Intraday Data Provider
==============================

Fetch intraday bars (1min, 5min, 15min, 1h) from Polygon.io API.
Used for intraday entry triggers (VWAP reclaim, first-hour high/low).

Requires paid Polygon subscription for real-time/delayed data.

Usage:
    from data.providers.polygon_intraday import fetch_intraday_bars, get_session_vwap

    # Get today's 5-min bars
    df = fetch_intraday_bars("AAPL", timeframe="5Min", limit=78)

    # Get session VWAP
    vwap = get_session_vwap("AAPL")
"""
from __future__ import annotations

import os
import time
import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Simple in-memory cache with TTL
_cache: Dict[str, tuple] = {}  # key -> (data, expiry_time)
CACHE_TTL_SECONDS = 60  # 1 minute cache


@dataclass
class IntradayBar:
    """Single intraday bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    trade_count: int


def _get_from_cache(key: str) -> Optional[Any]:
    """Get data from cache if not expired."""
    if key in _cache:
        data, expiry = _cache[key]
        if time.time() < expiry:
            return data
        del _cache[key]
    return None


def _set_cache(key: str, data: Any, ttl: int = CACHE_TTL_SECONDS) -> None:
    """Set data in cache with TTL."""
    _cache[key] = (data, time.time() + ttl)


def _timeframe_to_polygon(timeframe: str) -> tuple[int, str]:
    """
    Convert timeframe string to Polygon multiplier and timespan.

    Args:
        timeframe: "1Min", "5Min", "15Min", "1Hour"

    Returns:
        (multiplier, timespan) tuple for Polygon API
    """
    mapping = {
        "1Min": (1, "minute"),
        "5Min": (5, "minute"),
        "15Min": (15, "minute"),
        "1Hour": (1, "hour"),
        "1H": (1, "hour"),
    }
    return mapping.get(timeframe, (5, "minute"))


def fetch_intraday_bars(
    symbol: str,
    timeframe: str = "5Min",
    limit: int = 78,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[IntradayBar]:
    """
    Fetch intraday bars from Polygon.io API.

    Args:
        symbol: Stock ticker symbol
        timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour)
        limit: Maximum number of bars to return
        start: Start timestamp (ISO format), defaults to today's market open
        end: End timestamp (ISO format), defaults to now

    Returns:
        List of IntradayBar objects, sorted by timestamp ascending
    """
    cache_key = f"polygon_bars:{symbol}:{timeframe}:{limit}"
    cached = _get_from_cache(cache_key)
    if cached is not None:
        return cached

    api_key = os.getenv("POLYGON_API_KEY", "")

    if not api_key:
        logger.warning("Polygon API key not configured for intraday data")
        return []

    # Convert timeframe to Polygon format
    multiplier, timespan = _timeframe_to_polygon(timeframe)

    # Default to today
    today = date.today()
    if not start:
        start = today.isoformat()
    if not end:
        end = today.isoformat()

    # Build URL
    # GET /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}
    base_url = "https://api.polygon.io/v2"
    url = (
        f"{base_url}/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
        f"?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}"
    )

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if data.get("status") != "OK":
            logger.warning(f"Polygon API error for {symbol}: {data.get('status')}")
            return []

        results = data.get("results", [])
        bars = []

        for bar in results:
            # Polygon returns timestamp in milliseconds
            ts_ms = bar.get("t", 0)
            ts = datetime.fromtimestamp(ts_ms / 1000.0)

            bars.append(IntradayBar(
                timestamp=ts,
                open=bar.get("o", 0.0),
                high=bar.get("h", 0.0),
                low=bar.get("l", 0.0),
                close=bar.get("c", 0.0),
                volume=bar.get("v", 0),
                vwap=bar.get("vw", 0.0),  # Polygon includes VWAP
                trade_count=bar.get("n", 0),
            ))

        # Sort by timestamp ascending (should already be sorted)
        bars.sort(key=lambda b: b.timestamp)

        _set_cache(cache_key, bars)
        return bars

    except urllib.error.HTTPError as e:
        logger.warning(f"Polygon HTTP error for {symbol}: {e.code} {e.reason}")
        return []
    except Exception as e:
        logger.warning(f"Failed to fetch intraday bars for {symbol}: {e}")
        return []


def get_session_vwap(symbol: str) -> Optional[float]:
    """
    Calculate session VWAP from today's intraday bars.

    Returns the latest VWAP value from Polygon's bars (they include VWAP).
    """
    bars = fetch_intraday_bars(symbol, timeframe="5Min", limit=78)
    if not bars:
        return None

    # Polygon provides VWAP in each bar - use the latest
    # This is cumulative session VWAP
    return bars[-1].vwap


def get_first_hour_range(symbol: str) -> Optional[Dict[str, float]]:
    """
    Get the first trading hour's high and low.

    Returns:
        Dict with 'high', 'low', 'open' for first hour, or None if unavailable
    """
    bars = fetch_intraday_bars(symbol, timeframe="5Min", limit=78)
    if not bars:
        return None

    # First hour = first 12 bars of 5-min (9:30 - 10:30 ET)
    first_hour_bars = bars[:12]
    if not first_hour_bars:
        return None

    return {
        "high": max(b.high for b in first_hour_bars),
        "low": min(b.low for b in first_hour_bars),
        "open": first_hour_bars[0].open,
    }


def get_current_price(symbol: str) -> Optional[float]:
    """
    Get the most recent price from intraday bars.
    More reliable than quotes during market hours.
    """
    bars = fetch_intraday_bars(symbol, timeframe="1Min", limit=1)
    if bars:
        return bars[-1].close
    return None


def is_above_vwap(symbol: str) -> Optional[bool]:
    """
    Check if current price is above session VWAP.

    Returns:
        True if above VWAP, False if below, None if data unavailable
    """
    bars = fetch_intraday_bars(symbol, timeframe="5Min", limit=78)
    if not bars:
        return None

    last_bar = bars[-1]
    return last_bar.close > last_bar.vwap


def is_above_first_hour_high(symbol: str) -> Optional[bool]:
    """
    Check if current price is above first hour's high.
    Useful for breakout confirmation.
    """
    bars = fetch_intraday_bars(symbol, timeframe="5Min", limit=78)
    if len(bars) < 12:
        return None  # First hour not complete

    first_hour_high = max(b.high for b in bars[:12])
    current_price = bars[-1].close

    return current_price > first_hour_high


def is_below_first_hour_low(symbol: str) -> Optional[bool]:
    """
    Check if current price is below first hour's low.
    Useful for short entry confirmation.
    """
    bars = fetch_intraday_bars(symbol, timeframe="5Min", limit=78)
    if len(bars) < 12:
        return None  # First hour not complete

    first_hour_low = min(b.low for b in bars[:12])
    current_price = bars[-1].close

    return current_price < first_hour_low


def clear_cache() -> None:
    """Clear all cached intraday data."""
    global _cache
    _cache = {}
