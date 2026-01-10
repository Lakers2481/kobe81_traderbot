"""
Alpaca Intraday Data Provider
=============================

Fetch intraday bars (5min, 15min, 1h) from Alpaca Data API v2.
Used for intraday entry triggers (VWAP reclaim, first-hour high/low).

FIX (2026-01-04): Switched from urllib.request to requests library with
retry logic using core.rate_limiter.with_retry for resilience against
transient network errors and rate limits.

Usage:
    from data.providers.alpaca_intraday import fetch_intraday_bars, get_session_vwap

    # Get today's 5-min bars
    df = fetch_intraday_bars("AAPL", timeframe="5Min", limit=78)

    # Get session VWAP
    vwap = get_session_vwap("AAPL")
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

import requests

from core.rate_limiter import with_retry

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


def fetch_intraday_bars(
    symbol: str,
    timeframe: str = "5Min",
    limit: int = 78,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[IntradayBar]:
    """
    Fetch intraday bars from Alpaca Data API.

    Args:
        symbol: Stock ticker symbol
        timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour)
        limit: Maximum number of bars to return
        start: Start timestamp (ISO format), defaults to today's market open
        end: End timestamp (ISO format), defaults to now

    Returns:
        List of IntradayBar objects, sorted by timestamp ascending
    """
    cache_key = f"bars:{symbol}:{timeframe}:{limit}"
    cached = _get_from_cache(cache_key)
    if cached is not None:
        return cached

    api_key = os.getenv("ALPACA_API_KEY_ID", "")
    api_secret = os.getenv("ALPACA_API_SECRET_KEY", "")

    if not api_key or not api_secret:
        logger.warning("Alpaca credentials not configured for intraday data")
        return []

    # Use data API (not trading API)
    base_url = "https://data.alpaca.markets/v2"

    # Build URL
    url = f"{base_url}/stocks/{symbol}/bars"
    params: Dict[str, Any] = {
        "timeframe": timeframe,
        "limit": limit,
    }
    if start:
        params["start"] = start
    if end:
        params["end"] = end

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    def do_request() -> dict:
        """Make the API request (called by with_retry)."""
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    try:
        # FIX (2026-01-04): Use with_retry for resilience against transient errors
        data = with_retry(do_request, max_retries=3, base_delay_ms=500)

        bars_data = data.get("bars", [])
        bars = []
        for bar in bars_data:
            bars.append(IntradayBar(
                timestamp=datetime.fromisoformat(bar["t"].replace("Z", "+00:00")),
                open=bar["o"],
                high=bar["h"],
                low=bar["l"],
                close=bar["c"],
                volume=bar["v"],
                vwap=bar.get("vw", 0.0),
                trade_count=bar.get("n", 0),
            ))

        # Sort by timestamp ascending
        bars.sort(key=lambda b: b.timestamp)

        _set_cache(cache_key, bars)
        return bars

    except Exception as e:
        logger.warning(f"Failed to fetch intraday bars for {symbol} after retries: {e}")
        return []


def get_session_vwap(symbol: str) -> Optional[float]:
    """
    Calculate session VWAP from today's intraday bars.

    Returns the latest VWAP value from Alpaca's bars (they include VWAP).
    """
    bars = fetch_intraday_bars(symbol, timeframe="5Min", limit=78)
    if not bars:
        return None

    # Alpaca provides VWAP in each bar - use the latest
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
