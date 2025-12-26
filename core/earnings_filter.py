"""
Earnings Proximity Filter for Kobe81 Trading Bot.
Config-gated: only active when filters.earnings.enabled = true.

Skips trading signals near earnings announcement dates to avoid
IV crush and gap risk.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests

from config.settings_loader import get_earnings_filter_config


# In-memory cache of earnings dates per symbol
_earnings_cache: Dict[str, List[datetime]] = {}


def _get_cache_path() -> Path:
    """Get path to earnings cache file."""
    cfg = get_earnings_filter_config()
    return Path(cfg.get("cache_file", "state/earnings_cache.json"))


def _load_cache() -> Dict[str, List[str]]:
    """Load earnings cache from disk."""
    cache_path = _get_cache_path()
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_cache(cache: Dict[str, List[str]]) -> None:
    """Save earnings cache to disk."""
    cache_path = _get_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def fetch_earnings_dates(
    symbol: str,
    force_refresh: bool = False,
) -> List[datetime]:
    """
    Fetch earnings dates for a symbol.
    Uses Polygon API if available, with disk caching.

    Args:
        symbol: Stock ticker
        force_refresh: Force API refresh instead of using cache

    Returns:
        List of earnings announcement datetimes (sorted ascending)
    """
    global _earnings_cache

    symbol_upper = symbol.upper()

    # Check memory cache first
    if not force_refresh and symbol_upper in _earnings_cache:
        return _earnings_cache[symbol_upper]

    # Check disk cache
    disk_cache = _load_cache()
    if not force_refresh and symbol_upper in disk_cache:
        dates = [datetime.fromisoformat(d) for d in disk_cache[symbol_upper]]
        _earnings_cache[symbol_upper] = dates
        return dates

    # Fetch from Polygon API
    earnings_dates = _fetch_from_polygon(symbol_upper)

    # Update caches
    if earnings_dates:
        _earnings_cache[symbol_upper] = earnings_dates
        disk_cache[symbol_upper] = [d.isoformat() for d in earnings_dates]
        _save_cache(disk_cache)

    return earnings_dates


def _fetch_from_polygon(symbol: str) -> List[datetime]:
    """
    Fetch earnings dates from Polygon API.
    Uses the stock financials endpoint.
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        return []

    # Polygon reference data for earnings/financials
    # Note: This is a simplified approach; actual earnings calendar may require
    # a different endpoint or third-party data source
    url = f"https://api.polygon.io/vX/reference/financials?ticker={symbol}&limit=20&apiKey={api_key}"

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []

        data = resp.json()
        results = data.get("results", [])

        earnings_dates = []
        for r in results:
            # filing_date or period_of_report_date
            filing = r.get("filing_date") or r.get("period_of_report_date")
            if filing:
                try:
                    dt = datetime.fromisoformat(filing.replace("Z", ""))
                    earnings_dates.append(dt)
                except ValueError:
                    continue

        return sorted(set(earnings_dates))

    except Exception:
        return []


def is_near_earnings(
    symbol: str,
    check_date: datetime,
    days_before: Optional[int] = None,
    days_after: Optional[int] = None,
) -> bool:
    """
    Check if a date is near an earnings announcement.
    Config-gated: returns False if filter is disabled.

    Args:
        symbol: Stock ticker
        check_date: Date to check
        days_before: Days before earnings to skip (default from config)
        days_after: Days after earnings to skip (default from config)

    Returns:
        True if near earnings (should skip signal), False otherwise
    """
    cfg = get_earnings_filter_config()

    # Config gate: if disabled, never filter
    if not cfg.get("enabled", False):
        return False

    if days_before is None:
        days_before = cfg.get("days_before", 2)
    if days_after is None:
        days_after = cfg.get("days_after", 1)

    earnings_dates = fetch_earnings_dates(symbol)
    if not earnings_dates:
        return False

    check_date_only = check_date.replace(hour=0, minute=0, second=0, microsecond=0)

    for ed in earnings_dates:
        ed_only = ed.replace(hour=0, minute=0, second=0, microsecond=0)
        # Check if check_date falls within the exclusion window
        window_start = ed_only - timedelta(days=days_before)
        window_end = ed_only + timedelta(days=days_after)

        if window_start <= check_date_only <= window_end:
            return True

    return False


def filter_signals_by_earnings(
    signals: List[Dict],
    date_key: str = "timestamp",
    symbol_key: str = "symbol",
) -> List[Dict]:
    """
    Filter a list of signals, removing those near earnings.
    Config-gated: returns all signals if filter is disabled.

    Args:
        signals: List of signal dictionaries
        date_key: Key for datetime in signal dict
        symbol_key: Key for symbol in signal dict

    Returns:
        Filtered list of signals
    """
    cfg = get_earnings_filter_config()
    if not cfg.get("enabled", False):
        return signals

    filtered = []
    for sig in signals:
        symbol = sig.get(symbol_key, "")
        ts = sig.get(date_key)

        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", ""))
            except ValueError:
                filtered.append(sig)
                continue

        if not is_near_earnings(symbol, ts):
            filtered.append(sig)

    return filtered


def get_blackout_dates(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> Set[datetime]:
    """
    Get all blackout dates for a symbol within a date range.
    Useful for backtesting analysis.

    Args:
        symbol: Stock ticker
        start_date: Range start
        end_date: Range end

    Returns:
        Set of dates to exclude from trading
    """
    cfg = get_earnings_filter_config()
    days_before = cfg.get("days_before", 2)
    days_after = cfg.get("days_after", 1)

    earnings_dates = fetch_earnings_dates(symbol)
    blackout = set()

    for ed in earnings_dates:
        ed_only = ed.replace(hour=0, minute=0, second=0, microsecond=0)
        for offset in range(-days_before, days_after + 1):
            blackout_date = ed_only + timedelta(days=offset)
            if start_date <= blackout_date <= end_date:
                blackout.add(blackout_date)

    return blackout


def clear_cache() -> None:
    """Clear in-memory and disk cache (for testing)."""
    global _earnings_cache
    _earnings_cache = {}
    cache_path = _get_cache_path()
    if cache_path.exists():
        cache_path.unlink()
