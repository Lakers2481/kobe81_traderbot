"""
Earnings Proximity Filter for Kobe81 Trading Bot.
Config-gated: only active when filters.earnings.enabled = true.

Skips trading signals near earnings announcement dates to avoid
IV crush and gap risk.

FIX (2026-01-04): Updated to use correct Polygon events endpoint
(/v3/reference/tickers/{ticker}/events?types=earnings) instead of
financials endpoint (which gives filing dates, not earnings dates).
Added yfinance fallback for users without Polygon API key.

FIX (2026-01-05): Added source tagging for data provenance tracking
and canary function to detect when sources return zero events.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set
import requests

from config.settings_loader import get_earnings_filter_config

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================

EarningsSource = Literal["polygon", "yfinance", "none"]


@dataclass
class EarningsData:
    """Earnings data with source provenance."""

    dates: List[datetime]
    source: EarningsSource
    fetched_at: datetime


# In-memory cache of earnings dates per symbol
_earnings_cache: Dict[str, EarningsData] = {}


def _get_cache_path() -> Path:
    """Get path to earnings cache file."""
    cfg = get_earnings_filter_config()
    return Path(cfg.get("cache_file", "state/earnings_cache.json"))


def _load_cache() -> Dict[str, Dict]:
    """
    Load earnings cache from disk.

    FIX (2026-01-05): Updated to support new format with source tagging.
    Handles both old format (list of dates) and new format (dict with metadata).
    """
    cache_path = _get_cache_path()
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle old format migration
                migrated = {}
                for symbol, value in data.items():
                    if isinstance(value, list):
                        # Old format: just list of date strings
                        migrated[symbol] = {
                            "dates": value,
                            "source": "unknown",  # Legacy data
                            "fetched_at": None,
                        }
                    elif isinstance(value, dict):
                        # New format
                        migrated[symbol] = value
                    else:
                        # Skip invalid entries
                        continue
                return migrated
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_cache(cache: Dict[str, Dict]) -> None:
    """
    Save earnings cache to disk.

    FIX (2026-01-05): Updated to support new format with source tagging.
    """
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

    FIX (2026-01-05): Now tracks source provenance.

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
        return _earnings_cache[symbol_upper].dates

    # Check disk cache
    disk_cache = _load_cache()
    if not force_refresh and symbol_upper in disk_cache:
        cached = disk_cache[symbol_upper]
        dates = [datetime.fromisoformat(d) for d in cached.get("dates", [])]
        source = cached.get("source", "unknown")
        fetched_at_str = cached.get("fetched_at")
        fetched_at = (
            datetime.fromisoformat(fetched_at_str)
            if fetched_at_str
            else datetime.now()
        )
        _earnings_cache[symbol_upper] = EarningsData(
            dates=dates, source=source, fetched_at=fetched_at
        )
        return dates

    # Fetch from Polygon API (returns tuple: dates, source)
    earnings_dates, source = _fetch_from_polygon_with_source(symbol_upper)

    # Update caches
    now = datetime.now()
    earnings_data = EarningsData(dates=earnings_dates, source=source, fetched_at=now)
    _earnings_cache[symbol_upper] = earnings_data

    disk_cache[symbol_upper] = {
        "dates": [d.isoformat() for d in earnings_dates],
        "source": source,
        "fetched_at": now.isoformat(),
    }
    _save_cache(disk_cache)

    return earnings_dates


def get_earnings_source(symbol: str) -> Optional[EarningsSource]:
    """
    Get the data source for a symbol's earnings data.

    FIX (2026-01-05): Added for source provenance tracking.

    Args:
        symbol: Stock ticker

    Returns:
        Source name ("polygon", "yfinance", "none") or None if not cached
    """
    symbol_upper = symbol.upper()

    # Check memory cache
    if symbol_upper in _earnings_cache:
        return _earnings_cache[symbol_upper].source

    # Check disk cache
    disk_cache = _load_cache()
    if symbol_upper in disk_cache:
        return disk_cache[symbol_upper].get("source")

    return None


def _fetch_from_polygon_with_source(symbol: str) -> tuple[List[datetime], EarningsSource]:
    """
    Fetch earnings dates from Polygon API with source tracking.

    FIX (2026-01-04): Now uses /v3/reference/tickers/{ticker}/events endpoint
    which returns actual earnings announcement dates, instead of /vX/reference/financials
    which only returns SEC filing dates (not useful for earnings blackout).

    FIX (2026-01-05): Returns tuple of (dates, source) for provenance tracking.

    Falls back to yfinance if Polygon API key is not available.
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        logger.debug(f"No POLYGON_API_KEY, falling back to yfinance for {symbol}")
        dates = _fetch_from_yfinance(symbol)
        return (dates, "yfinance" if dates else "none")

    # Use events endpoint for actual earnings announcement dates
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol}/events"
    params = {
        "types": "earnings",
        "limit": 20,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Polygon events API returned {resp.status_code} for {symbol}")
            dates = _fetch_from_yfinance(symbol)
            return (dates, "yfinance" if dates else "none")

        data = resp.json()
        results = data.get("results", {})

        # Events endpoint returns { "events": [...] }
        events = results.get("events", []) if isinstance(results, dict) else []

        earnings_dates = []
        for event in events:
            # Look for earnings_date or date field
            event_date = event.get("earnings_date") or event.get("date")
            if event_date:
                try:
                    # Handle various date formats
                    if "T" in str(event_date):
                        dt = datetime.fromisoformat(str(event_date).replace("Z", ""))
                    else:
                        dt = datetime.strptime(str(event_date), "%Y-%m-%d")
                    earnings_dates.append(dt)
                except ValueError:
                    continue

        if not earnings_dates:
            # Fallback if Polygon returns empty
            logger.debug(f"Polygon returned no earnings for {symbol}, trying yfinance")
            dates = _fetch_from_yfinance(symbol)
            return (dates, "yfinance" if dates else "none")

        return (sorted(set(earnings_dates)), "polygon")

    except Exception as e:
        logger.warning(f"Polygon earnings fetch failed for {symbol}: {e}")
        dates = _fetch_from_yfinance(symbol)
        return (dates, "yfinance" if dates else "none")


def _fetch_from_yfinance(symbol: str) -> List[datetime]:
    """
    Fetch earnings dates from yfinance as fallback.

    Free source that doesn't require API key. Uses the earnings calendar
    from Yahoo Finance.

    Args:
        symbol: Stock ticker

    Returns:
        List of earnings dates (may be empty if unavailable)
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)

        # Try to get earnings calendar
        try:
            calendar = ticker.calendar
            if calendar is not None and not calendar.empty:
                # calendar is a DataFrame with 'Earnings Date' row
                if "Earnings Date" in calendar.index:
                    earnings_date = calendar.loc["Earnings Date"]
                    if hasattr(earnings_date, "iloc"):
                        # Multiple earnings dates
                        dates = []
                        for val in earnings_date:
                            if val is not None:
                                try:
                                    if hasattr(val, "to_pydatetime"):
                                        dates.append(val.to_pydatetime().replace(tzinfo=None))
                                    elif isinstance(val, datetime):
                                        dates.append(val.replace(tzinfo=None))
                                    elif isinstance(val, str):
                                        dates.append(datetime.fromisoformat(val.replace("Z", "")))
                                except (ValueError, AttributeError):
                                    continue
                        if dates:
                            return sorted(set(dates))
        except Exception as e:
            logger.debug(f"yfinance calendar failed for {symbol}: {e}")

        # Try earnings_dates attribute (historical earnings)
        try:
            earnings_df = ticker.earnings_dates
            if earnings_df is not None and not earnings_df.empty:
                dates = []
                for idx in earnings_df.index:
                    try:
                        if hasattr(idx, "to_pydatetime"):
                            dates.append(idx.to_pydatetime().replace(tzinfo=None))
                        elif isinstance(idx, datetime):
                            dates.append(idx.replace(tzinfo=None))
                    except (ValueError, AttributeError):
                        continue
                if dates:
                    return sorted(set(dates))
        except Exception as e:
            logger.debug(f"yfinance earnings_dates failed for {symbol}: {e}")

        return []

    except ImportError:
        logger.warning("yfinance not installed, cannot fetch earnings dates")
        return []
    except Exception as e:
        logger.debug(f"yfinance earnings fetch failed for {symbol}: {e}")
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


# =============================================================================
# Canary Functions (Data Quality Monitoring)
# =============================================================================

# Well-known large-cap stocks that always have quarterly earnings
# Used as canary to detect when data sources are broken
KNOWN_EARNINGS_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


def run_earnings_canary(
    symbols: Optional[List[str]] = None,
    lookback_days: int = 90,
) -> Dict[str, Dict]:
    """
    Run canary check to verify earnings data sources are working.

    FIX (2026-01-05): Added to detect when data sources silently fail.

    If a well-known earnings symbol returns zero events within lookback_days,
    it's a sign the data source is broken. Logs warning and increments
    Prometheus counter.

    Args:
        symbols: Symbols to check (default: KNOWN_EARNINGS_SYMBOLS)
        lookback_days: Days to look back for earnings (default: 90)

    Returns:
        Dict with results per symbol: {"AAPL": {"passed": True, "source": "polygon", ...}}
    """
    from trade_logging.prometheus_metrics import EARNINGS_CANARY_FAILED

    if symbols is None:
        symbols = KNOWN_EARNINGS_SYMBOLS

    results = {}
    now = datetime.now()
    lookback_start = now - timedelta(days=lookback_days)

    for symbol in symbols:
        try:
            # Force refresh to get fresh data
            dates = fetch_earnings_dates(symbol, force_refresh=True)
            source = get_earnings_source(symbol)

            # Check if any earnings dates are within lookback window
            recent_dates = [d for d in dates if d >= lookback_start]

            if not recent_dates:
                # Canary failed - no earnings in last 90 days for known symbol
                logger.warning(
                    f"Earnings canary FAILED for {symbol}: "
                    f"zero events in last {lookback_days} days from source '{source}'"
                )
                EARNINGS_CANARY_FAILED.labels(symbol=symbol, source=source or "unknown").inc()
                results[symbol] = {
                    "passed": False,
                    "source": source,
                    "dates_found": len(dates),
                    "recent_dates": 0,
                    "reason": f"No earnings in last {lookback_days} days",
                }
            else:
                results[symbol] = {
                    "passed": True,
                    "source": source,
                    "dates_found": len(dates),
                    "recent_dates": len(recent_dates),
                    "next_earnings": min(recent_dates).isoformat() if recent_dates else None,
                }

        except Exception as e:
            logger.error(f"Earnings canary error for {symbol}: {e}")
            EARNINGS_CANARY_FAILED.labels(symbol=symbol, source="error").inc()
            results[symbol] = {
                "passed": False,
                "source": "error",
                "dates_found": 0,
                "recent_dates": 0,
                "reason": str(e),
            }

    # Summary log
    passed = sum(1 for r in results.values() if r.get("passed"))
    total = len(results)
    if passed < total:
        logger.warning(f"Earnings canary: {passed}/{total} passed")
    else:
        logger.info(f"Earnings canary: {passed}/{total} passed")

    return results


def check_earnings_source_health() -> Dict[str, any]:
    """
    Check overall health of earnings data sources.

    FIX (2026-01-05): Added for preflight/health checks.

    Returns:
        Dict with:
        - healthy: bool - True if canary passes for majority of symbols
        - canary_results: Dict per symbol
        - source_stats: Dict with source distribution
    """
    canary_results = run_earnings_canary()

    # Calculate source distribution
    sources = {}
    for result in canary_results.values():
        src = result.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    passed = sum(1 for r in canary_results.values() if r.get("passed"))
    total = len(canary_results)

    return {
        "healthy": passed >= (total // 2 + 1),  # Majority must pass
        "passed": passed,
        "total": total,
        "canary_results": canary_results,
        "source_stats": sources,
    }


def clear_cache() -> None:
    """Clear in-memory and disk cache (for testing)."""
    global _earnings_cache
    _earnings_cache = {}
    cache_path = _get_cache_path()
    if cache_path.exists():
        cache_path.unlink()
