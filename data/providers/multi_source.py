from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import re
import warnings
import logging
import sys
import time
import random
import threading

import pandas as pd

from .polygon_eod import fetch_daily_bars_polygon
from .stooq_eod import fetch_daily_bars_stooq

# Suppress noisy yfinance warnings about "possibly delisted" for market-closed dates
logging.getLogger('yfinance').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# =============================================================================
# Phase 6: Data Provider Resilience (Codex #7)
# TTL Cache + Jittered Backoff + Provider Health Tracking
# =============================================================================

# Global provider health stats (thread-safe via lock)
_provider_stats_lock = threading.Lock()
_provider_stats: Dict[str, Dict[str, Any]] = {
    "polygon": {"success": 0, "failure": 0, "last_success": None, "last_failure": None},
    "yfinance": {"success": 0, "failure": 0, "last_success": None, "last_failure": None},
    "stooq": {"success": 0, "failure": 0, "last_success": None, "last_failure": None},
}

# In-memory TTL cache for fetched data
_ttl_cache_lock = threading.Lock()
_ttl_cache: Dict[str, Tuple[pd.DataFrame, float]] = {}  # {cache_key: (data, expiry_timestamp)}

# Default retry configuration
DEFAULT_RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay_sec": 1.0,
    "max_delay_sec": 30.0,
    "jitter_factor": 0.3,  # ±30% randomization
    "ttl_seconds": 3600,   # 1 hour cache TTL
}


def _calculate_backoff_with_jitter(attempt: int, config: Dict[str, Any] = None) -> float:
    """
    Calculate exponential backoff delay with jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration dict

    Returns:
        Delay in seconds with jitter applied
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG

    base_delay = config.get("base_delay_sec", 1.0)
    max_delay = config.get("max_delay_sec", 30.0)
    jitter_factor = config.get("jitter_factor", 0.3)

    # Exponential backoff: base * 2^attempt
    delay = min(base_delay * (2 ** attempt), max_delay)

    # Apply jitter: delay * (1 ± jitter_factor)
    jitter_range = delay * jitter_factor
    jitter = random.uniform(-jitter_range, jitter_range)

    return max(0.1, delay + jitter)


def _get_cache_key(symbol: str, start: str, end: str) -> str:
    """Generate a cache key for the data request."""
    return f"{symbol}_{start}_{end}"


def _check_ttl_cache(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Check TTL cache for valid data.

    Returns:
        Cached DataFrame if valid and not expired, None otherwise
    """
    cache_key = _get_cache_key(symbol, start, end)

    with _ttl_cache_lock:
        if cache_key in _ttl_cache:
            data, expiry = _ttl_cache[cache_key]
            if time.time() < expiry:
                logger.debug(f"TTL cache hit for {symbol}")
                return data.copy()
            else:
                # Expired - remove from cache
                del _ttl_cache[cache_key]
                logger.debug(f"TTL cache expired for {symbol}")

    return None


def _store_ttl_cache(symbol: str, start: str, end: str, data: pd.DataFrame, ttl_seconds: int = 3600) -> None:
    """Store data in TTL cache."""
    if data is None or data.empty:
        return

    cache_key = _get_cache_key(symbol, start, end)
    expiry = time.time() + ttl_seconds

    with _ttl_cache_lock:
        _ttl_cache[cache_key] = (data.copy(), expiry)

    logger.debug(f"Cached {len(data)} bars for {symbol} (TTL: {ttl_seconds}s)")


def _record_provider_result(provider: str, success: bool) -> None:
    """Record success/failure for a provider."""
    with _provider_stats_lock:
        if provider in _provider_stats:
            now = datetime.now().isoformat()
            if success:
                _provider_stats[provider]["success"] += 1
                _provider_stats[provider]["last_success"] = now
            else:
                _provider_stats[provider]["failure"] += 1
                _provider_stats[provider]["last_failure"] = now


def get_provider_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get provider health statistics.

    Returns:
        Dict with success/failure counts and success rate per provider
    """
    with _provider_stats_lock:
        stats = {}
        for provider, data in _provider_stats.items():
            total = data["success"] + data["failure"]
            success_rate = data["success"] / total if total > 0 else 0.0
            stats[provider] = {
                "success": data["success"],
                "failure": data["failure"],
                "success_rate": round(success_rate, 4),
                "last_success": data["last_success"],
                "last_failure": data["last_failure"],
            }
        return stats


def reset_provider_stats() -> None:
    """Reset provider statistics (for testing)."""
    with _provider_stats_lock:
        for provider in _provider_stats:
            _provider_stats[provider] = {
                "success": 0,
                "failure": 0,
                "last_success": None,
                "last_failure": None,
            }


def clear_ttl_cache() -> int:
    """
    Clear all TTL cache entries.

    Returns:
        Number of entries cleared
    """
    with _ttl_cache_lock:
        count = len(_ttl_cache)
        _ttl_cache.clear()
        return count


class _SuppressOutput:
    """Context manager to suppress stdout/stderr (for noisy yfinance prints)."""
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def _to_date_str(d: str | pd.Timestamp) -> str:
    return (pd.to_datetime(d).date()).isoformat()


def _yf_symbol(sym: str) -> str:
    # Yahoo uses '-' for BRK-B, BF-B, etc., while our universe may use 'BRK.B'
    return sym.replace('.', '-')


def fetch_daily_bars_yfinance(symbol: str, start: str, end: str, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Yahoo Finance via yfinance with auto-adjusted prices.
    Returns DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    """
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return pd.DataFrame(columns=['timestamp','symbol','open','high','low','close','volume'])

    cache_file: Optional[Path] = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{symbol}_{start}_{end}_yf.csv"
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, parse_dates=['timestamp'])
                return df
            except Exception:
                pass
        # superset search for yf cache
        try:
            s_req = pd.to_datetime(start)
            e_req = pd.to_datetime(end)
            pattern = re.compile(rf"^{re.escape(symbol)}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{4}}-\d{{2}}-\d{{2}})_yf\.csv$")
            # DETERMINISM FIX: Sort glob results for consistent cache file selection
            for f in sorted(cache_dir.glob(f"{symbol}_*_yf.csv")):
                m = pattern.match(f.name)
                if not m:
                    continue
                s_file = pd.to_datetime(m.group(1))
                e_file = pd.to_datetime(m.group(2))
                if s_file <= s_req and e_file >= e_req:
                    try:
                        big = pd.read_csv(f, parse_dates=['timestamp'])
                        big = big[(pd.to_datetime(big['timestamp']) >= s_req) & (pd.to_datetime(big['timestamp']) <= e_req)]
                        return big
                    except Exception:
                        continue
        except Exception:
            pass

    tkr = yf.Ticker(_yf_symbol(symbol))
    try:
        # Suppress yfinance print output for market-closed dates (holidays/weekends)
        with _SuppressOutput(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = tkr.history(start=start, end=end, interval='1d', auto_adjust=True)
    except Exception:
        return pd.DataFrame(columns=['timestamp','symbol','open','high','low','close','volume'])
    if df is None or df.empty:
        return pd.DataFrame(columns=['timestamp','symbol','open','high','low','close','volume'])
    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    df = df.reset_index().rename(columns={'Date':'timestamp'})
    df['symbol'] = symbol.upper()
    out = df[['timestamp','symbol','open','high','low','close','volume']].copy()
    # Normalize timestamp to date (no timezone drift) to match our engine expectations
    out['timestamp'] = pd.to_datetime(out['timestamp']).dt.normalize()
    if cache_file is not None:
        try:
            out.to_csv(cache_file, index=False)
        except Exception:
            pass
    return out


def fetch_daily_bars_stooq(symbol: str, start: str, end: str, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Stooq via pandas_datareader. Stooq prices are adjusted.
    Returns DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    """
    try:
        from pandas_datareader import data as web  # type: ignore
    except Exception:
        return pd.DataFrame(columns=['timestamp','symbol','open','high','low','close','volume'])

    cache_file: Optional[Path] = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{symbol}_{start}_{end}_stooq.csv"
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, parse_dates=['timestamp'])
                return df
            except Exception:
                pass
        # superset search for stooq cache
        try:
            s_req = pd.to_datetime(start)
            e_req = pd.to_datetime(end)
            pattern = re.compile(rf"^{re.escape(symbol)}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{4}}-\d{{2}}-\d{{2}})_stooq\.csv$")
            # DETERMINISM FIX: Sort glob results for consistent cache file selection
            for f in sorted(cache_dir.glob(f"{symbol}_*_stooq.csv")):
                m = pattern.match(f.name)
                if not m:
                    continue
                s_file = pd.to_datetime(m.group(1))
                e_file = pd.to_datetime(m.group(2))
                if s_file <= s_req and e_file >= e_req:
                    try:
                        big = pd.read_csv(f, parse_dates=['timestamp'])
                        big = big[(pd.to_datetime(big['timestamp']) >= s_req) & (pd.to_datetime(big['timestamp']) <= e_req)]
                        return big
                    except Exception:
                        continue
        except Exception:
            pass

    try:
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        df = web.DataReader(symbol, 'stooq', s, e)
    except Exception:
        return pd.DataFrame(columns=['timestamp','symbol','open','high','low','close','volume'])
    if df is None or df.empty:
        return pd.DataFrame(columns=['timestamp','symbol','open','high','low','close','volume'])
    # Stooq returns columns: Open, High, Low, Close, Volume with Date index (descending)
    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    df = df.reset_index().rename(columns={'Date':'timestamp'})
    df['symbol'] = symbol.upper()
    out = df[['timestamp','symbol','open','high','low','close','volume']].copy()
    out['timestamp'] = pd.to_datetime(out['timestamp']).dt.normalize()
    out = out.sort_values('timestamp').reset_index(drop=True)
    if cache_file is not None:
        try:
            out.to_csv(cache_file, index=False)
        except Exception:
            pass
    return out


def fetch_daily_bars_multi(symbol: str, start: str, end: str, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Combined fetcher: use Polygon for the requested window; if Polygon coverage
    starts after `start`, backfill the earlier missing segment from Stooq.

    FIX (2026-01-06): REMOVED yfinance fallback. yfinance is too slow, rate-limited,
    and causes "possibly delisted" errors. Now only uses Polygon + Stooq.
    """
    # Fetch from Polygon first (preferred)
    dfp = fetch_daily_bars_polygon(symbol, start, end, cache_dir=cache_dir)
    # Determine if we need older backfill
    need_backfill = False
    missing_end = None
    if dfp is None or dfp.empty:
        need_backfill = True
        missing_end = end
    else:
        earliest = pd.to_datetime(dfp['timestamp']).min()
        req_start = pd.to_datetime(start)
        if pd.isna(earliest) or earliest > req_start:
            need_backfill = True
            missing_end = (earliest - pd.Timedelta(days=1)).date().isoformat()

    if need_backfill:
        # FIX: Skip yfinance entirely - go straight to stooq for backfill
        # yfinance causes "possibly delisted" errors and is rate-limited
        dfo = fetch_daily_bars_stooq(symbol, start, missing_end, cache_dir=cache_dir)
    else:
        dfo = pd.DataFrame(columns=['timestamp','symbol','open','high','low','close','volume'])

    if dfp is None or dfp.empty:
        return dfo
    if dfo is None or dfo.empty:
        return dfp

    # Concatenate and prefer Polygon on overlaps
    # Normalize timestamps to tz-naive before merging to avoid comparison errors
    dfo['timestamp'] = pd.to_datetime(dfo['timestamp'], utc=True).dt.tz_localize(None)
    dfp['timestamp'] = pd.to_datetime(dfp['timestamp'], utc=True).dt.tz_localize(None)
    dfo['__src'] = 'yf'
    dfp['__src'] = 'poly'
    merged = pd.concat([dfo, dfp], ignore_index=True)
    merged['timestamp'] = pd.to_datetime(merged['timestamp'], utc=True).dt.tz_localize(None)
    merged = merged.sort_values(['timestamp','__src'])
    merged = merged.drop_duplicates(subset=['timestamp'], keep='last')  # keep Polygon rows when overlapping
    merged = merged.drop(columns=['__src'])
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    # Bound to [start, end] - compare on date only to avoid timezone hour issues
    # (raw timestamps may be at 05:00 UTC, but end date should include the whole day)
    s = pd.to_datetime(start).date()
    e = pd.to_datetime(end).date()
    merged = merged[(merged['timestamp'].dt.date >= s) & (merged['timestamp'].dt.date <= e)]
    return merged


def fetch_daily_bars_resilient(
    symbol: str,
    start: str,
    end: str,
    cache_dir: Optional[Path] = None,
    provider_order: list = None,
    retry_config: Dict[str, Any] = None,
    use_ttl_cache: bool = True,
) -> pd.DataFrame:
    """
    Resilient data fetcher with TTL cache, jittered backoff, and provider fallback.

    Phase 6 Implementation (Codex #7):
    - Checks TTL cache first (configurable TTL)
    - Tries each provider with exponential backoff + jitter
    - Tracks provider success/failure rates
    - Falls back to next provider on failure

    FIX (2026-01-06): Removed yfinance from default order - too slow, rate-limited,
    causes "possibly delisted" errors. Use Polygon + Stooq only.

    Args:
        symbol: Stock symbol
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        cache_dir: Optional cache directory for file-based caching
        provider_order: List of providers to try ["polygon", "stooq"]
        retry_config: Retry configuration dict
        use_ttl_cache: Whether to use in-memory TTL cache

    Returns:
        DataFrame with OHLCV data

    Raises:
        DataFetchError: If all providers fail
    """
    if provider_order is None:
        # FIX: Removed yfinance - causes rate limit errors and "possibly delisted" spam
        provider_order = ["polygon", "stooq"]
    if retry_config is None:
        retry_config = DEFAULT_RETRY_CONFIG

    # Check TTL cache first
    if use_ttl_cache:
        cached = _check_ttl_cache(symbol, start, end)
        if cached is not None and not cached.empty:
            return cached

    max_retries = retry_config.get("max_retries", 3)
    ttl_seconds = retry_config.get("ttl_seconds", 3600)

    # Provider function mapping
    provider_funcs = {
        "polygon": lambda: fetch_daily_bars_polygon(symbol, start, end, cache_dir=cache_dir),
        "yfinance": lambda: fetch_daily_bars_yfinance(symbol, start, end, cache_dir=cache_dir),
        "stooq": lambda: fetch_daily_bars_stooq(symbol, start, end, cache_dir=cache_dir),
    }

    last_error = None
    fallback_count = 0

    for provider in provider_order:
        if provider not in provider_funcs:
            logger.warning(f"Unknown provider: {provider}")
            continue

        fetch_func = provider_funcs[provider]

        for attempt in range(max_retries):
            try:
                logger.debug(f"Fetching {symbol} from {provider} (attempt {attempt + 1}/{max_retries})")
                data = fetch_func()

                if data is not None and not data.empty:
                    _record_provider_result(provider, success=True)

                    # Store in TTL cache
                    if use_ttl_cache:
                        _store_ttl_cache(symbol, start, end, data, ttl_seconds)

                    # Log fallback usage
                    if fallback_count > 0:
                        logger.info(f"Fetched {symbol} from {provider} after {fallback_count} fallback(s)")

                    return data

                # Empty data - treat as failure
                _record_provider_result(provider, success=False)
                logger.debug(f"Empty data from {provider} for {symbol}")

            except Exception as e:
                _record_provider_result(provider, success=False)
                last_error = e
                logger.debug(f"Provider {provider} failed for {symbol}: {e}")

                # Apply backoff before retry (except on last attempt)
                if attempt < max_retries - 1:
                    delay = _calculate_backoff_with_jitter(attempt, retry_config)
                    logger.debug(f"Retry in {delay:.2f}s...")
                    time.sleep(delay)

        # Provider exhausted, move to next
        fallback_count += 1
        logger.debug(f"Provider {provider} exhausted for {symbol}, trying next...")

    # All providers failed
    logger.warning(f"All providers failed for {symbol}: {last_error}")
    return pd.DataFrame(columns=['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])


class DataFetchError(Exception):
    """Raised when all data providers fail."""
    pass


def fetch_daily_bars_resilient_strict(
    symbol: str,
    start: str,
    end: str,
    cache_dir: Optional[Path] = None,
    provider_order: list = None,
    retry_config: Dict[str, Any] = None,
    use_ttl_cache: bool = True,
) -> pd.DataFrame:
    """
    Strict version of resilient fetcher that raises on complete failure.

    Same as fetch_daily_bars_resilient but raises DataFetchError instead
    of returning empty DataFrame when all providers fail.
    """
    result = fetch_daily_bars_resilient(
        symbol=symbol,
        start=start,
        end=end,
        cache_dir=cache_dir,
        provider_order=provider_order,
        retry_config=retry_config,
        use_ttl_cache=use_ttl_cache,
    )

    if result is None or result.empty:
        raise DataFetchError(f"All providers failed for {symbol} ({start} to {end})")

    return result
