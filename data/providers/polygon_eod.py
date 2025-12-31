from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Optional, Dict
import os
import time
import requests
import pandas as pd

from core.structured_log import jlog

POLYGON_AGGS_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
POLYGON_OPTIONS_CONTRACTS_URL = "https://api.polygon.io/v3/reference/options/contracts"

# Cache TTL in seconds (24 hours)
CACHE_TTL_SECONDS = 24 * 60 * 60


@dataclass
class PolygonConfig:
    api_key: str
    adjusted: bool = True
    sort: str = "asc"
    limit: int = 50000
    rate_sleep_sec: float = 0.30


def _is_cache_expired(cache_file: Path, ttl_seconds: int = CACHE_TTL_SECONDS) -> bool:
    """Check if cache file is expired based on TTL."""
    if not cache_file.exists():
        return True
    try:
        age_seconds = time.time() - cache_file.stat().st_mtime
        return age_seconds > ttl_seconds
    except Exception as e:
        jlog("cache_ttl_check_failed", level="WARNING", file=str(cache_file), error=str(e))
        return True


def fetch_daily_bars_polygon(
    symbol: str,
    start: str,
    end: str,
    cache_dir: Optional[Path] = None,
    cfg: Optional[PolygonConfig] = None,
    ignore_cache_ttl: bool = False,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars from Polygon in [start,end] (YYYY-MM-DD) and return a DataFrame
    with columns: timestamp, symbol, open, high, low, close, volume.
    Caches to CSV if cache_dir is provided.

    Args:
        symbol: Stock ticker symbol
        start: Start date YYYY-MM-DD
        end: End date YYYY-MM-DD
        cache_dir: Optional directory for CSV caching
        cfg: Optional PolygonConfig override
        ignore_cache_ttl: If True, use cache even if expired (for backtesting)

    Returns:
        DataFrame with OHLCV data
    """
    if cfg is None:
        api_key = os.getenv('POLYGON_API_KEY', '')
        if not api_key:
            jlog("polygon_no_api_key", level="WARNING", symbol=symbol)
            return pd.DataFrame(columns=['timestamp','symbol','open','high','low','close','volume'])
        cfg = PolygonConfig(api_key=api_key)

    # Cache handling
    cache_file: Optional[Path] = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        # Create a "polygon" subdirectory within the cache_dir
        polygon_cache_dir = cache_dir / "polygon"
        polygon_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = polygon_cache_dir / f"{symbol}_{start}_{end}.csv"

        # Check if cache exists and is not expired
        if cache_file.exists():
            if ignore_cache_ttl or not _is_cache_expired(cache_file):
                try:
                    df = pd.read_csv(cache_file, parse_dates=['timestamp'])
                    return df
                except Exception as e:
                    jlog("cache_read_failed", level="WARNING",
                         file=str(cache_file), error=str(e), symbol=symbol)
            else:
                jlog("cache_expired", level="DEBUG", file=str(cache_file), symbol=symbol)

        # Look for a superset cached range and slice it
        try:
            s_req = pd.to_datetime(start)
            e_req = pd.to_datetime(end)
            pattern = re.compile(rf"^{re.escape(symbol)}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$")
            # DETERMINISM FIX: Sort glob results for consistent cache file selection
            for f in sorted(cache_dir.glob(f"{symbol}_*.csv")):
                # Skip expired superset caches unless ignoring TTL
                if not ignore_cache_ttl and _is_cache_expired(f):
                    continue
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
                    except Exception as e:
                        jlog("superset_cache_read_failed", level="WARNING",
                             file=str(f), error=str(e), symbol=symbol)
                        continue
        except Exception as e:
            jlog("superset_cache_search_failed", level="WARNING",
                 error=str(e), symbol=symbol)

    url = POLYGON_AGGS_URL.format(ticker=symbol.upper(), start=start, end=end)
    params: Dict[str, str|int] = {
        'adjusted': 'true' if cfg.adjusted else 'false',
        'sort': cfg.sort,
        'limit': cfg.limit,
        'apiKey': cfg.api_key,
    }


    try:
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=45)
                if resp.status_code != 200:
                    jlog("polygon_http_error", level="WARNING",
                         symbol=symbol, status_code=resp.status_code, attempt=attempt+1)
                    last_exc = None
                    break
                data = resp.json()


                results = data.get('results', [])
                if not results:
                    return pd.DataFrame(columns=['timestamp','symbol','open','high','low','close','volume'])
                rows = []
                for r in results:

                    ts = pd.to_datetime(r.get('t'), unit='ms')
                    ts = ts.tz_localize(None) # Convert to timezone-naive
                    rows.append({
                        'timestamp': ts,
                        'symbol': symbol.upper(),
                        'open': float(r.get('o', 0)),
                        'high': float(r.get('h', 0)),
                        'low': float(r.get('l', 0)),
                        'close': float(r.get('c', 0)),
                        'volume': float(r.get('v', 0)),
                    })

                df_data = {col: [row[col] for row in rows] for col in ['timestamp','symbol','open','high','low','close','volume']}
                df = pd.DataFrame(df_data, columns=['timestamp','symbol','open','high','low','close','volume'])


                if cache_file:
                    try:
                        df.to_csv(cache_file, index=False)
                    except Exception as e:
                        jlog("cache_write_failed", level="WARNING",
                             file=str(cache_file), error=str(e), symbol=symbol)

                return df
            except Exception as e:
                last_exc = e
                jlog("polygon_request_failed", level="WARNING",
                     symbol=symbol, attempt=attempt+1, error=str(e))

                time.sleep(0.75 * (attempt + 1))
                continue
        # if we get here, either non-200 or exception: return empty
        if last_exc:
            jlog("polygon_all_retries_failed", level="ERROR",
                 symbol=symbol, error=str(last_exc))
        return pd.DataFrame(columns=['timestamp','symbol','open','high','low','close','volume'])
    finally:
        time.sleep(cfg.rate_sleep_sec)


def has_options_polygon(symbol: str, api_key: Optional[str] = None, timeout: int = 10) -> bool:
    """
    Return True if Polygon reports any options contracts for underlying symbol.
    Uses /v3/reference/options/contracts?underlying_ticker=SYMBOL&limit=1
    """
    key = api_key or os.getenv('POLYGON_API_KEY', '')
    if not key:
        return False
    params = {
        'underlying_ticker': symbol.upper(),
        'limit': 1,
        'apiKey': key,
    }
    try:
        r = requests.get(POLYGON_OPTIONS_CONTRACTS_URL, params=params, timeout=timeout)
        if r.status_code != 200:
            jlog("polygon_options_check_failed", level="WARNING",
                 symbol=symbol, status_code=r.status_code)
            return False
        data = r.json()
        results = data.get('results', [])
        return len(results) > 0
    except Exception as e:
        jlog("polygon_options_check_exception", level="WARNING",
             symbol=symbol, error=str(e))
        return False


def clear_expired_cache(cache_dir: Path, ttl_seconds: int = CACHE_TTL_SECONDS) -> int:
    """
    Remove expired cache files from the cache directory.

    Args:
        cache_dir: Directory containing cache files
        ttl_seconds: TTL in seconds

    Returns:
        Number of files removed
    """
    removed = 0
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0

    for f in cache_dir.glob("*.csv"):
        if _is_cache_expired(f, ttl_seconds):
            try:
                f.unlink()
                removed += 1
                jlog("cache_expired_removed", level="DEBUG", file=str(f))
            except Exception as e:
                jlog("cache_remove_failed", level="WARNING", file=str(f), error=str(e))

    return removed
