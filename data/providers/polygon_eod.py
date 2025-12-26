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

POLYGON_AGGS_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
POLYGON_OPTIONS_CONTRACTS_URL = "https://api.polygon.io/v3/reference/options/contracts"


@dataclass
class PolygonConfig:
    api_key: str
    adjusted: bool = True
    sort: str = "asc"
    limit: int = 50000
    rate_sleep_sec: float = 0.30


def fetch_daily_bars_polygon(
    symbol: str,
    start: str,
    end: str,
    cache_dir: Optional[Path] = None,
    cfg: Optional[PolygonConfig] = None,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars from Polygon in [start,end] (YYYY-MM-DD) and return a DataFrame
    with columns: timestamp, symbol, open, high, low, close, volume.
    Caches to CSV if cache_dir is provided.
    """
    if cfg is None:
        api_key = os.getenv('POLYGON_API_KEY', '')
        if not api_key:
            return pd.DataFrame(columns=['timestamp','symbol','open','high','low','close','volume'])
        cfg = PolygonConfig(api_key=api_key)

    # Cache handling
    cache_file: Optional[Path] = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{symbol}_{start}_{end}.csv"
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, parse_dates=['timestamp'])
                return df
            except Exception:
                pass
        # Look for a superset cached range and slice it
        try:
            s_req = pd.to_datetime(start)
            e_req = pd.to_datetime(end)
            pattern = re.compile(rf"^{re.escape(symbol)}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$")
            for f in cache_dir.glob(f"{symbol}_*.csv"):
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
                    last_exc = None
                    break
                data = resp.json()
                results = data.get('results', [])
                if not results:
                    return pd.DataFrame(columns=['timestamp','symbol','open','high','low','close','volume'])
                rows = []
                for r in results:
                    ts = pd.to_datetime(r.get('t'), unit='ms')
                    rows.append({
                        'timestamp': ts,
                        'symbol': symbol.upper(),
                        'open': float(r.get('o', 0)),
                        'high': float(r.get('h', 0)),
                        'low': float(r.get('l', 0)),
                        'close': float(r.get('c', 0)),
                        'volume': float(r.get('v', 0)),
                    })
                df = pd.DataFrame(rows)
                if cache_file:
                    try:
                        df.to_csv(cache_file, index=False)
                    except Exception:
                        pass
                return df
            except Exception as e:
                last_exc = e
                time.sleep(0.75 * (attempt + 1))
                continue
        # if we get here, either non-200 or exception: return empty
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
            return False
        data = r.json()
        results = data.get('results', [])
        return len(results) > 0
    except Exception:
        return False
