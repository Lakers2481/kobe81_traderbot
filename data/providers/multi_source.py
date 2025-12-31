from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Optional
import re
import warnings
import logging
import os
import sys

import pandas as pd

from .polygon_eod import fetch_daily_bars_polygon

# Suppress noisy yfinance warnings about "possibly delisted" for market-closed dates
logging.getLogger('yfinance').setLevel(logging.ERROR)


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
            for f in cache_dir.glob(f"{symbol}_*_yf.csv"):
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
            for f in cache_dir.glob(f"{symbol}_*_stooq.csv"):
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
        import datetime as dt
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
    starts after `start`, backfill the earlier missing segment from Yahoo Finance.
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
        dfo = fetch_daily_bars_yfinance(symbol, start, missing_end, cache_dir=cache_dir)
        if dfo is None or dfo.empty:
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
