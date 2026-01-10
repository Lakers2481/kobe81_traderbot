#!/usr/bin/env python3
"""
Rate-limited yfinance backfill for historical data (2015-2020)
Fills gaps where Polygon free tier doesn't have data.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv


def backfill_symbol(symbol: str, cache_dir: Path, start: str = "2015-01-01", end: str = "2020-12-27", delay: float = 2.0) -> bool:
    """
    Fetch historical data from yfinance for a single symbol.
    Returns True if successful, False otherwise.
    """
    import yfinance as yf

    cache_file = cache_dir / f"{symbol}_{start}_{end}_yf.csv"

    # Skip if already cached
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=['timestamp'])
            if len(df) > 100:  # Valid data
                return True
        except Exception:
            pass

    # Fetch from yfinance
    try:
        # Handle tickers with dots (BRK.B -> BRK-B)
        yf_symbol = symbol.replace('.', '-')
        t = yf.Ticker(yf_symbol)
        df = t.history(start=start, end=end, interval='1d', auto_adjust=True)

        if df is None or df.empty:
            print(f"  {symbol}: no data from yfinance")
            return False

        # Normalize columns
        df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        df = df.reset_index().rename(columns={'Date': 'timestamp'})
        df['symbol'] = symbol.upper()

        # Select and reorder columns
        out = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']].copy()
        out['timestamp'] = pd.to_datetime(out['timestamp']).dt.normalize()

        # Save to cache
        out.to_csv(cache_file, index=False)
        print(f"  {symbol}: {len(out)} rows saved")
        return True

    except Exception as e:
        print(f"  {symbol}: error - {e}")
        return False
    finally:
        time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Rate-limited yfinance backfill")
    parser.add_argument("--universe", type=str, default="data/universe/optionable_liquid_800.csv")
    parser.add_argument("--cache", type=str, default="data/cache")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2020-12-27")
    parser.add_argument("--cap", type=int, default=60, help="Max symbols to fetch")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between requests in seconds")
    parser.add_argument("--dotenv", type=str, default=None)
    args = parser.parse_args()

    if args.dotenv:
        load_dotenv(args.dotenv)

    # Load universe
    universe_path = Path(args.universe)
    if not universe_path.exists():
        print(f"Universe file not found: {universe_path}")
        return 1

    universe = pd.read_csv(universe_path)
    symbols = universe['symbol'].tolist()[:args.cap]

    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backfilling {len(symbols)} symbols from yfinance")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Cache dir: {cache_dir}")
    print(f"Delay: {args.delay}s between requests")
    print()

    success = 0
    failed = 0

    for i, sym in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] {sym}...")
        if backfill_symbol(sym, cache_dir, args.start, args.end, args.delay):
            success += 1
        else:
            failed += 1

    print()
    print(f"Done: {success} success, {failed} failed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
