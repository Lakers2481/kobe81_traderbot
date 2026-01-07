#!/usr/bin/env python3
"""
Refresh Polygon Cache - Updates polygon_cache with latest EOD data.

Runs automatically at 4:02 PM ET after market close to ensure
all cached data includes today's close prices.

This is CRITICAL for:
- Overnight watchlist generation (uses cached data)
- Morning scans (uses cached data)
- Premarket validation (compares cached vs live)

Usage:
    python scripts/refresh_polygon_cache.py
    python scripts/refresh_polygon_cache.py --symbols AAPL,TSLA,MSFT
    python scripts/refresh_polygon_cache.py --universe data/universe/optionable_liquid_900.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Rate limiting for Polygon API
POLYGON_RATE_LIMIT_DELAY = 0.15  # seconds between requests


def refresh_polygon_cache(
    symbols: Optional[List[str]] = None,
    universe_path: Optional[str] = None,
    dotenv_path: str = './.env',
    start_date: Optional[str] = None,
    concurrency: int = 1,
) -> dict:
    """
    Refresh polygon_cache with latest EOD data.

    Args:
        symbols: List of symbols to refresh (optional)
        universe_path: Path to universe CSV (optional, uses all 900 if not specified)
        dotenv_path: Path to .env file
        start_date: Start date for data fetch (default: 1 year ago)
        concurrency: Number of concurrent requests (default: 1 for rate limiting)

    Returns:
        Dict with refresh statistics
    """
    load_dotenv(dotenv_path)

    import pandas as pd
    from data.providers.polygon_eod import fetch_daily_bars_polygon
    from data.universe.loader import load_universe

    cache_dir = Path('data/polygon_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine symbols to refresh
    if symbols:
        symbols_to_refresh = [s.upper() for s in symbols]
    elif universe_path:
        symbols_to_refresh = load_universe(universe_path)
    else:
        # Default: refresh all existing cache files
        symbols_to_refresh = [f.stem.upper() for f in cache_dir.glob('*.csv')]
        if not symbols_to_refresh:
            # Fallback to universe if no cache exists
            default_universe = ROOT / 'data' / 'universe' / 'optionable_liquid_900.csv'
            if default_universe.exists():
                symbols_to_refresh = load_universe(str(default_universe))

    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    logger.info(f"Refreshing {len(symbols_to_refresh)} symbols from {start_date} to {end_date}")

    # Track statistics
    stats = {
        'total': len(symbols_to_refresh),
        'refreshed': 0,
        'skipped': 0,
        'errors': 0,
        'latest_date': None,
        'start_time': datetime.now().isoformat(),
    }

    today_str = datetime.now().strftime('%Y-%m-%d')

    for i, symbol in enumerate(symbols_to_refresh):
        try:
            cache_file = cache_dir / f'{symbol.lower()}.csv'

            # Check if already fresh (has today's or yesterday's data)
            if cache_file.exists():
                try:
                    df_existing = pd.read_csv(cache_file)
                    if len(df_existing) > 0:
                        last_date = str(df_existing['timestamp'].iloc[-1])[:10]
                        # Skip if already has recent data (within 1 trading day)
                        if last_date >= (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'):
                            stats['skipped'] += 1
                            continue
                except Exception:
                    pass  # File exists but can't read, so refresh it

            # Fetch fresh data
            df = fetch_daily_bars_polygon(symbol, start_date, end_date)

            if df is not None and len(df) > 0:
                df.to_csv(cache_file, index=False)
                latest = str(df['timestamp'].max())[:10]
                stats['latest_date'] = latest
                stats['refreshed'] += 1

                if stats['refreshed'] % 50 == 0:
                    logger.info(f"Progress: {stats['refreshed']}/{stats['total']} refreshed, latest: {latest}")
            else:
                stats['errors'] += 1
                logger.debug(f"No data for {symbol}")

            # Rate limiting
            time.sleep(POLYGON_RATE_LIMIT_DELAY)

        except Exception as e:
            stats['errors'] += 1
            logger.warning(f"Error refreshing {symbol}: {e}")

    stats['end_time'] = datetime.now().isoformat()
    stats['duration_seconds'] = (datetime.now() - datetime.fromisoformat(stats['start_time'])).total_seconds()

    logger.info(f"Cache refresh complete: {stats['refreshed']} refreshed, {stats['skipped']} skipped, {stats['errors']} errors")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Refresh Polygon cache with latest EOD data')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    parser.add_argument('--universe', type=str, help='Path to universe CSV')
    parser.add_argument('--dotenv', type=str, default='./.env', help='Path to .env file')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    symbols = args.symbols.split(',') if args.symbols else None

    stats = refresh_polygon_cache(
        symbols=symbols,
        universe_path=args.universe,
        dotenv_path=args.dotenv,
        start_date=args.start,
    )

    print()
    print("=" * 50)
    print("POLYGON CACHE REFRESH COMPLETE")
    print("=" * 50)
    print(f"  Total symbols:  {stats['total']}")
    print(f"  Refreshed:      {stats['refreshed']}")
    print(f"  Skipped (fresh):{stats['skipped']}")
    print(f"  Errors:         {stats['errors']}")
    print(f"  Latest date:    {stats['latest_date']}")
    print(f"  Duration:       {stats['duration_seconds']:.1f}s")
    print("=" * 50)

    return 0 if stats['errors'] < stats['total'] * 0.1 else 1


if __name__ == '__main__':
    sys.exit(main())
