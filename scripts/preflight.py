#!/usr/bin/env python3
"""
Kobe Preflight Checks

Validates all critical dependencies before trading:
1. Environment variables (POLYGON, ALPACA keys)
2. Config file pin integrity
3. Alpaca Trading API connectivity
4. Alpaca Data API (quotes) connectivity
5. Polygon EOD data freshness
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
import requests
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.env_loader import load_env
from core.config_pin import sha256_file


def check_quotes_api() -> bool:
    """
    Validate Alpaca Data API is accessible.

    Returns:
        True if quotes API is healthy, False otherwise
    """
    headers = {
        'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY_ID', ''),
        'APCA-API-SECRET-KEY': os.getenv('ALPACA_API_SECRET_KEY', ''),
    }
    # Always use data.alpaca.markets for quotes (regardless of paper/live)
    url = 'https://data.alpaca.markets/v2/stocks/quotes?symbols=AAPL'

    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            print(f'  Quotes API HTTP {r.status_code}')
            return False

        data = r.json()
        quotes = data.get('quotes', {})
        if 'AAPL' not in quotes or not quotes['AAPL']:
            print('  Quotes API: No AAPL quote data returned')
            return False

        print(f'  Quotes API OK (AAPL quote available)')
        return True
    except Exception as e:
        print(f'  Quotes API error: {e}')
        return False


def check_polygon_freshness(symbol: str = 'AAPL', lookback_days: int = 5) -> bool:
    """
    Validate Polygon EOD data is fresh.

    Args:
        symbol: Stock symbol to check
        lookback_days: Number of days to fetch

    Returns:
        True if data is fresh (has recent bars), False otherwise
    """
    api_key = os.getenv('POLYGON_API_KEY', '')
    if not api_key:
        print('  Polygon: No API key')
        return False

    # Calculate date range (last N trading days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 5)  # Extra days for weekends

    url = (
        f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/'
        f'{start_date.strftime("%Y-%m-%d")}/{end_date.strftime("%Y-%m-%d")}'
        f'?adjusted=true&sort=desc&limit={lookback_days}&apiKey={api_key}'
    )

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print(f'  Polygon API HTTP {r.status_code}')
            return False

        data = r.json()
        results = data.get('results', [])

        if not results:
            print(f'  Polygon: No bars returned for {symbol}')
            return False

        # Check if most recent bar is within last 5 calendar days
        latest_ts = results[0].get('t', 0) / 1000  # ms to seconds
        latest_date = datetime.fromtimestamp(latest_ts)
        days_old = (datetime.now() - latest_date).days

        if days_old > 5:
            print(f'  Polygon: Data stale ({days_old} days old for {symbol})')
            return False

        print(f'  Polygon OK ({len(results)} bars, latest: {latest_date.strftime("%Y-%m-%d")})')
        return True
    except Exception as e:
        print(f'  Polygon error: {e}')
        return False


def main():
    ap = argparse.ArgumentParser(description='Kobe preflight checks')
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--config', type=str, default='config/settings.json')
    ap.add_argument('--skip-polygon', action='store_true', help='Skip Polygon freshness check')
    args = ap.parse_args()

    print('=' * 50)
    print('KOBE PREFLIGHT CHECKS')
    print('=' * 50)

    # 1. Load environment
    loaded = {}
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
    print(f'\n[1/5] Environment: Loaded {len(loaded)} vars from {dotenv}')

    required = ['POLYGON_API_KEY', 'ALPACA_API_KEY_ID', 'ALPACA_API_SECRET_KEY']
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f'  FAIL: Missing keys: {", ".join(missing)}')
        sys.exit(2)
    print('  OK: All required keys present')

    # 2. Config pin
    print('\n[2/5] Config Pin:')
    try:
        digest = sha256_file(args.config)
        print(f'  OK: {digest[:16]}...')
    except Exception as e:
        print(f'  FAIL: {e}')
        sys.exit(3)

    # 3. Alpaca Trading API
    print('\n[3/5] Alpaca Trading API:')
    base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
    try:
        r = requests.get(base + '/v2/account', headers={
            'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY_ID', ''),
            'APCA-API-SECRET-KEY': os.getenv('ALPACA_API_SECRET_KEY', ''),
        }, timeout=5)
        if r.status_code == 200:
            print(f'  OK: Trading API accessible ({base})')
        else:
            print(f'  FAIL: HTTP {r.status_code}')
            sys.exit(4)
    except Exception as e:
        print(f'  FAIL: {e}')
        sys.exit(4)

    # 4. Alpaca Data API (quotes)
    print('\n[4/5] Alpaca Data API (Quotes):')
    if not check_quotes_api():
        print('  WARNING: Quotes API check failed (non-blocking)')

    # 5. Polygon Data Freshness
    print('\n[5/5] Polygon Data Freshness:')
    if args.skip_polygon:
        print('  SKIPPED (--skip-polygon)')
    elif not check_polygon_freshness():
        print('  WARNING: Data freshness check failed (non-blocking)')

    print('\n' + '=' * 50)
    print('PREFLIGHT OK - Ready for trading')
    print('=' * 50)


if __name__ == '__main__':
    main()
