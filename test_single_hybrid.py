#!/usr/bin/env python3
"""
Test single hybrid fetch with full debugging
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
load_env(ROOT / '.env')

import pandas as pd
import yfinance as yf
from data.providers.polygon_eod import fetch_daily_bars_polygon

symbol = 'AAPL'
start = '2015-01-01'
end = '2026-01-08'

print('=' * 70)
print('HYBRID FETCH DEBUG - AAPL')
print('=' * 70)
print()

# Step 1: Fetch Polygon
print('[1] Fetching from Polygon...')
dfp = fetch_daily_bars_polygon(symbol, start, end, cache_dir=Path('data/cache'))
print(f'  Polygon rows: {len(dfp)}')
if not dfp.empty:
    earliest_poly = pd.to_datetime(dfp['timestamp']).min()
    latest_poly = pd.to_datetime(dfp['timestamp']).max()
    print(f'  Polygon range: {earliest_poly.date()} to {latest_poly.date()}')
print()

# Step 2: Calculate backfill
req_start = pd.to_datetime(start)
earliest_poly = pd.to_datetime(dfp['timestamp']).min()
backfill_end = (earliest_poly - pd.Timedelta(days=1)).date().isoformat()

print(f'[2] Backfill calculation:')
print(f'  Requested start: {req_start.date()}')
print(f'  Polygon starts: {earliest_poly.date()}')
print(f'  Backfill period: {start} to {backfill_end}')
print()

# Step 3: Fetch YFinance
print(f'[3] Fetching from YFinance ({start} to {backfill_end})...')
dfy_raw = yf.download(symbol, start=start, end=backfill_end, progress=False)
print(f'  YFinance raw rows: {len(dfy_raw)}')

if not dfy_raw.empty:
    # Convert format
    dfy = dfy_raw.reset_index()
    dfy = dfy.rename(columns={
        'Date': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
    })
    dfy['symbol'] = symbol
    dfy = dfy[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

    print(f'  YFinance formatted rows: {len(dfy)}')
    print(f'  YFinance range: {dfy.iloc[0]["timestamp"]} to {dfy.iloc[-1]["timestamp"]}')
    print()

    # Step 4: Merge
    print('[4] Merging...')

    # Normalize timestamps
    dfy['timestamp'] = pd.to_datetime(dfy['timestamp']).dt.tz_localize(None)
    dfp['timestamp'] = pd.to_datetime(dfp['timestamp']).dt.tz_localize(None)

    print(f'  Before merge: YF={len(dfy)}, Poly={len(dfp)}')

    # Mark source
    dfy['__src'] = 'yf'
    dfp['__src'] = 'poly'

    # Concatenate
    merged = pd.concat([dfy, dfp], ignore_index=True)
    print(f'  After concat: {len(merged)} rows')

    # Sort
    merged = merged.sort_values(['timestamp', '__src'])
    print(f'  After sort: {len(merged)} rows')

    # Drop duplicates
    merged = merged.drop_duplicates(subset=['timestamp'], keep='last')
    print(f'  After dedup: {len(merged)} rows')

    # Remove source column
    merged = merged.drop(columns=['__src'])
    merged = merged.sort_values('timestamp').reset_index(drop=True)

    # Final date filter
    s = pd.to_datetime(start).date()
    e = pd.to_datetime(end).date()
    before_filter = len(merged)
    merged = merged[(merged['timestamp'].dt.date >= s) & (merged['timestamp'].dt.date <= e)]
    print(f'  After date filter ({s} to {e}): {len(merged)} rows (removed {before_filter - len(merged)})')

    print()
    print('[5] Final result:')
    print(f'  Total rows: {len(merged)}')
    first_date = pd.to_datetime(merged.iloc[0]['timestamp']).date()
    last_date = pd.to_datetime(merged.iloc[-1]['timestamp']).date()
    years = (last_date - first_date).days / 365.25
    print(f'  Date range: {first_date} to {last_date}')
    print(f'  Years: {years:.2f}')

    print()
    print('First 5 rows:')
    print(merged.head()[['timestamp', 'symbol', 'close', 'volume']])
    print()
    print('Last 5 rows:')
    print(merged.tail()[['timestamp', 'symbol', 'close', 'volume']])

    if years >= 10.0:
        print()
        print(f'[SUCCESS] {years:.1f} years of data!')
    else:
        print()
        print(f'[FAIL] Only {years:.1f} years - expected 10+')
