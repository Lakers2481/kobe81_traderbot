#!/usr/bin/env python3
"""
Debug Hybrid Data Strategy
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
load_env(ROOT / '.env')

import pandas as pd
from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.providers.stooq_eod import fetch_daily_bars_stooq

symbol = 'AAPL'
start = '2015-01-01'
end = '2026-01-08'

print('=' * 70)
print('DEBUGGING HYBRID FETCH')
print('=' * 70)
print()

# Step 1: Fetch from Polygon
print('[STEP 1] Fetching from Polygon (preferred source)...')
dfp = fetch_daily_bars_polygon(symbol, start, end, cache_dir=Path('data/cache'))
print(f'  Polygon rows: {len(dfp)}')
if not dfp.empty:
    earliest_poly = pd.to_datetime(dfp['timestamp']).min()
    latest_poly = pd.to_datetime(dfp['timestamp']).max()
    print(f'  Polygon range: {earliest_poly.date()} to {latest_poly.date()}')
print()

# Step 2: Check if backfill needed
req_start = pd.to_datetime(start)
need_backfill = False
missing_end = None

if dfp is None or dfp.empty:
    print('[STEP 2] Polygon returned NO data → need full backfill')
    need_backfill = True
    missing_end = end
else:
    earliest = pd.to_datetime(dfp['timestamp']).min()
    print(f'[STEP 2] Checking backfill need:')
    print(f'  Requested start: {req_start.date()}')
    print(f'  Polygon earliest: {earliest.date()}')
    print(f'  Gap: {(earliest - req_start).days} days')

    if pd.isna(earliest) or earliest > req_start:
        need_backfill = True
        missing_end = (earliest - pd.Timedelta(days=1)).date().isoformat()
        print(f'  >> BACKFILL NEEDED from {start} to {missing_end}')
    else:
        print(f'  >> No backfill needed (Polygon covers full range)')
print()

# Step 3: Fetch backfill if needed
if need_backfill:
    print(f'[STEP 3] Fetching backfill from Stooq ({start} to {missing_end})...')
    dfo = fetch_daily_bars_stooq(symbol, start, missing_end, cache_dir=Path('data/cache'))
    print(f'  Stooq rows: {len(dfo)}')
    if not dfo.empty:
        earliest_stooq = pd.to_datetime(dfo['timestamp']).min()
        latest_stooq = pd.to_datetime(dfo['timestamp']).max()
        print(f'  Stooq range: {earliest_stooq.date()} to {latest_stooq.date()}')
    else:
        print(f'  [ERROR] Stooq returned NO data!')
else:
    dfo = pd.DataFrame(columns=['timestamp','symbol','open','high','low','close','volume'])
    print('[STEP 3] Skipped backfill (not needed)')
print()

# Step 4: Merge
print('[STEP 4] Merging data...')
if dfp is None or dfp.empty:
    print('  Using Stooq only (Polygon empty)')
    final = dfo
elif dfo is None or dfo.empty:
    print('  Using Polygon only (Stooq empty)')
    final = dfp
else:
    print(f'  Merging {len(dfo)} Stooq rows + {len(dfp)} Polygon rows')

    # Normalize timestamps
    dfo['timestamp'] = pd.to_datetime(dfo['timestamp'], utc=True).dt.tz_localize(None)
    dfp['timestamp'] = pd.to_datetime(dfp['timestamp'], utc=True).dt.tz_localize(None)
    dfo['__src'] = 'stooq'
    dfp['__src'] = 'poly'

    merged = pd.concat([dfo, dfp], ignore_index=True)
    merged['timestamp'] = pd.to_datetime(merged['timestamp'], utc=True).dt.tz_localize(None)
    merged = merged.sort_values(['timestamp','__src'])
    merged = merged.drop_duplicates(subset=['timestamp'], keep='last')
    merged = merged.drop(columns=['__src'])
    merged = merged.sort_values('timestamp').reset_index(drop=True)

    # Bound to [start, end]
    s = pd.to_datetime(start).date()
    e = pd.to_datetime(end).date()
    merged = merged[(merged['timestamp'].dt.date >= s) & (merged['timestamp'].dt.date <= e)]

    final = merged
    print(f'  Merged result: {len(final)} rows')

print()

# Final verification
if final.empty:
    print('[FAIL] Final dataframe is empty!')
    sys.exit(1)

first_date = pd.to_datetime(final.iloc[0]['timestamp']).date()
last_date = pd.to_datetime(final.iloc[-1]['timestamp']).date()
years = (last_date - first_date).days / 365.25

print('=' * 70)
print('FINAL RESULT:')
print(f'  Rows: {len(final)}')
print(f'  Range: {first_date} to {last_date}')
print(f'  Years: {years:.2f}')
print()

if years >= 10.0:
    print(f'[SUCCESS] {years:.1f} years of data!')
else:
    print(f'[FAIL] Only {years:.1f} years')
print('=' * 70)
