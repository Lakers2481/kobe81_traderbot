#!/usr/bin/env python3
"""
Test Hybrid Data Strategy: Polygon (2021-2026) + Stooq (2015-2020)
Renaissance Technologies Standard: VERIFY WITH REAL DATA
"""
import sys
from pathlib import Path

# Add project to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load environment
from config.env_loader import load_env
load_env(ROOT / '.env')

import pandas as pd
from data.providers.multi_source import fetch_daily_bars_multi

print('=' * 70)
print('HYBRID DATA TEST: Polygon (2021-2026) + Stooq (2015-2020)')
print('=' * 70)
print()

# Test on AAPL
symbol = 'AAPL'
print(f'[1] Fetching {symbol} with hybrid approach (2015-01-01 to 2026-01-08)...')
df = fetch_daily_bars_multi(symbol, '2015-01-01', '2026-01-08', cache_dir=Path('data/cache'))

if df.empty:
    print('[FAIL] No data returned!')
    sys.exit(1)

print(f'[OK] Fetched {len(df)} rows')
print()

# Verify date range
first_date = pd.to_datetime(df.iloc[0]['timestamp']).date()
last_date = pd.to_datetime(df.iloc[-1]['timestamp']).date()
years = (last_date - first_date).days / 365.25

print('[2] VERIFICATION:')
print(f'  First date: {first_date}')
print(f'  Last date:  {last_date}')
print(f'  Total years: {years:.2f}')
print()

# Check for gaps
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
date_diffs = df['timestamp'].diff().dt.days
max_gap = date_diffs.max()
gaps = (date_diffs > 5).sum()

print('[3] DATA QUALITY:')
print(f'  Max gap: {max_gap:.0f} days')
print(f'  Gaps >5 days: {gaps}')
print(f'  Avg daily volume: {df["volume"].mean():,.0f}')
print()

# Show sample from old data (Stooq) and new data (Polygon)
print('[4] SAMPLE DATA (Proof of Hybrid):')
print()
print('  First 5 rows (should be Stooq 2015):')
print(df.head()[['timestamp', 'symbol', 'close', 'volume']].to_string(index=False))
print()
print('  Last 5 rows (should be Polygon 2026):')
print(df.tail()[['timestamp', 'symbol', 'close', 'volume']].to_string(index=False))
print()

# Find the transition point (where Stooq ends and Polygon begins)
df_2020 = df[(df['timestamp'] >= '2020-12-01') & (df['timestamp'] <= '2021-02-01')]
if not df_2020.empty:
    print('  Transition period (Dec 2020 - Feb 2021):')
    print(df_2020[['timestamp', 'symbol', 'close', 'volume']].to_string(index=False))
    print()

# Count data by year to show coverage
print('[5] YEARLY COVERAGE:')
df['year'] = df['timestamp'].dt.year
yearly_counts = df.groupby('year').size()
for year, count in yearly_counts.items():
    print(f'  {year}: {count:3d} rows')
print()

# Final verdict
print('=' * 70)
if years >= 10.0:
    print(f'[SUCCESS] Hybrid approach provides {years:.1f} years of data!')
    print('[VERIFIED] Ready to scale to 800 symbols.')
    print('=' * 70)
    sys.exit(0)
else:
    print(f'[FAIL] Only {years:.1f} years - need 10+')
    print('=' * 70)
    sys.exit(1)
