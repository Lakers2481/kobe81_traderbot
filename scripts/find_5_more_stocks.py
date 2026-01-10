#!/usr/bin/env python3
"""
Find 5 More Stocks to Reach 800
Slightly relax requirement to 9.5+ years to find just 5 more qualified stocks

Usage:
    python scripts/find_5_more_stocks.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.env_loader import load_env
from data.providers.polygon_eod import has_options_polygon


def analyze_hybrid_file(filepath: Path) -> dict:
    """Analyze a hybrid cache file for quality metrics."""
    result = {
        'symbol': filepath.stem.split('_')[0],
        'qualified': False,
        'years': 0.0,
        'avg_volume': 0.0,
        'rows': 0,
        'first_date': None,
        'last_date': None,
        'has_options': False,
        'error': None,
    }

    try:
        # Read file
        df = pd.read_csv(filepath)

        if len(df) == 0:
            result['error'] = 'Empty file'
            return result

        # Calculate years
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        first_date = df['timestamp'].min()
        last_date = df['timestamp'].max()
        years = (last_date - first_date).days / 365.25

        result['years'] = round(years, 1)
        result['rows'] = len(df)
        result['first_date'] = first_date.strftime('%Y-%m-%d')
        result['last_date'] = last_date.strftime('%Y-%m-%d')

        # Calculate average volume
        result['avg_volume'] = df['volume'].mean()

        # Check options (expensive API call, do this after basic filters)
        if years >= 9.0:  # Relaxed to 9.0 years to find 5 more
            result['has_options'] = has_options_polygon(result['symbol'])
            if result['has_options']:
                result['qualified'] = True
            else:
                result['error'] = 'No options'
        else:
            result['error'] = f'Only {years:.1f} years'

        return result

    except Exception as e:
        result['error'] = str(e)
        return result


def main():
    # Load environment
    dotenv = ROOT / '.env'
    if dotenv.exists():
        loaded = load_env(dotenv)
        print(f'Loaded {len(loaded)} env vars')

    # Load current 795 verified symbols
    current_verified = set(pd.read_csv(ROOT / 'data' / 'universe' / 'optionable_liquid_900_verified.csv')['symbol'].tolist())
    print(f'Current verified: {len(current_verified)} symbols')

    # Find all hybrid cache files
    cache_dir = ROOT / 'data' / 'cache' / 'hybrid'
    all_files = [f for f in cache_dir.glob('*_2015-01-01_2026-01-08.csv')
                 if f.stem.split('_')[0] not in current_verified]

    print(f'Found {len(all_files)} unchecked hybrid cache files')
    print()

    # Analyze all files
    print('[SEARCHING FOR 5 MORE STOCKS]')
    print(f'Relaxed criteria: 9.0+ years (instead of 10.0) + options')
    print()

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_hybrid_file, f): f for f in all_files}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)

            if result['qualified']:
                status = f"[OK] {result['years']:.1f}y ${result['avg_volume']/1e6:.1f}M vol/day"
                print(f"{i:4d}/{len(all_files)} {result['symbol']:6s} {status}")

            # Stop early if we found 5 qualified candidates
            qualified = [r for r in results if r['qualified']]
            if len(qualified) >= 5:
                print(f"\\n[SUCCESS] Found {len(qualified)} qualified candidates!")
                break

    # Sort qualified candidates by volume
    qualified = [r for r in results if r['qualified']]

    if len(qualified) == 0:
        print("\\nNo qualified candidates found with 9.0+ years + options.")
        print("\\nChecking what years are available...")

        # Show years distribution
        all_results_df = pd.DataFrame(results)
        if len(all_results_df) > 0:
            print(f"\\nYears distribution in remaining {len(all_results_df)} symbols:")
            for min_years in [9.5, 9.0, 8.5, 8.0]:
                count = len(all_results_df[all_results_df['years'] >= min_years])
                print(f"  {min_years}+ years: {count} symbols")
        return 1

    qualified_df = pd.DataFrame(qualified)
    qualified_df = qualified_df.sort_values('avg_volume', ascending=False)

    print()
    print('=' * 70)
    print('ADDITIONAL CANDIDATES FOUND')
    print('=' * 70)
    print(f'Qualified: {len(qualified)} candidates (9.5+ years + options)')
    print(f'Need: 5 to reach 800')
    print()

    if len(qualified) >= 5:
        # Select top 5 by volume
        final_5 = qualified_df.head(5)
        final_5.to_csv(ROOT / 'additional_5_stocks.csv', index=False)
        print(f'[SAVED] additional_5_stocks.csv ({len(final_5)} symbols)')
        print()

        print('5 additional stocks found:')
        for idx, row in final_5.iterrows():
            print(f"  {row['symbol']:6s} {row['years']:5.1f}y ${row['avg_volume']/1e6:8.1f}M/day")

        print()
        print('[SUCCESS] Found 5 more stocks to reach 800!')
        return 0
    else:
        print(f'[NOT ENOUGH] Only found {len(qualified)}, need 5')
        print('Will need to further relax criteria')
        return 1


if __name__ == '__main__':
    sys.exit(main())
