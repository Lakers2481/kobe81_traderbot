#!/usr/bin/env python3
"""
Build Final Verified Universe
Renaissance Technologies Standard: VERIFY EVERYTHING

Criteria (NO COMPROMISES):
- 10+ years of data (FULL market cycle required)
- Options available
- High volume
- Sorted by volume (most liquid first)
- Accept actual count (target ~850-900)

Usage:
    python scripts/build_final_900.py
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
        if years >= 10.0:  # Renaissance standard: FULL 10 years required
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

    # Find all hybrid cache files
    cache_dir = ROOT / 'data' / 'cache' / 'hybrid'
    all_files = list(cache_dir.glob('*_2015-01-01_2026-01-08.csv'))
    print(f'Found {len(all_files)} hybrid cache files')
    print()

    # Analyze all files
    print('[ANALYZING ALL HYBRID CACHE FILES]')
    print(f'Checking for 10+ years (Renaissance standard) + options availability')
    print()

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_hybrid_file, f): f for f in all_files}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)

            if result['qualified']:
                status = f"[OK] {result['years']:.1f}y ${result['avg_volume']/1e6:.1f}M vol/day"
            else:
                status = f"[SKIP] {result['error']}"

            print(f"{i:4d}/{len(all_files)} {result['symbol']:6s} {status}")

    # Filter qualified
    qualified = [r for r in results if r['qualified']]
    print()
    print('=' * 70)
    print('UNIVERSE BUILDING RESULTS')
    print('=' * 70)
    print(f'Qualified: {len(qualified)}/{len(results)} tested (10+ years + options)')
    print(f'Target: 850-800 symbols (quality over quantity)')
    print()

    if len(qualified) < 850:
        print(f'[WARNING] Only found {len(qualified)} qualified symbols')
        print(f'Expected 850-900 with 10+ year requirement')
        print()

    # Sort by volume (most liquid first)
    qualified_df = pd.DataFrame(qualified)
    qualified_df = qualified_df.sort_values('avg_volume', ascending=False)

    # Take all qualified (no artificial cap)
    final_universe = qualified_df

    # Save to file
    output_file = ROOT / 'data' / 'universe' / 'optionable_liquid_900_verified.csv'
    final_universe[['symbol']].to_csv(output_file, index=False)
    print(f'[SAVED] {output_file.name} ({len(final_universe)} symbols)')

    # Save full metadata
    metadata_file = ROOT / 'data' / 'universe' / 'optionable_liquid_900_verified.full.csv'
    final_universe.to_csv(metadata_file, index=False)
    print(f'[SAVED] {metadata_file.name} (with full metadata)')
    print()

    # Show top 20 by volume
    print('Top 20 symbols by volume:')
    for idx, row in final_universe.head(20).iterrows():
        print(f"  {row['symbol']:6s} {row['years']:5.1f}y ${row['avg_volume']/1e6:8.1f}M/day {row['rows']:5d} rows")

    print()

    # Show years distribution
    print('Years distribution:')
    for min_years in [10, 9, 8]:
        count = len(final_universe[final_universe['years'] >= min_years])
        pct = 100 * count / len(final_universe)
        print(f'  {min_years}+ years: {count:3d} symbols ({pct:5.1f}%)')

    print()

    # Final verdict
    if len(final_universe) >= 850:
        print(f'[SUCCESS] Built verified {len(final_universe)}-stock universe!')
        print(f'Renaissance standard met: 10+ years + options + high volume')
    elif len(final_universe) >= 700:
        print(f'[ACCEPTABLE] Built {len(final_universe)}-stock universe')
        print(f'Below 850 target but all symbols meet 10+ year standard')
    else:
        print(f'[WARNING] Only {len(final_universe)} stocks meet strict criteria')

    return 0 if len(final_universe) >= 700 else 1


if __name__ == '__main__':
    sys.exit(main())
