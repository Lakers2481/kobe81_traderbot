#!/usr/bin/env python3
"""
Find Replacement Stocks with 10+ Years Data
Renaissance Technologies Standard: VERIFY EVERYTHING

Usage:
    python scripts/find_replacement_stocks.py --needed 261
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.env_loader import load_env
from data.providers.polygon_eod import fetch_daily_bars_polygon, has_options_polygon
from scripts.hybrid_data_fetcher import fetch_hybrid

def test_candidate(symbol: str, cache_dir: Path) -> dict:
    """
    Test if candidate meets all criteria:
    - Has options
    - Has 10+ years of data
    - Returns stats
    """
    result = {
        'symbol': symbol,
        'qualified': False,
        'has_options': False,
        'years': 0.0,
        'avg_volume': 0.0,
        'last_price': 0.0,
        'error': None,
    }

    try:
        # Step 1: Check options availability
        result['has_options'] = has_options_polygon(symbol)
        if not result['has_options']:
            result['error'] = 'No options'
            return result

        # Step 2: Fetch hybrid data
        fetch_result = fetch_hybrid(symbol, '2015-01-01', '2026-01-08', cache_dir)

        if not fetch_result['success']:
            result['error'] = fetch_result.get('error', 'Unknown error')
            return result

        result['years'] = fetch_result['years']

        # Step 3: Check 10+ years requirement
        if result['years'] < 10.0:
            result['error'] = f'Only {result["years"]:.1f} years'
            return result

        # Step 4: Calculate volume stats
        df = pd.read_csv(cache_dir / 'hybrid' / f'{symbol}_2015-01-01_2026-01-08.csv')
        result['avg_volume'] = df['volume'].mean()
        result['last_price'] = df.iloc[-1]['close']

        # Step 5: Mark as qualified
        result['qualified'] = True
        return result

    except Exception as e:
        result['error'] = str(e)
        return result


def main():
    parser = argparse.ArgumentParser(description='Find replacement stocks')
    parser.add_argument('--needed', type=int, required=True, help='Number of replacements needed')
    parser.add_argument('--candidates', type=str, default='data/universe/optionable_liquid_candidates.csv')
    parser.add_argument('--verified', type=str, default='verified_symbols.csv')
    parser.add_argument('--cache', type=str, default='data/cache')
    parser.add_argument('--concurrency', type=int, default=10)
    parser.add_argument('--dotenv', type=str, default='./.env')
    args = parser.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
        print(f'Loaded {len(loaded)} env vars')

    # Load verified symbols (to exclude)
    verified = set(pd.read_csv(args.verified)['symbol'].tolist())
    print(f'Already verified: {len(verified)} symbols')

    # Load candidates
    all_candidates = pd.read_csv(args.candidates)['Symbol'].tolist()
    candidates = [s for s in all_candidates if s not in verified]
    print(f'Candidate pool: {len(candidates)} symbols')
    print()

    cache_dir = Path(args.cache)

    # Test candidates
    print(f'[TESTING CANDIDATES FOR REPLACEMENT]')
    print(f'Need: {args.needed} replacements')
    print()

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(test_candidate, sym, cache_dir): sym for sym in candidates}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)

            if result['qualified']:
                status = f"[OK] {result['years']:.1f}y ${result['avg_volume']/1e6:.1f}M vol/day"
            else:
                status = f"[SKIP] {result['error']}"

            print(f"{i:4d}/{len(candidates)} {result['symbol']:6s} {status}")

            # Stop early if we have enough qualified candidates
            qualified = [r for r in results if r['qualified']]
            if len(qualified) >= args.needed + 50:  # Extra buffer
                print(f"\n[SUCCESS] Found {len(qualified)} qualified candidates (needed {args.needed})")
                break

    # Sort qualified candidates by volume
    qualified = [r for r in results if r['qualified']]
    qualified_df = pd.DataFrame(qualified)
    qualified_df = qualified_df.sort_values('avg_volume', ascending=False)

    print()
    print('=' * 70)
    print('REPLACEMENT CANDIDATES FOUND')
    print('=' * 70)
    print(f'Qualified: {len(qualified)}/{len(results)} tested')
    print(f'Needed: {args.needed}')
    print()

    if len(qualified) >= args.needed:
        # Select top by volume
        replacements = qualified_df.head(args.needed)
        replacements.to_csv('replacement_candidates.csv', index=False)
        print(f'[SAVED] replacement_candidates.csv ({len(replacements)} symbols)')
        print()

        print('Top 20 replacements by volume:')
        for idx, row in replacements.head(20).iterrows():
            print(f"  {row['symbol']:6s} {row['years']:5.1f}y ${row['avg_volume']/1e6:8.1f}M/day")

        print()
        print('[SUCCESS] Found enough replacements!')
        return 0
    else:
        print(f'[NOT ENOUGH] Only found {len(qualified)}, need {args.needed}')
        print('Try expanding candidate pool or reducing requirements')
        return 1


if __name__ == '__main__':
    sys.exit(main())
