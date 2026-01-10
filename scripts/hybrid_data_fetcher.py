#!/usr/bin/env python3
"""
Hybrid Data Fetcher: Polygon (2021-2026) + YFinance (2015-2020)
Renaissance Technologies Standard: 10+ Years Verified Data

Usage:
    python scripts/hybrid_data_fetcher.py --universe data/universe/optionable_liquid_800.csv

Output:
    - Fetches 10+ years for all 800 symbols
    - Saves to data/cache/hybrid/ directory
    - Generates verification report
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import yfinance as yf

from config.env_loader import load_env
from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.universe.loader import load_universe

def fetch_hybrid(symbol: str, start: str, end: str, cache_dir: Path) -> dict:
    """
    Fetch hybrid data: Polygon (recent) + YFinance (historical).

    Returns dict with verification stats.
    """
    result = {
        'symbol': symbol,
        'success': False,
        'total_rows': 0,
        'years': 0.0,
        'first_date': None,
        'last_date': None,
        'polygon_rows': 0,
        'yfinance_rows': 0,
        'error': None,
    }

    try:
        # Step 1: Fetch from Polygon (2021-2026)
        dfp = fetch_daily_bars_polygon(symbol, start, end, cache_dir=cache_dir)
        result['polygon_rows'] = len(dfp) if not dfp.empty else 0

        # Step 2: Check if backfill needed
        need_backfill = False
        backfill_end = None

        if dfp is None or dfp.empty:
            need_backfill = True
            backfill_end = end
        else:
            earliest_poly = pd.to_datetime(dfp['timestamp']).min()
            req_start = pd.to_datetime(start)
            if earliest_poly > req_start:
                need_backfill = True
                backfill_end = (earliest_poly - pd.Timedelta(days=1)).date().isoformat()

        # Step 3: Fetch backfill from YFinance if needed
        if need_backfill:
            dfy = yf.download(symbol, start=start, end=backfill_end, progress=False)
            if not dfy.empty:
                # Convert yfinance format to standard format
                dfy = dfy.reset_index()

                # FIX: Flatten MultiIndex columns (yfinance returns ('Date', '') instead of 'Date')
                if isinstance(dfy.columns, pd.MultiIndex):
                    dfy.columns = dfy.columns.get_level_values(0)

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
                result['yfinance_rows'] = len(dfy)
            else:
                dfy = pd.DataFrame(columns=['timestamp','symbol','open','high','low','close','volume'])
        else:
            dfy = pd.DataFrame(columns=['timestamp','symbol','open','high','low','close','volume'])

        # Step 4: Merge
        if dfp is None or dfp.empty:
            final = dfy
        elif dfy is None or dfy.empty:
            final = dfp
        else:
            # Normalize timestamps
            dfy['timestamp'] = pd.to_datetime(dfy['timestamp']).dt.tz_localize(None)
            dfp['timestamp'] = pd.to_datetime(dfp['timestamp']).dt.tz_localize(None)

            # Mark source
            dfy['__src'] = 'yf'
            dfp['__src'] = 'poly'

            # Merge and prefer Polygon on overlaps
            merged = pd.concat([dfy, dfp], ignore_index=True)
            merged = merged.sort_values(['timestamp', '__src'])
            merged = merged.drop_duplicates(subset=['timestamp'], keep='last')
            merged = merged.drop(columns=['__src'])
            merged = merged.sort_values('timestamp').reset_index(drop=True)

            # Bound to date range
            s = pd.to_datetime(start).date()
            e = pd.to_datetime(end).date()
            merged = merged[(merged['timestamp'].dt.date >= s) & (merged['timestamp'].dt.date <= e)]
            final = merged

        if final.empty:
            result['error'] = 'No data after merge'
            return result

        # Step 5: Calculate verification stats
        result['total_rows'] = len(final)
        result['first_date'] = pd.to_datetime(final.iloc[0]['timestamp']).date().isoformat()
        result['last_date'] = pd.to_datetime(final.iloc[-1]['timestamp']).date().isoformat()

        first_dt = pd.to_datetime(result['first_date'])
        last_dt = pd.to_datetime(result['last_date'])
        result['years'] = (last_dt - first_dt).days / 365.25

        # Step 6: Save to cache
        hybrid_cache = cache_dir / 'hybrid'
        hybrid_cache.mkdir(parents=True, exist_ok=True)
        cache_file = hybrid_cache / f'{symbol}_{start}_{end}.csv'
        final.to_csv(cache_file, index=False)

        result['success'] = True
        return result

    except Exception as e:
        result['error'] = str(e)
        return result


def main():
    parser = argparse.ArgumentParser(description='Fetch hybrid data for universe')
    parser.add_argument('--universe', type=str, required=True)
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end', type=str, default='2026-01-08')
    parser.add_argument('--cache', type=str, default='data/cache')
    parser.add_argument('--concurrency', type=int, default=5)
    parser.add_argument('--dotenv', type=str, default='./.env')
    args = parser.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
        print(f'Loaded {len(loaded)} env vars from {dotenv}')

    # Load universe
    symbols = load_universe(Path(args.universe))
    print(f'Universe: {len(symbols)} symbols')
    print()

    cache_dir = Path(args.cache)

    # Fetch all symbols
    print('[FETCHING HYBRID DATA]')
    print(f'Strategy: Polygon (2021-2026) + YFinance (2015-2020)')
    print(f'Target: 10+ years per symbol')
    print()

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(fetch_hybrid, sym, args.start, args.end, cache_dir): sym for sym in symbols}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)

            status = '[OK]' if result['success'] else '[FAIL]'
            years = result.get('years', 0.0)
            print(f"{i:3d}/{len(symbols)} {status} {result['symbol']:6s} {years:5.1f}y {result['total_rows']:4d} rows (P:{result['polygon_rows']} Y:{result['yfinance_rows']})")

    # Generate verification report
    print()
    print('=' * 70)
    print('VERIFICATION REPORT')
    print('=' * 70)

    success_count = sum(1 for r in results if r['success'])
    total_years = [r['years'] for r in results if r['success']]

    print(f'Success: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)')
    if total_years:
        print(f'Min years: {min(total_years):.1f}')
        print(f'Avg years: {sum(total_years)/len(total_years):.1f}')
        print(f'Max years: {max(total_years):.1f}')

    # Check for failures
    failures = [r for r in results if not r['success']]
    if failures:
        print()
        print(f'FAILURES ({len(failures)}):')
        for r in failures[:10]:
            print(f"  {r['symbol']:6s}: {r['error']}")

    # Check symbols with <10 years
    insufficient = [r for r in results if r['success'] and r['years'] < 10.0]
    if insufficient:
        print()
        print(f'INSUFFICIENT HISTORY (<10 years): {len(insufficient)}')
        for r in insufficient[:10]:
            print(f"  {r['symbol']:6s}: {r['years']:.1f} years")

    # Final verdict
    print()
    print('=' * 70)
    ready_count = sum(1 for r in results if r['success'] and r['years'] >= 10.0)
    if ready_count >= 855:  # 95% of 900
        print(f'[SUCCESS] {ready_count}/800 symbols with 10+ years')
        print('[VERIFIED] Ready for Renaissance-quality backtesting!')
    else:
        print(f'[NOT READY] Only {ready_count}/800 symbols with 10+ years')
        print('Need at least 855 symbols (95%) with 10+ years')
    print('=' * 70)


if __name__ == '__main__':
    main()
