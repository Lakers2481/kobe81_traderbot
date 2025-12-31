#!/usr/bin/env python3
"""
FRESH SCAN FOR TOMORROW - Uses latest market data
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

import pandas as pd
from datetime import datetime, timedelta
from data.universe.loader import load_universe
from data.providers.multi_source import fetch_daily_bars_multi
from strategies.dual_strategy.combined import DualStrategyScanner, DualStrategyParams


def main():
    print('=' * 70)
    print('KOBE FRESH SCAN - DECEMBER 31, 2025')
    print('Using FRESH Dec 30 close data')
    print('=' * 70)
    print()

    # Load universe
    symbols = load_universe(ROOT / 'data/universe/optionable_liquid_900.csv', cap=300)
    print(f'Scanning {len(symbols)} symbols with FRESH data...')
    print()

    # Fetch fresh data (last 400 days ending TODAY Dec 30)
    end = datetime(2025, 12, 30)
    start = end - timedelta(days=400)

    print(f'Data range: {start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}')
    print()

    all_data = []
    errors = 0
    for i, sym in enumerate(symbols):
        if i > 0 and i % 50 == 0:
            print(f'  Progress: {i}/{len(symbols)} symbols loaded...')
        try:
            df = fetch_daily_bars_multi(sym, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            if df is not None and not df.empty and len(df) > 200:
                df['symbol'] = sym
                all_data.append(df)
        except Exception:
            errors += 1

    print(f'Loaded {len(all_data)} symbols (errors: {errors})')
    print()

    if not all_data:
        print('ERROR: No data loaded!')
        return

    # Combine and verify we have Dec 30 data
    combined = pd.concat(all_data, ignore_index=True)
    latest_date = pd.to_datetime(combined['timestamp']).max()
    print(f'Latest data date: {latest_date.strftime("%Y-%m-%d")}')

    if latest_date.strftime('%Y-%m-%d') != '2025-12-30':
        print(f'WARNING: Expected Dec 30 data, got {latest_date.strftime("%Y-%m-%d")}')
    else:
        print('CONFIRMED: Using Dec 30 close data (FRESH)')
    print()

    # Run scanner
    print('Running DualStrategyScanner v2.2...')
    scanner = DualStrategyScanner(DualStrategyParams())
    signals = scanner.generate_signals(combined)

    print()
    print('=' * 70)
    print(f'SIGNALS FOR TOMORROW (DEC 31, 2025): {len(signals)}')
    print('=' * 70)
    print()

    if signals.empty:
        print('No signals for tomorrow.')
        return

    signals = signals.sort_values('score', ascending=False)

    # Count by strategy
    ibs_count = len(signals[signals['reason'].str.contains('IBS_RSI')])
    ts_count = len(signals[signals['reason'].str.contains('TurtleSoup')])

    print(f'IBS+RSI: {ibs_count} | Turtle Soup: {ts_count} | TOTAL: {len(signals)}')
    print()
    print('ALL SIGNALS (sorted by score):')
    print('-' * 90)

    for i, (_, row) in enumerate(signals.iterrows(), 1):
        entry = float(row['entry_price']) if row['entry_price'] else 0
        stop = float(row['stop_loss']) if row['stop_loss'] else 0
        target = float(row['take_profit']) if row['take_profit'] else 0
        reason = str(row['reason'])[:40] if row['reason'] else 'Unknown'
        score = float(row['score']) if row['score'] else 0
        sym = str(row['symbol']) if row['symbol'] else '???'

        print(f'{i:2}. {sym:<6} {reason:<40} Entry: ${entry:.2f} Stop: ${stop:.2f} Score: {score:.1f}')

    print()
    print('=' * 70)
    print('TOP 3 PICKS FOR TOMORROW')
    print('=' * 70)

    for i, (_, row) in enumerate(signals.head(3).iterrows(), 1):
        entry = float(row['entry_price']) if row['entry_price'] else 0
        stop = float(row['stop_loss']) if row['stop_loss'] else 0
        target = float(row['take_profit']) if row['take_profit'] else 0
        score = float(row['score']) if row['score'] else 0
        risk = entry - stop if entry and stop else 0
        reward = target - entry if target and entry else 0
        rr = reward / risk if risk > 0 else 0

        print()
        print(f'#{i} {row["symbol"]} @ ${entry:.2f}')
        print(f'   Strategy: {row["reason"]}')
        print(f'   Stop: ${stop:.2f} (risk ${risk:.2f}/share)')
        print(f'   Target: ${target:.2f} (reward ${reward:.2f}/share)')
        print(f'   R:R: {rr:.1f}:1 | Score: {score:.1f}')

    # Save results
    output_file = ROOT / 'logs' / 'fresh_scan_20251231.csv'
    signals.to_csv(output_file, index=False)
    print()
    print(f'Results saved to: {output_file}')


if __name__ == '__main__':
    main()
