#!/usr/bin/env python3
"""
CRITICAL: Triple Scan Verification
Runs 3 independent scans on SAME data to verify consistency.
Also investigates Turtle Soup signals.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from datetime import datetime, timedelta
from config.env_loader import load_env
from data.universe.loader import load_universe
from data.providers.multi_source import fetch_daily_bars_multi
from strategies.dual_strategy.combined import DualStrategyScanner, DualStrategyParams


def main():
    load_env(ROOT / '.env')

    print('=' * 70)
    print('CRITICAL VERIFICATION: TRIPLE SCAN CONSISTENCY')
    print('=' * 70)
    print()

    # Step 1: Load data ONCE
    print('STEP 1: Loading data (will use SAME data for all 3 scans)')
    print('-' * 70)

    symbols = load_universe(ROOT / 'data/universe/optionable_liquid_800.csv', cap=900)
    print(f'Universe: {len(symbols)} symbols')

    end = datetime.now()
    start = end - timedelta(days=400)

    all_data = []
    for i, sym in enumerate(symbols):
        if i > 0 and i % 200 == 0:
            print(f'  Loading: {i}/{len(symbols)}')
        try:
            df = fetch_daily_bars_multi(sym, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            if df is not None and not df.empty and len(df) > 200:
                df['symbol'] = sym
                all_data.append(df)
        except Exception:
            pass

    combined = pd.concat(all_data, ignore_index=True)
    print(f'Loaded: {len(all_data)} symbols, {len(combined):,} bars')
    print(f'Date range: {combined["timestamp"].min()} to {combined["timestamp"].max()}')

    # Step 2: Verify scanner parameters
    print()
    print('STEP 2: Verifying Scanner Parameters')
    print('-' * 70)

    params = DualStrategyParams()
    print('IBS+RSI Parameters:')
    print(f'  ibs_entry:        {params.ibs_entry} (need < this)')
    print(f'  rsi_entry:        {params.rsi_entry} (need < this)')
    print(f'  ibs_rsi_stop_mult: {params.ibs_rsi_stop_mult}')
    print(f'  ibs_rsi_time_stop: {params.ibs_rsi_time_stop}')
    print()
    print('Turtle Soup Parameters:')
    print(f'  ts_lookback:              {params.ts_lookback}')
    print(f'  ts_min_bars_since_extreme: {params.ts_min_bars_since_extreme}')
    print(f'  ts_min_sweep_strength:    {params.ts_min_sweep_strength} (CRITICAL)')
    print(f'  ts_stop_buffer_mult:      {params.ts_stop_buffer_mult}')
    print(f'  ts_r_multiple:            {params.ts_r_multiple}')
    print(f'  ts_time_stop:             {params.ts_time_stop}')

    # Step 3: Run 3 independent scans
    print()
    print('STEP 3: Running 3 Independent Scans')
    print('-' * 70)

    results = []
    for scan_num in range(1, 4):
        scanner = DualStrategyScanner(DualStrategyParams())  # Fresh scanner each time
        signals = scanner.generate_signals(combined.copy())  # Copy data
        signals = signals.sort_values('score', ascending=False).reset_index(drop=True)

        # Get top 10 symbols and scores
        top10 = [(row['symbol'], round(row['score'], 2)) for _, row in signals.head(10).iterrows()]

        results.append({
            'scan': scan_num,
            'total': len(signals),
            'ibs_rsi': len(signals[signals['reason'].str.contains('IBS_RSI')]),
            'turtle_soup': len(signals[signals['reason'].str.contains('TurtleSoup')]),
            'top10': top10
        })

        print(f'Scan {scan_num}: {len(signals)} signals (IBS_RSI: {results[-1]["ibs_rsi"]}, TurtleSoup: {results[-1]["turtle_soup"]})')
        print(f'  Top 5: {[t[0] for t in top10[:5]]}')

    # Step 4: Verify consistency
    print()
    print('STEP 4: Consistency Verification')
    print('-' * 70)

    all_same = True
    for i in range(1, 3):
        if results[i]['total'] != results[0]['total']:
            all_same = False
            print(f'  MISMATCH: Scan {i+1} has {results[i]["total"]} signals vs Scan 1 has {results[0]["total"]}')
        if results[i]['top10'] != results[0]['top10']:
            all_same = False
            print(f'  MISMATCH: Top 10 differs between Scan 1 and Scan {i+1}')

    if all_same:
        print('  *** ALL 3 SCANS MATCH EXACTLY ***')

    # Step 5: Investigate Turtle Soup
    print()
    print('STEP 5: Turtle Soup Investigation')
    print('-' * 70)

    # Run scan_signals_over_time to get historical signals
    scanner = DualStrategyScanner(DualStrategyParams())
    all_signals = scanner.scan_signals_over_time(combined)

    # Filter to Turtle Soup only
    ts_signals = all_signals[all_signals['reason'].str.contains('TurtleSoup')]

    print(f'Total Turtle Soup signals in history: {len(ts_signals)}')

    if not ts_signals.empty:
        ts_signals['date'] = pd.to_datetime(ts_signals['timestamp']).dt.date

        # Show last 10 Turtle Soup signals
        print()
        print('Last 10 Turtle Soup signals:')
        for _, row in ts_signals.tail(10).iterrows():
            print(f'  {row["timestamp"]}: {row["symbol"]} - {row["reason"]} (score: {row["score"]:.1f})')

        # Check for Dec 29 signals
        dec29 = ts_signals[ts_signals['date'] == datetime(2025, 12, 29).date()]
        dec30 = ts_signals[ts_signals['date'] == datetime(2025, 12, 30).date()]

        print()
        print(f'Dec 29 Turtle Soup signals: {len(dec29)}')
        print(f'Dec 30 Turtle Soup signals: {len(dec30)}')
    else:
        print('No Turtle Soup signals in the dataset.')

    # Step 6: Final Summary
    print()
    print('=' * 70)
    print('FINAL SUMMARY')
    print('=' * 70)
    print(f'Total signals for tomorrow: {results[0]["total"]}')
    print(f'  - IBS+RSI: {results[0]["ibs_rsi"]}')
    print(f'  - Turtle Soup: {results[0]["turtle_soup"]}')
    print()
    print('Top 10 Signals (FROZEN):')
    for i, (sym, score) in enumerate(results[0]['top10'], 1):
        print(f'  {i:2}. {sym:<6} Score: {score}')
    print()
    if all_same:
        print('VERIFICATION: PASSED - All 3 scans identical')
    else:
        print('VERIFICATION: FAILED - Scans differ!')


if __name__ == '__main__':
    main()
