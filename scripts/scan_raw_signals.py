#!/usr/bin/env python3
"""
Pre-Game Briefing Scanner - Raw signals without quality gate
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
    print('KOBE PRE-GAME BRIEFING - DECEMBER 31, 2025')
    print('RAW SIGNALS (No Quality Gate) - Using Polygon Data')
    print('=' * 70)
    print()

    symbols = load_universe(ROOT / 'data/universe/optionable_liquid_900.csv', cap=300)
    print(f'Scanning {len(symbols)} symbols...')

    end = datetime.now()
    start = end - timedelta(days=400)

    all_data = []
    errors = 0
    for i, sym in enumerate(symbols):
        if i > 0 and i % 100 == 0:
            print(f'  Progress: {i}/{len(symbols)}')
        try:
            df = fetch_daily_bars_multi(sym, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            if df is not None and not df.empty and len(df) > 200:
                df['symbol'] = sym
                all_data.append(df)
        except Exception:
            errors += 1

    print(f'Loaded {len(all_data)} symbols (errors: {errors})')

    if not all_data:
        print('No data loaded!')
        return

    combined = pd.concat(all_data, ignore_index=True)
    scanner = DualStrategyScanner(DualStrategyParams())
    signals = scanner.generate_signals(combined)

    print()
    print('=' * 70)
    print(f'SIGNALS FOR TOMORROW (DEC 31): {len(signals)}')
    print('=' * 70)

    if signals.empty:
        print('No signals for tomorrow.')
        return

    signals = signals.sort_values('score', ascending=False)

    print()
    print('ALL SIGNALS (sorted by score):')
    print('-' * 90)

    for i, (_, row) in enumerate(signals.head(25).iterrows(), 1):
        entry = float(row['entry_price']) if row['entry_price'] else 0
        stop = float(row['stop_loss']) if row['stop_loss'] else 0
        target = float(row['take_profit']) if row['take_profit'] else 0
        reason = str(row['reason'])[:35] if row['reason'] else 'Unknown'
        score = float(row['score']) if row['score'] else 0
        sym = str(row['symbol']) if row['symbol'] else '???'
        print(f'{i:2}. {sym:<6} {reason:<35} ${entry:.2f} -> ${target:.2f} (stop ${stop:.2f}) Score:{score:.1f}')

    # Summary
    ibs_count = len(signals[signals['reason'].str.contains('IBS_RSI')])
    ts_count = len(signals[signals['reason'].str.contains('TurtleSoup')])
    print()
    print(f'IBS+RSI: {ibs_count} | Turtle Soup: {ts_count} | TOTAL: {len(signals)}')

    # Top 3
    print()
    print('=' * 70)
    print('TOP 3 PICKS FOR TOMORROW')
    print('=' * 70)

    for i, (_, row) in enumerate(signals.head(3).iterrows(), 1):
        entry = float(row['entry_price']) if row['entry_price'] else 0
        stop = float(row['stop_loss']) if row['stop_loss'] else 0
        target = float(row['take_profit']) if row['take_profit'] else 0
        score = float(row['score']) if row['score'] else 0
        risk = entry - stop
        reward = target - entry
        rr = reward / risk if risk > 0 else 0

        print()
        print(f'#{i} {row["symbol"]} @ ${entry:.2f}')
        print(f'   Strategy: {row["reason"]}')
        print(f'   Stop: ${stop:.2f} (risk ${risk:.2f}/share)')
        print(f'   Target: ${target:.2f} (reward ${reward:.2f}/share)')
        print(f'   R:R: {rr:.1f}:1 | Score: {score:.1f}')


if __name__ == '__main__':
    main()
