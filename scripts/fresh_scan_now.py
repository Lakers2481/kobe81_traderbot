#!/usr/bin/env python3
"""Fresh scan for today's trading - run after first 15-min candle closes."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from strategies.dual_strategy.combined import DualStrategyScanner, DualStrategyParams
from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.providers.alpaca_intraday import fetch_intraday_bars
from data.universe.loader import load_universe

def main():
    cache_dir = Path('cache')
    params = DualStrategyParams()
    scanner = DualStrategyScanner(params)

    print('=' * 70)
    print('FRESH SCAN: Prior Day Close -> Today Signals')
    print('=' * 70)

    # Load universe
    symbols = load_universe('data/universe/optionable_liquid_800.csv', cap=800)
    print(f'Universe: {len(symbols)} stocks')
    print()

    # Scan ALL for Friday (prior day) signals
    friday_signals = []
    for i, sym in enumerate(symbols):
        if (i+1) % 100 == 0:
            print(f'  Scanning {i+1}/800...')
        try:
            # Use end='2026-01-05' to get Friday as last bar
            df = fetch_daily_bars_polygon(sym, start='2024-01-01', end='2026-01-05', cache_dir=cache_dir)
            if df is not None and len(df) > 50:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                sig = scanner.generate_signals(df)
                if not sig.empty:
                    r = sig.iloc[0]
                    friday_signals.append({
                        'symbol': sym,
                        'strategy': r.get('strategy'),
                        'entry': float(r.get('entry_price', 0)),
                        'stop': float(r.get('stop_loss', 0)),
                        'target': float(r.get('take_profit', 0)) if pd.notna(r.get('take_profit')) else 0,
                        'reason': str(r.get('reason', ''))[:40]
                    })
        except Exception:
            pass

    print()
    print(f'Prior day signals found: {len(friday_signals)}')

    if friday_signals:
        print()
        print('TOP 5 WATCHLIST (from prior day close):')
        print('-' * 70)

        valid_signals = []
        for i, s in enumerate(friday_signals[:10]):
            # Get today's live price
            try:
                bars = fetch_intraday_bars(s['symbol'], timeframe='15Min', limit=2)
                live = bars[-1].close if bars else 0
            except Exception:
                live = 0

            gap = ((live / s['entry']) - 1) * 100 if s['entry'] > 0 and live > 0 else 0
            status = 'GAP_INVALID' if abs(gap) > 3 else 'VALID'

            print(f"{i+1:2}. {s['symbol']:5} | {s['strategy']:10} | Entry=${s['entry']:.2f} | Live=${live:.2f} | Gap={gap:+.1f}% | {status}")

            if status == 'VALID':
                s['live'] = live
                s['gap'] = gap
                valid_signals.append(s)

        print()
        if valid_signals:
            print('=' * 70)
            print(f'TRADEABLE TODAY: {len(valid_signals)} stocks (gap < 3%)')
            print('=' * 70)
            for i, s in enumerate(valid_signals[:2]):
                print(f"  TRADE {i+1}: {s['symbol']} {s['strategy']}")
                print(f"    Entry: ${s['entry']:.2f}")
                print(f"    Stop:  ${s['stop']:.2f}")
                print(f"    Live:  ${s['live']:.2f}")
                print()
        else:
            print('NO VALID TRADES - all signals gapped beyond 3%')
    else:
        print('NO SIGNALS from prior day - market was NOT oversold')

if __name__ == '__main__':
    main()
