#!/usr/bin/env python3
"""Full 900-stock scan with both strategies (IBS+RSI + TurtleSoup)."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from strategies.dual_strategy.combined import DualStrategyScanner, DualStrategyParams
from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.providers.alpaca_intraday import fetch_intraday_bars
from data.universe.loader import load_universe

def main():
    print('=' * 80)
    print('FULL 900-STOCK SCAN - Both Strategies (IBS+RSI + TurtleSoup)')
    print(f'Scan Time: {datetime.now().isoformat()}')
    print('Prior Close: Monday 2026-01-05')
    print('=' * 80)
    print()

    # Load full 900-stock universe
    universe_file = 'data/universe/optionable_liquid_800.csv'
    symbols = load_universe(universe_file, cap=900)
    print(f'Universe loaded: {len(symbols)} stocks')

    # Initialize scanner with both strategies
    params = DualStrategyParams()
    scanner = DualStrategyScanner(params)

    print('Scanning all stocks with fresh Monday (2026-01-05) data...')
    print()

    all_signals = []
    ibs_rsi_count = 0
    turtle_soup_count = 0
    errors = 0

    for i, sym in enumerate(symbols):
        if (i+1) % 100 == 0:
            print(f'  Progress: {i+1}/{len(symbols)} | Signals: {len(all_signals)} (IBS_RSI: {ibs_rsi_count}, TurtleSoup: {turtle_soup_count})')

        try:
            # Fetch fresh data (bypass cache)
            df = fetch_daily_bars_polygon(sym, start='2024-01-01', end='2026-01-05', cache_dir=None)
            if df is None or len(df) < 50:
                continue

            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Generate signal
            sig = scanner.generate_signals(df)
            if not sig.empty:
                r = sig.iloc[0]
                strategy = r.get('strategy', '')
                entry = float(r.get('entry_price', 0))
                stop = float(r.get('stop_loss', 0))
                target = float(r.get('take_profit', 0)) if pd.notna(r.get('take_profit')) else None

                # Calculate R:R
                if target and stop != entry:
                    rr = (target - entry) / (entry - stop)
                else:
                    rr = 0

                all_signals.append({
                    'symbol': sym,
                    'strategy': strategy,
                    'entry': entry,
                    'stop': stop,
                    'target': target,
                    'rr': rr,
                    'reason': str(r.get('reason', ''))[:60],
                    'score': float(r.get('score', 50))
                })

                if 'IBS' in strategy:
                    ibs_rsi_count += 1
                elif 'Turtle' in strategy:
                    turtle_soup_count += 1

        except Exception:
            errors += 1

    print()
    print('=' * 80)
    print('SCAN COMPLETE')
    print('=' * 80)
    print(f'Stocks scanned: {len(symbols)}')
    print(f'Total signals: {len(all_signals)}')
    print(f'  - IBS+RSI: {ibs_rsi_count}')
    print(f'  - TurtleSoup: {turtle_soup_count}')
    print(f'Errors: {errors}')
    print()

    if not all_signals:
        print('NO SIGNALS FOUND')
        print('Market was not oversold on Monday')
        return

    # Sort by score descending
    all_signals.sort(key=lambda x: (x['score'], x['rr']), reverse=True)

    # Top 5 watchlist
    top5 = all_signals[:5]

    print('TOP 5 WATCHLIST (by score):')
    print('-' * 80)
    print(f'{"Rank":<5} {"Symbol":<6} {"Strategy":<12} {"Entry":>8} {"Stop":>8} {"Target":>8} {"R:R":>6} {"Score":>6}')
    print('-' * 80)

    for i, s in enumerate(top5):
        target_str = f"${s['target']:.2f}" if s['target'] else "ATR"
        rr_str = f"{s['rr']:.2f}:1" if s['rr'] > 0 else "N/A"
        print(f"{i+1:<5} {s['symbol']:<6} {s['strategy']:<12} ${s['entry']:>7.2f} ${s['stop']:>7.2f} {target_str:>8} {rr_str:>6} {s['score']:>6.1f}")

    print()
    print('VALIDATING TOP 5 WITH TODAY\'S LIVE DATA:')
    print('-' * 80)

    validated = []
    for s in top5:
        try:
            bars = fetch_intraday_bars(s['symbol'], timeframe='15Min', limit=5)
            if bars:
                live = bars[-1].close
                gap = ((live / s['entry']) - 1) * 100
                gap_ok = abs(gap) < 3
                rr_ok = s['rr'] >= 1.5

                if gap_ok and rr_ok:
                    status = 'VALID'
                    s['live'] = live
                    s['gap'] = gap
                    validated.append(s)
                elif not gap_ok:
                    status = f'GAP_FAIL ({gap:+.1f}%)'
                else:
                    status = f'RR_FAIL ({s["rr"]:.2f}:1)'

                print(f"  {s['symbol']:5} | Live=${live:.2f} | Gap={gap:+.1f}% | R:R={s['rr']:.2f}:1 | {status}")
            else:
                print(f"  {s['symbol']:5} | No live data")
        except Exception as e:
            print(f"  {s['symbol']:5} | Error: {str(e)[:40]}")

    print()
    print('=' * 80)

    if validated:
        print(f'TOP 2 TRADEABLE ({len(validated)} passed quality gate):')
        print('=' * 80)

        for i, v in enumerate(validated[:2]):
            print(f"\nTRADE {i+1}: {v['symbol']} ({v['strategy']})")
            print(f"  Entry:  ${v['entry']:.2f}")
            print(f"  Stop:   ${v['stop']:.2f}")
            print(f"  Target: ${v['target']:.2f}" if v['target'] else "  Target: ATR-based")
            print(f"  Live:   ${v['live']:.2f} (gap {v['gap']:+.1f}%)")
            print(f"  R:R:    {v['rr']:.2f}:1")
            print(f"  Score:  {v['score']:.1f}")
            print(f"  Reason: {v['reason']}")
    else:
        print('NO SIGNALS PASS QUALITY GATE')
        print('  - All signals either gapped > 3% or R:R < 1.5:1')
        print('  - Wait for EOD scan at 3:30 PM ET')

    # Save results
    import json
    results = {
        'scan_time': datetime.now().isoformat(),
        'prior_close': '2026-01-05',
        'total_signals': len(all_signals),
        'ibs_rsi_count': ibs_rsi_count,
        'turtle_soup_count': turtle_soup_count,
        'top5': top5,
        'validated': validated,
        'top2': validated[:2] if validated else []
    }

    Path('state/watchlist').mkdir(parents=True, exist_ok=True)
    with open('state/watchlist/full_scan.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print('Results saved to: state/watchlist/full_scan.json')

if __name__ == '__main__':
    main()
