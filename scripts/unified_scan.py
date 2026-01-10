#!/usr/bin/env python3
"""Unified scan: Stocks + Crypto + Options -> Top 5 -> Top 2"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from data.providers.alpaca_intraday import fetch_intraday_bars

def main():
    # From the 900 scan, top signals:
    top_equity_signals = [
        {'symbol': 'CVE', 'strategy': 'TurtleSoup', 'entry': 16.64, 'stop': 15.72, 'target': 17.33, 'score': 141.2},
        {'symbol': 'VTR', 'strategy': 'TurtleSoup', 'entry': 76.44, 'stop': 74.71, 'target': 77.74, 'score': 97.8},
        {'symbol': 'JNJ', 'strategy': 'TurtleSoup', 'entry': 204.31, 'stop': 200.30, 'target': 207.31, 'score': 78.2},
        {'symbol': 'AEP', 'strategy': 'TurtleSoup', 'entry': 114.07, 'stop': 112.24, 'target': 115.44, 'score': 76.5},
        {'symbol': 'PNR', 'strategy': 'TurtleSoup', 'entry': 102.67, 'stop': 100.87, 'target': 104.02, 'score': 71.9},
        {'symbol': 'EVRG', 'strategy': 'TurtleSoup', 'entry': 72.31, 'stop': 71.22, 'target': 73.13, 'score': 57.7},
        {'symbol': 'XBI', 'strategy': 'TurtleSoup', 'entry': 120.15, 'stop': 117.81, 'target': 121.91, 'score': 56.1},
        {'symbol': 'LNT', 'strategy': 'TurtleSoup', 'entry': 64.90, 'stop': 63.71, 'target': 65.79, 'score': 53.4},
        {'symbol': 'GILD', 'strategy': 'TurtleSoup', 'entry': 118.30, 'stop': 116.39, 'target': 119.74, 'score': 52.2},
        {'symbol': 'WELL', 'strategy': 'TurtleSoup', 'entry': 184.73, 'stop': 181.24, 'target': 187.35, 'score': 50.2},
    ]

    print('=' * 80)
    print('UNIFIED SCAN: Stocks + Crypto + Options -> Top 5 -> Top 2')
    print(f'Time: {datetime.now().isoformat()}')
    print('Prior Close: Monday 2026-01-05')
    print('=' * 80)
    print()

    # Validate equities with live data
    print('EQUITY SIGNALS (from 900 scan) - Validating with live data:')
    print('-' * 80)

    validated_equities = []
    for s in top_equity_signals:
        try:
            bars = fetch_intraday_bars(s['symbol'], timeframe='15Min', limit=5)
            if bars:
                live = bars[-1].close
                gap = ((live / s['entry']) - 1) * 100
                risk = s['entry'] - s['stop']
                reward = s['target'] - s['entry']
                rr = reward / risk if risk > 0 else 0

                gap_ok = abs(gap) < 3
                rr_ok = rr >= 1.5

                if gap_ok and rr_ok:
                    status = 'VALID'
                    s['live'] = live
                    s['gap'] = gap
                    s['rr'] = rr
                    s['asset_class'] = 'EQUITY'
                    s['conf_score'] = s['score']
                    validated_equities.append(s)
                else:
                    reasons = []
                    if not gap_ok:
                        reasons.append(f'gap={gap:.1f}%')
                    if not rr_ok:
                        reasons.append(f'rr={rr:.2f}')
                    status = 'FAIL (' + ', '.join(reasons) + ')'

                print(f"  {s['symbol']:5} | Entry=${s['entry']:.2f} | Live=${live:.2f} | R:R={rr:.2f}:1 | {status}")
        except Exception as e:
            print(f"  {s['symbol']:5} | Error: {str(e)[:40]}")

    print()

    # Scan Crypto from Polygon
    print('CRYPTO SIGNALS (from Polygon):')
    print('-' * 80)

    crypto_signals = []
    try:
        from scanner.crypto_signals import scan_crypto
        crypto_df = scan_crypto(cap=8, max_signals=3, verbose=False)
        if crypto_df is not None and not crypto_df.empty:
            for _, row in crypto_df.iterrows():
                sym = str(row.get('symbol', ''))
                entry = float(row.get('entry_price', 0))
                conf = float(row.get('conf_score', 0))
                stop = float(row.get('stop_loss', 0))
                target = float(row.get('take_profit', 0)) if pd.notna(row.get('take_profit')) else None

                print(f"  {sym}: Entry=${entry:.2f} | Conf={conf:.2f}")

                if entry > 0 and stop > 0:
                    crypto_signals.append({
                        'symbol': sym,
                        'strategy': 'TurtleSoup_CRYPTO',
                        'entry': entry,
                        'stop': stop,
                        'target': target or entry * 1.03,
                        'live': entry,
                        'gap': 0,
                        'rr': (target - entry) / (entry - stop) if target and stop != entry else 1.0,
                        'asset_class': 'CRYPTO',
                        'conf_score': conf * 100,
                        'score': conf * 100,
                    })
        else:
            print('  No crypto signals found')
    except Exception as e:
        print(f'  Crypto scan error: {e}')

    print()

    # Combine all validated signals
    all_signals = validated_equities + crypto_signals

    # Sort by conf_score
    all_signals.sort(key=lambda x: x.get('conf_score', 0), reverse=True)

    print('=' * 80)
    print('FINAL RANKING (All Asset Classes):')
    print('=' * 80)

    if all_signals:
        print(f"{'Rank':<5} {'Symbol':<8} {'Class':<8} {'Entry':>8} {'Stop':>8} {'R:R':>6} {'Score':>6} {'Status'}")
        print('-' * 80)
        for i, s in enumerate(all_signals[:5]):
            rr_str = f"{s.get('rr', 0):.2f}:1"
            print(f"{i+1:<5} {s['symbol']:<8} {s.get('asset_class', 'EQ'):<8} ${s['entry']:>7.2f} ${s['stop']:>7.2f} {rr_str:>6} {s.get('conf_score', 0):>6.1f} VALID")

        print()
        print('=' * 80)
        print('TOP 2 TRADEABLE:')
        print('=' * 80)

        top2 = all_signals[:2]
        for i, s in enumerate(top2):
            print(f"\nTRADE {i+1}: {s['symbol']} ({s['strategy']})")
            print(f"  Entry:  ${s['entry']:.2f}")
            print(f"  Stop:   ${s['stop']:.2f}")
            print(f"  Target: ${s['target']:.2f}")
            print(f"  Live:   ${s['live']:.2f} (gap {s['gap']:+.1f}%)")
            print(f"  R:R:    {s['rr']:.2f}:1")
            print(f"  Score:  {s['conf_score']:.1f}")

        # Save results
        results = {
            'scan_time': datetime.now().isoformat(),
            'prior_close': '2026-01-05',
            'validated_count': len(all_signals),
            'top5': all_signals[:5],
            'top2': top2,
        }

        Path('state/watchlist').mkdir(parents=True, exist_ok=True)
        with open('state/watchlist/unified_scan.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print()
        print('Results saved to: state/watchlist/unified_scan.json')

    else:
        print('NO VALID SIGNALS - All failed quality gate')
        print('  - Equities: R:R too low (< 1.5:1)')
        print('  - Crypto: No signals')
        print()
        print('RECOMMENDATION: Wait for EOD scan at 3:30 PM ET')

if __name__ == '__main__':
    main()
