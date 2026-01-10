#!/usr/bin/env python3
"""
Unified Multi-Asset Scanner - Stocks + Crypto + Options -> Top 5 -> Top 2

This is the CORRECT way a quant professional scans:
1. Strategy-specific quality gates (not blanket R:R)
2. Expected Value (EV) based filtering
3. All asset classes compete on same conf_score scale
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.providers.alpaca_intraday import fetch_intraday_bars
from strategies.dual_strategy.combined import DualStrategyScanner, DualStrategyParams

# Strategy-specific stats (from backtest)
STRATEGY_STATS = {
    'TurtleSoup': {'wr': 0.61, 'min_rr': 0.50},
    'TurtleSoup_CRYPTO': {'wr': 0.55, 'min_rr': 0.60},
    'IBS_RSI': {'wr': 0.60, 'min_rr': 1.0},
    'IBS_RSI_CRYPTO': {'wr': 0.55, 'min_rr': 1.0},
}

def calculate_ev(wr: float, rr: float) -> float:
    """Expected Value per $1 risked"""
    return (wr * rr) - ((1 - wr) * 1.0)

def passes_quality_gate(strategy: str, rr: float, gap_pct: float) -> tuple:
    """Strategy-specific quality gate using EV"""
    # Gap check (universal)
    if abs(gap_pct) > 0.03:
        return False, f'Gap {abs(gap_pct)*100:.1f}% > 3%'

    # Get strategy stats
    base_strategy = strategy.replace('_CRYPTO', '').replace('_CALL', '').replace('_PUT', '')
    stats = STRATEGY_STATS.get(base_strategy, {'wr': 0.50, 'min_rr': 1.5})

    # Min R:R for strategy
    if rr < stats['min_rr']:
        return False, f'R:R {rr:.2f} < {stats["min_rr"]}'

    # EV check
    ev = calculate_ev(stats['wr'], rr)
    if ev <= 0:
        return False, f'EV {ev:.4f} <= 0'

    return True, f'EV +{ev:.4f}'


def scan_all_assets():
    """Scan stocks, crypto, options and rank unified."""
    print('=' * 80)
    print('UNIFIED MULTI-ASSET SCAN')
    print(f'Time: {datetime.now().isoformat()}')
    print('Prior Close: Monday 2026-01-05')
    print('=' * 80)

    all_signals = []

    # ========== 1. EQUITY SIGNALS (from 900 scan results) ==========
    print()
    print('1. EQUITY SIGNALS')
    print('-' * 40)

    equity_candidates = [
        {'symbol': 'CVE', 'strategy': 'TurtleSoup', 'entry': 16.64, 'stop': 15.72, 'target': 17.33, 'score': 141.2},
        {'symbol': 'VTR', 'strategy': 'TurtleSoup', 'entry': 76.44, 'stop': 74.71, 'target': 77.74, 'score': 97.8},
        {'symbol': 'JNJ', 'strategy': 'TurtleSoup', 'entry': 204.31, 'stop': 200.30, 'target': 207.31, 'score': 78.2},
        {'symbol': 'AEP', 'strategy': 'TurtleSoup', 'entry': 114.07, 'stop': 112.24, 'target': 115.44, 'score': 76.5},
        {'symbol': 'PNR', 'strategy': 'TurtleSoup', 'entry': 102.67, 'stop': 100.87, 'target': 104.02, 'score': 71.9},
    ]

    for s in equity_candidates:
        try:
            bars = fetch_intraday_bars(s['symbol'], timeframe='15Min', limit=5)
            if bars:
                live = bars[-1].close
                gap = (live / s['entry']) - 1
                risk = s['entry'] - s['stop']
                reward = s['target'] - s['entry']
                rr = reward / risk if risk > 0 else 0

                passed, reason = passes_quality_gate(s['strategy'], rr, gap)
                status = 'PASS' if passed else 'FAIL'

                print(f"   {s['symbol']:5} | Live=${live:.2f} | R:R={rr:.2f} | {status}: {reason}")

                if passed:
                    wr = STRATEGY_STATS.get(s['strategy'], {}).get('wr', 0.5)
                    all_signals.append({
                        'symbol': s['symbol'],
                        'asset_class': 'EQUITY',
                        'strategy': s['strategy'],
                        'entry': s['entry'],
                        'stop': s['stop'],
                        'target': s['target'],
                        'live': live,
                        'gap_pct': gap,
                        'rr': rr,
                        'ev': calculate_ev(wr, rr),
                        'conf_score': s['score'],  # Keep original score for ranking
                    })
        except Exception as e:
            print(f"   {s['symbol']:5} | Error: {str(e)[:40]}")

    # ========== 2. CRYPTO SIGNALS ==========
    print()
    print('2. CRYPTO SIGNALS')
    print('-' * 40)

    try:
        from data.providers.polygon_crypto import fetch_crypto_bars

        crypto_pairs = ['X:BTCUSD', 'X:ETHUSD', 'X:SOLUSD', 'X:AVAXUSD', 'X:LINKUSD']
        params = DualStrategyParams(min_price=0.0)
        scanner = DualStrategyScanner(params)

        for pair in crypto_pairs:
            try:
                df = fetch_crypto_bars(pair, start='2025-06-01', end='2026-01-05', timeframe='1d')
                if df is not None and len(df) > 50:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    sig = scanner.generate_signals(df)

                    if not sig.empty:
                        r = sig.iloc[0]
                        entry = float(r.get('entry_price', 0))
                        stop = float(r.get('stop_loss', 0))
                        target = float(r.get('take_profit', 0)) if pd.notna(r.get('take_profit')) else None

                        live = df.iloc[-1]['close']
                        gap = (live / entry) - 1 if entry > 0 else 0
                        risk = entry - stop if stop > 0 else 1
                        reward = (target - entry) if target else 0
                        rr = reward / risk if risk > 0 else 0

                        strategy = r.get('strategy', 'Unknown') + '_CRYPTO'
                        passed, reason = passes_quality_gate(strategy, rr, gap)

                        clean_sym = pair.replace('X:', '')
                        status = 'PASS' if passed else 'FAIL'
                        print(f"   {clean_sym:8} | Live=${live:.2f} | R:R={rr:.2f} | {status}: {reason}")

                        if passed:
                            all_signals.append({
                                'symbol': clean_sym,
                                'asset_class': 'CRYPTO',
                                'strategy': strategy,
                                'entry': entry,
                                'stop': stop,
                                'target': target or entry * 1.03,
                                'live': live,
                                'gap_pct': gap,
                                'rr': rr,
                                'ev': calculate_ev(0.55, rr),
                                'conf_score': float(r.get('score', 50)),
                            })
                    else:
                        clean_sym = pair.replace('X:', '')
                        print(f"   {clean_sym:8} | No signal (conditions not met)")
                else:
                    clean_sym = pair.replace('X:', '')
                    print(f"   {clean_sym:8} | Insufficient data")
            except Exception as e:
                clean_sym = pair.replace('X:', '')
                print(f"   {clean_sym:8} | Error: {str(e)[:40]}")
    except ImportError as e:
        print(f"   Crypto provider not available: {e}")

    # ========== 3. OPTIONS SIGNALS ==========
    print()
    print('3. OPTIONS SIGNALS (from top equities)')
    print('-' * 40)

    try:
        from scanner.options_signals import OptionsSignalGenerator

        # Get validated equity signals for options
        equity_for_options = [s for s in all_signals if s['asset_class'] == 'EQUITY'][:3]

        if equity_for_options:
            eq_df = pd.DataFrame(equity_for_options)
            eq_df['entry_price'] = eq_df['entry']

            # Get price data for volatility
            price_data_list = []
            for s in equity_for_options:
                try:
                    df = fetch_daily_bars_polygon(s['symbol'], start='2025-10-01', end='2026-01-05', cache_dir=None)
                    if df is not None and len(df) > 20:
                        df['symbol'] = s['symbol']
                        price_data_list.append(df)
                except Exception:
                    pass

            if price_data_list:
                price_data = pd.concat(price_data_list)
                gen = OptionsSignalGenerator(target_delta=0.30, target_dte=21)
                options_df = gen.generate_from_equity_signals(eq_df, price_data, max_signals=6)

                if not options_df.empty:
                    for _, row in options_df.iterrows():
                        sym = row.get('symbol', '')
                        opt_type = row.get('option_type', '')
                        strike = row.get('strike', 0)
                        price = row.get('option_price', 0)
                        delta = row.get('delta', 0)
                        conf = row.get('conf_score', 0.5)

                        print(f"   {sym} {strike:.0f}{opt_type[0]} | Premium=${price:.2f} | Delta={delta:.2f} | Conf={conf:.2f}")

                        # Options compete with adjusted conf_score (already discounted)
                        all_signals.append({
                            'symbol': f"{sym}_{strike:.0f}{opt_type[0]}",
                            'asset_class': 'OPTIONS',
                            'strategy': row.get('strategy', 'Options'),
                            'entry': price,
                            'stop': 0,  # Max loss is premium
                            'target': price * 2,  # 100% target
                            'live': price,
                            'gap_pct': 0,
                            'rr': 1.0,  # Defined risk/reward
                            'ev': conf * 0.5,  # Options EV estimate
                            'conf_score': conf * 100,  # Scale to match equities
                        })
                else:
                    print("   No options signals generated")
            else:
                print("   No price data for options")
        else:
            print("   No equity signals for options generation")
    except ImportError as e:
        print(f"   Options module error: {e}")
    except Exception as e:
        print(f"   Options error: {e}")

    # ========== UNIFIED RANKING ==========
    print()
    print('=' * 80)
    print('UNIFIED RANKING (All Asset Classes)')
    print('=' * 80)

    if not all_signals:
        print('NO SIGNALS PASSED QUALITY GATE')
        return

    # Sort by conf_score (maintains strategy-specific scoring)
    all_signals.sort(key=lambda x: x['conf_score'], reverse=True)

    print()
    print(f"{'Rank':<5} {'Symbol':<12} {'Class':<8} {'Entry':>10} {'R:R':>8} {'EV':>8} {'Score':>8}")
    print('-' * 80)

    for i, s in enumerate(all_signals[:10]):
        rr_str = f"{s['rr']:.2f}:1" if s['rr'] > 0 else 'N/A'
        ev_str = f"+{s['ev']:.4f}" if s['ev'] > 0 else f"{s['ev']:.4f}"
        print(f"{i+1:<5} {s['symbol']:<12} {s['asset_class']:<8} ${s['entry']:>9.2f} {rr_str:>8} {ev_str:>8} {s['conf_score']:>8.1f}")

    # Top 5 watchlist
    top5 = all_signals[:5]

    # Top 2 to trade
    top2 = all_signals[:2]

    print()
    print('=' * 80)
    print('TOP 2 TRADES')
    print('=' * 80)

    for i, t in enumerate(top2):
        print(f"\nTRADE {i+1}: {t['symbol']} ({t['asset_class']})")
        print(f"  Strategy: {t['strategy']}")
        print(f"  Entry:    ${t['entry']:.2f}")
        print(f"  Stop:     ${t['stop']:.2f}")
        print(f"  Target:   ${t['target']:.2f}")
        print(f"  Live:     ${t['live']:.2f} (gap {t['gap_pct']*100:+.1f}%)")
        print(f"  R:R:      {t['rr']:.2f}:1")
        print(f"  EV:       +{t['ev']:.4f} per $1 risked")
        print(f"  Score:    {t['conf_score']:.1f}")

    # Save results
    results = {
        'scan_time': datetime.now().isoformat(),
        'total_signals': len(all_signals),
        'by_asset_class': {
            'equity': len([s for s in all_signals if s['asset_class'] == 'EQUITY']),
            'crypto': len([s for s in all_signals if s['asset_class'] == 'CRYPTO']),
            'options': len([s for s in all_signals if s['asset_class'] == 'OPTIONS']),
        },
        'top5': top5,
        'top2': top2,
    }

    Path('state/watchlist').mkdir(parents=True, exist_ok=True)
    with open('state/watchlist/multi_asset_scan.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print('Results saved to: state/watchlist/multi_asset_scan.json')


if __name__ == '__main__':
    scan_all_assets()
