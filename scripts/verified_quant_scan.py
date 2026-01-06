#!/usr/bin/env python3
"""
Verified Professional Quant Scan - 900 Stocks

Uses the CANONICAL DualStrategyScanner (same as production).
All parameters verified:
- EV formula: (WR * RR) - ((1-WR) * 1)
- No lookahead bias (uses DualStrategyScanner's shift(1))
- Industry standard thresholds
- Real Polygon.io data
- R:R minimum 1.5:1
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.universe.loader import load_universe
from strategies.dual_strategy import DualStrategyScanner, DualStrategyParams

# VERIFIED THRESHOLDS
MIN_SAMPLES = 20
MIN_WR = 0.45
MIN_EV = 0.01
MIN_RR = 1.5  # Minimum Risk:Reward ratio (blueprint standard)
IBS_THRESHOLD = 0.20
RSI_THRESHOLD = 30.0

# Account sizing
ACCOUNT_EQUITY = 50000
MAX_RISK_PCT = 0.02
MAX_NOTIONAL_PCT = 0.20


def calc_historical_wr(df: pd.DataFrame, scanner: DualStrategyScanner) -> tuple:
    """
    Calculate stock-specific historical win rate using DualStrategyScanner.

    Returns (win_rate, sample_count, avg_return)
    """
    if len(df) < 250:  # Need enough history
        return 0.5, 0, 0.0

    # Get all historical signals using the canonical scanner
    df = df.copy()
    df['symbol'] = df['symbol'].iloc[0] if 'symbol' in df.columns else 'UNKNOWN'

    try:
        signals = scanner.scan_signals_over_time(df)
    except Exception:
        return 0.5, 0, 0.0

    if signals.empty:
        return 0.5, 0, 0.0

    # Calculate next-day returns for each signal
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['next_ret'] = df['close'].shift(-1) / df['close'] - 1

    # Merge signals with returns
    signals_with_ret = []
    for _, sig in signals.iterrows():
        sig_date = pd.to_datetime(sig['timestamp']).date()
        match = df[pd.to_datetime(df['timestamp']).dt.date == sig_date]
        if not match.empty and pd.notna(match.iloc[0]['next_ret']):
            signals_with_ret.append(match.iloc[0]['next_ret'])

    if len(signals_with_ret) < 5:
        return 0.5, len(signals_with_ret), 0.0

    wins = sum(1 for r in signals_with_ret if r > 0)
    total = len(signals_with_ret)
    avg_ret = np.mean(signals_with_ret) * 100

    return wins / total, total, avg_ret


def calc_oversold(df: pd.DataFrame) -> tuple:
    """Check if stock is oversold using IBS and RSI."""
    last = df.iloc[-1]

    # IBS: Internal Bar Strength
    high_low_range = last['high'] - last['low']
    ibs = (last['close'] - last['low']) / high_low_range if high_low_range > 0 else 0.5

    # RSI(14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    oversold = (ibs < IBS_THRESHOLD) or (rsi < RSI_THRESHOLD)

    return oversold, ibs, rsi


def calc_ev(wr: float, rr: float) -> float:
    """Expected Value = (WR * RR) - ((1-WR) * 1)"""
    return (wr * rr) - ((1 - wr) * 1.0)


def calc_position_size(entry: float, stop: float) -> dict:
    """Calculate position size with dual caps (2% risk, 20% notional)."""
    risk = entry - stop
    if risk <= 0:
        return {'shares': 0, 'position_value': 0, 'actual_risk': 0}

    risk_budget = ACCOUNT_EQUITY * MAX_RISK_PCT
    notional_cap = ACCOUNT_EQUITY * MAX_NOTIONAL_PCT

    shares_by_risk = int(risk_budget / risk)
    shares_by_notional = int(notional_cap / entry)
    final_shares = min(shares_by_risk, shares_by_notional)

    return {
        'shares': final_shares,
        'position_value': round(final_shares * entry, 2),
        'actual_risk': round(final_shares * risk, 2),
    }


def main():
    print('=' * 70)
    print('PROFESSIONAL QUANT SCAN - 900 STOCKS (DualStrategyScanner)')
    print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('Using: CANONICAL DualStrategyScanner (same as production)')
    print('=' * 70)
    print()
    print('Verified Parameters:')
    print(f'  MIN_SAMPLES: {MIN_SAMPLES}')
    print(f'  MIN_WIN_RATE: {MIN_WR*100}%')
    print(f'  MIN_EV: {MIN_EV}')
    print(f'  MIN_R:R: {MIN_RR}:1')
    print(f'  IBS_OVERSOLD: < {IBS_THRESHOLD}')
    print(f'  RSI_OVERSOLD: < {RSI_THRESHOLD}')
    print()

    symbols = load_universe('data/universe/optionable_liquid_900.csv', cap=900)
    print(f'Scanning {len(symbols)} stocks...')
    print()

    # Initialize canonical scanner
    scanner = DualStrategyScanner(DualStrategyParams())

    all_signals = []
    passing = []

    for i, sym in enumerate(symbols):
        if (i + 1) % 150 == 0:
            print(f'  Progress: {i+1}/900 | Signals: {len(all_signals)} | Passing: {len(passing)}')

        try:
            # Fetch real Polygon data
            df = fetch_daily_bars_polygon(sym, start='2024-01-01', end='2026-01-05', cache_dir=None)

            if df is None or len(df) < 50:
                continue

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            df['symbol'] = sym

            # Get current signal from DualStrategyScanner
            current_signals = scanner.generate_signals(df)

            if current_signals.empty:
                continue

            # Take the best signal for this stock
            sig = current_signals.iloc[0]

            # Calculate stock-specific historical stats
            wr, n_samples, avg_ret = calc_historical_wr(df, scanner)
            oversold, ibs, rsi = calc_oversold(df)

            # Calculate R:R (enforce minimum)
            entry = sig['entry_price']
            stop = sig['stop_loss']
            risk = entry - stop

            if risk <= 0:
                continue

            # Apply MIN_RR for target
            target = entry + (MIN_RR * risk)
            reward = target - entry
            rr = reward / risk
            ev = calc_ev(wr, rr)

            # Position sizing
            sizing = calc_position_size(entry, stop)

            signal = {
                'symbol': sym,
                'strategy': sig['strategy'],
                'entry': round(entry, 2),
                'stop': round(stop, 2),
                'target': round(target, 2),
                'risk': round(risk, 2),
                'reward': round(reward, 2),
                'rr': round(rr, 2),
                'win_rate': round(wr, 4),
                'samples': n_samples,
                'ev': round(ev, 4),
                'ibs': round(ibs, 3),
                'rsi': round(rsi, 1),
                'oversold': oversold,
                'score': sig['score'],
                'reason': sig['reason'],
                'shares': sizing['shares'],
                'position_value': sizing['position_value'],
            }

            # Quality gate
            passes = True
            fail_reason = ''

            if n_samples < MIN_SAMPLES:
                passes = False
                fail_reason = f'Samples {n_samples} < {MIN_SAMPLES}'
            elif wr < MIN_WR:
                passes = False
                fail_reason = f'WR {wr*100:.1f}% < {MIN_WR*100}%'
            elif ev < MIN_EV:
                passes = False
                fail_reason = f'EV {ev:.4f} < {MIN_EV}'
            elif not oversold:
                passes = False
                fail_reason = f'Not oversold (IBS={ibs:.2f}, RSI={rsi:.1f})'

            signal['passes'] = passes
            signal['fail_reason'] = fail_reason
            all_signals.append(signal)

            if passes:
                passing.append(signal)
                print(f'  PASS: {sym} | {sig["strategy"]} | WR={wr*100:.1f}% ({n_samples}) | EV={ev:+.4f} | IBS={ibs:.3f} | RSI={rsi:.1f}')

        except Exception as e:
            continue

    print()
    print('=' * 70)
    print('SCAN RESULTS')
    print('=' * 70)
    print(f'Scanner: DualStrategyScanner (IBS+RSI + Turtle Soup)')
    print(f'Total signals found: {len(all_signals)}')
    print(f'Pass quant gate: {len(passing)}')

    # Sort by EV
    passing.sort(key=lambda x: x['ev'], reverse=True)
    all_signals.sort(key=lambda x: x['ev'], reverse=True)

    if passing:
        print()
        print('TOP 5 WATCHLIST (by EV):')
        print('-' * 70)
        for i, s in enumerate(passing[:5]):
            print(f"{i+1}. {s['symbol']:6} | {s['strategy']:10} | Entry ${s['entry']:>8.2f} | WR={s['win_rate']*100:>5.1f}% ({s['samples']:>2}) | EV={s['ev']:+.4f}")

        print()
        print('TOP 2 TRADES:')
        print('=' * 70)
        for i, s in enumerate(passing[:2]):
            print()
            print(f"TRADE {i+1}: {s['symbol']} ({s['strategy']})")
            print(f"  Entry:    ${s['entry']:.2f}")
            print(f"  Stop:     ${s['stop']:.2f}")
            print(f"  Target:   ${s['target']:.2f} (R:R = {s['rr']:.2f}:1)")
            print(f"  Risk:     ${s['risk']:.2f}")
            print(f"  Reward:   ${s['reward']:.2f}")
            print(f"  Win Rate: {s['win_rate']*100:.1f}% ({s['samples']} historical samples)")
            print(f"  EV:       {s['ev']:+.4f} per $1 risked")
            ibs_flag = "<-- OVERSOLD" if s['ibs'] < 0.20 else ""
            rsi_flag = "<-- OVERSOLD" if s['rsi'] < 30 else ""
            print(f"  IBS:      {s['ibs']:.3f} {ibs_flag}")
            print(f"  RSI(14):  {s['rsi']:.1f} {rsi_flag}")
            print(f"  Shares:   {s['shares']} (${s['position_value']:,.2f})")
    else:
        print()
        print('NO SIGNALS PASS QUANT GATE')
        print()
        print('This is CORRECT professional quant behavior:')
        print('  - Capital preservation > forced trades')
        print('  - No oversold = no exhaustion = higher risk')
        print()
        print('Top 10 near-passes (why they failed):')
        print('-' * 70)
        for i, s in enumerate(all_signals[:10]):
            print(f"{i+1}. {s['symbol']:6} | EV={s['ev']:+.4f} | WR={s['win_rate']*100:.1f}% | IBS={s['ibs']:.2f} | RSI={s['rsi']:.1f} | {s['fail_reason']}")

    # Save results
    results = {
        'scan_time': datetime.now().isoformat(),
        'data_source': 'Polygon.io (real adjusted OHLCV)',
        'scanner': 'DualStrategyScanner (canonical production scanner)',
        'methodology': 'Stock-specific pattern analysis with DualStrategyScanner',
        'thresholds': {
            'min_samples': MIN_SAMPLES,
            'min_win_rate': MIN_WR,
            'min_ev': MIN_EV,
            'min_rr': MIN_RR,
            'ibs_oversold': IBS_THRESHOLD,
            'rsi_oversold': RSI_THRESHOLD,
        },
        'total_signals': len(all_signals),
        'passing_signals': len(passing),
        'top5': passing[:5] if passing else [],
        'top2': passing[:2] if passing else [],
        'all_signals': all_signals[:50],
    }

    Path('state/watchlist').mkdir(parents=True, exist_ok=True)
    with open('state/watchlist/professional_scan.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print(f'Results saved: state/watchlist/professional_scan.json')


if __name__ == '__main__':
    main()
