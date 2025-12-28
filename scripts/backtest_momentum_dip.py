#!/usr/bin/env python3
"""
Backtest Momentum Dip Strategy - Quant Interview Ready

Expected: 70% win rate, 1.3+ profit factor, 1+ trade/day from 900 stocks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

env_path = Path("C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env")
if env_path.exists():
    load_dotenv(env_path)

from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.universe.loader import load_universe
from strategies.momentum_dip import MomentumDipStrategy, MomentumDipParams


def run_backtest(symbols: List[str], start: str, end: str, params: MomentumDipParams, max_symbols: int) -> Dict:
    """Run backtest with proper trade simulation."""
    strategy = MomentumDipStrategy(params)
    cache_dir = Path("data/cache/polygon")

    print(f"Fetching data for {min(len(symbols), max_symbols)} symbols...")
    all_data = []
    for i, sym in enumerate(symbols[:max_symbols]):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{min(len(symbols), max_symbols)}...")
        try:
            df = fetch_daily_bars_polygon(sym, start, end, cache_dir=cache_dir)
            if df is not None and len(df) > 220:
                if 'symbol' not in df.columns:
                    df['symbol'] = sym
                all_data.append(df)
        except:
            pass

    if not all_data:
        return {'error': 'No data'}

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Got {len(all_data)} symbols, {len(combined)} bars")

    # Compute indicators
    combined = strategy._compute(combined)

    # Generate signals
    print("Generating signals...")
    signals = strategy.scan_signals_over_time(combined.copy())
    print(f"Generated {len(signals)} signals")

    if signals.empty:
        return {'signals': 0, 'trades': 0, 'win_rate': 0}

    # Simulate trades
    print("Simulating trades...")
    trades = []

    for _, sig in signals.iterrows():
        sym = sig['symbol']
        sym_data = combined[combined['symbol'] == sym].sort_values('timestamp')

        entry_ts = sig['timestamp']
        mask = sym_data['timestamp'] > entry_ts
        if not mask.any():
            continue

        entry_idx = mask.idxmax()
        entry_row = sym_data.loc[entry_idx]
        entry_price = float(entry_row['open'])
        stop_price = sig['stop_loss']

        exit_price = None
        exit_reason = None
        bars_held = 0

        future = sym_data.loc[entry_idx:].iloc[1:]
        for _, bar in future.iterrows():
            bars_held += 1
            close = float(bar['close'])
            low = float(bar['low'])
            rsi2 = bar.get('rsi2')

            # Check ATR stop
            if low <= stop_price:
                exit_price = stop_price
                exit_reason = 'atr_stop'
                break

            # Check RSI exit
            if rsi2 is not None and float(rsi2) > params.rsi_exit:
                exit_price = close
                exit_reason = 'rsi_exit'
                break

            # Time stop
            if bars_held >= params.time_stop_bars:
                exit_price = close
                exit_reason = 'time_stop'
                break

        if exit_price is None:
            if len(future) > 0:
                exit_price = float(future.iloc[-1]['close'])
                exit_reason = 'end_of_data'
            else:
                continue

        pnl = exit_price - entry_price
        pnl_pct = (pnl / entry_price) * 100

        trades.append({
            'symbol': sym,
            'entry_date': entry_ts,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'win': pnl > 0,
            'cum_rsi': sig['cum_rsi'],
        })

    if not trades:
        return {'signals': len(signals), 'trades': 0, 'win_rate': 0}

    trades_df = pd.DataFrame(trades)

    # Metrics
    wins = trades_df['win'].sum()
    losses = len(trades_df) - wins
    win_rate = wins / len(trades_df) * 100

    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if losses > 0 else 0

    # Trading days
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    days = (end_dt - start_dt).days * 252 / 365
    signals_per_day = len(signals) / days if days > 0 else 0

    return {
        'symbols_tested': len(all_data),
        'signals': len(signals),
        'trades': len(trades_df),
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 1),
        'profit_factor': round(pf, 2),
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'avg_bars_held': round(trades_df['bars_held'].mean(), 1),
        'signals_per_day': round(signals_per_day, 2),
        'exit_reasons': trades_df['exit_reason'].value_counts().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe', default='data/universe/optionable_liquid_900.csv')
    parser.add_argument('--start', default='2024-01-01')
    parser.add_argument('--end', default='2025-12-26')
    parser.add_argument('--cap', type=int, default=100)
    args = parser.parse_args()

    symbols = load_universe(args.universe)
    params = MomentumDipParams()

    print(f"\n{'='*60}")
    print("MOMENTUM DIP STRATEGY BACKTEST")
    print(f"{'='*60}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Universe: {len(symbols)} symbols (testing {args.cap})")
    print(f"Entry: CumRSI<{params.cum_rsi_entry}, >SMA200, >SMA50, Ret20>0")
    print(f"Exit: RSI>{params.rsi_exit}, ATR*{params.atr_stop_mult} stop, {params.time_stop_bars}-bar time")
    print(f"{'='*60}\n")

    results = run_backtest(symbols, args.start, args.end, params, args.cap)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    if 'error' in results:
        print(f"Error: {results['error']}")
        return 1

    print(f"Symbols Tested:    {results['symbols_tested']}")
    print(f"Signals Generated: {results['signals']}")
    print(f"Trades Executed:   {results['trades']}")
    print(f"Signals/Day:       {results['signals_per_day']}")

    print(f"\nPerformance:")
    print(f"  Win Rate:        {results['win_rate']}%")
    print(f"  Wins/Losses:     {results['wins']}/{results['losses']}")
    print(f"  Profit Factor:   {results['profit_factor']}")
    print(f"  Avg Win:         +{results['avg_win_pct']}%")
    print(f"  Avg Loss:        {results['avg_loss_pct']}%")
    print(f"  Avg Bars Held:   {results['avg_bars_held']}")

    print(f"\nExit Reasons:")
    for reason, count in results.get('exit_reasons', {}).items():
        print(f"  {reason}: {count}")

    # Scale to 900 stocks
    scale = 900 / results['symbols_tested'] if results['symbols_tested'] > 0 else 1
    projected_spd = results['signals_per_day'] * scale

    print(f"\n{'='*60}")
    print("QUANT INTERVIEW CRITERIA")
    print(f"{'='*60}")

    wr_pass = results['win_rate'] >= 60
    pf_pass = results['profit_factor'] >= 1.3
    spd_pass = projected_spd >= 1.0

    print(f"Win Rate >= 60%:            {'PASS' if wr_pass else 'FAIL'} ({results['win_rate']}%)")
    print(f"Profit Factor >= 1.3:       {'PASS' if pf_pass else 'FAIL'} ({results['profit_factor']})")
    print(f"Signals/Day >= 1 (900 stk): {'PASS' if spd_pass else 'FAIL'} ({projected_spd:.1f} projected)")

    if wr_pass and pf_pass and spd_pass:
        print(f"\n{'*'*60}")
        print("*** ALL CRITERIA PASSED - QUANT INTERVIEW READY ***")
        print(f"{'*'*60}")
    else:
        print("\n*** OPTIMIZATION NEEDED ***")

    return 0


if __name__ == '__main__':
    sys.exit(main())
