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

# from strategies.momentum_dip import MomentumDipStrategy, MomentumDipParams


def run_backtest(symbols: List[str], start: str, end: str, params: Any, max_symbols: int) -> Dict:
    """Run backtest with proper trade simulation."""
    # strategy = MomentumDipStrategy(params) # Commented out strategy initialization
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
    # combined = strategy._compute(combined) # Commented out indicator computation

    # Generate signals
    print("Generating signals...")
    # signals = strategy.scan_signals_over_time(combined.copy()) # Commented out signal generation
    # ...
    return {'error': 'Strategy not implemented'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe', default='data/universe/optionable_liquid_900.csv')
    parser.add_argument('--start', default='2024-01-01')
    parser.add_argument('--end', default='2025-12-26')
    parser.add_argument('--cap', type=int, default=100)
    args = parser.parse_args()

    symbols = load_universe(args.universe)
    # params = MomentumDipParams() # Commented out params initialization

    print(f"\n{'='*60}")
    print("MOMENTUM DIP STRATEGY BACKTEST")
    print(f"{'='*60}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Universe: {len(symbols)} symbols (testing {args.cap})")
    # print(f"Entry: CumRSI<{params.cum_rsi_entry}, >SMA200, >SMA50, Ret20>0") # Commented out params print
    # print(f"Exit: RSI>{params.rsi_exit}, ATR*{params.atr_stop_mult} stop, {params.time_stop_bars}-bar time") # Commented out params print
    print(f"{'='*60}\n")

    results = run_backtest(symbols, args.start, args.end, None, args.cap) # Changed params to None

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
    # scale = 900 / results['symbols_tested'] if results['symbols_tested'] > 0 else 1
    # projected_spd = results['signals_per_day'] * scale

    print(f"\n{'='*60}")
    print("QUANT INTERVIEW CRITERIA")
    print(f"{'='*60}")

    # wr_pass = results['win_rate'] >= 60
    # pf_pass = results['profit_factor'] >= 1.3
    # spd_pass = projected_spd >= 1.0

    # print(f"Win Rate >= 60%:            {'PASS' if wr_pass else 'FAIL'} ({results['win_rate']}%)")
    # print(f"Profit Factor >= 1.3:       {'PASS' if pf_pass else 'FAIL'} ({results['profit_factor']})")
    # print(f"Signals/Day >= 1 (900 stk): {'PASS' if spd_pass else 'FAIL'} ({projected_spd:.1f} projected)")

    # if wr_pass and pf_pass and spd_pass:
    #     print(f"\n{'*'*60}")
    #     print("*** ALL CRITERIA PASSED - QUANT INTERVIEW READY ***")
    #     print(f"{'*'*60}")
    # else:
    #     print("\n*** OPTIMIZATION NEEDED ***")
    print("Skipping success criteria check for unimplemented strategy.")
    return 1 # Indicate error/skip
