#!/usr/bin/env python3
"""
Backtest Mean Reversion Strategy on 2025 data.

Usage:
    python scripts/backtest_mean_reversion.py --universe data/universe/optionable_liquid_900.csv --start 2025-01-01 --end 2025-12-27
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
env_path = Path("C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env")
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try local .env
    load_dotenv()

# from strategies.mean_reversion import MeanReversionStrategy, MeanReversionParams


def run_backtest(
    symbols: List[str],
    start: str,
    end: str,
    params: Any, # Changed to Any as params type will be undefined
    capital: float = 100_000,
    risk_per_trade: float = 0.01,
    max_symbols: int = 50,
) -> Dict:
    """Run backtest on mean reversion strategy.

    Returns dict with metrics.
    """
    # strategy = MeanReversionStrategy(params) # Commented out strategy initialization
    cache_dir = Path("data/cache/polygon")

    # Fetch data for all symbols
    print(f"Fetching data for {len(symbols)} symbols from {start} to {end}...")
    all_data = []
    symbols_with_data = 0

    for i, sym in enumerate(symbols[:max_symbols]):
        if (i + 1) % 10 == 0:
            print(f"  Fetching {i+1}/{min(len(symbols), max_symbols)}...")
        try:
            df = fetch_daily_bars_polygon(sym, start, end, cache_dir=cache_dir)
            if df is not None and len(df) > 220:  # Need 200+ bars for SMA(200)
                # Ensure symbol column exists
                if 'symbol' not in df.columns:
                    df['symbol'] = sym
                all_data.append(df)
                symbols_with_data += 1
        except Exception as e:
            pass  # Skip symbols with errors

    if not all_data:
        return {'error': 'No data available'}

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Got data for {symbols_with_data} symbols, {len(combined)} total bars")

    # Generate signals
    print("Generating signals...")
    # signals = strategy.scan_signals_over_time(combined) # Commented out signal generation
    # ...
    # Calculate R multiple
    # risk = entry_price - stop if stop else entry_price * 0.02
    # pnl = exit_price - entry_price
    # r_mult = pnl / risk if risk > 0 else 0

    # trades.append({
    #     'symbol': sym,
    #     'entry_date': entry_ts,
    #     'entry_price': entry_price,
    #     'exit_price': exit_price,
    #     'exit_reason': exit_reason,
    #     'bars_held': bars_held,
    #     'pnl': pnl,
    #     'r_multiple': r_mult,
    #     'win': pnl > 0,
    #     'reason': sig['reason'],
    # })
    # ...
    return {'error': 'Strategy not implemented'}


def main():
    parser = argparse.ArgumentParser(description='Backtest Mean Reversion Strategy')
    parser.add_argument('--universe', type=str,
                        default='data/universe/optionable_liquid_900.csv',
                        help='Path to universe CSV')
    parser.add_argument('--start', type=str, default='2025-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-12-27',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--cap', type=int, default=100,
                        help='Max symbols to test')
    parser.add_argument('--rsi-entry', type=float, default=10.0,
                        help='RSI(2) entry threshold')
    parser.add_argument('--ibs-entry', type=float, default=0.25,
                        help='IBS entry threshold')
    parser.add_argument('--down-days', type=int, default=2,
                        help='Consecutive down days threshold')
    parser.add_argument('--require-multiple', action='store_true',
                        help='Require 2+ conditions for entry')
    args = parser.parse_args()

    # Load universe
    print(f"Loading universe from {args.universe}...")
    symbols = load_universe(args.universe)
    print(f"Loaded {len(symbols)} symbols")

    # Create params
    # params = MeanReversionParams( # Commented out params initialization
    #     rsi_entry=args.rsi_entry,
    #     ibs_entry=args.ibs_entry,
    #     down_days=args.down_days,
    #     require_multiple=args.require_multiple,
    # )

    print(f"\n{'='*60}")
    print("MEAN REVERSION STRATEGY BACKTEST")
    print(f"{'='*60}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Universe: {len(symbols)} symbols (testing {args.cap})")
    # print(f"Params:")
    # print(f"  RSI(2) entry: <= {params.rsi_entry}")
    # print(f"  IBS entry: <= {params.ibs_entry}")
    # print(f"  Down days: >= {params.down_days}")
    # print(f"  Require multiple: {params.require_multiple}")
    print(f"{'='*60}\n")

    # Run backtest
    results = run_backtest(
        symbols=symbols,
        start=args.start,
        end=args.end,
        params=None, # Changed to None
        max_symbols=args.cap,
    )

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    if 'error' in results:
        print(f"Error: {results['error']}")
        return 1

    print(f"Signals Generated: {results['signals']}")
    print(f"Trades Executed:   {results['trades']}")
    print(f"Signals/Day:       {results['signals_per_day']}")
    print(f"\nPerformance:")
    print(f"  Win Rate:        {results['win_rate']}%")
    print(f"  Wins/Losses:     {results['wins']}/{results['losses']}")
    print(f"  Profit Factor:   {results['profit_factor']}")
    print(f"  Total R:         {results['total_r']}R")
    print(f"  Avg R/Trade:     {results['avg_r']}R")
    print(f"  Avg Bars Held:   {results['avg_bars_held']}")

    print(f"\nExit Reasons:")
    for reason, count in results.get('exit_reasons', {}).items():
        print(f"  {reason}: {count}")

    print(f"\nSignal Types:")
    for sig_type, count in results.get('signal_types', {}).items():
        print(f"  {sig_type}: {count}")

    # Success criteria
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*60}")
    # wr_pass = results['win_rate'] >= 58
    # spd_pass = results['signals_per_day'] >= 1.0
    # pf_pass = results['profit_factor'] >= 1.3

    # print(f"Win Rate >= 58%:       {'PASS' if wr_pass else 'FAIL'} ({results['win_rate']}%)")
    # print(f"Signals/Day >= 1.0:    {'PASS' if spd_pass else 'FAIL'} ({results['signals_per_day']})")
    # print(f"Profit Factor >= 1.3:  {'PASS' if pf_pass else 'FAIL'} ({results['profit_factor']})")

    # if wr_pass and spd_pass and pf_pass:
    #     print("\n*** ALL CRITERIA PASSED ***")
    # else:
    #     print("\n*** CRITERIA NOT MET - OPTIMIZATION NEEDED ***")
    print("Skipping success criteria check for unimplemented strategy.")
    return 1 # Indicate error/skip
