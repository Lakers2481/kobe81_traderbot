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

from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.universe.loader import load_universe
from strategies.mean_reversion import MeanReversionStrategy, MeanReversionParams


def run_backtest(
    symbols: List[str],
    start: str,
    end: str,
    params: MeanReversionParams,
    capital: float = 100_000,
    risk_per_trade: float = 0.01,
    max_symbols: int = 50,
) -> Dict:
    """Run backtest on mean reversion strategy.

    Returns dict with metrics.
    """
    strategy = MeanReversionStrategy(params)
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
    signals = strategy.scan_signals_over_time(combined)
    print(f"Generated {len(signals)} signals")

    if signals.empty:
        return {
            'signals': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_r': 0,
            'signals_per_day': 0,
        }

    # Run backtest
    print("Running backtest...")
    trades = []
    for _, sig in signals.iterrows():
        sym = sig['symbol']
        sym_data = combined[combined['symbol'] == sym].sort_values('timestamp')

        # Find entry bar
        entry_ts = sig['timestamp']
        mask = sym_data['timestamp'] > entry_ts
        if not mask.any():
            continue

        entry_idx = mask.idxmax()
        entry_row = sym_data.loc[entry_idx]
        entry_price = float(entry_row['open'])  # Fill at next bar open

        stop = sig['stop_loss']
        tp = sig['take_profit']
        time_stop_bars = sig.get('time_stop_bars', 5)

        # Find exit
        exit_price = None
        exit_reason = None
        bars_held = 0

        future_bars = sym_data.loc[entry_idx:].iloc[1:]  # Bars after entry
        for i, (_, bar) in enumerate(future_bars.iterrows()):
            bars_held += 1

            # Check stop loss
            if stop and float(bar['low']) <= stop:
                exit_price = stop
                exit_reason = 'stop_loss'
                break

            # Check take profit
            if tp and float(bar['high']) >= tp:
                exit_price = tp
                exit_reason = 'take_profit'
                break

            # Check RSI exit
            rsi = bar.get('rsi2') if 'rsi2' in bar.index else None
            if rsi and float(rsi) >= params.rsi_exit:
                exit_price = float(bar['close'])
                exit_reason = 'rsi_exit'
                break

            # Check IBS exit
            ibs_val = bar.get('ibs') if 'ibs' in bar.index else None
            if ibs_val and float(ibs_val) >= params.ibs_exit:
                exit_price = float(bar['close'])
                exit_reason = 'ibs_exit'
                break

            # Check time stop
            if bars_held >= time_stop_bars:
                exit_price = float(bar['close'])
                exit_reason = 'time_stop'
                break

        if exit_price is None:
            # Force exit at last bar
            if len(future_bars) > 0:
                exit_price = float(future_bars.iloc[-1]['close'])
                exit_reason = 'end_of_data'
            else:
                continue

        # Calculate R multiple
        risk = entry_price - stop if stop else entry_price * 0.02
        pnl = exit_price - entry_price
        r_mult = pnl / risk if risk > 0 else 0

        trades.append({
            'symbol': sym,
            'entry_date': entry_ts,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'pnl': pnl,
            'r_multiple': r_mult,
            'win': pnl > 0,
            'reason': sig['reason'],
        })

    if not trades:
        return {
            'signals': len(signals),
            'trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_r': 0,
            'signals_per_day': 0,
        }

    trades_df = pd.DataFrame(trades)

    # Calculate metrics
    wins = trades_df['win'].sum()
    losses = len(trades_df) - wins
    win_rate = wins / len(trades_df) * 100

    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    total_r = trades_df['r_multiple'].sum()
    avg_r = trades_df['r_multiple'].mean()

    # Calculate signals per trading day
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    trading_days = (end_dt - start_dt).days * 252 / 365  # Approx trading days
    signals_per_day = len(signals) / trading_days if trading_days > 0 else 0

    # Exit reason breakdown
    exit_reasons = trades_df['exit_reason'].value_counts().to_dict()

    # Signal type breakdown
    signal_types = trades_df['reason'].value_counts().to_dict()

    return {
        'signals': len(signals),
        'trades': len(trades_df),
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2),
        'total_r': round(total_r, 2),
        'avg_r': round(avg_r, 3),
        'avg_bars_held': round(trades_df['bars_held'].mean(), 1),
        'signals_per_day': round(signals_per_day, 2),
        'exit_reasons': exit_reasons,
        'signal_types': signal_types,
    }


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
    params = MeanReversionParams(
        rsi_entry=args.rsi_entry,
        ibs_entry=args.ibs_entry,
        down_days=args.down_days,
        require_multiple=args.require_multiple,
    )

    print(f"\n{'='*60}")
    print("MEAN REVERSION STRATEGY BACKTEST")
    print(f"{'='*60}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Universe: {len(symbols)} symbols (testing {args.cap})")
    print(f"Params:")
    print(f"  RSI(2) entry: <= {params.rsi_entry}")
    print(f"  IBS entry: <= {params.ibs_entry}")
    print(f"  Down days: >= {params.down_days}")
    print(f"  Require multiple: {params.require_multiple}")
    print(f"{'='*60}\n")

    # Run backtest
    results = run_backtest(
        symbols=symbols,
        start=args.start,
        end=args.end,
        params=params,
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
    wr_pass = results['win_rate'] >= 58
    spd_pass = results['signals_per_day'] >= 1.0
    pf_pass = results['profit_factor'] >= 1.3

    print(f"Win Rate >= 58%:       {'PASS' if wr_pass else 'FAIL'} ({results['win_rate']}%)")
    print(f"Signals/Day >= 1.0:    {'PASS' if spd_pass else 'FAIL'} ({results['signals_per_day']})")
    print(f"Profit Factor >= 1.3:  {'PASS' if pf_pass else 'FAIL'} ({results['profit_factor']})")

    if wr_pass and spd_pass and pf_pass:
        print("\n*** ALL CRITERIA PASSED ***")
    else:
        print("\n*** CRITERIA NOT MET - OPTIMIZATION NEEDED ***")

    return 0


if __name__ == '__main__':
    sys.exit(main())
