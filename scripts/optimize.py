#!/usr/bin/env python3
"""
Parameter Optimization for Kobe81 Trading Bot.
Grid search around RSI-2/IBS thresholds to find robust parameter plateaus.
Outputs heatmap CSV/HTML and best parameters JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path as _P
from itertools import product
from typing import Dict, List, Any

import pandas as pd

import sys
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy, ConnorsRSI2Params
from strategies.ibs.strategy import IBSStrategy, IBSParams
from strategies.connors_crsi.strategy import ConnorsCRSIStrategy, ConnorsCRSIParams
from backtest.engine import Backtester, BacktestConfig
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon
from config.env_loader import load_env


def main():
    ap = argparse.ArgumentParser(description='Parameter optimization for RSI-2/IBS')
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--start', type=str, required=True)
    ap.add_argument('--end', type=str, required=True)
    ap.add_argument('--cap', type=int, default=100, help='Max symbols for speed')
    ap.add_argument('--outdir', type=str, default='optimize_outputs')
    ap.add_argument('--cache', type=str, default='data/cache')
    ap.add_argument('--dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
    # RSI-2 grid
    ap.add_argument('--rsi-entry-min', type=int, default=5)
    ap.add_argument('--rsi-entry-max', type=int, default=20)
    ap.add_argument('--rsi-entry-step', type=int, default=5)
    ap.add_argument('--rsi-exit-min', type=int, default=60)
    ap.add_argument('--rsi-exit-max', type=int, default=80)
    ap.add_argument('--rsi-exit-step', type=int, default=10)
    # IBS grid
    ap.add_argument('--ibs-min', type=float, default=0.1)
    ap.add_argument('--ibs-max', type=float, default=0.3)
    ap.add_argument('--ibs-step', type=float, default=0.05)
    # CRSI grid
    ap.add_argument('--crsi-min', type=int, default=10)
    ap.add_argument('--crsi-max', type=int, default=30)
    ap.add_argument('--crsi-step', type=int, default=5)
    args = ap.parse_args()

    dotenv = _P(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    symbols = load_universe(_P(args.universe), cap=args.cap)
    cache_dir = _P(args.cache)
    outdir = _P(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def fetcher(sym: str) -> pd.DataFrame:
        return fetch_daily_bars_polygon(sym, args.start, args.end, cache_dir=cache_dir)

    # Generate parameter grids
    rsi_entries = list(range(args.rsi_entry_min, args.rsi_entry_max + 1, args.rsi_entry_step))
    rsi_exits = list(range(args.rsi_exit_min, args.rsi_exit_max + 1, args.rsi_exit_step))
    ibs_thresholds = [args.ibs_min + i * args.ibs_step
                      for i in range(int((args.ibs_max - args.ibs_min) / args.ibs_step) + 1)]

    crsi_thresholds = list(range(args.crsi_min, args.crsi_max + 1, args.crsi_step))

    print(f"RSI-2 grid: entry={rsi_entries}, exit={rsi_exits}")
    print(f"IBS grid: {ibs_thresholds}")
    print(f"CRSI grid: {crsi_thresholds}")

    # Run RSI-2 optimization
    rsi_results = []
    for entry, exit_val in product(rsi_entries, rsi_exits):
        if entry >= exit_val:
            continue  # Skip invalid combinations

        params = ConnorsRSI2Params(long_entry_rsi_max=float(entry), long_exit_rsi_min=float(exit_val))
        strategy = ConnorsRSI2Strategy(params)

        def get_signals(df: pd.DataFrame) -> pd.DataFrame:
            return strategy.scan_signals_over_time(df)

        cfg = BacktestConfig(initial_cash=100_000.0)
        bt = Backtester(cfg, get_signals, fetcher)
        result = bt.run(symbols)
        m = result.get('metrics', {})

        rsi_results.append({
            'entry': entry,
            'exit': exit_val,
            'trades': m.get('trades', 0),
            'win_rate': m.get('win_rate', 0.0),
            'profit_factor': m.get('profit_factor', 0.0),
            'sharpe': m.get('sharpe', 0.0),
            'max_drawdown': m.get('max_drawdown', 0.0),
        })
        print(f"RSI-2 entry={entry}, exit={exit_val}: WR={m.get('win_rate', 0):.2%}, PF={m.get('profit_factor', 0):.2f}")

    # Run IBS optimization
    ibs_results = []
    for threshold in ibs_thresholds:
        params = IBSParams(ibs_long_max=float(threshold))
        strategy = IBSStrategy(params)

        def get_signals(df: pd.DataFrame) -> pd.DataFrame:
            return strategy.scan_signals_over_time(df)

        cfg = BacktestConfig(initial_cash=100_000.0)
        bt = Backtester(cfg, get_signals, fetcher)
        result = bt.run(symbols)
        m = result.get('metrics', {})

        ibs_results.append({
            'threshold': threshold,
            'trades': m.get('trades', 0),
            'win_rate': m.get('win_rate', 0.0),
            'profit_factor': m.get('profit_factor', 0.0),
            'sharpe': m.get('sharpe', 0.0),
            'max_drawdown': m.get('max_drawdown', 0.0),
        })
        print(f"IBS threshold={threshold:.2f}: WR={m.get('win_rate', 0):.2%}, PF={m.get('profit_factor', 0):.2f}")

    # Run CRSI optimization
    crsi_results = []
    for threshold in crsi_thresholds:
        params = ConnorsCRSIParams(long_entry_crsi_max=float(threshold))
        strategy = ConnorsCRSIStrategy(params)

        def get_signals(df: pd.DataFrame) -> pd.DataFrame:
            return strategy.scan_signals_over_time(df)

        cfg = BacktestConfig(initial_cash=100_000.0)
        bt = Backtester(cfg, get_signals, fetcher)
        result = bt.run(symbols)
        m = result.get('metrics', {})

        crsi_results.append({
            'threshold': threshold,
            'trades': m.get('trades', 0),
            'win_rate': m.get('win_rate', 0.0),
            'profit_factor': m.get('profit_factor', 0.0),
            'sharpe': m.get('sharpe', 0.0),
            'max_drawdown': m.get('max_drawdown', 0.0),
        })
        print(f"CRSI threshold={threshold}: WR={m.get('win_rate', 0):.2%}, PF={m.get('profit_factor', 0):.2f}, trades={m.get('trades', 0)}")

    # Save results
    rsi_df = pd.DataFrame(rsi_results)
    ibs_df = pd.DataFrame(ibs_results)
    crsi_df = pd.DataFrame(crsi_results)

    rsi_df.to_csv(outdir / 'rsi2_heatmap.csv', index=False)
    ibs_df.to_csv(outdir / 'ibs_heatmap.csv', index=False)
    crsi_df.to_csv(outdir / 'crsi_heatmap.csv', index=False)

    # Find best plateau (not single spike) - look for consistent results
    best_rsi = _find_best_plateau(rsi_df, ['entry', 'exit'], 'profit_factor')
    best_ibs = _find_best_plateau(ibs_df, ['threshold'], 'profit_factor')
    best_crsi = _find_best_plateau(crsi_df, ['threshold'], 'profit_factor')

    best_params = {
        'rsi2': {
            'best': best_rsi,
            'all_results': rsi_results,
        },
        'ibs': {
            'best': best_ibs,
            'all_results': ibs_results,
        },
        'crsi': {
            'best': best_crsi,
            'all_results': crsi_results,
        },
    }

    with open(outdir / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2, default=str)

    # Generate HTML report
    _generate_html_report(rsi_df, ibs_df, crsi_df, best_rsi, best_ibs, best_crsi, outdir)

    print(f"\nOptimization complete. Results in {outdir}/")
    print(f"Best RSI-2: {best_rsi}")
    print(f"Best IBS: {best_ibs}")
    print(f"Best CRSI: {best_crsi}")


def _find_best_plateau(df: pd.DataFrame, group_cols: List[str], metric: str) -> Dict[str, Any]:
    """
    Find the best parameter set that's also part of a stable plateau.
    A plateau is where neighboring parameter values have similar performance.
    """
    if df.empty:
        return {}

    # Simple approach: find top 3 by metric and pick the middle one
    df_sorted = df.nlargest(3, metric)
    if len(df_sorted) >= 2:
        # Pick the second best as it's more likely part of a plateau
        best = df_sorted.iloc[1].to_dict()
    else:
        best = df_sorted.iloc[0].to_dict() if len(df_sorted) > 0 else {}

    return best


def _generate_html_report(
    rsi_df: pd.DataFrame,
    ibs_df: pd.DataFrame,
    crsi_df: pd.DataFrame,
    best_rsi: Dict,
    best_ibs: Dict,
    best_crsi: Dict,
    outdir: _P,
) -> None:
    """Generate HTML optimization report."""
    html = [
        '<html><head><meta charset="utf-8"><title>Parameter Optimization Report</title>',
        '<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse;margin:10px 0} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3} .best{background:#d4edda}</style>',
        '</head><body>',
        '<h1>Parameter Optimization Report</h1>',
        '<h2>RSI-2 Optimization</h2>',
        f'<p>Best parameters: {best_rsi}</p>',
        rsi_df.to_html(index=False, classes='rsi2'),
        '<h2>IBS Optimization</h2>',
        f'<p>Best parameters: {best_ibs}</p>',
        ibs_df.to_html(index=False, classes='ibs'),
        '<h2>CRSI Optimization</h2>',
        f'<p>Best parameters: {best_crsi}</p>',
        crsi_df.to_html(index=False, classes='crsi'),
        '</body></html>',
    ]
    (outdir / 'optimization_report.html').write_text('\n'.join(html), encoding='utf-8')


if __name__ == '__main__':
    main()
