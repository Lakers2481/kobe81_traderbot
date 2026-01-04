#!/usr/bin/env python3
"""
Parameter Optimization for Kobe81 Trading Bot (IBS+RSI / Turtle Soup).

Runs a compact grid search over strategy parameters using the event-driven
backtest engine. Designed for quick, reproducible scans on small universes to
identify promising parameter regions — not a final, production calibration.

Outputs:
- <outdir>/<strategy>_grid.csv
- <outdir>/best_params.json

Strategies:
- IBS+RSI (IbsRsiStrategy): varies `ibs_max`, `rsi_max`, `atr_mult`,
  `r_multiple`, and `time_stop_bars`.
- Turtle Soup (TurtleSoupStrategy): varies `lookback`, `min_bars_since_extreme`,
  `stop_buffer_mult`, `r_multiple`, and `time_stop_bars`.
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

# Use canonical DualStrategyScanner - deprecated standalone strategies removed
from strategies.registry import get_production_scanner
from strategies.dual_strategy.combined import DualStrategyParams
# Legacy imports for parameter optimization (suppress deprecation warnings for this specific use case)
from strategies.ibs_rsi.strategy import IbsRsiStrategy, IbsRsiParams
from strategies.ict.turtle_soup import TurtleSoupStrategy, TurtleSoupParams
from backtest.engine import Backtester, BacktestConfig
from data.universe.loader import load_universe
from data.providers.multi_source import fetch_daily_bars_multi
from config.env_loader import load_env


def main():
    ap = argparse.ArgumentParser(description='Parameter optimization for IBS_RSI / ICT Turtle Soup')
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--start', type=str, required=True)
    ap.add_argument('--end', type=str, required=True)
    ap.add_argument('--cap', type=int, default=100, help='Max symbols for speed')
    ap.add_argument('--outdir', type=str, default='optimize_outputs')
    ap.add_argument('--cache', type=str, default='data/cache')
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--strategy', type=str, choices=['ibs_rsi','turtle_soup'], default='ibs_rsi')
    # IBS+RSI grid
    ap.add_argument('--ibs-max', type=str, default='0.10,0.15,0.20')
    ap.add_argument('--rsi-max', type=str, default='5,10,15')
    ap.add_argument('--atr-mults', type=str, default='0.8,1.0,1.2')
    ap.add_argument('--r-mults', type=str, default='1.5,2.0,2.5')
    ap.add_argument('--time-stops', type=str, default='5,7')
    # Turtle Soup grid
    ap.add_argument('--ict-lookbacks', type=str, default='20,30')
    ap.add_argument('--ict-min-bars', type=str, default='3,5')
    ap.add_argument('--ict-stop-bufs', type=str, default='0.5,1.0')
    ap.add_argument('--ict-time-stops', type=str, default='5,7')
    ap.add_argument('--ict-r-mults', type=str, default='2.0,3.0')
    args = ap.parse_args()

    dotenv = _P(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    symbols = load_universe(_P(args.universe), cap=args.cap)
    cache_dir = _P(args.cache)
    outdir = _P(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def fetcher(sym: str) -> pd.DataFrame:
        return fetch_daily_bars_multi(sym, args.start, args.end, cache_dir=cache_dir)

    results: List[Dict[str, Any]] = []
    if args.strategy == 'ibs_rsi':
        ibs_maxs = [float(x) for x in args.ibs_max.split(',') if x]
        rsi_maxs = [float(x) for x in args.rsi_max.split(',') if x]
        atr_mults = [float(x) for x in args.atr_mults.split(',') if x]
        r_mults = [float(x) for x in args.r_mults.split(',') if x]
        time_stops = [int(x) for x in args.time_stops.split(',') if x]
        print(f"IBS_RSI grid: ibs_max={ibs_maxs}, rsi_max={rsi_maxs}, atr_mults={atr_mults}, r_mults={r_mults}, time_stops={time_stops}")
        for ibs_m, rsi_m, atr_m, rm, ts in product(ibs_maxs, rsi_maxs, atr_mults, r_mults, time_stops):
            params = IbsRsiParams(ibs_max=ibs_m, rsi_max=rsi_m, atr_mult=atr_m, r_multiple=rm, time_stop_bars=ts)
            strat = IbsRsiStrategy(params)
            def get_signals(df: pd.DataFrame) -> pd.DataFrame:
                return strat.scan_signals_over_time(df)
            cfg = BacktestConfig(initial_cash=100_000.0)
            bt = Backtester(cfg, get_signals, fetcher)
            result = bt.run(symbols)
            m = result.get('metrics', {})
            rec = {
                'ibs_max': ibs_m, 'rsi_max': rsi_m, 'atr_mult': atr_m, 'r_multiple': rm, 'time_stop_bars': ts,
                'trades': m.get('trades', 0), 'win_rate': m.get('win_rate', 0.0),
                'profit_factor': m.get('profit_factor', 0.0), 'sharpe': m.get('sharpe', 0.0),
                'max_drawdown': m.get('max_drawdown', 0.0),
            }
            results.append(rec)
            print(f"IBS_RSI ibs={ibs_m} rsi={rsi_m} atr={atr_m} ts={ts} rm={rm} => WR={m.get('win_rate',0):.2%} PF={m.get('profit_factor',0):.2f}")
    else:
        lookbacks = [int(x) for x in args.ict_lookbacks.split(',') if x]
        min_bars = [int(x) for x in args.ict_min_bars.split(',') if x]
        stop_bufs = [float(x) for x in args.ict_stop_bufs.split(',') if x]
        time_stops = [int(x) for x in args.ict_time_stops.split(',') if x]
        r_mults = [float(x) for x in args.ict_r_mults.split(',') if x]
        print(f"TurtleSoup grid: lookbacks={lookbacks}, min_bars={min_bars}, stop_bufs={stop_bufs}, time_stops={time_stops}, r_mults={r_mults}")
        for lb, mb, sb, ts, rm in product(lookbacks, min_bars, stop_bufs, time_stops, r_mults):
            params = TurtleSoupParams(lookback=lb, min_bars_since_extreme=mb, stop_buffer_mult=sb, time_stop_bars=ts, r_multiple=rm)
            strategy = TurtleSoupStrategy(params)
            def get_signals(df: pd.DataFrame) -> pd.DataFrame:
                return strategy.scan_signals_over_time(df)
            cfg = BacktestConfig(initial_cash=100_000.0)
            bt = Backtester(cfg, get_signals, fetcher)
            result = bt.run(symbols)
            m = result.get('metrics', {})
            rec = {
                'lookback': lb, 'min_bars': mb, 'stop_buffer_mult': sb, 'time_stop_bars': ts, 'r_multiple': rm,
                'trades': m.get('trades', 0), 'win_rate': m.get('win_rate', 0.0),
                'profit_factor': m.get('profit_factor', 0.0), 'sharpe': m.get('sharpe', 0.0),
                'max_drawdown': m.get('max_drawdown', 0.0),
            }
            results.append(rec)
            print(f"TS lb={lb} mb={mb} sb={sb} ts={ts} rm={rm} => WR={m.get('win_rate',0):.2%} PF={m.get('profit_factor',0):.2f}")

    # Save results
    df_out = pd.DataFrame(results)
    grid_file = outdir / f'{args.strategy}_grid.csv'
    df_out.to_csv(grid_file, index=False)
    # Pick best by PF then WR
    best: Dict[str, Any] = {}
    if not df_out.empty:
        s = df_out.sort_values(['profit_factor','win_rate'], ascending=[False, False]).iloc[0]
        best = s.to_dict()
    with open(outdir / 'best_params.json', 'w') as f:
        json.dump(best, f, indent=2)
    print(f'Wrote {grid_file} and best_params.json')


if __name__ == '__main__':
    main()


