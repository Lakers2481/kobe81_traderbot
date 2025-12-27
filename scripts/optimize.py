#!/usr/bin/env python3
"""
Parameter Optimization for Kobe81 Trading Bot (Donchian / ICT).

Runs a small grid search over key parameters for:
- Donchian breakout: lookback, stop_mult, time_stop_bars, r_multiple
- ICT Turtle Soup: lookback, min_bars_since_extreme, time_stop_bars, r_multiple

Outputs CSV and best_params.json.
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

from strategies.donchian.strategy import DonchianBreakoutStrategy, DonchianParams
from strategies.ict.turtle_soup import TurtleSoupStrategy, TurtleSoupParams
from backtest.engine import Backtester, BacktestConfig
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon
from config.env_loader import load_env


def main():
    ap = argparse.ArgumentParser(description='Parameter optimization for Donchian / ICT Turtle Soup')
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--start', type=str, required=True)
    ap.add_argument('--end', type=str, required=True)
    ap.add_argument('--cap', type=int, default=100, help='Max symbols for speed')
    ap.add_argument('--outdir', type=str, default='optimize_outputs')
    ap.add_argument('--cache', type=str, default='data/cache')
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--strategy', type=str, choices=['donchian','turtle_soup','ict'], default='donchian')
    # Donchian grid
    ap.add_argument('--donchian-lookbacks', type=str, default='20,55')
    ap.add_argument('--donchian-stop-mults', type=str, default='1.5,2.0,2.5')
    ap.add_argument('--donchian-time-stops', type=str, default='10,20,30')
    ap.add_argument('--donchian-r-mults', type=str, default='2.0,2.5,3.0')
    # ICT grid
    ap.add_argument('--ict-lookbacks', type=str, default='20,30')
    ap.add_argument('--ict-min-bars', type=str, default='3,5')
    ap.add_argument('--ict-time-stops', type=str, default='5,7')
    ap.add_argument('--ict-r-mults', type=str, default='2.0,2.5')
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

    results: List[Dict[str, Any]] = []
    if args.strategy == 'donchian':
        lookbacks = [int(x) for x in args.donchian_lookbacks.split(',') if x]
        stop_mults = [float(x) for x in args.donchian_stop_mults.split(',') if x]
        time_stops = [int(x) for x in args.donchian_time_stops.split(',') if x]
        r_mults = [float(x) for x in args.donchian_r_mults.split(',') if x]
        print(f"Donchian grid: lookbacks={lookbacks}, stop_mults={stop_mults}, time_stops={time_stops}, r_mults={r_mults}")
        for lb, sm, ts, rm in product(lookbacks, stop_mults, time_stops, r_mults):
            params = DonchianParams(lookback=lb, stop_mult=sm, time_stop_bars=ts, r_multiple=rm)
            strategy = DonchianBreakoutStrategy(params)
            def get_signals(df: pd.DataFrame) -> pd.DataFrame:
                return strategy.scan_signals_over_time(df)
            cfg = BacktestConfig(initial_cash=100_000.0)
            bt = Backtester(cfg, get_signals, fetcher)
            result = bt.run(symbols)
            m = result.get('metrics', {})
            rec = {
                'lookback': lb, 'stop_mult': sm, 'time_stop_bars': ts, 'r_multiple': rm,
                'trades': m.get('trades', 0), 'win_rate': m.get('win_rate', 0.0),
                'profit_factor': m.get('profit_factor', 0.0), 'sharpe': m.get('sharpe', 0.0),
                'max_drawdown': m.get('max_drawdown', 0.0),
            }
            results.append(rec)
            print(f"DON lb={lb} sm={sm} ts={ts} rm={rm} => WR={m.get('win_rate',0):.2%} PF={m.get('profit_factor',0):.2f}")
    else:
        lookbacks = [int(x) for x in args.ict_lookbacks.split(',') if x]
        min_bars = [int(x) for x in args.ict_min_bars.split(',') if x]
        time_stops = [int(x) for x in args.ict_time_stops.split(',') if x]
        r_mults = [float(x) for x in args.ict_r_mults.split(',') if x]
        print(f"ICT grid: lookbacks={lookbacks}, min_bars={min_bars}, time_stops={time_stops}, r_mults={r_mults}")
        for lb, mb, ts, rm in product(lookbacks, min_bars, time_stops, r_mults):
            params = TurtleSoupParams(lookback=lb, min_bars_since_extreme=mb, time_stop_bars=ts, r_multiple=rm)
            strategy = TurtleSoupStrategy(params)
            def get_signals(df: pd.DataFrame) -> pd.DataFrame:
                return strategy.scan_signals_over_time(df)
            cfg = BacktestConfig(initial_cash=100_000.0)
            bt = Backtester(cfg, get_signals, fetcher)
            result = bt.run(symbols)
            m = result.get('metrics', {})
            rec = {
                'lookback': lb, 'min_bars': mb, 'time_stop_bars': ts, 'r_multiple': rm,
                'trades': m.get('trades', 0), 'win_rate': m.get('win_rate', 0.0),
                'profit_factor': m.get('profit_factor', 0.0), 'sharpe': m.get('sharpe', 0.0),
                'max_drawdown': m.get('max_drawdown', 0.0),
            }
            results.append(rec)
            print(f"ICT lb={lb} mb={mb} ts={ts} rm={rm} => WR={m.get('win_rate',0):.2%} PF={m.get('profit_factor',0):.2f}")

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

