#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path as _P
from datetime import date

import pandas as pd

import sys
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
from strategies.ibs.strategy import IBSStrategy
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.providers.multi_source import fetch_daily_bars_multi
from backtest.walk_forward import generate_splits, run_walk_forward, summarize_results
from backtest.engine import BacktestConfig
from config.env_loader import load_env


def main():
    ap = argparse.ArgumentParser(description='Walk-forward backtest (Polygon) with RSI-2, IBS, and AND')
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--start', type=str, required=True, help='YYYY-MM-DD')
    ap.add_argument('--end', type=str, required=True, help='YYYY-MM-DD')
    ap.add_argument('--train-days', type=int, default=252)
    ap.add_argument('--test-days', type=int, default=63)
    ap.add_argument('--anchored', action='store_true', default=False)
    ap.add_argument('--cap', type=int, default=950)
    ap.add_argument('--outdir', type=str, default='wf_outputs')
    ap.add_argument('--cache', type=str, default='data/cache')
    ap.add_argument('--fallback-free', action='store_true', default=False, help='Backfill pre-Polygon coverage with Yahoo Finance')
    ap.add_argument('--dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
    args = ap.parse_args()

    universe = _P(args.universe)
    dotenv = _P(args.dotenv)
    if dotenv.exists():
        _loaded = load_env(dotenv)
        print('Loaded %d env vars from %s' % (len(_loaded), dotenv))
    symbols = load_universe(universe, cap=args.cap)

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    splits = generate_splits(start_date, end_date, train_days=args.train_days, test_days=args.test_days, anchored=args.anchored)

    cache_dir = _P(args.cache)
    outdir = _P(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def fetcher(sym: str, start_s: str, end_s: str) -> pd.DataFrame:
        if args.fallback_free:
            return fetch_daily_bars_multi(sym, start_s, end_s, cache_dir=cache_dir)
        return fetch_daily_bars_polygon(sym, start_s, end_s, cache_dir=cache_dir)

    # Strategies
    rsi2 = ConnorsRSI2Strategy()
    ibs = IBSStrategy()

    def get_rsi2(df: pd.DataFrame) -> pd.DataFrame:
        return rsi2.scan_signals_over_time(df)

    def get_ibs(df: pd.DataFrame) -> pd.DataFrame:
        return ibs.scan_signals_over_time(df)

    def get_and(df: pd.DataFrame) -> pd.DataFrame:
        a = rsi2.scan_signals_over_time(df)
        b = ibs.scan_signals_over_time(df)
        if a.empty or b.empty:
            return pd.DataFrame(columns=a.columns if not a.empty else b.columns)
        merged = pd.merge(a, b, on=['timestamp','symbol','side'], suffixes=('_rsi2','_ibs'))
        if merged.empty:
            return pd.DataFrame(columns=['timestamp','symbol','side','entry_price','stop_loss','take_profit','reason'])
        out = merged[['timestamp','symbol','side']].copy()
        out['entry_price'] = merged['entry_price_rsi2']
        out['stop_loss'] = merged['stop_loss_rsi2']
        out['take_profit'] = merged.get('take_profit_rsi2', pd.NA)
        out['reason'] = 'RSI2+IBS AND'
        if 'rsi2' in merged.columns: out['rsi2'] = merged['rsi2']
        if 'ibs' in merged.columns: out['ibs'] = merged['ibs']
        return out

    # Run WF per strategy
    rsi2_results = run_walk_forward(symbols, fetcher, get_rsi2, splits, outdir=str(outdir / 'rsi2'))
    ibs_results = run_walk_forward(symbols, fetcher, get_ibs, splits, outdir=str(outdir / 'ibs'))
    and_results = run_walk_forward(symbols, fetcher, get_and, splits, outdir=str(outdir / 'and'))

    # Summaries
    rsi2_summary = summarize_results(rsi2_results)
    ibs_summary = summarize_results(ibs_results)
    and_summary = summarize_results(and_results)

    # Combined side-by-side CSV
    rows = []
    rows.append({'strategy': 'RSI2', **rsi2_summary})
    rows.append({'strategy': 'IBS', **ibs_summary})
    rows.append({'strategy': 'AND', **and_summary})
    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(outdir / 'wf_summary_compare.csv', index=False)

    # Also write detailed split metrics
    pd.DataFrame(rsi2_results).to_csv(outdir / 'rsi2' / 'wf_splits.csv', index=False)
    pd.DataFrame(ibs_results).to_csv(outdir / 'ibs' / 'wf_splits.csv', index=False)
    pd.DataFrame(and_results).to_csv(outdir / 'and' / 'wf_splits.csv', index=False)

    print('Walk-forward complete. Summary:')
    print(compare_df)


if __name__ == '__main__':
    main()
