#!/usr/bin/env python3
"""
Walk-forward backtest for crypto pairs using Polygon hourly data.
Same strategy logic (RSI-2, IBS, AND) adapted for 1-hour bars.
"""
from __future__ import annotations

import argparse
from pathlib import Path as _P
from datetime import date

import pandas as pd

import sys
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
from strategies.ibs.strategy import IBSStrategy
from data.universe.loader import load_universe
from data.providers.polygon_crypto import fetch_crypto_bars
from backtest.walk_forward import generate_splits, run_walk_forward, summarize_results
from config.env_loader import load_env


def main():
    ap = argparse.ArgumentParser(description='Walk-forward backtest (Crypto) with RSI-2, IBS, and AND')
    ap.add_argument('--universe', type=str, required=True, help='Path to crypto universe CSV')
    ap.add_argument('--start', type=str, required=True, help='YYYY-MM-DD')
    ap.add_argument('--end', type=str, required=True, help='YYYY-MM-DD')
    ap.add_argument('--train-days', type=int, default=252, help='Training window in days')
    ap.add_argument('--test-days', type=int, default=63, help='Test window in days')
    ap.add_argument('--anchored', action='store_true', default=False)
    ap.add_argument('--outdir', type=str, default='wf_outputs_crypto')
    ap.add_argument('--cache', type=str, default='data/cache/crypto')
    ap.add_argument('--timeframe', type=str, default='1h', help='Bar timeframe (1h, 4h, etc)')
    ap.add_argument('--dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
    args = ap.parse_args()

    universe = _P(args.universe)
    dotenv = _P(args.dotenv)
    if dotenv.exists():
        _loaded = load_env(dotenv)
        print('Loaded %d env vars from %s' % (len(_loaded), dotenv))

    symbols = load_universe(universe)
    print(f'Loaded {len(symbols)} crypto symbols')

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    splits = generate_splits(start_date, end_date, train_days=args.train_days, test_days=args.test_days, anchored=args.anchored)

    cache_dir = _P(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    outdir = _P(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def fetcher(sym: str, start_s: str, end_s: str) -> pd.DataFrame:
        df = fetch_crypto_bars(sym, start_s, end_s, timeframe=args.timeframe, cache_dir=cache_dir)
        # Convert to tz-naive timestamp for consistency with backtest engine
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        return df

    # Strategies (same as equities)
    rsi2 = ConnorsRSI2Strategy()
    ibs = IBSStrategy()

    def get_rsi2(df: pd.DataFrame) -> pd.DataFrame:
        return rsi2.scan_signals_over_time(df)

    def get_ibs(df: pd.DataFrame) -> pd.DataFrame:
        # IBS with zero-division guard
        return _safe_ibs_scan(ibs, df)

    def get_and(df: pd.DataFrame) -> pd.DataFrame:
        a = rsi2.scan_signals_over_time(df)
        b = _safe_ibs_scan(ibs, df)
        if a.empty or b.empty:
            return pd.DataFrame(columns=a.columns if not a.empty else b.columns)
        merged = pd.merge(a, b, on=['timestamp', 'symbol', 'side'], suffixes=('_rsi2', '_ibs'))
        if merged.empty:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'entry_price', 'stop_loss', 'take_profit', 'reason'])
        out = merged[['timestamp', 'symbol', 'side']].copy()
        out['entry_price'] = merged['entry_price_rsi2']
        out['stop_loss'] = merged['stop_loss_rsi2']
        out['take_profit'] = merged.get('take_profit_rsi2', pd.NA)
        out['reason'] = 'RSI2+IBS AND (Crypto)'
        if 'rsi2' in merged.columns:
            out['rsi2'] = merged['rsi2']
        if 'ibs' in merged.columns:
            out['ibs'] = merged['ibs']
        return out

    # Run WF per strategy
    print('Running RSI-2...')
    rsi2_results = run_walk_forward(symbols, fetcher, get_rsi2, splits, outdir=str(outdir / 'rsi2'))
    print('Running IBS...')
    ibs_results = run_walk_forward(symbols, fetcher, get_ibs, splits, outdir=str(outdir / 'ibs'))
    print('Running AND...')
    and_results = run_walk_forward(symbols, fetcher, get_and, splits, outdir=str(outdir / 'and'))

    # Summaries
    rsi2_summary = summarize_results(rsi2_results)
    ibs_summary = summarize_results(ibs_results)
    and_summary = summarize_results(and_results)

    # Combined side-by-side CSV
    rows = []
    rows.append({'strategy': 'RSI2_crypto', **rsi2_summary})
    rows.append({'strategy': 'IBS_crypto', **ibs_summary})
    rows.append({'strategy': 'AND_crypto', **and_summary})
    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(outdir / 'wf_summary_compare.csv', index=False)

    # Write detailed split metrics
    (outdir / 'rsi2').mkdir(parents=True, exist_ok=True)
    (outdir / 'ibs').mkdir(parents=True, exist_ok=True)
    (outdir / 'and').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rsi2_results).to_csv(outdir / 'rsi2' / 'wf_splits.csv', index=False)
    pd.DataFrame(ibs_results).to_csv(outdir / 'ibs' / 'wf_splits.csv', index=False)
    pd.DataFrame(and_results).to_csv(outdir / 'and' / 'wf_splits.csv', index=False)

    print('Walk-forward (crypto) complete. Summary:')
    print(compare_df)


def _safe_ibs_scan(ibs_strategy, df: pd.DataFrame) -> pd.DataFrame:
    """
    IBS scan with zero-division guard.
    If (high - low) == 0, IBS is undefined; exclude that bar.
    """
    if df.empty:
        return pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'entry_price', 'stop_loss', 'take_profit', 'reason'])

    # Filter out bars where high == low (flat bars, no range)
    df_filtered = df[df['high'] != df['low']].copy()
    if df_filtered.empty:
        return pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'entry_price', 'stop_loss', 'take_profit', 'reason'])

    return ibs_strategy.scan_signals_over_time(df_filtered)


if __name__ == '__main__':
    main()
