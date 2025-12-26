#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path as _P

import pandas as pd

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
from strategies.ibs.strategy import IBSStrategy
from backtest.engine import Backtester, BacktestConfig
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon
from config.env_loader import load_env


def main():
    ap = argparse.ArgumentParser(description='Run backtest on Polygon data')
    ap.add_argument('--universe', type=str, required=True, help='CSV with symbol column')
    ap.add_argument('--start', type=str, required=True, help='YYYY-MM-DD')
    ap.add_argument('--end', type=str, required=True, help='YYYY-MM-DD')
    ap.add_argument('--strategy', type=str, default='rsi2', choices=['rsi2','ibs','and'])
    ap.add_argument('--cache', type=str, default='data/cache')
    ap.add_argument('--cap', type=int, default=950, help='Max symbols to use')
    ap.add_argument('--dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
    args = ap.parse_args()

    uni = _P(args.universe)
    dotenv_path = _P(args.dotenv)
    if dotenv_path.exists():
        _loaded = load_env(dotenv_path)
        print('Loaded %d env vars from %s' % (len(_loaded), dotenv_path))

    symbols = load_universe(uni, cap=args.cap)

    cache_dir = _P(args.cache)

    # Strategy setup
    rsi2 = ConnorsRSI2Strategy()
    ibs = IBSStrategy()

    def fetcher(sym: str) -> pd.DataFrame:
        return fetch_daily_bars_polygon(sym, args.start, args.end, cache_dir=cache_dir)

    # Select get_signals function
    if args.strategy == 'rsi2':
        def get_signals(df: pd.DataFrame) -> pd.DataFrame:
            return rsi2.scan_signals_over_time(df)
    elif args.strategy == 'ibs':
        def get_signals(df: pd.DataFrame) -> pd.DataFrame:
            return ibs.scan_signals_over_time(df)
    else:  # 'and' => combine filters: require both conditions same bar
        def get_signals(df: pd.DataFrame) -> pd.DataFrame:
            a = rsi2.scan_signals_over_time(df)
            b = ibs.scan_signals_over_time(df)
            if a.empty or b.empty:
                return pd.DataFrame(columns=a.columns if not a.empty else b.columns)
            merged = pd.merge(a, b, on=['timestamp','symbol','side'], suffixes=('_rsi2','_ibs'))
            if merged.empty:
                return merged
            out = merged[['timestamp','symbol','side']].copy()
            out['entry_price'] = merged['entry_price_rsi2']
            out['stop_loss'] = merged['stop_loss_rsi2']
            out['take_profit'] = merged.get('take_profit_rsi2', pd.NA)
            out['reason'] = 'RSI2+IBS AND'
            if 'rsi2' in merged.columns: out['rsi2'] = merged['rsi2']
            if 'ibs' in merged.columns: out['ibs'] = merged['ibs']
            return out

    cfg = BacktestConfig(initial_cash=100_000.0)
    bt = Backtester(cfg, get_signals, fetcher)
    res = bt.run(symbols, outdir='outputs')
    m = res.get('metrics', {})
    print('Symbols: %d | Trades: %d | PnL: %.2f | WR: %.2f | PF: %.2f | Sharpe: %.2f | MaxDD: %.2f' % (
        len(symbols), len(res['trades']), res.get('pnl', 0.0), m.get('win_rate', 0.0), m.get('profit_factor', 0.0), m.get('sharpe', 0.0), m.get('max_drawdown', 0.0)
    ))

if __name__ == '__main__':
    main()

