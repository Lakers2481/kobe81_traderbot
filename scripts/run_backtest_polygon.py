#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path as _P

import pandas as pd

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategies.registry import get_production_scanner
from backtest.engine import Backtester, BacktestConfig
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon
from config.env_loader import load_env


def main():
    ap = argparse.ArgumentParser(description='Run backtest on Polygon data')
    ap.add_argument('--universe', type=str, required=True, help='CSV with symbol column')
    ap.add_argument('--start', type=str, required=True, help='YYYY-MM-DD')
    ap.add_argument('--end', type=str, required=True, help='YYYY-MM-DD')
    # Deprecated: --strategy flag removed. Always uses DualStrategyScanner (IBS+RSI + Turtle Soup combined)
    ap.add_argument('--cache', type=str, default='data/cache')
    ap.add_argument('--cap', type=int, default=900, help='Max symbols to use')
    ap.add_argument('--dotenv', type=str, default='./.env')
    args = ap.parse_args()

    uni = _P(args.universe)
    dotenv_path = _P(args.dotenv)
    if dotenv_path.exists():
        _loaded = load_env(dotenv_path)
        print('Loaded %d env vars from %s' % (len(_loaded), dotenv_path))

    symbols = load_universe(uni, cap=args.cap)

    cache_dir = _P(args.cache)

    # Use canonical DualStrategyScanner (combines IBS+RSI and Turtle Soup with 0.3 ATR filter)
    scanner = get_production_scanner()

    def fetcher(sym: str) -> pd.DataFrame:
        return fetch_daily_bars_polygon(sym, args.start, args.end, cache_dir=cache_dir)

    def get_signals(df: pd.DataFrame) -> pd.DataFrame:
        return scanner.scan_signals_over_time(df)

    cfg = BacktestConfig(initial_cash=100_000.0)
    bt = Backtester(cfg, get_signals, fetcher)
    res = bt.run(symbols, outdir='outputs')
    m = res.get('metrics', {})
    print('Symbols: %d | Trades: %d | PnL: %.2f | WR: %.2f | PF: %.2f | Sharpe: %.2f | MaxDD: %.2f' % (
        len(symbols), len(res['trades']), res.get('pnl', 0.0), m.get('win_rate', 0.0), m.get('profit_factor', 0.0), m.get('sharpe', 0.0), m.get('max_drawdown', 0.0)
    ))

if __name__ == '__main__':
    main()
