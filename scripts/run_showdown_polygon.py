#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path as _P

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from strategies.ibs_rsi.strategy import IbsRsiStrategy
from strategies.ict.turtle_soup import TurtleSoupStrategy
from backtest.engine import Backtester, BacktestConfig
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon
from config.env_loader import load_env
from config.settings_loader import (
    get_regime_filter_config, is_regime_filter_enabled,
    get_selection_config, is_selection_enabled,
    is_earnings_filter_enabled,
)
from core.regime_filter import filter_signals_by_regime, fetch_spy_bars
from core.earnings_filter import filter_signals_by_earnings


def main():
    ap = argparse.ArgumentParser(description='Showdown backtest: IBS+RSI vs ICT Turtle Soup over full period')
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--start', type=str, required=True)
    ap.add_argument('--end', type=str, required=True)
    ap.add_argument('--cap', type=int, default=900)
    ap.add_argument('--outdir', type=str, default='showdown_outputs')
    ap.add_argument('--cache', type=str, default='data/cache')
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--regime-on', action='store_true', default=False, help='Force regime filter ON')
    ap.add_argument('--topn-on', action='store_true', default=False, help='Force top-N selection ON')
    args = ap.parse_args()

    dotenv = _P(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)
    symbols = load_universe(_P(args.universe), cap=args.cap)
    cache_dir = _P(args.cache)
    outdir = _P(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Check config-gated features
    regime_cfg = get_regime_filter_config()
    selection_cfg = get_selection_config()
    use_regime = args.regime_on or regime_cfg.get('enabled', False)
    use_topn = args.topn_on or selection_cfg.get('enabled', False)

    # Load SPY bars for regime filter if enabled
    spy_bars = pd.DataFrame()
    if use_regime:
        print('Regime filter enabled - loading SPY bars...')
        spy_bars = fetch_spy_bars(args.start, args.end, cache_dir=cache_dir)
        print(f'Loaded {len(spy_bars)} SPY bars for regime filtering')

    # Fetcher over full window
    def fetcher(sym: str) -> pd.DataFrame:
        return fetch_daily_bars_polygon(sym, args.start, args.end, cache_dir=cache_dir)

    # Strategies
    ibs = IbsRsiStrategy()
    ict = TurtleSoupStrategy()

    def apply_regime_filter(signals: pd.DataFrame) -> pd.DataFrame:
        if signals.empty:
            return signals
        if use_regime and not spy_bars.empty:
            signals = filter_signals_by_regime(signals, spy_bars, regime_cfg)
        return signals

    def apply_earnings_filter(signals: pd.DataFrame) -> pd.DataFrame:
        if signals.empty:
            return signals
        if is_earnings_filter_enabled():
            recs = signals.to_dict('records')
            filtered = filter_signals_by_earnings(recs, date_key='timestamp', symbol_key='symbol')
            return pd.DataFrame(filtered)
        return signals

    # No TOPN cross-sectional ranking in showdown when only comparing two distinct strategies

    def get_ibs_rsi(df: pd.DataFrame) -> pd.DataFrame:
        signals = ibs.generate_signals(df)
        signals = apply_regime_filter(signals)
        signals = apply_earnings_filter(signals)
        return signals

    def get_turtle_soup(df: pd.DataFrame) -> pd.DataFrame:
        signals = ict.scan_signals_over_time(df)
        signals = apply_regime_filter(signals)
        signals = apply_earnings_filter(signals)
        return signals

    # Run per strategy
    cfg = BacktestConfig(initial_cash=100_000.0)
    bt_ibs = Backtester(cfg, get_ibs_rsi, fetcher)
    r1 = bt_ibs.run(symbols, outdir=str(outdir / 'ibs_rsi'))
    bt_ict = Backtester(cfg, get_turtle_soup, fetcher)
    r2 = bt_ict.run(symbols, outdir=str(outdir / 'turtle_soup'))

    # Combined summary
    rows = []
    for name, res in (('IBS_RSI', r1), ('TURTLE_SOUP', r2)):
        m = res.get('metrics', {})
        rows.append({
            'strategy': name,
            'trades': m.get('trades', 0),
            'win_rate': m.get('win_rate', 0.0),
            'profit_factor': m.get('profit_factor', 0.0),
            'sharpe': m.get('sharpe', 0.0),
            'max_drawdown': m.get('max_drawdown', 0.0),
            'final_equity': m.get('final_equity', 0.0),
            'gross_pnl': m.get('gross_pnl', 0.0),
            'net_pnl': m.get('net_pnl', 0.0),
            'total_fees': m.get('total_fees', 0.0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / 'showdown_summary.csv', index=False)

    # Simple HTML
    html = ['<html><head><meta charset="utf-8"><title>Showdown Report</title>',
            '<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3}</style>',
            '</head><body><h1>Showdown Summary (IBS+RSI vs ICT Turtle Soup)</h1>', df.to_html(index=False), '</body></html>']
    (outdir / 'showdown_report.html').write_text('\n'.join(html), encoding='utf-8')
    print('\nShowdown complete. Summary (IBS+RSI vs ICT Turtle Soup):')
    print(df)


if __name__ == '__main__':
    main()
