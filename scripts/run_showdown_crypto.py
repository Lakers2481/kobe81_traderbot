#!/usr/bin/env python3
"""
Showdown backtest for crypto: IBS_RSI vs ICT Turtle Soup over full period.
Uses Polygon hourly crypto data.
"""
from __future__ import annotations

import argparse
from pathlib import Path as _P

import pandas as pd

import sys
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from strategies.ibs_rsi.strategy import IbsRsiStrategy
from strategies.ict.turtle_soup import TurtleSoupStrategy
from backtest.engine import Backtester, BacktestConfig
from data.universe.loader import load_universe
from data.providers.polygon_crypto import fetch_crypto_bars
from config.env_loader import load_env


def main():
    ap = argparse.ArgumentParser(description='Showdown backtest: IBS_RSI vs ICT over full period (Crypto)')
    ap.add_argument('--universe', type=str, required=True, help='Path to crypto universe CSV')
    ap.add_argument('--start', type=str, required=True, help='YYYY-MM-DD')
    ap.add_argument('--end', type=str, required=True, help='YYYY-MM-DD')
    ap.add_argument('--outdir', type=str, default='showdown_outputs_crypto')
    ap.add_argument('--cache', type=str, default='data/cache/crypto')
    ap.add_argument('--timeframe', type=str, default='1h', help='Bar timeframe (1h, 4h, etc)')
    ap.add_argument('--dotenv', type=str, default='./.env')
    args = ap.parse_args()

    dotenv = _P(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    symbols = load_universe(_P(args.universe))
    print(f'Loaded {len(symbols)} crypto symbols')

    cache_dir = _P(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    outdir = _P(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Fetcher over full window
    def fetcher(sym: str) -> pd.DataFrame:
        df = fetch_crypto_bars(sym, args.start, args.end, timeframe=args.timeframe, cache_dir=cache_dir)
        # Convert to tz-naive timestamp for consistency
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        return df

    # Strategies
    don = IbsRsiStrategy()
    ict = TurtleSoupStrategy()

    def get_ibs_rsi(df: pd.DataFrame) -> pd.DataFrame:
        return don.scan_signals_over_time(df)

    def get_turtle_soup(df: pd.DataFrame) -> pd.DataFrame:
        return ict.scan_signals_over_time(df)

    # Run per strategy
    print('Running IBS_RSI...')
    cfg = BacktestConfig(initial_cash=100_000.0)
    bt_don = Backtester(cfg, get_ibs_rsi, fetcher)
    r1 = bt_don.run(symbols, outdir=str(outdir / 'ibs_rsi'))

    print('Running ICT Turtle Soup...')
    bt_ict = Backtester(cfg, get_turtle_soup, fetcher)
    r2 = bt_ict.run(symbols, outdir=str(outdir / 'turtle_soup'))

    # Combined summary
    rows = []
    for name, res in (('IBS_RSI_crypto', r1), ('TURTLE_SOUP_crypto', r2)):
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
    html = [
        '<html><head><meta charset="utf-8"><title>Crypto Showdown Report</title>',
        '<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3}</style>',
        '</head><body><h1>Crypto Showdown Summary</h1>',
        f'<p>Universe: {args.universe} | Period: {args.start} to {args.end} | Timeframe: {args.timeframe}</p>',
        df.to_html(index=False),
        '</body></html>'
    ]
    (outdir / 'showdown_report.html').write_text('\n'.join(html), encoding='utf-8')
    print('Showdown (crypto) complete. Summary (IBS_RSI vs ICT):')
    print(df)


    # No IBS in this setup
    

if __name__ == '__main__':
    main()
