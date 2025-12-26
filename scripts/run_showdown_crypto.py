#!/usr/bin/env python3
"""
Showdown backtest for crypto: RSI-2 vs IBS vs AND over full period.
Uses Polygon hourly crypto data.
"""
from __future__ import annotations

import argparse
from pathlib import Path as _P

import pandas as pd

import sys
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
from strategies.ibs.strategy import IBSStrategy
from backtest.engine import Backtester, BacktestConfig
from data.universe.loader import load_universe
from data.providers.polygon_crypto import fetch_crypto_bars
from config.env_loader import load_env


def main():
    ap = argparse.ArgumentParser(description='Showdown backtest: RSI-2 vs IBS vs AND over full period (Crypto)')
    ap.add_argument('--universe', type=str, required=True, help='Path to crypto universe CSV')
    ap.add_argument('--start', type=str, required=True, help='YYYY-MM-DD')
    ap.add_argument('--end', type=str, required=True, help='YYYY-MM-DD')
    ap.add_argument('--outdir', type=str, default='showdown_outputs_crypto')
    ap.add_argument('--cache', type=str, default='data/cache/crypto')
    ap.add_argument('--timeframe', type=str, default='1h', help='Bar timeframe (1h, 4h, etc)')
    ap.add_argument('--dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
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
    rsi2 = ConnorsRSI2Strategy()
    ibs = IBSStrategy()

    def get_rsi2(df: pd.DataFrame) -> pd.DataFrame:
        return rsi2.scan_signals_over_time(df)

    def get_ibs(df: pd.DataFrame) -> pd.DataFrame:
        return _safe_ibs_scan(ibs, df)

    def get_and(df: pd.DataFrame) -> pd.DataFrame:
        a = rsi2.scan_signals_over_time(df)
        b = _safe_ibs_scan(ibs, df)
        if a.empty or b.empty:
            return pd.DataFrame(columns=a.columns if not a.empty else b.columns)
        merged = pd.merge(a, b, on=['timestamp', 'symbol', 'side'], suffixes=('_rsi2', '_ibs'))
        if merged.empty:
            return merged
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

    # Run per strategy
    print('Running RSI-2...')
    cfg = BacktestConfig(initial_cash=100_000.0)
    bt_rsi2 = Backtester(cfg, get_rsi2, fetcher)
    r1 = bt_rsi2.run(symbols, outdir=str(outdir / 'rsi2'))

    print('Running IBS...')
    bt_ibs = Backtester(cfg, get_ibs, fetcher)
    r2 = bt_ibs.run(symbols, outdir=str(outdir / 'ibs'))

    print('Running AND...')
    bt_and = Backtester(cfg, get_and, fetcher)
    r3 = bt_and.run(symbols, outdir=str(outdir / 'and'))

    # Combined summary
    rows = []
    for name, res in (('RSI2_crypto', r1), ('IBS_crypto', r2), ('AND_crypto', r3)):
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
    print('Showdown (crypto) complete. Summary:')
    print(df)


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
