#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path as _P

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
from strategies.ibs.strategy import IBSStrategy
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
    ap = argparse.ArgumentParser(description='Showdown backtest: RSI-2 vs IBS vs AND over full period')
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--start', type=str, required=True)
    ap.add_argument('--end', type=str, required=True)
    ap.add_argument('--cap', type=int, default=950)
    ap.add_argument('--outdir', type=str, default='showdown_outputs')
    ap.add_argument('--cache', type=str, default='data/cache')
    ap.add_argument('--dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
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
    rsi2 = ConnorsRSI2Strategy()
    ibs = IBSStrategy()

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

    def apply_topn_crosssectional(signals: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        if signals.empty:
            return signals
        top_n = cfg.get('top_n', 10)
        weights = cfg.get('score_weights', {})
        w_rsi2 = weights.get('rsi2', 0.6)
        w_ibs = weights.get('ibs', 0.4)
        min_price = cfg.get('min_price', 5.0)

        df = signals.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date

        if 'entry_price' in df.columns:
            df = df[df['entry_price'] >= min_price]
        if df.empty:
            return pd.DataFrame(columns=signals.columns)

        def compute_daily_ranks(group):
            n = len(group)
            if n == 0:
                return group
            g = group.copy()
            if 'rsi2' in g.columns:
                g['rsi_raw_rank'] = g['rsi2'].rank(method='average', ascending=True)
                g['rsi_rank'] = 1 - (g['rsi_raw_rank'] - 1) / max(n - 1, 1)
            else:
                g['rsi_rank'] = 0.5
            if 'ibs' in g.columns:
                g['ibs_raw_rank'] = g['ibs'].rank(method='average', ascending=True)
                g['ibs_rank'] = 1 - (g['ibs_raw_rank'] - 1) / max(n - 1, 1)
            else:
                g['ibs_rank'] = 0.5
            g['composite_score'] = w_rsi2 * g['rsi_rank'] + w_ibs * g['ibs_rank']
            return g

        df = df.groupby('date', group_keys=False).apply(compute_daily_ranks)

        selected = []
        for day, grp in df.groupby('date'):
            grp_sorted = grp.sort_values('composite_score', ascending=False)
            n_sel = min(top_n, len(grp_sorted))
            selected.append(grp_sorted.head(n_sel))

        if not selected:
            return pd.DataFrame(columns=signals.columns)

        result = pd.concat(selected, ignore_index=True)
        drop_cols = ['date', 'rsi_raw_rank', 'rsi_rank', 'ibs_raw_rank', 'ibs_rank', 'composite_score']
        result = result.drop(columns=[c for c in drop_cols if c in result.columns], errors='ignore')
        return result.reset_index(drop=True)

    def get_rsi2(df: pd.DataFrame) -> pd.DataFrame:
        signals = rsi2.scan_signals_over_time(df)
        signals = apply_regime_filter(signals)
        signals = apply_earnings_filter(signals)
        return signals

    def get_ibs(df: pd.DataFrame) -> pd.DataFrame:
        signals = ibs.scan_signals_over_time(df)
        signals = apply_regime_filter(signals)
        signals = apply_earnings_filter(signals)
        return signals

    def get_and(df: pd.DataFrame) -> pd.DataFrame:
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
        if 'rsi2_rsi2' in merged.columns: out['rsi2'] = merged['rsi2_rsi2']
        elif 'rsi2' in merged.columns: out['rsi2'] = merged['rsi2']
        if 'ibs_ibs' in merged.columns: out['ibs'] = merged['ibs_ibs']
        elif 'ibs' in merged.columns: out['ibs'] = merged['ibs']
        out = apply_regime_filter(out)
        out = apply_earnings_filter(out)
        return out

    def get_topn(df: pd.DataFrame) -> pd.DataFrame:
        if selection_cfg.get('include_and_guard', True):
            signals = get_and(df)
        else:
            a = rsi2.scan_signals_over_time(df)
            b = ibs.scan_signals_over_time(df)
            signals = pd.concat([a, b], ignore_index=True).drop_duplicates(subset=['timestamp', 'symbol', 'side'])
            signals = apply_regime_filter(signals)
            signals = apply_earnings_filter(signals)
        return apply_topn_crosssectional(signals, selection_cfg)

    # Run per strategy
    cfg = BacktestConfig(initial_cash=100_000.0)
    bt_rsi2 = Backtester(cfg, get_rsi2, fetcher)
    r1 = bt_rsi2.run(symbols, outdir=str(outdir / 'rsi2'))
    bt_ibs = Backtester(cfg, get_ibs, fetcher)
    r2 = bt_ibs.run(symbols, outdir=str(outdir / 'ibs'))
    bt_and = Backtester(cfg, get_and, fetcher)
    r3 = bt_and.run(symbols, outdir=str(outdir / 'and'))

    # TOPN variant if enabled
    r4 = None
    if use_topn:
        print(f'\nRunning TOPN variant (top_n={selection_cfg.get("top_n", 10)})...')
        bt_topn = Backtester(cfg, get_topn, fetcher)
        r4 = bt_topn.run(symbols, outdir=str(outdir / 'topn'))

    # Combined summary
    rows = []
    for name, res in (('RSI2', r1), ('IBS', r2), ('AND', r3)):
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

    # Add TOPN row if enabled
    if use_topn and r4:
        m = r4.get('metrics', {})
        rows.append({
            'strategy': 'TOPN',
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
            '</head><body><h1>Showdown Summary</h1>', df.to_html(index=False), '</body></html>']
    (outdir / 'showdown_report.html').write_text('\n'.join(html), encoding='utf-8')
    print('\nShowdown complete. Summary:')
    print(df)


if __name__ == '__main__':
    main()
