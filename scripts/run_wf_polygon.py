#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path as _P
from datetime import date

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy, ConnorsRSI2Params
from strategies.connors_crsi.strategy import ConnorsCRSIStrategy, ConnorsCRSIParams
from strategies.ibs.strategy import IBSStrategy, IBSParams
from strategies.donchian.strategy import DonchianBreakoutStrategy, DonchianParams
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.providers.multi_source import fetch_daily_bars_multi
from backtest.walk_forward import generate_splits, run_walk_forward, summarize_results
from backtest.engine import BacktestConfig
from config.env_loader import load_env
from config.settings_loader import (
    get_regime_filter_config, is_regime_filter_enabled,
    get_selection_config, is_selection_enabled,
    is_earnings_filter_enabled,
)
from core.regime_filter import filter_signals_by_regime, fetch_spy_bars
from core.earnings_filter import filter_signals_by_earnings


def main():
    ap = argparse.ArgumentParser(description='Walk-forward backtest (Polygon) with RSI-2, IBS, CRSI, AND, and TOPN')
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
    ap.add_argument('--regime-on', action='store_true', default=False, help='Force regime filter ON (overrides config)')
    ap.add_argument('--topn-on', action='store_true', default=False, help='Force top-N selection ON (overrides config)')
    # Optional WR tuning knobs
    ap.add_argument('--rsi2-long-max', type=float, default=None, help='RSI2 long entry threshold (default 10)')
    ap.add_argument('--ibs-long-max', type=float, default=None, help='IBS long entry threshold (default 0.2)')
    ap.add_argument('--crsi-long-max', type=float, default=None, help='CRSI long entry threshold (default 10)')
    ap.add_argument('--time-stop-bars', type=int, default=None, help='Override time stop bars (default 5)')
    ap.add_argument('--topn-k', type=int, default=None, help='Override TOPN k (default from config)')
    ap.add_argument('--min-price', type=float, default=None, help='Override min price filter for selection')
    # Donchian breakout params
    ap.add_argument('--donchian-on', action='store_true', default=False, help='Run Donchian breakout overlay')
    ap.add_argument('--donchian-lookback', type=int, default=55)
    ap.add_argument('--donchian-stop-mult', type=float, default=2.0)
    ap.add_argument('--donchian-time-stop', type=int, default=20)
    ap.add_argument('--donchian-r-mult', type=float, default=2.5, help='Take-profit multiple of risk (entry-stop)')
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

    # Check config-gated features
    regime_cfg = get_regime_filter_config()
    selection_cfg = get_selection_config()
    # Optional overrides for selection
    if args.topn_k is not None and args.topn_k > 0:
        selection_cfg['top_n'] = int(args.topn_k)
    if args.min_price is not None and args.min_price > 0:
        selection_cfg['min_price'] = float(args.min_price)
    use_regime = args.regime_on or regime_cfg.get('enabled', False)
    use_topn = args.topn_on or selection_cfg.get('enabled', False)

    # Load SPY bars for regime filter if enabled
    spy_bars = pd.DataFrame()
    if use_regime:
        print('Regime filter enabled - loading SPY bars...')
        spy_bars = fetch_spy_bars(args.start, args.end, cache_dir=cache_dir)
        print(f'Loaded {len(spy_bars)} SPY bars for regime filtering')

    # Strategies with optional param overrides
    rsi2_params = ConnorsRSI2Params()
    ibs_params = IBSParams()
    crsi_params = ConnorsCRSIParams()
    if args.rsi2_long_max is not None:
        rsi2_params.long_entry_rsi_max = float(args.rsi2_long_max)
    if args.ibs_long_max is not None:
        ibs_params.ibs_long_max = float(args.ibs_long_max)
    if args.crsi_long_max is not None:
        crsi_params.long_entry_crsi_max = float(args.crsi_long_max)
    if args.time_stop_bars is not None and args.time_stop_bars > 0:
        rsi2_params.time_stop_bars = int(args.time_stop_bars)
        ibs_params.time_stop_bars = int(args.time_stop_bars)
        crsi_params.time_stop_bars = int(args.time_stop_bars)

    rsi2 = ConnorsRSI2Strategy(rsi2_params)
    ibs = IBSStrategy(ibs_params)
    crsi = ConnorsCRSIStrategy(crsi_params)
    don_params = DonchianParams(
        lookback=int(args.donchian_lookback),
        stop_mult=float(args.donchian_stop_mult),
        time_stop_bars=int(args.donchian_time_stop),
        min_price=float(selection_cfg.get('min_price', 5.0)),
        r_multiple=float(args.donchian_r_mult),
    )
    don = DonchianBreakoutStrategy(don_params)

    def apply_regime_filter(signals: pd.DataFrame) -> pd.DataFrame:
        """Apply regime filter if enabled."""
        if signals.empty:
            return signals
        if use_regime and not spy_bars.empty:
            signals = filter_signals_by_regime(signals, spy_bars, regime_cfg)
        return signals

    def apply_earnings_filter(signals: pd.DataFrame) -> pd.DataFrame:
        """Apply earnings proximity filter if enabled (config-gated)."""
        if signals.empty:
            return signals
        if is_earnings_filter_enabled():
            # Filter expects list[dict]; convert back to DataFrame
            recs = signals.to_dict('records')
            filtered = filter_signals_by_earnings(recs, date_key='timestamp', symbol_key='symbol')
            return pd.DataFrame(filtered)
        return signals

    def apply_topn_crosssectional(signals: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        """
        Cross-sectional ranking: compute ranks within each day's eligible set.
        rsi_rank = 1 - rank(rsi2)/N (lower RSI-2 => higher rank)
        ibs_rank = 1 - rank(ibs)/N (lower IBS => higher rank)
        score = w_rsi * rsi_rank + w_ibs * ibs_rank
        """
        if signals.empty:
            return signals
        top_n = cfg.get('top_n', 10)
        weights = cfg.get('score_weights', {})
        w_rsi2 = float(weights.get('rsi2', 0.6))
        w_ibs = float(weights.get('ibs', 0.4))
        w_crsi = float(weights.get('crsi', 0.0))
        min_price = cfg.get('min_price', 5.0)

        df = signals.copy()
        # Engine is long-only in v1; drop shorts to avoid selecting non-executable signals
        if 'side' in df.columns:
            df = df[df['side'].astype(str).str.lower() == 'long']
        if df.empty:
            return pd.DataFrame(columns=signals.columns)
        df['date'] = pd.to_datetime(df['timestamp']).dt.date

        # Min price filter
        if 'entry_price' in df.columns:
            df = df[df['entry_price'] >= min_price]
        if df.empty:
            return pd.DataFrame(columns=signals.columns)

        # Cross-sectional ranks per day
        def compute_daily_ranks(group):
            n = len(group)
            if n == 0:
                return group
            g = group.copy()
            # Lower RSI2 => rank 1 => rsi_rank close to 1
            if 'rsi2' in g.columns:
                g['rsi_raw_rank'] = g['rsi2'].rank(method='average', ascending=True)
                g['rsi_rank'] = 1 - (g['rsi_raw_rank'] - 1) / max(n - 1, 1)
            else:
                g['rsi_rank'] = 0.5
            # Lower IBS => rank 1 => ibs_rank close to 1
            if 'ibs' in g.columns:
                g['ibs_raw_rank'] = g['ibs'].rank(method='average', ascending=True)
                g['ibs_rank'] = 1 - (g['ibs_raw_rank'] - 1) / max(n - 1, 1)
            else:
                g['ibs_rank'] = 0.5
            # Lower CRSI => more oversold => higher rank
            if 'crsi' in g.columns:
                g['crsi_raw_rank'] = g['crsi'].rank(method='average', ascending=True)
                g['crsi_rank'] = 1 - (g['crsi_raw_rank'] - 1) / max(n - 1, 1)
            else:
                g['crsi_rank'] = 0.5
            w_sum = w_rsi2 + w_ibs + w_crsi
            if w_sum <= 0:
                g['composite_score'] = (g['rsi_rank'] + g['ibs_rank'] + g['crsi_rank']) / 3.0
            else:
                g['composite_score'] = (
                    (w_rsi2 * g['rsi_rank']) + (w_ibs * g['ibs_rank']) + (w_crsi * g['crsi_rank'])
                ) / w_sum
            return g

        df = df.groupby('date', group_keys=False).apply(compute_daily_ranks)

        # Select top N per day
        selected = []
        for day, grp in df.groupby('date'):
            grp_sorted = grp.sort_values('composite_score', ascending=False)
            n_sel = min(top_n, len(grp_sorted))
            selected.append(grp_sorted.head(n_sel))
            if use_topn:
                print(f'Day {day}: selected N={n_sel} of {len(grp)} eligible')

        if not selected:
            return pd.DataFrame(columns=signals.columns)

        result = pd.concat(selected, ignore_index=True)
        # Clean up helper columns
        drop_cols = ['date', 'rsi_raw_rank', 'rsi_rank', 'ibs_raw_rank', 'ibs_rank', 'crsi_raw_rank', 'crsi_rank', 'composite_score']
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
            return pd.DataFrame(columns=['timestamp','symbol','side','entry_price','stop_loss','take_profit','reason'])
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
        """Top-N cross-sectional selection using composite scoring."""
        # Start from AND signals if include_and_guard, else union of RSI2 + IBS
        if selection_cfg.get('include_and_guard', True):
            signals = get_and(df)
            # Enrich with CRSI values for ranking (merge by timestamp/symbol/side)
            cr = crsi.scan_signals_over_time(df)
            if not cr.empty:
                cr = cr[['timestamp','symbol','side','crsi']]
                signals = pd.merge(signals, cr, on=['timestamp','symbol','side'], how='left')
        else:
            # Union of RSI2 and IBS signals
            a = rsi2.scan_signals_over_time(df)
            b = ibs.scan_signals_over_time(df)
            base = pd.concat([a, b], ignore_index=True).drop_duplicates(subset=['timestamp', 'symbol', 'side'])
            # Enrich with CRSI values for ranking
            cr = crsi.scan_signals_over_time(df)
            if not cr.empty:
                cr = cr[['timestamp','symbol','side','crsi']]
                signals = pd.merge(base, cr, on=['timestamp','symbol','side'], how='left')
            else:
                signals = base
            signals = apply_regime_filter(signals)
            signals = apply_earnings_filter(signals)
        return apply_topn_crosssectional(signals, selection_cfg)

    def get_donchian(df: pd.DataFrame) -> pd.DataFrame:
        sigs = don.scan_signals_over_time(df)
        sigs = apply_regime_filter(sigs)
        sigs = apply_earnings_filter(sigs)
        # Selection can apply here too if desired; for now, take all donchian signals
        # and rely on regime/vol gating to limit trades.
        return sigs

    # Run WF per strategy
    rsi2_results = run_walk_forward(symbols, fetcher, get_rsi2, splits, outdir=str(outdir / 'rsi2'))
    ibs_results = run_walk_forward(symbols, fetcher, get_ibs, splits, outdir=str(outdir / 'ibs'))
    and_results = run_walk_forward(symbols, fetcher, get_and, splits, outdir=str(outdir / 'and'))
    crsi_results = run_walk_forward(symbols, fetcher, lambda df: apply_earnings_filter(apply_regime_filter(crsi.scan_signals_over_time(df))), splits, outdir=str(outdir / 'crsi'))
    don_results = []
    if args.donchian_on:
        don_results = run_walk_forward(symbols, fetcher, get_donchian, splits, outdir=str(outdir / 'donchian'))

    # TOPN variant (only if selection enabled)
    topn_results = []
    if use_topn:
        print(f'\nRunning TOPN variant (top_n={selection_cfg.get("top_n", 10)})...')
        topn_results = run_walk_forward(symbols, fetcher, get_topn, splits, outdir=str(outdir / 'topn'))

    # Summaries
    rsi2_summary = summarize_results(rsi2_results)
    ibs_summary = summarize_results(ibs_results)
    and_summary = summarize_results(and_results)
    crsi_summary = summarize_results(crsi_results)

    # Combined side-by-side CSV
    rows = []
    rows.append({'strategy': 'RSI2', **rsi2_summary})
    rows.append({'strategy': 'IBS', **ibs_summary})
    rows.append({'strategy': 'AND', **and_summary})
    rows.append({'strategy': 'CRSI', **crsi_summary})
    if don_results:
        rows.append({'strategy': 'DONCHIAN', **summarize_results(don_results)})

    # Add TOPN row if enabled
    if use_topn and topn_results:
        topn_summary = summarize_results(topn_results)
        rows.append({'strategy': 'TOPN', **topn_summary})
        pd.DataFrame(topn_results).to_csv(outdir / 'topn' / 'wf_splits.csv', index=False)

    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(outdir / 'wf_summary_compare.csv', index=False)

    # Also write detailed split metrics
    pd.DataFrame(rsi2_results).to_csv(outdir / 'rsi2' / 'wf_splits.csv', index=False)
    pd.DataFrame(ibs_results).to_csv(outdir / 'ibs' / 'wf_splits.csv', index=False)
    pd.DataFrame(and_results).to_csv(outdir / 'and' / 'wf_splits.csv', index=False)
    pd.DataFrame(crsi_results).to_csv(outdir / 'crsi' / 'wf_splits.csv', index=False)
    if don_results:
        pd.DataFrame(don_results).to_csv(outdir / 'donchian' / 'wf_splits.csv', index=False)

    print('\nWalk-forward complete. Summary:')
    print(compare_df)
    # Update metrics snapshot with averages (optional)
    try:
        from monitor.health_endpoints import update_performance_metrics
        update_performance_metrics(
            win_rate=float(compare_df['win_rate_avg'].mean()),
            profit_factor=float(compare_df['profit_factor_avg'].mean()),
            sharpe=float(compare_df['sharpe_avg'].mean()),
        )
    except Exception:
        pass


if __name__ == '__main__':
    main()
