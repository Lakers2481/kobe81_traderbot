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

from strategies.ibs_rsi.strategy import IbsRsiStrategy, IbsRsiParams
from strategies.ict.turtle_soup import TurtleSoupStrategy, TurtleSoupParams
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.providers.multi_source import fetch_daily_bars_multi
from backtest.walk_forward import generate_splits, run_walk_forward, summarize_results
from backtest.engine import BacktestConfig, CommissionConfig
from config.env_loader import load_env
from config.settings_loader import (
    get_regime_filter_config, is_regime_filter_enabled,
    get_selection_config, is_selection_enabled,
    is_earnings_filter_enabled,
    get_setting, get_commission_config,
)
from core.regime_filter import filter_signals_by_regime, fetch_spy_bars
from core.earnings_filter import filter_signals_by_earnings


def main():
    ap = argparse.ArgumentParser(description='Walk-forward backtest (Polygon) with IBS+RSI and ICT Turtle Soup')
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--start', type=str, required=True, help='YYYY-MM-DD')
    ap.add_argument('--end', type=str, required=True, help='YYYY-MM-DD')
    ap.add_argument('--train-days', type=int, default=252)
    ap.add_argument('--test-days', type=int, default=63)
    ap.add_argument('--anchored', action='store_true', default=False)
    ap.add_argument('--cap', type=int, default=900)
    ap.add_argument('--outdir', type=str, default='wf_outputs')
    ap.add_argument('--cache', type=str, default='data/cache')
    ap.add_argument('--fallback-free', action='store_true', default=False, help='Backfill pre-Polygon coverage with Yahoo Finance')
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--regime-on', action='store_true', default=False, help='Force regime filter ON (overrides config)')
    # Streamlined CLI (no RSI2/IBS/CRSI/TOPN knobs in two-strategy setup)
    # IBS+RSI params
    ap.add_argument('--ibs-on', action='store_true', default=False, help='Run IBS+RSI overlay')
    ap.add_argument('--ibs-max', type=float, default=0.15)
    ap.add_argument('--rsi-max', type=float, default=10.0)
    ap.add_argument('--ibs-atr-mult', type=float, default=1.0)
    ap.add_argument('--ibs-r-mult', type=float, default=2.0)
    ap.add_argument('--ibs-time-stop', type=int, default=5)
    # Turtle Soup (ICT Liquidity Sweep) params
    ap.add_argument('--turtle-soup-on', action='store_true', default=False, help='Run Turtle Soup (ICT liquidity sweep) strategy')
    ap.add_argument('--turtle-soup-lookback', type=int, default=20, help='N-day channel lookback (default 20)')
    ap.add_argument('--turtle-soup-min-bars', type=int, default=3, help='Min bars since prior extreme (default 3)')
    ap.add_argument('--turtle-soup-r-mult', type=float, default=2.0, help='Take-profit R-multiple (default 2.0)')
    ap.add_argument('--turtle-soup-time-stop', type=int, default=5, help='Time stop bars (default 5)')
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
    use_regime = args.regime_on or regime_cfg.get('enabled', False)

    # BacktestConfig factory (inject slippage/commissions from config)
    def make_bt_cfg() -> BacktestConfig:
        slippage_pct = float(get_setting('backtest.slippage_pct', 0.0005) or 0.0005)
        slippage_bps = float(slippage_pct * 10000.0)
        c = get_commission_config()
        commissions = CommissionConfig(
            enabled=bool(c.get('enabled', False)),
            per_share=float(c.get('per_share', 0.0)),
            min_per_order=float(c.get('min_per_order', 0.0)),
            bps=float(c.get('bps', 0.0)),
            sec_fee_per_dollar=float(c.get('sec_fee_per_dollar', 0.0000278)),
            taf_fee_per_share=float(c.get('taf_fee_per_share', 0.000166)),
        )
        return BacktestConfig(initial_cash=100_000.0, slippage_bps=slippage_bps, commissions=commissions)

    # Load SPY bars for regime filter if enabled
    spy_bars = pd.DataFrame()
    if use_regime:
        print('Regime filter enabled - loading SPY bars...')
        spy_bars = fetch_spy_bars(args.start, args.end, cache_dir=cache_dir)
        print(f'Loaded {len(spy_bars)} SPY bars for regime filtering')

    # Strategies
    ibs_params = IbsRsiParams(
        ibs_max=float(args.ibs_max),
        rsi_max=float(args.rsi_max),
        atr_mult=float(args.ibs_atr_mult),
        r_multiple=float(args.ibs_r_mult),
        time_stop_bars=int(args.ibs_time_stop),
        min_price=float(get_setting('selection.min_price', 10.0)),
    )
    ibs = IbsRsiStrategy(ibs_params)

    # Turtle Soup (ICT Liquidity Sweep) strategy
    ts_params = TurtleSoupParams(
        lookback=int(args.turtle_soup_lookback),
        min_bars_since_extreme=int(args.turtle_soup_min_bars),
        r_multiple=float(args.turtle_soup_r_mult),
        time_stop_bars=int(args.turtle_soup_time_stop),
        min_price=float(get_setting('selection.min_price', 10.0)),
    )
    turtle_soup = TurtleSoupStrategy(ts_params)

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

    # No cross-sectional TOPN ranking in this two-strategy setup

    def get_ibs(df: pd.DataFrame) -> pd.DataFrame:
        sigs = ibs.generate_signals(df)
        sigs = apply_regime_filter(sigs)
        sigs = apply_earnings_filter(sigs)
        return sigs

    def get_turtle_soup(df: pd.DataFrame) -> pd.DataFrame:
        """Turtle Soup (ICT Liquidity Sweep) - trades failed breakouts."""
        sigs = turtle_soup.scan_signals_over_time(df)
        sigs = apply_regime_filter(sigs)
        sigs = apply_earnings_filter(sigs)
        return sigs

    # Run WF per selected strategies
    print('\nRunning IBS+RSI Mean Reversion...')
    don_results = run_walk_forward(symbols, fetcher, get_ibs, splits, outdir=str(outdir / 'ibs_rsi'), config_factory=make_bt_cfg)
    print('Running ICT Turtle Soup...')
    ts_results = run_walk_forward(symbols, fetcher, get_turtle_soup, splits, outdir=str(outdir / 'turtle_soup'), config_factory=make_bt_cfg)

    # Ensure subdirs exist for CSV outputs
    (outdir / 'ibs_rsi').mkdir(parents=True, exist_ok=True)
    (outdir / 'turtle_soup').mkdir(parents=True, exist_ok=True)

    # Summaries
    # Combined side-by-side CSV
    rows = []
    rows.append({'strategy': 'IBS_RSI', **summarize_results(don_results)})
    rows.append({'strategy': 'TURTLE_SOUP', **summarize_results(ts_results)})

    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(outdir / 'wf_summary_compare.csv', index=False)

    # Also write detailed split metrics
    pd.DataFrame(don_results).to_csv(outdir / 'ibs_rsi' / 'wf_splits.csv', index=False)
    pd.DataFrame(ts_results).to_csv(outdir / 'turtle_soup' / 'wf_splits.csv', index=False)

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
