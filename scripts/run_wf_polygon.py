#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path as _P
from datetime import date

import pandas as pd

import sys
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from strategies.registry import get_production_scanner
from strategies.dual_strategy.combined import DualStrategyParams
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.providers.multi_source import fetch_daily_bars_multi
from backtest.walk_forward import generate_splits, run_walk_forward, run_walk_forward_parallel, summarize_results
from backtest.engine import BacktestConfig, CommissionConfig
from config.env_loader import load_env
from config.settings_loader import (
    get_regime_filter_config, is_earnings_filter_enabled,
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
    # IBS+RSI params (v2.2 defaults)
    ap.add_argument('--ibs-on', action='store_true', default=False, help='Run IBS+RSI overlay')
    ap.add_argument('--ibs-max', type=float, default=0.08, help='IBS threshold (default 0.08)')
    ap.add_argument('--rsi-max', type=float, default=5.0, help='RSI(2) threshold (default 5.0)')
    ap.add_argument('--ibs-atr-mult', type=float, default=2.0, help='ATR stop multiple (default 2.0)')
    ap.add_argument('--ibs-r-mult', type=float, default=2.0, help='Take-profit R multiple (default 2.0)')
    ap.add_argument('--ibs-time-stop', type=int, default=7, help='Time stop bars (default 7)')
    # Turtle Soup (ICT Liquidity Sweep) params (v2.2 defaults)
    ap.add_argument('--turtle-soup-on', action='store_true', default=False, help='Run Turtle Soup (ICT liquidity sweep) strategy')
    ap.add_argument('--turtle-soup-lookback', type=int, default=20, help='N-day channel lookback (default 20)')
    ap.add_argument('--turtle-soup-min-bars', type=int, default=3, help='Min bars since prior extreme (default 3)')
    ap.add_argument('--turtle-soup-stop-buf', type=float, default=0.2, help='ATR stop buffer multiple (default 0.2)')
    ap.add_argument('--turtle-soup-r-mult', type=float, default=0.5, help='Take-profit R-multiple (default 0.5)')
    ap.add_argument('--turtle-soup-time-stop', type=int, default=3, help='Time stop bars (default 3)')
    ap.add_argument('--turtle-soup-min-sweep', type=float, default=0.3, help='Min sweep strength in ATR to accept (default 0.3)')
    # Parallel execution
    ap.add_argument('--parallel', action='store_true', default=False, help='Run splits in parallel (4x speedup)')
    ap.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default 4)')
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

    # Use canonical DualStrategyScanner with CLI-overridable params
    dual_params = DualStrategyParams(
        # IBS+RSI params
        ibs_entry=float(args.ibs_max),
        rsi_entry=float(args.rsi_max),
        ibs_rsi_stop_mult=float(args.ibs_atr_mult),
        ibs_rsi_time_stop=int(args.ibs_time_stop),
        # Turtle Soup params
        ts_lookback=int(args.turtle_soup_lookback),
        ts_min_bars_since_extreme=int(args.turtle_soup_min_bars),
        ts_stop_buffer_mult=float(args.turtle_soup_stop_buf),
        ts_r_multiple=float(args.turtle_soup_r_mult),
        ts_time_stop=int(args.turtle_soup_time_stop),
        ts_min_sweep_strength=float(args.turtle_soup_min_sweep),  # CRITICAL: v2.2 sweep filter
        # Common params
        min_price=float(get_setting('selection.min_price', 10.0)),
    )
    scanner = get_production_scanner(dual_params)

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
        """Generate IBS+RSI signals using canonical DualStrategyScanner."""
        sigs = scanner.scan_signals_over_time(df)
        # Filter to IBS_RSI strategy only
        if not sigs.empty and 'strategy' in sigs.columns:
            sigs = sigs[sigs['strategy'] == 'IBS_RSI']
        sigs = apply_regime_filter(sigs)
        sigs = apply_earnings_filter(sigs)
        return sigs

    def get_turtle_soup(df: pd.DataFrame) -> pd.DataFrame:
        """Generate Turtle Soup signals using canonical DualStrategyScanner."""
        sigs = scanner.scan_signals_over_time(df)
        # Filter to Turtle_Soup strategy only (sweep filter already applied in DualStrategyScanner)
        if not sigs.empty and 'strategy' in sigs.columns:
            sigs = sigs[sigs['strategy'].isin(['Turtle_Soup', 'TurtleSoup'])]
        sigs = apply_regime_filter(sigs)
        sigs = apply_earnings_filter(sigs)
        return sigs

    # Determine which strategies to run:
    # If neither flag is set, run both. If any is set, run only the selected ones.
    run_ibs = args.ibs_on or (not args.ibs_on and not args.turtle_soup_on)
    run_ts = args.turtle_soup_on or (not args.ibs_on and not args.turtle_soup_on)

    don_results = []
    ts_results = []

    # Prepare config dict for parallel execution
    bt_cfg = make_bt_cfg()
    config_dict = {
        'initial_cash': bt_cfg.initial_cash,
        'slippage_bps': bt_cfg.slippage_bps,
    }
    if bt_cfg.commissions:
        config_dict['commissions'] = {
            'enabled': bt_cfg.commissions.enabled,
            'per_share': bt_cfg.commissions.per_share,
            'min_per_order': bt_cfg.commissions.min_per_order,
            'bps': bt_cfg.commissions.bps,
            'sec_fee_per_dollar': bt_cfg.commissions.sec_fee_per_dollar,
            'taf_fee_per_share': bt_cfg.commissions.taf_fee_per_share,
        }

    # Strategy params dict for parallel execution
    # Note: Only include params that DualStrategyParams accepts
    strategy_params = {
        '_class': 'DualStrategyParams',
        'ibs_entry': float(args.ibs_max),
        'rsi_entry': float(args.rsi_max),
        'ibs_rsi_stop_mult': float(args.ibs_atr_mult),
        'ibs_rsi_time_stop': int(args.ibs_time_stop),
        'ts_lookback': int(args.turtle_soup_lookback),
        'ts_min_bars_since_extreme': int(args.turtle_soup_min_bars),
        'ts_stop_buffer_mult': float(args.turtle_soup_stop_buf),
        'ts_r_multiple': float(args.turtle_soup_r_mult),
        'ts_time_stop': int(args.turtle_soup_time_stop),
        'ts_min_sweep_strength': float(args.turtle_soup_min_sweep),
        'min_price': float(get_setting('selection.min_price', 10.0)),
    }

    # Data params for parallel execution
    data_params_ibs = {
        'fetch_func': 'fetch_daily_bars_multi' if args.fallback_free else 'fetch_daily_bars_polygon',
        'cache_dir': str(cache_dir),
        'strategy_filter': ['IBS_RSI'],
    }
    data_params_ts = {
        'fetch_func': 'fetch_daily_bars_multi' if args.fallback_free else 'fetch_daily_bars_polygon',
        'cache_dir': str(cache_dir),
        'strategy_filter': ['Turtle_Soup', 'TurtleSoup'],
    }
    data_provider = 'data.providers.multi_source' if args.fallback_free else 'data.providers.polygon_eod'

    if run_ibs:
        print('\nRunning IBS+RSI Mean Reversion...')
        if args.parallel:
            print(f'  Using PARALLEL mode with {args.workers} workers')
            don_results = run_walk_forward_parallel(
                symbols=symbols,
                splits=splits,
                outdir=str(outdir / 'ibs_rsi'),
                strategy_module='strategies.dual_strategy.combined',
                strategy_class='DualStrategyScanner',
                strategy_params=strategy_params,
                data_provider=data_provider,
                data_params=data_params_ibs,
                config_dict=config_dict,
                max_workers=args.workers,
            )
        else:
            don_results = run_walk_forward(symbols, fetcher, get_ibs, splits, outdir=str(outdir / 'ibs_rsi'), config_factory=make_bt_cfg)
    else:
        print('\nSkipping IBS+RSI (flag not set)')

    if run_ts:
        print('Running ICT Turtle Soup...')
        if args.parallel:
            print(f'  Using PARALLEL mode with {args.workers} workers')
            ts_results = run_walk_forward_parallel(
                symbols=symbols,
                splits=splits,
                outdir=str(outdir / 'turtle_soup'),
                strategy_module='strategies.dual_strategy.combined',
                strategy_class='DualStrategyScanner',
                strategy_params=strategy_params,
                data_provider=data_provider,
                data_params=data_params_ts,
                config_dict=config_dict,
                max_workers=args.workers,
            )
        else:
            ts_results = run_walk_forward(symbols, fetcher, get_turtle_soup, splits, outdir=str(outdir / 'turtle_soup'), config_factory=make_bt_cfg)
    else:
        print('Skipping ICT Turtle Soup (flag not set)')

    # Ensure subdirs exist for CSV outputs
    (outdir / 'ibs_rsi').mkdir(parents=True, exist_ok=True)
    (outdir / 'turtle_soup').mkdir(parents=True, exist_ok=True)

    # Summaries
    # Combined side-by-side CSV
    rows = []
    if run_ibs:
        rows.append({'strategy': 'IBS_RSI', **summarize_results(don_results)})
    if run_ts:
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
