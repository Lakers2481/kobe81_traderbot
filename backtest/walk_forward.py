from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, List, Dict, Any, Optional

import pandas as pd

from .engine import Backtester, BacktestConfig


@dataclass
class WFSplit:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def generate_splits(
    start: date,
    end: date,
    train_days: int = 252,
    test_days: int = 63,
    anchored: bool = False,
) -> List[WFSplit]:
    """Generate rolling or anchored walk-forward splits in trading days (approx calendar days)."""
    splits: List[WFSplit] = []
    cur_train_start = start
    while True:
        if anchored:
            cur_train_start = start
        train_end = cur_train_start + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days - 1)
        if test_start > end:
            break
        if test_end > end:
            test_end = end
        splits.append(WFSplit(
            train_start=train_start_to_date(cur_train_start),
            train_end=train_start_to_date(train_end),
            test_start=train_start_to_date(test_start),
            test_end=train_start_to_date(test_end),
        ))
        # Advance by test_days
        cur_train_start = cur_train_start + timedelta(days=test_days)
        if cur_train_start + timedelta(days=train_days) > end:
            break
    return splits


def train_start_to_date(d: date | datetime) -> date:
    return d if isinstance(d, date) and not isinstance(d, datetime) else d.date()


def run_walk_forward(
    symbols: List[str],
    fetch_bars_for_window: Callable[[str, str, str], pd.DataFrame],
    get_signals: Callable[[pd.DataFrame], pd.DataFrame],
    splits: List[WFSplit],
    outdir: str,
    initial_cash: float = 100_000.0,
    config_factory: Optional[Callable[[], BacktestConfig]] = None,
) -> List[Dict[str, Any]]:
    """
    Run walk-forward backtests over splits.

    fetch_bars_for_window: (symbol, start_str, end_str) -> DataFrame
    get_signals: strategy function mapping merged OHLCV -> signal DataFrame
    """
    results: List[Dict[str, Any]] = []
    for i, sp in enumerate(splits, start=1):
        # Fetch from train_start to ensure indicators (e.g., SMA200) have sufficient lookback
        start_s = sp.train_start.isoformat()
        end_s = sp.test_end.isoformat()

        def fetcher(sym: str) -> pd.DataFrame:
            return fetch_bars_for_window(sym, start_s, end_s)

        cfg = config_factory() if config_factory is not None else BacktestConfig(initial_cash=initial_cash)
        bt = Backtester(cfg, get_signals, fetcher)
        res = bt.run(symbols, outdir=f"{outdir}/split_{i:02d}")
        m = res.get("metrics", {})
        results.append({
            "split": i,
            "test_start": start_s,
            "test_end": end_s,
            **m,
        })
    return results


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"splits": 0}
    df = pd.DataFrame(results)
    summary = {
        "splits": len(results),
        "trades_total": int(df["trades"].sum()),
        "win_rate_avg": float(df["win_rate"].mean()),
        "profit_factor_avg": float(df["profit_factor"].replace([float('inf')], pd.NA).dropna().mean() or 0.0),
        "sharpe_avg": float(df["sharpe"].mean()),
        "max_drawdown_avg": float(df["max_drawdown"].mean()),
    }
    # Include totals for gross/net PnL and fees if present
    for col in ("gross_pnl", "net_pnl", "total_fees"):
        if col in df.columns:
            key = f"{col}_total"
            try:
                summary[key] = float(df[col].sum())
            except Exception as e:
                # Log the error but continue with default value
                import logging
                logging.warning(f"Failed to compute {key}: {e}")
                summary[key] = 0.0
    return summary


# ============================================================================
# PARALLEL WALK-FORWARD (ProcessPoolExecutor)
# ============================================================================

def _run_single_split_worker(args: tuple) -> Dict[str, Any]:
    """
    Worker function for parallel walk-forward. Must be top-level for pickling.

    Args is a tuple of (split_idx, split, symbols, outdir, config_dict,
                        strategy_module, strategy_class, strategy_params,
                        data_provider, data_params)
    """
    (
        split_idx,
        split_dict,
        symbols,
        outdir,
        config_dict,
        strategy_module,
        strategy_class,
        strategy_params,
        data_provider,
        data_params,
    ) = args

    import importlib
    from pathlib import Path

    # Reconstruct the split
    sp_train_start = split_dict['train_start']
    sp_train_end = split_dict['train_end']
    sp_test_start = split_dict['test_start']
    sp_test_end = split_dict['test_end']

    start_s = sp_train_start
    end_s = sp_test_end

    # Dynamically import and instantiate strategy
    strat_mod = importlib.import_module(strategy_module)
    strategy_cls = getattr(strat_mod, strategy_class)
    if strategy_params:
        params_cls_name = strategy_params.get('_class')
        if params_cls_name:
            params_cls = getattr(strat_mod, params_cls_name)
            params_dict = {k: v for k, v in strategy_params.items() if k != '_class'}
            params = params_cls(**params_dict)
            scanner = strategy_cls(params)
        else:
            scanner = strategy_cls()
    else:
        scanner = strategy_cls()

    # Dynamically import data fetcher
    data_mod = importlib.import_module(data_provider)
    fetch_func_name = data_params.get('fetch_func', 'fetch_daily_bars_multi')
    fetch_func = getattr(data_mod, fetch_func_name)
    cache_dir = Path(data_params.get('cache_dir', 'data/cache'))

    def fetcher(sym: str) -> pd.DataFrame:
        return fetch_func(sym, start_s, end_s, cache_dir=cache_dir)

    def get_signals(df: pd.DataFrame) -> pd.DataFrame:
        sigs = scanner.scan_signals_over_time(df)
        # Apply strategy filter if specified
        strategy_filter = data_params.get('strategy_filter')
        if strategy_filter and not sigs.empty and 'strategy' in sigs.columns:
            sigs = sigs[sigs['strategy'].isin(strategy_filter)]
        return sigs

    # Reconstruct BacktestConfig
    from .engine import BacktestConfig, CommissionConfig

    commissions = None
    if 'commissions' in config_dict and config_dict['commissions']:
        comm_dict = config_dict['commissions']
        commissions = CommissionConfig(**comm_dict)

    cfg = BacktestConfig(
        initial_cash=config_dict.get('initial_cash', 100_000.0),
        slippage_bps=config_dict.get('slippage_bps', 5.0),
        commissions=commissions,
    )

    bt = Backtester(cfg, get_signals, fetcher)
    split_outdir = f"{outdir}/split_{split_idx:02d}"
    res = bt.run(symbols, outdir=split_outdir)
    m = res.get("metrics", {})

    return {
        "split": split_idx,
        "test_start": start_s,
        "test_end": end_s,
        **m,
    }


def run_walk_forward_parallel(
    symbols: List[str],
    splits: List[WFSplit],
    outdir: str,
    strategy_module: str,
    strategy_class: str,
    strategy_params: Optional[Dict[str, Any]] = None,
    data_provider: str = "data.providers.multi_source",
    data_params: Optional[Dict[str, Any]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    max_workers: int = 4,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run walk-forward backtests in parallel using ProcessPoolExecutor.

    This version uses serializable parameters instead of callbacks to enable
    multiprocessing. Each worker imports the necessary modules dynamically.

    Args:
        symbols: List of ticker symbols
        splits: List of WFSplit objects
        outdir: Output directory for results
        strategy_module: Full module path (e.g., 'strategies.dual_strategy.combined')
        strategy_class: Class name (e.g., 'DualStrategyScanner')
        strategy_params: Dict of params to pass to strategy (include '_class' for params class name)
        data_provider: Module path for data fetcher
        data_params: Dict with 'fetch_func', 'cache_dir', 'strategy_filter' keys
        config_dict: BacktestConfig as dict (initial_cash, slippage_bps, commissions)
        max_workers: Number of parallel processes (default 4)
        verbose: Print progress updates

    Returns:
        List of result dicts for each split

    Example:
        results = run_walk_forward_parallel(
            symbols=['AAPL', 'MSFT'],
            splits=splits,
            outdir='wf_outputs/ibs_rsi',
            strategy_module='strategies.dual_strategy.combined',
            strategy_class='DualStrategyScanner',
            strategy_params={
                '_class': 'DualStrategyParams',
                'ibs_entry': 0.08,
                'rsi_entry': 5.0,
            },
            data_provider='data.providers.multi_source',
            data_params={
                'fetch_func': 'fetch_daily_bars_multi',
                'cache_dir': 'data/cache',
                'strategy_filter': ['IBS_RSI'],
            },
            config_dict={
                'initial_cash': 100_000.0,
                'slippage_bps': 5.0,
            },
            max_workers=4,
        )
    """
    if data_params is None:
        data_params = {'fetch_func': 'fetch_daily_bars_multi', 'cache_dir': 'data/cache'}
    if config_dict is None:
        config_dict = {'initial_cash': 100_000.0, 'slippage_bps': 5.0}
    if strategy_params is None:
        strategy_params = {}

    # Prepare worker arguments
    worker_args = []
    for i, sp in enumerate(splits, start=1):
        split_dict = {
            'train_start': sp.train_start.isoformat(),
            'train_end': sp.train_end.isoformat(),
            'test_start': sp.test_start.isoformat(),
            'test_end': sp.test_end.isoformat(),
        }
        worker_args.append((
            i,
            split_dict,
            symbols,
            outdir,
            config_dict,
            strategy_module,
            strategy_class,
            strategy_params,
            data_provider,
            data_params,
        ))

    # Limit workers to number of splits
    actual_workers = min(max_workers, len(splits))

    if verbose:
        print(f"Running {len(splits)} splits with {actual_workers} parallel workers...")

    # Handle empty splits case
    if len(splits) == 0:
        if verbose:
            print("  No splits to run (date range too short for train+test periods)")
        return []

    results = []

    # Use spawn context for Windows compatibility
    ctx = multiprocessing.get_context('spawn')

    with ProcessPoolExecutor(max_workers=actual_workers, mp_context=ctx) as executor:
        future_to_split = {executor.submit(_run_single_split_worker, args): args[0] for args in worker_args}
        completed = 0

        for future in as_completed(future_to_split):
            split_idx = future_to_split[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)
                if verbose:
                    trades = result.get('trades', 0)
                    wr = result.get('win_rate', 0)
                    print(f"  [{completed}/{len(splits)}] Split {split_idx}: {trades} trades, {wr:.1%} WR", flush=True)
            except Exception as e:
                if verbose:
                    print(f"  [{completed}/{len(splits)}] Split {split_idx}: FAILED - {e}", flush=True)
                results.append({
                    "split": split_idx,
                    "error": str(e),
                    "trades": 0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "sharpe": 0.0,
                    "max_drawdown": 0.0,
                })

    # Sort by split number
    results.sort(key=lambda x: x.get('split', 0))

    return results
