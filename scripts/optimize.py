#!/usr/bin/env python3
"""
Parameter Optimization via Grid Search

- Grid search over strategy parameters
- Test parameter combinations via backtest
- Find optimal parameters with validation
- Guard against overfitting (train/test split)

Usage:
    python optimize.py --strategy rsi2 --universe universe.csv --start 2020-01-01 --end 2023-12-31
    python optimize.py --strategy ibs --param-grid ibs_params.json --validate 0.3
    python optimize.py --strategy rsi2 --dotenv path/to/.env
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Type, Callable

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.env_loader import load_env
from backtest.engine import Backtester, BacktestConfig
from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy, ConnorsRSI2Params
from strategies.ibs.strategy import IBSStrategy, IBSParams


# Default parameter grids for each strategy
DEFAULT_PARAM_GRIDS = {
    'rsi2': {
        'rsi_period': [2, 3],
        'sma_period': [150, 200, 250],
        'atr_period': [10, 14, 20],
        'atr_stop_mult': [1.5, 2.0, 2.5],
        'long_entry_rsi_max': [5, 10, 15],
        'short_entry_rsi_min': [85, 90, 95],
    },
    'ibs': {
        'sma_period': [150, 200, 250],
        'atr_period': [10, 14, 20],
        'atr_stop_mult': [1.5, 2.0, 2.5],
        'ibs_long_max': [0.1, 0.2, 0.3],
        'ibs_short_min': [0.7, 0.8, 0.9],
    },
}


@dataclass
class OptimizationResult:
    """Result of a single parameter combination test."""
    params: Dict[str, Any]
    train_metrics: Dict[str, Any]
    test_metrics: Optional[Dict[str, Any]]
    score: float  # Optimization target (e.g., Sharpe)


def load_param_grid(path: Optional[Path], strategy: str) -> Dict[str, List[Any]]:
    """
    Load parameter grid from JSON file or return default.

    JSON format:
    {
        "param_name": [value1, value2, ...],
        ...
    }
    """
    if path and path.exists():
        try:
            with open(path, 'r') as f:
                grid = json.load(f)
            print(f"[INFO] Loaded parameter grid from {path}")
            return grid
        except Exception as e:
            print(f"[WARN] Failed to load param grid: {e}. Using default.")

    if strategy in DEFAULT_PARAM_GRIDS:
        print(f"[INFO] Using default parameter grid for {strategy}")
        return DEFAULT_PARAM_GRIDS[strategy]

    print(f"[WARN] No parameter grid available for {strategy}")
    return {}


def generate_param_combinations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters from grid."""
    if not grid:
        return [{}]

    keys = list(grid.keys())
    values = list(grid.values())

    combinations = []
    for combo in itertools.product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)

    return combinations


def create_strategy(strategy_name: str, params: Dict[str, Any]):
    """Create strategy instance with given parameters."""
    if strategy_name == 'rsi2':
        param_obj = ConnorsRSI2Params(**{k: v for k, v in params.items()
                                         if k in [f.name for f in fields(ConnorsRSI2Params)]})
        return ConnorsRSI2Strategy(params=param_obj)
    elif strategy_name == 'ibs':
        param_obj = IBSParams(**{k: v for k, v in params.items()
                                 if k in [f.name for f in fields(IBSParams)]})
        return IBSStrategy(params=param_obj)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def split_date_range(
    start: str,
    end: str,
    validate_ratio: float = 0.3,
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """
    Split date range into train and test periods.

    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        validate_ratio: Fraction of data for validation/test

    Returns:
        ((train_start, train_end), (test_start, test_end))
    """
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt = datetime.strptime(end, '%Y-%m-%d')

    total_days = (end_dt - start_dt).days
    train_days = int(total_days * (1 - validate_ratio))

    train_end_dt = start_dt + timedelta(days=train_days)
    test_start_dt = train_end_dt + timedelta(days=1)

    return (
        (start, train_end_dt.strftime('%Y-%m-%d')),
        (test_start_dt.strftime('%Y-%m-%d'), end),
    )


def run_backtest_for_params(
    strategy_name: str,
    params: Dict[str, Any],
    symbols: List[str],
    start: str,
    end: str,
    fetch_bars: Callable[[str], pd.DataFrame],
    initial_cash: float = 100000.0,
) -> Dict[str, Any]:
    """Run backtest with given strategy parameters."""
    try:
        strategy = create_strategy(strategy_name, params)

        def get_signals(df: pd.DataFrame) -> pd.DataFrame:
            return strategy.scan_signals_over_time(df)

        cfg = BacktestConfig(initial_cash=initial_cash)
        bt = Backtester(cfg, get_signals, fetch_bars)
        result = bt.run(symbols)

        return result.get('metrics', {})
    except Exception as e:
        print(f"[WARN] Backtest failed for params {params}: {e}")
        return {}


def calculate_score(metrics: Dict[str, Any], scoring: str = 'sharpe') -> float:
    """
    Calculate optimization score from metrics.

    Scoring options:
    - sharpe: Sharpe ratio (higher is better)
    - profit_factor: Profit factor (higher is better)
    - win_rate: Win rate (higher is better)
    - combined: Weighted combination
    """
    if not metrics:
        return float('-inf')

    if scoring == 'sharpe':
        return float(metrics.get('sharpe', float('-inf')))
    elif scoring == 'profit_factor':
        pf = metrics.get('profit_factor', 0)
        return float(pf) if pf != float('inf') else 10.0
    elif scoring == 'win_rate':
        return float(metrics.get('win_rate', 0))
    elif scoring == 'combined':
        # Weighted combination
        sharpe = float(metrics.get('sharpe', 0))
        win_rate = float(metrics.get('win_rate', 0))
        pf = metrics.get('profit_factor', 0)
        pf = float(pf) if pf != float('inf') else 10.0
        max_dd = abs(float(metrics.get('max_drawdown', 0)))

        # Score formula: prioritize risk-adjusted returns
        score = sharpe * 0.4 + win_rate * 0.2 + min(pf / 2, 2) * 0.2 - max_dd * 0.2
        return score
    else:
        return float(metrics.get(scoring, float('-inf')))


def format_results_table(results: List[OptimizationResult], top_n: int = 20) -> str:
    """Format optimization results as text table."""
    if not results:
        return "[WARN] No results to display"

    lines = []
    lines.append("")
    lines.append("=" * 100)
    lines.append("PARAMETER OPTIMIZATION RESULTS")
    lines.append("=" * 100)
    lines.append("")

    # Sort by score descending
    sorted_results = sorted(results, key=lambda x: x.score, reverse=True)[:top_n]

    # Build header based on first result's params
    param_names = list(sorted_results[0].params.keys()) if sorted_results else []

    # Table header
    header_parts = ['Rank']
    header_parts.extend([p[:10] for p in param_names])  # Truncate param names
    header_parts.extend(['Sharpe', 'WinRate', 'PF', 'MaxDD', 'Score'])
    if sorted_results[0].test_metrics:
        header_parts.extend(['T.Sharpe', 'T.WR', 'T.PF'])

    header = ' | '.join([f'{h:>8}' for h in header_parts])
    lines.append(header)
    lines.append('-' * len(header))

    # Table rows
    for rank, res in enumerate(sorted_results, 1):
        row_parts = [str(rank)]

        # Parameter values
        for p in param_names:
            val = res.params.get(p, '')
            if isinstance(val, float):
                row_parts.append(f'{val:.2f}')
            else:
                row_parts.append(str(val)[:10])

        # Train metrics
        tm = res.train_metrics
        row_parts.append(f"{tm.get('sharpe', 0):.2f}")
        row_parts.append(f"{tm.get('win_rate', 0)*100:.1f}%")
        pf = tm.get('profit_factor', 0)
        pf_str = f"{pf:.2f}" if pf != float('inf') else 'inf'
        row_parts.append(pf_str)
        row_parts.append(f"{abs(tm.get('max_drawdown', 0))*100:.1f}%")
        row_parts.append(f"{res.score:.3f}")

        # Test metrics (if available)
        if res.test_metrics:
            ttm = res.test_metrics
            row_parts.append(f"{ttm.get('sharpe', 0):.2f}")
            row_parts.append(f"{ttm.get('win_rate', 0)*100:.1f}%")
            tpf = ttm.get('profit_factor', 0)
            tpf_str = f"{tpf:.2f}" if tpf != float('inf') else 'inf'
            row_parts.append(tpf_str)

        row = ' | '.join([f'{v:>8}' for v in row_parts])
        lines.append(row)

    lines.append("")
    lines.append("=" * 100)

    # Best parameters summary
    if sorted_results:
        best = sorted_results[0]
        lines.append("")
        lines.append("BEST PARAMETERS")
        lines.append("-" * 40)
        for k, v in best.params.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append("BEST TRAIN METRICS")
        lines.append("-" * 40)
        for k, v in best.train_metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        if best.test_metrics:
            lines.append("")
            lines.append("BEST TEST METRICS (out-of-sample)")
            lines.append("-" * 40)
            for k, v in best.test_metrics.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")

    lines.append("")
    return '\n'.join(lines)


def save_results(results: List[OptimizationResult], outdir: Path, strategy: str) -> None:
    """Save optimization results to files."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    rows = []
    for res in results:
        row = dict(res.params)
        for k, v in res.train_metrics.items():
            row[f'train_{k}'] = v
        if res.test_metrics:
            for k, v in res.test_metrics.items():
                row[f'test_{k}'] = v
        row['score'] = res.score
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('score', ascending=False)
    df.to_csv(outdir / f'{strategy}_optimization_results.csv', index=False)

    # Save best parameters as JSON
    if results:
        best = sorted(results, key=lambda x: x.score, reverse=True)[0]
        best_params = {
            'strategy': strategy,
            'params': best.params,
            'train_metrics': best.train_metrics,
            'test_metrics': best.test_metrics,
            'score': best.score,
        }
        with open(outdir / f'{strategy}_best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2, default=str)

    print(f"[INFO] Results saved to {outdir}")


def main():
    ap = argparse.ArgumentParser(
        description='Parameter optimization via grid search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize.py --strategy rsi2 --universe data/universe.csv --start 2020-01-01 --end 2023-12-31
  python optimize.py --strategy ibs --param-grid custom_grid.json --validate 0.3
  python optimize.py --strategy rsi2 --scoring combined --top 30
        """
    )
    ap.add_argument('--strategy', type=str, required=True, choices=['rsi2', 'ibs'],
                    help='Strategy to optimize')
    ap.add_argument('--universe', type=str, default='data/universe.csv',
                    help='Path to universe CSV file')
    ap.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    ap.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    ap.add_argument('--param-grid', type=str, default=None,
                    help='Path to parameter grid JSON file')
    ap.add_argument('--validate', type=float, default=0.0,
                    help='Validation split ratio (0-1). 0 = no validation (default)')
    ap.add_argument('--scoring', type=str, default='sharpe',
                    choices=['sharpe', 'profit_factor', 'win_rate', 'combined'],
                    help='Scoring metric for optimization')
    ap.add_argument('--initial-cash', type=float, default=100000.0,
                    help='Initial cash for backtest')
    ap.add_argument('--cap', type=int, default=100,
                    help='Max symbols to use from universe')
    ap.add_argument('--cache', type=str, default='data/cache',
                    help='Cache directory for price data')
    ap.add_argument('--outdir', type=str, default='outputs/optimization',
                    help='Output directory for results')
    ap.add_argument('--top', type=int, default=20,
                    help='Number of top results to display')
    ap.add_argument('--dotenv', type=str, default=None, help='Path to .env file')
    ap.add_argument('--quiet', action='store_true', help='Suppress progress output')
    args = ap.parse_args()

    # Load environment variables
    if args.dotenv:
        dotenv_path = Path(args.dotenv)
        if dotenv_path.exists():
            loaded = load_env(dotenv_path)
            if not args.quiet:
                print(f"[INFO] Loaded {len(loaded)} env vars from {dotenv_path}")

    # Load universe
    from data.universe.loader import load_universe
    universe_path = Path(args.universe)
    if not universe_path.exists():
        print(f"[ERROR] Universe file not found: {universe_path}")
        sys.exit(1)

    symbols = load_universe(universe_path, cap=args.cap)
    print(f"[INFO] Loaded {len(symbols)} symbols from universe")

    # Load parameter grid
    param_grid_path = Path(args.param_grid) if args.param_grid else None
    param_grid = load_param_grid(param_grid_path, args.strategy)

    if not param_grid:
        print("[ERROR] No parameter grid available")
        sys.exit(1)

    # Generate parameter combinations
    combinations = generate_param_combinations(param_grid)
    total_combos = len(combinations)
    print(f"[INFO] Testing {total_combos} parameter combinations")

    # Split data if validation requested
    if args.validate > 0:
        (train_start, train_end), (test_start, test_end) = split_date_range(
            args.start, args.end, args.validate
        )
        print(f"[INFO] Train period: {train_start} to {train_end}")
        print(f"[INFO] Test period: {test_start} to {test_end}")
    else:
        train_start, train_end = args.start, args.end
        test_start, test_end = None, None
        print(f"[INFO] Full period: {train_start} to {train_end} (no validation split)")

    # Setup data fetcher
    cache_dir = Path(args.cache)

    try:
        from data.providers.polygon_eod import fetch_daily_bars_polygon

        def make_fetcher(start: str, end: str):
            def fetcher(sym: str) -> pd.DataFrame:
                return fetch_daily_bars_polygon(sym, start, end, cache_dir=cache_dir)
            return fetcher
    except ImportError:
        print("[WARN] Polygon provider not available, using synthetic data")

        def make_fetcher(start: str, end: str):
            def fetcher(sym: str) -> pd.DataFrame:
                np.random.seed(abs(hash(sym + start + end)) % 2**32)
                days = 260
                dates = pd.date_range(end=end, periods=days, freq='B')
                rets = np.random.normal(0.0004, 0.01, days)
                close = 100 * np.cumprod(1 + rets)
                return pd.DataFrame({
                    'timestamp': dates,
                    'symbol': sym,
                    'open': close * (1 + np.random.uniform(-0.002, 0.002, days)),
                    'high': close * (1 + np.random.uniform(0, 0.01, days)),
                    'low': close * (1 - np.random.uniform(0, 0.01, days)),
                    'close': close,
                    'volume': np.random.randint(1_000_000, 5_000_000, days),
                })
            return fetcher

    # Run optimization
    results: List[OptimizationResult] = []
    train_fetcher = make_fetcher(train_start, train_end)
    test_fetcher = make_fetcher(test_start, test_end) if test_start else None

    progress_step = max(1, total_combos // 20)

    for i, params in enumerate(combinations):
        if not args.quiet and i % progress_step == 0:
            pct = int(100 * i / total_combos)
            print(f"\r[PROGRESS] Optimization: {pct}% ({i}/{total_combos})", end='', flush=True)

        # Run train backtest
        train_metrics = run_backtest_for_params(
            args.strategy, params, symbols, train_start, train_end,
            train_fetcher, args.initial_cash
        )

        # Run test backtest if validation enabled
        test_metrics = None
        if test_fetcher and test_start:
            test_metrics = run_backtest_for_params(
                args.strategy, params, symbols, test_start, test_end,
                test_fetcher, args.initial_cash
            )

        # Calculate score
        score = calculate_score(train_metrics, args.scoring)

        results.append(OptimizationResult(
            params=params,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            score=score,
        ))

    if not args.quiet:
        print(f"\r[PROGRESS] Optimization: 100% ({total_combos}/{total_combos})")

    # Display results
    print(format_results_table(results, args.top))

    # Overfitting analysis (if validation enabled)
    if args.validate > 0 and results:
        lines = []
        lines.append("")
        lines.append("OVERFITTING ANALYSIS")
        lines.append("=" * 60)

        # Get best by train score
        best_train = sorted(results, key=lambda x: x.score, reverse=True)[0]
        train_sharpe = best_train.train_metrics.get('sharpe', 0)
        test_sharpe = best_train.test_metrics.get('sharpe', 0) if best_train.test_metrics else 0

        lines.append(f"Best Train Sharpe: {train_sharpe:.3f}")
        lines.append(f"Test Sharpe: {test_sharpe:.3f}")

        if train_sharpe > 0:
            degradation = (train_sharpe - test_sharpe) / train_sharpe * 100
            lines.append(f"Performance Degradation: {degradation:.1f}%")

            if degradation > 50:
                lines.append("[WARNING] Significant overfitting detected (>50% degradation)")
            elif degradation > 25:
                lines.append("[CAUTION] Moderate overfitting detected (25-50% degradation)")
            else:
                lines.append("[OK] Results appear robust (<25% degradation)")

        # Rank correlation between train and test scores
        if all(r.test_metrics for r in results):
            train_scores = [calculate_score(r.train_metrics, args.scoring) for r in results]
            test_scores = [calculate_score(r.test_metrics, args.scoring) for r in results]

            # Spearman rank correlation
            from scipy import stats
            try:
                corr, pval = stats.spearmanr(train_scores, test_scores)
                lines.append(f"\nRank Correlation (train vs test): {corr:.3f}")
                if corr > 0.7:
                    lines.append("[OK] Strong correlation suggests stable rankings")
                elif corr > 0.4:
                    lines.append("[CAUTION] Moderate correlation - some instability")
                else:
                    lines.append("[WARNING] Weak correlation - rankings unstable")
            except ImportError:
                pass

        lines.append("")
        print('\n'.join(lines))

    # Save results
    save_results(results, Path(args.outdir), args.strategy)


if __name__ == '__main__':
    main()
