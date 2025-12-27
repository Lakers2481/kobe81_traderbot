#!/usr/bin/env python3
"""
Benchmark Comparison Tool for Kobe Trading System.

Compares strategy performance against SPY benchmark.
Calculates alpha, beta, correlation, and relative performance.

Usage:
    python benchmark.py --wfdir wf_outputs/donchian --start 2021-01-01 --end 2022-12-31
    python benchmark.py --equity equity_curve.csv --dotenv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BenchmarkMetrics:
    """Container for benchmark comparison metrics."""
    # Strategy metrics
    strategy_total_return: float = 0.0
    strategy_cagr: float = 0.0
    strategy_volatility: float = 0.0
    strategy_sharpe: float = 0.0
    strategy_max_drawdown: float = 0.0

    # Benchmark metrics
    benchmark_total_return: float = 0.0
    benchmark_cagr: float = 0.0
    benchmark_volatility: float = 0.0
    benchmark_sharpe: float = 0.0
    benchmark_max_drawdown: float = 0.0

    # Relative metrics
    alpha: float = 0.0
    beta: float = 0.0
    correlation: float = 0.0
    r_squared: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    excess_return: float = 0.0

    # Period data
    start_date: str = ""
    end_date: str = ""
    trading_days: int = 0


def load_env(dotenv_path: Optional[str] = None) -> None:
    """Load environment variables from .env file."""
    if dotenv_path is None:
        dotenv_path = Path(__file__).parent.parent / '.env'
    else:
        dotenv_path = Path(dotenv_path)

    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, val = line.split('=', 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ[key] = val


def fetch_spy_data(
    start_date: datetime,
    end_date: datetime,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch SPY data from Polygon.io or generate synthetic data for demo.

    In production, this would fetch real data from Polygon.io API.
    For testing/demo, generates synthetic SPY-like returns.
    """
    api_key = api_key or os.environ.get('POLYGON_API_KEY')

    if api_key:
        try:
            return _fetch_spy_polygon(start_date, end_date, api_key)
        except Exception as e:
            print(f"Warning: Failed to fetch SPY from Polygon: {e}")
            print("Falling back to synthetic data...")

    # Generate synthetic SPY data for demo/testing
    return _generate_synthetic_spy(start_date, end_date)


def _fetch_spy_polygon(
    start_date: datetime,
    end_date: datetime,
    api_key: str
) -> pd.DataFrame:
    """Fetch real SPY data from Polygon.io."""
    import urllib.request
    import urllib.error

    # Format dates
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/"
        f"{start_str}/{end_str}?adjusted=true&sort=asc&apiKey={api_key}"
    )

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Polygon API error: {e.code} {e.reason}")

    if 'results' not in data or not data['results']:
        raise RuntimeError("No data returned from Polygon API")

    records = []
    for bar in data['results']:
        records.append({
            'timestamp': pd.to_datetime(bar['t'], unit='ms'),
            'open': bar['o'],
            'high': bar['h'],
            'low': bar['l'],
            'close': bar['c'],
            'volume': bar['v'],
        })

    df = pd.DataFrame(records)
    df = df.set_index('timestamp')
    df['returns'] = df['close'].pct_change()

    return df


def _generate_synthetic_spy(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Generate synthetic SPY-like data for testing."""
    # Generate business days
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # SPY-like parameters: ~10% annual return, ~15% volatility
    np.random.seed(42)  # For reproducibility
    n_days = len(dates)
    daily_return = 0.10 / 252  # ~10% annual
    daily_vol = 0.15 / np.sqrt(252)  # ~15% annual volatility

    returns = np.random.normal(daily_return, daily_vol, n_days)

    # Start at 400 (approximate SPY level)
    prices = 400 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'close': prices,
        'returns': returns,
    }, index=dates)

    df.index.name = 'timestamp'

    return df


def load_equity_curve(path: Path) -> pd.DataFrame:
    """Load equity curve from CSV file."""
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    df.columns = df.columns.str.lower().str.strip()

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    return df


def discover_equity_files(base_dir: Path, strategy: Optional[str] = None) -> List[Path]:
    """Discover equity curve files in a directory structure."""
    files = []

    if strategy:
        strategy_dir = base_dir / strategy
        if strategy_dir.exists():
            files.extend(strategy_dir.glob('**/equity_curve.csv'))
    else:
        files.extend(base_dir.glob('**/equity_curve.csv'))

    return sorted(files)


def merge_equity_curves(equity_files: List[Path]) -> pd.DataFrame:
    """Merge multiple equity curve files."""
    all_equity = []

    for ef in equity_files:
        df = load_equity_curve(ef)
        if not df.empty and 'equity' in df.columns:
            all_equity.append(df[['equity']])

    if not all_equity:
        return pd.DataFrame()

    # Merge on index (timestamp), taking mean of equity values
    merged = pd.concat(all_equity, axis=1)
    result = pd.DataFrame({
        'equity': merged.mean(axis=1)
    })

    return result


def compute_benchmark_metrics(
    strategy_equity: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    trading_days: int = 252
) -> BenchmarkMetrics:
    """Compute benchmark comparison metrics."""
    metrics = BenchmarkMetrics()

    if strategy_equity.empty or benchmark_df.empty:
        return metrics

    # Ensure equity column
    if 'equity' not in strategy_equity.columns:
        return metrics

    # Align data on common dates
    strategy = strategy_equity['equity'].copy()
    benchmark = benchmark_df['close'].copy() if 'close' in benchmark_df.columns else benchmark_df['equity'].copy()

    # Find common index
    common_idx = strategy.index.intersection(benchmark.index)
    if len(common_idx) < 10:
        print(f"Warning: Only {len(common_idx)} overlapping dates found")
        return metrics

    strategy = strategy.loc[common_idx]
    benchmark = benchmark.loc[common_idx]

    # Calculate returns
    strategy_returns = strategy.pct_change().dropna()
    benchmark_returns = benchmark.pct_change().dropna()

    # Align returns
    common_return_idx = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.loc[common_return_idx]
    benchmark_returns = benchmark_returns.loc[common_return_idx]

    if len(strategy_returns) < 5:
        return metrics

    metrics.trading_days = len(common_return_idx)
    metrics.start_date = str(common_idx.min().date())
    metrics.end_date = str(common_idx.max().date())

    # Strategy metrics
    metrics.strategy_total_return = (strategy.iloc[-1] / strategy.iloc[0]) - 1
    strategy_std = strategy_returns.std()
    if strategy_std > 0:
        metrics.strategy_volatility = float(strategy_std * np.sqrt(trading_days))
        excess_ret = strategy_returns.mean() - (risk_free_rate / trading_days)
        metrics.strategy_sharpe = float(excess_ret / strategy_std * np.sqrt(trading_days))

    # Strategy CAGR
    n_years = len(common_idx) / trading_days
    if n_years > 0:
        metrics.strategy_cagr = (1 + metrics.strategy_total_return) ** (1 / n_years) - 1

    # Strategy max drawdown
    cummax = strategy.cummax()
    drawdown = (strategy - cummax) / cummax
    metrics.strategy_max_drawdown = float(drawdown.min())

    # Benchmark metrics
    metrics.benchmark_total_return = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
    benchmark_std = benchmark_returns.std()
    if benchmark_std > 0:
        metrics.benchmark_volatility = float(benchmark_std * np.sqrt(trading_days))
        excess_ret = benchmark_returns.mean() - (risk_free_rate / trading_days)
        metrics.benchmark_sharpe = float(excess_ret / benchmark_std * np.sqrt(trading_days))

    # Benchmark CAGR
    if n_years > 0:
        metrics.benchmark_cagr = (1 + metrics.benchmark_total_return) ** (1 / n_years) - 1

    # Benchmark max drawdown
    cummax = benchmark.cummax()
    drawdown = (benchmark - cummax) / cummax
    metrics.benchmark_max_drawdown = float(drawdown.min())

    # Relative metrics
    metrics.excess_return = metrics.strategy_total_return - metrics.benchmark_total_return

    # Correlation
    metrics.correlation = float(strategy_returns.corr(benchmark_returns))

    # Beta and Alpha (CAPM regression)
    # Strategy Return = Alpha + Beta * Benchmark Return + Epsilon
    if len(strategy_returns) > 2:
        # Using numpy for simple linear regression
        X = benchmark_returns.values
        Y = strategy_returns.values

        # Add constant for intercept
        X_mean = X.mean()
        Y_mean = Y.mean()

        # Beta = Cov(S, B) / Var(B)
        covariance = ((X - X_mean) * (Y - Y_mean)).sum()
        variance = ((X - X_mean) ** 2).sum()

        if variance > 0:
            metrics.beta = float(covariance / variance)
            metrics.alpha = float((Y_mean - metrics.beta * X_mean) * trading_days)  # Annualized

        # R-squared
        metrics.r_squared = metrics.correlation ** 2

    # Tracking Error (standard deviation of excess returns)
    excess_daily = strategy_returns - benchmark_returns
    metrics.tracking_error = float(excess_daily.std() * np.sqrt(trading_days))

    # Information Ratio
    if metrics.tracking_error > 0:
        annualized_excess = excess_daily.mean() * trading_days
        metrics.information_ratio = float(annualized_excess / metrics.tracking_error)

    return metrics


def format_comparison_table(metrics: BenchmarkMetrics) -> str:
    """Format benchmark comparison as a text table."""
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append("BENCHMARK COMPARISON REPORT (vs SPY)")
    lines.append("=" * 70)
    lines.append(f"Period: {metrics.start_date} to {metrics.end_date} ({metrics.trading_days} trading days)")
    lines.append("")

    # Side-by-side comparison
    lines.append("-" * 70)
    lines.append(f"{'Metric':<30} {'Strategy':>18} {'SPY (Benchmark)':>18}")
    lines.append("-" * 70)
    lines.append(f"{'Total Return':<30} {metrics.strategy_total_return:>17.2%} {metrics.benchmark_total_return:>17.2%}")
    lines.append(f"{'CAGR':<30} {metrics.strategy_cagr:>17.2%} {metrics.benchmark_cagr:>17.2%}")
    lines.append(f"{'Volatility (Annual)':<30} {metrics.strategy_volatility:>17.2%} {metrics.benchmark_volatility:>17.2%}")
    lines.append(f"{'Sharpe Ratio':<30} {metrics.strategy_sharpe:>18.3f} {metrics.benchmark_sharpe:>18.3f}")
    lines.append(f"{'Max Drawdown':<30} {metrics.strategy_max_drawdown:>17.2%} {metrics.benchmark_max_drawdown:>17.2%}")
    lines.append("")

    # Relative Metrics
    lines.append("-" * 70)
    lines.append("RELATIVE METRICS")
    lines.append("-" * 70)
    lines.append(f"{'Excess Return:':<30} {metrics.excess_return:>17.2%}")
    lines.append(f"{'Alpha (annualized):':<30} {metrics.alpha:>17.2%}")
    lines.append(f"{'Beta:':<30} {metrics.beta:>18.3f}")
    lines.append(f"{'Correlation:':<30} {metrics.correlation:>18.3f}")
    lines.append(f"{'R-Squared:':<30} {metrics.r_squared:>18.3f}")
    lines.append(f"{'Tracking Error:':<30} {metrics.tracking_error:>17.2%}")
    lines.append(f"{'Information Ratio:':<30} {metrics.information_ratio:>18.3f}")
    lines.append("")

    # Interpretation
    lines.append("-" * 70)
    lines.append("INTERPRETATION")
    lines.append("-" * 70)

    if metrics.alpha > 0.02:
        lines.append("  [+] Strong positive alpha - strategy adds significant value")
    elif metrics.alpha > 0:
        lines.append("  [+] Positive alpha - strategy outperforms risk-adjusted benchmark")
    elif metrics.alpha > -0.02:
        lines.append("  [-] Slight negative alpha - strategy slightly underperforms")
    else:
        lines.append("  [-] Negative alpha - strategy underperforms risk-adjusted benchmark")

    if metrics.beta < 0.8:
        lines.append("  [*] Low beta - strategy is defensive, less market exposure")
    elif metrics.beta > 1.2:
        lines.append("  [*] High beta - strategy is aggressive, amplified market exposure")
    else:
        lines.append("  [*] Beta near 1.0 - strategy tracks market exposure")

    if metrics.correlation > 0.8:
        lines.append("  [*] High correlation - strategy moves with the market")
    elif metrics.correlation < 0.3:
        lines.append("  [*] Low correlation - strategy provides diversification")

    if metrics.information_ratio > 0.5:
        lines.append("  [+] Good Information Ratio - efficient use of tracking error")
    elif metrics.information_ratio < 0:
        lines.append("  [-] Negative Information Ratio - tracking error not compensated")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def generate_chart_data(
    strategy_equity: pd.DataFrame,
    benchmark_df: pd.DataFrame
) -> Dict[str, Any]:
    """Generate data for performance charts."""
    if strategy_equity.empty or benchmark_df.empty:
        return {}

    # Normalize both to start at 100
    strategy = strategy_equity['equity'].copy()
    benchmark = benchmark_df['close'].copy() if 'close' in benchmark_df.columns else benchmark_df['equity'].copy()

    # Align on common dates
    common_idx = strategy.index.intersection(benchmark.index)
    if len(common_idx) < 2:
        return {}

    strategy = strategy.loc[common_idx]
    benchmark = benchmark.loc[common_idx]

    # Normalize
    strategy_norm = (strategy / strategy.iloc[0]) * 100
    benchmark_norm = (benchmark / benchmark.iloc[0]) * 100

    # Calculate cumulative excess return
    excess_return = strategy_norm - benchmark_norm

    # Rolling metrics (30-day)
    strategy_returns = strategy.pct_change()
    benchmark_returns = benchmark.pct_change()

    rolling_corr = strategy_returns.rolling(30).corr(benchmark_returns)
    rolling_beta = strategy_returns.rolling(30).cov(benchmark_returns) / benchmark_returns.rolling(30).var()

    return {
        'dates': [str(d.date()) for d in common_idx],
        'strategy_normalized': strategy_norm.tolist(),
        'benchmark_normalized': benchmark_norm.tolist(),
        'excess_return': excess_return.tolist(),
        'rolling_correlation': rolling_corr.fillna(0).tolist(),
        'rolling_beta': rolling_beta.fillna(1).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare strategy performance against SPY benchmark'
    )
    parser.add_argument(
        '--wfdir',
        type=str,
        default='wf_outputs',
        help='Directory containing WF outputs (default: wf_outputs)'
    )
    parser.add_argument(
        '--equity',
        type=str,
        help='Path to specific equity curve CSV file'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        help='Filter by strategy name (e.g., donchian, TURTLE_SOUP)'
    )
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--dotenv',
        action='store_true',
        help='Load environment variables from .env file'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output metrics as JSON'
    )
    parser.add_argument(
        '--chart-data',
        action='store_true',
        help='Include chart data in JSON output'
    )
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=0.0,
        help='Annual risk-free rate (default: 0.0)'
    )

    args = parser.parse_args()

    if args.dotenv:
        load_env()

    # Determine base directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    # Load strategy equity curve
    if args.equity:
        equity_path = Path(args.equity)
        if not equity_path.is_absolute():
            equity_path = project_dir / equity_path
        strategy_equity = load_equity_curve(equity_path)
    else:
        wf_dir = Path(args.wfdir)
        if not wf_dir.is_absolute():
            wf_dir = project_dir / wf_dir

        equity_files = discover_equity_files(wf_dir, args.strategy)
        if not equity_files:
            print(f"No equity curve files found in {wf_dir}")
            sys.exit(1)

        strategy_equity = merge_equity_curves(equity_files)

    if strategy_equity.empty:
        print("No strategy equity data loaded")
        sys.exit(1)

    # Determine date range
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    else:
        start_date = strategy_equity.index.min()
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.to_pydatetime()

    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    else:
        end_date = strategy_equity.index.max()
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.to_pydatetime()

    # Filter strategy equity by date range
    strategy_equity = strategy_equity.loc[
        (strategy_equity.index >= start_date) &
        (strategy_equity.index <= end_date)
    ]

    # Fetch benchmark data
    print(f"Fetching SPY data from {start_date.date()} to {end_date.date()}...")
    benchmark_df = fetch_spy_data(start_date, end_date)

    if benchmark_df.empty:
        print("Failed to fetch benchmark data")
        sys.exit(1)

    # Compute metrics
    metrics = compute_benchmark_metrics(
        strategy_equity,
        benchmark_df,
        args.risk_free_rate
    )

    if args.json:
        output = {
            'period': {
                'start_date': metrics.start_date,
                'end_date': metrics.end_date,
                'trading_days': metrics.trading_days,
            },
            'strategy': {
                'total_return': metrics.strategy_total_return,
                'cagr': metrics.strategy_cagr,
                'volatility': metrics.strategy_volatility,
                'sharpe': metrics.strategy_sharpe,
                'max_drawdown': metrics.strategy_max_drawdown,
            },
            'benchmark': {
                'total_return': metrics.benchmark_total_return,
                'cagr': metrics.benchmark_cagr,
                'volatility': metrics.benchmark_volatility,
                'sharpe': metrics.benchmark_sharpe,
                'max_drawdown': metrics.benchmark_max_drawdown,
            },
            'relative': {
                'alpha': metrics.alpha,
                'beta': metrics.beta,
                'correlation': metrics.correlation,
                'r_squared': metrics.r_squared,
                'tracking_error': metrics.tracking_error,
                'information_ratio': metrics.information_ratio,
                'excess_return': metrics.excess_return,
            }
        }

        if args.chart_data:
            output['chart_data'] = generate_chart_data(strategy_equity, benchmark_df)

        print(json.dumps(output, indent=2))
    else:
        print(format_comparison_table(metrics))


if __name__ == '__main__':
    main()

