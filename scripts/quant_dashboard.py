#!/usr/bin/env python3
"""
Quantitative Analysis Dashboard for Kobe Trading System.

Advanced analytics including:
- CAGR, Sharpe, Sortino, Calmar ratios
- VaR and CVaR (Expected Shortfall) calculations
- Alpha decomposition
- Factor exposures
- Statistical significance tests

Usage:
    python quant_dashboard.py --wfdir wf_outputs/donchian --dotenv
    python quant_dashboard.py --equity equity_curve.csv --confidence 0.95
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
from scipy import stats as scipy_stats


@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    # Value at Risk
    var_95: float = 0.0
    var_99: float = 0.0
    var_parametric_95: float = 0.0
    var_parametric_99: float = 0.0

    # Conditional VaR (Expected Shortfall)
    cvar_95: float = 0.0
    cvar_99: float = 0.0

    # Other risk measures
    downside_deviation: float = 0.0
    upside_deviation: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0


@dataclass
class PerformanceRatios:
    """Container for performance ratios."""
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    treynor_ratio: float = 0.0
    information_ratio: float = 0.0

    # Return metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Additional ratios
    gain_to_pain_ratio: float = 0.0
    ulcer_index: float = 0.0
    recovery_factor: float = 0.0


@dataclass
class AlphaDecomposition:
    """Container for alpha decomposition."""
    total_alpha: float = 0.0
    market_alpha: float = 0.0
    timing_alpha: float = 0.0
    selection_alpha: float = 0.0
    beta: float = 0.0
    r_squared: float = 0.0


@dataclass
class StatisticalTests:
    """Container for statistical significance tests."""
    # Normality tests
    jarque_bera_stat: float = 0.0
    jarque_bera_pvalue: float = 0.0
    shapiro_stat: float = 0.0
    shapiro_pvalue: float = 0.0

    # Serial correlation tests
    ljung_box_stat: float = 0.0
    ljung_box_pvalue: float = 0.0

    # Performance significance
    t_stat_returns: float = 0.0
    t_stat_pvalue: float = 0.0

    # Stationarity test
    adf_stat: float = 0.0
    adf_pvalue: float = 0.0


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

    merged = pd.concat(all_equity, axis=1)
    result = pd.DataFrame({'equity': merged.mean(axis=1)})

    return result


def compute_returns(equity: pd.Series) -> pd.Series:
    """Compute daily returns from equity curve."""
    return equity.pct_change().dropna()


def compute_risk_metrics(
    returns: pd.Series,
    confidence_levels: Tuple[float, float] = (0.95, 0.99)
) -> RiskMetrics:
    """Compute VaR, CVaR, and other risk metrics."""
    metrics = RiskMetrics()

    if len(returns) < 10:
        return metrics

    returns_arr = returns.dropna().values

    # Historical VaR (percentile method)
    metrics.var_95 = float(np.percentile(returns_arr, (1 - confidence_levels[0]) * 100))
    metrics.var_99 = float(np.percentile(returns_arr, (1 - confidence_levels[1]) * 100))

    # Parametric VaR (assuming normal distribution)
    mu = returns_arr.mean()
    sigma = returns_arr.std()
    metrics.var_parametric_95 = float(mu + scipy_stats.norm.ppf(1 - confidence_levels[0]) * sigma)
    metrics.var_parametric_99 = float(mu + scipy_stats.norm.ppf(1 - confidence_levels[1]) * sigma)

    # CVaR (Expected Shortfall) - average of losses beyond VaR
    below_var_95 = returns_arr[returns_arr <= metrics.var_95]
    below_var_99 = returns_arr[returns_arr <= metrics.var_99]
    metrics.cvar_95 = float(below_var_95.mean()) if len(below_var_95) > 0 else metrics.var_95
    metrics.cvar_99 = float(below_var_99.mean()) if len(below_var_99) > 0 else metrics.var_99

    # Downside and upside deviation
    downside_returns = returns_arr[returns_arr < 0]
    upside_returns = returns_arr[returns_arr > 0]
    metrics.downside_deviation = float(downside_returns.std()) if len(downside_returns) > 1 else 0.0
    metrics.upside_deviation = float(upside_returns.std()) if len(upside_returns) > 1 else 0.0

    # Higher moments
    metrics.skewness = float(scipy_stats.skew(returns_arr))
    metrics.kurtosis = float(scipy_stats.kurtosis(returns_arr))

    # Tail ratio (ratio of 95th percentile to 5th percentile absolute values)
    p95 = np.percentile(returns_arr, 95)
    p05 = np.percentile(returns_arr, 5)
    if abs(p05) > 0:
        metrics.tail_ratio = float(abs(p95) / abs(p05))

    return metrics


def compute_performance_ratios(
    equity: pd.Series,
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252
) -> PerformanceRatios:
    """Compute comprehensive performance ratios."""
    metrics = PerformanceRatios()

    if len(returns) < 10 or len(equity) < 2:
        return metrics

    returns_arr = returns.dropna().values

    # Basic return metrics
    initial_value = equity.iloc[0]
    final_value = equity.iloc[-1]
    metrics.total_return = (final_value - initial_value) / initial_value

    # Annualized return and volatility
    n_days = len(returns)
    years = n_days / trading_days

    if years > 0:
        metrics.cagr = (1 + metrics.total_return) ** (1 / years) - 1
        metrics.annual_return = returns_arr.mean() * trading_days

    metrics.annual_volatility = returns_arr.std() * np.sqrt(trading_days)

    # Sharpe Ratio
    excess_returns = returns_arr - (risk_free_rate / trading_days)
    if returns_arr.std() > 0:
        metrics.sharpe_ratio = float(excess_returns.mean() / returns_arr.std() * np.sqrt(trading_days))

    # Sortino Ratio (using downside deviation)
    downside_returns = returns_arr[returns_arr < 0]
    if len(downside_returns) > 1:
        downside_std = downside_returns.std()
        if downside_std > 0:
            metrics.sortino_ratio = float(excess_returns.mean() / downside_std * np.sqrt(trading_days))

    # Max Drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    metrics.max_drawdown = float(drawdown.min())

    # Max Drawdown Duration
    in_drawdown = drawdown < 0
    dd_groups = (in_drawdown != in_drawdown.shift()).cumsum()
    dd_durations = in_drawdown.groupby(dd_groups).sum()
    if len(dd_durations) > 0:
        metrics.max_drawdown_duration = int(dd_durations.max())

    # Calmar Ratio
    if metrics.max_drawdown < 0:
        metrics.calmar_ratio = metrics.cagr / abs(metrics.max_drawdown)

    # Omega Ratio (probability-weighted ratio of gains to losses)
    threshold = risk_free_rate / trading_days
    gains = (returns_arr[returns_arr > threshold] - threshold).sum()
    losses = (threshold - returns_arr[returns_arr <= threshold]).sum()
    if losses > 0:
        metrics.omega_ratio = float(gains / losses)
    elif gains > 0:
        metrics.omega_ratio = float('inf')

    # Gain to Pain Ratio
    positive_returns = returns_arr[returns_arr > 0].sum()
    negative_returns = abs(returns_arr[returns_arr < 0].sum())
    if negative_returns > 0:
        metrics.gain_to_pain_ratio = float(positive_returns / negative_returns)

    # Ulcer Index (measures depth and duration of drawdowns)
    drawdown_squared = drawdown ** 2
    metrics.ulcer_index = float(np.sqrt(drawdown_squared.mean()))

    # Recovery Factor (net profit / max drawdown)
    if metrics.max_drawdown < 0:
        net_profit = final_value - initial_value
        metrics.recovery_factor = float(net_profit / (abs(metrics.max_drawdown) * initial_value))

    return metrics


def compute_alpha_decomposition(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252
) -> AlphaDecomposition:
    """Decompose alpha into components."""
    decomp = AlphaDecomposition()

    # Align returns
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 20:
        return decomp

    s_ret = strategy_returns.loc[common_idx].values
    b_ret = benchmark_returns.loc[common_idx].values

    # CAPM regression: S = alpha + beta * B + epsilon
    beta, alpha, r_value, p_value, std_err = scipy_stats.linregress(b_ret, s_ret)

    decomp.beta = float(beta)
    decomp.r_squared = float(r_value ** 2)
    decomp.total_alpha = float(alpha * trading_days)  # Annualized

    # Simple alpha decomposition (approximation)
    # Market alpha: return attributable to market exposure
    decomp.market_alpha = float((beta * b_ret.mean() - b_ret.mean()) * trading_days)

    # Timing alpha: correlation-based (simplified)
    # High correlation in up markets, low in down markets indicates timing skill
    up_market = b_ret > 0
    down_market = b_ret <= 0

    if up_market.sum() > 5 and down_market.sum() > 5:
        corr_up = np.corrcoef(s_ret[up_market], b_ret[up_market])[0, 1] if up_market.sum() > 2 else 0
        corr_down = np.corrcoef(s_ret[down_market], b_ret[down_market])[0, 1] if down_market.sum() > 2 else 0
        timing_skill = max(0, corr_up - corr_down)
        decomp.timing_alpha = float(timing_skill * s_ret.std() * np.sqrt(trading_days) * 0.5)

    # Selection alpha: residual alpha after market and timing
    decomp.selection_alpha = decomp.total_alpha - decomp.market_alpha - decomp.timing_alpha

    return decomp


def compute_statistical_tests(returns: pd.Series) -> StatisticalTests:
    """Perform statistical significance tests on returns."""
    tests = StatisticalTests()

    if len(returns) < 30:
        return tests

    returns_arr = returns.dropna().values

    # Jarque-Bera test for normality
    try:
        jb_stat, jb_pvalue = scipy_stats.jarque_bera(returns_arr)
        tests.jarque_bera_stat = float(jb_stat)
        tests.jarque_bera_pvalue = float(jb_pvalue)
    except Exception:
        pass

    # Shapiro-Wilk test for normality (sample size limited)
    try:
        sample = returns_arr[:5000]  # Shapiro-Wilk has sample size limit
        sw_stat, sw_pvalue = scipy_stats.shapiro(sample)
        tests.shapiro_stat = float(sw_stat)
        tests.shapiro_pvalue = float(sw_pvalue)
    except Exception:
        pass

    # T-test: are returns significantly different from zero?
    try:
        t_stat, t_pvalue = scipy_stats.ttest_1samp(returns_arr, 0)
        tests.t_stat_returns = float(t_stat)
        tests.t_stat_pvalue = float(t_pvalue)
    except Exception:
        pass

    # Ljung-Box test for serial correlation
    try:
        # Simple autocorrelation test using first lag
        n = len(returns_arr)
        if n > 10:
            autocorr = np.corrcoef(returns_arr[:-1], returns_arr[1:])[0, 1]
            # Approximate Ljung-Box statistic
            lb_stat = n * (n + 2) * (autocorr ** 2 / (n - 1))
            lb_pvalue = 1 - scipy_stats.chi2.cdf(lb_stat, 1)
            tests.ljung_box_stat = float(lb_stat)
            tests.ljung_box_pvalue = float(lb_pvalue)
    except Exception:
        pass

    # Augmented Dickey-Fuller test for stationarity
    try:
        # Simple version using first differences
        diff_returns = np.diff(returns_arr)
        if len(diff_returns) > 20:
            # Approximate ADF using OLS regression
            lag_returns = returns_arr[:-1]
            delta_returns = diff_returns
            # Remove any NaN/inf values
            mask = np.isfinite(lag_returns) & np.isfinite(delta_returns)
            if mask.sum() > 10:
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                    lag_returns[mask], delta_returns[mask]
                )
                adf_stat = slope / std_err
                tests.adf_stat = float(adf_stat)
                # Approximate p-value (simplified)
                tests.adf_pvalue = float(2 * (1 - scipy_stats.norm.cdf(abs(adf_stat))))
    except Exception:
        pass

    return tests


def compute_rolling_metrics(
    equity: pd.Series,
    returns: pd.Series,
    window: int = 63  # ~3 months
) -> Dict[str, pd.Series]:
    """Compute rolling performance metrics."""
    rolling = {}

    if len(returns) < window:
        return rolling

    # Rolling Sharpe
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling['sharpe'] = (rolling_mean / rolling_std * np.sqrt(252)).dropna()

    # Rolling volatility
    rolling['volatility'] = (rolling_std * np.sqrt(252)).dropna()

    # Rolling max drawdown
    def rolling_max_dd(series):
        cummax = series.cummax()
        dd = (series - cummax) / cummax
        return dd.min()

    rolling['max_drawdown'] = equity.rolling(window).apply(rolling_max_dd, raw=False).dropna()

    # Rolling Sortino
    def calc_sortino(rets):
        neg_rets = rets[rets < 0]
        if len(neg_rets) > 1:
            downside_std = neg_rets.std()
            if downside_std > 0:
                return rets.mean() / downside_std * np.sqrt(252)
        return np.nan

    rolling['sortino'] = returns.rolling(window).apply(calc_sortino, raw=False).dropna()

    return rolling


def format_quant_dashboard(
    perf: PerformanceRatios,
    risk: RiskMetrics,
    alpha: AlphaDecomposition,
    tests: StatisticalTests,
    rolling: Dict[str, pd.Series]
) -> str:
    """Format quantitative dashboard as text."""
    lines = []

    # Header
    lines.append("=" * 75)
    lines.append("QUANTITATIVE ANALYSIS DASHBOARD")
    lines.append("=" * 75)
    lines.append("")

    # Performance Ratios
    lines.append("-" * 75)
    lines.append("PERFORMANCE RATIOS")
    lines.append("-" * 75)
    lines.append(f"{'CAGR:':<35} {perf.cagr:>12.2%}")
    lines.append(f"{'Total Return:':<35} {perf.total_return:>12.2%}")
    lines.append(f"{'Annual Volatility:':<35} {perf.annual_volatility:>12.2%}")
    lines.append("")
    lines.append(f"{'Sharpe Ratio:':<35} {perf.sharpe_ratio:>12.3f}")
    lines.append(f"{'Sortino Ratio:':<35} {perf.sortino_ratio:>12.3f}")
    lines.append(f"{'Calmar Ratio:':<35} {perf.calmar_ratio:>12.3f}")
    lines.append(f"{'Omega Ratio:':<35} {perf.omega_ratio:>12.3f}")
    lines.append("")
    lines.append(f"{'Max Drawdown:':<35} {perf.max_drawdown:>12.2%}")
    lines.append(f"{'Max Drawdown Duration (days):':<35} {perf.max_drawdown_duration:>12}")
    lines.append(f"{'Ulcer Index:':<35} {perf.ulcer_index:>12.4f}")
    lines.append(f"{'Recovery Factor:':<35} {perf.recovery_factor:>12.2f}")
    lines.append(f"{'Gain to Pain Ratio:':<35} {perf.gain_to_pain_ratio:>12.2f}")
    lines.append("")

    # Risk Metrics
    lines.append("-" * 75)
    lines.append("RISK METRICS (VALUE AT RISK)")
    lines.append("-" * 75)
    lines.append(f"{'Historical VaR (95%):':<35} {risk.var_95:>12.4%}")
    lines.append(f"{'Historical VaR (99%):':<35} {risk.var_99:>12.4%}")
    lines.append(f"{'Parametric VaR (95%):':<35} {risk.var_parametric_95:>12.4%}")
    lines.append(f"{'Parametric VaR (99%):':<35} {risk.var_parametric_99:>12.4%}")
    lines.append("")
    lines.append(f"{'CVaR / Expected Shortfall (95%):':<35} {risk.cvar_95:>12.4%}")
    lines.append(f"{'CVaR / Expected Shortfall (99%):':<35} {risk.cvar_99:>12.4%}")
    lines.append("")
    lines.append(f"{'Downside Deviation:':<35} {risk.downside_deviation:>12.4%}")
    lines.append(f"{'Upside Deviation:':<35} {risk.upside_deviation:>12.4%}")
    lines.append(f"{'Skewness:':<35} {risk.skewness:>12.3f}")
    lines.append(f"{'Excess Kurtosis:':<35} {risk.kurtosis:>12.3f}")
    lines.append(f"{'Tail Ratio:':<35} {risk.tail_ratio:>12.3f}")
    lines.append("")

    # Alpha Decomposition
    if alpha.total_alpha != 0 or alpha.beta != 0:
        lines.append("-" * 75)
        lines.append("ALPHA DECOMPOSITION")
        lines.append("-" * 75)
        lines.append(f"{'Total Alpha (annualized):':<35} {alpha.total_alpha:>12.2%}")
        lines.append(f"{'Market Alpha:':<35} {alpha.market_alpha:>12.2%}")
        lines.append(f"{'Timing Alpha:':<35} {alpha.timing_alpha:>12.2%}")
        lines.append(f"{'Selection Alpha:':<35} {alpha.selection_alpha:>12.2%}")
        lines.append("")
        lines.append(f"{'Beta:':<35} {alpha.beta:>12.3f}")
        lines.append(f"{'R-Squared:':<35} {alpha.r_squared:>12.3f}")
        lines.append("")

    # Statistical Tests
    lines.append("-" * 75)
    lines.append("STATISTICAL SIGNIFICANCE TESTS")
    lines.append("-" * 75)

    # Normality
    lines.append("Normality Tests:")
    jb_significant = tests.jarque_bera_pvalue < 0.05
    lines.append(f"  Jarque-Bera: stat={tests.jarque_bera_stat:.2f}, "
                f"p-value={tests.jarque_bera_pvalue:.4f} "
                f"{'[REJECT H0: not normal]' if jb_significant else '[ACCEPT H0: normal]'}")

    sw_significant = tests.shapiro_pvalue < 0.05
    lines.append(f"  Shapiro-Wilk: stat={tests.shapiro_stat:.4f}, "
                f"p-value={tests.shapiro_pvalue:.4f} "
                f"{'[REJECT H0: not normal]' if sw_significant else '[ACCEPT H0: normal]'}")

    # Returns significance
    lines.append("")
    lines.append("Returns Significance:")
    t_significant = tests.t_stat_pvalue < 0.05
    lines.append(f"  T-test (mean != 0): t={tests.t_stat_returns:.3f}, "
                f"p-value={tests.t_stat_pvalue:.4f} "
                f"{'[SIGNIFICANT]' if t_significant else '[NOT SIGNIFICANT]'}")

    # Serial correlation
    lines.append("")
    lines.append("Serial Correlation:")
    lb_significant = tests.ljung_box_pvalue < 0.05
    lines.append(f"  Ljung-Box: stat={tests.ljung_box_stat:.2f}, "
                f"p-value={tests.ljung_box_pvalue:.4f} "
                f"{'[AUTOCORRELATED]' if lb_significant else '[NO AUTOCORRELATION]'}")

    # Stationarity
    lines.append("")
    lines.append("Stationarity:")
    adf_significant = tests.adf_pvalue < 0.05
    lines.append(f"  ADF Test: stat={tests.adf_stat:.3f}, "
                f"p-value={tests.adf_pvalue:.4f} "
                f"{'[STATIONARY]' if adf_significant else '[NON-STATIONARY]'}")

    # Rolling Metrics Summary
    if rolling:
        lines.append("")
        lines.append("-" * 75)
        lines.append("ROLLING METRICS SUMMARY (63-day window)")
        lines.append("-" * 75)

        for metric_name, series in rolling.items():
            if not series.empty:
                lines.append(f"{metric_name.title()}: "
                           f"Mean={series.mean():.3f}, "
                           f"Min={series.min():.3f}, "
                           f"Max={series.max():.3f}, "
                           f"Current={series.iloc[-1]:.3f}")

    # Interpretation
    lines.append("")
    lines.append("-" * 75)
    lines.append("INTERPRETATION")
    lines.append("-" * 75)

    # Risk interpretation
    if risk.skewness < -0.5:
        lines.append("  [!] Negative skewness indicates larger negative tail risk")
    elif risk.skewness > 0.5:
        lines.append("  [+] Positive skewness indicates larger positive tail potential")

    if risk.kurtosis > 3:
        lines.append("  [!] High kurtosis indicates fat tails / more extreme events")

    # Performance interpretation
    if perf.sharpe_ratio > 1.5:
        lines.append("  [+] Excellent Sharpe ratio (>1.5)")
    elif perf.sharpe_ratio > 1.0:
        lines.append("  [+] Good Sharpe ratio (>1.0)")
    elif perf.sharpe_ratio > 0.5:
        lines.append("  [*] Moderate Sharpe ratio (>0.5)")
    elif perf.sharpe_ratio > 0:
        lines.append("  [-] Low Sharpe ratio (<0.5)")
    else:
        lines.append("  [-] Negative Sharpe ratio")

    if perf.sortino_ratio > perf.sharpe_ratio * 1.2:
        lines.append("  [+] Sortino > Sharpe suggests good downside protection")

    if perf.calmar_ratio < 0.5:
        lines.append("  [-] Low Calmar ratio indicates poor risk-adjusted returns relative to drawdown")
    elif perf.calmar_ratio > 2:
        lines.append("  [+] High Calmar ratio indicates excellent risk-adjusted returns")

    lines.append("")
    lines.append("=" * 75)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Quantitative analysis dashboard for Kobe Trading System'
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
        '--benchmark',
        type=str,
        help='Path to benchmark equity/returns file for alpha decomposition'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.95,
        help='Confidence level for VaR (default: 0.95)'
    )
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=0.0,
        help='Annual risk-free rate (default: 0.0)'
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

    args = parser.parse_args()

    if args.dotenv:
        load_env()

    # Determine base directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    # Load equity curve
    if args.equity:
        equity_path = Path(args.equity)
        if not equity_path.is_absolute():
            equity_path = project_dir / equity_path
        equity_df = load_equity_curve(equity_path)
    else:
        wf_dir = Path(args.wfdir)
        if not wf_dir.is_absolute():
            wf_dir = project_dir / wf_dir

        equity_files = discover_equity_files(wf_dir, args.strategy)
        if not equity_files:
            print(f"No equity curve files found in {wf_dir}")
            sys.exit(1)

        equity_df = merge_equity_curves(equity_files)

    if equity_df.empty or 'equity' not in equity_df.columns:
        print("No valid equity data loaded")
        sys.exit(1)

    equity = equity_df['equity']
    returns = compute_returns(equity)

    if len(returns) < 10:
        print(f"Insufficient data: only {len(returns)} return observations")
        sys.exit(1)

    # Compute metrics
    perf = compute_performance_ratios(equity, returns, args.risk_free_rate)
    risk = compute_risk_metrics(returns, (args.confidence, 0.99))
    tests = compute_statistical_tests(returns)
    rolling = compute_rolling_metrics(equity, returns)

    # Alpha decomposition (if benchmark provided)
    alpha = AlphaDecomposition()
    if args.benchmark:
        benchmark_path = Path(args.benchmark)
        if not benchmark_path.is_absolute():
            benchmark_path = project_dir / benchmark_path
        benchmark_df = load_equity_curve(benchmark_path)
        if not benchmark_df.empty and 'equity' in benchmark_df.columns:
            benchmark_returns = compute_returns(benchmark_df['equity'])
            alpha = compute_alpha_decomposition(returns, benchmark_returns, args.risk_free_rate)

    if args.json:
        output = {
            'performance': {
                'cagr': perf.cagr,
                'total_return': perf.total_return,
                'annual_volatility': perf.annual_volatility,
                'sharpe_ratio': perf.sharpe_ratio,
                'sortino_ratio': perf.sortino_ratio,
                'calmar_ratio': perf.calmar_ratio,
                'omega_ratio': perf.omega_ratio,
                'max_drawdown': perf.max_drawdown,
                'max_drawdown_duration': perf.max_drawdown_duration,
                'ulcer_index': perf.ulcer_index,
                'recovery_factor': perf.recovery_factor,
                'gain_to_pain_ratio': perf.gain_to_pain_ratio,
            },
            'risk': {
                'var_95': risk.var_95,
                'var_99': risk.var_99,
                'var_parametric_95': risk.var_parametric_95,
                'var_parametric_99': risk.var_parametric_99,
                'cvar_95': risk.cvar_95,
                'cvar_99': risk.cvar_99,
                'downside_deviation': risk.downside_deviation,
                'upside_deviation': risk.upside_deviation,
                'skewness': risk.skewness,
                'kurtosis': risk.kurtosis,
                'tail_ratio': risk.tail_ratio,
            },
            'alpha_decomposition': {
                'total_alpha': alpha.total_alpha,
                'market_alpha': alpha.market_alpha,
                'timing_alpha': alpha.timing_alpha,
                'selection_alpha': alpha.selection_alpha,
                'beta': alpha.beta,
                'r_squared': alpha.r_squared,
            },
            'statistical_tests': {
                'jarque_bera': {'stat': tests.jarque_bera_stat, 'pvalue': tests.jarque_bera_pvalue},
                'shapiro': {'stat': tests.shapiro_stat, 'pvalue': tests.shapiro_pvalue},
                't_test': {'stat': tests.t_stat_returns, 'pvalue': tests.t_stat_pvalue},
                'ljung_box': {'stat': tests.ljung_box_stat, 'pvalue': tests.ljung_box_pvalue},
                'adf': {'stat': tests.adf_stat, 'pvalue': tests.adf_pvalue},
            },
            'rolling_summary': {
                name: {'mean': s.mean(), 'min': s.min(), 'max': s.max(), 'current': s.iloc[-1]}
                for name, s in rolling.items() if not s.empty
            }
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_quant_dashboard(perf, risk, alpha, tests, rolling))


if __name__ == '__main__':
    main()

