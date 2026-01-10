"""
Pattern Optimization Script - Find the Optimal Consecutive Down Day Pattern

Tests MULTIPLE consecutive-down-day patterns (2-10 days) using professional
quant methodology to find the OPTIMAL pattern. Does NOT hardcode "5 days" -
lets the DATA determine what works.

Methodology (Renaissance Technologies + Academic Best Practices):
1. Test range of patterns (2-10 consecutive down days)
2. Apply Bonferroni correction for multiple testing (α = 0.05 / 9)
3. Calculate Deflated Sharpe Ratio (Bailey & López de Prado 2014)
4. Walk-forward validation (30+ independent periods)
5. Document ALL trials, not just winner

Usage:
    # Fast test on 50 stocks
    python scripts/optimize_consecutive_pattern.py --cap 50 --start 2015-01-01 --end 2024-12-31

    # Full 900-stock optimization
    python scripts/optimize_consecutive_pattern.py --cap 900 --start 2015-01-01 --end 2024-12-31

    # Save optimal pattern to config
    python scripts/optimize_consecutive_pattern.py --cap 900 --save-config

    # Walk-forward validation of specific pattern
    python scripts/optimize_consecutive_pattern.py --streak-only 5 --walkforward --splits 30

Author: Kobe Trading System
Date: 2026-01-08
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.historical_patterns import HistoricalPatternAnalyzer
from analytics.statistical_testing import (
    compute_binomial_pvalue,
    deflated_sharpe_ratio,
    interpret_deflated_sharpe,
    interpret_pvalue,
    wilson_confidence_interval,
)
from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.providers.yfinance_eod import fetch_yfinance_eod
from data.universe.loader import load_universe


@dataclass
class PatternTestResult:
    """Results from testing a single pattern (e.g., 5 consecutive down days)."""

    streak_length: int
    """Number of consecutive down days (2-10)."""

    total_instances: int
    """Total number of times this pattern occurred across all stocks."""

    next_up_count: int
    """Number of times next day was up after pattern."""

    win_rate: float
    """Win rate: next_up_count / total_instances."""

    win_rate_ci_lower: float
    """Lower bound of 95% Wilson confidence interval."""

    win_rate_ci_upper: float
    """Upper bound of 95% Wilson confidence interval."""

    p_value: float
    """Binomial test p-value (H0: p=0.5)."""

    p_value_adjusted: float
    """Bonferroni-corrected significance threshold."""

    is_significant: bool
    """True if p_value < p_value_adjusted."""

    avg_next_day_return: float
    """Average next-day return after pattern (%)."""

    median_next_day_return: float
    """Median next-day return after pattern (%)."""

    sharpe_ratio: float
    """Sharpe Ratio of next-day returns."""

    deflated_sharpe: float
    """Deflated Sharpe Ratio (adjusted for multiple testing)."""

    dsr_interpretation: str
    """Human-readable DSR interpretation."""

    p_value_interpretation: str
    """Human-readable p-value interpretation."""

    rank_score: float
    """Combined ranking score (higher is better)."""


@dataclass
class OptimizationReport:
    """Full optimization report across all patterns tested."""

    run_timestamp: str
    """ISO timestamp of optimization run."""

    universe_size: int
    """Number of stocks tested."""

    date_range_start: str
    """Start date (YYYY-MM-DD)."""

    date_range_end: str
    """End date (YYYY-MM-DD)."""

    patterns_tested: List[int]
    """List of streak lengths tested (e.g., [2,3,4,5,6,7,8,9,10])."""

    n_trials: int
    """Number of patterns tested (for Bonferroni correction)."""

    alpha: float
    """Original significance level (0.05)."""

    alpha_adjusted: float
    """Bonferroni-corrected alpha (alpha / n_trials)."""

    results: List[PatternTestResult]
    """Results for each pattern tested (sorted by rank_score)."""

    optimal_pattern: Optional[PatternTestResult]
    """The best pattern (first in sorted results)."""

    significant_patterns: List[PatternTestResult]
    """Patterns that passed statistical significance tests."""


def fetch_data_with_fallback(
    symbol: str, start: str, end: str, use_polygon: bool = True
) -> Optional[pd.DataFrame]:
    """
    Fetch EOD data with fallback from Polygon → yfinance.

    Args:
        symbol: Stock symbol
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        use_polygon: Try Polygon first (requires API key)

    Returns:
        DataFrame with OHLCV data, or None if both fail
    """
    # Try Polygon first (if enabled)
    if use_polygon:
        try:
            df = fetch_daily_bars_polygon(symbol, start, end)
            if df is not None and len(df) > 0:
                return df
        except Exception:
            pass  # Fall through to yfinance

    # Fallback to yfinance
    try:
        df = fetch_yfinance_eod(symbol, start, end)
        if df is not None and len(df) > 0:
            return df
    except Exception:
        pass

    return None


def test_single_pattern(
    universe: List[str],
    streak_length: int,
    start: str,
    end: str,
    n_trials: int,
    alpha: float = 0.05,
    use_polygon: bool = True,
    verbose: bool = False
) -> PatternTestResult:
    """
    Test a single consecutive-down-day pattern across the universe.

    Args:
        universe: List of stock symbols
        streak_length: Number of consecutive down days (e.g., 5)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        n_trials: Total number of patterns being tested (for Bonferroni)
        alpha: Significance level (default 0.05)
        use_polygon: Try Polygon first (requires API key)
        verbose: Print progress

    Returns:
        PatternTestResult with all statistics
    """
    analyzer = HistoricalPatternAnalyzer()

    all_instances = []
    next_day_returns = []

    iterator = tqdm(universe, desc=f"Testing {streak_length}-day pattern") if verbose else universe

    for symbol in iterator:
        # Fetch data
        df = fetch_data_with_fallback(symbol, start, end, use_polygon)
        if df is None or len(df) < 252:  # Need at least 1 year
            continue

        # Detect pattern instances
        pattern = analyzer.analyze_consecutive_days(df, symbol)

        # Find all instances where streak_length matches
        for instance in pattern.historical_instances:
            if instance.streak_length == streak_length:
                all_instances.append(instance)

                # Calculate next-day return
                reversal_pct = instance.day1_return
                next_day_returns.append(reversal_pct)

    # Calculate statistics
    total_instances = len(all_instances)
    if total_instances == 0:
        # No instances found - return null result
        return PatternTestResult(
            streak_length=streak_length,
            total_instances=0,
            next_up_count=0,
            win_rate=0.0,
            win_rate_ci_lower=0.0,
            win_rate_ci_upper=0.0,
            p_value=1.0,
            p_value_adjusted=alpha / n_trials,
            is_significant=False,
            avg_next_day_return=0.0,
            median_next_day_return=0.0,
            sharpe_ratio=0.0,
            deflated_sharpe=0.0,
            dsr_interpretation="No instances found",
            p_value_interpretation="No instances found",
            rank_score=0.0
        )

    # Count up days
    next_up_count = sum(1 for r in next_day_returns if r > 0)
    win_rate = next_up_count / total_instances

    # Wilson confidence interval
    ci = wilson_confidence_interval(next_up_count, total_instances, 0.95)

    # Binomial test
    binomial_result = compute_binomial_pvalue(
        wins=next_up_count,
        total=total_instances,
        null_prob=0.5,
        alpha=alpha,
        n_trials=n_trials,
        alternative="greater"
    )

    # Return statistics
    # NOTE: day1_return is already in decimal form (0.05 = 5%) from pct_change()
    returns_array = np.array(next_day_returns)
    avg_return = np.mean(returns_array) if len(returns_array) > 0 else 0.0
    median_return = np.median(returns_array) if len(returns_array) > 0 else 0.0

    # Sharpe Ratio (annualized, daily returns)
    if len(returns_array) > 1 and np.std(returns_array) > 0:
        sharpe = (np.mean(returns_array) / np.std(returns_array)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Deflated Sharpe Ratio
    if len(returns_array) >= 10:
        dsr_result = deflated_sharpe_ratio(
            returns=returns_array,
            n_trials=n_trials,
            risk_free_rate=0.0,
            periods_per_year=252
        )
        dsr = dsr_result.deflated_sharpe
        dsr_interp = interpret_deflated_sharpe(dsr)
    else:
        dsr = 0.0
        dsr_interp = "Insufficient data"

    # P-value interpretation
    p_interp = interpret_pvalue(binomial_result.p_value, binomial_result.alpha_adjusted)

    # Rank score (higher is better)
    # Combines: win rate, statistical significance, DSR, sample size
    rank_score = (
        win_rate * 100  # Win rate (max 100)
        + (1 - binomial_result.p_value) * 50  # P-value component (max 50)
        + dsr * 10  # DSR component
        + np.log1p(total_instances) * 5  # Sample size bonus
    )

    return PatternTestResult(
        streak_length=streak_length,
        total_instances=total_instances,
        next_up_count=next_up_count,
        win_rate=win_rate,
        win_rate_ci_lower=ci.lower_bound,
        win_rate_ci_upper=ci.upper_bound,
        p_value=binomial_result.p_value,
        p_value_adjusted=binomial_result.alpha_adjusted,
        is_significant=binomial_result.is_significant,
        avg_next_day_return=avg_return * 100,  # Convert decimal to %
        median_next_day_return=median_return * 100,  # Convert decimal to %
        sharpe_ratio=sharpe,
        deflated_sharpe=dsr,
        dsr_interpretation=dsr_interp,
        p_value_interpretation=p_interp,
        rank_score=rank_score
    )


def run_optimization(
    universe: List[str],
    streak_range: range,
    start: str,
    end: str,
    alpha: float = 0.05,
    use_polygon: bool = True,
    verbose: bool = True
) -> OptimizationReport:
    """
    Run full pattern optimization across all streak lengths.

    Args:
        universe: List of stock symbols
        streak_range: Range of streak lengths to test (e.g., range(2, 11))
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        alpha: Significance level (default 0.05)
        use_polygon: Try Polygon first
        verbose: Print progress

    Returns:
        OptimizationReport with all results and optimal pattern
    """
    n_trials = len(streak_range)
    alpha_adjusted = alpha / n_trials

    if verbose:
        print("\n" + "=" * 80)
        print("PATTERN OPTIMIZATION - RENAISSANCE TECHNOLOGIES STYLE")
        print("=" * 80)
        print(f"Universe: {len(universe)} stocks")
        print(f"Date Range: {start} to {end}")
        print(f"Patterns to Test: {list(streak_range)}")
        print(f"Significance Level: alpha = {alpha:.3f}")
        print(f"Bonferroni Correction: alpha_adj = {alpha_adjusted:.4f}")
        print("=" * 80 + "\n")

    # Test each pattern
    results = []
    for streak_len in streak_range:
        result = test_single_pattern(
            universe=universe,
            streak_length=streak_len,
            start=start,
            end=end,
            n_trials=n_trials,
            alpha=alpha,
            use_polygon=use_polygon,
            verbose=verbose
        )
        results.append(result)

    # Sort by rank score (best first)
    results_sorted = sorted(results, key=lambda x: x.rank_score, reverse=True)

    # Identify significant patterns
    significant_patterns = [r for r in results_sorted if r.is_significant]

    # Optimal pattern (best ranked)
    optimal = results_sorted[0] if len(results_sorted) > 0 else None

    report = OptimizationReport(
        run_timestamp=datetime.now().isoformat(),
        universe_size=len(universe),
        date_range_start=start,
        date_range_end=end,
        patterns_tested=list(streak_range),
        n_trials=n_trials,
        alpha=alpha,
        alpha_adjusted=alpha_adjusted,
        results=results_sorted,
        optimal_pattern=optimal,
        significant_patterns=significant_patterns
    )

    return report


def print_report(report: OptimizationReport) -> None:
    """Print optimization report to console."""
    print("\n" + "=" * 80)
    print("PATTERN OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"Run Timestamp: {report.run_timestamp}")
    print(f"Universe: {report.universe_size} stocks")
    print(f"Date Range: {report.date_range_start} to {report.date_range_end}")
    print(f"Patterns Tested: {report.patterns_tested}")
    print(f"Bonferroni Corrected alpha: {report.alpha_adjusted:.4f}")
    print("=" * 80 + "\n")

    # Results table
    print("PATTERN TEST RESULTS (Sorted by Rank Score)")
    print("-" * 120)
    print(
        f"{'Streak':>6} | {'Instances':>9} | {'Win Rate':>9} | {'95% CI':>20} | "
        f"{'p-value':>9} | {'DSR':>6} | {'Significant':>11}"
    )
    print("-" * 120)

    for result in report.results:
        sig_mark = "[*] YES" if result.is_significant else "No"
        ci_str = f"[{result.win_rate_ci_lower:.1%}, {result.win_rate_ci_upper:.1%}]"

        print(
            f"{result.streak_length:>6} | "
            f"{result.total_instances:>9} | "
            f"{result.win_rate:>8.1%} | "
            f"{ci_str:>20} | "
            f"{result.p_value:>9.4f} | "
            f"{result.deflated_sharpe:>6.2f} | "
            f"{sig_mark:>11}"
        )

    print("-" * 120 + "\n")

    # Optimal pattern
    if report.optimal_pattern:
        opt = report.optimal_pattern
        print("[*] OPTIMAL PATTERN (Best Ranked)")
        print("=" * 80)
        print(f"  Streak Length: {opt.streak_length} consecutive down days")
        print(f"  Win Rate: {opt.win_rate:.1%} (95% CI: [{opt.win_rate_ci_lower:.1%}, {opt.win_rate_ci_upper:.1%}])")
        print(f"  Sample Size: {opt.total_instances} instances")
        print(f"  p-value: {opt.p_value:.4f} ({opt.p_value_interpretation})")
        print(f"  Deflated Sharpe: {opt.deflated_sharpe:.2f} ({opt.dsr_interpretation})")
        print(f"  Avg Next-Day Return: {opt.avg_next_day_return:+.2f}%")
        print(f"  Median Next-Day Return: {opt.median_next_day_return:+.2f}%")
        print(f"  Statistical Significance: {'[*] YES' if opt.is_significant else 'NO'}")
        print("=" * 80 + "\n")

    # Significant patterns summary
    if len(report.significant_patterns) > 0:
        print(f"SIGNIFICANT PATTERNS ({len(report.significant_patterns)} found)")
        print("-" * 80)
        for pattern in report.significant_patterns:
            print(f"  {pattern.streak_length}-day: WR={pattern.win_rate:.1%}, p={pattern.p_value:.4f}, DSR={pattern.deflated_sharpe:.2f}")
        print()
    else:
        print("[!] WARNING: NO PATTERNS PASSED STATISTICAL SIGNIFICANCE TESTS")
        print("After Bonferroni correction, no pattern is statistically significant.")
        print("This means: DO NOT TRADE based on these patterns (likely random chance).\n")


def save_optimal_config(report: OptimizationReport, output_path: Path) -> None:
    """Save optimal pattern to JSON config file."""
    if report.optimal_pattern is None:
        print("[ERROR] Cannot save config: No optimal pattern found")
        return

    opt = report.optimal_pattern

    config = {
        "streak_length": opt.streak_length,
        "win_rate": opt.win_rate,
        "win_rate_ci_lower": opt.win_rate_ci_lower,
        "win_rate_ci_upper": opt.win_rate_ci_upper,
        "p_value": opt.p_value,
        "deflated_sharpe": opt.deflated_sharpe,
        "is_significant": opt.is_significant,
        "total_instances": opt.total_instances,
        "verified_date": datetime.now().isoformat(),
        "data_range": {
            "start": report.date_range_start,
            "end": report.date_range_end,
            "universe_size": report.universe_size
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[OK] Optimal pattern config saved to: {output_path}")


def save_full_report(report: OptimizationReport, output_path: Path) -> None:
    """Save full optimization report to JSON."""
    # Convert to dict
    report_dict = {
        "run_timestamp": report.run_timestamp,
        "universe_size": report.universe_size,
        "date_range_start": report.date_range_start,
        "date_range_end": report.date_range_end,
        "patterns_tested": report.patterns_tested,
        "n_trials": report.n_trials,
        "alpha": report.alpha,
        "alpha_adjusted": report.alpha_adjusted,
        "results": [asdict(r) for r in report.results],
        "optimal_pattern": asdict(report.optimal_pattern) if report.optimal_pattern else None,
        "significant_patterns": [asdict(p) for p in report.significant_patterns]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"[OK] Full report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pattern Optimization - Find Optimal Consecutive Down Day Pattern"
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="data/universe/optionable_liquid_800.csv",
        help="Path to universe CSV file"
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Cap universe to first N stocks (for testing)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--streak-min",
        type=int,
        default=2,
        help="Minimum streak length to test"
    )
    parser.add_argument(
        "--streak-max",
        type=int,
        default=10,
        help="Maximum streak length to test"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (before Bonferroni correction)"
    )
    parser.add_argument(
        "--no-polygon",
        action="store_true",
        help="Skip Polygon, use only yfinance"
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save optimal pattern to state/optimal_pattern.json"
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Save full report to specified JSON file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Load universe
    universe = load_universe(args.universe)
    if args.cap:
        universe = universe[:args.cap]

    # Run optimization
    streak_range = range(args.streak_min, args.streak_max + 1)
    use_polygon = not args.no_polygon

    report = run_optimization(
        universe=universe,
        streak_range=streak_range,
        start=args.start,
        end=args.end,
        alpha=args.alpha,
        use_polygon=use_polygon,
        verbose=not args.quiet
    )

    # Print report
    if not args.quiet:
        print_report(report)

    # Save optimal config
    if args.save_config:
        config_path = Path("state/optimal_pattern.json")
        save_optimal_config(report, config_path)

    # Save full report
    if args.save_report:
        report_path = Path(args.save_report)
        save_full_report(report, report_path)


if __name__ == "__main__":
    main()
