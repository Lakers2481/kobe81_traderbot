"""
Verify Walk-Forward Degradation (Jim Simons Standard)
======================================================

Analyzes train vs test performance degradation to detect overfitting.

Degradation Formula:
    degradation = (train_metric - test_metric) / train_metric * 100

Thresholds (Jim Simons Standard):
    - Win Rate: >10% degradation = REJECT
    - Profit Factor: >15% degradation = REJECT
    - Sharpe Ratio: >20% degradation = ACCEPTABLE (more volatile metric)

Usage:
    python scripts/verify_wf_degradation.py --wf-dir wf_outputs
    python scripts/verify_wf_degradation.py --demo  # Synthetic example
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.structured_log import jlog


@dataclass
class SplitResult:
    """Results from a single walk-forward split."""

    split_id: str
    train_wr: float
    test_wr: float
    train_pf: float
    test_pf: float
    train_sharpe: float
    test_sharpe: float
    train_trades: int
    test_trades: int


@dataclass
class DegradationAnalysis:
    """Analysis of train vs test degradation."""

    strategy: str
    total_splits: int
    avg_train_wr: float
    avg_test_wr: float
    avg_train_pf: float
    avg_test_pf: float
    avg_train_sharpe: float
    avg_test_sharpe: float
    wr_degradation_pct: float
    pf_degradation_pct: float
    sharpe_degradation_pct: float
    passed: bool
    failure_reasons: List[str]


def calculate_degradation(train_val: float, test_val: float) -> float:
    """
    Calculate degradation percentage.

    Formula: (train - test) / train * 100

    Returns:
        Degradation percentage (positive = performance dropped in test)
    """
    if train_val == 0:
        return 0.0
    return ((train_val - test_val) / train_val) * 100


def analyze_wf_results(wf_dir: Path, strategy: str) -> Optional[DegradationAnalysis]:
    """
    Analyze walk-forward results for a strategy.

    Args:
        wf_dir: Walk-forward outputs directory
        strategy: Strategy name (e.g., 'ibs_rsi', 'turtle_soup')

    Returns:
        DegradationAnalysis or None if no valid data
    """
    strategy_dir = wf_dir / strategy

    if not strategy_dir.exists():
        jlog("strategy_dir_not_found", level="WARNING", strategy=strategy)
        return None

    # Find all split directories
    split_dirs = sorted([d for d in strategy_dir.iterdir() if d.is_dir()])

    if not split_dirs:
        jlog("no_splits_found", level="WARNING", strategy=strategy)
        return None

    # Load results from each split
    splits = []
    for split_dir in split_dirs:
        summary_file = split_dir / "summary.json"

        if not summary_file.exists():
            continue

        try:
            with open(summary_file, "r") as f:
                data = json.load(f)

            # Check if this is train or test split
            # Naming convention: train periods are even (split_02, split_04)
            # Test periods are odd (split_01, split_03)
            split_num = int(split_dir.name.split("_")[1])
            is_train = (split_num % 2) == 0

            trades = data.get("trades", 0)
            if trades == 0:
                continue

            result_data = {
                "split_id": split_dir.name,
                "trades": trades,
                "wr": data.get("win_rate", 0.0),
                "pf": data.get("profit_factor", 0.0),
                "sharpe": data.get("sharpe", 0.0),
            }

            if is_train:
                # Store for matching with test
                splits.append({"train": result_data, "test": None})
            else:
                # Find matching train split
                if splits and splits[-1]["test"] is None:
                    splits[-1]["test"] = result_data

        except Exception as e:
            jlog("split_load_error", level="WARNING", split=split_dir.name, error=str(e))
            continue

    # Filter complete train/test pairs
    complete_splits = [s for s in splits if s["train"] and s["test"]]

    if not complete_splits:
        jlog("no_complete_splits", level="WARNING", strategy=strategy)
        return None

    # Calculate average metrics
    train_wrs = [s["train"]["wr"] for s in complete_splits]
    test_wrs = [s["test"]["wr"] for s in complete_splits]
    train_pfs = [s["train"]["pf"] for s in complete_splits]
    test_pfs = [s["test"]["pf"] for s in complete_splits]
    train_sharpes = [s["train"]["sharpe"] for s in complete_splits]
    test_sharpes = [s["test"]["sharpe"] for s in complete_splits]

    avg_train_wr = sum(train_wrs) / len(train_wrs)
    avg_test_wr = sum(test_wrs) / len(test_wrs)
    avg_train_pf = sum(train_pfs) / len(train_pfs)
    avg_test_pf = sum(test_pfs) / len(test_pfs)
    avg_train_sharpe = sum(train_sharpes) / len(train_sharpes)
    avg_test_sharpe = sum(test_sharpes) / len(test_sharpes)

    # Calculate degradation
    wr_degradation = calculate_degradation(avg_train_wr, avg_test_wr)
    pf_degradation = calculate_degradation(avg_train_pf, avg_test_pf)
    sharpe_degradation = calculate_degradation(avg_train_sharpe, avg_test_sharpe)

    # Determine if passed
    failure_reasons = []

    if wr_degradation > 10.0:
        failure_reasons.append(
            f"Win rate degraded {wr_degradation:.1f}% (threshold: 10%)"
        )

    if pf_degradation > 15.0:
        failure_reasons.append(
            f"Profit factor degraded {pf_degradation:.1f}% (threshold: 15%)"
        )

    if sharpe_degradation > 20.0:
        failure_reasons.append(
            f"Sharpe ratio degraded {sharpe_degradation:.1f}% (threshold: 20%)"
        )

    passed = len(failure_reasons) == 0

    return DegradationAnalysis(
        strategy=strategy,
        total_splits=len(complete_splits),
        avg_train_wr=avg_train_wr,
        avg_test_wr=avg_test_wr,
        avg_train_pf=avg_train_pf,
        avg_test_pf=avg_test_pf,
        avg_train_sharpe=avg_train_sharpe,
        avg_test_sharpe=avg_test_sharpe,
        wr_degradation_pct=wr_degradation,
        pf_degradation_pct=pf_degradation,
        sharpe_degradation_pct=sharpe_degradation,
        passed=passed,
        failure_reasons=failure_reasons,
    )


def print_analysis(analysis: DegradationAnalysis):
    """Pretty print degradation analysis."""
    print("\n" + "=" * 80)
    print(f"Strategy: {analysis.strategy.upper()}")
    print("=" * 80)

    print(f"\nTotal Train/Test Pairs: {analysis.total_splits}")

    print("\nTRAIN Performance:")
    print(f"  Win Rate: {analysis.avg_train_wr:.2%}")
    print(f"  Profit Factor: {analysis.avg_train_pf:.2f}")
    print(f"  Sharpe Ratio: {analysis.avg_train_sharpe:.2f}")

    print("\nTEST Performance:")
    print(f"  Win Rate: {analysis.avg_test_wr:.2%}")
    print(f"  Profit Factor: {analysis.avg_test_pf:.2f}")
    print(f"  Sharpe Ratio: {analysis.avg_test_sharpe:.2f}")

    print("\nDegradation:")
    wr_status = "[OK]" if analysis.wr_degradation_pct <= 10 else "[FAIL]"
    pf_status = "[OK]" if analysis.pf_degradation_pct <= 15 else "[FAIL]"
    sharpe_status = "[OK]" if analysis.sharpe_degradation_pct <= 20 else "[FAIL]"

    print(f"  {wr_status} Win Rate: {analysis.wr_degradation_pct:+.1f}% (max: 10%)")
    print(
        f"  {pf_status} Profit Factor: {analysis.pf_degradation_pct:+.1f}% (max: 15%)"
    )
    print(
        f"  {sharpe_status} Sharpe Ratio: {analysis.sharpe_degradation_pct:+.1f}% (max: 20%)"
    )

    if analysis.passed:
        print("\n[OK] PASSED - No significant overfitting detected")
    else:
        print("\n[FAIL] OVERFITTING DETECTED:")
        for reason in analysis.failure_reasons:
            print(f"  - {reason}")


def demo_degradation_analysis():
    """Demonstrate degradation analysis with synthetic examples."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD DEGRADATION DEMO")
    print("Jim Simons / Renaissance Technologies Standard")
    print("=" * 80)

    # Example 1: GOOD strategy (low degradation)
    print("\n" + "-" * 80)
    print("Example 1: GOOD Strategy (Low Degradation)")
    print("-" * 80)

    good = DegradationAnalysis(
        strategy="good_strategy",
        total_splits=10,
        avg_train_wr=0.65,
        avg_test_wr=0.63,  # 3.1% degradation
        avg_train_pf=1.60,
        avg_test_pf=1.50,  # 6.3% degradation
        avg_train_sharpe=1.20,
        avg_test_sharpe=1.05,  # 12.5% degradation
        wr_degradation_pct=calculate_degradation(0.65, 0.63),
        pf_degradation_pct=calculate_degradation(1.60, 1.50),
        sharpe_degradation_pct=calculate_degradation(1.20, 1.05),
        passed=True,
        failure_reasons=[],
    )

    print_analysis(good)

    # Example 2: BAD strategy (high degradation = overfitting)
    print("\n" + "-" * 80)
    print("Example 2: BAD Strategy (Overfitting Detected)")
    print("-" * 80)

    bad = DegradationAnalysis(
        strategy="overfit_strategy",
        total_splits=10,
        avg_train_wr=0.75,
        avg_test_wr=0.55,  # 26.7% degradation!
        avg_train_pf=2.00,
        avg_test_pf=1.10,  # 45% degradation!
        avg_train_sharpe=1.50,
        avg_test_sharpe=0.80,  # 46.7% degradation!
        wr_degradation_pct=calculate_degradation(0.75, 0.55),
        pf_degradation_pct=calculate_degradation(2.00, 1.10),
        sharpe_degradation_pct=calculate_degradation(1.50, 0.80),
        passed=False,
        failure_reasons=[
            "Win rate degraded 26.7% (threshold: 10%)",
            "Profit factor degraded 45.0% (threshold: 15%)",
            "Sharpe ratio degraded 46.7% (threshold: 20%)",
        ],
    )

    print_analysis(bad)

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS:")
    print("=" * 80)
    print("1. Small degradation (< 10%) is NORMAL - data is different between periods")
    print("2. Large degradation (> 15%) suggests OVERFITTING to training data")
    print("3. Jim Simons would reject any strategy with >10% WR degradation")
    print("4. Use walk-forward testing to detect and prevent overfitting")


def main():
    parser = argparse.ArgumentParser(
        description="Verify walk-forward degradation (Jim Simons standard)"
    )
    parser.add_argument(
        "--wf-dir",
        type=str,
        default="wf_outputs",
        help="Walk-forward outputs directory",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration with synthetic examples",
    )

    args = parser.parse_args()

    if args.demo:
        demo_degradation_analysis()
        sys.exit(0)

    wf_dir = Path(args.wf_dir)

    if not wf_dir.exists():
        print(f"ERROR: WF directory not found: {wf_dir}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("WALK-FORWARD DEGRADATION ANALYSIS")
    print("Jim Simons / Renaissance Technologies Standard")
    print("=" * 80)
    print(f"\nAnalyzing: {wf_dir}")

    # Find all strategies
    strategies = [d.name for d in wf_dir.iterdir() if d.is_dir()]
    print(f"Found {len(strategies)} strategies: {strategies}")

    results = []

    for strategy in strategies:
        analysis = analyze_wf_results(wf_dir, strategy)

        if analysis:
            results.append(analysis)
            print_analysis(analysis)

    if not results:
        print("\n[WARNING] No valid walk-forward results found.")
        print("  Either:")
        print("  - WF outputs are incomplete (0 trades)")
        print("  - Summary files don't follow expected format")
        print("  - Run walk-forward test first: python scripts/run_wf_polygon.py")
        print("\nRun with --demo to see expected behavior")
        sys.exit(1)

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)

    print(f"\nStrategies Analyzed: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_count - passed_count}")

    if passed_count == total_count:
        print("\n[OK] All strategies passed degradation tests!")
        sys.exit(0)
    else:
        print(
            f"\n[FAIL] {total_count - passed_count} strategies show overfitting - review and reject!"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
