"""
RL Trading Agent Benchmark Evaluation - Renaissance Standard

Comprehensive benchmark evaluation for RL trading agents.
Compares agent performance vs baseline algorithms with statistical rigor.

Jim Simons Quality Standard:
- Statistical significance testing (t-tests, p-values)
- Regime-specific validation (Bull/Bear/Neutral)
- Transaction cost sensitivity analysis
- Walk-forward validation (out-of-sample)
- Comprehensive reporting with deployment recommendation

Usage:
    # Basic evaluation (agent vs baselines)
    python scripts/run_rl_benchmark.py \
        --agent-trades exports/ppo_trades.csv \
        --baselines exports/sac_trades.csv exports/dqn_trades.csv \
        --baseline-names SAC DQN \
        --output reports/rl_benchmark.md

    # With regime data
    python scripts/run_rl_benchmark.py \
        --agent-trades exports/ppo_trades.csv \
        --baselines exports/sac_trades.csv \
        --baseline-names SAC \
        --regime-data state/regime/hmm_regime_history.csv \
        --output reports/rl_benchmark_with_regimes.md

    # Walk-forward validation
    python scripts/run_rl_benchmark.py \
        --agent-trades exports/ppo_trades.csv \
        --walk-forward \
        --train-pct 0.70 \
        --output reports/rl_benchmark_wf.md

Author: Kobe Trading System
Date: 2026-01-08
Version: 1.0
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from evaluation.rl_benchmark import RLBenchmarkFramework, BenchmarkComparison

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_basic_benchmark(
    agent_name: str,
    agent_trades_path: str,
    baseline_paths: List[str],
    baseline_names: List[str],
    regime_data_path: Optional[str] = None,
    output_path: str = "reports/rl_benchmark.md",
) -> BenchmarkComparison:
    """
    Run basic benchmark: agent vs baselines.

    Args:
        agent_name: Name of RL agent
        agent_trades_path: Path to agent trade history CSV
        baseline_paths: List of paths to baseline trade histories
        baseline_names: List of baseline algorithm names
        regime_data_path: Path to regime data CSV (optional)
        output_path: Path to save report

    Returns:
        BenchmarkComparison results
    """
    logger.info("=" * 80)
    logger.info("RL TRADING AGENT BENCHMARK EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Agent: {agent_name}")
    logger.info(f"Baselines: {', '.join(baseline_names)}")
    logger.info(f"Report: {output_path}")
    logger.info("=" * 80)

    # Initialize benchmark framework
    benchmark = RLBenchmarkFramework()

    # Load agent trades
    logger.info(f"\nLoading agent trades: {agent_trades_path}")
    agent_trades = benchmark.load_trade_history(agent_trades_path)

    # Load baseline trades
    baseline_trades = {}
    for name, path in zip(baseline_names, baseline_paths):
        logger.info(f"Loading baseline trades ({name}): {path}")
        baseline_trades[name] = benchmark.load_trade_history(path)

    # Load regime data (optional)
    regime_data = None
    if regime_data_path:
        logger.info(f"Loading regime data: {regime_data_path}")
        regime_data = pd.read_csv(regime_data_path, parse_dates=['timestamp'])

    # Compute agent metrics
    logger.info(f"\nComputing metrics for {agent_name}...")
    agent_metrics = benchmark.compute_metrics(agent_trades, regime_data)

    logger.info(f"  Sharpe Ratio: {agent_metrics.sharpe_ratio:.2f}")
    logger.info(f"  Win Rate: {agent_metrics.win_rate:.1%}")
    logger.info(f"  Profit Factor: {agent_metrics.profit_factor:.2f}")
    logger.info(f"  Max Drawdown: {agent_metrics.max_drawdown_pct:.1%}")

    # Compare vs baselines
    logger.info(f"\nComparing {agent_name} vs baselines...")
    comparison = benchmark.compare_agents(
        agent_name=agent_name,
        agent_trades=agent_trades,
        baseline_trades=baseline_trades,
        regime_data=regime_data,
    )

    logger.info(f"  Rank: {comparison.rank} of {len(baseline_names) + 1}")
    logger.info(f"  Outperforms Baselines: {comparison.outperforms_baselines}")
    logger.info(f"  Statistical Significance: {comparison.significant_improvement}")
    logger.info(f"  Effect Size: {comparison.effect_size}")

    # Validate against Renaissance standards
    passes, failures = benchmark.validate_meets_standards(agent_metrics)

    if passes:
        logger.info("\n✅ AGENT MEETS INSTITUTIONAL STANDARDS")
    else:
        logger.warning("\n❌ AGENT DOES NOT MEET INSTITUTIONAL STANDARDS")
        for failure in failures:
            logger.warning(f"  - {failure}")

    # Generate report
    logger.info(f"\nGenerating comprehensive report: {output_path}")
    benchmark.generate_report(comparison, output_path)

    logger.info("\n✅ Benchmark evaluation complete!")
    return comparison


def run_walk_forward_validation(
    agent_name: str,
    agent_trades_path: str,
    train_pct: float = 0.70,
    output_path: str = "reports/rl_benchmark_wf.md",
) -> Dict:
    """
    Run walk-forward validation (out-of-sample testing).

    Args:
        agent_name: Name of RL agent
        agent_trades_path: Path to agent trade history CSV
        train_pct: Percentage of data for training (default: 70%)
        output_path: Path to save report

    Returns:
        Dict with train and test metrics
    """
    logger.info("=" * 80)
    logger.info("WALK-FORWARD VALIDATION (OUT-OF-SAMPLE TESTING)")
    logger.info("=" * 80)
    logger.info(f"Agent: {agent_name}")
    logger.info(f"Train: {train_pct:.0%}, Test: {1-train_pct:.0%}")
    logger.info("=" * 80)

    # Initialize benchmark
    benchmark = RLBenchmarkFramework()

    # Load trades
    logger.info(f"\nLoading trades: {agent_trades_path}")
    trades = benchmark.load_trade_history(agent_trades_path)

    # Split into train/test
    split_idx = int(len(trades) * train_pct)
    train_trades = trades.iloc[:split_idx]
    test_trades = trades.iloc[split_idx:]

    logger.info(f"Train trades: {len(train_trades)} ({train_pct:.0%})")
    logger.info(f"Test trades: {len(test_trades)} ({1-train_pct:.0%})")

    # Compute metrics for train and test
    logger.info("\nComputing training metrics...")
    train_metrics = benchmark.compute_metrics(train_trades)

    logger.info("  Train Sharpe: {:.2f}".format(train_metrics.sharpe_ratio))
    logger.info("  Train Win Rate: {:.1%}".format(train_metrics.win_rate))

    logger.info("\nComputing test metrics (OUT-OF-SAMPLE)...")
    test_metrics = benchmark.compute_metrics(test_trades)

    logger.info("  Test Sharpe: {:.2f}".format(test_metrics.sharpe_ratio))
    logger.info("  Test Win Rate: {:.1%}".format(test_metrics.win_rate))

    # Check for overfitting
    sharpe_degradation = (train_metrics.sharpe_ratio - test_metrics.sharpe_ratio) / train_metrics.sharpe_ratio if train_metrics.sharpe_ratio > 0 else 0.0
    winrate_degradation = (train_metrics.win_rate - test_metrics.win_rate) / train_metrics.win_rate if train_metrics.win_rate > 0 else 0.0

    logger.info("\nOverfitting Analysis:")
    logger.info(f"  Sharpe Degradation: {sharpe_degradation:.1%}")
    logger.info(f"  Win Rate Degradation: {winrate_degradation:.1%}")

    if sharpe_degradation > 0.20:
        logger.warning("  ⚠️ WARNING: Sharpe degradation > 20% (possible overfitting)")
    elif sharpe_degradation < -0.10:
        logger.info("  ✅ Test performance IMPROVED (agent generalizes well)")
    else:
        logger.info("  ✅ Performance stable (low overfitting risk)")

    # Generate walk-forward report
    logger.info(f"\nGenerating walk-forward report: {output_path}")

    report = f"""# Walk-Forward Validation Report: {agent_name}

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Agent:** {agent_name}
**Train:** {train_pct:.0%} ({len(train_trades)} trades)
**Test:** {1-train_pct:.0%} ({len(test_trades)} trades)

---

## Executive Summary

**Conclusion:** {"✅ AGENT GENERALIZES WELL" if sharpe_degradation < 0.20 else "⚠️ POSSIBLE OVERFITTING"}

- **Train Sharpe:** {train_metrics.sharpe_ratio:.2f}
- **Test Sharpe:** {test_metrics.sharpe_ratio:.2f}
- **Degradation:** {sharpe_degradation:.1%} {"(ACCEPTABLE)" if sharpe_degradation < 0.20 else "(HIGH RISK)"}

**Overfitting Risk:**
- Sharpe degradation > 20%: High risk ❌
- Sharpe degradation 10-20%: Moderate risk ⚠️
- Sharpe degradation < 10%: Low risk ✅

---

## Performance Comparison

| Metric | Train (In-Sample) | Test (Out-of-Sample) | Degradation |
|--------|-------------------|----------------------|-------------|
| Sharpe Ratio | {train_metrics.sharpe_ratio:.2f} | {test_metrics.sharpe_ratio:.2f} | {sharpe_degradation:+.1%} |
| Sortino Ratio | {train_metrics.sortino_ratio:.2f} | {test_metrics.sortino_ratio:.2f} | {((train_metrics.sortino_ratio - test_metrics.sortino_ratio) / train_metrics.sortino_ratio if train_metrics.sortino_ratio > 0 else 0):+.1%} |
| Win Rate | {train_metrics.win_rate:.1%} | {test_metrics.win_rate:.1%} | {winrate_degradation:+.1%} |
| Profit Factor | {train_metrics.profit_factor:.2f} | {test_metrics.profit_factor:.2f} | {((train_metrics.profit_factor - test_metrics.profit_factor) / train_metrics.profit_factor if train_metrics.profit_factor > 0 else 0):+.1%} |
| Max Drawdown | {train_metrics.max_drawdown_pct:.1%} | {test_metrics.max_drawdown_pct:.1%} | {((test_metrics.max_drawdown_pct - train_metrics.max_drawdown_pct) / train_metrics.max_drawdown_pct if train_metrics.max_drawdown_pct > 0 else 0):+.1%} |

---

## Recommendation

"""

    if sharpe_degradation < 0.10:
        report += """**DEPLOY TO PRODUCTION**

✅ Out-of-sample performance stable (< 10% degradation)
✅ Agent generalizes well to unseen data
✅ Low overfitting risk

**Next Steps:**
1. Paper trade for 2 weeks
2. Monitor out-of-sample performance
3. Deploy to live trading with position limits
"""
    elif sharpe_degradation < 0.20:
        report += """**CAUTIOUS DEPLOYMENT - MONITOR CLOSELY**

⚠️ Moderate performance degradation (10-20%)
⚠️ Some overfitting risk

**Next Steps:**
1. Paper trade for 4 weeks (longer validation)
2. Monitor out-of-sample performance closely
3. Consider: More training data, regularization, simpler model
4. Deploy to live trading only if paper trading confirms stability
"""
    else:
        report += """**DO NOT DEPLOY - OVERFITTING DETECTED**

❌ High performance degradation (> 20%)
❌ Agent likely overfit to training data

**Actions:**
1. Increase training data (more symbols, longer history)
2. Add regularization (L1/L2, dropout)
3. Simplify model (fewer features, smaller network)
4. Cross-validation (multiple train/test splits)
5. Re-train and re-evaluate before considering deployment
"""

    report += "\n\n---\n\n**Generated by:** Kobe RL Benchmark Framework (Walk-Forward Validation)\n"

    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)

    logger.info("✅ Walk-forward validation complete!")

    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'sharpe_degradation': sharpe_degradation,
        'winrate_degradation': winrate_degradation,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RL Trading Agent Benchmark Evaluation")

    # Required arguments
    parser.add_argument('--agent-trades', type=str, required=True, help="Path to agent trade history CSV")
    parser.add_argument('--agent-name', type=str, default="PPO", help="Agent name (default: PPO)")

    # Baseline comparison
    parser.add_argument('--baselines', nargs='+', help="Paths to baseline trade history CSVs")
    parser.add_argument('--baseline-names', nargs='+', help="Baseline algorithm names")

    # Regime data
    parser.add_argument('--regime-data', type=str, help="Path to regime data CSV (optional)")

    # Walk-forward validation
    parser.add_argument('--walk-forward', action='store_true', help="Run walk-forward validation")
    parser.add_argument('--train-pct', type=float, default=0.70, help="Train percentage for walk-forward (default: 0.70)")

    # Output
    parser.add_argument('--output', type=str, default="reports/rl_benchmark.md", help="Output report path")

    args = parser.parse_args()

    # Validation
    if not args.walk_forward and (not args.baselines or not args.baseline_names):
        parser.error("Either --walk-forward or both --baselines and --baseline-names are required")

    if args.baselines and args.baseline_names:
        if len(args.baselines) != len(args.baseline_names):
            parser.error("Number of baselines and baseline names must match")

    # Run benchmark
    if args.walk_forward:
        run_walk_forward_validation(
            agent_name=args.agent_name,
            agent_trades_path=args.agent_trades,
            train_pct=args.train_pct,
            output_path=args.output,
        )
    else:
        run_basic_benchmark(
            agent_name=args.agent_name,
            agent_trades_path=args.agent_trades,
            baseline_paths=args.baselines,
            baseline_names=args.baseline_names,
            regime_data_path=args.regime_data,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
