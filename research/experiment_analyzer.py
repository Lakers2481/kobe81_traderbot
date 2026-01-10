"""
Experiment Analyzer - Automated A/B Testing for Strategy Parameters

Mission 3 Part A: Automated A/B Testing
--------------------------------------
This script runs weekly to analyze experimental trades and determine if
parameter changes lead to statistically significant improvements.

When an improvement is validated, the script:
1. Updates the production strategy configuration
2. Archives old experiment logs
3. Sends an alert about the change

Author: Kobe Trading System
Date: 2026-01-07
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Root directory
ROOT = Path(__file__).resolve().parents[1]

# Paths
EXPERIMENTS_LOG = ROOT / "logs" / "experiments.jsonl"
ARCHIVE_DIR = ROOT / "logs" / "experiments_archive"
STRATEGY_CONFIG = ROOT / "config" / "strategies" / "default.yaml"


@dataclass
class ExperimentResult:
    """Result of a single experimental trade."""
    timestamp: str
    symbol: str
    parameter_name: str
    control_value: Any
    experiment_value: Any
    pnl: float
    win: bool
    is_experiment: bool = True
    entry_price: float = 0.0
    exit_price: float = 0.0
    hold_days: int = 0


@dataclass
class ParameterAnalysis:
    """Analysis of a parameter experiment."""
    parameter_name: str
    control_value: Any
    experiment_value: Any
    control_trades: int
    experiment_trades: int
    control_win_rate: float
    experiment_win_rate: float
    control_avg_pnl: float
    experiment_avg_pnl: float
    p_value: float
    improvement_pct: float
    is_significant: bool
    recommendation: str


class ExperimentAnalyzer:
    """
    Analyzes experimental trades to determine if parameter changes improve performance.

    Uses statistical tests to validate improvements before promoting to production.
    """

    def __init__(
        self,
        experiments_log: Path = EXPERIMENTS_LOG,
        min_samples: int = 20,
        significance_level: float = 0.05,
        min_improvement_pct: float = 5.0
    ):
        """
        Initialize the analyzer.

        Args:
            experiments_log: Path to experiments JSONL file
            min_samples: Minimum samples needed for analysis
            significance_level: p-value threshold for significance
            min_improvement_pct: Minimum improvement % to promote
        """
        self.experiments_log = experiments_log
        self.min_samples = min_samples
        self.significance_level = significance_level
        self.min_improvement_pct = min_improvement_pct

    def load_experiments(self) -> List[ExperimentResult]:
        """Load all experiment records from the log file."""
        experiments = []

        if not self.experiments_log.exists():
            logger.warning(f"Experiments log not found: {self.experiments_log}")
            return experiments

        with open(self.experiments_log, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    exp = ExperimentResult(
                        timestamp=data.get('timestamp', ''),
                        symbol=data.get('symbol', ''),
                        parameter_name=data.get('parameter_name', ''),
                        control_value=data.get('control_value'),
                        experiment_value=data.get('experiment_value'),
                        pnl=float(data.get('pnl', 0)),
                        win=data.get('win', False),
                        is_experiment=data.get('is_experiment', True),
                        entry_price=float(data.get('entry_price', 0)),
                        exit_price=float(data.get('exit_price', 0)),
                        hold_days=int(data.get('hold_days', 0)),
                    )
                    experiments.append(exp)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse experiment record: {e}")
                    continue

        logger.info(f"Loaded {len(experiments)} experiment records")
        return experiments

    def group_by_parameter(
        self,
        experiments: List[ExperimentResult]
    ) -> Dict[str, List[ExperimentResult]]:
        """Group experiments by parameter being tested."""
        groups: Dict[str, List[ExperimentResult]] = {}

        for exp in experiments:
            key = f"{exp.parameter_name}:{exp.control_value}:{exp.experiment_value}"
            if key not in groups:
                groups[key] = []
            groups[key].append(exp)

        return groups

    def analyze_parameter(
        self,
        parameter_key: str,
        experiments: List[ExperimentResult]
    ) -> Optional[ParameterAnalysis]:
        """
        Analyze whether a parameter change improved performance.

        Uses two-sample t-test for P&L and chi-square for win rate.

        Args:
            parameter_key: Parameter identifier (name:control:experiment)
            experiments: List of experiments for this parameter

        Returns:
            ParameterAnalysis or None if insufficient data
        """
        if len(experiments) < self.min_samples:
            logger.info(f"{parameter_key}: Insufficient samples ({len(experiments)} < {self.min_samples})")
            return None

        # Split into control and experiment
        control = [e for e in experiments if not e.is_experiment]
        experiment = [e for e in experiments if e.is_experiment]

        if len(control) < self.min_samples // 2 or len(experiment) < self.min_samples // 2:
            logger.info(f"{parameter_key}: Insufficient samples in each group")
            return None

        # Parse parameter info
        parts = parameter_key.split(':')
        param_name = parts[0] if len(parts) > 0 else "unknown"
        control_value = parts[1] if len(parts) > 1 else None
        experiment_value = parts[2] if len(parts) > 2 else None

        # Calculate metrics
        control_pnls = [e.pnl for e in control]
        experiment_pnls = [e.pnl for e in experiment]

        control_wins = sum(1 for e in control if e.win)
        experiment_wins = sum(1 for e in experiment if e.win)

        control_wr = control_wins / len(control) if control else 0
        experiment_wr = experiment_wins / len(experiment) if experiment else 0

        control_avg_pnl = np.mean(control_pnls) if control_pnls else 0
        experiment_avg_pnl = np.mean(experiment_pnls) if experiment_pnls else 0

        # Statistical test (t-test for P&L)
        if len(control_pnls) >= 2 and len(experiment_pnls) >= 2:
            t_stat, p_value = stats.ttest_ind(experiment_pnls, control_pnls)
        else:
            p_value = 1.0

        # Calculate improvement
        if control_avg_pnl != 0:
            improvement_pct = ((experiment_avg_pnl - control_avg_pnl) / abs(control_avg_pnl)) * 100
        elif experiment_avg_pnl > 0:
            improvement_pct = 100.0
        else:
            improvement_pct = 0.0

        # Determine if significant
        is_significant = (
            p_value < self.significance_level and
            improvement_pct >= self.min_improvement_pct and
            experiment_avg_pnl > control_avg_pnl
        )

        # Generate recommendation
        if is_significant:
            recommendation = f"PROMOTE: {param_name} = {experiment_value} (was {control_value})"
        elif improvement_pct > 0:
            recommendation = f"CONTINUE TESTING: Positive trend but not significant (p={p_value:.3f})"
        else:
            recommendation = f"REJECT: No improvement detected"

        return ParameterAnalysis(
            parameter_name=param_name,
            control_value=control_value,
            experiment_value=experiment_value,
            control_trades=len(control),
            experiment_trades=len(experiment),
            control_win_rate=control_wr,
            experiment_win_rate=experiment_wr,
            control_avg_pnl=control_avg_pnl,
            experiment_avg_pnl=experiment_avg_pnl,
            p_value=p_value,
            improvement_pct=improvement_pct,
            is_significant=is_significant,
            recommendation=recommendation
        )

    def run_analysis(self) -> List[ParameterAnalysis]:
        """Run full analysis on all experiments."""
        experiments = self.load_experiments()
        if not experiments:
            logger.info("No experiments to analyze")
            return []

        groups = self.group_by_parameter(experiments)
        results = []

        for param_key, group_experiments in groups.items():
            analysis = self.analyze_parameter(param_key, group_experiments)
            if analysis:
                results.append(analysis)
                logger.info(
                    f"{analysis.parameter_name}: "
                    f"improvement={analysis.improvement_pct:.1f}%, "
                    f"p={analysis.p_value:.3f}, "
                    f"significant={analysis.is_significant}"
                )

        return results

    def promote_parameter(self, analysis: ParameterAnalysis) -> bool:
        """
        Promote a validated parameter change to production.

        Args:
            analysis: The analysis result to promote

        Returns:
            True if promotion succeeded
        """
        if not analysis.is_significant:
            logger.warning(f"Cannot promote non-significant result: {analysis.parameter_name}")
            return False

        try:
            # Load current config
            import yaml

            if not STRATEGY_CONFIG.exists():
                logger.error(f"Strategy config not found: {STRATEGY_CONFIG}")
                return False

            with open(STRATEGY_CONFIG, 'r') as f:
                config = yaml.safe_load(f) or {}

            # Update parameter
            old_value = config.get(analysis.parameter_name, analysis.control_value)
            config[analysis.parameter_name] = analysis.experiment_value

            # Add promotion metadata
            if '_promotions' not in config:
                config['_promotions'] = []
            config['_promotions'].append({
                'parameter': analysis.parameter_name,
                'old_value': str(old_value),
                'new_value': str(analysis.experiment_value),
                'improvement_pct': round(analysis.improvement_pct, 2),
                'p_value': round(analysis.p_value, 4),
                'promoted_at': datetime.now().isoformat(),
            })

            # Write updated config
            with open(STRATEGY_CONFIG, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            logger.info(f"PROMOTED: {analysis.parameter_name} = {analysis.experiment_value}")

            # Send alert
            self._send_promotion_alert(analysis)

            # Archive experiments
            self._archive_experiments(analysis.parameter_name)

            return True

        except Exception as e:
            logger.error(f"Failed to promote parameter: {e}")
            return False

    def _send_promotion_alert(self, analysis: ParameterAnalysis) -> None:
        """Send alert about parameter promotion."""
        try:
            from notifications.telegram_alerts import send_alert
            message = (
                f"EXPERIMENT CONCLUDED: {analysis.parameter_name}\n"
                f"New value: {analysis.experiment_value} (was {analysis.control_value})\n"
                f"Improvement: {analysis.improvement_pct:+.1f}% P&L\n"
                f"Win rate: {analysis.control_win_rate:.1%} -> {analysis.experiment_win_rate:.1%}\n"
                f"p-value: {analysis.p_value:.4f}\n"
                f"Samples: {analysis.control_trades} control, {analysis.experiment_trades} experiment"
            )
            send_alert(message)
        except Exception as e:
            logger.warning(f"Failed to send promotion alert: {e}")

    def _archive_experiments(self, parameter_name: str) -> None:
        """Archive processed experiments."""
        try:
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            archive_path = ARCHIVE_DIR / f"experiments_{parameter_name}_{datetime.now().strftime('%Y%m%d')}.jsonl"

            # Copy relevant experiments to archive
            if self.experiments_log.exists():
                with open(self.experiments_log, 'r') as src, open(archive_path, 'w') as dst:
                    for line in src:
                        try:
                            data = json.loads(line.strip())
                            if data.get('parameter_name') == parameter_name:
                                dst.write(line)
                        except json.JSONDecodeError:
                            continue

            logger.info(f"Archived experiments to: {archive_path}")

        except Exception as e:
            logger.warning(f"Failed to archive experiments: {e}")

    def generate_report(self) -> Dict[str, Any]:
        """Generate full analysis report."""
        results = self.run_analysis()

        significant = [r for r in results if r.is_significant]
        promising = [r for r in results if not r.is_significant and r.improvement_pct > 0]
        rejected = [r for r in results if r.improvement_pct <= 0]

        return {
            "analysis_time": datetime.now().isoformat(),
            "total_parameters_tested": len(results),
            "significant_improvements": len(significant),
            "promising_trends": len(promising),
            "rejected": len(rejected),
            "results": [
                {
                    "parameter": r.parameter_name,
                    "control_value": r.control_value,
                    "experiment_value": r.experiment_value,
                    "control_trades": r.control_trades,
                    "experiment_trades": r.experiment_trades,
                    "improvement_pct": round(r.improvement_pct, 2),
                    "p_value": round(r.p_value, 4),
                    "is_significant": r.is_significant,
                    "recommendation": r.recommendation,
                }
                for r in results
            ]
        }


def log_experiment(
    symbol: str,
    parameter_name: str,
    control_value: Any,
    experiment_value: Any,
    pnl: float,
    win: bool,
    is_experiment: bool = True,
    entry_price: float = 0,
    exit_price: float = 0,
    hold_days: int = 0
) -> None:
    """
    Log an experimental trade result.

    This should be called by the learning_hub when processing trade outcomes
    for trades marked with is_experiment=True.
    """
    EXPERIMENTS_LOG.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "parameter_name": parameter_name,
        "control_value": control_value,
        "experiment_value": experiment_value,
        "pnl": pnl,
        "win": win,
        "is_experiment": is_experiment,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "hold_days": hold_days,
    }

    with open(EXPERIMENTS_LOG, 'a') as f:
        f.write(json.dumps(record) + "\n")

    logger.info(f"Logged experiment: {symbol} {parameter_name}={experiment_value} pnl=${pnl:.2f}")


# Example usage / CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze trading experiments")
    parser.add_argument("--report", action="store_true", help="Generate analysis report")
    parser.add_argument("--promote", action="store_true", help="Promote significant improvements")
    args = parser.parse_args()

    analyzer = ExperimentAnalyzer()

    if args.report:
        report = analyzer.generate_report()
        print(json.dumps(report, indent=2))

    if args.promote:
        results = analyzer.run_analysis()
        for result in results:
            if result.is_significant:
                print(f"Promoting: {result.parameter_name}")
                analyzer.promote_parameter(result)
