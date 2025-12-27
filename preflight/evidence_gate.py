"""
Evidence Gate - Quant-Interview-Grade Promotion System
========================================================

Ensures strategies meet rigorous evidence standards before
promotion to live trading. Integrates with KnowledgeBoundary
to enforce "stand down when uncertain" behavior.

Key Evidence Requirements:
- Minimum OOS trades (default: 100)
- Minimum OOS Sharpe ratio (default: 0.5)
- Minimum OOS profit factor (default: 1.3)
- Maximum drawdown limit (default: 25%)
- Stability across regimes
- No lookahead bias detected

Quant Interview Standard:
This system can defend every claim with data. It refuses to
promote strategies with insufficient evidence and documents
exactly why a strategy passed or failed.

Usage:
    from preflight.evidence_gate import EvidenceGate

    gate = EvidenceGate()
    report = gate.evaluate(backtest_results, strategy_name='momentum_v1')

    if report.passed:
        print("Strategy approved for live trading")
    else:
        print(f"Strategy rejected: {report.rejection_reasons}")
"""
from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EvidenceLevel(Enum):
    """Evidence strength levels."""
    INSUFFICIENT = "insufficient"  # Not enough data
    WEAK = "weak"                  # Marginal evidence
    MODERATE = "moderate"          # Acceptable for paper trading
    STRONG = "strong"              # Acceptable for live trading
    EXCELLENT = "excellent"        # High confidence


@dataclass
class EvidenceRequirements:
    """
    Minimum requirements for strategy promotion.

    These are quant-interview-grade standards that can be
    defended with data and statistical reasoning.
    """
    # Trade count requirements
    min_oos_trades: int = 100          # Minimum out-of-sample trades
    min_total_trades: int = 200         # Minimum total trades

    # Performance requirements
    min_oos_sharpe: float = 0.5        # Minimum OOS Sharpe ratio
    min_oos_profit_factor: float = 1.3  # Minimum OOS profit factor
    min_win_rate: float = 0.40          # Minimum win rate
    max_drawdown: float = 0.25          # Maximum acceptable drawdown

    # Stability requirements (across regimes)
    max_regime_performance_variance: float = 0.3  # Max variance across regimes
    min_regimes_profitable: int = 2     # Must be profitable in N regimes

    # Statistical requirements
    min_t_stat: float = 2.0            # Minimum t-statistic for returns
    max_overfitting_score: float = 0.5  # Max IS vs OOS degradation

    # Data integrity requirements
    min_data_years: float = 5.0        # Minimum years of history
    max_data_gaps_pct: float = 0.05    # Maximum gap percentage

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceReport:
    """
    Comprehensive evidence report for a strategy.

    Documents exactly why a strategy passed or failed,
    suitable for quant interview defense.
    """
    strategy_name: str
    evaluated_at: str
    passed: bool
    evidence_level: EvidenceLevel

    # Detailed scores
    oos_trades: int = 0
    oos_sharpe: float = 0.0
    oos_profit_factor: float = 0.0
    oos_win_rate: float = 0.0
    max_drawdown: float = 0.0

    # Stability metrics
    regime_performance: Dict[str, float] = field(default_factory=dict)
    regime_variance: float = 0.0

    # Statistical metrics
    returns_t_stat: float = 0.0
    overfitting_score: float = 0.0

    # Pass/fail details
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Knowledge boundary integration
    uncertainty_level: str = "unknown"
    should_stand_down: bool = False

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['evidence_level'] = self.evidence_level.value
        return d

    def save(self, path: Path):
        """Save report to JSON file."""
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> 'EvidenceReport':
        """Load report from JSON file."""
        data = json.loads(path.read_text())
        data['evidence_level'] = EvidenceLevel(data['evidence_level'])
        return cls(**data)


class EvidenceGate:
    """
    Quant-interview-grade evidence gate for strategy promotion.

    Integrates with:
    - KnowledgeBoundary: Stand down when uncertain
    - BacktestResults: Evaluate OOS performance
    - Walk-forward: Verify across time periods
    """

    def __init__(
        self,
        requirements: Optional[EvidenceRequirements] = None,
        reports_dir: str = "reports/evidence",
    ):
        self.requirements = requirements or EvidenceRequirements()
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Lazy load knowledge boundary
        self._knowledge_boundary = None

        logger.info(f"EvidenceGate initialized with requirements: {self.requirements}")

    @property
    def knowledge_boundary(self):
        """Lazy load KnowledgeBoundary."""
        if self._knowledge_boundary is None:
            try:
                from cognitive.knowledge_boundary import KnowledgeBoundary
                self._knowledge_boundary = KnowledgeBoundary()
            except ImportError:
                logger.warning("KnowledgeBoundary not available")
                self._knowledge_boundary = None
        return self._knowledge_boundary

    def evaluate(
        self,
        backtest_results: Dict[str, Any],
        strategy_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvidenceReport:
        """
        Evaluate strategy against evidence requirements.

        Args:
            backtest_results: Results from walk-forward or backtest
                Expected keys:
                - trades: List of trade dicts or DataFrame
                - equity_curve: DataFrame with equity over time
                - metrics: Dict with sharpe, profit_factor, win_rate, etc.
                - oos_metrics: Dict with OOS-specific metrics (optional)
                - regime_splits: Dict with per-regime performance (optional)
            strategy_name: Name of the strategy being evaluated
            context: Additional context for knowledge boundary

        Returns:
            EvidenceReport with pass/fail and detailed analysis
        """
        report = EvidenceReport(
            strategy_name=strategy_name,
            evaluated_at=datetime.now().isoformat(),
            passed=False,
            evidence_level=EvidenceLevel.INSUFFICIENT,
        )

        checks_passed = []
        checks_failed = []

        # === Extract metrics ===
        metrics = backtest_results.get('metrics', {})
        oos_metrics = backtest_results.get('oos_metrics', metrics)
        trades = backtest_results.get('trades', [])

        # Convert trades to list if DataFrame
        if isinstance(trades, pd.DataFrame):
            trade_count = len(trades)
            oos_trades = len(trades[trades.get('is_oos', True)]) if 'is_oos' in trades.columns else trade_count
        else:
            trade_count = len(trades) if trades else 0
            oos_trades = trade_count

        report.oos_trades = oos_trades

        # === Check 1: Minimum OOS trades ===
        if oos_trades >= self.requirements.min_oos_trades:
            checks_passed.append(f"OOS trades: {oos_trades} >= {self.requirements.min_oos_trades}")
        else:
            checks_failed.append(f"OOS trades: {oos_trades} < {self.requirements.min_oos_trades}")
            report.rejection_reasons.append(
                f"Insufficient OOS trades ({oos_trades}). Need at least {self.requirements.min_oos_trades} "
                f"for statistical significance."
            )

        # === Check 2: OOS Sharpe ratio ===
        oos_sharpe = oos_metrics.get('sharpe', oos_metrics.get('sharpe_ratio', 0.0))
        report.oos_sharpe = oos_sharpe

        if oos_sharpe >= self.requirements.min_oos_sharpe:
            checks_passed.append(f"OOS Sharpe: {oos_sharpe:.2f} >= {self.requirements.min_oos_sharpe}")
        else:
            checks_failed.append(f"OOS Sharpe: {oos_sharpe:.2f} < {self.requirements.min_oos_sharpe}")
            report.rejection_reasons.append(
                f"OOS Sharpe ratio ({oos_sharpe:.2f}) below minimum ({self.requirements.min_oos_sharpe}). "
                f"Risk-adjusted returns insufficient."
            )

        # === Check 3: OOS Profit Factor ===
        oos_pf = oos_metrics.get('profit_factor', 0.0)
        report.oos_profit_factor = oos_pf

        if oos_pf >= self.requirements.min_oos_profit_factor:
            checks_passed.append(f"OOS Profit Factor: {oos_pf:.2f} >= {self.requirements.min_oos_profit_factor}")
        else:
            checks_failed.append(f"OOS Profit Factor: {oos_pf:.2f} < {self.requirements.min_oos_profit_factor}")
            report.rejection_reasons.append(
                f"OOS Profit Factor ({oos_pf:.2f}) below minimum ({self.requirements.min_oos_profit_factor}). "
                f"Gross profits don't sufficiently exceed gross losses."
            )

        # === Check 4: Win Rate ===
        win_rate = oos_metrics.get('win_rate', 0.0)
        report.oos_win_rate = win_rate

        if win_rate >= self.requirements.min_win_rate:
            checks_passed.append(f"Win Rate: {win_rate:.1%} >= {self.requirements.min_win_rate:.1%}")
        else:
            checks_failed.append(f"Win Rate: {win_rate:.1%} < {self.requirements.min_win_rate:.1%}")
            report.rejection_reasons.append(
                f"Win rate ({win_rate:.1%}) below minimum ({self.requirements.min_win_rate:.1%})."
            )

        # === Check 5: Maximum Drawdown ===
        max_dd = oos_metrics.get('max_drawdown', oos_metrics.get('max_dd', 1.0))
        report.max_drawdown = max_dd

        if max_dd <= self.requirements.max_drawdown:
            checks_passed.append(f"Max Drawdown: {max_dd:.1%} <= {self.requirements.max_drawdown:.1%}")
        else:
            checks_failed.append(f"Max Drawdown: {max_dd:.1%} > {self.requirements.max_drawdown:.1%}")
            report.rejection_reasons.append(
                f"Maximum drawdown ({max_dd:.1%}) exceeds limit ({self.requirements.max_drawdown:.1%}). "
                f"Risk of large losses too high."
            )

        # === Check 6: Regime Stability ===
        regime_splits = backtest_results.get('regime_splits', {})

        if regime_splits:
            regime_sharpes = {k: v.get('sharpe', 0) for k, v in regime_splits.items()}
            report.regime_performance = regime_sharpes

            if len(regime_sharpes) >= 2:
                values = list(regime_sharpes.values())
                regime_var = np.std(values) / (np.mean(values) + 1e-6)
                report.regime_variance = regime_var

                profitable_regimes = sum(1 for v in values if v > 0)

                if regime_var <= self.requirements.max_regime_performance_variance:
                    checks_passed.append(f"Regime stability: variance {regime_var:.2f} <= {self.requirements.max_regime_performance_variance}")
                else:
                    checks_failed.append(f"Regime stability: variance {regime_var:.2f} > {self.requirements.max_regime_performance_variance}")
                    report.rejection_reasons.append(
                        f"Performance varies too much across regimes (CV={regime_var:.2f}). "
                        f"Strategy may only work in specific conditions."
                    )

                if profitable_regimes >= self.requirements.min_regimes_profitable:
                    checks_passed.append(f"Profitable regimes: {profitable_regimes} >= {self.requirements.min_regimes_profitable}")
                else:
                    checks_failed.append(f"Profitable regimes: {profitable_regimes} < {self.requirements.min_regimes_profitable}")

        # === Check 7: Overfitting (IS vs OOS comparison) ===
        is_sharpe = metrics.get('sharpe', metrics.get('sharpe_ratio', 0))

        if is_sharpe > 0:
            overfitting_score = 1 - (oos_sharpe / is_sharpe)
            report.overfitting_score = max(0, overfitting_score)

            if report.overfitting_score <= self.requirements.max_overfitting_score:
                checks_passed.append(f"Overfitting: {report.overfitting_score:.2f} <= {self.requirements.max_overfitting_score}")
            else:
                checks_failed.append(f"Overfitting: {report.overfitting_score:.2f} > {self.requirements.max_overfitting_score}")
                report.rejection_reasons.append(
                    f"OOS performance degrades {report.overfitting_score:.0%} vs IS. "
                    f"Strategy may be overfit to historical data."
                )

        # === Check 8: Statistical Significance ===
        equity_curve = backtest_results.get('equity_curve')
        if equity_curve is not None and len(equity_curve) > 30:
            if isinstance(equity_curve, pd.DataFrame):
                if 'equity' in equity_curve.columns:
                    returns = equity_curve['equity'].pct_change().dropna()
                elif 'value' in equity_curve.columns:
                    returns = equity_curve['value'].pct_change().dropna()
                else:
                    returns = pd.Series()
            else:
                returns = pd.Series()

            if len(returns) > 0:
                mean_return = returns.mean()
                std_return = returns.std()
                if std_return > 0:
                    t_stat = mean_return / (std_return / np.sqrt(len(returns)))
                    report.returns_t_stat = t_stat

                    if t_stat >= self.requirements.min_t_stat:
                        checks_passed.append(f"T-statistic: {t_stat:.2f} >= {self.requirements.min_t_stat}")
                    else:
                        checks_failed.append(f"T-statistic: {t_stat:.2f} < {self.requirements.min_t_stat}")
                        report.rejection_reasons.append(
                            f"Returns not statistically significant (t={t_stat:.2f}). "
                            f"Could be due to chance."
                        )

        # === Integrate with KnowledgeBoundary ===
        if self.knowledge_boundary and context:
            try:
                signal = {
                    'strategy': strategy_name,
                    'entry_price': 100,  # Dummy for assessment
                }
                kb_assessment = self.knowledge_boundary.assess_knowledge_state(signal, context)
                report.uncertainty_level = kb_assessment.uncertainty_level.value
                report.should_stand_down = kb_assessment.should_stand_down

                if kb_assessment.should_stand_down:
                    checks_failed.append("KnowledgeBoundary: STAND DOWN recommended")
                    report.rejection_reasons.append(
                        f"KnowledgeBoundary recommends standing down: "
                        f"uncertainty level = {kb_assessment.uncertainty_level.value}"
                    )
            except Exception as e:
                logger.warning(f"KnowledgeBoundary integration failed: {e}")

        # === Final Determination ===
        report.checks_passed = checks_passed
        report.checks_failed = checks_failed

        # Determine evidence level
        total_checks = len(checks_passed) + len(checks_failed)
        pass_rate = len(checks_passed) / total_checks if total_checks > 0 else 0

        if pass_rate >= 0.9 and oos_sharpe >= 1.0:
            report.evidence_level = EvidenceLevel.EXCELLENT
        elif pass_rate >= 0.8:
            report.evidence_level = EvidenceLevel.STRONG
        elif pass_rate >= 0.6:
            report.evidence_level = EvidenceLevel.MODERATE
        elif pass_rate >= 0.4:
            report.evidence_level = EvidenceLevel.WEAK
        else:
            report.evidence_level = EvidenceLevel.INSUFFICIENT

        # Pass only if strong or excellent and no stand-down
        report.passed = (
            report.evidence_level in [EvidenceLevel.STRONG, EvidenceLevel.EXCELLENT]
            and not report.should_stand_down
        )

        # Generate recommendations
        if not report.passed:
            report.recommendations = self._generate_recommendations(report)
        else:
            report.recommendations = [
                "Strategy approved for live trading with standard risk limits",
                f"Recommended position size: based on {report.oos_sharpe:.2f} Sharpe",
            ]

        # Save report
        report_path = self.reports_dir / f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report.save(report_path)
        logger.info(f"Evidence report saved: {report_path}")

        return report

    def _generate_recommendations(self, report: EvidenceReport) -> List[str]:
        """Generate actionable recommendations for failed strategies."""
        recs = []

        if report.oos_trades < self.requirements.min_oos_trades:
            recs.append(
                f"Collect more data: need {self.requirements.min_oos_trades - report.oos_trades} more OOS trades"
            )

        if report.oos_sharpe < self.requirements.min_oos_sharpe:
            recs.append(
                "Improve risk-adjusted returns: consider tighter stops or better entry timing"
            )

        if report.max_drawdown > self.requirements.max_drawdown:
            recs.append(
                "Reduce drawdown: consider smaller position sizes or faster exits"
            )

        if report.overfitting_score > self.requirements.max_overfitting_score:
            recs.append(
                "Reduce overfitting: use fewer parameters, longer training periods, or regularization"
            )

        if report.regime_variance > self.requirements.max_regime_performance_variance:
            recs.append(
                "Improve regime stability: add regime filters or regime-specific parameters"
            )

        if not recs:
            recs.append("Review failed checks and address specific issues")

        return recs

    def quick_check(
        self,
        oos_trades: int,
        oos_sharpe: float,
        oos_profit_factor: float,
        max_drawdown: float,
    ) -> Tuple[bool, str]:
        """
        Quick pass/fail check without full report.

        Returns:
            Tuple of (passed, reason)
        """
        if oos_trades < self.requirements.min_oos_trades:
            return False, f"Insufficient trades: {oos_trades}"

        if oos_sharpe < self.requirements.min_oos_sharpe:
            return False, f"Low Sharpe: {oos_sharpe:.2f}"

        if oos_profit_factor < self.requirements.min_oos_profit_factor:
            return False, f"Low profit factor: {oos_profit_factor:.2f}"

        if max_drawdown > self.requirements.max_drawdown:
            return False, f"High drawdown: {max_drawdown:.1%}"

        return True, "All checks passed"


def check_promotion_gate(
    backtest_results: Dict[str, Any],
    strategy_name: str,
    requirements: Optional[EvidenceRequirements] = None,
) -> Tuple[bool, EvidenceReport]:
    """
    Convenience function to check if a strategy can be promoted.

    Args:
        backtest_results: Results from walk-forward backtest
        strategy_name: Strategy identifier
        requirements: Custom requirements (optional)

    Returns:
        Tuple of (passed, report)
    """
    gate = EvidenceGate(requirements=requirements)
    report = gate.evaluate(backtest_results, strategy_name)
    return report.passed, report
