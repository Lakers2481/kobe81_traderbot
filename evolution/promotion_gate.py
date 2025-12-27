"""
Promotion Gate for Strategy Evolution
======================================

Controls which evolved strategies get promoted to production.
Only strategies passing rigorous walk-forward validation
and meeting performance criteria are promoted.

Prevents overfitting by requiring out-of-sample performance
and stability checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class PromotionStatus(Enum):
    """Status of a promotion attempt."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    DEFERRED = "deferred"  # Needs more data


@dataclass
class PromotionCriteria:
    """Criteria for strategy promotion."""
    # Minimum performance thresholds
    min_sharpe: float = 0.5
    min_profit_factor: float = 1.2
    min_win_rate: float = 0.45
    min_trades: int = 30

    # Maximum risk thresholds
    max_drawdown: float = 0.20
    max_consecutive_losses: int = 10

    # Stability requirements
    min_oos_sharpe_ratio: float = 0.6  # OOS Sharpe must be >= 60% of IS
    max_param_sensitivity: float = 0.3  # Performance shouldn't vary >30% with small param changes

    # Walk-forward requirements
    min_wf_splits: int = 4
    min_profitable_splits: float = 0.6  # At least 60% of splits profitable

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'min_sharpe': self.min_sharpe,
            'min_profit_factor': self.min_profit_factor,
            'min_win_rate': self.min_win_rate,
            'min_trades': self.min_trades,
            'max_drawdown': self.max_drawdown,
            'max_consecutive_losses': self.max_consecutive_losses,
            'min_oos_sharpe_ratio': self.min_oos_sharpe_ratio,
            'max_param_sensitivity': self.max_param_sensitivity,
            'min_wf_splits': self.min_wf_splits,
            'min_profitable_splits': self.min_profitable_splits,
        }


@dataclass
class PromotionResult:
    """Result of a promotion check."""
    strategy_name: str
    status: PromotionStatus = PromotionStatus.PENDING
    checked_at: datetime = field(default_factory=datetime.now)

    # Check results
    passed_checks: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Recommendation
    recommendation: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'status': self.status.value,
            'checked_at': self.checked_at.isoformat(),
            'passed_checks': self.passed_checks,
            'failed_checks': self.failed_checks,
            'warnings': self.warnings,
            'metrics': self.metrics,
            'recommendation': self.recommendation,
            'confidence': self.confidence,
        }

    @property
    def passed(self) -> bool:
        """Whether the strategy passed promotion."""
        return self.status == PromotionStatus.PASSED


@dataclass
class WalkForwardResult:
    """Result from a walk-forward test split."""
    split_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    # In-sample metrics
    is_sharpe: float = 0.0
    is_profit_factor: float = 0.0
    is_win_rate: float = 0.0
    is_trades: int = 0

    # Out-of-sample metrics
    oos_sharpe: float = 0.0
    oos_profit_factor: float = 0.0
    oos_win_rate: float = 0.0
    oos_trades: int = 0
    oos_return: float = 0.0

    @property
    def is_profitable(self) -> bool:
        """Whether OOS was profitable."""
        return self.oos_return > 0


class PromotionGate:
    """
    Controls strategy promotion to production.

    Requires strategies to pass multiple validation checks
    including walk-forward testing and stability analysis.
    """

    def __init__(
        self,
        criteria: Optional[PromotionCriteria] = None,
        require_wf: bool = True,
    ):
        """
        Initialize the promotion gate.

        Args:
            criteria: Promotion criteria (uses defaults if None)
            require_wf: Whether to require walk-forward results
        """
        self.criteria = criteria or PromotionCriteria()
        self.require_wf = require_wf

        # History of promotion attempts
        self._history: List[PromotionResult] = []

        logger.info(
            f"PromotionGate initialized with "
            f"min_sharpe={self.criteria.min_sharpe}, "
            f"min_trades={self.criteria.min_trades}"
        )

    def _check_performance(
        self,
        metrics: Dict[str, Any],
        result: PromotionResult,
    ) -> bool:
        """Check basic performance criteria."""
        all_passed = True

        # Sharpe ratio
        sharpe = metrics.get('sharpe', 0)
        result.metrics['sharpe'] = sharpe
        if sharpe >= self.criteria.min_sharpe:
            result.passed_checks.append(f"Sharpe ratio: {sharpe:.2f} >= {self.criteria.min_sharpe}")
        else:
            result.failed_checks.append(f"Sharpe ratio: {sharpe:.2f} < {self.criteria.min_sharpe}")
            all_passed = False

        # Profit factor
        pf = metrics.get('profit_factor', 0)
        result.metrics['profit_factor'] = pf
        if pf >= self.criteria.min_profit_factor:
            result.passed_checks.append(f"Profit factor: {pf:.2f} >= {self.criteria.min_profit_factor}")
        else:
            result.failed_checks.append(f"Profit factor: {pf:.2f} < {self.criteria.min_profit_factor}")
            all_passed = False

        # Win rate
        wr = metrics.get('win_rate', 0)
        result.metrics['win_rate'] = wr
        if wr >= self.criteria.min_win_rate:
            result.passed_checks.append(f"Win rate: {wr:.2%} >= {self.criteria.min_win_rate:.2%}")
        else:
            result.failed_checks.append(f"Win rate: {wr:.2%} < {self.criteria.min_win_rate:.2%}")
            all_passed = False

        # Trade count
        trades = metrics.get('total_trades', metrics.get('trades', 0))
        result.metrics['total_trades'] = trades
        if trades >= self.criteria.min_trades:
            result.passed_checks.append(f"Trade count: {trades} >= {self.criteria.min_trades}")
        else:
            result.failed_checks.append(f"Trade count: {trades} < {self.criteria.min_trades}")
            all_passed = False

        # Drawdown
        dd = metrics.get('max_drawdown', 0)
        result.metrics['max_drawdown'] = dd
        if dd <= self.criteria.max_drawdown:
            result.passed_checks.append(f"Max drawdown: {dd:.2%} <= {self.criteria.max_drawdown:.2%}")
        else:
            result.failed_checks.append(f"Max drawdown: {dd:.2%} > {self.criteria.max_drawdown:.2%}")
            all_passed = False

        return all_passed

    def _check_walk_forward(
        self,
        wf_results: List[WalkForwardResult],
        result: PromotionResult,
    ) -> bool:
        """Check walk-forward validation results."""
        if not wf_results:
            if self.require_wf:
                result.failed_checks.append("No walk-forward results provided")
                return False
            else:
                result.warnings.append("Walk-forward not performed")
                return True

        all_passed = True

        # Check number of splits
        n_splits = len(wf_results)
        result.metrics['wf_splits'] = n_splits
        if n_splits >= self.criteria.min_wf_splits:
            result.passed_checks.append(f"WF splits: {n_splits} >= {self.criteria.min_wf_splits}")
        else:
            result.failed_checks.append(f"WF splits: {n_splits} < {self.criteria.min_wf_splits}")
            all_passed = False

        # Check profitable splits ratio
        profitable = sum(1 for r in wf_results if r.is_profitable)
        profitable_ratio = profitable / n_splits if n_splits > 0 else 0
        result.metrics['profitable_splits_ratio'] = profitable_ratio
        if profitable_ratio >= self.criteria.min_profitable_splits:
            result.passed_checks.append(
                f"Profitable splits: {profitable_ratio:.2%} >= {self.criteria.min_profitable_splits:.2%}"
            )
        else:
            result.failed_checks.append(
                f"Profitable splits: {profitable_ratio:.2%} < {self.criteria.min_profitable_splits:.2%}"
            )
            all_passed = False

        # Check OOS Sharpe ratio relative to IS
        is_sharpes = [r.is_sharpe for r in wf_results if r.is_sharpe > 0]
        oos_sharpes = [r.oos_sharpe for r in wf_results]

        if is_sharpes and oos_sharpes:
            avg_is_sharpe = np.mean(is_sharpes)
            avg_oos_sharpe = np.mean(oos_sharpes)
            oos_ratio = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe > 0 else 0

            result.metrics['avg_is_sharpe'] = avg_is_sharpe
            result.metrics['avg_oos_sharpe'] = avg_oos_sharpe
            result.metrics['oos_is_sharpe_ratio'] = oos_ratio

            if oos_ratio >= self.criteria.min_oos_sharpe_ratio:
                result.passed_checks.append(
                    f"OOS/IS Sharpe ratio: {oos_ratio:.2%} >= {self.criteria.min_oos_sharpe_ratio:.2%}"
                )
            else:
                result.failed_checks.append(
                    f"OOS/IS Sharpe ratio: {oos_ratio:.2%} < {self.criteria.min_oos_sharpe_ratio:.2%}"
                )
                all_passed = False

        # Aggregate OOS trades
        total_oos_trades = sum(r.oos_trades for r in wf_results)
        result.metrics['total_oos_trades'] = total_oos_trades

        return all_passed

    def _check_stability(
        self,
        sensitivity_results: Optional[Dict[str, float]],
        result: PromotionResult,
    ) -> bool:
        """Check parameter sensitivity/stability."""
        if sensitivity_results is None:
            result.warnings.append("Parameter sensitivity not tested")
            return True

        max_variation = sensitivity_results.get('max_variation', 0)
        result.metrics['param_sensitivity'] = max_variation

        if max_variation <= self.criteria.max_param_sensitivity:
            result.passed_checks.append(
                f"Parameter sensitivity: {max_variation:.2%} <= {self.criteria.max_param_sensitivity:.2%}"
            )
            return True
        else:
            result.failed_checks.append(
                f"Parameter sensitivity: {max_variation:.2%} > {self.criteria.max_param_sensitivity:.2%}"
            )
            return False

    def check_promotion(
        self,
        strategy_name: str,
        metrics: Dict[str, Any],
        wf_results: Optional[List[WalkForwardResult]] = None,
        sensitivity_results: Optional[Dict[str, float]] = None,
    ) -> PromotionResult:
        """
        Check if a strategy should be promoted.

        Args:
            strategy_name: Name of the strategy
            metrics: Performance metrics
            wf_results: Walk-forward test results
            sensitivity_results: Parameter sensitivity analysis

        Returns:
            PromotionResult with status and details
        """
        result = PromotionResult(strategy_name=strategy_name)

        # Run all checks
        perf_passed = self._check_performance(metrics, result)
        wf_passed = self._check_walk_forward(wf_results or [], result)
        stability_passed = self._check_stability(sensitivity_results, result)

        # Determine overall status
        if perf_passed and wf_passed and stability_passed:
            result.status = PromotionStatus.PASSED
            result.recommendation = "Strategy approved for production deployment"
            result.confidence = 0.8 + 0.1 * (len(result.passed_checks) / 10)
        elif len(result.failed_checks) == 0:
            result.status = PromotionStatus.DEFERRED
            result.recommendation = "Strategy needs more data for evaluation"
            result.confidence = 0.5
        else:
            result.status = PromotionStatus.FAILED
            failures = ", ".join(result.failed_checks[:3])
            result.recommendation = f"Strategy failed: {failures}"
            result.confidence = 0.2

        # Record in history
        self._history.append(result)

        logger.info(
            f"Promotion check for {strategy_name}: {result.status.value} "
            f"({len(result.passed_checks)} passed, {len(result.failed_checks)} failed)"
        )

        return result

    def get_history(
        self,
        strategy_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[PromotionResult]:
        """Get promotion history, optionally filtered by strategy."""
        history = self._history
        if strategy_name:
            history = [r for r in history if r.strategy_name == strategy_name]
        return history[-limit:]

    def get_promoted_strategies(self) -> List[str]:
        """Get list of strategies that have passed promotion."""
        promoted = set()
        for result in self._history:
            if result.status == PromotionStatus.PASSED:
                promoted.add(result.strategy_name)
        return list(promoted)


def check_promotion(
    strategy_name: str,
    metrics: Dict[str, Any],
    wf_results: Optional[List[WalkForwardResult]] = None,
) -> PromotionResult:
    """Convenience function to check strategy promotion."""
    gate = PromotionGate()
    return gate.check_promotion(strategy_name, metrics, wf_results)


# Module-level gate instance
_gate: Optional[PromotionGate] = None


def get_gate() -> PromotionGate:
    """Get or create the global promotion gate instance."""
    global _gate
    if _gate is None:
        _gate = PromotionGate()
    return _gate
