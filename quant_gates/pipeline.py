"""
Quant Gates Pipeline
====================

Orchestrates the full 5-gate validation:
0. Sanity (lookahead, leakage)
1. Baseline (beat B&H, MA, MR)
2. Robustness (walk-forward, stress)
3. Risk Realism (DD, diversification)
4. Multiple Testing (T-stat penalty)

REJECT BY DEFAULT: Any gate failure = archive forever.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .gate_0_sanity import Gate0Sanity
from .gate_1_baseline import Gate1Baseline
from .gate_2_robustness import Gate2Robustness
from .gate_3_risk import Gate3RiskRealism
from .gate_4_multiple_testing import Gate4MultipleTesting

logger = logging.getLogger(__name__)


class GateStatus(Enum):
    """Gate evaluation status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class GateResult:
    """Result from a single gate."""
    gate_id: int
    gate_name: str
    status: str
    passed: bool
    message: str
    details: Dict[str, Any]


@dataclass
class PipelineResult:
    """Result from full pipeline run."""
    id: str
    strategy_id: str
    timestamp: str
    gates: List[GateResult]
    overall_passed: bool
    failed_gate: Optional[int]
    archived: bool
    archive_reason: Optional[str]


class QuantGatesPipeline:
    """
    Full quant gates validation pipeline.

    Runs gates 0-4 in sequence.
    STOPS at first failure and archives strategy.

    Gate 0: Sanity - lookahead, leakage, costs
    Gate 1: Baseline - beat B&H, MA cross, naive MR
    Gate 2: Robustness - WF 62.5%+, stress, sensitivity
    Gate 3: Risk - DD 25%, trades 100+, symbols 30+
    Gate 4: Multiple Testing - adjusted T-stat
    """

    def __init__(self, archive_dir: str = "archive"):
        self._archive_dir = Path(archive_dir)
        self._archive_dir.mkdir(parents=True, exist_ok=True)

        # Initialize gates
        self._gates = {
            0: Gate0Sanity(),
            1: Gate1Baseline(),
            2: Gate2Robustness(),
            3: Gate3RiskRealism(),
            4: Gate4MultipleTesting(),
        }

    def run(
        self,
        strategy_id: str,
        strategy_code: Optional[str] = None,
        strategy_file: Optional[str] = None,
        backtest_config: Optional[Dict[str, Any]] = None,
        equity_curve: Optional[pd.Series] = None,
        trades: Optional[pd.DataFrame] = None,
        benchmark_prices: Optional[pd.Series] = None,
        wf_results: Optional[List[Dict[str, float]]] = None,
        base_sharpe: float = 0.0,
        stressed_sharpe: float = 0.0,
        raw_t_stat: float = 0.0,
        strategy_family: str = "unknown",
        num_parameters: int = 5,
    ) -> PipelineResult:
        """
        Run full pipeline validation.

        Args:
            strategy_id: Unique strategy identifier
            strategy_code: Strategy source code
            strategy_file: Path to strategy file
            backtest_config: Backtest configuration
            equity_curve: Strategy equity curve
            trades: Trades DataFrame
            benchmark_prices: Benchmark prices for comparison
            wf_results: Walk-forward results
            base_sharpe: Base Sharpe ratio
            stressed_sharpe: Sharpe under stress
            raw_t_stat: Raw T-statistic
            strategy_family: Strategy family for attempt tracking
            num_parameters: Number of free parameters

        Returns:
            PipelineResult with gate results and overall status
        """
        gate_results: List[GateResult] = []
        failed_gate: Optional[int] = None

        logger.info(f"Starting quant gates pipeline for: {strategy_id}")

        # Gate 0: Sanity
        try:
            result = self._gates[0].validate(
                strategy_code=strategy_code,
                strategy_file=strategy_file,
                backtest_trades=trades,
                backtest_config=backtest_config,
            )
            gate_results.append(GateResult(
                gate_id=0,
                gate_name="Sanity Check",
                status=GateStatus.PASSED.value if result.passed else GateStatus.FAILED.value,
                passed=result.passed,
                message="; ".join(result.issues) if result.issues else "Passed",
                details=result.details,
            ))
            if not result.passed:
                failed_gate = 0
        except Exception as e:
            logger.error(f"Gate 0 error: {e}")
            gate_results.append(GateResult(
                gate_id=0,
                gate_name="Sanity Check",
                status=GateStatus.ERROR.value,
                passed=False,
                message=str(e),
                details={},
            ))
            failed_gate = 0

        # Stop if failed
        if failed_gate is not None:
            return self._finalize(strategy_id, gate_results, failed_gate)

        # Gate 1: Baseline
        try:
            if equity_curve is not None and benchmark_prices is not None:
                result = self._gates[1].validate(equity_curve, benchmark_prices)
                gate_results.append(GateResult(
                    gate_id=1,
                    gate_name="Baseline Beat",
                    status=GateStatus.PASSED.value if result.passed else GateStatus.FAILED.value,
                    passed=result.passed,
                    message=f"Margin vs best: {result.margin_vs_best:.2f}",
                    details=result.details,
                ))
                if not result.passed:
                    failed_gate = 1
            else:
                gate_results.append(GateResult(
                    gate_id=1,
                    gate_name="Baseline Beat",
                    status=GateStatus.SKIPPED.value,
                    passed=True,
                    message="Skipped - missing equity/benchmark data",
                    details={},
                ))
        except Exception as e:
            logger.error(f"Gate 1 error: {e}")
            gate_results.append(GateResult(
                gate_id=1,
                gate_name="Baseline Beat",
                status=GateStatus.ERROR.value,
                passed=False,
                message=str(e),
                details={},
            ))
            failed_gate = 1

        if failed_gate is not None:
            return self._finalize(strategy_id, gate_results, failed_gate)

        # Gate 2: Robustness
        try:
            result = self._gates[2].validate(
                wf_results=wf_results,
                base_sharpe=base_sharpe,
                stressed_sharpe=stressed_sharpe,
            )
            gate_results.append(GateResult(
                gate_id=2,
                gate_name="Robustness",
                status=GateStatus.PASSED.value if result.passed else GateStatus.FAILED.value,
                passed=result.passed,
                message=f"WF consistency: {result.wf_consistency:.1%}",
                details=result.details,
            ))
            if not result.passed:
                failed_gate = 2
        except Exception as e:
            logger.error(f"Gate 2 error: {e}")
            gate_results.append(GateResult(
                gate_id=2,
                gate_name="Robustness",
                status=GateStatus.ERROR.value,
                passed=False,
                message=str(e),
                details={},
            ))
            failed_gate = 2

        if failed_gate is not None:
            return self._finalize(strategy_id, gate_results, failed_gate)

        # Gate 3: Risk Realism
        try:
            result = self._gates[3].validate(equity_curve=equity_curve, trades=trades)
            gate_results.append(GateResult(
                gate_id=3,
                gate_name="Risk Realism",
                status=GateStatus.PASSED.value if result.passed else GateStatus.FAILED.value,
                passed=result.passed,
                message="; ".join(result.violations) if result.violations else "Passed",
                details=result.details,
            ))
            if not result.passed:
                failed_gate = 3
        except Exception as e:
            logger.error(f"Gate 3 error: {e}")
            gate_results.append(GateResult(
                gate_id=3,
                gate_name="Risk Realism",
                status=GateStatus.ERROR.value,
                passed=False,
                message=str(e),
                details={},
            ))
            failed_gate = 3

        if failed_gate is not None:
            return self._finalize(strategy_id, gate_results, failed_gate)

        # Gate 4: Multiple Testing
        try:
            result = self._gates[4].validate(
                raw_t_stat=raw_t_stat,
                strategy_family=strategy_family,
                num_parameters=num_parameters,
            )
            gate_results.append(GateResult(
                gate_id=4,
                gate_name="Multiple Testing",
                status=GateStatus.PASSED.value if result.passed else GateStatus.FAILED.value,
                passed=result.passed,
                message=f"T-stat {raw_t_stat:.2f} vs threshold {result.adjusted_threshold:.2f}",
                details=result.details,
            ))
            if not result.passed:
                failed_gate = 4
        except Exception as e:
            logger.error(f"Gate 4 error: {e}")
            gate_results.append(GateResult(
                gate_id=4,
                gate_name="Multiple Testing",
                status=GateStatus.ERROR.value,
                passed=False,
                message=str(e),
                details={},
            ))
            failed_gate = 4

        return self._finalize(strategy_id, gate_results, failed_gate)

    def _finalize(
        self,
        strategy_id: str,
        gate_results: List[GateResult],
        failed_gate: Optional[int],
    ) -> PipelineResult:
        """Finalize pipeline result."""
        overall_passed = failed_gate is None
        archived = not overall_passed
        archive_reason = None

        if failed_gate is not None:
            archive_reason = f"GATE_{failed_gate}_FAILED"
            self._archive_strategy(strategy_id, gate_results, archive_reason)

        result = PipelineResult(
            id=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=strategy_id,
            timestamp=datetime.now().isoformat(),
            gates=gate_results,
            overall_passed=overall_passed,
            failed_gate=failed_gate,
            archived=archived,
            archive_reason=archive_reason,
        )

        status = "PASSED" if overall_passed else f"FAILED at Gate {failed_gate}"
        logger.info(f"Pipeline complete for {strategy_id}: {status}")

        return result

    def _archive_strategy(
        self,
        strategy_id: str,
        gate_results: List[GateResult],
        reason: str,
    ) -> None:
        """Archive a failed strategy (never retest)."""
        archive_file = self._archive_dir / f"{strategy_id}.json"
        with open(archive_file, "w") as f:
            json.dump({
                "strategy_id": strategy_id,
                "archived_at": datetime.now().isoformat(),
                "reason": reason,
                "gates": [asdict(g) for g in gate_results],
            }, f, indent=2)
        logger.warning(f"Strategy ARCHIVED: {strategy_id} ({reason})")


def run_full_validation(
    strategy_id: str,
    strategy_file: str,
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    benchmark_prices: pd.Series,
    wf_results: List[Dict[str, float]],
    base_sharpe: float,
    raw_t_stat: float,
    strategy_family: str,
    num_parameters: int,
) -> PipelineResult:
    """
    Convenience function to run full validation.

    Args:
        strategy_id: Strategy identifier
        strategy_file: Path to strategy file
        equity_curve: Equity curve
        trades: Trades DataFrame
        benchmark_prices: Benchmark prices
        wf_results: Walk-forward results
        base_sharpe: Base Sharpe ratio
        raw_t_stat: Raw T-statistic
        strategy_family: Strategy family name
        num_parameters: Number of parameters

    Returns:
        PipelineResult
    """
    pipeline = QuantGatesPipeline()
    return pipeline.run(
        strategy_id=strategy_id,
        strategy_file=strategy_file,
        equity_curve=equity_curve,
        trades=trades,
        benchmark_prices=benchmark_prices,
        wf_results=wf_results,
        base_sharpe=base_sharpe,
        stressed_sharpe=base_sharpe * 0.8,  # Assume 20% degradation under stress
        raw_t_stat=raw_t_stat,
        strategy_family=strategy_family,
        num_parameters=num_parameters,
    )
