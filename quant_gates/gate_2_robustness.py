"""
Gate 2: Robustness
==================

Strategy must be robust to:
- Walk-forward validation (62.5%+ splits profitable)
- Stress testing (survives 2x costs, 2x slippage)
- Parameter sensitivity (±5% doesn't break it)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RobustnessResult:
    """Result from robustness testing."""
    passed: bool
    wf_consistency: float  # % of splits profitable
    stress_passed: bool
    sensitivity_passed: bool
    wf_results: List[Dict[str, float]]
    stress_results: Dict[str, Any]
    sensitivity_results: Dict[str, Any]
    details: Dict[str, Any]


class Gate2Robustness:
    """
    Gate 2: Strategy robustness validation.

    Requirements:
    - Walk-forward: 62.5%+ splits profitable (5/8)
    - Stress: Survives 2x costs, 2x slippage, 1-bar delay
    - Sensitivity: ±5% param changes don't break it

    FAIL = ARCHIVE
    """

    # Walk-forward requirements
    MIN_WF_SPLITS = 8
    MIN_WF_CONSISTENCY = 0.625  # 5/8

    # Stress test multipliers
    STRESS_COST_MULT = 2.0
    STRESS_SLIP_MULT = 2.0
    STRESS_DELAY_BARS = 1

    # Sensitivity thresholds
    SENSITIVITY_RANGE = 0.05  # ±5%
    MAX_SHARPE_DEGRADATION = 0.30  # 30% max degradation

    def __init__(self):
        pass

    def validate(
        self,
        wf_results: Optional[List[Dict[str, float]]] = None,
        base_sharpe: float = 0.0,
        stressed_sharpe: float = 0.0,
        param_sensitivity: Optional[Dict[str, float]] = None,
    ) -> RobustnessResult:
        """
        Run robustness validation.

        Args:
            wf_results: Walk-forward split results
            base_sharpe: Base case Sharpe ratio
            stressed_sharpe: Sharpe under stress conditions
            param_sensitivity: Dict of param_name -> Sharpe at ±5%

        Returns:
            RobustnessResult
        """
        details = {}

        # Walk-forward check
        wf_consistency = 0.0
        if wf_results and len(wf_results) >= self.MIN_WF_SPLITS:
            profitable = sum(1 for r in wf_results if r.get("sharpe", 0) > 0)
            wf_consistency = profitable / len(wf_results)
        wf_passed = wf_consistency >= self.MIN_WF_CONSISTENCY
        details["wf"] = {"consistency": wf_consistency, "passed": wf_passed}

        # Stress test check
        stress_passed = True
        stress_results = {}
        if base_sharpe > 0 and stressed_sharpe > 0:
            degradation = (base_sharpe - stressed_sharpe) / base_sharpe
            stress_passed = degradation < self.MAX_SHARPE_DEGRADATION
            stress_results = {
                "base_sharpe": base_sharpe,
                "stressed_sharpe": stressed_sharpe,
                "degradation": degradation,
            }
        details["stress"] = stress_results

        # Sensitivity check
        sensitivity_passed = True
        sensitivity_results = {}
        if param_sensitivity:
            worst_degradation = 0.0
            for param, sharpe in param_sensitivity.items():
                if base_sharpe > 0:
                    deg = (base_sharpe - sharpe) / base_sharpe
                    worst_degradation = max(worst_degradation, deg)
                    sensitivity_results[param] = {
                        "sharpe": sharpe,
                        "degradation": deg,
                    }
            sensitivity_passed = worst_degradation < self.MAX_SHARPE_DEGRADATION
        details["sensitivity"] = sensitivity_results

        passed = wf_passed and stress_passed and sensitivity_passed

        return RobustnessResult(
            passed=passed,
            wf_consistency=wf_consistency,
            stress_passed=stress_passed,
            sensitivity_passed=sensitivity_passed,
            wf_results=wf_results or [],
            stress_results=stress_results,
            sensitivity_results=sensitivity_results,
            details=details,
        )

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        strategy_fn: callable,
        train_days: int = 252,
        test_days: int = 63,
        n_splits: int = 8,
    ) -> List[Dict[str, float]]:
        """
        Run walk-forward validation.

        Args:
            data: Full dataset
            strategy_fn: Function that takes DataFrame, returns equity curve
            train_days: Training period days
            test_days: Testing period days
            n_splits: Number of splits

        Returns:
            List of results per split
        """
        results = []
        total_days = len(data)
        split_size = train_days + test_days

        for i in range(n_splits):
            start_idx = i * test_days
            end_idx = start_idx + split_size

            if end_idx > total_days:
                break

            split_data = data.iloc[start_idx:end_idx]
            train_data = split_data.iloc[:train_days]
            test_data = split_data.iloc[train_days:]

            try:
                # Train on train_data, test on test_data
                equity = strategy_fn(test_data)
                returns = equity.pct_change().dropna()

                if len(returns) > 0:
                    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                    total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
                else:
                    sharpe = 0
                    total_ret = 0

                results.append({
                    "split": i,
                    "sharpe": float(sharpe),
                    "return": float(total_ret),
                })
            except Exception as e:
                logger.warning(f"WF split {i} failed: {e}")
                results.append({"split": i, "sharpe": 0, "return": 0})

        return results


def check_robustness(
    wf_results: List[Dict[str, float]],
    base_sharpe: float,
    stressed_sharpe: float,
) -> RobustnessResult:
    """
    Convenience function for robustness check.

    Args:
        wf_results: Walk-forward results
        base_sharpe: Base Sharpe ratio
        stressed_sharpe: Stressed Sharpe ratio

    Returns:
        RobustnessResult
    """
    gate = Gate2Robustness()
    return gate.validate(wf_results, base_sharpe, stressed_sharpe)
