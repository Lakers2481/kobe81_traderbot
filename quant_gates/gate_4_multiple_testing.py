"""
Gate 4: Multiple Testing Correction
====================================

Penalizes strategies for:
- Number of attempts (data snooping)
- Number of free parameters (overfitting)

Adjusted T-stat threshold:
  threshold = 2.0 + 0.1*(attempts/10) + 0.1*params

This prevents data mining bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

logger = logging.getLogger(__name__)


@dataclass
class MultipleTestingResult:
    """Result from multiple testing correction."""
    passed: bool
    raw_t_stat: float
    adjusted_threshold: float
    num_attempts: int
    num_parameters: int
    penalty_breakdown: Dict[str, float]
    details: Dict[str, Any]


class Gate4MultipleTesting:
    """
    Gate 4: Multiple testing penalty.

    Combats data mining bias by:
    1. Tracking attempts per strategy family
    2. Penalizing for free parameters
    3. Raising T-stat threshold accordingly

    Threshold = 2.0 + 0.1*(attempts/10) + 0.1*params

    Example:
    - 50 attempts, 8 params: 2.0 + 0.5 + 0.8 = 3.3 required T-stat

    FAIL = ARCHIVE
    """

    # Base T-stat threshold
    BASE_T_STAT = 2.0

    # Penalties
    ATTEMPT_PENALTY_PER_10 = 0.1
    PARAM_PENALTY_PER = 0.1

    # Registry file for tracking attempts
    REGISTRY_FILE = "state/strategy_attempts.json"

    def __init__(self, registry_path: Optional[str] = None):
        self._registry_path = Path(registry_path or self.REGISTRY_FILE)
        self._registry: Dict[str, int] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load attempt registry from file."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path) as f:
                    self._registry = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load attempt registry: {e}")

    def _save_registry(self) -> None:
        """Save attempt registry to file."""
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._registry_path, "w") as f:
                json.dump(self._registry, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save attempt registry: {e}")

    def record_attempt(self, strategy_family: str) -> int:
        """
        Record an attempt for a strategy family.

        Args:
            strategy_family: Family name (e.g., "ibs_rsi", "turtle_soup")

        Returns:
            New attempt count
        """
        self._registry[strategy_family] = self._registry.get(strategy_family, 0) + 1
        self._save_registry()
        return self._registry[strategy_family]

    def get_attempts(self, strategy_family: str) -> int:
        """Get attempt count for a strategy family."""
        return self._registry.get(strategy_family, 0)

    def calculate_threshold(
        self,
        num_attempts: int,
        num_parameters: int,
    ) -> float:
        """
        Calculate adjusted T-stat threshold.

        Args:
            num_attempts: Number of attempts for this strategy family
            num_parameters: Number of free parameters

        Returns:
            Adjusted T-stat threshold
        """
        attempt_penalty = self.ATTEMPT_PENALTY_PER_10 * (num_attempts / 10)
        param_penalty = self.PARAM_PENALTY_PER * num_parameters

        return self.BASE_T_STAT + attempt_penalty + param_penalty

    def validate(
        self,
        raw_t_stat: float,
        strategy_family: str,
        num_parameters: int,
        record_attempt: bool = True,
    ) -> MultipleTestingResult:
        """
        Validate strategy against multiple testing threshold.

        Args:
            raw_t_stat: Raw T-statistic from backtest
            strategy_family: Strategy family name
            num_parameters: Number of free parameters
            record_attempt: Whether to record this as an attempt

        Returns:
            MultipleTestingResult
        """
        # Record attempt if requested
        if record_attempt:
            self.record_attempt(strategy_family)

        num_attempts = self.get_attempts(strategy_family)

        # Calculate penalties
        attempt_penalty = self.ATTEMPT_PENALTY_PER_10 * (num_attempts / 10)
        param_penalty = self.PARAM_PENALTY_PER * num_parameters
        total_penalty = attempt_penalty + param_penalty

        # Calculate threshold
        threshold = self.BASE_T_STAT + total_penalty

        # Check if passes
        passed = raw_t_stat >= threshold

        return MultipleTestingResult(
            passed=passed,
            raw_t_stat=raw_t_stat,
            adjusted_threshold=threshold,
            num_attempts=num_attempts,
            num_parameters=num_parameters,
            penalty_breakdown={
                "base": self.BASE_T_STAT,
                "attempt_penalty": attempt_penalty,
                "param_penalty": param_penalty,
                "total_penalty": total_penalty,
            },
            details={
                "strategy_family": strategy_family,
                "passed": passed,
                "margin": raw_t_stat - threshold,
            },
        )


def check_multiple_testing(
    raw_t_stat: float,
    strategy_family: str,
    num_parameters: int,
) -> MultipleTestingResult:
    """
    Convenience function for multiple testing check.

    Args:
        raw_t_stat: Raw T-statistic
        strategy_family: Strategy family name
        num_parameters: Number of parameters

    Returns:
        MultipleTestingResult
    """
    gate = Gate4MultipleTesting()
    return gate.validate(raw_t_stat, strategy_family, num_parameters)
