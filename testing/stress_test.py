"""
Stress Testing for Trading Strategies
======================================

Simulates adverse market conditions to test strategy robustness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Type of stress scenario."""
    CRASH = "crash"           # Sudden market crash
    RALLY = "rally"           # Sharp rally
    VOLATILITY_SPIKE = "volatility_spike"
    VOLATILITY_CRUSH = "volatility_crush"
    CORRELATION_BREAK = "correlation_break"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    FLASH_CRASH = "flash_crash"
    SIDEWAYS = "sideways"     # Extended choppy market


@dataclass
class StressScenario:
    """Definition of a stress scenario."""
    name: str
    scenario_type: ScenarioType
    description: str = ""

    # Return modifications
    return_shock: float = 0.0  # One-time return shock
    volatility_mult: float = 1.0  # Multiply volatility
    trend_bias: float = 0.0  # Daily bias

    # Duration
    duration_days: int = 5

    # Spread/liquidity
    spread_mult: float = 1.0
    slippage_mult: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.scenario_type.value,
            'return_shock': self.return_shock,
            'volatility_mult': self.volatility_mult,
            'duration_days': self.duration_days,
        }


@dataclass
class StressTestResult:
    """Result of stress test."""
    scenario: StressScenario
    original_return: float
    stressed_return: float
    original_sharpe: float
    stressed_sharpe: float
    max_drawdown: float
    survived: bool
    details: Dict[str, Any] = field(default_factory=dict)


class StressTester:
    """
    Applies stress scenarios to trading strategies.
    """

    # Standard stress scenarios
    STANDARD_SCENARIOS = [
        StressScenario(
            name="Black Monday",
            scenario_type=ScenarioType.CRASH,
            return_shock=-0.20,
            volatility_mult=3.0,
            duration_days=3,
        ),
        StressScenario(
            name="2020 COVID Crash",
            scenario_type=ScenarioType.CRASH,
            return_shock=-0.35,
            volatility_mult=4.0,
            duration_days=20,
        ),
        StressScenario(
            name="Flash Crash",
            scenario_type=ScenarioType.FLASH_CRASH,
            return_shock=-0.10,
            volatility_mult=5.0,
            duration_days=1,
        ),
        StressScenario(
            name="VIX Spike",
            scenario_type=ScenarioType.VOLATILITY_SPIKE,
            volatility_mult=3.0,
            duration_days=10,
        ),
        StressScenario(
            name="Sideways Chop",
            scenario_type=ScenarioType.SIDEWAYS,
            volatility_mult=0.5,
            duration_days=60,
        ),
        StressScenario(
            name="Bull Rally",
            scenario_type=ScenarioType.RALLY,
            return_shock=0.15,
            trend_bias=0.002,
            duration_days=20,
        ),
    ]

    def __init__(self, ruin_threshold: float = -0.50):
        self.ruin_threshold = ruin_threshold
        logger.info("StressTester initialized")

    def apply_scenario(
        self,
        returns: pd.Series,
        scenario: StressScenario,
    ) -> pd.Series:
        """Apply stress scenario to return series."""
        stressed = returns.copy()
        n = len(stressed)

        # Apply shock at start
        if scenario.return_shock != 0:
            shock_days = min(scenario.duration_days, n)
            stressed.iloc[:shock_days] = (
                stressed.iloc[:shock_days] * scenario.volatility_mult +
                scenario.return_shock / shock_days
            )

        # Apply volatility multiplier
        if scenario.volatility_mult != 1.0:
            mean = stressed.mean()
            stressed = (stressed - mean) * scenario.volatility_mult + mean

        # Apply trend bias
        if scenario.trend_bias != 0:
            stressed = stressed + scenario.trend_bias

        return stressed

    def run_test(
        self,
        returns: pd.Series,
        scenario: StressScenario,
    ) -> StressTestResult:
        """Run single stress test."""
        # Original metrics
        orig_total = (1 + returns).prod() - 1
        orig_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Apply stress
        stressed = self.apply_scenario(returns, scenario)
        stressed_total = (1 + stressed).prod() - 1
        stressed_sharpe = stressed.mean() / stressed.std() * np.sqrt(252) if stressed.std() > 0 else 0

        # Max drawdown
        cumulative = (1 + stressed).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        survived = stressed_total > self.ruin_threshold

        return StressTestResult(
            scenario=scenario,
            original_return=orig_total,
            stressed_return=stressed_total,
            original_sharpe=orig_sharpe,
            stressed_sharpe=stressed_sharpe,
            max_drawdown=max_dd,
            survived=survived,
        )

    def run_all_scenarios(
        self,
        returns: pd.Series,
    ) -> List[StressTestResult]:
        """Run all standard stress scenarios."""
        results = []
        for scenario in self.STANDARD_SCENARIOS:
            result = self.run_test(returns, scenario)
            results.append(result)
        return results


def run_stress_test(
    returns: pd.Series,
    scenario_name: str = "Black Monday",
) -> StressTestResult:
    """Convenience function for stress testing."""
    tester = StressTester()
    scenario = next(
        (s for s in tester.STANDARD_SCENARIOS if s.name == scenario_name),
        tester.STANDARD_SCENARIOS[0]
    )
    return tester.run_test(returns, scenario)


def get_standard_scenarios() -> List[StressScenario]:
    """Get list of standard stress scenarios."""
    return StressTester.STANDARD_SCENARIOS.copy()
