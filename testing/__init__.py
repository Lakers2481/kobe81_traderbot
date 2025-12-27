"""
Synthetic & Adversarial Testing Module
======================================

Stress testing, Monte Carlo simulation, and adversarial
scenario generation for robust strategy validation.

Components:
- MonteCarloSimulator: Monte Carlo simulation for returns
- StressScenario: Predefined stress test scenarios
- AdversarialGenerator: Generate adverse market conditions
"""

from .monte_carlo import (
    MonteCarloSimulator,
    SimulationResult,
    simulate_returns,
    run_monte_carlo,
)

from .stress_test import (
    StressTester,
    StressScenario,
    ScenarioType,
    run_stress_test,
    get_standard_scenarios,
)

__all__ = [
    'MonteCarloSimulator',
    'SimulationResult',
    'simulate_returns',
    'run_monte_carlo',
    'StressTester',
    'StressScenario',
    'ScenarioType',
    'run_stress_test',
    'get_standard_scenarios',
]
