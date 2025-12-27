"""
Monte Carlo Simulation for Trading Systems
===========================================

Simulates thousands of possible return paths to assess
strategy robustness and risk metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result of Monte Carlo simulation."""
    n_simulations: int
    n_periods: int

    # Distribution metrics
    mean_return: float
    median_return: float
    std_return: float

    # Risk metrics
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    max_drawdown_mean: float
    max_drawdown_worst: float

    # Outcome probabilities
    prob_positive: float
    prob_double: float  # Probability of doubling
    prob_ruin: float  # Probability of 50% loss

    # Percentiles
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'n_simulations': self.n_simulations,
            'n_periods': self.n_periods,
            'mean_return': self.mean_return,
            'median_return': self.median_return,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'max_drawdown_mean': self.max_drawdown_mean,
            'prob_positive': self.prob_positive,
            'prob_ruin': self.prob_ruin,
        }


class MonteCarloSimulator:
    """
    Monte Carlo simulator for trading returns.

    Generates multiple possible paths based on historical
    return distribution and calculates risk metrics.
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        random_seed: Optional[int] = None,
    ):
        self.n_simulations = n_simulations
        if random_seed is not None:
            np.random.seed(random_seed)
        logger.info(f"MonteCarloSimulator initialized with {n_simulations} simulations")

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from return series."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return float(np.min(drawdowns))

    def simulate(
        self,
        mean_return: float,
        std_return: float,
        n_periods: int = 252,
        initial_capital: float = 10000,
        distribution: str = "normal",
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Args:
            mean_return: Expected daily return
            std_return: Standard deviation of daily returns
            n_periods: Number of periods (e.g., 252 trading days)
            initial_capital: Starting capital
            distribution: "normal" or "student_t"

        Returns:
            SimulationResult with all metrics
        """
        # Generate random returns
        if distribution == "student_t":
            # Heavier tails
            returns = np.random.standard_t(df=5, size=(self.n_simulations, n_periods))
            returns = returns * std_return + mean_return
        else:
            returns = np.random.normal(mean_return, std_return, (self.n_simulations, n_periods))

        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns, axis=1)
        final_values = cumulative[:, -1] * initial_capital
        total_returns = final_values / initial_capital - 1

        # Calculate max drawdowns for each path
        max_drawdowns = np.array([self._calculate_max_drawdown(r) for r in returns])

        # Calculate VaR and CVaR
        var_95 = np.percentile(total_returns, 5)
        var_99 = np.percentile(total_returns, 1)
        cvar_95 = np.mean(total_returns[total_returns <= var_95])

        result = SimulationResult(
            n_simulations=self.n_simulations,
            n_periods=n_periods,
            mean_return=float(np.mean(total_returns)),
            median_return=float(np.median(total_returns)),
            std_return=float(np.std(total_returns)),
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            max_drawdown_mean=float(np.mean(max_drawdowns)),
            max_drawdown_worst=float(np.min(max_drawdowns)),
            prob_positive=float(np.mean(total_returns > 0)),
            prob_double=float(np.mean(total_returns > 1.0)),
            prob_ruin=float(np.mean(total_returns < -0.5)),
            percentile_5=float(np.percentile(total_returns, 5)),
            percentile_25=float(np.percentile(total_returns, 25)),
            percentile_75=float(np.percentile(total_returns, 75)),
            percentile_95=float(np.percentile(total_returns, 95)),
        )

        logger.info(
            f"Simulation complete: mean={result.mean_return:.2%}, "
            f"VaR95={result.var_95:.2%}, prob_positive={result.prob_positive:.1%}"
        )

        return result

    def simulate_from_trades(
        self,
        trades: List[Dict[str, Any]],
        n_trades: int = 100,
    ) -> SimulationResult:
        """Simulate based on historical trade distribution."""
        pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
        if not pnls:
            return self.simulate(0.0, 0.01, n_trades)

        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        capital = sum(abs(p) for p in pnls) / len(pnls) * 10  # Rough estimate

        return self.simulate(
            mean_return=mean_pnl / capital,
            std_return=std_pnl / capital,
            n_periods=n_trades,
        )


def simulate_returns(
    mean_return: float,
    std_return: float,
    n_periods: int = 252,
) -> SimulationResult:
    """Convenience function for Monte Carlo simulation."""
    sim = MonteCarloSimulator()
    return sim.simulate(mean_return, std_return, n_periods)


def run_monte_carlo(
    trades: List[Dict[str, Any]],
    n_simulations: int = 10000,
) -> SimulationResult:
    """Run Monte Carlo from trade history."""
    sim = MonteCarloSimulator(n_simulations=n_simulations)
    return sim.simulate_from_trades(trades)
