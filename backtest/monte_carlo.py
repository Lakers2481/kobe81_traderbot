"""
Monte Carlo Simulation for Forward Testing
===========================================

Provides statistical confidence intervals for backtest results.

Features:
- Trade sequence resampling (bootstrap)
- Equity path simulation
- Drawdown distribution analysis
- Confidence interval estimation
- Risk of ruin calculation

Usage:
    from backtest.monte_carlo import MonteCarloSimulator

    mc = MonteCarloSimulator(trades_df)
    results = mc.run(n_simulations=10000)
    print(results.summary())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try numba for performance
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    n_simulations: int = 10000
    initial_capital: float = 100_000.0
    confidence_levels: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95])
    random_seed: Optional[int] = 42
    max_trades_per_sim: Optional[int] = None  # If None, use actual trade count


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation."""
    final_equity_distribution: np.ndarray
    max_drawdown_distribution: np.ndarray
    equity_paths: Optional[np.ndarray]  # Shape: (n_simulations, n_trades)
    confidence_intervals: Dict[str, Dict[str, float]]
    risk_of_ruin: float  # Probability of losing X% of capital
    expected_return: float
    expected_max_drawdown: float
    config: MonteCarloConfig

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 50,
            "MONTE CARLO SIMULATION RESULTS",
            "=" * 50,
            "",
            f"Simulations: {self.config.n_simulations:,}",
            f"Initial Capital: ${self.config.initial_capital:,.2f}",
            "",
            "--- Final Equity Distribution ---",
        ]

        for pct, values in self.confidence_intervals['final_equity'].items():
            lines.append(f"  {pct}: ${values:,.2f}")

        lines.extend([
            "",
            f"Expected Return: {self.expected_return:+.2f}%",
            f"Expected Max Drawdown: {self.expected_max_drawdown:.2f}%",
            f"Risk of Ruin (50% loss): {self.risk_of_ruin:.2%}",
            "",
            "--- Max Drawdown Distribution ---",
        ])

        for pct, values in self.confidence_intervals['max_drawdown'].items():
            lines.append(f"  {pct}: {values:.2f}%")

        lines.extend([
            "",
            "=" * 50,
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'n_simulations': self.config.n_simulations,
            'expected_return': float(self.expected_return),
            'expected_max_drawdown': float(self.expected_max_drawdown),
            'risk_of_ruin': float(self.risk_of_ruin),
            'confidence_intervals': self.confidence_intervals,
            'equity_stats': {
                'mean': float(np.mean(self.final_equity_distribution)),
                'std': float(np.std(self.final_equity_distribution)),
                'min': float(np.min(self.final_equity_distribution)),
                'max': float(np.max(self.final_equity_distribution)),
            },
            'drawdown_stats': {
                'mean': float(np.mean(self.max_drawdown_distribution)),
                'std': float(np.std(self.max_drawdown_distribution)),
                'min': float(np.min(self.max_drawdown_distribution)),
                'max': float(np.max(self.max_drawdown_distribution)),
            },
        }


@jit(nopython=True, cache=True)
def _simulate_single_path(
    returns: np.ndarray,
    initial_capital: float,
    n_trades: int,
) -> Tuple[float, float, np.ndarray]:
    """
    Simulate a single equity path (numba-optimized).

    Returns:
        Tuple of (final_equity, max_drawdown_pct, equity_path)
    """
    equity = np.empty(n_trades + 1)
    equity[0] = initial_capital

    peak = initial_capital
    max_dd = 0.0

    for i in range(n_trades):
        equity[i + 1] = equity[i] * (1 + returns[i])

        if equity[i + 1] > peak:
            peak = equity[i + 1]

        if peak > 0:
            dd = (peak - equity[i + 1]) / peak
            if dd > max_dd:
                max_dd = dd

    return equity[-1], max_dd * 100, equity


class MonteCarloSimulator:
    """
    Monte Carlo simulator for forward testing.

    Uses bootstrap resampling of historical trades to generate
    possible future equity paths and estimate confidence intervals.
    """

    def __init__(
        self,
        trades: pd.DataFrame,
        config: Optional[MonteCarloConfig] = None,
    ):
        """
        Initialize Monte Carlo simulator.

        Args:
            trades: DataFrame with 'pnl' column (trade returns)
            config: Simulation configuration
        """
        self.trades = trades
        self.config = config or MonteCarloConfig()

        # Extract trade returns
        if 'pnl' in trades.columns and 'entry_price' in trades.columns:
            # Calculate percentage returns
            self.returns = (trades['pnl'] / (trades['entry_price'] * trades.get('qty', 1))).dropna().values
        elif 'pnl_pct' in trades.columns:
            self.returns = trades['pnl_pct'].dropna().values
        elif 'pnl' in trades.columns:
            # Assume PnL is already in dollar terms, normalize
            trades['pnl'].sum()
            self.returns = (trades['pnl'] / self.config.initial_capital).dropna().values
        else:
            raise ValueError("Trades DataFrame must have 'pnl' column")

        if len(self.returns) == 0:
            raise ValueError("No valid trade returns found")

        # Set random seed
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    def run(
        self,
        n_simulations: Optional[int] = None,
        store_paths: bool = False,
    ) -> MonteCarloResults:
        """
        Run Monte Carlo simulation.

        Args:
            n_simulations: Override config simulation count
            store_paths: Store all equity paths (memory intensive)

        Returns:
            MonteCarloResults with distributions and confidence intervals
        """
        n_sims = n_simulations or self.config.n_simulations
        n_trades = self.config.max_trades_per_sim or len(self.returns)

        logger.info(f"Running {n_sims:,} Monte Carlo simulations with {n_trades} trades each")

        # Pre-allocate result arrays
        final_equities = np.empty(n_sims)
        max_drawdowns = np.empty(n_sims)
        equity_paths = np.empty((n_sims, n_trades + 1)) if store_paths else None

        # Run simulations
        for sim in range(n_sims):
            # Bootstrap resample trades
            sampled_indices = np.random.randint(0, len(self.returns), size=n_trades)
            sampled_returns = self.returns[sampled_indices]

            # Simulate equity path
            final_eq, max_dd, path = _simulate_single_path(
                sampled_returns,
                self.config.initial_capital,
                n_trades,
            )

            final_equities[sim] = final_eq
            max_drawdowns[sim] = max_dd

            if store_paths:
                equity_paths[sim] = path

        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            final_equities, max_drawdowns
        )

        # Calculate risk of ruin (probability of losing 50% or more)
        ruin_threshold = self.config.initial_capital * 0.5
        risk_of_ruin = np.mean(final_equities < ruin_threshold)

        # Expected values
        expected_return = (np.mean(final_equities) / self.config.initial_capital - 1) * 100
        expected_max_dd = np.mean(max_drawdowns)

        results = MonteCarloResults(
            final_equity_distribution=final_equities,
            max_drawdown_distribution=max_drawdowns,
            equity_paths=equity_paths,
            confidence_intervals=confidence_intervals,
            risk_of_ruin=risk_of_ruin,
            expected_return=expected_return,
            expected_max_drawdown=expected_max_dd,
            config=self.config,
        )

        logger.info(f"Monte Carlo complete: Expected Return {expected_return:+.2f}%, "
                   f"Risk of Ruin {risk_of_ruin:.2%}")

        return results

    def _calculate_confidence_intervals(
        self,
        final_equities: np.ndarray,
        max_drawdowns: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate confidence intervals for key metrics."""
        intervals = {
            'final_equity': {},
            'max_drawdown': {},
        }

        for level in self.config.confidence_levels:
            pct_label = f"{int(level * 100)}th percentile"

            intervals['final_equity'][pct_label] = float(np.percentile(final_equities, level * 100))
            intervals['max_drawdown'][pct_label] = float(np.percentile(max_drawdowns, level * 100))

        return intervals

    def run_with_sizing(
        self,
        position_sizes: np.ndarray,
        n_simulations: Optional[int] = None,
    ) -> MonteCarloResults:
        """
        Run simulation with varying position sizes.

        Args:
            position_sizes: Array of position sizes (0-1) for each trade
            n_simulations: Override config simulation count
        """
        if len(position_sizes) != len(self.returns):
            raise ValueError("Position sizes must match number of trades")

        # Adjust returns by position size
        adjusted_returns = self.returns * position_sizes

        # Create temporary simulator with adjusted returns
        temp_trades = self.trades.copy()
        temp_trades['pnl_pct'] = adjusted_returns

        temp_sim = MonteCarloSimulator(temp_trades, self.config)
        return temp_sim.run(n_simulations)

    def calculate_optimal_f(
        self,
        min_f: float = 0.01,
        max_f: float = 1.0,
        steps: int = 100,
    ) -> Tuple[float, float]:
        """
        Calculate optimal fraction (Kelly-inspired) for position sizing.

        Returns:
            Tuple of (optimal_f, expected_growth_rate)
        """
        f_values = np.linspace(min_f, max_f, steps)
        growth_rates = []

        for f in f_values:
            # Calculate geometric mean of (1 + f * return)
            adjusted = 1 + f * self.returns
            # Handle negative results (would lead to ruin)
            adjusted = np.maximum(adjusted, 0.001)
            growth_rate = np.exp(np.mean(np.log(adjusted))) - 1
            growth_rates.append(growth_rate)

        growth_rates = np.array(growth_rates)
        optimal_idx = np.argmax(growth_rates)
        optimal_f = f_values[optimal_idx]

        return float(optimal_f), float(growth_rates[optimal_idx])


def run_monte_carlo_analysis(
    trades: pd.DataFrame,
    n_simulations: int = 10000,
    initial_capital: float = 100_000,
) -> Dict[str, Any]:
    """
    Convenience function to run complete Monte Carlo analysis.

    Args:
        trades: DataFrame with trade data
        n_simulations: Number of simulations
        initial_capital: Starting capital

    Returns:
        Dictionary with analysis results
    """
    config = MonteCarloConfig(
        n_simulations=n_simulations,
        initial_capital=initial_capital,
    )

    mc = MonteCarloSimulator(trades, config)
    results = mc.run()

    # Calculate optimal f
    optimal_f, optimal_growth = mc.calculate_optimal_f()

    return {
        'results': results.to_dict(),
        'summary': results.summary(),
        'optimal_f': optimal_f,
        'optimal_growth_rate': optimal_growth,
    }
