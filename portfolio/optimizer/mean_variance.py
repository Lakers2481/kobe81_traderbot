"""
Mean-Variance Optimizer - Markowitz Portfolio Optimization

Classic mean-variance optimization with practical constraints
for solo traders.

Key Features:
- Efficient frontier calculation
- Maximum Sharpe ratio portfolio
- Minimum volatility portfolio
- Constraint handling (position limits, long-only)

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.structured_log import get_logger

logger = get_logger(__name__)


@dataclass
class OptimalPortfolio:
    """Result of portfolio optimization."""
    weights: Dict[str, float]           # Symbol -> weight
    expected_return: float              # Annualized
    volatility: float                   # Annualized
    sharpe_ratio: float
    max_weight: float                   # Largest position
    min_weight: float                   # Smallest position
    effective_n: float                  # Diversification
    optimization_type: str              # max_sharpe, min_vol, target_return
    constraints_satisfied: bool
    constraint_violations: List[str]
    as_of: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_weight": self.max_weight,
            "min_weight": self.min_weight,
            "effective_n": self.effective_n,
            "optimization_type": self.optimization_type,
            "constraints_satisfied": self.constraints_satisfied,
            "constraint_violations": self.constraint_violations,
            "as_of": self.as_of.isoformat(),
        }

    def get_allocation(self, total_capital: float) -> Dict[str, float]:
        """Get dollar allocation for each position."""
        return {
            symbol: weight * total_capital
            for symbol, weight in self.weights.items()
            if weight > 0.01  # Ignore tiny positions
        }

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"**Portfolio Optimization ({self.optimization_type})**",
            "",
            f"Expected Return: {self.expected_return:.1%}",
            f"Volatility: {self.volatility:.1%}",
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"Effective N: {self.effective_n:.1f}",
            "",
            "**Top Positions:**",
        ]

        # Sort by weight
        sorted_weights = sorted(
            self.weights.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for symbol, weight in sorted_weights[:5]:
            if weight > 0.01:
                lines.append(f"  {symbol}: {weight:.1%}")

        if self.constraint_violations:
            lines.append("")
            lines.append("**Constraint Violations:**")
            for violation in self.constraint_violations:
                lines.append(f"  - {violation}")

        return "\n".join(lines)


class MeanVarianceOptimizer:
    """
    Mean-Variance (Markowitz) Portfolio Optimizer.

    Features:
    - Maximum Sharpe ratio optimization
    - Minimum volatility optimization
    - Target return optimization
    - Constraint handling
    """

    STATE_FILE = Path("state/portfolio/mv_optimizer.json")

    # Default constraints
    DEFAULT_MAX_WEIGHT = 0.20       # 20% max per position
    DEFAULT_MIN_WEIGHT = 0.00       # Allow 0% (don't force all positions)
    DEFAULT_RISK_FREE_RATE = 0.05   # 5% risk-free rate

    def __init__(
        self,
        max_weight: float = DEFAULT_MAX_WEIGHT,
        min_weight: float = DEFAULT_MIN_WEIGHT,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ):
        """Initialize optimizer."""
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.risk_free_rate = risk_free_rate

        self._last_portfolio: Optional[OptimalPortfolio] = None

        # Ensure directory
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _calculate_portfolio_stats(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Calculate portfolio expected return, volatility, and Sharpe.

        Args:
            weights: Portfolio weights (N,)
            expected_returns: Expected returns (N,)
            cov_matrix: Covariance matrix (N, N)

        Returns:
            (expected_return, volatility, sharpe_ratio)
        """
        port_return = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        return port_return, port_vol, sharpe

    def _negative_sharpe(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> float:
        """Objective function: negative Sharpe (for minimization)."""
        _, _, sharpe = self._calculate_portfolio_stats(
            weights, expected_returns, cov_matrix
        )
        return -sharpe

    def _portfolio_volatility(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> float:
        """Objective function: portfolio volatility."""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def optimize(
        self,
        symbols: List[str],
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        optimization_type: str = "max_sharpe",
        target_return: Optional[float] = None,
        sector_limits: Optional[Dict[str, float]] = None,
        symbol_sectors: Optional[Dict[str, str]] = None,
    ) -> OptimalPortfolio:
        """
        Optimize portfolio weights.

        Args:
            symbols: List of symbols to include
            expected_returns: Dict of symbol -> expected annual return
            covariance_matrix: DataFrame covariance matrix
            optimization_type: 'max_sharpe', 'min_vol', or 'target_return'
            target_return: Required if optimization_type is 'target_return'
            sector_limits: Optional sector exposure limits
            symbol_sectors: Optional symbol -> sector mapping

        Returns:
            OptimalPortfolio
        """
        n = len(symbols)

        # Convert to numpy arrays
        returns_array = np.array([expected_returns.get(s, 0.0) for s in symbols])
        cov_array = covariance_matrix.loc[symbols, symbols].values

        # Initial guess (equal weight)
        init_weights = np.array([1.0 / n] * n)

        # Bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))

        # Constraints
        constraints = [
            # Weights sum to 1
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]

        # Target return constraint
        if optimization_type == "target_return" and target_return is not None:
            constraints.append({
                "type": "eq",
                "fun": lambda w: np.dot(w, returns_array) - target_return
            })

        # Sector constraints
        if sector_limits and symbol_sectors:
            for sector, limit in sector_limits.items():
                sector_mask = np.array([
                    1.0 if symbol_sectors.get(s) == sector else 0.0
                    for s in symbols
                ])
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, m=sector_mask, lim=limit: lim - np.dot(w, m)
                })

        # Optimize
        try:
            if optimization_type == "max_sharpe":
                result = minimize(
                    self._negative_sharpe,
                    init_weights,
                    args=(returns_array, cov_array),
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                )
            elif optimization_type == "min_vol":
                result = minimize(
                    self._portfolio_volatility,
                    init_weights,
                    args=(cov_array,),
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                )
            else:  # target_return
                result = minimize(
                    self._portfolio_volatility,
                    init_weights,
                    args=(cov_array,),
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                )

            optimal_weights = result.x

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Fall back to equal weight
            optimal_weights = init_weights

        # Calculate portfolio stats
        exp_ret, vol, sharpe = self._calculate_portfolio_stats(
            optimal_weights, returns_array, cov_array
        )

        # Build weights dict
        weights_dict = {
            symbol: float(weight)
            for symbol, weight in zip(symbols, optimal_weights)
        }

        # Calculate metrics
        max_weight = max(optimal_weights)
        min_weight = min(optimal_weights)
        hhi = sum(w ** 2 for w in optimal_weights)
        effective_n = 1.0 / hhi if hhi > 0 else 0

        # Check constraint violations
        violations = []
        if max_weight > self.max_weight + 0.01:
            violations.append(f"Max weight exceeded: {max_weight:.1%} > {self.max_weight:.1%}")
        if abs(sum(optimal_weights) - 1.0) > 0.01:
            violations.append(f"Weights don't sum to 1: {sum(optimal_weights):.2f}")

        portfolio = OptimalPortfolio(
            weights=weights_dict,
            expected_return=float(exp_ret),
            volatility=float(vol),
            sharpe_ratio=float(sharpe),
            max_weight=float(max_weight),
            min_weight=float(min_weight),
            effective_n=float(effective_n),
            optimization_type=optimization_type,
            constraints_satisfied=len(violations) == 0,
            constraint_violations=violations,
        )

        self._last_portfolio = portfolio

        # Save state
        self._save_state(portfolio)

        return portfolio

    def efficient_frontier(
        self,
        symbols: List[str],
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        n_points: int = 50,
    ) -> List[Tuple[float, float]]:
        """
        Calculate the efficient frontier.

        Returns:
            List of (volatility, return) tuples
        """
        returns_array = np.array([expected_returns.get(s, 0.0) for s in symbols])

        min_ret = min(returns_array)
        max_ret = max(returns_array)

        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier = []

        for target in target_returns:
            try:
                portfolio = self.optimize(
                    symbols=symbols,
                    expected_returns=expected_returns,
                    covariance_matrix=covariance_matrix,
                    optimization_type="target_return",
                    target_return=target,
                )
                frontier.append((portfolio.volatility, portfolio.expected_return))
            except Exception:
                continue

        return frontier

    def _save_state(self, portfolio: OptimalPortfolio) -> None:
        """Save optimization state."""
        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump({
                    "last_portfolio": portfolio.to_dict(),
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save optimizer state: {e}")

    def get_last_portfolio(self) -> Optional[OptimalPortfolio]:
        """Get last optimized portfolio."""
        return self._last_portfolio


# Singleton
_optimizer: Optional[MeanVarianceOptimizer] = None


def get_mv_optimizer() -> MeanVarianceOptimizer:
    """Get or create singleton optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = MeanVarianceOptimizer()
    return _optimizer


if __name__ == "__main__":
    # Demo
    optimizer = MeanVarianceOptimizer()

    print("=== Mean-Variance Optimizer Demo ===\n")

    # Sample data
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    # Expected returns (annualized)
    expected_returns = {
        "AAPL": 0.15,
        "MSFT": 0.14,
        "GOOGL": 0.12,
        "AMZN": 0.18,
        "NVDA": 0.25,
    }

    # Covariance matrix (simplified)
    np.random.seed(42)
    n = len(symbols)
    vol = np.array([0.25, 0.22, 0.28, 0.32, 0.45])
    corr = np.array([
        [1.0, 0.6, 0.5, 0.4, 0.3],
        [0.6, 1.0, 0.5, 0.4, 0.4],
        [0.5, 0.5, 1.0, 0.5, 0.3],
        [0.4, 0.4, 0.5, 1.0, 0.3],
        [0.3, 0.4, 0.3, 0.3, 1.0],
    ])
    cov = np.outer(vol, vol) * corr
    cov_df = pd.DataFrame(cov, index=symbols, columns=symbols)

    # Optimize
    portfolio = optimizer.optimize(
        symbols=symbols,
        expected_returns=expected_returns,
        covariance_matrix=cov_df,
        optimization_type="max_sharpe",
    )

    print(portfolio.to_summary())

    # Allocation for $100K
    print("\n**Allocation for $100,000:**")
    for symbol, amount in portfolio.get_allocation(100_000).items():
        print(f"  {symbol}: ${amount:,.0f}")
