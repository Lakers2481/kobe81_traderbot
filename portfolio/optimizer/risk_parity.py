"""
Risk Parity Optimizer - Equal Risk Contribution

Allocate portfolio so each position contributes equally to total risk.
This avoids concentration in low-volatility assets.

Key Features:
- Equal risk contribution (ERC)
- Volatility targeting
- Robust to estimation error
- Better diversification than mean-variance

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
class RiskParityPortfolio:
    """Result of risk parity optimization."""
    weights: Dict[str, float]               # Symbol -> weight
    risk_contributions: Dict[str, float]    # Symbol -> risk contribution
    total_volatility: float                 # Portfolio volatility
    max_risk_contribution: float            # Largest risk contribution
    risk_concentration: float               # HHI of risk contributions
    target_volatility: Optional[float]      # If volatility targeting
    leverage: float                         # Leverage applied (1.0 = none)
    as_of: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "risk_contributions": self.risk_contributions,
            "total_volatility": self.total_volatility,
            "max_risk_contribution": self.max_risk_contribution,
            "risk_concentration": self.risk_concentration,
            "target_volatility": self.target_volatility,
            "leverage": self.leverage,
            "as_of": self.as_of.isoformat(),
        }

    def get_allocation(self, total_capital: float) -> Dict[str, float]:
        """Get dollar allocation for each position."""
        return {
            symbol: weight * total_capital * self.leverage
            for symbol, weight in self.weights.items()
            if weight > 0.01
        }

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "**Risk Parity Portfolio**",
            "",
            f"Total Volatility: {self.total_volatility:.1%}",
            f"Max Risk Contribution: {self.max_risk_contribution:.1%}",
            f"Risk Concentration (HHI): {self.risk_concentration:.3f}",
            f"Leverage: {self.leverage:.2f}x",
            "",
            "**Weights & Risk Contributions:**",
        ]

        # Sort by weight
        sorted_weights = sorted(
            self.weights.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for symbol, weight in sorted_weights:
            if weight > 0.01:
                risk_contrib = self.risk_contributions.get(symbol, 0)
                lines.append(f"  {symbol}: {weight:.1%} (risk: {risk_contrib:.1%})")

        return "\n".join(lines)


class RiskParityOptimizer:
    """
    Risk Parity (Equal Risk Contribution) Optimizer.

    Features:
    - Equal marginal risk contribution
    - Volatility targeting
    - Leverage calculation
    - Robust optimization
    """

    STATE_FILE = Path("state/portfolio/rp_optimizer.json")

    def __init__(
        self,
        target_volatility: Optional[float] = None,
        max_leverage: float = 2.0,
    ):
        """Initialize optimizer."""
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage

        self._last_portfolio: Optional[RiskParityPortfolio] = None

        # Ensure directory
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _risk_budget_objective(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        risk_budget: np.ndarray,
    ) -> float:
        """
        Objective: minimize deviation from target risk budget.

        Risk contribution of asset i = w_i * (Σw)_i / sqrt(w'Σw)
        Target: RC_i = risk_budget_i * total_vol
        """
        weights = np.maximum(weights, 1e-10)  # Avoid division by zero

        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        if port_vol < 1e-10:
            return 1e10

        # Marginal risk contributions
        marginal_contrib = np.dot(cov_matrix, weights) / port_vol

        # Risk contributions
        risk_contrib = weights * marginal_contrib

        # Normalize
        risk_contrib_pct = risk_contrib / port_vol

        # Objective: sum of squared deviations from target
        target_contrib = risk_budget * port_vol
        deviation = risk_contrib - target_contrib

        return np.sum(deviation ** 2)

    def _calculate_risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate risk contributions for each asset.

        Returns:
            (risk_contributions, total_volatility)
        """
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        if port_vol < 1e-10:
            return np.zeros(len(weights)), 0.0

        # Marginal contributions
        marginal = np.dot(cov_matrix, weights) / port_vol

        # Risk contributions (percentage)
        risk_contrib = (weights * marginal) / port_vol

        return risk_contrib, port_vol

    def optimize(
        self,
        symbols: List[str],
        covariance_matrix: pd.DataFrame,
        risk_budgets: Optional[Dict[str, float]] = None,
    ) -> RiskParityPortfolio:
        """
        Optimize for equal risk contribution.

        Args:
            symbols: List of symbols
            covariance_matrix: Covariance matrix
            risk_budgets: Optional custom risk budgets (defaults to equal)

        Returns:
            RiskParityPortfolio
        """
        n = len(symbols)
        cov_array = covariance_matrix.loc[symbols, symbols].values

        # Default to equal risk budget
        if risk_budgets is None:
            risk_budget = np.ones(n) / n
        else:
            risk_budget = np.array([
                risk_budgets.get(s, 1.0 / n)
                for s in symbols
            ])
            risk_budget = risk_budget / risk_budget.sum()

        # Initial guess (inverse volatility weighted)
        vols = np.sqrt(np.diag(cov_array))
        init_weights = (1.0 / vols) / np.sum(1.0 / vols)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]

        # Bounds (long only, no single position > 50%)
        bounds = tuple((0.01, 0.50) for _ in range(n))

        # Optimize
        try:
            result = minimize(
                self._risk_budget_objective,
                init_weights,
                args=(cov_array, risk_budget),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            optimal_weights = result.x
            optimal_weights = optimal_weights / optimal_weights.sum()  # Ensure sums to 1

        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            optimal_weights = init_weights

        # Calculate risk contributions
        risk_contrib, port_vol = self._calculate_risk_contributions(
            optimal_weights, cov_array
        )

        # Build dicts
        weights_dict = {
            symbol: float(weight)
            for symbol, weight in zip(symbols, optimal_weights)
        }

        risk_contrib_dict = {
            symbol: float(contrib)
            for symbol, contrib in zip(symbols, risk_contrib)
        }

        # Risk concentration (HHI)
        risk_hhi = sum(c ** 2 for c in risk_contrib)

        # Calculate leverage for volatility targeting
        leverage = 1.0
        if self.target_volatility and port_vol > 0:
            raw_leverage = self.target_volatility / port_vol
            leverage = min(raw_leverage, self.max_leverage)

        portfolio = RiskParityPortfolio(
            weights=weights_dict,
            risk_contributions=risk_contrib_dict,
            total_volatility=float(port_vol),
            max_risk_contribution=float(max(risk_contrib)),
            risk_concentration=float(risk_hhi),
            target_volatility=self.target_volatility,
            leverage=float(leverage),
        )

        self._last_portfolio = portfolio
        self._save_state(portfolio)

        return portfolio

    def inverse_volatility_weights(
        self,
        symbols: List[str],
        covariance_matrix: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Simple inverse volatility weighting.

        A practical approximation to risk parity that ignores correlations.
        """
        cov_array = covariance_matrix.loc[symbols, symbols].values
        vols = np.sqrt(np.diag(cov_array))

        inv_vol_weights = (1.0 / vols) / np.sum(1.0 / vols)

        return {
            symbol: float(weight)
            for symbol, weight in zip(symbols, inv_vol_weights)
        }

    def _save_state(self, portfolio: RiskParityPortfolio) -> None:
        """Save optimization state."""
        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump({
                    "last_portfolio": portfolio.to_dict(),
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save optimizer state: {e}")

    def get_last_portfolio(self) -> Optional[RiskParityPortfolio]:
        """Get last optimized portfolio."""
        return self._last_portfolio


# Singleton
_optimizer: Optional[RiskParityOptimizer] = None


def get_rp_optimizer() -> RiskParityOptimizer:
    """Get or create singleton optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = RiskParityOptimizer()
    return _optimizer


if __name__ == "__main__":
    # Demo
    optimizer = RiskParityOptimizer(target_volatility=0.10)

    print("=== Risk Parity Optimizer Demo ===\n")

    # Sample data
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    # Covariance matrix (different volatilities)
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
        covariance_matrix=cov_df,
    )

    print(portfolio.to_summary())

    # Compare to inverse volatility
    print("\n**Inverse Volatility Weights (Simple):**")
    inv_vol = optimizer.inverse_volatility_weights(symbols, cov_df)
    for sym, w in sorted(inv_vol.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sym}: {w:.1%}")

    # Allocation for $100K
    print("\n**Allocation for $100,000:**")
    for symbol, amount in portfolio.get_allocation(100_000).items():
        print(f"  {symbol}: ${amount:,.0f}")
