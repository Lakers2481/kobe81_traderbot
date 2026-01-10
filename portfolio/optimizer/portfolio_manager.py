"""
Portfolio Manager - Unified Portfolio Intelligence

Coordinates all portfolio optimization components:
- Mean-variance optimization
- Risk parity allocation
- Rebalancing recommendations
- Risk monitoring

Solo Trader Features:
- Simple daily position guidance
- Optimal size for new signals
- Rebalancing alerts
- Risk budget tracking

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum
import json

import numpy as np
import pandas as pd

from core.structured_log import get_logger
from .mean_variance import MeanVarianceOptimizer
from .risk_parity import RiskParityOptimizer
from .rebalancer import PortfolioRebalancer, RebalanceRecommendation, RebalanceUrgency

logger = get_logger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization method."""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOL = "inverse_volatility"


@dataclass
class PortfolioState:
    """Current portfolio state and recommendations."""
    # Current state
    positions: Dict[str, Dict]      # symbol -> {shares, price, value, weight}
    total_value: float
    cash: float
    invested: float

    # Risk metrics
    estimated_volatility: float
    estimated_beta: float
    max_position_weight: float
    effective_n: float

    # Optimization
    target_weights: Dict[str, float]
    optimization_method: OptimizationMethod

    # Rebalancing
    rebalance_urgency: RebalanceUrgency
    rebalance_trades: int
    days_since_rebalance: int

    # Risk budget
    risk_budget_used: float         # % of risk budget used
    can_add_position: bool          # Room for new positions

    as_of: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positions": self.positions,
            "total_value": self.total_value,
            "cash": self.cash,
            "invested": self.invested,
            "estimated_volatility": self.estimated_volatility,
            "estimated_beta": self.estimated_beta,
            "max_position_weight": self.max_position_weight,
            "effective_n": self.effective_n,
            "target_weights": self.target_weights,
            "optimization_method": self.optimization_method.value,
            "rebalance_urgency": self.rebalance_urgency.value,
            "rebalance_trades": self.rebalance_trades,
            "days_since_rebalance": self.days_since_rebalance,
            "risk_budget_used": self.risk_budget_used,
            "can_add_position": self.can_add_position,
            "as_of": self.as_of.isoformat(),
        }

    def to_summary(self) -> str:
        """Generate dashboard summary."""
        lines = [
            "=" * 50,
            "PORTFOLIO STATUS",
            "=" * 50,
            "",
            f"Total Value: ${self.total_value:,.0f}",
            f"Cash: ${self.cash:,.0f} ({self.cash/self.total_value:.1%})" if self.total_value > 0 else f"Cash: ${self.cash:,.0f}",
            f"Invested: ${self.invested:,.0f}",
            "",
            "**Risk Metrics:**",
            f"  Volatility: {self.estimated_volatility:.1%}",
            f"  Beta: {self.estimated_beta:.2f}",
            f"  Max Position: {self.max_position_weight:.1%}",
            f"  Effective N: {self.effective_n:.1f}",
            "",
            f"**Rebalancing:** {self.rebalance_urgency.name}",
            f"  Trades needed: {self.rebalance_trades}",
            f"  Days since last: {self.days_since_rebalance}",
            "",
            f"**Risk Budget:** {self.risk_budget_used:.0%} used",
            f"  Can add position: {'YES' if self.can_add_position else 'NO'}",
            "",
        ]

        if self.positions:
            lines.append("**Positions:**")
            sorted_pos = sorted(
                self.positions.items(),
                key=lambda x: x[1].get("weight", 0),
                reverse=True
            )
            for symbol, pos in sorted_pos[:5]:
                weight = pos.get("weight", 0)
                value = pos.get("value", 0)
                lines.append(f"  {symbol}: {weight:.1%} (${value:,.0f})")

        return "\n".join(lines)


class PortfolioManager:
    """
    Unified portfolio management.

    Features:
    - Portfolio state tracking
    - Multi-method optimization
    - Rebalancing recommendations
    - New position sizing
    """

    STATE_FILE = Path("state/portfolio/manager.json")

    # Risk budget settings
    MAX_PORTFOLIO_VOL = 0.20        # 20% annual vol target
    MAX_POSITIONS = 10              # Max concurrent positions
    MIN_POSITION_WEIGHT = 0.05      # 5% minimum per position
    MAX_POSITION_WEIGHT = 0.20      # 20% maximum per position

    def __init__(
        self,
        optimization_method: OptimizationMethod = OptimizationMethod.RISK_PARITY,
    ):
        """Initialize portfolio manager."""
        self.optimization_method = optimization_method

        self._mv_optimizer = MeanVarianceOptimizer(
            max_weight=self.MAX_POSITION_WEIGHT
        )
        self._rp_optimizer = RiskParityOptimizer(
            target_volatility=self.MAX_PORTFOLIO_VOL
        )
        self._rebalancer = PortfolioRebalancer()

        self._current_state: Optional[PortfolioState] = None

        # Ensure directory
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _estimate_covariance(
        self,
        symbols: List[str],
        returns_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Estimate covariance matrix.

        If no returns data provided, use reasonable defaults.
        """
        n = len(symbols)

        if returns_data is not None and not returns_data.empty:
            # Calculate from historical returns
            cov = returns_data[symbols].cov() * 252  # Annualize
            return cov

        # Default: assume 25% vol with 0.3 correlation
        default_vol = 0.25
        default_corr = 0.3

        corr = np.eye(n) + default_corr * (np.ones((n, n)) - np.eye(n))
        vols = np.array([default_vol] * n)
        cov = np.outer(vols, vols) * corr

        return pd.DataFrame(cov, index=symbols, columns=symbols)

    def _estimate_returns(
        self,
        symbols: List[str],
        returns_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Estimate expected returns.

        Uses shrinkage towards market return if historical data is noisy.
        """
        market_return = 0.10  # 10% long-term market return

        if returns_data is not None and not returns_data.empty:
            historical = returns_data[symbols].mean() * 252
            # Shrink towards market (50% shrinkage)
            shrunk = 0.5 * historical + 0.5 * market_return
            return shrunk.to_dict()

        # Default to market return
        return {s: market_return for s in symbols}

    def update_state(
        self,
        positions: Dict[str, Dict],
        cash: float,
        returns_data: Optional[pd.DataFrame] = None,
    ) -> PortfolioState:
        """
        Update portfolio state with current positions.

        Args:
            positions: {symbol: {shares, price, cost_basis, ...}}
            cash: Available cash
            returns_data: Optional historical returns for optimization

        Returns:
            PortfolioState
        """
        # Calculate portfolio value
        invested = sum(
            pos.get("shares", 0) * pos.get("price", 0)
            for pos in positions.values()
        )
        total_value = invested + cash

        # Calculate current weights
        for symbol, pos in positions.items():
            value = pos.get("shares", 0) * pos.get("price", 0)
            pos["value"] = value
            pos["weight"] = value / total_value if total_value > 0 else 0

        # Get symbols
        symbols = list(positions.keys())

        if not symbols:
            # Empty portfolio
            state = PortfolioState(
                positions={},
                total_value=total_value,
                cash=cash,
                invested=0,
                estimated_volatility=0,
                estimated_beta=0,
                max_position_weight=0,
                effective_n=0,
                target_weights={},
                optimization_method=self.optimization_method,
                rebalance_urgency=RebalanceUrgency.NONE,
                rebalance_trades=0,
                days_since_rebalance=0,
                risk_budget_used=0,
                can_add_position=True,
            )
            self._current_state = state
            return state

        # Estimate covariance and returns
        cov_matrix = self._estimate_covariance(symbols, returns_data)
        expected_returns = self._estimate_returns(symbols, returns_data)

        # Optimize
        if self.optimization_method == OptimizationMethod.MEAN_VARIANCE:
            opt_portfolio = self._mv_optimizer.optimize(
                symbols=symbols,
                expected_returns=expected_returns,
                covariance_matrix=cov_matrix,
            )
            target_weights = opt_portfolio.weights
            est_vol = opt_portfolio.volatility

        elif self.optimization_method == OptimizationMethod.RISK_PARITY:
            rp_portfolio = self._rp_optimizer.optimize(
                symbols=symbols,
                covariance_matrix=cov_matrix,
            )
            target_weights = rp_portfolio.weights
            est_vol = rp_portfolio.total_volatility

        elif self.optimization_method == OptimizationMethod.INVERSE_VOL:
            target_weights = self._rp_optimizer.inverse_volatility_weights(
                symbols, cov_matrix
            )
            est_vol = 0.15  # Approximate

        else:  # Equal weight
            target_weights = {s: 1.0 / len(symbols) for s in symbols}
            est_vol = 0.15  # Approximate

        # Rebalancing recommendation
        rebalance = self._rebalancer.recommend(
            current_positions=positions,
            target_weights=target_weights,
            total_value=total_value,
        )

        # Calculate metrics
        current_weights = [pos.get("weight", 0) for pos in positions.values()]
        max_weight = max(current_weights) if current_weights else 0
        hhi = sum(w ** 2 for w in current_weights)
        effective_n = 1.0 / hhi if hhi > 0 else 0

        # Risk budget
        risk_budget_used = est_vol / self.MAX_PORTFOLIO_VOL
        can_add = (
            len(positions) < self.MAX_POSITIONS and
            risk_budget_used < 0.9 and
            cash / total_value > 0.05
        )

        state = PortfolioState(
            positions=positions,
            total_value=total_value,
            cash=cash,
            invested=invested,
            estimated_volatility=est_vol,
            estimated_beta=1.0,  # Would need market data
            max_position_weight=max_weight,
            effective_n=effective_n,
            target_weights=target_weights,
            optimization_method=self.optimization_method,
            rebalance_urgency=rebalance.urgency,
            rebalance_trades=len([t for t in rebalance.trades if t.shares_to_trade > 0]),
            days_since_rebalance=rebalance.days_since_last,
            risk_budget_used=risk_budget_used,
            can_add_position=can_add,
        )

        self._current_state = state
        self._save_state()

        return state

    def get_optimal_size(
        self,
        symbol: str,
        signal_confidence: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Get optimal position size for a new signal.

        Args:
            symbol: Symbol to add
            signal_confidence: Quality gate confidence (0-1)

        Returns:
            {weight, dollar_amount, shares_estimate, reason}
        """
        if not self._current_state:
            return {
                "weight": self.MIN_POSITION_WEIGHT,
                "dollar_amount": 0,
                "shares_estimate": 0,
                "reason": "No portfolio state available",
            }

        state = self._current_state

        if not state.can_add_position:
            return {
                "weight": 0,
                "dollar_amount": 0,
                "shares_estimate": 0,
                "reason": "Risk budget exhausted or max positions reached",
            }

        # Base weight
        base_weight = self.MIN_POSITION_WEIGHT

        # Adjust by confidence
        confidence_multiplier = 0.5 + signal_confidence  # 0.5x to 1.5x
        adjusted_weight = base_weight * confidence_multiplier

        # Cap at max weight
        final_weight = min(adjusted_weight, self.MAX_POSITION_WEIGHT)

        # Calculate dollar amount
        dollar_amount = final_weight * state.total_value

        return {
            "weight": final_weight,
            "dollar_amount": dollar_amount,
            "shares_estimate": 0,  # Would need price
            "reason": f"Confidence-adjusted: {signal_confidence:.0%} -> {final_weight:.1%} weight",
        }

    def _save_state(self) -> None:
        """Save manager state."""
        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump({
                    "current_state": self._current_state.to_dict() if self._current_state else None,
                    "optimization_method": self.optimization_method.value,
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save manager state: {e}")

    def get_state(self) -> Optional[PortfolioState]:
        """Get current portfolio state."""
        return self._current_state

    def get_rebalance_recommendation(self) -> Optional[RebalanceRecommendation]:
        """Get latest rebalancing recommendation."""
        if not self._current_state:
            return None

        return self._rebalancer.recommend(
            current_positions=self._current_state.positions,
            target_weights=self._current_state.target_weights,
            total_value=self._current_state.total_value,
        )


# Singleton
_manager: Optional[PortfolioManager] = None


def get_portfolio_manager() -> PortfolioManager:
    """Get or create singleton manager."""
    global _manager
    if _manager is None:
        _manager = PortfolioManager()
    return _manager


if __name__ == "__main__":
    # Demo
    manager = PortfolioManager(
        optimization_method=OptimizationMethod.RISK_PARITY
    )

    print("=== Portfolio Manager Demo ===\n")

    # Sample positions
    positions = {
        "AAPL": {"shares": 100, "price": 175, "cost_basis": 150},
        "MSFT": {"shares": 50, "price": 380, "cost_basis": 350},
        "GOOGL": {"shares": 30, "price": 140, "cost_basis": 130},
        "NVDA": {"shares": 40, "price": 480, "cost_basis": 300},
    }

    cash = 20000

    # Update state
    state = manager.update_state(
        positions=positions,
        cash=cash,
    )

    print(state.to_summary())

    # Get optimal size for new position
    print("\n**Optimal Size for New Position (AMZN):**")
    sizing = manager.get_optimal_size("AMZN", signal_confidence=0.75)
    print(f"  Weight: {sizing['weight']:.1%}")
    print(f"  Dollar Amount: ${sizing['dollar_amount']:,.0f}")
    print(f"  Reason: {sizing['reason']}")

    # Rebalancing recommendation
    print("\n**Rebalancing:**")
    rebal = manager.get_rebalance_recommendation()
    if rebal:
        print(f"  Urgency: {rebal.urgency.name}")
        print(f"  Reason: {rebal.reason}")
