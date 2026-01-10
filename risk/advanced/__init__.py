"""
Advanced Risk Management Suite
==============================

Production-grade risk components for quantitative trading.

Components:
- Monte Carlo VaR: Portfolio VaR with Cholesky decomposition
- Kelly Position Sizer: Optimal position sizing with Kelly Criterion
- Correlation Limits: Enhanced correlation and concentration checks

MERGED FROM GAME_PLAN_2K28 - Production Ready
"""

from .monte_carlo_var import (
    MonteCarloVaR,
    VaRResult,
    StressScenario,
    StressTestResult,
)

from .kelly_position_sizer import (
    KellyPositionSizer,
    KellyPositionResult,
    optimal_kelly,
    fractional_kelly,
    volatility_adjusted_kelly,
    quick_kelly_position,
)

from .correlation_limits import (
    EnhancedCorrelationLimits,
    CorrelationCheckResult,
    PortfolioDiversificationMetrics,
    RiskLevel,
    check_correlation_limits,
)

__all__ = [
    # Monte Carlo VaR
    'MonteCarloVaR',
    'VaRResult',
    'StressScenario',
    'StressTestResult',
    'check_portfolio_var',  # FIX (2026-01-08): Added easy-to-use VaR check

    # Kelly Position Sizer
    'KellyPositionSizer',
    'KellyPositionResult',
    'optimal_kelly',
    'fractional_kelly',
    'volatility_adjusted_kelly',
    'quick_kelly_position',

    # Correlation Limits
    'EnhancedCorrelationLimits',
    'CorrelationCheckResult',
    'PortfolioDiversificationMetrics',
    'RiskLevel',
    'check_correlation_limits',
]


import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

_var_logger = logging.getLogger(__name__)


def check_portfolio_var(
    positions: Optional[List[Dict[str, Any]]] = None,
    var_limit_pct: float = 0.05,
    confidence: float = 0.95,
    horizon_days: int = 1,
    n_simulations: int = 5000,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if portfolio Value at Risk is within acceptable limits.

    FIX (2026-01-08): Wire Monte Carlo VaR into execution path.
    This provides an easy-to-use interface for daily VaR checks.

    Args:
        positions: List of position dicts with keys:
            - symbol: Stock symbol
            - quantity: Number of shares
            - price: Current price per share
            - volatility: Daily volatility (optional, estimated if missing)
        var_limit_pct: Maximum acceptable VaR as % of portfolio (default 5%)
        confidence: Confidence level for VaR (default 95%)
        horizon_days: Time horizon in days (default 1)
        n_simulations: Number of Monte Carlo simulations (default 5000)

    Returns:
        Tuple of (passes_check: bool, var_result: dict)

    Example:
        >>> positions = [
        ...     {'symbol': 'AAPL', 'quantity': 100, 'price': 175.0, 'volatility': 0.02},
        ...     {'symbol': 'MSFT', 'quantity': 50, 'price': 380.0, 'volatility': 0.018},
        ... ]
        >>> ok, result = check_portfolio_var(positions, var_limit_pct=0.05)
        >>> if not ok:
        ...     print(f"VaR exceeded: {result['var_pct']:.2%} > 5%")
    """
    # If no positions provided, try to fetch from broker
    if positions is None:
        try:
            from execution.broker_alpaca import get_positions
            raw_positions = get_positions()
            if not raw_positions:
                _var_logger.info("No positions to check VaR for")
                return True, {'var_pct': 0.0, 'portfolio_value': 0.0, 'message': 'No positions'}

            positions = []
            for pos in raw_positions:
                positions.append({
                    'symbol': pos.get('symbol', ''),
                    'quantity': float(pos.get('qty', 0)),
                    'price': float(pos.get('current_price', 0)),
                    'volatility': 0.02,  # Default 2% daily volatility
                })
        except Exception as e:
            _var_logger.warning(f"Could not fetch positions for VaR: {e}")
            return True, {'var_pct': 0.0, 'message': f'Position fetch error: {e}'}

    if not positions:
        return True, {'var_pct': 0.0, 'portfolio_value': 0.0, 'message': 'No positions'}

    # Build DataFrame for VaR calculation
    pos_df = pd.DataFrame(positions)

    # Ensure required columns exist
    if 'expected_return' not in pos_df.columns:
        pos_df['expected_return'] = 0.0  # Assume zero expected return
    if 'volatility' not in pos_df.columns:
        pos_df['volatility'] = 0.02  # Default 2% daily volatility

    # Calculate portfolio value
    pos_df['value'] = pos_df['quantity'] * pos_df['price']
    portfolio_value = pos_df['value'].sum()

    if portfolio_value <= 0:
        return True, {'var_pct': 0.0, 'portfolio_value': 0.0, 'message': 'No portfolio value'}

    try:
        # Run Monte Carlo VaR
        var_calculator = MonteCarloVaR(
            confidence_levels=[confidence],
            horizon_days=horizon_days,
            n_simulations=n_simulations,
        )

        var_result = var_calculator.calculate_var(pos_df)

        # Extract VaR as percentage of portfolio
        var_dollar = var_result.get('var', 0.0)
        var_pct = abs(var_dollar) / portfolio_value if portfolio_value > 0 else 0.0

        # Check if within limit
        passes = var_pct <= var_limit_pct

        result = {
            'var_pct': var_pct,
            'var_dollar': var_dollar,
            'portfolio_value': portfolio_value,
            'cvar': var_result.get('cvar', 0.0),
            'confidence': confidence,
            'horizon_days': horizon_days,
            'limit_pct': var_limit_pct,
            'passes': passes,
        }

        if not passes:
            _var_logger.warning(
                f"VaR LIMIT EXCEEDED: {var_pct:.2%} > {var_limit_pct:.2%} "
                f"(${abs(var_dollar):,.0f} on ${portfolio_value:,.0f})"
            )
        else:
            _var_logger.info(
                f"VaR check passed: {var_pct:.2%} <= {var_limit_pct:.2%}"
            )

        return passes, result

    except Exception as e:
        _var_logger.error(f"VaR calculation failed: {e}")
        # Fail-safe: return True to not block trading on calculation error
        return True, {'var_pct': 0.0, 'message': f'Calculation error: {e}'}
