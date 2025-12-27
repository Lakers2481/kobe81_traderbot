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
