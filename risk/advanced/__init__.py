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
    create_var_calculator
)

from .kelly_position_sizer import (
    KellyPositionSizer,
    KellyResult,
    optimal_kelly,
    fractional_kelly,
    volatility_adjusted_kelly,
    create_kelly_sizer
)

from .correlation_limits import (
    EnhancedCorrelationLimits,
    CorrelationCheckResult,
    create_correlation_checker
)

__all__ = [
    # Monte Carlo VaR
    'MonteCarloVaR',
    'VaRResult',
    'StressScenario',
    'StressTestResult',
    'create_var_calculator',

    # Kelly Position Sizer
    'KellyPositionSizer',
    'KellyResult',
    'optimal_kelly',
    'fractional_kelly',
    'volatility_adjusted_kelly',
    'create_kelly_sizer',

    # Correlation Limits
    'EnhancedCorrelationLimits',
    'CorrelationCheckResult',
    'create_correlation_checker',
]
