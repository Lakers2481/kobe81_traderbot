"""
Risk Management Module
======================

Core risk management for the Kobe trading system.

Components:
- policy_gate: Per-order ($75) and daily ($1k) budget enforcement
- liquidity_gate: ADV and spread checks for live execution
- position_limit_gate: Max concurrent positions enforcement
- advanced/: Advanced risk analytics (VaR, Kelly, Correlation)
"""

from .policy_gate import PolicyGate
from .liquidity_gate import LiquidityGate, LiquidityCheck, check_liquidity
from .position_limit_gate import (
    PositionLimitGate,
    PositionLimits,
    get_position_limit_gate,
)

# Note: advanced submodule is available but not auto-imported
# to avoid dependency issues. Import explicitly:
#   from risk.advanced import MonteCarloVaR, KellyPositionSizer

__all__ = [
    'PolicyGate',
    'LiquidityGate',
    'LiquidityCheck',
    'check_liquidity',
    'PositionLimitGate',
    'PositionLimits',
    'get_position_limit_gate',
]
