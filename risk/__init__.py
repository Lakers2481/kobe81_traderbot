"""
Risk Management Module
======================

Core risk management for the Kobe trading system.

Components:
- policy_gate: Per-order ($75) and daily ($1k) budget enforcement
- advanced/: Advanced risk analytics (VaR, Kelly, Correlation)
"""

from .policy_gate import PolicyGate

# Note: advanced submodule is available but not auto-imported
# to avoid dependency issues. Import explicitly:
#   from risk.advanced import MonteCarloVaR, KellyPositionSizer

__all__ = [
    'PolicyGate',
]
