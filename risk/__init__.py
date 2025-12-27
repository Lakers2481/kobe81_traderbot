"""
Risk Management Module
======================

Core risk management for the Kobe trading system.

Components:
- policy_gate: Per-order ($75) and daily ($1k) budget enforcement
- advanced/: Advanced risk analytics (VaR, Kelly, Correlation)
"""

from .policy_gate import PolicyGate

# Import advanced submodule for explicit access
from . import advanced

__all__ = [
    'PolicyGate',
    'advanced',
]
