"""
Alpha Decay Monitor - Know When Your Edge is Dying

Strategies have a shelf life. This module tells you when to retire
them BEFORE you lose money.

Components:
- SignalDecay: How fast signals lose value
- InformationRatio: Rolling IC tracking
- CrowdingDetector: Is everyone trading this?
- RegimeShift: Has the market changed?
- DecayAlerter: Alert when edge dying

Solo Trader Features:
- Simple health dashboard (green/yellow/red)
- Auto-reduce allocation to degrading strategies
- Alert when strategy should be retired
- Suggest when to re-evaluate

Author: Kobe Trading System
Created: 2026-01-04
"""

from .alpha_monitor import (
    AlphaDecayMonitor,
    AlphaHealth,
    StrategyHealth,
    get_alpha_monitor,
)

__all__ = [
    "AlphaDecayMonitor",
    "AlphaHealth",
    "StrategyHealth",
    "get_alpha_monitor",
]
