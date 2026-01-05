"""
Circuit Breaker System - Institutional-Grade Auto-Protection

Automatic protection that STOPS TRADING when things go wrong.
You're not watching - the system protects you 24/7.

Components:
- DrawdownBreaker: Daily/weekly/max drawdown limits
- VolatilityBreaker: VIX-based position reduction/halt
- StreakBreaker: Consecutive loss protection
- CorrelationBreaker: Correlation regime change detection
- ExecutionBreaker: Slippage anomaly detection

Author: Kobe Trading System
Created: 2026-01-04
"""

from .breaker_manager import (
    BreakerManager,
    BreakerAction,
    BreakerStatus,
    BreakerAlert,
    get_breaker_manager,
    check_all_breakers,
    get_breaker_status,
)
from .drawdown_breaker import DrawdownBreaker
from .volatility_breaker import VolatilityBreaker
from .streak_breaker import StreakBreaker
from .correlation_breaker import CorrelationBreaker
from .execution_breaker import ExecutionBreaker

__all__ = [
    "BreakerManager",
    "BreakerAction",
    "BreakerStatus",
    "BreakerAlert",
    "get_breaker_manager",
    "check_all_breakers",
    "get_breaker_status",
    "DrawdownBreaker",
    "VolatilityBreaker",
    "StreakBreaker",
    "CorrelationBreaker",
    "ExecutionBreaker",
]
