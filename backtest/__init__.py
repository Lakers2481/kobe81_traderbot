"""
Backtest module for Kobe Trading System.
Provides backtesting engine and walk-forward analysis.
"""
from __future__ import annotations

from .engine import (
    Backtester,
    BacktestConfig,
    CommissionConfig,
    Trade,
    Position,
)
from .walk_forward import (
    WFSplit,
    generate_splits,
    run_walk_forward,
    summarize_results,
    train_start_to_date,
)

__all__ = [
    # Engine exports
    "Backtester",
    "BacktestConfig",
    "CommissionConfig",
    "Trade",
    "Position",
    # Walk-forward exports
    "WFSplit",
    "generate_splits",
    "run_walk_forward",
    "summarize_results",
    "train_start_to_date",
]
