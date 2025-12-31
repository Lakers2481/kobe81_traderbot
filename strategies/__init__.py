"""
Kobe Trading Strategies
=======================

Production System: Dual Strategy (IBS+RSI + Turtle Soup)

v2.2 VERIFIED PERFORMANCE (2015-2024, 200 symbols tested):
- IBS+RSI:     59.9% WR, 1.46 PF (867 trades)
- Turtle Soup: 61.0% WR, 1.37 PF (305 trades)
- Combined:    60.2% WR, 1.44 PF (1,172 trades)

Key v2.2 Optimization:
- LOOSER entry + TIGHTER exits = higher WR for mean-reversion
- Turtle Soup: 0.3 ATR sweep, 0.5R target, 3-bar time stop
- IBS+RSI: IBS < 0.08, RSI < 5, 7-bar time stop

Replication:
    python scripts/backtest_dual_strategy.py --cap 200 --start 2015-01-01 --end 2024-12-31

Usage:
    from strategies.dual_strategy import DualStrategyScanner
    scanner = DualStrategyScanner()
    signals = scanner.generate_signals(df)
    top3 = signals.head(3)
    totd = signals.iloc[0]

See docs/V2.2_OPTIMIZATION_GUIDE.md for full methodology.
"""

# Primary Strategy - Use This
from .dual_strategy.combined import DualStrategyScanner, DualStrategyParams

# Production scanner factory
from .registry import get_production_scanner

# DEPRECATED: Standalone strategies are NOT exported at package level
# They exist for reference/testing only. Use DualStrategyScanner instead.
# DO NOT import: IbsRsiStrategy, TurtleSoupStrategy
# These are kept in their modules but not exported here.

__all__ = [
    # Production - Use These
    "DualStrategyScanner",
    "DualStrategyParams",
    "get_production_scanner",
]
