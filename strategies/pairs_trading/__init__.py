"""
Pairs Trading Module - Renaissance-Inspired Statistical Arbitrage

This module implements pairs trading (statistical arbitrage) inspired by
Renaissance Technologies' approach:
- Cointegration-based pair selection
- Z-score spread trading
- HMM regime integration
- Kelly-optimal position sizing

Key Components:
- CorrelationScanner: Find cointegrated pairs in 900-stock universe
- SpreadCalculator: Calculate spreads, z-scores, half-life
- PairsStrategy: Generate long/short signals
- PairsBacktest: Validate strategy performance

Renaissance Insight:
"We're right 50.75% of the time... but we're 100% right 50.75% of the time."
- Robert Mercer

The edge comes from:
1. Statistical edge (cointegration = mean reversion guarantee)
2. Diversification (many uncorrelated pairs)
3. Risk management (z-score based entry/exit)
4. Regime awareness (only trade in mean-reverting regimes)
"""

from .correlation_scanner import CorrelationScanner, PairCandidate
from .spread_calculator import SpreadCalculator, SpreadState
from .pairs_strategy import PairsStrategy, PairsSignal, PairsParams
from .regime_integration import RegimeAwarePairs

__all__ = [
    'CorrelationScanner',
    'PairCandidate',
    'SpreadCalculator',
    'SpreadState',
    'PairsStrategy',
    'PairsSignal',
    'PairsParams',
    'RegimeAwarePairs',
]

__version__ = '1.0.0'
