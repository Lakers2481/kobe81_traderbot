"""
Analysis Module - Historical Patterns and Market Analysis
=========================================================

This module provides comprehensive market analysis tools for the Pre-Game Blueprint:

- Historical pattern analysis (consecutive days, reversal rates)
- Options expected move calculation
- Support/resistance identification
- Volume profile analysis
- Sector relative strength

Usage:
    from analysis.historical_patterns import HistoricalPatternAnalyzer
    from analysis.options_expected_move import ExpectedMoveCalculator
"""

from analysis.historical_patterns import (
    HistoricalPatternAnalyzer,
    ConsecutiveDayPattern,
    SupportResistanceLevel,
)
from analysis.options_expected_move import (
    ExpectedMoveCalculator,
    ExpectedMove,
)

__all__ = [
    'HistoricalPatternAnalyzer',
    'ConsecutiveDayPattern',
    'SupportResistanceLevel',
    'ExpectedMoveCalculator',
    'ExpectedMove',
]
