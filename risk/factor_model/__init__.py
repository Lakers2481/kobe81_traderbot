"""
Factor Risk Model - Understand Your TRUE Exposures

You might think you're diversified but actually all-in on momentum.
This module decomposes your portfolio by systematic risk factors.

Components:
- FactorCalculator: Calculate factor exposures
- FactorDecomposition: Decompose returns by factor
- BetaTracker: Market beta monitoring
- SectorExposure: Sector concentration
- FactorReport: Exposure reports

Solo Trader Features:
- Automatic alerts if concentration too high
- Factor drift monitoring
- Sector neutralization option
- Beta hedge suggestions

Author: Kobe Trading System
Created: 2026-01-04
"""

from .factor_calculator import (
    FactorCalculator,
    FactorExposures,
    get_factor_calculator,
)
from .sector_exposure import (
    SectorAnalyzer,
    SectorExposures,
)
from .factor_report import (
    FactorRiskReporter,
    FactorRiskReport,
    generate_factor_report,
)

__all__ = [
    "FactorCalculator",
    "FactorExposures",
    "get_factor_calculator",
    "SectorAnalyzer",
    "SectorExposures",
    "FactorRiskReporter",
    "FactorRiskReport",
    "generate_factor_report",
]
