"""
Analytics Module for Edge Analysis and Strategy Monitoring.

Provides tools for understanding and monitoring trading edge:
- Edge Decomposition: Stratify performance by multiple dimensions
- Factor Attribution: Identify which signal components drive profits
- Auto Stand-down: Automatic position reduction on edge degradation
"""

from analytics.edge_decomposition import (
    EdgeDecomposition,
    DecompositionResult,
    DimensionStats,
)
from analytics.factor_attribution import (
    FactorAttribution,
    AttributionResult,
    FactorContribution,
)
from analytics.auto_standdown import (
    AutoStanddown,
    StanddownRecommendation,
    StanddownSeverity,
)

__all__ = [
    # Edge Decomposition
    "EdgeDecomposition",
    "DecompositionResult",
    "DimensionStats",
    # Factor Attribution
    "FactorAttribution",
    "AttributionResult",
    "FactorContribution",
    # Auto Stand-down
    "AutoStanddown",
    "StanddownRecommendation",
    "StanddownSeverity",
]
