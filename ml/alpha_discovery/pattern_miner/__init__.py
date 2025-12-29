"""
Pattern Miner - ML-based pattern discovery via clustering.

Discovers recurring price patterns from historical trade data using
KMeans and DBSCAN clustering algorithms.
"""

from .clustering import (
    PatternCluster,
    PatternClusteringEngine,
)
from .pattern_library import PatternLibrary

__all__ = [
    'PatternCluster',
    'PatternClusteringEngine',
    'PatternLibrary',
]
