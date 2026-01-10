"""
Feature Registry Module.

This module provides centralized feature management for the Kobe trading system.
It includes:
- FeatureMetadata: Metadata about each feature (name, version, dependencies)
- FeatureRegistry: Central registry for all features
- FeatureStore: Persistence layer for computed features

Blueprint Alignment:
    This implements the "Feature Registry" requirement from Section 2.3
    of the production-grade trading system blueprint.

Usage:
    from features import FeatureRegistry, FeatureMetadata

    # Register a feature
    registry = FeatureRegistry()
    registry.register(FeatureMetadata(
        name="rsi_2",
        version="1.0.0",
        category="momentum",
        lookback_periods=2,
        is_shifted=True,
    ))

    # Get feature info
    meta = registry.get("rsi_2")
    print(meta.lookback_periods)
"""

from features.registry import (
    FeatureMetadata,
    FeatureRegistry,
    FeatureCategory,
    get_global_registry,
)
from features.store import (
    FeatureStore,
    FeatureSetManifest,
    get_global_store,
)

__all__ = [
    # Registry
    "FeatureMetadata",
    "FeatureRegistry",
    "FeatureCategory",
    "get_global_registry",
    # Store
    "FeatureStore",
    "FeatureSetManifest",
    "get_global_store",
]
