"""
Dynamic Data Exploration Module
================================

Tools for analyzing feature importance, discovering patterns,
and registering available data sources.

Components:
- FeatureImportance: Analyze which features drive predictions
- DataRegistry: Track available data sources and features
- FeatureDiscovery: Automatically discover useful features

Usage:
    from data_exploration import FeatureImportance, analyze_importance

    analyzer = FeatureImportance()
    importance = analyzer.analyze(X, y, feature_names)
"""

from .feature_importance import (
    FeatureImportance,
    ImportanceResult,
    ImportanceMethod,
    analyze_importance,
    get_top_features,
)

from .data_registry import (
    DataRegistry,
    DataSource,
    FeatureDefinition,
    register_source,
    get_available_features,
)

from .feature_discovery import (
    FeatureDiscovery,
    DiscoveredFeature,
    discover_features,
    suggest_features,
)

__all__ = [
    # Feature Importance
    'FeatureImportance',
    'ImportanceResult',
    'ImportanceMethod',
    'analyze_importance',
    'get_top_features',
    # Data Registry
    'DataRegistry',
    'DataSource',
    'FeatureDefinition',
    'register_source',
    'get_available_features',
    # Feature Discovery
    'FeatureDiscovery',
    'DiscoveredFeature',
    'discover_features',
    'suggest_features',
]
