"""
Data Registry for Feature Management
======================================

Tracks available data sources, features, and their metadata.
Provides a central catalog of all data available to the
trading system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureCategory(Enum):
    """Category of feature."""
    PRICE = "price"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    SENTIMENT = "sentiment"
    FUNDAMENTAL = "fundamental"
    DERIVED = "derived"


class DataFrequency(Enum):
    """Frequency of data updates."""
    TICK = "tick"
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class FeatureDefinition:
    """Definition of a feature."""
    name: str
    category: FeatureCategory
    description: str = ""
    source: str = ""
    frequency: DataFrequency = DataFrequency.DAILY

    # Value characteristics
    dtype: str = "float"
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # Feature engineering info
    lookback: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)

    # Usage tracking
    used_in_strategies: List[str] = field(default_factory=list)
    importance_score: Optional[float] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'category': self.category.value,
            'description': self.description,
            'source': self.source,
            'frequency': self.frequency.value,
            'dtype': self.dtype,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'lookback': self.lookback,
            'dependencies': self.dependencies,
            'used_in_strategies': self.used_in_strategies,
            'importance_score': self.importance_score,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class DataSource:
    """Definition of a data source."""
    name: str
    provider: str
    description: str = ""
    frequency: DataFrequency = DataFrequency.DAILY

    # Connection info
    api_endpoint: Optional[str] = None
    requires_auth: bool = False

    # Coverage
    symbols_covered: List[str] = field(default_factory=list)
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None

    # Features provided
    features: List[str] = field(default_factory=list)

    # Health status
    is_active: bool = True
    last_updated: Optional[datetime] = None
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'provider': self.provider,
            'description': self.description,
            'frequency': self.frequency.value,
            'requires_auth': self.requires_auth,
            'symbols_covered_count': len(self.symbols_covered),
            'features': self.features,
            'is_active': self.is_active,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
        }


class DataRegistry:
    """
    Central registry for data sources and features.

    Maintains catalog of available data and their metadata
    for discovery, validation, and dependency tracking.
    """

    def __init__(
        self,
        registry_path: Optional[Path] = None,
        auto_save: bool = True,
    ):
        """
        Initialize the data registry.

        Args:
            registry_path: Path to persist registry
            auto_save: Whether to auto-save on changes
        """
        self.registry_path = Path(registry_path) if registry_path else Path("data/registry.json")
        self.auto_save = auto_save

        # In-memory registries
        self._sources: Dict[str, DataSource] = {}
        self._features: Dict[str, FeatureDefinition] = {}

        # Load existing registry
        self._load()

        logger.info(
            f"DataRegistry initialized with {len(self._sources)} sources, "
            f"{len(self._features)} features"
        )

    def _load(self):
        """Load registry from disk."""
        if not self.registry_path.exists():
            return

        try:
            with open(self.registry_path) as f:
                data = json.load(f)

            # Load sources
            for source_data in data.get('sources', []):
                source = DataSource(
                    name=source_data['name'],
                    provider=source_data['provider'],
                    description=source_data.get('description', ''),
                    features=source_data.get('features', []),
                )
                self._sources[source.name] = source

            # Load features
            for feat_data in data.get('features', []):
                feature = FeatureDefinition(
                    name=feat_data['name'],
                    category=FeatureCategory(feat_data.get('category', 'derived')),
                    description=feat_data.get('description', ''),
                    source=feat_data.get('source', ''),
                )
                self._features[feature.name] = feature

        except Exception as e:
            logger.warning(f"Failed to load registry: {e}")

    def _save(self):
        """Save registry to disk."""
        if not self.auto_save:
            return

        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'sources': [s.to_dict() for s in self._sources.values()],
                'features': [f.to_dict() for f in self._features.values()],
                'updated_at': datetime.now().isoformat(),
            }

            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def register_source(self, source: DataSource):
        """Register a data source."""
        self._sources[source.name] = source

        # Register features from source
        for feature_name in source.features:
            if feature_name not in self._features:
                self._features[feature_name] = FeatureDefinition(
                    name=feature_name,
                    category=FeatureCategory.DERIVED,
                    source=source.name,
                )

        self._save()
        logger.info(f"Registered source: {source.name}")

    def register_feature(self, feature: FeatureDefinition):
        """Register a feature definition."""
        self._features[feature.name] = feature
        self._save()
        logger.info(f"Registered feature: {feature.name}")

    def get_source(self, name: str) -> Optional[DataSource]:
        """Get a data source by name."""
        return self._sources.get(name)

    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """Get a feature by name."""
        return self._features.get(name)

    def list_sources(self) -> List[DataSource]:
        """List all data sources."""
        return list(self._sources.values())

    def list_features(
        self,
        category: Optional[FeatureCategory] = None,
        source: Optional[str] = None,
    ) -> List[FeatureDefinition]:
        """List features with optional filtering."""
        features = list(self._features.values())

        if category:
            features = [f for f in features if f.category == category]

        if source:
            features = [f for f in features if f.source == source]

        return features

    def get_feature_dependencies(self, name: str) -> List[str]:
        """Get all dependencies for a feature."""
        feature = self._features.get(name)
        if not feature:
            return []

        deps = set(feature.dependencies)

        # Recursively get dependencies
        for dep in feature.dependencies:
            deps.update(self.get_feature_dependencies(dep))

        return list(deps)

    def search_features(self, query: str) -> List[FeatureDefinition]:
        """Search features by name or description."""
        query = query.lower()
        return [
            f for f in self._features.values()
            if query in f.name.lower() or query in f.description.lower()
        ]

    def get_features_for_category(
        self,
        category: FeatureCategory,
    ) -> List[FeatureDefinition]:
        """Get all features in a category."""
        return [f for f in self._features.values() if f.category == category]

    def update_importance(self, feature_name: str, importance: float):
        """Update the importance score for a feature."""
        if feature_name in self._features:
            self._features[feature_name].importance_score = importance
            self._features[feature_name].updated_at = datetime.now()
            self._save()

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        categories = {}
        for f in self._features.values():
            cat = f.category.value
            categories[cat] = categories.get(cat, 0) + 1

        return {
            'total_sources': len(self._sources),
            'active_sources': sum(1 for s in self._sources.values() if s.is_active),
            'total_features': len(self._features),
            'features_by_category': categories,
        }

    def export_catalog(self) -> Dict[str, Any]:
        """Export the full catalog."""
        return {
            'sources': [s.to_dict() for s in self._sources.values()],
            'features': [f.to_dict() for f in self._features.values()],
            'stats': self.get_stats(),
            'exported_at': datetime.now().isoformat(),
        }


def register_source(
    name: str,
    provider: str,
    features: List[str],
    **kwargs,
):
    """Convenience function to register a data source."""
    registry = get_registry()
    source = DataSource(
        name=name,
        provider=provider,
        features=features,
        **kwargs,
    )
    registry.register_source(source)


def get_available_features(
    category: Optional[str] = None,
) -> List[str]:
    """Get list of available feature names."""
    registry = get_registry()
    features = registry.list_features(
        category=FeatureCategory(category) if category else None
    )
    return [f.name for f in features]


# Module-level registry
_registry: Optional[DataRegistry] = None


def get_registry() -> DataRegistry:
    """Get or create the global registry instance."""
    global _registry
    if _registry is None:
        _registry = DataRegistry()
    return _registry
