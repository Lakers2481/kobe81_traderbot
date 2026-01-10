"""
Feature Registry for centralized feature management.

This module provides:
- FeatureMetadata: Complete metadata about a feature
- FeatureRegistry: Central registry with versioning and lineage tracking
- FeatureCategory: Enum of feature categories

Blueprint Alignment:
    Implements Section 2.3 requirements for feature management with:
    - Version tracking
    - Dependency tracking
    - Lookahead bias prevention (is_shifted flag)
    - Lineage tracking
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import hashlib


class FeatureCategory(Enum):
    """Categories for features."""
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    VOLUME = "volume"
    PRICE_ACTION = "price_action"
    PATTERN = "pattern"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    DERIVED = "derived"
    CUSTOM = "custom"


@dataclass
class FeatureMetadata:
    """
    Complete metadata about a feature.

    This metadata enables:
    - Version tracking for reproducibility
    - Dependency management
    - Lookahead bias prevention
    - Feature lineage

    Attributes:
        name: Unique feature name
        version: Semantic version (e.g., "1.0.0")
        category: Feature category (momentum, volatility, etc.)
        description: Human-readable description
        dependencies: List of feature names this depends on
        lookback_periods: Number of historical bars required
        is_shifted: Whether the feature uses .shift(1) for lookahead safety
        author: Who created this feature
        created_at: When the feature was registered
        params: Feature-specific parameters
    """
    name: str
    version: str = "1.0.0"
    category: FeatureCategory = FeatureCategory.CUSTOM
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    lookback_periods: int = 0
    is_shifted: bool = True  # Default to shifted for safety
    author: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate metadata after initialization."""
        if not self.name:
            raise ValueError("Feature name cannot be empty")
        if not self.version:
            raise ValueError("Feature version cannot be empty")

    @property
    def full_name(self) -> str:
        """Get fully qualified name with version."""
        return f"{self.name}@{self.version}"

    @property
    def feature_hash(self) -> str:
        """Generate deterministic hash of feature definition."""
        hash_input = json.dumps({
            "name": self.name,
            "version": self.version,
            "category": self.category.value,
            "lookback_periods": self.lookback_periods,
            "is_shifted": self.is_shifted,
            "params": self.params,
        }, sort_keys=True)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "category": self.category.value,
            "description": self.description,
            "dependencies": self.dependencies,
            "lookback_periods": self.lookback_periods,
            "is_shifted": self.is_shifted,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "params": self.params,
            "feature_hash": self.feature_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureMetadata":
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            category=FeatureCategory(data.get("category", "custom")),
            description=data.get("description", ""),
            dependencies=data.get("dependencies", []),
            lookback_periods=data.get("lookback_periods", 0),
            is_shifted=data.get("is_shifted", True),
            author=data.get("author", "system"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            params=data.get("params", {}),
        )


class FeatureRegistry:
    """
    Central registry for all features.

    The registry maintains:
    - All registered features with their metadata
    - Version history for each feature
    - Dependency graph for lineage tracking

    Thread-safe through immutable operations.
    """

    def __init__(self, persist_path: Optional[Path] = None):
        """
        Initialize the registry.

        Args:
            persist_path: Optional path to persist registry to disk
        """
        self._features: Dict[str, Dict[str, FeatureMetadata]] = {}  # name -> version -> metadata
        self._persist_path = persist_path

        # Load from disk if available
        if persist_path and persist_path.exists():
            self._load()

    def register(
        self,
        metadata: FeatureMetadata,
        overwrite: bool = False
    ) -> None:
        """
        Register a feature.

        Args:
            metadata: Feature metadata to register
            overwrite: If True, allow overwriting existing version

        Raises:
            ValueError: If feature version exists and overwrite=False
        """
        name = metadata.name
        version = metadata.version

        if name not in self._features:
            self._features[name] = {}

        if version in self._features[name] and not overwrite:
            raise ValueError(
                f"Feature {name}@{version} already registered. "
                f"Use overwrite=True to replace."
            )

        self._features[name][version] = metadata

        # Persist if path configured
        if self._persist_path:
            self._save()

    def get(
        self,
        name: str,
        version: str = "latest"
    ) -> Optional[FeatureMetadata]:
        """
        Get feature metadata.

        Args:
            name: Feature name
            version: Specific version or "latest"

        Returns:
            FeatureMetadata if found, None otherwise
        """
        if name not in self._features:
            return None

        versions = self._features[name]

        if version == "latest":
            # Get the highest version
            if not versions:
                return None
            latest = max(versions.keys(), key=self._parse_version)
            return versions[latest]

        return versions.get(version)

    def _parse_version(self, version: str) -> tuple:
        """Parse semantic version string to tuple for comparison."""
        try:
            parts = version.split(".")
            return tuple(int(p) for p in parts)
        except (ValueError, AttributeError):
            return (0, 0, 0)

    def get_all(self, name: str) -> Dict[str, FeatureMetadata]:
        """
        Get all versions of a feature.

        Args:
            name: Feature name

        Returns:
            Dict mapping version to metadata
        """
        return self._features.get(name, {})

    def list_features(
        self,
        category: Optional[FeatureCategory] = None,
        shifted_only: bool = False
    ) -> List[FeatureMetadata]:
        """
        List all registered features.

        Args:
            category: Optional filter by category
            shifted_only: If True, only return shifted features

        Returns:
            List of feature metadata (latest version of each)
        """
        result = []
        for name in self._features:
            meta = self.get(name, "latest")
            if meta is None:
                continue

            if category and meta.category != category:
                continue

            if shifted_only and not meta.is_shifted:
                continue

            result.append(meta)

        return result

    def get_lineage(self, name: str) -> Dict[str, Any]:
        """
        Get feature lineage (dependency tree).

        Args:
            name: Feature name

        Returns:
            Dict with dependency information
        """
        meta = self.get(name)
        if meta is None:
            return {"name": name, "error": "Feature not found"}

        lineage = {
            "name": name,
            "version": meta.version,
            "category": meta.category.value,
            "dependencies": [],
        }

        for dep in meta.dependencies:
            dep_lineage = self.get_lineage(dep)
            lineage["dependencies"].append(dep_lineage)

        return lineage

    def validate_dependencies(self, name: str) -> tuple[bool, List[str]]:
        """
        Validate that all dependencies of a feature are registered.

        Args:
            name: Feature name

        Returns:
            Tuple of (valid, missing_dependencies)
        """
        meta = self.get(name)
        if meta is None:
            return False, [f"{name} (not found)"]

        missing = []
        for dep in meta.dependencies:
            if self.get(dep) is None:
                missing.append(dep)

        return len(missing) == 0, missing

    def get_required_lookback(self, names: List[str]) -> int:
        """
        Get maximum lookback required for a set of features.

        This is crucial for ensuring enough historical data.

        Args:
            names: List of feature names

        Returns:
            Maximum lookback periods required
        """
        max_lookback = 0
        visited: Set[str] = set()

        def _get_lookback(name: str) -> int:
            if name in visited:
                return 0
            visited.add(name)

            meta = self.get(name)
            if meta is None:
                return 0

            lb = meta.lookback_periods
            for dep in meta.dependencies:
                lb = max(lb, _get_lookback(dep))

            return lb

        for name in names:
            max_lookback = max(max_lookback, _get_lookback(name))

        return max_lookback

    def check_lookahead_safety(self, names: List[str]) -> tuple[bool, List[str]]:
        """
        Check that all features are shifted (lookahead-safe).

        Args:
            names: List of feature names

        Returns:
            Tuple of (all_safe, unsafe_features)
        """
        unsafe = []
        for name in names:
            meta = self.get(name)
            if meta and not meta.is_shifted:
                unsafe.append(name)

        return len(unsafe) == 0, unsafe

    def _save(self) -> None:
        """Save registry to disk."""
        if not self._persist_path:
            return

        data = {}
        for name, versions in self._features.items():
            data[name] = {
                version: meta.to_dict()
                for version, meta in versions.items()
            }

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        """Load registry from disk."""
        if not self._persist_path or not self._persist_path.exists():
            return

        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for name, versions in data.items():
                self._features[name] = {
                    version: FeatureMetadata.from_dict(meta_dict)
                    for version, meta_dict in versions.items()
                }
        except Exception:
            # If loading fails, start with empty registry
            pass

    def to_dict(self) -> Dict[str, Any]:
        """Export registry as dictionary."""
        return {
            name: {
                version: meta.to_dict()
                for version, meta in versions.items()
            }
            for name, versions in self._features.items()
        }


# Global registry instance
_global_registry: Optional[FeatureRegistry] = None


def get_global_registry() -> FeatureRegistry:
    """
    Get the global feature registry instance.

    Returns:
        Global FeatureRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        persist_path = Path("state/feature_registry.json")
        _global_registry = FeatureRegistry(persist_path)
        _register_core_features(_global_registry)
    return _global_registry


def _register_core_features(registry: FeatureRegistry) -> None:
    """Register core features used by the Kobe trading system."""
    core_features = [
        FeatureMetadata(
            name="rsi_2",
            version="1.0.0",
            category=FeatureCategory.MOMENTUM,
            description="2-period RSI indicator",
            lookback_periods=3,
            is_shifted=True,
            params={"period": 2},
        ),
        FeatureMetadata(
            name="ibs",
            version="1.0.0",
            category=FeatureCategory.PRICE_ACTION,
            description="Internal Bar Strength: (Close - Low) / (High - Low)",
            lookback_periods=1,
            is_shifted=True,
        ),
        FeatureMetadata(
            name="sma_200",
            version="1.0.0",
            category=FeatureCategory.TREND,
            description="200-period Simple Moving Average",
            lookback_periods=200,
            is_shifted=True,
            params={"period": 200},
        ),
        FeatureMetadata(
            name="atr_14",
            version="1.0.0",
            category=FeatureCategory.VOLATILITY,
            description="14-period Average True Range",
            lookback_periods=15,
            is_shifted=True,
            params={"period": 14},
        ),
        FeatureMetadata(
            name="sweep_strength",
            version="1.0.0",
            category=FeatureCategory.PRICE_ACTION,
            description="Liquidity sweep strength (low penetration / ATR)",
            lookback_periods=15,
            is_shifted=True,
            dependencies=["atr_14"],
        ),
        FeatureMetadata(
            name="close_above_sma200",
            version="1.0.0",
            category=FeatureCategory.TREND,
            description="Binary flag: Close > SMA(200)",
            lookback_periods=200,
            is_shifted=True,
            dependencies=["sma_200"],
        ),
        FeatureMetadata(
            name="markov_state",
            version="1.0.0",
            category=FeatureCategory.PATTERN,
            description="Markov chain state classification (Up/Down/Flat)",
            lookback_periods=20,
            is_shifted=True,
        ),
        FeatureMetadata(
            name="consecutive_down_days",
            version="1.0.0",
            category=FeatureCategory.PATTERN,
            description="Count of consecutive down days",
            lookback_periods=10,
            is_shifted=True,
        ),
    ]

    for meta in core_features:
        try:
            registry.register(meta)
        except ValueError:
            # Already registered, skip
            pass
