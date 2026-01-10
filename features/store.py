"""
Feature Store for computed feature persistence.

This module provides storage and retrieval of computed features,
enabling:
- Feature caching for performance
- Feature set versioning for reproducibility
- Time-travel queries for historical features

Blueprint Alignment:
    Implements Section 2.3 requirements for feature persistence with:
    - Parquet storage for efficient columnar access
    - SHA256 hashes for integrity verification
    - Feature set versioning
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import pandas as pd


@dataclass
class FeatureSetManifest:
    """
    Manifest for a stored feature set.

    Contains metadata about the stored features for reproducibility.
    """
    feature_set_id: str
    created_at: datetime
    symbols: List[str]
    features: List[str]
    date_range: tuple[str, str]
    file_path: str
    file_hash: str
    row_count: int
    registry_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "feature_set_id": self.feature_set_id,
            "created_at": self.created_at.isoformat(),
            "symbols": self.symbols,
            "features": self.features,
            "date_range": list(self.date_range),
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "row_count": self.row_count,
            "registry_snapshot": self.registry_snapshot,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSetManifest":
        """Deserialize from dictionary."""
        return cls(
            feature_set_id=data["feature_set_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            symbols=data["symbols"],
            features=data["features"],
            date_range=tuple(data["date_range"]),
            file_path=data["file_path"],
            file_hash=data["file_hash"],
            row_count=data["row_count"],
            registry_snapshot=data.get("registry_snapshot", {}),
        )


class FeatureStore:
    """
    Storage layer for computed features.

    The store provides:
    - Save/load of feature DataFrames
    - Feature set versioning
    - Integrity verification
    - Cache management
    """

    def __init__(
        self,
        base_dir: Path = Path("data/features"),
        manifest_path: Optional[Path] = None
    ):
        """
        Initialize the feature store.

        Args:
            base_dir: Base directory for feature storage
            manifest_path: Optional path to manifest file
        """
        self.base_dir = Path(base_dir)
        self.manifest_path = manifest_path or (self.base_dir / "manifest.json")
        self._manifests: Dict[str, FeatureSetManifest] = {}

        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Load existing manifests
        self._load_manifests()

    def save_features(
        self,
        features_df: pd.DataFrame,
        feature_names: List[str],
        symbols: Optional[List[str]] = None,
        registry_snapshot: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save computed features to the store.

        Args:
            features_df: DataFrame with features (must have 'symbol', 'date' columns)
            feature_names: List of feature column names
            symbols: Optional list of symbols (inferred from df if not provided)
            registry_snapshot: Optional registry metadata snapshot

        Returns:
            Feature set ID for later retrieval
        """
        # Validate required columns
        required_cols = {"symbol", "date"}
        if not required_cols.issubset(features_df.columns):
            missing = required_cols - set(features_df.columns)
            raise ValueError(f"DataFrame missing required columns: {missing}")

        # Validate feature columns exist
        missing_features = set(feature_names) - set(features_df.columns)
        if missing_features:
            raise ValueError(f"DataFrame missing feature columns: {missing_features}")

        # Get date range
        dates = pd.to_datetime(features_df["date"])
        date_range = (dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d"))

        # Get symbols
        if symbols is None:
            symbols = sorted(features_df["symbol"].unique().tolist())

        # Generate feature set ID
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        content_hash = self._compute_df_hash(features_df[["symbol", "date"] + feature_names])
        feature_set_id = f"fs_{timestamp}_{content_hash[:8]}"

        # Save to parquet
        file_path = self.base_dir / f"{feature_set_id}.parquet"
        features_df.to_parquet(file_path, index=False)

        # Compute file hash
        file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

        # Create manifest
        manifest = FeatureSetManifest(
            feature_set_id=feature_set_id,
            created_at=datetime.utcnow(),
            symbols=symbols,
            features=feature_names,
            date_range=date_range,
            file_path=str(file_path),
            file_hash=file_hash,
            row_count=len(features_df),
            registry_snapshot=registry_snapshot or {},
        )

        # Store and save manifest
        self._manifests[feature_set_id] = manifest
        self._save_manifests()

        return feature_set_id

    def load_features(
        self,
        feature_set_id: str,
        symbols: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        verify_integrity: bool = True,
    ) -> pd.DataFrame:
        """
        Load features from the store.

        Args:
            feature_set_id: ID of the feature set to load
            symbols: Optional filter to specific symbols
            features: Optional filter to specific features
            verify_integrity: If True, verify file hash before loading

        Returns:
            DataFrame with requested features

        Raises:
            KeyError: If feature set not found
            ValueError: If integrity check fails
        """
        if feature_set_id not in self._manifests:
            raise KeyError(f"Feature set not found: {feature_set_id}")

        manifest = self._manifests[feature_set_id]
        file_path = Path(manifest.file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Feature file not found: {file_path}")

        # Verify integrity
        if verify_integrity:
            actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if actual_hash != manifest.file_hash:
                raise ValueError(
                    f"Feature file integrity check failed for {feature_set_id}. "
                    f"Expected hash {manifest.file_hash[:12]}..., got {actual_hash[:12]}..."
                )

        # Load data
        df = pd.read_parquet(file_path)

        # Filter symbols if requested
        if symbols:
            df = df[df["symbol"].isin(symbols)]

        # Filter features if requested
        if features:
            # Always include symbol and date
            cols = ["symbol", "date"] + [f for f in features if f in df.columns]
            df = df[cols]

        return df

    def get_manifest(self, feature_set_id: str) -> Optional[FeatureSetManifest]:
        """Get manifest for a feature set."""
        return self._manifests.get(feature_set_id)

    def list_feature_sets(
        self,
        symbols: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        after: Optional[datetime] = None,
    ) -> List[FeatureSetManifest]:
        """
        List available feature sets with optional filtering.

        Args:
            symbols: Filter to sets containing these symbols
            features: Filter to sets containing these features
            after: Filter to sets created after this time

        Returns:
            List of matching manifests
        """
        result = []

        for manifest in self._manifests.values():
            # Filter by creation time
            if after and manifest.created_at < after:
                continue

            # Filter by symbols
            if symbols:
                if not set(symbols).issubset(set(manifest.symbols)):
                    continue

            # Filter by features
            if features:
                if not set(features).issubset(set(manifest.features)):
                    continue

            result.append(manifest)

        # Sort by creation time descending
        result.sort(key=lambda m: m.created_at, reverse=True)

        return result

    def get_latest(
        self,
        symbols: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Get the latest feature set ID matching the criteria.

        Args:
            symbols: Filter to sets containing these symbols
            features: Filter to sets containing these features

        Returns:
            Feature set ID or None
        """
        matching = self.list_feature_sets(symbols=symbols, features=features)
        return matching[0].feature_set_id if matching else None

    def delete(self, feature_set_id: str, delete_file: bool = True) -> bool:
        """
        Delete a feature set.

        Args:
            feature_set_id: ID of set to delete
            delete_file: If True, also delete the data file

        Returns:
            True if deleted, False if not found
        """
        if feature_set_id not in self._manifests:
            return False

        manifest = self._manifests[feature_set_id]

        # Delete file if requested
        if delete_file:
            file_path = Path(manifest.file_path)
            if file_path.exists():
                file_path.unlink()

        # Remove from manifests
        del self._manifests[feature_set_id]
        self._save_manifests()

        return True

    def cleanup_old(self, keep_latest: int = 5) -> int:
        """
        Clean up old feature sets, keeping the latest N.

        Args:
            keep_latest: Number of latest sets to keep

        Returns:
            Number of sets deleted
        """
        all_sets = list(self._manifests.values())
        all_sets.sort(key=lambda m: m.created_at, reverse=True)

        to_delete = all_sets[keep_latest:]
        deleted = 0

        for manifest in to_delete:
            if self.delete(manifest.feature_set_id):
                deleted += 1

        return deleted

    def _compute_df_hash(self, df: pd.DataFrame) -> str:
        """Compute deterministic hash of DataFrame contents."""
        # Sort for determinism
        df_sorted = df.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Hash the bytes representation
        content = df_sorted.to_csv(index=False).encode()
        return hashlib.sha256(content).hexdigest()

    def _save_manifests(self) -> None:
        """Save all manifests to disk."""
        data = {
            fsid: manifest.to_dict()
            for fsid, manifest in self._manifests.items()
        }
        self.manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_manifests(self) -> None:
        """Load manifests from disk."""
        if not self.manifest_path.exists():
            return

        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._manifests = {
                fsid: FeatureSetManifest.from_dict(manifest_data)
                for fsid, manifest_data in data.items()
            }
        except Exception:
            # If loading fails, start fresh
            pass

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the feature store cache."""
        total_size = 0
        for manifest in self._manifests.values():
            file_path = Path(manifest.file_path)
            if file_path.exists():
                total_size += file_path.stat().st_size

        return {
            "total_feature_sets": len(self._manifests),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "base_dir": str(self.base_dir),
        }


# Global store instance
_global_store: Optional[FeatureStore] = None


def get_global_store() -> FeatureStore:
    """
    Get the global feature store instance.

    Returns:
        Global FeatureStore instance
    """
    global _global_store
    if _global_store is None:
        _global_store = FeatureStore()
    return _global_store
