"""
Clustering algorithms for pattern discovery.

Uses KMeans for initial broad clustering and DBSCAN for density-based refinement.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..common.metrics import calculate_win_rate, calculate_profit_factor, calculate_cluster_stats

logger = logging.getLogger(__name__)


@dataclass
class PatternCluster:
    """Represents a discovered pattern cluster."""
    cluster_id: str
    centroid: np.ndarray
    n_samples: int
    win_rate: float
    profit_factor: float
    avg_return: float
    regime_distribution: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    sample_trades: List[Dict] = field(default_factory=list)
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_validated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cluster_id': self.cluster_id,
            'centroid': self.centroid.tolist() if isinstance(self.centroid, np.ndarray) else self.centroid,
            'n_samples': self.n_samples,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_return': self.avg_return,
            'regime_distribution': self.regime_distribution,
            'feature_importance': self.feature_importance,
            'sample_trades': self.sample_trades[:5],  # Limit samples
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'last_validated': self.last_validated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternCluster':
        """Create from dictionary."""
        return cls(
            cluster_id=data['cluster_id'],
            centroid=np.array(data['centroid']),
            n_samples=data['n_samples'],
            win_rate=data['win_rate'],
            profit_factor=data['profit_factor'],
            avg_return=data.get('avg_return', 0.0),
            regime_distribution=data.get('regime_distribution', {}),
            feature_importance=data.get('feature_importance', {}),
            sample_trades=data.get('sample_trades', []),
            confidence=data.get('confidence', 0.0),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.utcnow(),
            last_validated=datetime.fromisoformat(data['last_validated']) if 'last_validated' in data else datetime.utcnow(),
        )


class PatternClusteringEngine:
    """
    Main clustering engine for pattern discovery.

    Uses KMeans for initial clustering and DBSCAN for density-based refinement.
    Calculates performance metrics for each cluster.
    """

    def __init__(
        self,
        n_clusters: int = 20,
        min_samples_dbscan: int = 10,
        eps: float = 0.5,
        random_state: int = 42,
    ):
        """
        Initialize clustering engine.

        Args:
            n_clusters: Number of KMeans clusters
            min_samples_dbscan: Minimum samples for DBSCAN core points
            eps: DBSCAN epsilon (neighborhood radius)
            random_state: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for clustering. Install with: pip install scikit-learn")

        self.n_clusters = n_clusters
        self.min_samples_dbscan = min_samples_dbscan
        self.eps = eps
        self.random_state = random_state

        self._scaler = StandardScaler()
        self._kmeans: Optional[KMeans] = None
        self._dbscan: Optional[DBSCAN] = None
        self._clusters: List[PatternCluster] = []
        self._feature_names: List[str] = []

    def fit(
        self,
        trades_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        pnl_col: str = 'pnl',
    ) -> List[PatternCluster]:
        """
        Fit clustering models on trade features.

        Args:
            trades_df: DataFrame with trade data and features
            feature_cols: List of feature column names (auto-detect if None)
            pnl_col: Column name for P&L

        Returns:
            List of discovered PatternCluster objects
        """
        if trades_df.empty:
            logger.warning("Empty trades DataFrame, no clusters to discover")
            return []

        # Auto-detect feature columns
        if feature_cols is None:
            exclude_cols = {'timestamp', 'symbol', 'side', 'strategy', 'split', 'won', pnl_col}
            feature_cols = [c for c in trades_df.columns
                           if c not in exclude_cols and trades_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

        if not feature_cols:
            logger.warning("No numeric feature columns found")
            return []

        self._feature_names = feature_cols
        logger.info(f"Using {len(feature_cols)} features for clustering")

        # Extract features
        X = trades_df[feature_cols].fillna(0).values

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Run KMeans
        self._kmeans = KMeans(
            n_clusters=min(self.n_clusters, len(X)),
            random_state=self.random_state,
            n_init=10,
        )
        kmeans_labels = self._kmeans.fit_predict(X_scaled)

        # Analyze clusters
        self._clusters = []
        for cluster_id in range(self._kmeans.n_clusters):
            mask = kmeans_labels == cluster_id
            cluster_trades = trades_df[mask]

            if len(cluster_trades) < 5:
                continue

            # Calculate cluster statistics
            pnl_values = cluster_trades[pnl_col].values if pnl_col in cluster_trades.columns else []
            stats = calculate_cluster_stats(cluster_trades, pnl_col)

            # Calculate feature importance (distance from overall centroid)
            cluster_centroid = X_scaled[mask].mean(axis=0)
            overall_centroid = X_scaled.mean(axis=0)
            importance = np.abs(cluster_centroid - overall_centroid)
            feature_importance = dict(zip(feature_cols, importance.tolist()))

            # Calculate confidence based on sample size and consistency
            confidence = self._calculate_confidence(
                n_samples=len(cluster_trades),
                win_rate=stats['win_rate'],
                profit_factor=stats['profit_factor'],
            )

            # Get regime distribution if available
            regime_dist = {}
            if 'regime' in cluster_trades.columns:
                regime_counts = cluster_trades['regime'].value_counts(normalize=True)
                regime_dist = regime_counts.to_dict()

            # Sample trades for narrative generation
            sample_trades = cluster_trades.head(5).to_dict('records')

            cluster = PatternCluster(
                cluster_id=f"kmeans_{cluster_id:03d}",
                centroid=cluster_centroid,
                n_samples=len(cluster_trades),
                win_rate=stats['win_rate'],
                profit_factor=stats['profit_factor'],
                avg_return=stats['avg_pnl'],
                regime_distribution=regime_dist,
                feature_importance=feature_importance,
                sample_trades=sample_trades,
                confidence=confidence,
            )
            self._clusters.append(cluster)

        logger.info(f"Discovered {len(self._clusters)} pattern clusters")
        return self._clusters

    def _calculate_confidence(
        self,
        n_samples: int,
        win_rate: float,
        profit_factor: float,
    ) -> float:
        """Calculate confidence score for a cluster."""
        # Sample size factor (more samples = higher confidence)
        size_factor = min(1.0, n_samples / 100)

        # Win rate factor (higher = better, but not if too extreme)
        wr_factor = 0.0
        if 0.4 <= win_rate <= 0.8:
            wr_factor = (win_rate - 0.4) / 0.4  # 0 to 1 scale

        # Profit factor bonus
        pf_factor = min(1.0, profit_factor / 2.0) if profit_factor > 0 else 0.0

        # Combined score
        confidence = (size_factor * 0.4 + wr_factor * 0.4 + pf_factor * 0.2)
        return round(confidence, 3)

    def predict_cluster(
        self,
        features: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Predict which cluster a new trade belongs to.

        Args:
            features: Feature vector for the trade

        Returns:
            Tuple of (cluster_id, confidence)
        """
        if self._kmeans is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        features_scaled = self._scaler.transform(features)
        cluster_id = self._kmeans.predict(features_scaled)[0]

        # Calculate confidence based on distance to centroid
        centroid = self._kmeans.cluster_centers_[cluster_id]
        distance = np.linalg.norm(features_scaled[0] - centroid)
        confidence = max(0.0, 1.0 - distance / 5.0)  # Normalize

        return int(cluster_id), float(confidence)

    def get_high_edge_clusters(
        self,
        min_win_rate: float = 0.55,
        min_samples: int = 30,
        min_profit_factor: float = 1.0,
    ) -> List[PatternCluster]:
        """
        Return clusters that meet edge criteria.

        Args:
            min_win_rate: Minimum win rate threshold
            min_samples: Minimum number of samples
            min_profit_factor: Minimum profit factor

        Returns:
            List of high-edge PatternCluster objects
        """
        return [
            c for c in self._clusters
            if c.win_rate >= min_win_rate
            and c.n_samples >= min_samples
            and c.profit_factor >= min_profit_factor
        ]

    def get_all_clusters(self) -> List[PatternCluster]:
        """Get all discovered clusters."""
        return self._clusters

    def save(self, path: str) -> None:
        """Save clustering model and results."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save clusters
        clusters_data = [c.to_dict() for c in self._clusters]
        with open(save_path / 'clusters.json', 'w') as f:
            json.dump(clusters_data, f, indent=2, default=str)

        # Save feature names
        with open(save_path / 'feature_names.json', 'w') as f:
            json.dump(self._feature_names, f)

        logger.info(f"Saved {len(self._clusters)} clusters to {save_path}")

    def load(self, path: str) -> None:
        """Load clustering model and results."""
        load_path = Path(path)

        clusters_file = load_path / 'clusters.json'
        if clusters_file.exists():
            with open(clusters_file) as f:
                clusters_data = json.load(f)
            self._clusters = [PatternCluster.from_dict(c) for c in clusters_data]

        features_file = load_path / 'feature_names.json'
        if features_file.exists():
            with open(features_file) as f:
                self._feature_names = json.load(f)

        logger.info(f"Loaded {len(self._clusters)} clusters from {load_path}")
