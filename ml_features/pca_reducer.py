"""
PCA Dimensionality Reduction Module.

Provides Principal Component Analysis (PCA) for reducing the 150+ features
to a smaller set of uncorrelated components. This helps:
- Prevent overfitting in ML models
- Speed up training time
- Remove multicollinearity between features

Inspired by: MML Book (mml-book.github.io) mathematical foundations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.structured_log import jlog


@dataclass
class PCAConfig:
    """Configuration for PCA dimensionality reduction."""
    # Variance threshold - keep components explaining this much variance
    variance_threshold: float = 0.95  # 95% variance retained

    # Maximum number of components (None = auto based on variance)
    max_components: Optional[int] = None

    # Minimum number of components to keep
    min_components: int = 5

    # Scale features before PCA (recommended)
    scale_first: bool = True

    # Column prefix for PCA features
    feature_prefix: str = "pca_"


class PCAReducer:
    """
    Principal Component Analysis for feature dimensionality reduction.

    Reduces 150+ technical indicators to a smaller set of uncorrelated
    components while retaining most of the information (variance).

    Example:
        >>> reducer = PCAReducer(PCAConfig(variance_threshold=0.95))
        >>> df_reduced = reducer.fit_transform(df_features)
        >>> print(f"Reduced from {df_features.shape[1]} to {df_reduced.shape[1]} features")
    """

    def __init__(self, config: Optional[PCAConfig] = None):
        self.config = config or PCAConfig()
        self._pca: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._feature_cols: List[str] = []
        self._is_fitted: bool = False
        self._explained_variance_ratio: Optional[np.ndarray] = None
        self._n_components: int = 0

    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "PCAReducer":
        """
        Fit PCA on the feature data.

        Args:
            df: DataFrame with features
            feature_cols: Optional list of feature columns (auto-detect if None)

        Returns:
            self for method chaining
        """
        if not SKLEARN_AVAILABLE:
            jlog("pca_sklearn_missing", level="WARNING",
                 message="scikit-learn not installed, PCA disabled")
            return self

        # Auto-detect feature columns
        if feature_cols is None:
            exclude = {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'date'}
            feature_cols = [c for c in df.columns if c not in exclude
                          and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

        self._feature_cols = feature_cols

        if len(feature_cols) < self.config.min_components:
            jlog("pca_too_few_features", level="WARNING",
                 n_features=len(feature_cols),
                 min_required=self.config.min_components)
            return self

        # Extract feature matrix
        X = df[feature_cols].fillna(0).values

        if len(X) < 10:
            jlog("pca_too_few_samples", level="WARNING", n_samples=len(X))
            return self

        try:
            # Scale features first (recommended for PCA)
            if self.config.scale_first:
                self._scaler = StandardScaler()
                X = self._scaler.fit_transform(X)

            # Determine number of components
            if self.config.max_components:
                n_components = min(self.config.max_components, len(feature_cols), len(X))
            else:
                # Use variance threshold to determine components
                n_components = min(len(feature_cols), len(X))

            # Fit PCA
            self._pca = PCA(n_components=n_components)
            self._pca.fit(X)

            # Determine final components based on variance threshold
            cumulative_variance = np.cumsum(self._pca.explained_variance_ratio_)
            n_keep = np.searchsorted(cumulative_variance, self.config.variance_threshold) + 1
            n_keep = max(n_keep, self.config.min_components)
            n_keep = min(n_keep, n_components)

            self._n_components = n_keep
            self._explained_variance_ratio = self._pca.explained_variance_ratio_[:n_keep]
            self._is_fitted = True

            jlog("pca_fitted", level="INFO",
                 original_features=len(feature_cols),
                 pca_components=n_keep,
                 variance_retained=float(cumulative_variance[n_keep-1]))

        except Exception as e:
            jlog("pca_fit_error", level="ERROR", error=str(e))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted PCA.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with PCA features added
        """
        if not self._is_fitted or self._pca is None:
            return df

        df = df.copy()

        # Extract feature matrix
        X = df[self._feature_cols].fillna(0).values

        try:
            # Scale if configured
            if self.config.scale_first and self._scaler is not None:
                X = self._scaler.transform(X)

            # Transform with PCA
            X_pca = self._pca.transform(X)[:, :self._n_components]

            # Add PCA columns to dataframe
            for i in range(self._n_components):
                col_name = f"{self.config.feature_prefix}{i+1}"
                df[col_name] = X_pca[:, i]

            jlog("pca_transformed", level="DEBUG",
                 n_samples=len(df),
                 n_components=self._n_components)

        except Exception as e:
            jlog("pca_transform_error", level="ERROR", error=str(e))

        return df

    def fit_transform(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit PCA and transform in one step.

        Args:
            df: DataFrame with features
            feature_cols: Optional list of feature columns

        Returns:
            DataFrame with PCA features added
        """
        self.fit(df, feature_cols)
        return self.transform(df)

    def get_component_loadings(self) -> Optional[pd.DataFrame]:
        """
        Get the loadings (weights) of original features on each component.

        Returns:
            DataFrame with shape (n_features, n_components)
        """
        if not self._is_fitted or self._pca is None:
            return None

        loadings = self._pca.components_[:self._n_components].T
        return pd.DataFrame(
            loadings,
            index=self._feature_cols,
            columns=[f"{self.config.feature_prefix}{i+1}" for i in range(self._n_components)]
        )

    def get_explained_variance(self) -> Dict[str, Any]:
        """
        Get explained variance information.

        Returns:
            Dict with variance info per component
        """
        if not self._is_fitted or self._explained_variance_ratio is None:
            return {}

        return {
            "n_components": self._n_components,
            "total_variance_retained": float(self._explained_variance_ratio.sum()),
            "per_component": [
                {
                    "component": i+1,
                    "variance_ratio": float(v),
                    "cumulative": float(self._explained_variance_ratio[:i+1].sum())
                }
                for i, v in enumerate(self._explained_variance_ratio)
            ]
        }

    def get_top_features_per_component(self, n_top: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top contributing features for each PCA component.

        Args:
            n_top: Number of top features per component

        Returns:
            Dict mapping component name to list of (feature, loading) tuples
        """
        loadings = self.get_component_loadings()
        if loadings is None:
            return {}

        result = {}
        for col in loadings.columns:
            abs_loadings = loadings[col].abs().sort_values(ascending=False)
            top_features = [
                (feat, float(loadings.loc[feat, col]))
                for feat in abs_loadings.head(n_top).index
            ]
            result[col] = top_features

        return result

    @property
    def is_fitted(self) -> bool:
        """Check if PCA is fitted."""
        return self._is_fitted

    @property
    def n_components(self) -> int:
        """Get number of PCA components."""
        return self._n_components

    @property
    def original_features(self) -> List[str]:
        """Get list of original feature columns."""
        return self._feature_cols.copy()


# Convenience function
def reduce_features_pca(
    df: pd.DataFrame,
    variance_threshold: float = 0.95,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, PCAReducer]:
    """
    Reduce features using PCA (convenience function).

    Args:
        df: DataFrame with features
        variance_threshold: Variance to retain (default 95%)
        feature_cols: Optional feature columns

    Returns:
        Tuple of (transformed DataFrame, fitted PCAReducer)
    """
    config = PCAConfig(variance_threshold=variance_threshold)
    reducer = PCAReducer(config)
    df_reduced = reducer.fit_transform(df, feature_cols)
    return df_reduced, reducer
