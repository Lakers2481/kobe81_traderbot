"""
Feature Discovery for Trading Systems
=======================================

Automatically discovers potentially useful features from
market data through statistical analysis and pattern detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureTransform(Enum):
    """Types of feature transformations."""
    RETURNS = "returns"           # Price returns
    LOG_RETURNS = "log_returns"   # Log returns
    MOMENTUM = "momentum"         # N-period momentum
    VOLATILITY = "volatility"     # Rolling volatility
    ZSCORE = "zscore"             # Z-score normalization
    PERCENTILE = "percentile"     # Rolling percentile
    RATIO = "ratio"               # Ratio of two series
    DIFF = "diff"                 # Difference from MA


@dataclass
class DiscoveredFeature:
    """A discovered feature with its properties."""
    name: str
    transform: FeatureTransform
    base_column: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Quality metrics
    correlation_with_target: float = 0.0
    information_ratio: float = 0.0
    stability_score: float = 0.0

    # Metadata
    discovered_at: datetime = field(default_factory=datetime.now)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'transform': self.transform.value,
            'base_column': self.base_column,
            'parameters': self.parameters,
            'correlation_with_target': self.correlation_with_target,
            'information_ratio': self.information_ratio,
            'stability_score': self.stability_score,
            'description': self.description,
        }

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Apply the feature transformation to data."""
        col = self.base_column

        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")

        series = df[col]
        params = self.parameters

        if self.transform == FeatureTransform.RETURNS:
            period = params.get('period', 1)
            return series.pct_change(period)

        elif self.transform == FeatureTransform.LOG_RETURNS:
            period = params.get('period', 1)
            return np.log(series / series.shift(period))

        elif self.transform == FeatureTransform.MOMENTUM:
            period = params.get('period', 10)
            return series - series.shift(period)

        elif self.transform == FeatureTransform.VOLATILITY:
            window = params.get('window', 20)
            return series.pct_change().rolling(window).std()

        elif self.transform == FeatureTransform.ZSCORE:
            window = params.get('window', 20)
            roll_mean = series.rolling(window).mean()
            roll_std = series.rolling(window).std()
            return (series - roll_mean) / roll_std

        elif self.transform == FeatureTransform.PERCENTILE:
            window = params.get('window', 20)
            return series.rolling(window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                raw=False
            )

        elif self.transform == FeatureTransform.RATIO:
            other_col = params.get('other_column')
            if other_col and other_col in df.columns:
                return series / df[other_col]
            return series / series.rolling(params.get('window', 20)).mean()

        elif self.transform == FeatureTransform.DIFF:
            window = params.get('window', 20)
            return series - series.rolling(window).mean()

        return series


class FeatureDiscovery:
    """
    Discovers useful features from market data.

    Automatically generates and evaluates potential features
    based on price, volume, and derived indicators.
    """

    # Standard feature templates
    TEMPLATES = [
        (FeatureTransform.RETURNS, {'period': 1}),
        (FeatureTransform.RETURNS, {'period': 5}),
        (FeatureTransform.RETURNS, {'period': 20}),
        (FeatureTransform.VOLATILITY, {'window': 10}),
        (FeatureTransform.VOLATILITY, {'window': 20}),
        (FeatureTransform.ZSCORE, {'window': 20}),
        (FeatureTransform.ZSCORE, {'window': 50}),
        (FeatureTransform.PERCENTILE, {'window': 20}),
        (FeatureTransform.MOMENTUM, {'period': 5}),
        (FeatureTransform.MOMENTUM, {'period': 10}),
    ]

    def __init__(
        self,
        min_correlation: float = 0.05,
        min_stability: float = 0.3,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize feature discovery.

        Args:
            min_correlation: Minimum correlation to keep feature
            min_stability: Minimum stability score to keep
            random_seed: Random seed for reproducibility
        """
        self.min_correlation = min_correlation
        self.min_stability = min_stability

        if random_seed is not None:
            np.random.seed(random_seed)

        logger.info(
            f"FeatureDiscovery initialized with "
            f"min_corr={min_correlation}, min_stability={min_stability}"
        )

    def _calculate_correlation(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> float:
        """Calculate correlation between feature and target."""
        valid_mask = feature.notna() & target.notna()
        if valid_mask.sum() < 30:
            return 0.0

        corr = feature.loc[valid_mask].corr(target.loc[valid_mask])
        return corr if not np.isnan(corr) else 0.0

    def _calculate_stability(
        self,
        feature: pd.Series,
        target: pd.Series,
        n_splits: int = 5,
    ) -> float:
        """Calculate stability of feature across time splits."""
        valid_mask = feature.notna() & target.notna()
        n_valid = valid_mask.sum()

        if n_valid < n_splits * 10:
            return 0.0

        # Split into time periods
        split_size = n_valid // n_splits
        correlations = []

        valid_indices = feature.loc[valid_mask].index

        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size
            split_indices = valid_indices[start_idx:end_idx]

            if len(split_indices) < 10:
                continue

            split_corr = feature.loc[split_indices].corr(target.loc[split_indices])
            if not np.isnan(split_corr):
                correlations.append(split_corr)

        if len(correlations) < 2:
            return 0.0

        # Stability = consistency of correlation across splits
        # High stability = all correlations have same sign and similar magnitude
        correlations = np.array(correlations)
        same_sign = np.all(correlations > 0) or np.all(correlations < 0)
        low_variance = np.std(correlations) < 0.3

        if same_sign and low_variance:
            return 1.0 - np.std(correlations) / np.mean(np.abs(correlations))

        return 0.0

    def _calculate_information_ratio(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> float:
        """Calculate information ratio (correlation / std of correlation)."""
        corr = self._calculate_correlation(feature, target)
        if abs(corr) < 0.01:
            return 0.0

        # Rolling correlation to estimate variance
        valid_mask = feature.notna() & target.notna()
        if valid_mask.sum() < 60:
            return abs(corr)

        rolling_corr = (
            feature.loc[valid_mask]
            .rolling(30)
            .corr(target.loc[valid_mask])
        )
        std_corr = rolling_corr.std()

        if std_corr > 0:
            return abs(corr) / std_corr
        return abs(corr)

    def discover_from_columns(
        self,
        df: pd.DataFrame,
        target_column: str,
        base_columns: Optional[List[str]] = None,
    ) -> List[DiscoveredFeature]:
        """
        Discover features from DataFrame columns.

        Args:
            df: DataFrame with price/volume data
            target_column: Target to predict (e.g., 'forward_return')
            base_columns: Columns to derive features from

        Returns:
            List of discovered features
        """
        if target_column not in df.columns:
            logger.warning(f"Target column {target_column} not found")
            return []

        target = df[target_column]

        if base_columns is None:
            base_columns = [
                c for c in df.columns
                if c != target_column and df[c].dtype in [np.float64, np.int64]
            ]

        discovered = []

        for col in base_columns:
            for transform, params in self.TEMPLATES:
                try:
                    # Create feature
                    feature = DiscoveredFeature(
                        name=f"{col}_{transform.value}_{list(params.values())[0]}",
                        transform=transform,
                        base_column=col,
                        parameters=params.copy(),
                    )

                    # Apply transformation
                    feature_values = feature.apply(df)

                    # Calculate quality metrics
                    feature.correlation_with_target = self._calculate_correlation(
                        feature_values, target
                    )
                    feature.stability_score = self._calculate_stability(
                        feature_values, target
                    )
                    feature.information_ratio = self._calculate_information_ratio(
                        feature_values, target
                    )

                    # Filter by quality
                    if (abs(feature.correlation_with_target) >= self.min_correlation and
                        feature.stability_score >= self.min_stability):
                        feature.description = (
                            f"{transform.value.capitalize()} of {col} "
                            f"with params {params}"
                        )
                        discovered.append(feature)

                except Exception as e:
                    logger.debug(f"Failed to create {col} {transform.value}: {e}")
                    continue

        # Sort by absolute correlation
        discovered.sort(key=lambda x: abs(x.correlation_with_target), reverse=True)

        logger.info(f"Discovered {len(discovered)} features from {len(base_columns)} columns")

        return discovered

    def suggest_combinations(
        self,
        discovered: List[DiscoveredFeature],
        max_combinations: int = 10,
    ) -> List[DiscoveredFeature]:
        """Suggest feature combinations."""
        suggestions = []

        # Get top features by correlation
        top_features = sorted(
            discovered,
            key=lambda x: abs(x.correlation_with_target),
            reverse=True
        )[:5]

        # Suggest ratios between correlated features
        for i, f1 in enumerate(top_features):
            for f2 in top_features[i+1:]:
                if len(suggestions) >= max_combinations:
                    break

                # Same base column, different transforms
                if f1.base_column == f2.base_column:
                    continue

                suggestion = DiscoveredFeature(
                    name=f"{f1.name}_ratio_{f2.name}",
                    transform=FeatureTransform.RATIO,
                    base_column=f1.base_column,
                    parameters={
                        'other_column': f2.base_column,
                        'feature1': f1.to_dict(),
                        'feature2': f2.to_dict(),
                    },
                    description=f"Ratio of {f1.name} to {f2.name}",
                )
                suggestions.append(suggestion)

        return suggestions


def discover_features(
    df: pd.DataFrame,
    target_column: str,
) -> List[DiscoveredFeature]:
    """Convenience function to discover features."""
    discovery = FeatureDiscovery()
    return discovery.discover_from_columns(df, target_column)


def suggest_features(
    df: pd.DataFrame,
    target_column: str,
    n: int = 10,
) -> List[str]:
    """Get suggested feature names."""
    discovered = discover_features(df, target_column)
    return [f.name for f in discovered[:n]]


# Module-level discovery instance
_discovery: Optional[FeatureDiscovery] = None


def get_discovery() -> FeatureDiscovery:
    """Get or create the global discovery instance."""
    global _discovery
    if _discovery is None:
        _discovery = FeatureDiscovery()
    return _discovery
