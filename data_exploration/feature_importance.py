"""
Feature Importance Analysis
============================

Analyzes which features are most important for predictions.
Supports multiple methods including permutation importance,
correlation analysis, and tree-based importance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ImportanceMethod(Enum):
    """Method for calculating feature importance."""
    CORRELATION = "correlation"
    PERMUTATION = "permutation"
    MUTUAL_INFO = "mutual_info"
    TREE_BASED = "tree_based"


@dataclass
class ImportanceResult:
    """Result of feature importance analysis."""
    feature_name: str
    importance: float
    rank: int
    method: ImportanceMethod
    std_error: Optional[float] = None
    p_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feature_name': self.feature_name,
            'importance': self.importance,
            'rank': self.rank,
            'method': self.method.value,
            'std_error': self.std_error,
            'p_value': self.p_value,
        }


class FeatureImportance:
    """
    Analyzes feature importance for trading models.

    Supports multiple methods for robustness and can
    generate importance rankings for feature selection.
    """

    def __init__(
        self,
        default_method: ImportanceMethod = ImportanceMethod.CORRELATION,
        n_permutations: int = 100,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the feature importance analyzer.

        Args:
            default_method: Default importance calculation method
            n_permutations: Number of permutations for permutation importance
            random_seed: Random seed for reproducibility
        """
        self.default_method = default_method
        self.n_permutations = n_permutations
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        logger.info(f"FeatureImportance initialized with method={default_method.value}")

    def _correlation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> List[Tuple[str, float]]:
        """Calculate importance using correlation."""
        importance = []

        for col in X.columns:
            # Handle missing values
            valid_mask = X[col].notna() & y.notna()
            if valid_mask.sum() < 10:
                importance.append((col, 0.0))
                continue

            corr = X.loc[valid_mask, col].corr(y.loc[valid_mask])
            importance.append((col, abs(corr) if not np.isnan(corr) else 0.0))

        return importance

    def _permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Optional[Any] = None,
    ) -> List[Tuple[str, float, float]]:
        """Calculate permutation importance."""
        # If no model provided, use correlation as base metric
        if model is None:
            base_scores = {}
            for col in X.columns:
                valid_mask = X[col].notna() & y.notna()
                if valid_mask.sum() < 10:
                    base_scores[col] = 0.0
                else:
                    base_scores[col] = abs(X.loc[valid_mask, col].corr(y.loc[valid_mask]))

            importance = []
            for col in X.columns:
                # Permute the feature and measure score change
                scores = []
                for _ in range(min(self.n_permutations, 10)):
                    X_perm = X.copy()
                    X_perm[col] = np.random.permutation(X_perm[col].values)

                    valid_mask = X_perm[col].notna() & y.notna()
                    if valid_mask.sum() < 10:
                        continue

                    perm_corr = abs(X_perm.loc[valid_mask, col].corr(y.loc[valid_mask]))
                    score_drop = base_scores.get(col, 0) - (perm_corr if not np.isnan(perm_corr) else 0)
                    scores.append(score_drop)

                if scores:
                    mean_drop = np.mean(scores)
                    std_drop = np.std(scores)
                else:
                    mean_drop = 0.0
                    std_drop = 0.0

                importance.append((col, mean_drop, std_drop))

            return importance

        # With model, use actual predictions
        try:
            y_pred = model.predict(X)
            base_score = np.corrcoef(y, y_pred)[0, 1]
        except Exception:
            base_score = 0.0

        importance = []
        for col in X.columns:
            scores = []
            for _ in range(self.n_permutations):
                X_perm = X.copy()
                X_perm[col] = np.random.permutation(X_perm[col].values)

                try:
                    y_pred_perm = model.predict(X_perm)
                    perm_score = np.corrcoef(y, y_pred_perm)[0, 1]
                    scores.append(base_score - perm_score)
                except Exception:
                    continue

            if scores:
                importance.append((col, np.mean(scores), np.std(scores)))
            else:
                importance.append((col, 0.0, 0.0))

        return importance

    def _mutual_info_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> List[Tuple[str, float]]:
        """Calculate mutual information importance."""
        importance = []

        for col in X.columns:
            valid_mask = X[col].notna() & y.notna()
            if valid_mask.sum() < 10:
                importance.append((col, 0.0))
                continue

            x_vals = X.loc[valid_mask, col].values
            y_vals = y.loc[valid_mask].values

            # Discretize continuous variables for MI calculation
            try:
                x_bins = pd.qcut(x_vals, q=10, duplicates='drop').codes
                y_bins = pd.qcut(y_vals, q=10, duplicates='drop').codes

                # Calculate entropy and mutual information
                n = len(x_bins)
                p_x = np.bincount(x_bins) / n
                p_y = np.bincount(y_bins) / n

                # Joint probability
                joint = np.zeros((len(np.unique(x_bins)), len(np.unique(y_bins))))
                for i, (xi, yi) in enumerate(zip(x_bins, y_bins)):
                    joint[xi, yi] += 1
                joint /= n

                # Mutual information
                mi = 0.0
                for i in range(joint.shape[0]):
                    for j in range(joint.shape[1]):
                        if joint[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                            mi += joint[i, j] * np.log2(joint[i, j] / (p_x[i] * p_y[j]))

                importance.append((col, mi))

            except Exception:
                importance.append((col, 0.0))

        return importance

    def analyze(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: Optional[ImportanceMethod] = None,
        model: Optional[Any] = None,
    ) -> List[ImportanceResult]:
        """
        Analyze feature importance.

        Args:
            X: Feature matrix
            y: Target variable
            method: Importance calculation method
            model: Optional trained model for permutation importance

        Returns:
            List of ImportanceResult sorted by importance
        """
        method = method or self.default_method

        if method == ImportanceMethod.CORRELATION:
            raw_importance = self._correlation_importance(X, y)
            results = [
                ImportanceResult(
                    feature_name=name,
                    importance=imp,
                    rank=0,
                    method=method,
                )
                for name, imp in raw_importance
            ]

        elif method == ImportanceMethod.PERMUTATION:
            raw_importance = self._permutation_importance(X, y, model)
            results = [
                ImportanceResult(
                    feature_name=name,
                    importance=imp,
                    rank=0,
                    method=method,
                    std_error=std,
                )
                for name, imp, std in raw_importance
            ]

        elif method == ImportanceMethod.MUTUAL_INFO:
            raw_importance = self._mutual_info_importance(X, y)
            results = [
                ImportanceResult(
                    feature_name=name,
                    importance=imp,
                    rank=0,
                    method=method,
                )
                for name, imp in raw_importance
            ]

        else:
            raise ValueError(f"Unsupported method: {method}")

        # Sort by importance and assign ranks
        results.sort(key=lambda x: x.importance, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1

        logger.info(
            f"Analyzed {len(results)} features using {method.value}. "
            f"Top feature: {results[0].feature_name if results else 'N/A'}"
        )

        return results

    def get_top_n(
        self,
        results: List[ImportanceResult],
        n: int = 10,
    ) -> List[ImportanceResult]:
        """Get top N most important features."""
        return sorted(results, key=lambda x: x.importance, reverse=True)[:n]

    def compare_methods(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, List[ImportanceResult]]:
        """Compare importance across different methods."""
        comparison = {}

        for method in [ImportanceMethod.CORRELATION, ImportanceMethod.MUTUAL_INFO]:
            try:
                results = self.analyze(X, y, method)
                comparison[method.value] = results
            except Exception as e:
                logger.warning(f"Failed to calculate {method.value}: {e}")

        return comparison

    def to_dataframe(
        self,
        results: List[ImportanceResult],
    ) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([r.to_dict() for r in results])


def analyze_importance(
    X: pd.DataFrame,
    y: pd.Series,
    method: ImportanceMethod = ImportanceMethod.CORRELATION,
) -> List[ImportanceResult]:
    """Convenience function to analyze feature importance."""
    analyzer = FeatureImportance(default_method=method)
    return analyzer.analyze(X, y)


def get_top_features(
    X: pd.DataFrame,
    y: pd.Series,
    n: int = 10,
) -> List[str]:
    """Get names of top N features by importance."""
    results = analyze_importance(X, y)
    return [r.feature_name for r in results[:n]]


# Module-level analyzer
_analyzer: Optional[FeatureImportance] = None


def get_analyzer() -> FeatureImportance:
    """Get or create the global analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = FeatureImportance()
    return _analyzer
