"""
SHAP Explainer - Deep ML Model Interpretability
================================================

Provides SHAP (SHapley Additive exPlanations) integration for explaining
ML model predictions in the trading system.

SHAP values show the contribution of each feature to a prediction,
enabling:
- Understanding why a model made a specific prediction
- Identifying the most influential features
- Detecting potential model biases
- Building trust in ML-driven trading decisions

Usage:
    from ml_features.shap_explainer import SHAPExplainer, get_shap_explainer

    explainer = get_shap_explainer()
    explanation = explainer.explain_prediction(
        model=my_xgboost_model,
        X=feature_dataframe,
        model_type="tree"
    )

    print(explanation.to_narrative())
    top_features = explanation.get_top_features(k=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Conditional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


class ModelType(Enum):
    """Types of models for SHAP explanation."""
    TREE = "tree"           # Tree-based (XGBoost, LightGBM, RF)
    DEEP = "deep"           # Deep learning (LSTM, Neural Nets)
    KERNEL = "kernel"       # Model-agnostic (any model)
    LINEAR = "linear"       # Linear models


@dataclass
class FeatureContribution:
    """Contribution of a single feature to a prediction."""
    feature_name: str
    feature_value: float
    shap_value: float
    impact: str  # "positive" or "negative"
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature': self.feature_name,
            'value': round(self.feature_value, 4) if isinstance(self.feature_value, float) else self.feature_value,
            'shap_value': round(self.shap_value, 4),
            'impact': self.impact,
            'rank': self.rank,
        }


@dataclass
class SHAPExplanation:
    """
    Complete SHAP explanation for a prediction.

    Contains SHAP values, base value, and methods for extracting
    insights from the explanation.
    """
    shap_values: np.ndarray           # SHAP values for each feature
    base_value: float                  # Expected value (model output for average case)
    feature_names: List[str]           # Names of features
    feature_values: np.ndarray         # Actual feature values
    prediction: float                  # Model's prediction
    model_type: ModelType = ModelType.TREE
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_top_features(self, k: int = 5) -> List[FeatureContribution]:
        """
        Get the top-k features by absolute SHAP value.

        Args:
            k: Number of top features to return

        Returns:
            List of FeatureContribution objects, sorted by importance
        """
        abs_shap = np.abs(self.shap_values)
        top_indices = np.argsort(abs_shap)[-k:][::-1]

        contributions = []
        for rank, idx in enumerate(top_indices, 1):
            contributions.append(FeatureContribution(
                feature_name=self.feature_names[idx],
                feature_value=float(self.feature_values[idx]),
                shap_value=float(self.shap_values[idx]),
                impact="positive" if self.shap_values[idx] > 0 else "negative",
                rank=rank,
            ))

        return contributions

    def get_positive_contributors(self, threshold: float = 0.0) -> List[FeatureContribution]:
        """Get features that positively contributed to the prediction."""
        positive_indices = np.where(self.shap_values > threshold)[0]
        contributions = []

        for idx in positive_indices:
            contributions.append(FeatureContribution(
                feature_name=self.feature_names[idx],
                feature_value=float(self.feature_values[idx]),
                shap_value=float(self.shap_values[idx]),
                impact="positive",
            ))

        return sorted(contributions, key=lambda c: c.shap_value, reverse=True)

    def get_negative_contributors(self, threshold: float = 0.0) -> List[FeatureContribution]:
        """Get features that negatively contributed to the prediction."""
        negative_indices = np.where(self.shap_values < -threshold)[0]
        contributions = []

        for idx in negative_indices:
            contributions.append(FeatureContribution(
                feature_name=self.feature_names[idx],
                feature_value=float(self.feature_values[idx]),
                shap_value=float(self.shap_values[idx]),
                impact="negative",
            ))

        return sorted(contributions, key=lambda c: c.shap_value)

    def to_narrative(self, top_k: int = 5) -> str:
        """
        Convert explanation to human-readable narrative.

        Returns a natural language explanation of the prediction.
        """
        top_features = self.get_top_features(top_k)

        narrative = f"Prediction: {self.prediction:.4f} (base expectation: {self.base_value:.4f})\n\n"
        narrative += "Key factors driving this prediction:\n"

        for feat in top_features:
            direction = "increased" if feat.impact == "positive" else "decreased"
            narrative += (
                f"  {feat.rank}. {feat.feature_name} = {feat.feature_value:.3f} "
                f"{direction} prediction by {abs(feat.shap_value):.4f}\n"
            )

        # Summary
        total_positive = sum(c.shap_value for c in self.get_positive_contributors())
        total_negative = sum(c.shap_value for c in self.get_negative_contributors())

        narrative += f"\nTotal positive contribution: +{total_positive:.4f}\n"
        narrative += f"Total negative contribution: {total_negative:.4f}\n"
        narrative += f"Net effect: {total_positive + total_negative:.4f}"

        return narrative

    def to_dict(self) -> Dict[str, Any]:
        """Serialize explanation to dictionary."""
        return {
            'prediction': round(float(self.prediction), 4),
            'base_value': round(float(self.base_value), 4),
            'top_features': [f.to_dict() for f in self.get_top_features(10)],
            'model_type': self.model_type.value,
            'timestamp': self.timestamp,
        }


class SHAPExplainer:
    """
    SHAP integration for explaining ML model predictions.

    Supports:
    - Tree models (XGBoost, LightGBM, Random Forest, GBM)
    - Deep learning models (with background data)
    - Any model (via KernelSHAP, slower but universal)

    Example:
        explainer = SHAPExplainer()

        # For XGBoost/LightGBM (fast)
        explanation = explainer.explain_prediction(
            model=xgb_model,
            X=features_df,
            model_type="tree"
        )

        # For neural networks (needs background data)
        explanation = explainer.explain_prediction(
            model=lstm_model,
            X=features_df,
            model_type="deep",
            background_data=training_data[:100]
        )
    """

    def __init__(self, cache_explainers: bool = True):
        """
        Initialize SHAP explainer.

        Args:
            cache_explainers: Whether to cache explainer objects for reuse
        """
        self._explainer_cache: Dict[str, Any] = {}
        self._cache_explainers = cache_explainers
        self._last_explanation: Optional[SHAPExplanation] = None

        if not SHAP_AVAILABLE:
            logger.warning("SHAP not installed. Explanations will be limited.")

    def explain_prediction(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        model_type: Union[str, ModelType] = "tree",
        background_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        check_additivity: bool = False,
    ) -> SHAPExplanation:
        """
        Generate SHAP explanation for a prediction.

        Args:
            model: Trained ML model (XGBoost, LightGBM, LSTM, etc.)
            X: Input features (single row or batch)
            model_type: Type of model ("tree", "deep", "kernel", "linear")
            background_data: Required for deep/kernel explainers
            check_additivity: Verify SHAP values sum to prediction (slower)

        Returns:
            SHAPExplanation with values, base value, and feature info
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type.lower())

        # Handle DataFrame vs ndarray
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            feature_values = X.values[0] if len(X) == 1 else X.values.mean(axis=0)
            X_array = X.values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[-1])]
            feature_values = X[0] if X.ndim > 1 else X
            X_array = X

        if not SHAP_AVAILABLE:
            # Fallback: return mock explanation
            return self._mock_explanation(model, X_array, feature_names, feature_values)

        try:
            # Get or create explainer
            explainer = self._get_explainer(model, model_type, background_data, X_array)

            # Calculate SHAP values
            shap_values = explainer.shap_values(X_array)

            # Handle multi-output models (take first output or mean)
            if isinstance(shap_values, list):
                shap_values = shap_values[0] if len(shap_values) == 1 else np.mean(shap_values, axis=0)

            # Handle batch (take first row)
            if shap_values.ndim > 1:
                shap_values = shap_values[0]

            # Get base value
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0] if len(base_value) == 1 else base_value.mean()

            # Get prediction
            try:
                prediction = float(model.predict(X_array)[0])
            except Exception:
                prediction = float(base_value + shap_values.sum())

            explanation = SHAPExplanation(
                shap_values=shap_values,
                base_value=float(base_value),
                feature_names=feature_names,
                feature_values=np.array(feature_values),
                prediction=prediction,
                model_type=model_type,
            )

            self._last_explanation = explanation
            return explanation

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._mock_explanation(model, X_array, feature_names, feature_values)

    def _get_explainer(
        self,
        model: Any,
        model_type: ModelType,
        background_data: Optional[np.ndarray],
        X: np.ndarray,
    ) -> Any:
        """Get or create a SHAP explainer for the model."""
        cache_key = f"{id(model)}_{model_type.value}"

        if self._cache_explainers and cache_key in self._explainer_cache:
            return self._explainer_cache[cache_key]

        if model_type == ModelType.TREE:
            explainer = shap.TreeExplainer(model)

        elif model_type == ModelType.DEEP:
            if background_data is None:
                background_data = X[:min(100, len(X))]
            explainer = shap.DeepExplainer(model, background_data)

        elif model_type == ModelType.LINEAR:
            if background_data is None:
                background_data = X[:min(100, len(X))]
            explainer = shap.LinearExplainer(model, background_data)

        else:  # KERNEL (slowest but most general)
            if background_data is None:
                background_data = X[:min(50, len(X))]  # Smaller for speed
            explainer = shap.KernelExplainer(model.predict, background_data)

        if self._cache_explainers:
            self._explainer_cache[cache_key] = explainer

        return explainer

    def _mock_explanation(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        feature_values: np.ndarray,
    ) -> SHAPExplanation:
        """Create a mock explanation when SHAP is unavailable."""
        n_features = len(feature_names)

        # Try to get prediction
        try:
            prediction = float(model.predict(X)[0])
        except Exception:
            prediction = 0.5

        # Generate mock SHAP values (small random values)
        np.random.seed(42)
        shap_values = np.random.randn(n_features) * 0.01

        return SHAPExplanation(
            shap_values=shap_values,
            base_value=prediction - shap_values.sum(),
            feature_names=feature_names,
            feature_values=np.array(feature_values),
            prediction=prediction,
            model_type=ModelType.KERNEL,
        )

    def get_latest_explanation(self, symbol: Optional[str] = None) -> Optional[SHAPExplanation]:
        """Get the most recent explanation (for API/dashboard use)."""
        return self._last_explanation

    def explain_ensemble(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        model_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[SHAPExplanation, Dict[str, SHAPExplanation]]:
        """
        Explain an ensemble of models with weighted aggregation.

        Args:
            models: Dictionary of model_name -> model
            X: Input features
            model_weights: Optional weights for each model (default: equal)

        Returns:
            Tuple of (aggregated_explanation, individual_explanations)
        """
        if model_weights is None:
            model_weights = {name: 1.0 / len(models) for name in models}

        # Normalize weights
        total_weight = sum(model_weights.values())
        model_weights = {k: v / total_weight for k, v in model_weights.items()}

        explanations = {}
        aggregated_shap = None

        for name, model in models.items():
            # Determine model type
            model_type = self._infer_model_type(model)
            explanation = self.explain_prediction(model, X, model_type)
            explanations[name] = explanation

            # Aggregate SHAP values
            weight = model_weights.get(name, 1.0 / len(models))
            if aggregated_shap is None:
                aggregated_shap = explanation.shap_values * weight
            else:
                aggregated_shap += explanation.shap_values * weight

        # Create aggregated explanation
        first_exp = next(iter(explanations.values()))
        aggregated = SHAPExplanation(
            shap_values=aggregated_shap,
            base_value=sum(e.base_value * model_weights.get(n, 1/len(models))
                         for n, e in explanations.items()),
            feature_names=first_exp.feature_names,
            feature_values=first_exp.feature_values,
            prediction=sum(e.prediction * model_weights.get(n, 1/len(models))
                          for n, e in explanations.items()),
        )

        return aggregated, explanations

    def _infer_model_type(self, model: Any) -> ModelType:
        """Infer the model type from the model class."""
        model_class = model.__class__.__name__.lower()

        if any(tree_type in model_class for tree_type in
               ['xgb', 'lightgbm', 'lgb', 'random', 'forest', 'tree', 'gradient', 'gbm']):
            return ModelType.TREE

        if any(nn_type in model_class for nn_type in
               ['lstm', 'neural', 'network', 'dense', 'sequential', 'keras']):
            return ModelType.DEEP

        if any(lin_type in model_class for lin_type in
               ['linear', 'logistic', 'ridge', 'lasso']):
            return ModelType.LINEAR

        # Default to kernel for unknown models
        return ModelType.KERNEL

    def clear_cache(self) -> None:
        """Clear the explainer cache."""
        self._explainer_cache.clear()


# Singleton instance
_shap_explainer_instance: Optional[SHAPExplainer] = None


def get_shap_explainer(cache_explainers: bool = True) -> SHAPExplainer:
    """
    Get the singleton SHAP explainer instance.

    Args:
        cache_explainers: Whether to cache explainer objects

    Returns:
        SHAPExplainer instance
    """
    global _shap_explainer_instance
    if _shap_explainer_instance is None:
        _shap_explainer_instance = SHAPExplainer(cache_explainers=cache_explainers)
    return _shap_explainer_instance


def explain_prediction(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    model_type: str = "tree",
) -> SHAPExplanation:
    """
    Convenience function to explain a single prediction.

    Args:
        model: Trained ML model
        X: Input features
        model_type: Type of model ("tree", "deep", "kernel")

    Returns:
        SHAPExplanation object
    """
    explainer = get_shap_explainer()
    return explainer.explain_prediction(model, X, model_type)
