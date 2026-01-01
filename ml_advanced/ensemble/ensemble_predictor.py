"""
Ensemble Prediction System
===========================

Multi-model ensemble predictor that combines predictions from LSTM,
XGBoost, and LightGBM models using weighted voting and confidence scoring.

The ensemble calculates prediction confidence based on model agreement:
- High agreement (low std) = high confidence
- Low agreement (high std) = low confidence

MERGED FROM GAME_PLAN_2K28 - Production Ready
"""

import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from itertools import product

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Result from ensemble prediction."""
    prediction: float
    confidence: float
    model_predictions: Dict[str, float]
    model_weights: Dict[str, float]
    std_dev: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prediction": round(self.prediction, 4),
            "confidence": round(self.confidence, 4),
            "model_predictions": {k: round(v, 4) for k, v in self.model_predictions.items()},
            "model_weights": {k: round(v, 4) for k, v in self.model_weights.items()},
            "std_dev": round(self.std_dev, 4),
            "timestamp": self.timestamp.isoformat()
        }


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers."""

    def __init__(self, name: str):
        self.name = name
        self.model = None

    @abstractmethod
    def load(self, model_path: Path) -> None:
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> float:
        pass

    def is_loaded(self) -> bool:
        return self.model is not None


class XGBoostWrapper(BaseModelWrapper):
    """Wrapper for XGBoost model."""

    # Feature names expected by the model (from training)
    FEATURE_NAMES = ['atr14', 'sma20_over_200', 'rv20', 'don20_width',
                     'pos_in_don20', 'ret5', 'log_vol']

    def __init__(self, name: str = "xgboost", task_type: str = "classification"):
        super().__init__(name)
        self.task_type = task_type
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. pip install xgboost")

    def load(self, model_path: Path) -> None:
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))
        logger.info(f"XGBoost model '{self.name}' loaded from {model_path}")

    def predict(self, features: np.ndarray) -> float:
        if not self.is_loaded():
            raise ValueError(f"Model '{self.name}' not loaded")

        if features.ndim == 1:
            features = features.reshape(1, -1)
        elif features.ndim == 3:
            features = features.reshape(features.shape[0], -1)

        # Create DMatrix with feature names to match trained model
        dmatrix = xgb.DMatrix(features, feature_names=self.FEATURE_NAMES)
        prediction = self.model.predict(dmatrix)[0]

        if self.task_type == "classification":
            prediction = float(np.clip(prediction, 0, 1))
        return prediction


class LightGBMWrapper(BaseModelWrapper):
    """Wrapper for LightGBM model."""

    def __init__(self, name: str = "lightgbm", task_type: str = "classification"):
        super().__init__(name)
        self.task_type = task_type
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. pip install lightgbm")

    def load(self, model_path: Path) -> None:
        self.model = lgb.Booster(model_file=str(model_path))
        logger.info(f"LightGBM model '{self.name}' loaded from {model_path}")

    def predict(self, features: np.ndarray) -> float:
        if not self.is_loaded():
            raise ValueError(f"Model '{self.name}' not loaded")

        if features.ndim == 1:
            features = features.reshape(1, -1)
        elif features.ndim == 3:
            features = features.reshape(features.shape[0], -1)

        prediction = self.model.predict(features)[0]

        if self.task_type == "classification":
            prediction = float(np.clip(prediction, 0, 1))
        return prediction


class EnsemblePredictor:
    """
    Multi-model ensemble predictor with weighted voting.

    Combines predictions from multiple models using configurable weights.
    Calculates confidence based on model agreement.
    """

    def __init__(self, default_weights: Optional[Dict[str, float]] = None):
        self.models: Dict[str, BaseModelWrapper] = {}
        self._raw_weights: Dict[str, float] = {}

        if default_weights is None:
            self.default_weights = {"lstm": 0.4, "xgboost": 0.3, "lightgbm": 0.3}
        else:
            self.default_weights = default_weights

        self.prediction_history: List[EnsemblePrediction] = []
        logger.info("EnsemblePredictor initialized")

    @property
    def weights(self) -> Dict[str, float]:
        return self._get_normalized_weights()

    def add_model(self, name: str, model: BaseModelWrapper, weight: float = 1.0) -> None:
        """Register a model with the ensemble."""
        if name in self.models:
            logger.warning(f"Model '{name}' already exists, replacing")

        self.models[name] = model
        self._raw_weights[name] = weight

        normalized_weights = self._get_normalized_weights()
        logger.info(f"Model '{name}' added with weight {normalized_weights[name]:.3f}")

    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found")
        del self.models[name]
        del self._raw_weights[name]
        logger.info(f"Model '{name}' removed")

    def _get_normalized_weights(self) -> Dict[str, float]:
        """Get normalized weights that sum to 1.0."""
        total = sum(self._raw_weights.values())
        if total > 0:
            return {k: v / total for k, v in self._raw_weights.items()}
        return {}

    def predict(self, features: np.ndarray) -> float:
        """Get ensemble prediction using weighted averaging."""
        if not self.models:
            raise ValueError("No models loaded in ensemble")

        predictions = []
        weights = []

        for name, model in self.models.items():
            if model.is_loaded():
                pred = model.predict(features)
                predictions.append(pred)
                weights.append(self.weights[name])

        if not predictions:
            raise ValueError("No loaded models available for prediction")

        weights_array = np.array(weights)
        weights_array = weights_array / weights_array.sum()

        return float(np.average(predictions, weights=weights_array))

    def predict_with_confidence(self, features: np.ndarray) -> EnsemblePrediction:
        """Get ensemble prediction with confidence score."""
        if not self.models:
            raise ValueError("No models loaded in ensemble")

        model_predictions = {}
        predictions = []
        active_weights = {}

        for name, model in self.models.items():
            if model.is_loaded():
                try:
                    pred = model.predict(features)
                    model_predictions[name] = pred
                    predictions.append(pred)
                    active_weights[name] = self.weights[name]
                except Exception as e:
                    logger.warning(f"Model '{name}' prediction failed: {e}")

        if not predictions:
            raise ValueError("No loaded models available for prediction")

        total_weight = sum(active_weights.values())
        active_weights = {k: v / total_weight for k, v in active_weights.items()}

        weights_array = np.array([active_weights[name] for name in model_predictions.keys()])
        preds_array = np.array(list(model_predictions.values()))

        ensemble_prediction = np.average(preds_array, weights=weights_array)
        std_dev = float(np.std(preds_array))
        confidence = 1.0 / (1.0 + std_dev)

        result = EnsemblePrediction(
            prediction=float(ensemble_prediction),
            confidence=confidence,
            model_predictions=model_predictions,
            model_weights=active_weights,
            std_dev=std_dev,
            timestamp=datetime.now()
        )

        self.prediction_history.append(result)
        return result

    def optimize_weights(
        self,
        validation_data: List[Tuple[np.ndarray, float]],
        metric: str = "mse"
    ) -> Dict[str, float]:
        """Optimize model weights using grid search."""
        if not self.models:
            raise ValueError("No models loaded")

        logger.info(f"Optimizing weights on {len(validation_data)} samples using {metric}")

        model_names = [name for name, m in self.models.items() if m.is_loaded()]
        n_models = len(model_names)

        if n_models == 0:
            raise ValueError("No loaded models available")

        model_preds = {name: [] for name in model_names}
        targets = []

        for features, target in validation_data:
            targets.append(target)
            for name in model_names:
                try:
                    pred = self.models[name].predict(features)
                    model_preds[name].append(pred)
                except Exception:
                    model_preds[name].append(0.5)

        targets = np.array(targets)
        model_preds = {k: np.array(v) for k, v in model_preds.items()}

        best_weights = None
        best_score = float('inf') if metric in ['mse', 'mae'] else 0

        if n_models <= 3:
            weight_options = [i/10 for i in range(11)]

            for weights_tuple in product(weight_options, repeat=n_models):
                if abs(sum(weights_tuple) - 1.0) > 0.01:
                    continue

                weights_dict = dict(zip(model_names, weights_tuple))
                ensemble_preds = sum(model_preds[name] * weight for name, weight in weights_dict.items())

                if metric == "mse":
                    score = np.mean((ensemble_preds - targets) ** 2)
                    is_better = score < best_score
                elif metric == "mae":
                    score = np.mean(np.abs(ensemble_preds - targets))
                    is_better = score < best_score
                elif metric == "accuracy":
                    score = np.mean((ensemble_preds > 0.5) == (targets > 0.5))
                    is_better = score > best_score
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                if is_better:
                    best_score = score
                    best_weights = weights_dict.copy()
        else:
            best_weights = {name: 1.0 / n_models for name in model_names}

        if best_weights is None:
            best_weights = {name: 1.0 / n_models for name in model_names}

        self._raw_weights.update(best_weights)
        logger.info(f"Optimized weights: {self._get_normalized_weights()}")
        return self._get_normalized_weights().copy()

    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about ensemble models."""
        return {
            "total_models": len(self.models),
            "loaded_models": sum(1 for m in self.models.values() if m.is_loaded()),
            "weights": self.weights.copy(),
            "total_predictions": len(self.prediction_history)
        }

    def calculate_average_confidence(self) -> float:
        """Calculate average confidence across all predictions."""
        if not self.prediction_history:
            return 0.0
        return float(np.mean([p.confidence for p in self.prediction_history]))

    def export_predictions(self, output_path: Path) -> None:
        """Export prediction history to CSV."""
        if not self.prediction_history:
            logger.warning("No predictions to export")
            return

        records = [p.to_dict() for p in self.prediction_history]
        df = pd.DataFrame(records)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(records)} predictions to {output_path}")

    def clear_history(self) -> None:
        """Clear prediction history."""
        self.prediction_history.clear()
        logger.info("Prediction history cleared")
