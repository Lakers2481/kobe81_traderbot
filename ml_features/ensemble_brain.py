"""
Ensemble Brain - Multi-Model Decision Engine for Kobe Trading System.

Combines multiple ML models for robust signal generation:
- XGBoost (gradient boosting)
- LightGBM (fast gradient boosting)
- Random Forest (ensemble voting)
- Neural Network (pattern recognition)
- Meta-learner (stacking ensemble)

Features:
- Automatic feature importance analysis
- Model disagreement detection
- Confidence intervals
- Graceful degradation when models unavailable
- Optimized for mean-reversion strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pickle
import json
import logging
import warnings

# DEPRECATION WARNING (2026-01-08): This brain is supplementary
# The canonical brain is now autonomous.brain.AutonomousBrain
# This module provides ML ensemble predictions used BY the main brain
warnings.warn(
    "ml_features.ensemble_brain.EnsembleBrain is DEPRECATED as a standalone brain. "
    "Use autonomous.brain.AutonomousBrain instead, which integrates ensemble predictions. "
    "This module will continue to provide ML predictions but is not the primary decision maker.",
    DeprecationWarning,
    stacklevel=2
)

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for ensemble prediction results."""
    signal: float  # -1 to 1 (short to long)
    confidence: float  # 0 to 1
    direction: str  # 'long', 'short', 'neutral'
    model_votes: Dict[str, float] = field(default_factory=dict)
    model_confidences: Dict[str, float] = field(default_factory=dict)
    disagreement_score: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'signal': round(self.signal, 4),
            'confidence': round(self.confidence, 4),
            'direction': self.direction,
            'model_votes': {k: round(v, 4) for k, v in self.model_votes.items()},
            'model_confidences': {k: round(v, 4) for k, v in self.model_confidences.items()},
            'disagreement_score': round(self.disagreement_score, 4),
            'feature_importance': {k: round(v, 4) for k, v in self.feature_importance.items()},
            'metadata': self.metadata,
        }


class EnsembleBrain:
    """
    Multi-Model AI Ensemble for Trading Decisions.

    Combines XGBoost, LightGBM, Random Forest, and Neural Networks
    with a meta-learner for optimal signal generation.

    Optimized for mean-reversion strategies (Connors RSI-2, IBS).
    """

    MODEL_NAMES = ['xgboost', 'lightgbm', 'random_forest', 'neural_net']
    DEFAULT_WEIGHTS = {
        'xgboost': 0.30,
        'lightgbm': 0.30,
        'random_forest': 0.20,
        'neural_net': 0.20
    }

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        enable_training: bool = True
    ):
        """
        Initialize ensemble brain.

        Args:
            model_dir: Directory to save/load models
            enable_training: Whether to allow model training
        """
        self.model_dir = model_dir or Path(__file__).parent.parent / "models" / "ensemble"
        self.enable_training = enable_training
        self.models: Dict[str, Any] = {}
        self.meta_learner: Any = None
        self.feature_names: List[str] = []
        self.is_fitted = False
        self.model_weights = self.DEFAULT_WEIGHTS.copy()

        # Track which libraries are available
        self.available_libs = self._check_available_libraries()

        # Create model directory if needed
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _check_available_libraries(self) -> Dict[str, bool]:
        """Check which ML libraries are available."""
        available = {}

        try:
            import xgboost
            available['xgboost'] = True
        except ImportError:
            available['xgboost'] = False

        try:
            import lightgbm
            available['lightgbm'] = True
        except ImportError:
            available['lightgbm'] = False

        try:
            from sklearn.ensemble import RandomForestClassifier
            available['random_forest'] = True
        except ImportError:
            available['random_forest'] = False

        try:
            from sklearn.neural_network import MLPClassifier
            available['neural_net'] = True
        except ImportError:
            available['neural_net'] = False

        try:
            from sklearn.linear_model import LogisticRegression
            available['meta_learner'] = True
        except ImportError:
            available['meta_learner'] = False

        return available

    def _init_models(self, n_features: int):
        """Initialize all available models."""
        self.models = {}

        # XGBoost
        if self.available_libs.get('xgboost'):
            try:
                import xgboost as xgb
                self.models['xgboost'] = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    objective='binary:logistic',
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    n_jobs=-1
                )
            except Exception as e:
                logger.warning(f"XGBoost init failed: {e}")

        # LightGBM
        if self.available_libs.get('lightgbm'):
            try:
                import lightgbm as lgb
                self.models['lightgbm'] = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    objective='binary',
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            except Exception as e:
                logger.warning(f"LightGBM init failed: {e}")

        # Random Forest
        if self.available_libs.get('random_forest'):
            try:
                from sklearn.ensemble import RandomForestClassifier
                self.models['random_forest'] = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    random_state=42,
                    n_jobs=-1
                )
            except Exception as e:
                logger.warning(f"Random Forest init failed: {e}")

        # Neural Network
        if self.available_libs.get('neural_net'):
            try:
                from sklearn.neural_network import MLPClassifier
                self.models['neural_net'] = MLPClassifier(
                    hidden_layer_sizes=(64, 32, 16),
                    activation='relu',
                    solver='adam',
                    learning_rate_init=0.001,
                    max_iter=200,
                    random_state=42
                )
            except Exception as e:
                logger.warning(f"Neural Network init failed: {e}")

        # Meta-learner
        if self.available_libs.get('meta_learner'):
            try:
                from sklearn.linear_model import LogisticRegression
                self.meta_learner = LogisticRegression(
                    random_state=42,
                    max_iter=1000
                )
            except Exception as e:
                logger.warning(f"Meta-learner init failed: {e}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train all ensemble models.

        Args:
            X: Feature DataFrame
            y: Target labels (0 = loss, 1 = profit)
            validation_split: Fraction for validation

        Returns:
            Dictionary of model accuracies
        """
        if not self.enable_training:
            raise RuntimeError("Training is disabled for this instance")

        self.feature_names = list(X.columns)
        n_features = len(self.feature_names)

        # Initialize models
        self._init_models(n_features)

        if not self.models:
            raise RuntimeError("No ML models available. Install scikit-learn, xgboost, or lightgbm.")

        # Prepare data
        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(int)

        # Handle NaN/inf
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Train/validation split (preserving time order)
        n_val = int(len(X_arr) * validation_split)
        X_train, X_val = X_arr[:-n_val], X_arr[-n_val:]
        y_train, y_val = y_arr[:-n_val], y_arr[-n_val:]

        accuracies = {}
        val_predictions = {}

        # Train each model
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train)

                # Validate
                val_pred = model.predict(X_val)
                val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else val_pred.astype(float)

                acc = (val_pred == y_val).mean()
                accuracies[name] = acc
                val_predictions[name] = val_proba

                logger.info(f"  {name} accuracy: {acc:.4f}")

            except Exception as e:
                logger.error(f"  {name} training failed: {e}")
                accuracies[name] = 0.0

        # Train meta-learner on validation predictions
        if self.meta_learner and len(val_predictions) > 1:
            try:
                meta_X = np.column_stack([val_predictions[n] for n in val_predictions])
                self.meta_learner.fit(meta_X, y_val)
                meta_pred = self.meta_learner.predict(meta_X)
                meta_acc = (meta_pred == y_val).mean()
                accuracies['meta_learner'] = meta_acc
                logger.info(f"  meta_learner accuracy: {meta_acc:.4f}")
            except Exception as e:
                logger.warning(f"  meta_learner training failed: {e}")

        # Update model weights based on accuracy
        self._update_weights(accuracies)

        self.is_fitted = True
        return accuracies

    def _update_weights(self, accuracies: Dict[str, float]):
        """Update model weights based on validation accuracy."""
        total = sum(acc for name, acc in accuracies.items() if name in self.MODEL_NAMES and acc > 0.5)

        if total > 0:
            for name in self.MODEL_NAMES:
                if name in accuracies and accuracies[name] > 0.5:
                    self.model_weights[name] = accuracies[name] / total
                else:
                    self.model_weights[name] = 0.0

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_proba: bool = True
    ) -> PredictionResult:
        """
        Generate ensemble prediction.

        Args:
            X: Single sample or DataFrame of features
            return_proba: Whether to return probability estimates

        Returns:
            PredictionResult with signal, confidence, and model details
        """
        # Handle single sample
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        elif isinstance(X, np.ndarray) and X.ndim == 1:
            X = X.reshape(1, -1)

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = X.astype(np.float32)

        # Handle NaN/inf
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Get predictions from each model
        model_votes = {}
        model_confidences = {}

        for name, model in self.models.items():
            try:
                if return_proba and hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_arr)[0, 1]
                    model_votes[name] = proba
                    model_confidences[name] = abs(proba - 0.5) * 2
                else:
                    pred = model.predict(X_arr)[0]
                    model_votes[name] = float(pred)
                    model_confidences[name] = 0.5
            except Exception:
                pass

        if not model_votes:
            return PredictionResult(
                signal=0.0,
                confidence=0.0,
                direction='neutral',
                metadata={'error': 'No models available'}
            )

        # Weighted ensemble
        weighted_sum = 0.0
        total_weight = 0.0

        for name, vote in model_votes.items():
            weight = self.model_weights.get(name, 0.25)
            weighted_sum += vote * weight
            total_weight += weight

        if total_weight > 0:
            ensemble_proba = weighted_sum / total_weight
        else:
            ensemble_proba = sum(model_votes.values()) / len(model_votes)

        # Use meta-learner if available
        if self.meta_learner and self.is_fitted and len(model_votes) > 1:
            try:
                meta_X = np.array([[model_votes.get(n, 0.5) for n in self.MODEL_NAMES if n in model_votes]])
                if meta_X.shape[1] > 1:
                    meta_proba = self.meta_learner.predict_proba(meta_X)[0, 1]
                    ensemble_proba = 0.7 * meta_proba + 0.3 * ensemble_proba
            except Exception:
                pass

        # Convert to signal (-1 to 1)
        signal = (ensemble_proba - 0.5) * 2

        # Confidence from model agreement
        if len(model_votes) > 1:
            std_dev = np.std(list(model_votes.values()))
            disagreement = min(std_dev * 2, 1.0)
        else:
            disagreement = 0.5

        confidence = 1.0 - disagreement

        # Direction
        if signal > 0.1:
            direction = 'long'
        elif signal < -0.1:
            direction = 'short'
        else:
            direction = 'neutral'

        # Feature importance
        feature_importance = self._get_feature_importance()

        return PredictionResult(
            signal=signal,
            confidence=confidence,
            direction=direction,
            model_votes=model_votes,
            model_confidences=model_confidences,
            disagreement_score=disagreement,
            feature_importance=feature_importance,
            metadata={
                'ensemble_proba': ensemble_proba,
                'model_weights': self.model_weights,
                'n_models_voted': len(model_votes)
            }
        )

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance from tree models."""
        if not self.feature_names:
            return {}

        importance = np.zeros(len(self.feature_names))
        count = 0

        # XGBoost importance
        if 'xgboost' in self.models:
            try:
                imp = self.models['xgboost'].feature_importances_
                if len(imp) == len(self.feature_names):
                    importance += imp
                    count += 1
            except Exception:
                pass

        # LightGBM importance
        if 'lightgbm' in self.models:
            try:
                imp = self.models['lightgbm'].feature_importances_
                if len(imp) == len(self.feature_names):
                    importance += imp
                    count += 1
            except Exception:
                pass

        # Random Forest importance
        if 'random_forest' in self.models:
            try:
                imp = self.models['random_forest'].feature_importances_
                if len(imp) == len(self.feature_names):
                    importance += imp
                    count += 1
            except Exception:
                pass

        if count > 0:
            importance /= count
            return dict(zip(self.feature_names, importance.tolist()))

        return {}

    def save(self, path: Optional[Path] = None):
        """Save trained models to disk."""
        path = path or self.model_dir

        state = {
            'feature_names': self.feature_names,
            'model_weights': self.model_weights,
            'is_fitted': self.is_fitted,
            'available_libs': self.available_libs,
            'timestamp': datetime.now().isoformat()
        }

        with open(path / 'ensemble_state.json', 'w') as f:
            json.dump(state, f, indent=2)

        for name, model in self.models.items():
            try:
                model_path = path / f'{name}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            except Exception as e:
                logger.warning(f"Failed to save {name}: {e}")

        if self.meta_learner:
            try:
                with open(path / 'meta_learner.pkl', 'wb') as f:
                    pickle.dump(self.meta_learner, f)
            except Exception as e:
                logger.warning(f"Failed to save meta_learner: {e}")

        logger.info(f"Ensemble saved to {path}")

    def load(self, path: Optional[Path] = None) -> bool:
        """Load trained models from disk."""
        path = path or self.model_dir

        try:
            with open(path / 'ensemble_state.json', 'r') as f:
                state = json.load(f)

            self.feature_names = state.get('feature_names', [])
            self.model_weights = state.get('model_weights', self.DEFAULT_WEIGHTS)
            self.is_fitted = state.get('is_fitted', False)

            self.models = {}
            for name in self.MODEL_NAMES:
                model_path = path / f'{name}.pkl'
                if model_path.exists():
                    try:
                        with open(model_path, 'rb') as f:
                            self.models[name] = pickle.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load {name}: {e}")

            meta_path = path / 'meta_learner.pkl'
            if meta_path.exists():
                try:
                    with open(meta_path, 'rb') as f:
                        self.meta_learner = pickle.load(f)
                except Exception:
                    pass

            logger.info(f"Ensemble loaded from {path}")
            return bool(self.models)

        except Exception as e:
            logger.warning(f"Failed to load ensemble: {e}")
            return False

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        return {
            'available_libraries': self.available_libs,
            'loaded_models': list(self.models.keys()),
            'is_fitted': self.is_fitted,
            'model_weights': self.model_weights,
            'feature_count': len(self.feature_names)
        }


class QuickEnsemble:
    """
    Lightweight ensemble that works without heavy ML libraries.
    Uses statistical methods for predictions.
    """

    def __init__(self):
        self.thresholds: Dict[str, tuple] = {}
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Learn optimal thresholds from data."""
        for col in X.columns:
            best_acc = 0.5
            X[col].median()

            for pct in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                thresh = X[col].quantile(pct)

                # Above threshold = 1, below = 0
                pred = (X[col] > thresh).astype(int)
                acc = (pred == y).mean()

                # Try inverse too
                pred_inv = (X[col] < thresh).astype(int)
                acc_inv = (pred_inv == y).mean()

                if acc > best_acc:
                    best_acc = acc
                    self.thresholds[col] = (thresh, 1, acc)

                if acc_inv > best_acc:
                    best_acc = acc_inv
                    self.thresholds[col] = (thresh, -1, acc_inv)

        self.fitted = True

    def predict(self, X: pd.DataFrame) -> PredictionResult:
        """Generate prediction using learned thresholds."""
        if not self.fitted:
            return PredictionResult(
                signal=0.0,
                confidence=0.0,
                direction='neutral'
            )

        votes = []
        weights = []

        for col, (thresh, direction, acc) in self.thresholds.items():
            if col in X.columns:
                val = X[col].iloc[0] if isinstance(X, pd.DataFrame) else X[col]
                if direction == 1:
                    vote = 1 if val > thresh else 0
                else:
                    vote = 1 if val < thresh else 0
                votes.append(vote)
                weights.append(max(acc - 0.5, 0.01))

        if not votes:
            return PredictionResult(
                signal=0.0,
                confidence=0.0,
                direction='neutral'
            )

        # Weighted vote
        weighted_vote = np.average(votes, weights=weights)
        signal = (weighted_vote - 0.5) * 2

        if signal > 0.1:
            direction = 'long'
        elif signal < -0.1:
            direction = 'short'
        else:
            direction = 'neutral'

        return PredictionResult(
            signal=signal,
            confidence=abs(signal),
            direction=direction,
            metadata={'method': 'quick_ensemble', 'n_features': len(votes)}
        )


# Singleton instance
_ensemble_brain: Optional[EnsembleBrain] = None


def get_ensemble_brain(model_dir: Optional[Path] = None) -> EnsembleBrain:
    """Get or create the global ensemble brain instance."""
    global _ensemble_brain
    if _ensemble_brain is None:
        _ensemble_brain = EnsembleBrain(model_dir=model_dir)
    return _ensemble_brain
