"""
Multi-Output LSTM Model Architecture
====================================

Three-headed LSTM for predicting:
1. Direction (binary classification)
2. Magnitude (regression)
3. Trade Success (binary classification)

MERGED FROM GAME_PLAN_2K28 - Production Ready
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, LSTM, Dense, Dropout, BatchNormalization
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau
    )
    TF_AVAILABLE = True
    KERAS_AVAILABLE = True
except Exception:
    # Catch all exceptions - TensorFlow can crash with access violations on Windows
    TF_AVAILABLE = False
    KERAS_AVAILABLE = False
    Model = Any
    keras = None
    tf = None
    logging.warning("TensorFlow/Keras not available. Model training disabled.")

from .config import LSTMConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Container for model predictions."""
    direction_prob: float
    magnitude_pct: float
    success_prob: float
    combined_confidence: float
    grade: str


class LSTMConfidenceModel:
    """
    Multi-output LSTM model for signal confidence scoring.

    Architecture:
        Input (lookback, features)
            |
        LSTM (128 units, return_sequences=True)
            |
        Dropout (0.3)
            |
        LSTM (64 units)
            |
        Dropout (0.3)
            |
        +-------------------+-------------------+
        |                   |                   |
        Dense(32)          Dense(32)          Dense(32)
        |                   |                   |
        Dense(1, sigmoid)  Dense(1, linear)   Dense(1, sigmoid)
        |                   |                   |
        DIRECTION          MAGNITUDE           SUCCESS
    """

    def __init__(self, config: LSTMConfig = DEFAULT_CONFIG):
        self.config = config
        self.model: Optional[Model] = None
        self._built = False

        if not KERAS_AVAILABLE:
            raise ImportError("TensorFlow/Keras required. pip install tensorflow")

    def build(self) -> Model:
        """Build the multi-output LSTM model."""
        logger.info("Building LSTM Confidence Model...")

        inputs = Input(
            shape=(self.config.lookback_bars, self.config.n_features),
            name='lstm_input'
        )

        # Shared LSTM backbone
        x = LSTM(self.config.lstm_units_1, return_sequences=True, name='lstm_1')(inputs)
        x = Dropout(self.config.dropout_rate, name='dropout_1')(x)
        x = LSTM(self.config.lstm_units_2, return_sequences=False, name='lstm_2')(x)
        x = Dropout(self.config.dropout_rate, name='dropout_2')(x)
        x = BatchNormalization(name='batch_norm')(x)

        # Direction output head
        direction_branch = Dense(32, activation='relu', name='direction_dense_1')(x)
        direction_branch = Dropout(0.2, name='direction_dropout')(direction_branch)
        direction_output = Dense(1, activation='sigmoid', name='direction_output')(direction_branch)

        # Magnitude output head
        magnitude_branch = Dense(32, activation='relu', name='magnitude_dense_1')(x)
        magnitude_branch = Dropout(0.2, name='magnitude_dropout')(magnitude_branch)
        magnitude_output = Dense(1, activation='linear', name='magnitude_output')(magnitude_branch)

        # Success output head
        success_branch = Dense(32, activation='relu', name='success_dense_1')(x)
        success_branch = Dropout(0.2, name='success_dropout')(success_branch)
        success_output = Dense(1, activation='sigmoid', name='success_output')(success_branch)

        self.model = Model(
            inputs=inputs,
            outputs=[direction_output, magnitude_output, success_output],
            name='lstm_confidence_model'
        )

        self.model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss={
                'direction_output': 'binary_crossentropy',
                'magnitude_output': 'mse',
                'success_output': 'binary_crossentropy'
            },
            loss_weights={
                'direction_output': self.config.loss_weight_direction,
                'magnitude_output': self.config.loss_weight_magnitude,
                'success_output': self.config.loss_weight_success
            },
            metrics={
                'direction_output': ['accuracy', 'AUC'],
                'magnitude_output': ['mae'],
                'success_output': ['accuracy', 'AUC']
            }
        )

        self._built = True
        logger.info(f"Model built: {self.model.count_params():,} parameters")
        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_direction: np.ndarray,
        y_magnitude: np.ndarray,
        y_success: np.ndarray,
        validation_data: Optional[Tuple] = None,
        callbacks: Optional[list] = None,
        class_weight: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """Train the model."""
        if self.model is None:
            self.build()

        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor=self.config.early_stopping_monitor,
                    patience=self.config.early_stopping_patience,
                    mode=self.config.early_stopping_mode,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.config.lr_reduce_factor,
                    patience=self.config.lr_reduce_patience,
                    min_lr=self.config.lr_min,
                    verbose=1
                ),
            ]

        logger.info(f"Training on {len(X_train):,} samples...")

        fit_kwargs = {
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs,
            'validation_split': self.config.validation_split if validation_data is None else 0.0,
            'validation_data': validation_data,
            'callbacks': callbacks,
            'verbose': 1
        }

        if class_weight is not None:
            fit_kwargs['class_weight'] = {
                'direction_output': class_weight,
                'success_output': class_weight,
            }
            logger.info(f"Applying class weights: {class_weight}")

        history = self.model.fit(
            X_train,
            {
                'direction_output': y_direction,
                'magnitude_output': y_magnitude,
                'success_output': y_success
            },
            **fit_kwargs
        )

        logger.info("Training complete.")
        return history.history

    def predict_raw(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get raw model predictions."""
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        predictions = self.model.predict(X, verbose=0)
        return predictions[0], predictions[1], predictions[2]

    def predict(self, X: np.ndarray) -> list:
        """Get structured predictions with confidence scores."""
        direction_probs, magnitudes, success_probs = self.predict_raw(X)

        predictions = []
        for i in range(len(X)):
            direction_prob = float(direction_probs[i][0])
            magnitude_pct = float(magnitudes[i][0])
            success_prob = float(success_probs[i][0])

            magnitude_norm = (magnitude_pct - self.config.magnitude_clip_min) / (
                self.config.magnitude_clip_max - self.config.magnitude_clip_min
            )
            magnitude_norm = np.clip(magnitude_norm, 0, 1)

            combined_confidence = (
                self.config.confidence_weight_direction * direction_prob +
                self.config.confidence_weight_success * success_prob +
                self.config.confidence_weight_magnitude * magnitude_norm
            )

            if combined_confidence >= self.config.threshold_grade_a:
                grade = "A"
            elif combined_confidence >= self.config.threshold_grade_b:
                grade = "B"
            elif combined_confidence >= self.config.threshold_grade_c:
                grade = "C"
            else:
                grade = "REJECTED"

            predictions.append(ModelPrediction(
                direction_prob=direction_prob,
                magnitude_pct=magnitude_pct,
                success_prob=success_prob,
                combined_confidence=combined_confidence,
                grade=grade
            ))

        return predictions

    def predict_single(self, X: np.ndarray) -> ModelPrediction:
        """Predict for a single sample."""
        if X.ndim == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])
        return self.predict(X)[0]

    def save(self, path: Optional[Path] = None):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        save_path = Path(path) if path else Path(self.config.model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")

    def load(self, path: Optional[Path] = None):
        """Load model from disk."""
        load_path = Path(path) if path else Path(self.config.model_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found at {load_path}")

        self.model = keras.models.load_model(load_path, compile=False)
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss={
                'direction_output': 'binary_crossentropy',
                'magnitude_output': 'mse',
                'success_output': 'binary_crossentropy'
            },
            loss_weights={
                'direction_output': self.config.loss_weight_direction,
                'magnitude_output': self.config.loss_weight_magnitude,
                'success_output': self.config.loss_weight_success
            }
        )
        self._built = True
        logger.info(f"Model loaded from {load_path}")

    @classmethod
    def from_pretrained(cls, path: Path, config: LSTMConfig = DEFAULT_CONFIG) -> "LSTMConfidenceModel":
        """Load a pre-trained model."""
        model = cls(config)
        model.load(path)
        return model

    def evaluate(
        self,
        X_test: np.ndarray,
        y_direction: np.ndarray,
        y_magnitude: np.ndarray,
        y_success: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        if self.model is None:
            raise ValueError("Model not built or loaded.")

        results = self.model.evaluate(
            X_test,
            {
                'direction_output': y_direction,
                'magnitude_output': y_magnitude,
                'success_output': y_success
            },
            verbose=0
        )
        return dict(zip(self.model.metrics_names, results))


def create_model(config: LSTMConfig = DEFAULT_CONFIG) -> LSTMConfidenceModel:
    """Factory function to create and build model."""
    model = LSTMConfidenceModel(config)
    model.build()
    return model
