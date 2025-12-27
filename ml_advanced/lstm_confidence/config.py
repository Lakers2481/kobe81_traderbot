"""
LSTM Confidence Model Configuration
====================================

Configuration parameters for the multi-output LSTM model.

MERGED FROM GAME_PLAN_2K28 - Production Ready
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LSTMConfig:
    """Configuration for LSTM Confidence Model."""

    # Architecture
    lookback_bars: int = 30
    n_features: int = 12
    lstm_units_1: int = 128
    lstm_units_2: int = 64
    dropout_rate: float = 0.3

    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2

    # Loss weights
    loss_weight_direction: float = 0.4
    loss_weight_magnitude: float = 0.3
    loss_weight_success: float = 0.3

    # Confidence calculation weights
    confidence_weight_direction: float = 0.40
    confidence_weight_success: float = 0.30
    confidence_weight_magnitude: float = 0.30

    # Grade thresholds
    threshold_grade_a: float = 0.70
    threshold_grade_b: float = 0.60
    threshold_grade_c: float = 0.50

    # Magnitude normalization
    magnitude_clip_min: float = -10.0
    magnitude_clip_max: float = 10.0

    # Early stopping
    early_stopping_monitor: str = 'val_loss'
    early_stopping_patience: int = 15
    early_stopping_mode: str = 'min'

    # Learning rate reduction
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 5
    lr_min: float = 0.00001

    # Model paths
    model_path: str = "models/lstm_confidence/model.keras"
    checkpoint_dir: str = "models/lstm_confidence/checkpoints"


# Default configuration
DEFAULT_CONFIG = LSTMConfig()
