"""
LSTM Confidence Scoring System
==============================

Multi-output LSTM model for signal confidence scoring.

Outputs:
1. Direction probability (up/down)
2. Magnitude prediction (% move)
3. Success probability (profitable trade)

Grades signals A/B/C/REJECTED based on combined confidence.

MERGED FROM GAME_PLAN_2K28 - Production Ready
"""

from .config import LSTMConfig, DEFAULT_CONFIG

# Conditional import for TensorFlow
try:
    from .model import (
        LSTMConfidenceModel,
        ModelPrediction,
        create_model
    )
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    LSTMConfidenceModel = None
    ModelPrediction = None
    create_model = None

__all__ = [
    'LSTMConfig',
    'DEFAULT_CONFIG',
    'LSTMConfidenceModel',
    'ModelPrediction',
    'create_model',
    'LSTM_AVAILABLE',
]
