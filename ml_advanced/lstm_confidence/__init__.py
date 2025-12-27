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

# TensorFlow can crash with access violations on Windows before Python can catch it
# Check if TensorFlow is safe to import by looking for a marker file or env var
import os

_TF_DISABLED = os.environ.get('KOBE_DISABLE_TENSORFLOW', '0') == '1'

# Also check if there's a local disable marker
_TF_MARKER_FILE = os.path.join(os.path.dirname(__file__), '.tf_disabled')
if os.path.exists(_TF_MARKER_FILE):
    _TF_DISABLED = True

if _TF_DISABLED:
    # Skip TensorFlow entirely
    LSTM_AVAILABLE = False
    TF_AVAILABLE = False
    LSTMConfidenceModel = None
    ModelPrediction = None
    create_model = None
else:
    # Attempt conditional import for TensorFlow
    try:
        from .model import (
            LSTMConfidenceModel,
            ModelPrediction,
            create_model,
            TF_AVAILABLE,
        )
        LSTM_AVAILABLE = TF_AVAILABLE
    except Exception:
        # TensorFlow not available or import failed
        LSTM_AVAILABLE = False
        TF_AVAILABLE = False
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
    'TF_AVAILABLE',
]
