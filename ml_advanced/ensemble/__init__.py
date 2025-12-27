"""
Ensemble Prediction System
==========================

Multi-model ensemble predictor combining:
- LSTM neural networks
- XGBoost gradient boosting
- LightGBM gradient boosting

Features:
- Weighted voting with configurable weights
- Confidence scoring based on model agreement
- Weight optimization via grid search
- Prediction history tracking

MERGED FROM GAME_PLAN_2K28 - Production Ready
"""

from .ensemble_predictor import (
    EnsemblePredictor,
    EnsemblePrediction,
    BaseModelWrapper,
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
)

# Conditional imports for optional dependencies
if XGBOOST_AVAILABLE:
    from .ensemble_predictor import XGBoostWrapper
else:
    XGBoostWrapper = None

if LIGHTGBM_AVAILABLE:
    from .ensemble_predictor import LightGBMWrapper
else:
    LightGBMWrapper = None

__all__ = [
    'EnsemblePredictor',
    'EnsemblePrediction',
    'BaseModelWrapper',
    'XGBoostWrapper',
    'LightGBMWrapper',
    'XGBOOST_AVAILABLE',
    'LIGHTGBM_AVAILABLE',
]
