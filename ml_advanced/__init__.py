"""
Advanced ML/AI Trading Suite
============================

Machine learning and AI components for quantitative trading.

Modules:
- hmm_regime_detector: Market regime detection using Hidden Markov Models
- lstm_confidence: Multi-output LSTM for signal confidence scoring
- ensemble: Multi-model ensemble prediction with weighted voting
- online_learning: Incremental learning with concept drift detection

MERGED FROM GAME_PLAN_2K28 - Production Ready

Usage:
    from ml_advanced.hmm_regime_detector import HMMRegimeDetector
    from ml_advanced.lstm_confidence import LSTMConfidenceModel
    from ml_advanced.ensemble import EnsemblePredictor
    from ml_advanced.online_learning import OnlineLearningManager
"""

# Lazy imports to avoid dependency issues in CI
# Import modules explicitly when needed

__all__ = [
    'hmm_regime_detector',
    'lstm_confidence',
    'ensemble',
    'online_learning',
]
