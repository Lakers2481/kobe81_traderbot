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
"""

from .hmm_regime_detector import (
    HMMRegimeDetector,
    AdaptiveRegimeDetector,
    RegimeState,
    create_regime_detector,
    HMM_AVAILABLE
)

from .online_learning import (
    OnlineLearningManager,
    ExperienceReplayBuffer,
    ConceptDriftDetector,
    TradeOutcome,
    create_online_learning_manager
)

# Conditional imports for optional dependencies
__all__ = [
    # HMM Regime Detection
    'HMMRegimeDetector',
    'AdaptiveRegimeDetector',
    'RegimeState',
    'create_regime_detector',
    'HMM_AVAILABLE',

    # Online Learning
    'OnlineLearningManager',
    'ExperienceReplayBuffer',
    'ConceptDriftDetector',
    'TradeOutcome',
    'create_online_learning_manager',
]

# Import submodules for explicit access
from . import lstm_confidence
from . import ensemble
