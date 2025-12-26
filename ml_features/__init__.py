"""
ML Features Module for Kobe Trading System.

Provides machine learning enhanced features for trading strategies:
- TechnicalFeatures: 150+ technical indicators via pandas-ta
- AnomalyDetector: Matrix profile anomaly detection via stumpy
- FeaturePipeline: Unified feature extraction pipeline
- SignalConfidence: ML-based signal confidence scoring
- RegimeDetector: ML-based market regime detection
"""
from __future__ import annotations

from .technical_features import (
    TechnicalFeatures,
    compute_momentum_features,
    compute_volatility_features,
    compute_trend_features,
    compute_volume_features,
)
from .anomaly_detection import (
    AnomalyDetector,
    detect_price_anomalies,
    detect_volume_anomalies,
    get_anomaly_score,
)
from .feature_pipeline import (
    FeaturePipeline,
    FeatureConfig,
    extract_all_features,
)
from .signal_confidence import (
    SignalConfidence,
    compute_signal_confidence,
    ConfidenceLevel,
)
from .regime_ml import (
    RegimeDetectorML,
    RegimeState,
    detect_regime_ml,
)
from .strategy_enhancer import (
    StrategyEnhancer,
    EnhancerConfig,
    enhance_strategy,
)
from .regime_hmm import (
    MarketRegimeDetector,
    MarketRegime,
    RegimeResult,
    get_regime_detector,
)
from .conviction_scorer import (
    ConvictionScorer,
    ConvictionBreakdown,
    get_conviction_scorer,
)
from .ensemble_brain import (
    EnsembleBrain,
    QuickEnsemble,
    PredictionResult,
    get_ensemble_brain,
)

__all__ = [
    # Technical Features
    "TechnicalFeatures",
    "compute_momentum_features",
    "compute_volatility_features",
    "compute_trend_features",
    "compute_volume_features",
    # Anomaly Detection
    "AnomalyDetector",
    "detect_price_anomalies",
    "detect_volume_anomalies",
    "get_anomaly_score",
    # Feature Pipeline
    "FeaturePipeline",
    "FeatureConfig",
    "extract_all_features",
    # Signal Confidence
    "SignalConfidence",
    "compute_signal_confidence",
    "ConfidenceLevel",
    # Regime Detection
    "RegimeDetectorML",
    "RegimeState",
    "detect_regime_ml",
    # Strategy Enhancement
    "StrategyEnhancer",
    "EnhancerConfig",
    "enhance_strategy",
    # Regime Detection (HMM)
    "MarketRegimeDetector",
    "MarketRegime",
    "RegimeResult",
    "get_regime_detector",
    # Conviction Scoring
    "ConvictionScorer",
    "ConvictionBreakdown",
    "get_conviction_scorer",
    # Ensemble Brain
    "EnsembleBrain",
    "QuickEnsemble",
    "PredictionResult",
    "get_ensemble_brain",
]
