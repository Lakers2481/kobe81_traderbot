"""
ML Confidence Integrator
=========================

Unified confidence scoring that combines:
1. ConvictionScorer (rule-based technical analysis)
2. EnsemblePredictor (ML-based model predictions)
3. LSTM Confidence (if available)

Provides a single confidence score (0-1) for position sizing integration
with PortfolioRiskManager.

Usage:
    from ml_features.confidence_integrator import get_ml_confidence

    confidence = get_ml_confidence(
        signal={'symbol': 'AAPL', 'entry_price': 150, 'stop_loss': 145},
        price_data=df,
        spy_data=spy_df,
        vix_level=18.5
    )

    # Use with PortfolioRiskManager
    decision = risk_manager.evaluate_trade(signal, positions, price_data, confidence)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceResult:
    """Combined confidence result."""
    confidence: float           # Final confidence (0-1)
    conviction_score: int       # Rule-based score (0-100)
    ml_confidence: Optional[float]  # ML ensemble confidence (0-1)
    lstm_confidence: Optional[float]  # LSTM confidence if available
    ensemble_std: Optional[float]  # Model disagreement
    tier: str                   # EXCEPTIONAL, EXCELLENT, GOOD, ACCEPTABLE, WEAK
    action: str                 # STRONG BUY, BUY, CONSIDER, PASS
    components_used: list       # Which components contributed

    def to_dict(self) -> Dict[str, Any]:
        return {
            'confidence': round(self.confidence, 4),
            'conviction_score': self.conviction_score,
            'ml_confidence': round(self.ml_confidence, 4) if self.ml_confidence else None,
            'lstm_confidence': round(self.lstm_confidence, 4) if self.lstm_confidence else None,
            'ensemble_std': round(self.ensemble_std, 4) if self.ensemble_std else None,
            'tier': self.tier,
            'action': self.action,
            'components_used': self.components_used,
        }


class ConfidenceIntegrator:
    """
    Integrates multiple confidence sources into a unified score.

    Weights:
    - ConvictionScorer: 40% (always available, rule-based)
    - EnsemblePredictor: 35% (if ML models loaded)
    - LSTM Confidence: 25% (if LSTM available)

    Falls back gracefully when components aren't available.
    """

    def __init__(
        self,
        conviction_weight: float = 0.40,
        ensemble_weight: float = 0.35,
        lstm_weight: float = 0.25,
        min_confidence_floor: float = 0.1,  # Never return below this
    ):
        self.conviction_weight = conviction_weight
        self.ensemble_weight = ensemble_weight
        self.lstm_weight = lstm_weight
        self.min_confidence_floor = min_confidence_floor

        # Lazy-loaded components
        self._conviction_scorer = None
        self._ensemble_predictor = None
        self._lstm_model = None

        logger.info("ConfidenceIntegrator initialized")

    @property
    def conviction_scorer(self):
        """Lazy load ConvictionScorer."""
        if self._conviction_scorer is None:
            try:
                from ml_features.conviction_scorer import get_conviction_scorer
                self._conviction_scorer = get_conviction_scorer()
            except ImportError:
                logger.warning("ConvictionScorer not available")
        return self._conviction_scorer

    @property
    def ensemble_predictor(self):
        """Lazy load EnsemblePredictor."""
        if self._ensemble_predictor is None:
            try:
                from ml_advanced.ensemble.ensemble_predictor import EnsemblePredictor
                self._ensemble_predictor = EnsemblePredictor()
                # Note: Models need to be loaded separately
            except ImportError:
                logger.warning("EnsemblePredictor not available")
        return self._ensemble_predictor

    @property
    def lstm_model(self):
        """Lazy load LSTM model."""
        if self._lstm_model is None:
            try:
                # Import from package __init__ which has safe exception handling
                from ml_advanced.lstm_confidence import LSTMConfidenceModel, LSTM_AVAILABLE
                if LSTM_AVAILABLE and LSTMConfidenceModel is not None:
                    self._lstm_model = LSTMConfidenceModel()
                else:
                    logger.info("TensorFlow not available, LSTM disabled")
            except Exception:
                # Catch all exceptions - TensorFlow can crash on Windows
                logger.warning("LSTM model not available")
        return self._lstm_model

    def calculate_confidence(
        self,
        signal: Dict[str, Any],
        price_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame] = None,
        vix_level: Optional[float] = None,
        features: Optional[np.ndarray] = None,
    ) -> ConfidenceResult:
        """
        Calculate unified confidence score.

        Args:
            signal: Trade signal dict with symbol, entry_price, stop_loss, etc.
            price_data: Price history for the symbol
            spy_data: Optional SPY price data for market context
            vix_level: Optional current VIX level
            features: Optional pre-computed feature array for ML models

        Returns:
            ConfidenceResult with unified confidence and breakdown
        """
        components_used = []
        weighted_scores = []
        weights_used = []

        conviction_score = 50  # Default neutral
        ml_confidence = None
        lstm_confidence = None
        ensemble_std = None
        tier = "ACCEPTABLE"
        action = "CONSIDER"

        # === Component 1: Conviction Scorer (Rule-Based) ===
        if self.conviction_scorer:
            try:
                breakdown = self.conviction_scorer.calculate_conviction(
                    signal=signal,
                    price_data=price_data,
                    spy_data=spy_data,
                    vix_level=vix_level
                )
                conviction_score = breakdown.total_score
                tier = breakdown.tier
                action = breakdown.action

                # Convert 0-100 to 0-1
                conviction_confidence = conviction_score / 100.0
                weighted_scores.append(conviction_confidence)
                weights_used.append(self.conviction_weight)
                components_used.append("conviction")

            except Exception as e:
                logger.warning(f"ConvictionScorer failed: {e}")

        # === Component 2: Ensemble Predictor (ML-Based) ===
        if self.ensemble_predictor and features is not None:
            try:
                # Check if any models are loaded
                if self.ensemble_predictor.models:
                    ensemble_result = self.ensemble_predictor.predict_with_confidence(features)
                    ml_confidence = ensemble_result.confidence
                    ensemble_std = ensemble_result.std_dev

                    weighted_scores.append(ml_confidence)
                    weights_used.append(self.ensemble_weight)
                    components_used.append("ensemble")

            except Exception as e:
                logger.warning(f"EnsemblePredictor failed: {e}")

        # === Component 3: LSTM Confidence (Deep Learning) ===
        if self.lstm_model and features is not None:
            try:
                if self.lstm_model.model is not None:
                    # Reshape for LSTM if needed
                    lstm_features = features
                    if lstm_features.ndim == 1:
                        lstm_features = lstm_features.reshape(1, -1, 1)
                    elif lstm_features.ndim == 2:
                        lstm_features = lstm_features.reshape(1, lstm_features.shape[0], lstm_features.shape[1])

                    lstm_pred = self.lstm_model.predict(lstm_features)
                    lstm_confidence = float(lstm_pred)

                    weighted_scores.append(lstm_confidence)
                    weights_used.append(self.lstm_weight)
                    components_used.append("lstm")

            except Exception as e:
                logger.warning(f"LSTM model failed: {e}")

        # === Calculate Final Confidence ===
        if weighted_scores:
            # Normalize weights
            total_weight = sum(weights_used)
            normalized_weights = [w / total_weight for w in weights_used]

            # Weighted average
            final_confidence = sum(
                score * weight for score, weight in zip(weighted_scores, normalized_weights)
            )
        else:
            # Fallback: use conviction score only
            final_confidence = conviction_score / 100.0

        # Apply floor
        final_confidence = max(final_confidence, self.min_confidence_floor)

        return ConfidenceResult(
            confidence=final_confidence,
            conviction_score=conviction_score,
            ml_confidence=ml_confidence,
            lstm_confidence=lstm_confidence,
            ensemble_std=ensemble_std,
            tier=tier,
            action=action,
            components_used=components_used
        )

    def get_simple_confidence(
        self,
        signal: Dict[str, Any],
        price_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame] = None,
        vix_level: Optional[float] = None,
    ) -> float:
        """
        Quick method to get just the confidence score (0-1).

        Use this for integration with PortfolioRiskManager.
        """
        result = self.calculate_confidence(signal, price_data, spy_data, vix_level)
        return result.confidence

    def get_status(self) -> Dict[str, Any]:
        """Get status of all confidence components."""
        return {
            'conviction_scorer': self._conviction_scorer is not None,
            'ensemble_predictor': self._ensemble_predictor is not None,
            'lstm_model': self._lstm_model is not None,
            'weights': {
                'conviction': self.conviction_weight,
                'ensemble': self.ensemble_weight,
                'lstm': self.lstm_weight,
            }
        }


# Singleton instance
_confidence_integrator: Optional[ConfidenceIntegrator] = None


def get_confidence_integrator() -> ConfidenceIntegrator:
    """Get or create singleton ConfidenceIntegrator."""
    global _confidence_integrator
    if _confidence_integrator is None:
        _confidence_integrator = ConfidenceIntegrator()
    return _confidence_integrator


def get_ml_confidence(
    signal: Dict[str, Any],
    price_data: pd.DataFrame,
    spy_data: Optional[pd.DataFrame] = None,
    vix_level: Optional[float] = None,
) -> float:
    """
    Convenience function to get ML confidence score.

    Example:
        confidence = get_ml_confidence(
            signal={'symbol': 'AAPL', 'entry_price': 150, 'stop_loss': 145},
            price_data=aapl_df,
            spy_data=spy_df,
            vix_level=18.5
        )

        # Use with PortfolioRiskManager
        from portfolio.risk_manager import get_risk_manager
        manager = get_risk_manager(equity=100000)
        decision = manager.evaluate_trade(signal, positions, price_data, confidence)
    """
    integrator = get_confidence_integrator()
    return integrator.get_simple_confidence(signal, price_data, spy_data, vix_level)
