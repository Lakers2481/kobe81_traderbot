"""
Hidden Markov Model Regime Detection for Kobe Trading System

Probabilistic market regime classification using HMM.
Identifies hidden states (bull/bear/sideways) from observable market data.

Key Features:
- 3-state HMM (bull/neutral/bear) for simplicity
- Probabilistic confidence scores for each regime
- Transition probability matrix for regime forecasting
- Position sizing multipliers based on regime + confidence
- Graceful fallback to rule-based detection

MERGED FROM GAME_PLAN_2K28 - Production Ready
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from datetime import datetime
import pickle

import numpy as np
import pandas as pd

# Wrap imports for CI compatibility
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

logger = logging.getLogger(__name__)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to title case (Open, High, Low, Close, Volume)."""
    col_map = {
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume',
        'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low',
        'CLOSE': 'Close', 'VOLUME': 'Volume',
    }
    return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})


class MarketRegime(Enum):
    """Market regime states."""
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"


@dataclass
class RegimeState:
    """Current regime state with probabilities."""
    regime: MarketRegime
    confidence: float
    probabilities: Dict[MarketRegime, float]
    transition_probs: Dict[MarketRegime, float]
    days_in_regime: int
    expected_duration: float
    feature_snapshot: Dict[str, float]
    timestamp: datetime


class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection.

    Uses Gaussian HMM with observable features:
    - SPY returns (momentum)
    - SPY volatility (realized vol)
    - VIX level (implied vol)
    - Market breadth (advance/decline)
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        covariance_type: str = 'full',
        random_state: int = 42,
        cache_dir: str = "data/ai_learning/hmm_regimes"
    ):
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn not installed. Run: pip install hmmlearn\n"
                "For CI/testing without hmmlearn, use AdaptiveRegimeDetector instead."
            )

        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            verbose=False
        )
        self.is_fitted = False
        self.state_labels: Dict[int, MarketRegime] = {}
        self.feature_names = ['returns', 'volatility', 'vix', 'breadth']
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}

        self.current_regime: Optional[RegimeState] = None
        self.regime_history: List[RegimeState] = []
        self.regime_sequence: List[int] = []

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"HMMRegimeDetector initialized (n_states={n_states})")

    def prepare_features(
        self,
        spy_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        breadth_data: Optional[pd.DataFrame] = None,
        lookback_vol: int = 21
    ) -> pd.DataFrame:
        """Prepare observable features for HMM."""
        # Normalize column names (handle lowercase 'close' from multi-source provider)
        spy_data = _normalize_columns(spy_data)
        vix_data = _normalize_columns(vix_data)
        spy_close = spy_data['Close']
        vix_close = vix_data['Close']

        common_dates = spy_close.index.intersection(vix_close.index)
        spy_close = spy_close.loc[common_dates]
        vix_close = vix_close.loc[common_dates]

        features = pd.DataFrame(index=common_dates)

        # SPY returns (momentum)
        features['returns'] = spy_close.pct_change(10) * 100

        # Realized volatility
        daily_returns = spy_close.pct_change()
        features['volatility'] = daily_returns.rolling(lookback_vol).std() * np.sqrt(252) * 100

        # VIX level
        features['vix'] = vix_close

        # Market breadth
        if breadth_data is not None and 'breadth' in breadth_data.columns:
            breadth_aligned = breadth_data['breadth'].reindex(common_dates)
            features['breadth'] = breadth_aligned.fillna(0)
        else:
            sma_50 = spy_close.rolling(50).mean()
            features['breadth'] = (spy_close / sma_50 - 1) * 100

        features = features.dropna()

        if len(features) < 100:
            logger.warning(f"Only {len(features)} samples. Need 100+ for reliable training.")

        return features

    def fit(
        self,
        spy_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        breadth_data: Optional[pd.DataFrame] = None
    ) -> 'HMMRegimeDetector':
        """Train HMM on historical data."""
        logger.info("Training HMM regime detector...")

        X = self.prepare_features(spy_data, vix_data, breadth_data)

        if len(X) < 100:
            raise ValueError(f"Insufficient data: {len(X)} samples (need 100+)")

        self.feature_means = X.mean().to_dict()
        self.feature_stds = X.std().to_dict()

        X_standardized = (X - X.mean()) / X.std()
        self.model.fit(X_standardized.values)

        regime_sequence = self.model.predict(X_standardized.values)
        self._label_states(X, regime_sequence)

        self.is_fitted = True

        log_likelihood = self.model.score(X_standardized.values)
        logger.info(f"HMM training complete. Log-likelihood: {log_likelihood:.2f}")

        return self

    def _label_states(self, features: pd.DataFrame, regime_sequence: np.ndarray) -> None:
        """Label HMM states based on feature characteristics."""
        state_characteristics = {}

        for state_id in range(self.n_states):
            mask = regime_sequence == state_id
            state_features = features[mask]

            if len(state_features) == 0:
                state_characteristics[state_id] = {'mean_return': 0.0}
                continue

            state_characteristics[state_id] = {
                'mean_return': state_features['returns'].mean(),
                'mean_vol': state_features['volatility'].mean(),
                'mean_vix': state_features['vix'].mean(),
                'n_samples': len(state_features)
            }

        sorted_states = sorted(state_characteristics.items(), key=lambda x: x[1]['mean_return'])

        if len(sorted_states) >= 3:
            self.state_labels[sorted_states[0][0]] = MarketRegime.BEARISH
            self.state_labels[sorted_states[1][0]] = MarketRegime.NEUTRAL
            self.state_labels[sorted_states[2][0]] = MarketRegime.BULLISH
        else:
            for i, (state_id, _) in enumerate(sorted_states):
                if i == 0:
                    self.state_labels[state_id] = MarketRegime.BEARISH
                elif i == len(sorted_states) - 1:
                    self.state_labels[state_id] = MarketRegime.BULLISH
                else:
                    self.state_labels[state_id] = MarketRegime.NEUTRAL

        logger.info(f"State labels assigned: {self.state_labels}")

    def detect_regime(
        self,
        spy_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        breadth_data: Optional[pd.DataFrame] = None
    ) -> RegimeState:
        """Detect current market regime with probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")

        X = self.prepare_features(spy_data, vix_data, breadth_data)

        if len(X) == 0:
            raise ValueError("Insufficient data to detect regime")

        X_last = X.iloc[[-1]]
        X_standardized = (X_last - pd.Series(self.feature_means)) / pd.Series(self.feature_stds)

        state_id = self.model.predict(X_standardized.values)[0]
        state_probs = self.model.predict_proba(X_standardized.values)[0]

        regime = self.state_labels.get(state_id, MarketRegime.NEUTRAL)
        confidence = float(state_probs[state_id])

        probabilities = {
            self.state_labels.get(i, MarketRegime.NEUTRAL): float(state_probs[i])
            for i in range(self.n_states)
        }

        transition_probs = self._get_transition_probabilities(state_id)
        days_in_regime = self._count_days_in_regime(X, regime)
        expected_duration = self._estimate_regime_duration(state_id)

        regime_state = RegimeState(
            regime=regime,
            confidence=confidence,
            probabilities=probabilities,
            transition_probs=transition_probs,
            days_in_regime=days_in_regime,
            expected_duration=expected_duration,
            feature_snapshot=X_last.iloc[0].to_dict(),
            timestamp=datetime.now()
        )

        self.current_regime = regime_state
        self.regime_history.append(regime_state)
        self.regime_sequence.append(state_id)

        return regime_state

    def _get_transition_probabilities(self, state_id: int) -> Dict[MarketRegime, float]:
        """Get probability of transitioning to each regime."""
        transition_probs = {}
        for next_state_id in range(self.n_states):
            next_regime = self.state_labels.get(next_state_id, MarketRegime.NEUTRAL)
            prob = float(self.model.transmat_[state_id, next_state_id])
            transition_probs[next_regime] = prob
        return transition_probs

    def _count_days_in_regime(self, features: pd.DataFrame, regime: MarketRegime) -> int:
        """Count consecutive days in current regime."""
        if len(self.regime_sequence) == 0:
            return 0

        current_state_id = None
        for regime_id, regime_label in self.state_labels.items():
            if regime_label == regime:
                current_state_id = regime_id
                break

        if current_state_id is None:
            return 0

        days = 0
        for state_id in reversed(self.regime_sequence):
            if state_id == current_state_id:
                days += 1
            else:
                break

        return days

    def _estimate_regime_duration(self, state_id: int) -> float:
        """Estimate expected duration of current regime."""
        p_stay = float(self.model.transmat_[state_id, state_id])
        if p_stay >= 1.0:
            return float('inf')
        return 1.0 / (1.0 - p_stay)

    def get_position_multiplier(
        self,
        regime_state: RegimeState,
        confidence_threshold: float = 0.6
    ) -> float:
        """
        Get position sizing multiplier based on regime and confidence.

        Strategy:
        - Bull + high confidence: 1.0 (full size)
        - Bull + low confidence: 0.75
        - Neutral: 0.5
        - Bear + high confidence: 0.25
        - Bear + low confidence: 0.0 (no new positions)
        """
        regime = regime_state.regime
        confidence = regime_state.confidence

        base_multipliers = {
            MarketRegime.BULLISH: 1.0,
            MarketRegime.NEUTRAL: 0.5,
            MarketRegime.BEARISH: 0.25
        }

        base_mult = base_multipliers.get(regime, 0.5)

        if confidence < confidence_threshold:
            confidence_factor = confidence / confidence_threshold
            adjusted_mult = base_mult * confidence_factor
        else:
            adjusted_mult = base_mult

        if regime == MarketRegime.BEARISH and confidence > 0.8:
            adjusted_mult = 0.0

        return max(0.0, min(1.0, adjusted_mult))

    def save_model(self, filepath: Optional[str] = None) -> str:
        """Save trained model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.cache_dir / f"hmm_regime_model_{timestamp}.pkl"
        else:
            filepath = Path(filepath)

        model_data = {
            'model': self.model,
            'state_labels': self.state_labels,
            'feature_names': self.feature_names,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'n_states': self.n_states,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")
        return str(filepath)

    def load_model(self, filepath: str) -> 'HMMRegimeDetector':
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.state_labels = model_data['state_labels']
        self.feature_names = model_data['feature_names']
        self.feature_means = model_data['feature_means']
        self.feature_stds = model_data['feature_stds']
        self.n_states = model_data['n_states']
        self.is_fitted = model_data['is_fitted']

        logger.info(f"Model loaded from {filepath}")
        return self


class AdaptiveRegimeDetector:
    """
    Combines rule-based and HMM regime detection for robustness.
    Falls back to rule-based if HMM unavailable or uncertain.
    """

    def __init__(
        self,
        hmm_weight: float = 0.7,
        rule_weight: float = 0.3,
        min_confidence: float = 0.6,
        use_hmm: bool = True
    ):
        self.hmm_weight = hmm_weight
        self.rule_weight = rule_weight
        self.min_confidence = min_confidence

        self.hmm_detector: Optional[HMMRegimeDetector] = None
        if use_hmm and HMM_AVAILABLE:
            try:
                self.hmm_detector = HMMRegimeDetector()
                logger.info("AdaptiveRegimeDetector: HMM mode enabled")
            except Exception as e:
                logger.warning(f"Could not initialize HMM: {e}. Using rule-based only.")
        else:
            logger.info("AdaptiveRegimeDetector: Rule-based mode only")

    def fit(
        self,
        spy_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        breadth_data: Optional[pd.DataFrame] = None
    ) -> 'AdaptiveRegimeDetector':
        """Train HMM component if available."""
        if self.hmm_detector is not None:
            self.hmm_detector.fit(spy_data, vix_data, breadth_data)
            logger.info("AdaptiveRegimeDetector: HMM component trained")
        return self

    def detect_regime(
        self,
        spy_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        breadth_data: Optional[pd.DataFrame] = None
    ) -> RegimeState:
        """Detect regime using ensemble of HMM and rules."""
        # Try HMM first
        if self.hmm_detector is not None and self.hmm_detector.is_fitted:
            hmm_regime_state = self.hmm_detector.detect_regime(spy_data, vix_data, breadth_data)

            if hmm_regime_state.confidence >= self.min_confidence:
                logger.debug(f"Using HMM regime: {hmm_regime_state.regime.value}")
                return hmm_regime_state
            else:
                rule_regime = self._rule_based_regime(spy_data, vix_data)
                if hmm_regime_state.regime == rule_regime:
                    return hmm_regime_state
                else:
                    return self._blend_regimes(hmm_regime_state, rule_regime)

        # Fallback: rule-based only
        logger.debug("Using rule-based regime detection")
        rule_regime = self._rule_based_regime(spy_data, vix_data)

        return RegimeState(
            regime=rule_regime,
            confidence=0.7,
            probabilities={
                MarketRegime.BULLISH: 1.0 if rule_regime == MarketRegime.BULLISH else 0.0,
                MarketRegime.NEUTRAL: 1.0 if rule_regime == MarketRegime.NEUTRAL else 0.0,
                MarketRegime.BEARISH: 1.0 if rule_regime == MarketRegime.BEARISH else 0.0
            },
            transition_probs={r: 0.33 for r in MarketRegime},
            days_in_regime=1,
            expected_duration=10.0,
            feature_snapshot={},
            timestamp=datetime.now()
        )

    def _rule_based_regime(self, spy_data: pd.DataFrame, vix_data: Optional[pd.DataFrame] = None) -> MarketRegime:
        """Simple rule-based regime detection."""
        # Normalize column names (handle lowercase 'close' from multi-source provider)
        spy_data = _normalize_columns(spy_data)
        spy_close = spy_data['Close']

        spy_price = spy_close.iloc[-1]
        spy_sma200 = spy_close.rolling(200).mean().iloc[-1]

        # Handle VIX data (use default of 20 if not available)
        if vix_data is not None and not vix_data.empty:
            vix_data = _normalize_columns(vix_data)
        if vix_data is not None and not vix_data.empty and 'Close' in vix_data.columns:
            vix_price = vix_data['Close'].iloc[-1]
        else:
            vix_price = 20.0  # Neutral VIX assumption

        if len(spy_close) >= 50:
            spy_momentum_50d = (spy_price / spy_close.iloc[-50] - 1) * 100
        else:
            spy_momentum_50d = 0.0

        if spy_price > spy_sma200 and vix_price < 20 and spy_momentum_50d > 0:
            return MarketRegime.BULLISH
        elif spy_price < spy_sma200 or vix_price > 30:
            return MarketRegime.BEARISH
        else:
            return MarketRegime.NEUTRAL

    def _blend_regimes(
        self,
        hmm_regime_state: RegimeState,
        rule_regime: MarketRegime
    ) -> RegimeState:
        """Blend HMM and rule-based regimes when they disagree."""
        blended_confidence = hmm_regime_state.confidence * 0.8

        adjusted_probs = hmm_regime_state.probabilities.copy()
        for regime in MarketRegime:
            if regime == rule_regime:
                adjusted_probs[regime] = (adjusted_probs[regime] + 0.3) / 1.3
            else:
                adjusted_probs[regime] = adjusted_probs[regime] / 1.3

        if hmm_regime_state.probabilities[hmm_regime_state.regime] > adjusted_probs[rule_regime]:
            final_regime = hmm_regime_state.regime
        else:
            final_regime = rule_regime

        return RegimeState(
            regime=final_regime,
            confidence=blended_confidence,
            probabilities=adjusted_probs,
            transition_probs=hmm_regime_state.transition_probs,
            days_in_regime=hmm_regime_state.days_in_regime,
            expected_duration=hmm_regime_state.expected_duration,
            feature_snapshot=hmm_regime_state.feature_snapshot,
            timestamp=datetime.now()
        )

    def get_position_multiplier(self, regime_state: RegimeState) -> float:
        """Get position sizing multiplier from regime state."""
        if self.hmm_detector is not None:
            return self.hmm_detector.get_position_multiplier(regime_state)
        else:
            multipliers = {
                MarketRegime.BULLISH: 1.0,
                MarketRegime.NEUTRAL: 0.5,
                MarketRegime.BEARISH: 0.25
            }
            return multipliers.get(regime_state.regime, 0.5)


def create_regime_detector(use_hmm: bool = True) -> AdaptiveRegimeDetector:
    """Factory function to create regime detector."""
    return AdaptiveRegimeDetector(use_hmm=use_hmm)
