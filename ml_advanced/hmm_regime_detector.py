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

from typing import Any, Dict, List, Optional, Tuple
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

        # Staleness tracking for auto-retrain
        self.last_train_timestamp: Optional[datetime] = None
        self.staleness_threshold_days: int = 30  # Retrain after 30 days
        self.training_window_days: int = 504  # ~2 years of trading days

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"HMMRegimeDetector initialized (n_states={n_states})")

    # Historical VIX statistics (1990-2024) for fallback estimation
    HISTORICAL_VIX_MEAN = 19.5
    HISTORICAL_VIX_MEDIAN = 17.5
    HISTORICAL_VIX_PERCENTILE_75 = 23.0

    def _estimate_vix_from_realized_vol(
        self,
        spy_data: pd.DataFrame,
        lookback: int = 21
    ) -> float:
        """Estimate implied VIX from realized SPY volatility.

        Empirical relationship: VIX ≈ 0.9 * RealizedVol * sqrt(252) * 100 + 2
        (VIX typically trades at a premium to realized vol)

        Args:
            spy_data: SPY OHLCV data
            lookback: Days for realized volatility calculation (default: 21)

        Returns:
            Estimated VIX level (clamped to [10, 80])
        """
        spy_data = _normalize_columns(spy_data)
        if 'Close' not in spy_data.columns:
            return self.HISTORICAL_VIX_MEAN

        returns = spy_data['Close'].pct_change().dropna()

        if len(returns) < lookback:
            return self.HISTORICAL_VIX_MEAN

        realized_vol = returns.iloc[-lookback:].std() * np.sqrt(252) * 100
        estimated_vix = realized_vol * 0.9 + 2.0  # VIX premium adjustment

        # Clamp to reasonable range [10, 80]
        return max(10.0, min(80.0, estimated_vix))

    def _get_vix_with_fallback(
        self,
        vix_data: Optional[pd.DataFrame],
        spy_data: pd.DataFrame
    ) -> float:
        """Get VIX level with intelligent fallback chain.

        Fallback order:
        1. Use provided VIX data if valid
        2. Estimate from SPY realized volatility
        3. Use historical VIX mean as last resort

        Args:
            vix_data: VIX OHLCV data (may be None or empty)
            spy_data: SPY OHLCV data for realized vol estimation

        Returns:
            VIX level (actual or estimated)
        """
        # Try 1: Use provided VIX data
        if vix_data is not None and not vix_data.empty:
            vix_data = _normalize_columns(vix_data)
            if 'Close' in vix_data.columns and len(vix_data) > 0:
                vix_level = float(vix_data['Close'].iloc[-1])
                if not np.isnan(vix_level) and vix_level > 0:
                    return vix_level

        # Try 2: Estimate from realized volatility
        estimated_vix = self._estimate_vix_from_realized_vol(spy_data)

        logger.warning(
            f"VIX data unavailable. Using estimated VIX={estimated_vix:.1f} "
            f"(from realized vol)"
        )

        return estimated_vix

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

    def is_stale(self) -> bool:
        """Check if model needs retraining.

        Returns True if:
        - Model is not fitted
        - last_train_timestamp is None
        - More than staleness_threshold_days have passed since training
        """
        if not self.is_fitted or self.last_train_timestamp is None:
            return True
        days_since_training = (datetime.now() - self.last_train_timestamp).days
        return days_since_training > self.staleness_threshold_days

    def retrain_if_stale(
        self,
        spy_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        breadth_data: Optional[pd.DataFrame] = None,
        force: bool = False
    ) -> bool:
        """Retrain model if stale. Returns True if retrained.

        Uses a sliding window of training_window_days (~2 years) for training
        to keep the model adapted to recent market conditions.

        Args:
            spy_data: SPY OHLCV data
            vix_data: VIX OHLCV data
            breadth_data: Optional market breadth data
            force: Force retrain even if not stale

        Returns:
            True if model was retrained, False otherwise
        """
        if not force and not self.is_stale():
            logger.debug(f"Model not stale (trained {self.last_train_timestamp})")
            return False

        # Use only recent data (sliding window)
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=self.training_window_days)

        # Filter to recent data if index is datetime
        if hasattr(spy_data.index, 'tz_localize') or isinstance(spy_data.index, pd.DatetimeIndex):
            spy_recent = spy_data[spy_data.index >= cutoff]
            vix_recent = vix_data[vix_data.index >= cutoff] if vix_data is not None else None
        else:
            # If no datetime index, use all data
            spy_recent = spy_data
            vix_recent = vix_data

        logger.info(f"Retraining HMM (stale={self.is_stale()}, force={force}, samples={len(spy_recent)})")

        self.fit(spy_recent, vix_recent, breadth_data)
        self.last_train_timestamp = datetime.now()

        # Save model with timestamp
        model_path = self.save_model()

        # Log retrain event
        try:
            from core.structured_log import jlog
            jlog('hmm_retrain',
                 model_path=model_path,
                 training_samples=len(spy_recent),
                 staleness_days=self.staleness_threshold_days,
                 forced=force)
        except ImportError:
            pass  # Graceful degradation if jlog not available

        return True

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

    def _log_regime_transition(
        self,
        previous_regime: Optional[RegimeState],
        current_regime: RegimeState
    ) -> None:
        """Log regime transition to structured event log.

        Logs detailed information about regime state and transitions
        to logs/events.jsonl for post-hoc analysis and audit trails.
        """
        try:
            from core.structured_log import jlog
        except ImportError:
            return  # Graceful degradation

        is_transition = (
            previous_regime is not None and
            previous_regime.regime != current_regime.regime
        )

        event_data = {
            'regime': current_regime.regime.value,
            'confidence': round(current_regime.confidence, 4),
            'days_in_regime': current_regime.days_in_regime,
            'expected_duration': round(current_regime.expected_duration, 1),
            'is_transition': is_transition,
        }

        # Add probabilities
        for regime, prob in current_regime.probabilities.items():
            event_data[f'prob_{regime.value.lower()}'] = round(prob, 4)

        # Add transition probs
        for regime, prob in current_regime.transition_probs.items():
            event_data[f'trans_to_{regime.value.lower()}'] = round(prob, 4)

        # Add feature snapshot
        for feature, value in current_regime.feature_snapshot.items():
            if isinstance(value, float):
                event_data[f'feat_{feature}'] = round(value, 4)
            else:
                event_data[f'feat_{feature}'] = value

        if is_transition:
            event_data['previous_regime'] = previous_regime.regime.value
            event_data['previous_confidence'] = round(previous_regime.confidence, 4)
            jlog('regime_transition', **event_data)
            logger.info(
                f"REGIME TRANSITION: {previous_regime.regime.value} -> "
                f"{current_regime.regime.value} (conf={current_regime.confidence:.2f})"
            )
        else:
            jlog('regime_update', **event_data)

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

        # Store previous for transition detection
        previous_regime = self.current_regime

        self.current_regime = regime_state
        self.regime_history.append(regime_state)
        self.regime_sequence.append(state_id)

        # Log regime transition/update to events.jsonl
        self._log_regime_transition(previous_regime, regime_state)

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

    def forecast_regime(
        self,
        days_ahead: int = 5,
        current_state: Optional[RegimeState] = None
    ) -> Dict[str, Any]:
        """Forecast regime probabilities N days ahead.

        Uses Chapman-Kolmogorov theorem: P(t+n) = P(t) @ A^n
        where A is the transition matrix.

        Args:
            days_ahead: Number of days to forecast (default 5)
            current_state: Current regime state (uses self.current_regime if None)

        Returns:
            Dict with:
            - 'forecast_probs': {MarketRegime: probability}
            - 'most_likely_regime': MarketRegime
            - 'confidence': float (probability of most likely)
            - 'regime_change_prob': float (probability of leaving current regime)
            - 'days_ahead': int
            - 'current_regime': MarketRegime
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")

        state = current_state or self.current_regime
        if state is None:
            raise ValueError("No current regime. Call detect_regime() first.")

        # Get current state index
        current_state_id = None
        for state_id, regime in self.state_labels.items():
            if regime == state.regime:
                current_state_id = state_id
                break

        if current_state_id is None:
            raise ValueError(f"Unknown regime: {state.regime}")

        # Compute transition matrix power: A^n (Chapman-Kolmogorov)
        trans_matrix = self.model.transmat_
        trans_matrix_n = np.linalg.matrix_power(trans_matrix, days_ahead)

        # Get probabilities from current state row
        future_probs = trans_matrix_n[current_state_id, :]

        # Map to regimes
        forecast_probs = {}
        for state_id, regime in self.state_labels.items():
            forecast_probs[regime] = float(future_probs[state_id])

        # Find most likely regime
        most_likely_id = int(np.argmax(future_probs))
        most_likely_regime = self.state_labels.get(most_likely_id, MarketRegime.NEUTRAL)
        confidence = float(future_probs[most_likely_id])

        # Probability of leaving current regime
        stay_prob = float(trans_matrix_n[current_state_id, current_state_id])
        change_prob = 1.0 - stay_prob

        return {
            'forecast_probs': forecast_probs,
            'most_likely_regime': most_likely_regime,
            'confidence': confidence,
            'regime_change_prob': change_prob,
            'days_ahead': days_ahead,
            'current_regime': state.regime,
        }

    def get_forecast_adjusted_multiplier(
        self,
        regime_state: RegimeState,
        forecast_days: int = 5
    ) -> float:
        """Get position multiplier adjusted for regime forecast.

        If regime likely to worsen in next N days, reduce position size proactively.
        This allows the system to reduce exposure BEFORE a regime change occurs.

        Args:
            regime_state: Current regime state
            forecast_days: Number of days to look ahead (default 5)

        Returns:
            Adjusted position multiplier (0.0 to 1.0)
        """
        base_mult = self.get_position_multiplier(regime_state)

        try:
            forecast = self.forecast_regime(days_ahead=forecast_days, current_state=regime_state)
        except Exception:
            return base_mult  # Fall back to base if forecast fails

        # If high probability of transitioning to worse regime, reduce size
        current = regime_state.regime
        future = forecast['most_likely_regime']
        change_prob = forecast['regime_change_prob']

        # Regime ordering: BULLISH > NEUTRAL > BEARISH
        regime_order = {MarketRegime.BULLISH: 2, MarketRegime.NEUTRAL: 1, MarketRegime.BEARISH: 0}

        current_score = regime_order.get(current, 1)
        future_score = regime_order.get(future, 1)

        if future_score < current_score and change_prob > 0.3:
            # Regime likely to worsen - reduce position proactively
            reduction = change_prob * 0.5  # Up to 50% reduction
            adjusted = base_mult * (1.0 - reduction)
            logger.info(
                f"Forecast adjustment: {current.value} -> {future.value} "
                f"(prob={change_prob:.2f}), mult {base_mult:.2f} -> {adjusted:.2f}"
            )
            return adjusted

        return base_mult

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
            'is_fitted': self.is_fitted,
            # Staleness tracking
            'last_train_timestamp': self.last_train_timestamp,
            'training_window_days': self.training_window_days,
            'staleness_threshold_days': self.staleness_threshold_days,
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

        # Restore staleness tracking (with defaults for older models)
        self.last_train_timestamp = model_data.get('last_train_timestamp')
        self.training_window_days = model_data.get('training_window_days', 504)
        self.staleness_threshold_days = model_data.get('staleness_threshold_days', 30)

        logger.info(f"Model loaded from {filepath} (trained={self.last_train_timestamp})")
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

        # Handle VIX data with smart fallback
        vix_price = None
        if vix_data is not None and not vix_data.empty:
            vix_data = _normalize_columns(vix_data)
            if 'Close' in vix_data.columns and len(vix_data) > 0:
                vix_price = float(vix_data['Close'].iloc[-1])
                if np.isnan(vix_price) or vix_price <= 0:
                    vix_price = None

        # Smart fallback: estimate from realized volatility
        if vix_price is None:
            returns = spy_close.pct_change().dropna()
            if len(returns) >= 21:
                realized_vol = returns.iloc[-21:].std() * np.sqrt(252) * 100
                vix_price = realized_vol * 0.9 + 2.0  # VIX premium
                vix_price = max(10.0, min(80.0, vix_price))
            else:
                vix_price = 19.5  # Historical mean fallback

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
