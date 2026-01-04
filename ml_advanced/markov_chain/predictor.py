"""
Markov Chain Predictor for Trading Signals

Generates trading signals from Markov chain predictions.
Integrates 1st-order, higher-order, and stationary distribution
analysis into actionable buy/sell/hold signals.

Signal Logic:
- BUY: P(Up) > threshold AND current_state != Up
- SELL: P(Down) > threshold AND current_state != Down
- HOLD: Otherwise

Features:
- Multi-order prediction (combines 1st and 2nd order)
- Stationary distribution deviation scoring
- Signal confidence calculation
- Integration with existing signal confidence system

Usage:
    predictor = MarkovPredictor()
    predictor.fit(returns)

    prediction = predictor.predict(df, symbol="AAPL")
    # Returns: {"signal": "BUY", "prob_up": 0.65, "confidence": 0.72, ...}

Created: 2026-01-04
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .higher_order import HigherOrderMarkov
from .state_classifier import StateClassifier, compute_returns
from .stationary_dist import StationaryDistribution
from .transition_matrix import TransitionMatrix

logger = logging.getLogger(__name__)


@dataclass
class PredictorConfig:
    """Configuration for Markov predictor."""

    # Signal thresholds
    buy_threshold: float = 0.55  # P(Up) > this → BUY signal
    sell_threshold: float = 0.55  # P(Down) > this → SELL signal

    # Feature flags
    use_stationary: bool = True  # Include stationary deviation
    use_higher_order: bool = True  # Use 2nd order when available

    # State classification
    n_states: int = 3
    classification_method: str = "threshold"

    # Higher order settings
    max_order: int = 2
    min_samples_higher: int = 20

    # Confidence calculation
    confidence_weight_prob: float = 0.5  # Weight for transition probability
    confidence_weight_stat: float = 0.3  # Weight for stationary deviation
    confidence_weight_samples: float = 0.2  # Weight for sample size


class MarkovPrediction:
    """Result of a Markov chain prediction."""

    def __init__(
        self,
        signal: str,
        current_state: int,
        prob_up: float,
        prob_down: float,
        prob_flat: float,
        confidence: float,
        stationary_deviation: float = 0.0,
        order_used: int = 1,
        recent_states: Optional[List[int]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.signal = signal
        self.current_state = current_state
        self.prob_up = prob_up
        self.prob_down = prob_down
        self.prob_flat = prob_flat
        self.confidence = confidence
        self.stationary_deviation = stationary_deviation
        self.order_used = order_used
        self.recent_states = recent_states or []
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal": self.signal,
            "current_state": self.current_state,
            "prob_up": self.prob_up,
            "prob_down": self.prob_down,
            "prob_flat": self.prob_flat,
            "confidence": self.confidence,
            "stationary_deviation": self.stationary_deviation,
            "order_used": self.order_used,
            "recent_states": self.recent_states,
            **self.details,
        }

    def __repr__(self) -> str:
        return (
            f"MarkovPrediction(signal={self.signal}, "
            f"P(Up)={self.prob_up:.3f}, conf={self.confidence:.3f})"
        )


class MarkovPredictor:
    """
    Generate trading signals from Markov chain analysis.

    Combines multiple Markov chain components:
    1. First-order transition matrix (basic predictions)
    2. Higher-order chains (multi-day patterns)
    3. Stationary distribution (mean-reversion signals)

    The predictor generates signals with confidence scores that
    can be integrated into the existing signal quality system.

    Example:
        predictor = MarkovPredictor(buy_threshold=0.6)
        predictor.fit(historical_returns)

        # Get prediction for today
        pred = predictor.predict(current_df, symbol="AAPL")

        if pred.signal == "BUY" and pred.confidence > 0.7:
            execute_trade()
    """

    def __init__(
        self,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.55,
        use_stationary: bool = True,
        use_higher_order: bool = True,
        n_states: int = 3,
        classification_method: str = "threshold",
        max_order: int = 2,
    ):
        """
        Initialize Markov predictor.

        Args:
            buy_threshold: P(Up) threshold for BUY signal
            sell_threshold: P(Down) threshold for SELL signal
            use_stationary: Include stationary distribution analysis
            use_higher_order: Use higher-order chains when available
            n_states: Number of states (2, 3, or 5)
            classification_method: Method for state classification
            max_order: Maximum order for higher-order chains
        """
        self.config = PredictorConfig(
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            use_stationary=use_stationary,
            use_higher_order=use_higher_order,
            n_states=n_states,
            classification_method=classification_method,
            max_order=max_order,
        )

        # Components
        self.classifier = StateClassifier(
            n_states=n_states,
            method=classification_method,
        )
        self.transition_matrix = TransitionMatrix(n_states=n_states)
        self.higher_order: Optional[HigherOrderMarkov] = None
        self.stationary_dist = StationaryDistribution()

        # State
        self._fitted = False
        self._states: Optional[np.ndarray] = None
        self._pi: Optional[np.ndarray] = None

        logger.debug(f"MarkovPredictor initialized: thresholds=({buy_threshold}, {sell_threshold})")

    def fit(self, returns: Union[pd.Series, np.ndarray]) -> "MarkovPredictor":
        """
        Fit predictor on historical returns.

        Args:
            returns: Historical daily returns

        Returns:
            Self for method chaining
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        returns = returns.dropna()

        if len(returns) < 30:
            logger.warning(f"Insufficient data: {len(returns)} < 30")
            return self

        # Classify returns into states
        self.classifier.fit(returns)
        self._states = self.classifier.classify(returns)

        # Build first-order transition matrix
        self.transition_matrix.fit(self._states)

        # Build higher-order chain if enabled
        if self.config.use_higher_order and len(self._states) > self.config.max_order + 10:
            self.higher_order = HigherOrderMarkov(
                order=self.config.max_order,
                n_states=self.config.n_states,
            )
            self.higher_order.fit(self._states)

        # Compute stationary distribution
        if self.config.use_stationary:
            self._pi = self.stationary_dist.compute(self.transition_matrix.matrix)

        self._fitted = True
        logger.debug(f"Fitted on {len(returns)} returns, {self.transition_matrix.total_transitions} transitions")

        return self

    def predict(
        self,
        df: Optional[pd.DataFrame] = None,
        returns: Optional[pd.Series] = None,
        symbol: str = "",
    ) -> MarkovPrediction:
        """
        Generate prediction for current state.

        Args:
            df: DataFrame with OHLCV data (uses 'close' for returns)
            returns: Alternative: pass returns directly
            symbol: Symbol name for logging

        Returns:
            MarkovPrediction with signal, probabilities, confidence
        """
        if not self._fitted:
            logger.warning("Predictor not fitted, returning HOLD")
            return MarkovPrediction(
                signal="HOLD",
                current_state=-1,
                prob_up=0.33,
                prob_down=0.33,
                prob_flat=0.34,
                confidence=0.0,
            )

        # Get returns
        if returns is None and df is not None:
            returns = compute_returns(df)
        elif returns is None:
            raise ValueError("Must provide either df or returns")

        # Get recent states
        recent_returns = returns.dropna().tail(self.config.max_order + 1)
        recent_states = self.classifier.classify(recent_returns)

        if len(recent_states) == 0:
            return MarkovPrediction(
                signal="HOLD",
                current_state=-1,
                prob_up=0.33,
                prob_down=0.33,
                prob_flat=0.34,
                confidence=0.0,
            )

        current_state = int(recent_states[-1])

        # Get first-order prediction
        probs_1st = self.transition_matrix.predict_next(current_state)

        # Get higher-order prediction if available
        order_used = 1
        probs = probs_1st.copy()

        if (self.config.use_higher_order and
            self.higher_order is not None and
            len(recent_states) >= self.config.max_order):

            recent_tuple = tuple(recent_states[-self.config.max_order:])

            if self.higher_order.is_reliable(*recent_tuple):
                probs_higher = self.higher_order.predict(*recent_tuple)
                # Blend: weight higher-order more when confident
                probs = 0.4 * probs_1st + 0.6 * probs_higher
                order_used = self.config.max_order

        # Extract probabilities
        n = self.config.n_states
        prob_down = float(probs[0])
        prob_up = float(probs[n - 1])
        prob_flat = float(probs[1]) if n >= 3 else 0.0

        # Calculate stationary deviation
        stat_deviation = 0.0
        if self.config.use_stationary and self._pi is not None:
            # Positive deviation = current state below equilibrium
            stat_deviation = float(self._pi[n - 1] - (1.0 if current_state == n - 1 else 0.0))

        # Generate signal
        signal = self._generate_signal(
            current_state=current_state,
            prob_up=prob_up,
            prob_down=prob_down,
            stat_deviation=stat_deviation,
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            prob_up=prob_up,
            prob_down=prob_down,
            signal=signal,
            stat_deviation=stat_deviation,
        )

        return MarkovPrediction(
            signal=signal,
            current_state=current_state,
            prob_up=prob_up,
            prob_down=prob_down,
            prob_flat=prob_flat,
            confidence=confidence,
            stationary_deviation=stat_deviation,
            order_used=order_used,
            recent_states=recent_states.tolist(),
            details={
                "symbol": symbol,
                "probs_1st": probs_1st.tolist(),
                "pi": self._pi.tolist() if self._pi is not None else None,
            },
        )

    def _generate_signal(
        self,
        current_state: int,
        prob_up: float,
        prob_down: float,
        stat_deviation: float,
    ) -> str:
        """Generate BUY/SELL/HOLD signal."""
        n = self.config.n_states
        up_state = n - 1
        down_state = 0

        # BUY conditions:
        # 1. High probability of UP
        # 2. NOT already in UP state (avoid buying tops)
        if prob_up > self.config.buy_threshold and current_state != up_state:
            return "BUY"

        # SELL conditions:
        # 1. High probability of DOWN
        # 2. NOT already in DOWN state (avoid selling bottoms)
        if prob_down > self.config.sell_threshold and current_state != down_state:
            return "SELL"

        # Mean-reversion boost: if far below equilibrium, bias toward BUY
        if self.config.use_stationary and stat_deviation > 0.1:
            if prob_up > 0.45 and current_state == down_state:
                return "BUY"

        return "HOLD"

    def _calculate_confidence(
        self,
        prob_up: float,
        prob_down: float,
        signal: str,
        stat_deviation: float,
    ) -> float:
        """
        Calculate confidence score for the signal.

        Components:
        1. Probability strength (how far above threshold)
        2. Stationary deviation (mean-reversion support)
        3. Sample size reliability
        """
        cfg = self.config

        # Base confidence from probability
        if signal == "BUY":
            prob_confidence = min(1.0, (prob_up - 0.33) / 0.37)  # Scale to 0-1
        elif signal == "SELL":
            prob_confidence = min(1.0, (prob_down - 0.33) / 0.37)
        else:
            prob_confidence = 0.5  # Neutral for HOLD

        # Stationary deviation contribution
        if signal == "BUY":
            stat_confidence = max(0.0, min(1.0, stat_deviation * 5 + 0.5))
        elif signal == "SELL":
            stat_confidence = max(0.0, min(1.0, -stat_deviation * 5 + 0.5))
        else:
            stat_confidence = 0.5

        # Sample size contribution
        total = self.transition_matrix.total_transitions
        sample_confidence = min(1.0, total / 500)  # Max confidence at 500+ samples

        # Weighted combination
        confidence = (
            cfg.confidence_weight_prob * prob_confidence +
            cfg.confidence_weight_stat * stat_confidence +
            cfg.confidence_weight_samples * sample_confidence
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def score_signal(
        self,
        existing_signal: Dict[str, Any],
        df: Optional[pd.DataFrame] = None,
        returns: Optional[pd.Series] = None,
    ) -> float:
        """
        Score an existing signal based on Markov agreement.

        Use this to boost/penalize signals from other strategies
        based on Markov chain agreement.

        Args:
            existing_signal: Signal dict with 'side' key ("long" or "short")
            df: OHLCV DataFrame
            returns: Or pass returns directly

        Returns:
            Score from 0.0 (strong disagreement) to 1.0 (strong agreement)
        """
        prediction = self.predict(df=df, returns=returns)

        side = existing_signal.get("side", "long").lower()

        if side == "long":
            # Long signal - agreement based on P(Up)
            if prediction.signal == "BUY":
                return 0.7 + 0.3 * prediction.confidence  # Strong agreement
            elif prediction.signal == "HOLD":
                return 0.4 + 0.2 * prediction.prob_up  # Neutral
            else:
                return 0.3 * prediction.prob_up  # Disagreement

        else:  # short
            # Short signal - agreement based on P(Down)
            if prediction.signal == "SELL":
                return 0.7 + 0.3 * prediction.confidence
            elif prediction.signal == "HOLD":
                return 0.4 + 0.2 * prediction.prob_down
            else:
                return 0.3 * prediction.prob_down

    def get_bounce_probability(self, returns: pd.Series) -> Tuple[float, int]:
        """
        Get probability of bounce after recent down days.

        Args:
            returns: Recent returns

        Returns:
            Tuple of (bounce_probability, consecutive_down_days)
        """
        recent = returns.dropna().tail(5)
        states = self.classifier.classify(recent)

        # Count consecutive down days from end
        consecutive_down = 0
        for s in reversed(states):
            if s == 0:  # DOWN
                consecutive_down += 1
            else:
                break

        if consecutive_down == 0:
            return 0.33, 0  # Not in down streak

        # Use higher-order if available
        if (self.higher_order is not None and
            consecutive_down >= self.config.max_order):

            down_seq = tuple([0] * self.config.max_order)
            if self.higher_order.is_reliable(*down_seq):
                prob = self.higher_order.get_probability(*down_seq, next_state=self.config.n_states - 1)
                return float(prob), consecutive_down

        # Fall back to first-order
        prob = self.transition_matrix.get_probability(0, self.config.n_states - 1)
        return float(prob), consecutive_down

    @property
    def is_fitted(self) -> bool:
        """Whether predictor has been fitted."""
        return self._fitted

    def to_dict(self) -> Dict[str, Any]:
        """Serialize predictor state."""
        return {
            "config": {
                "buy_threshold": self.config.buy_threshold,
                "sell_threshold": self.config.sell_threshold,
                "use_stationary": self.config.use_stationary,
                "use_higher_order": self.config.use_higher_order,
                "n_states": self.config.n_states,
                "classification_method": self.config.classification_method,
                "max_order": self.config.max_order,
            },
            "classifier": self.classifier.to_dict(),
            "transition_matrix": self.transition_matrix.to_dict(),
            "higher_order": self.higher_order.to_dict() if self.higher_order else None,
            "pi": self._pi.tolist() if self._pi is not None else None,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MarkovPredictor":
        """Deserialize predictor from dict."""
        cfg = data["config"]
        predictor = cls(
            buy_threshold=cfg["buy_threshold"],
            sell_threshold=cfg["sell_threshold"],
            use_stationary=cfg["use_stationary"],
            use_higher_order=cfg["use_higher_order"],
            n_states=cfg["n_states"],
            classification_method=cfg["classification_method"],
            max_order=cfg["max_order"],
        )

        predictor.classifier = StateClassifier.from_dict(data["classifier"])
        predictor.transition_matrix = TransitionMatrix.from_dict(data["transition_matrix"])

        if data.get("higher_order"):
            predictor.higher_order = HigherOrderMarkov.from_dict(data["higher_order"])

        if data.get("pi"):
            predictor._pi = np.array(data["pi"])

        predictor._fitted = data.get("fitted", False)

        return predictor


def predict_batch(
    returns_dict: Dict[str, pd.Series],
    predictor: Optional[MarkovPredictor] = None,
    fit_each: bool = True,
) -> pd.DataFrame:
    """
    Generate predictions for multiple symbols.

    Args:
        returns_dict: Dict mapping symbol to returns series
        predictor: Optional shared predictor (will fit on each if fit_each=True)
        fit_each: Whether to fit predictor on each symbol's data

    Returns:
        DataFrame with predictions for each symbol
    """
    if predictor is None:
        predictor = MarkovPredictor()

    results = []

    for symbol, returns in returns_dict.items():
        try:
            if fit_each:
                predictor.fit(returns)

            pred = predictor.predict(returns=returns, symbol=symbol)
            result = pred.to_dict()
            result["symbol"] = symbol
            results.append(result)

        except Exception as e:
            logger.warning(f"Failed to predict for {symbol}: {e}")

    return pd.DataFrame(results)
