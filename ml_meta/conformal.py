"""
Conformal Prediction Framework.

Provides distribution-free uncertainty quantification for trading predictions.
Conformal prediction creates prediction intervals with guaranteed coverage
regardless of the underlying model or data distribution.

Key concepts:
- Non-conformity score: How "unusual" a prediction is
- Coverage guarantee: If we want 90% coverage, ~90% of true values will be in intervals
- Uncertainty score: Higher = less confident = smaller position size

Usage:
    from ml_meta.conformal import ConformalPredictor

    # Fit on validation residuals
    predictor = ConformalPredictor(target_coverage=0.90)
    predictor.fit(val_predictions, val_actuals)

    # Get uncertainty for new predictions
    result = predictor.predict_with_uncertainty(new_prediction)
    print(f"Prediction: {result.prediction}, Uncertainty: {result.uncertainty_score}")

    # Use uncertainty for position sizing
    position_size = base_size * (1 - result.uncertainty_score * scale_factor)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ConformalResult:
    """Result of conformal prediction."""
    prediction: float
    lower_bound: float
    upper_bound: float
    uncertainty_score: float  # 0 = very certain, 1 = very uncertain
    interval_width: float
    coverage_target: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConformalStats:
    """Statistics about conformal predictor performance."""
    n_calibration_samples: int
    threshold: float
    target_coverage: float
    empirical_coverage: Optional[float]
    avg_interval_width: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Conformal Predictor
# ============================================================================

class ConformalPredictor:
    """
    Split conformal prediction for uncertainty quantification.

    Uses non-conformity scores (absolute residuals) from a calibration set
    to construct prediction intervals with guaranteed coverage.

    For trading:
    - High uncertainty → reduce position size
    - Low uncertainty → maintain or increase position size
    """

    def __init__(
        self,
        target_coverage: float = 0.90,
        uncertainty_scale: float = 0.5,
    ):
        """
        Initialize conformal predictor.

        Args:
            target_coverage: Desired coverage probability (e.g., 0.90 for 90%)
            uncertainty_scale: Max position reduction from uncertainty (0.5 = 50% max reduction)
        """
        if not 0 < target_coverage < 1:
            raise ValueError("target_coverage must be between 0 and 1")

        self.target_coverage = target_coverage
        self.uncertainty_scale = uncertainty_scale

        self._nonconformity_scores: Optional[np.ndarray] = None
        self._threshold: Optional[float] = None
        self._fitted = False

        # Track performance
        self._predictions_made = 0
        self._interval_widths: List[float] = []

    def fit(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> 'ConformalPredictor':
        """
        Fit the conformal predictor on calibration data.

        Computes non-conformity scores (residuals) and determines
        the threshold for the target coverage level.

        Args:
            predictions: Model predictions on calibration set
            actuals: Actual values on calibration set

        Returns:
            self for chaining
        """
        predictions = np.asarray(predictions).flatten()
        actuals = np.asarray(actuals).flatten()

        if len(predictions) != len(actuals):
            raise ValueError("predictions and actuals must have same length")

        if len(predictions) < 10:
            logger.warning("Conformal predictor needs at least 10 samples")
            self._fitted = False
            return self

        # Compute non-conformity scores (absolute residuals)
        self._nonconformity_scores = np.abs(predictions - actuals)

        # Compute threshold at (1 - alpha) quantile
        # For 90% coverage, we use the 90th percentile of residuals
        1 - self.target_coverage
        n = len(self._nonconformity_scores)

        # Finite sample correction
        q_level = np.ceil((n + 1) * self.target_coverage) / n
        q_level = min(q_level, 1.0)

        self._threshold = float(np.quantile(self._nonconformity_scores, q_level))
        self._fitted = True

        logger.info(
            f"ConformalPredictor fitted: n={n}, coverage={self.target_coverage:.0%}, "
            f"threshold={self._threshold:.4f}"
        )
        return self

    def predict_with_uncertainty(
        self,
        prediction: float,
        return_interval: bool = True,
    ) -> ConformalResult:
        """
        Get prediction with uncertainty quantification.

        Args:
            prediction: Point prediction from model
            return_interval: Whether to compute prediction interval

        Returns:
            ConformalResult with prediction, bounds, and uncertainty score
        """
        if not self._fitted:
            # Return default uncertainty if not fitted
            return ConformalResult(
                prediction=prediction,
                lower_bound=prediction - 0.1,
                upper_bound=prediction + 0.1,
                uncertainty_score=0.5,  # Default moderate uncertainty
                interval_width=0.2,
                coverage_target=self.target_coverage,
            )

        # Prediction interval using threshold
        lower = prediction - self._threshold
        upper = prediction + self._threshold
        interval_width = 2 * self._threshold

        # Compute uncertainty score based on how wide the interval is
        # relative to the prediction magnitude
        if abs(prediction) > 0.01:
            relative_width = interval_width / abs(prediction)
        else:
            relative_width = interval_width

        # Map to [0, 1] uncertainty score
        # Higher relative width = higher uncertainty
        uncertainty_score = min(1.0, relative_width / 2.0)

        # Track for statistics
        self._predictions_made += 1
        self._interval_widths.append(interval_width)

        return ConformalResult(
            prediction=prediction,
            lower_bound=lower,
            upper_bound=upper,
            uncertainty_score=uncertainty_score,
            interval_width=interval_width,
            coverage_target=self.target_coverage,
        )

    def batch_predict_with_uncertainty(
        self,
        predictions: np.ndarray,
    ) -> List[ConformalResult]:
        """
        Get uncertainty for multiple predictions.

        Args:
            predictions: Array of point predictions

        Returns:
            List of ConformalResult objects
        """
        predictions = np.asarray(predictions).flatten()
        return [self.predict_with_uncertainty(p) for p in predictions]

    def compute_position_multiplier(
        self,
        uncertainty_score: float,
    ) -> float:
        """
        Compute position size multiplier based on uncertainty.

        Higher uncertainty → lower multiplier → smaller position.

        Args:
            uncertainty_score: Uncertainty from predict_with_uncertainty()

        Returns:
            Multiplier in range [1 - uncertainty_scale, 1.0]
        """
        # Linear scaling: uncertainty 0 → mult 1.0, uncertainty 1 → mult (1 - scale)
        multiplier = 1.0 - (uncertainty_score * self.uncertainty_scale)
        return max(0.5, min(1.0, multiplier))  # Clip to [0.5, 1.0]

    def get_stats(self) -> ConformalStats:
        """Get statistics about the conformal predictor."""
        return ConformalStats(
            n_calibration_samples=len(self._nonconformity_scores) if self._nonconformity_scores is not None else 0,
            threshold=self._threshold if self._threshold is not None else 0.0,
            target_coverage=self.target_coverage,
            empirical_coverage=None,  # Computed post-hoc
            avg_interval_width=float(np.mean(self._interval_widths)) if self._interval_widths else 0.0,
            timestamp=datetime.utcnow().isoformat(),
        )

    def evaluate_coverage(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Evaluate actual coverage on test data.

        Args:
            predictions: Model predictions
            actuals: Actual values

        Returns:
            Tuple of (coverage_rate, avg_interval_width)
        """
        predictions = np.asarray(predictions).flatten()
        actuals = np.asarray(actuals).flatten()

        if not self._fitted:
            return 0.0, 0.0

        # Check how many actuals fall within prediction intervals
        results = self.batch_predict_with_uncertainty(predictions)

        covered = 0
        total_width = 0.0

        for result, actual in zip(results, actuals):
            if result.lower_bound <= actual <= result.upper_bound:
                covered += 1
            total_width += result.interval_width

        n = len(predictions)
        coverage_rate = covered / n if n > 0 else 0.0
        avg_width = total_width / n if n > 0 else 0.0

        return coverage_rate, avg_width

    def save(self, path: Path) -> None:
        """Save conformal predictor state to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'target_coverage': self.target_coverage,
            'uncertainty_scale': self.uncertainty_scale,
            'nonconformity_scores': self._nonconformity_scores.tolist() if self._nonconformity_scores is not None else None,
            'threshold': self._threshold,
            'fitted': self._fitted,
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved ConformalPredictor to {path}")

    @classmethod
    def load(cls, path: Path) -> 'ConformalPredictor':
        """Load conformal predictor from file."""
        with open(path, 'r') as f:
            state = json.load(f)

        predictor = cls(
            target_coverage=state['target_coverage'],
            uncertainty_scale=state['uncertainty_scale'],
        )

        if state['nonconformity_scores'] is not None:
            predictor._nonconformity_scores = np.array(state['nonconformity_scores'])

        predictor._threshold = state['threshold']
        predictor._fitted = state['fitted']

        logger.info(f"Loaded ConformalPredictor from {path}")
        return predictor

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ============================================================================
# Adaptive Conformal Predictor
# ============================================================================

class AdaptiveConformalPredictor(ConformalPredictor):
    """
    Conformal predictor with adaptive threshold adjustment.

    Tracks recent coverage and adjusts threshold to maintain
    target coverage over time (handles distribution shift).
    """

    def __init__(
        self,
        target_coverage: float = 0.90,
        uncertainty_scale: float = 0.5,
        window_size: int = 100,
        adjustment_rate: float = 0.01,
    ):
        super().__init__(target_coverage, uncertainty_scale)
        self.window_size = window_size
        self.adjustment_rate = adjustment_rate

        # Track recent predictions for adaptive adjustment
        self._recent_covered: List[bool] = []
        self._recent_actuals: List[float] = []
        self._recent_predictions: List[float] = []

    def update(
        self,
        prediction: float,
        actual: float,
    ) -> None:
        """
        Update with new observation for adaptive adjustment.

        Args:
            prediction: Model prediction
            actual: Actual value that was observed
        """
        if not self._fitted:
            return

        # Check if actual was covered
        result = self.predict_with_uncertainty(prediction)
        covered = result.lower_bound <= actual <= result.upper_bound

        self._recent_covered.append(covered)
        self._recent_actuals.append(actual)
        self._recent_predictions.append(prediction)

        # Maintain window size
        if len(self._recent_covered) > self.window_size:
            self._recent_covered.pop(0)
            self._recent_actuals.pop(0)
            self._recent_predictions.pop(0)

        # Adjust threshold if we have enough samples
        if len(self._recent_covered) >= 20:
            recent_coverage = sum(self._recent_covered) / len(self._recent_covered)

            if recent_coverage < self.target_coverage:
                # Under-coverage: increase threshold (wider intervals)
                self._threshold *= (1 + self.adjustment_rate)
            elif recent_coverage > self.target_coverage + 0.05:
                # Over-coverage: decrease threshold (tighter intervals)
                self._threshold *= (1 - self.adjustment_rate)

            logger.debug(
                f"AdaptiveConformal: coverage={recent_coverage:.2%}, "
                f"threshold adjusted to {self._threshold:.4f}"
            )


# ============================================================================
# Global State
# ============================================================================

_global_conformal: Optional[ConformalPredictor] = None


def set_global_conformal(predictor: ConformalPredictor) -> None:
    """Set the global conformal predictor for use across modules."""
    global _global_conformal
    _global_conformal = predictor


def get_global_conformal() -> Optional[ConformalPredictor]:
    """Get the global conformal predictor if set."""
    return _global_conformal


def get_uncertainty_score(prediction: float) -> float:
    """
    Get uncertainty score for a prediction using global conformal predictor.

    If no predictor is set, returns moderate uncertainty (0.3).

    Args:
        prediction: Model prediction

    Returns:
        Uncertainty score [0, 1]
    """
    if _global_conformal is None or not _global_conformal.is_fitted:
        return 0.3  # Default moderate uncertainty

    result = _global_conformal.predict_with_uncertainty(prediction)
    return result.uncertainty_score


def get_position_multiplier(prediction: float) -> float:
    """
    Get position size multiplier based on uncertainty.

    If no predictor is set, returns 1.0 (no adjustment).

    Args:
        prediction: Model prediction

    Returns:
        Position multiplier [0.5, 1.0]
    """
    if _global_conformal is None or not _global_conformal.is_fitted:
        return 1.0

    uncertainty = get_uncertainty_score(prediction)
    return _global_conformal.compute_position_multiplier(uncertainty)
