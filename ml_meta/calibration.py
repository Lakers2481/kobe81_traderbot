"""
Probability Calibration Framework.

Provides calibration methods to improve the reliability of probability predictions.
Calibrated probabilities ensure that when the model predicts 80% confidence,
approximately 80% of those trades should actually win.

Methods:
- IsotonicCalibrator: Non-parametric monotonic calibration
- PlattCalibrator: Parametric (logistic) calibration
- CalibrationMetrics: ECE, MCE, Brier score calculation

Usage:
    from ml_meta.calibration import IsotonicCalibrator, compute_brier_score

    # Fit calibrator on validation data
    calibrator = IsotonicCalibrator()
    calibrator.fit(val_probs, val_outcomes)

    # Calibrate new predictions
    calibrated_probs = calibrator.calibrate(raw_probs)

    # Compute metrics
    brier = compute_brier_score(calibrated_probs, outcomes)
    ece = compute_expected_calibration_error(calibrated_probs, outcomes)
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# Try importing sklearn for calibration
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    IsotonicRegression = None
    LogisticRegression = None


# ============================================================================
# Calibration Metrics
# ============================================================================

def compute_brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Compute Brier score (mean squared error of probabilities).

    Lower is better. Range: [0, 1]
    - 0.0 = perfect predictions
    - 0.25 = baseline for 50/50 random predictions

    Args:
        probabilities: Predicted probabilities (0 to 1)
        outcomes: Actual binary outcomes (0 or 1)

    Returns:
        Brier score
    """
    probabilities = np.asarray(probabilities).flatten()
    outcomes = np.asarray(outcomes).flatten()

    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have same length")

    if len(probabilities) == 0:
        return 0.0

    return float(np.mean((probabilities - outcomes) ** 2))


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the average gap between predicted confidence and actual accuracy.
    Lower is better. Range: [0, 1]

    Args:
        probabilities: Predicted probabilities (0 to 1)
        outcomes: Actual binary outcomes (0 or 1)
        n_bins: Number of bins for grouping predictions

    Returns:
        ECE value
    """
    probabilities = np.asarray(probabilities).flatten()
    outcomes = np.asarray(outcomes).flatten()

    if len(probabilities) == 0:
        return 0.0

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(probabilities)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find samples in this bin
        if i == n_bins - 1:
            # Include upper boundary in last bin
            in_bin = (probabilities >= bin_lower) & (probabilities <= bin_upper)
        else:
            in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size == 0:
            continue

        # Compute average confidence and accuracy in this bin
        bin_confidence = np.mean(probabilities[in_bin])
        bin_accuracy = np.mean(outcomes[in_bin])

        # Weighted contribution to ECE
        ece += (bin_size / total_samples) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_max_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE measures the worst-case gap between predicted confidence and actual accuracy.

    Args:
        probabilities: Predicted probabilities (0 to 1)
        outcomes: Actual binary outcomes (0 or 1)
        n_bins: Number of bins for grouping predictions

    Returns:
        MCE value
    """
    probabilities = np.asarray(probabilities).flatten()
    outcomes = np.asarray(outcomes).flatten()

    if len(probabilities) == 0:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_error = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (probabilities >= bin_lower) & (probabilities <= bin_upper)
        else:
            in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size == 0:
            continue

        bin_confidence = np.mean(probabilities[in_bin])
        bin_accuracy = np.mean(outcomes[in_bin])
        error = abs(bin_accuracy - bin_confidence)
        max_error = max(max_error, error)

    return float(max_error)


def compute_reliability_diagram(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, List[float]]:
    """
    Compute reliability diagram data (for visualization).

    Returns bin midpoints, accuracies, and counts for plotting.

    Args:
        probabilities: Predicted probabilities (0 to 1)
        outcomes: Actual binary outcomes (0 or 1)
        n_bins: Number of bins

    Returns:
        Dict with 'bin_midpoints', 'accuracies', 'counts', 'confidences'
    """
    probabilities = np.asarray(probabilities).flatten()
    outcomes = np.asarray(outcomes).flatten()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_midpoints = []
    accuracies = []
    confidences = []
    counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        midpoint = (bin_lower + bin_upper) / 2

        if i == n_bins - 1:
            in_bin = (probabilities >= bin_lower) & (probabilities <= bin_upper)
        else:
            in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)

        bin_size = int(np.sum(in_bin))
        bin_midpoints.append(midpoint)
        counts.append(bin_size)

        if bin_size > 0:
            accuracies.append(float(np.mean(outcomes[in_bin])))
            confidences.append(float(np.mean(probabilities[in_bin])))
        else:
            accuracies.append(None)
            confidences.append(None)

    return {
        'bin_midpoints': bin_midpoints,
        'accuracies': accuracies,
        'confidences': confidences,
        'counts': counts,
    }


# ============================================================================
# Calibration Result
# ============================================================================

@dataclass
class CalibrationResult:
    """Result of calibration metrics computation."""
    brier_score: float
    ece: float
    mce: float
    n_samples: int
    reliability_diagram: Dict[str, List[float]]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_predictions(
        cls,
        probabilities: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> 'CalibrationResult':
        """Compute all calibration metrics from predictions."""
        return cls(
            brier_score=compute_brier_score(probabilities, outcomes),
            ece=compute_expected_calibration_error(probabilities, outcomes, n_bins),
            mce=compute_max_calibration_error(probabilities, outcomes, n_bins),
            n_samples=len(probabilities),
            reliability_diagram=compute_reliability_diagram(probabilities, outcomes, n_bins),
            timestamp=datetime.utcnow().isoformat(),
        )


# ============================================================================
# Calibrators
# ============================================================================

class IsotonicCalibrator:
    """
    Isotonic regression calibrator.

    Non-parametric method that fits a monotonically increasing function
    to map raw probabilities to calibrated probabilities.

    Best for: Large datasets, when monotonicity is desired.
    """

    def __init__(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for IsotonicCalibrator")
        self._model = IsotonicRegression(out_of_bounds='clip')
        self._fitted = False

    def fit(self, probabilities: np.ndarray, outcomes: np.ndarray) -> 'IsotonicCalibrator':
        """
        Fit the calibrator on validation data.

        Args:
            probabilities: Raw predicted probabilities
            outcomes: Actual binary outcomes (0 or 1)

        Returns:
            self for chaining
        """
        probabilities = np.asarray(probabilities).flatten()
        outcomes = np.asarray(outcomes).flatten()

        if len(probabilities) < 10:
            logger.warning("Isotonic calibration needs at least 10 samples")
            self._fitted = False
            return self

        self._model.fit(probabilities, outcomes)
        self._fitted = True
        logger.info(f"IsotonicCalibrator fitted on {len(probabilities)} samples")
        return self

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calibrate raw probabilities.

        Args:
            probabilities: Raw predicted probabilities

        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            logger.warning("Calibrator not fitted, returning raw probabilities")
            return np.asarray(probabilities)

        probabilities = np.asarray(probabilities).flatten()
        calibrated = self._model.predict(probabilities)
        # Clip to [0, 1] range
        return np.clip(calibrated, 0, 1)

    def save(self, path: Path) -> None:
        """Save calibrator to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'model': self._model, 'fitted': self._fitted}, f)
        logger.info(f"Saved IsotonicCalibrator to {path}")

    @classmethod
    def load(cls, path: Path) -> 'IsotonicCalibrator':
        """Load calibrator from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        cal = cls()
        cal._model = data['model']
        cal._fitted = data['fitted']
        logger.info(f"Loaded IsotonicCalibrator from {path}")
        return cal

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class PlattCalibrator:
    """
    Platt scaling calibrator (logistic regression).

    Parametric method that fits a logistic function to map
    raw probabilities to calibrated probabilities.

    Best for: Small datasets, when a smooth calibration curve is preferred.
    """

    def __init__(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for PlattCalibrator")
        self._model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self._fitted = False

    def fit(self, probabilities: np.ndarray, outcomes: np.ndarray) -> 'PlattCalibrator':
        """
        Fit the calibrator on validation data.

        Args:
            probabilities: Raw predicted probabilities
            outcomes: Actual binary outcomes (0 or 1)

        Returns:
            self for chaining
        """
        probabilities = np.asarray(probabilities).flatten()
        outcomes = np.asarray(outcomes).flatten()

        if len(probabilities) < 10:
            logger.warning("Platt calibration needs at least 10 samples")
            self._fitted = False
            return self

        # Reshape for sklearn (needs 2D input)
        X = probabilities.reshape(-1, 1)
        self._model.fit(X, outcomes)
        self._fitted = True
        logger.info(f"PlattCalibrator fitted on {len(probabilities)} samples")
        return self

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calibrate raw probabilities.

        Args:
            probabilities: Raw predicted probabilities

        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            logger.warning("Calibrator not fitted, returning raw probabilities")
            return np.asarray(probabilities)

        probabilities = np.asarray(probabilities).flatten()
        X = probabilities.reshape(-1, 1)
        calibrated = self._model.predict_proba(X)[:, 1]
        return calibrated

    def save(self, path: Path) -> None:
        """Save calibrator to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'model': self._model, 'fitted': self._fitted}, f)
        logger.info(f"Saved PlattCalibrator to {path}")

    @classmethod
    def load(cls, path: Path) -> 'PlattCalibrator':
        """Load calibrator from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        cal = cls()
        cal._model = data['model']
        cal._fitted = data['fitted']
        logger.info(f"Loaded PlattCalibrator from {path}")
        return cal

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ============================================================================
# Calibrator Factory
# ============================================================================

def get_calibrator(method: str = 'isotonic') -> IsotonicCalibrator | PlattCalibrator:
    """
    Get a calibrator by method name.

    Args:
        method: 'isotonic' or 'platt'

    Returns:
        Calibrator instance
    """
    if method.lower() == 'isotonic':
        return IsotonicCalibrator()
    elif method.lower() == 'platt':
        return PlattCalibrator()
    else:
        raise ValueError(f"Unknown calibration method: {method}")


# ============================================================================
# Global Calibrator State (for integration)
# ============================================================================

_global_calibrator: Optional[IsotonicCalibrator | PlattCalibrator] = None


def set_global_calibrator(calibrator: IsotonicCalibrator | PlattCalibrator) -> None:
    """Set the global calibrator for use across modules."""
    global _global_calibrator
    _global_calibrator = calibrator


def get_global_calibrator() -> Optional[IsotonicCalibrator | PlattCalibrator]:
    """Get the global calibrator if set."""
    return _global_calibrator


def calibrate_probability(raw_prob: float) -> float:
    """
    Calibrate a single probability using the global calibrator.

    If no calibrator is set, returns the raw probability.

    Args:
        raw_prob: Raw predicted probability

    Returns:
        Calibrated probability
    """
    if _global_calibrator is None or not _global_calibrator.is_fitted:
        return raw_prob
    calibrated = _global_calibrator.calibrate(np.array([raw_prob]))
    return float(calibrated[0])
