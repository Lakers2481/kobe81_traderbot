"""
Probability Calibration Module
==============================

Tracks and evaluates the calibration of probability predictions.
A well-calibrated model should have predicted probabilities that match actual frequencies.

Metrics:
- Brier Score: Mean squared error of probability predictions (lower is better)
- Calibration Error: Difference between predicted and actual frequencies by bucket
- Reliability Diagram: Visual representation of calibration

Usage:
    from monitor.calibration import CalibrationTracker, get_calibration_tracker

    tracker = get_calibration_tracker()

    # Record predictions and outcomes
    tracker.record(predicted_prob=0.72, actual_outcome=True)
    tracker.record(predicted_prob=0.35, actual_outcome=False)

    # Get calibration report
    report = tracker.get_report()
    print(f"Brier Score: {report.brier_score:.4f}")
    print(f"Calibration Error: {report.calibration_error:.4f}")

    # Check if model is well-calibrated
    if report.is_calibrated:
        print("Model is well-calibrated")
"""
from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Record of a single prediction."""
    timestamp: datetime
    predicted_prob: float
    actual_outcome: bool
    strategy: str = "unknown"
    context: Dict = field(default_factory=dict)


@dataclass
class CalibrationReport:
    """Report of calibration analysis."""
    generated_at: datetime
    predictions_analyzed: int

    # Core metrics
    brier_score: float
    calibration_error: float

    # Derived
    is_calibrated: bool
    calibration_grade: str  # A, B, C, D, F

    # Bucket analysis
    bucket_analysis: Dict[str, Dict]

    # Thresholds used
    brier_threshold: float
    calibration_threshold: float

    def to_dict(self) -> Dict:
        return {
            'generated_at': self.generated_at.isoformat(),
            'predictions_analyzed': self.predictions_analyzed,
            'brier_score': self.brier_score,
            'calibration_error': self.calibration_error,
            'is_calibrated': self.is_calibrated,
            'calibration_grade': self.calibration_grade,
            'bucket_analysis': self.bucket_analysis,
            'thresholds': {
                'brier': self.brier_threshold,
                'calibration': self.calibration_threshold,
            },
        }


class CalibrationTracker:
    """
    Tracks probability calibration over time.

    Evaluates whether predicted probabilities match actual frequencies.
    A model predicting 70% should win approximately 70% of the time.
    """

    def __init__(
        self,
        brier_threshold: float = 0.25,
        calibration_threshold: float = 0.10,
        max_history: int = 1000,
        state_dir: Optional[Path] = None,
        auto_persist: bool = True,
    ):
        """
        Initialize calibration tracker.

        Args:
            brier_threshold: Maximum acceptable Brier score (default: 0.25)
            calibration_threshold: Maximum acceptable calibration error (default: 0.10)
            max_history: Maximum predictions to store
            state_dir: Directory for persistence
            auto_persist: Whether to auto-save state
        """
        self.brier_threshold = brier_threshold
        self.calibration_threshold = calibration_threshold
        self.state_dir = Path(state_dir) if state_dir else Path("state/monitoring")
        self.auto_persist = auto_persist

        self._predictions: deque[PredictionRecord] = deque(maxlen=max_history)
        self._last_report: Optional[CalibrationReport] = None

        self._load_state()

    def record(
        self,
        predicted_prob: float,
        actual_outcome: bool,
        strategy: str = "unknown",
        context: Optional[Dict] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record a prediction and its actual outcome.

        Args:
            predicted_prob: Predicted probability of success (0.0 to 1.0)
            actual_outcome: Whether the prediction was correct
            strategy: Strategy that made the prediction
            context: Additional context (regime, etc.)
            timestamp: Time of prediction (defaults to now)
        """
        if not 0.0 <= predicted_prob <= 1.0:
            logger.warning(f"Invalid predicted_prob: {predicted_prob}, clipping to [0, 1]")
            predicted_prob = max(0.0, min(1.0, predicted_prob))

        record = PredictionRecord(
            timestamp=timestamp or datetime.now(),
            predicted_prob=predicted_prob,
            actual_outcome=actual_outcome,
            strategy=strategy,
            context=context or {},
        )
        self._predictions.append(record)

        if self.auto_persist:
            self._save_state()

    def get_report(self, min_samples: int = 20) -> CalibrationReport:
        """
        Generate calibration report.

        Args:
            min_samples: Minimum samples required for analysis

        Returns:
            CalibrationReport with metrics and analysis
        """
        now = datetime.now()

        if len(self._predictions) < min_samples:
            return CalibrationReport(
                generated_at=now,
                predictions_analyzed=len(self._predictions),
                brier_score=1.0,
                calibration_error=1.0,
                is_calibrated=False,
                calibration_grade="F",
                bucket_analysis={},
                brier_threshold=self.brier_threshold,
                calibration_threshold=self.calibration_threshold,
            )

        predictions = list(self._predictions)

        # Calculate Brier score
        brier_scores = []
        for p in predictions:
            outcome = 1.0 if p.actual_outcome else 0.0
            brier_scores.append((p.predicted_prob - outcome) ** 2)
        brier_score = statistics.mean(brier_scores)

        # Calculate bucket-wise calibration
        bucket_analysis = self._analyze_buckets(predictions)

        # Calculate calibration error
        calibration_errors = []
        for bucket, data in bucket_analysis.items():
            if data['count'] >= 5:
                error = abs(data['predicted_mean'] - data['actual_mean'])
                calibration_errors.append(error)

        calibration_error = statistics.mean(calibration_errors) if calibration_errors else 1.0

        # Determine if calibrated
        is_calibrated = (
            brier_score <= self.brier_threshold and
            calibration_error <= self.calibration_threshold
        )

        # Calculate grade
        grade = self._calculate_grade(brier_score, calibration_error)

        report = CalibrationReport(
            generated_at=now,
            predictions_analyzed=len(predictions),
            brier_score=brier_score,
            calibration_error=calibration_error,
            is_calibrated=is_calibrated,
            calibration_grade=grade,
            bucket_analysis=bucket_analysis,
            brier_threshold=self.brier_threshold,
            calibration_threshold=self.calibration_threshold,
        )

        self._last_report = report
        return report

    def _analyze_buckets(self, predictions: List[PredictionRecord]) -> Dict[str, Dict]:
        """Analyze predictions by probability bucket."""
        buckets: Dict[str, List[PredictionRecord]] = {}

        # Create buckets: 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
        for p in predictions:
            bucket_idx = min(int(p.predicted_prob * 10), 9)
            bucket_name = f"{bucket_idx/10:.1f}-{(bucket_idx+1)/10:.1f}"

            if bucket_name not in buckets:
                buckets[bucket_name] = []
            buckets[bucket_name].append(p)

        # Calculate stats for each bucket
        analysis = {}
        for bucket_name, bucket_preds in sorted(buckets.items()):
            predicted_probs = [p.predicted_prob for p in bucket_preds]
            actual_outcomes = [1.0 if p.actual_outcome else 0.0 for p in bucket_preds]

            analysis[bucket_name] = {
                'count': len(bucket_preds),
                'predicted_mean': statistics.mean(predicted_probs),
                'actual_mean': statistics.mean(actual_outcomes),
                'calibration_error': abs(
                    statistics.mean(predicted_probs) - statistics.mean(actual_outcomes)
                ),
            }

        return analysis

    def _calculate_grade(self, brier_score: float, calibration_error: float) -> str:
        """Calculate letter grade based on metrics."""
        # Combined score (weighted average)
        combined = 0.6 * brier_score + 0.4 * calibration_error

        if combined <= 0.05:
            return "A"
        elif combined <= 0.10:
            return "B"
        elif combined <= 0.20:
            return "C"
        elif combined <= 0.30:
            return "D"
        else:
            return "F"

    def get_stats(self) -> Dict:
        """Get summary statistics."""
        if len(self._predictions) == 0:
            return {
                'predictions_count': 0,
                'brier_score': None,
                'calibration_error': None,
                'is_calibrated': None,
            }

        report = self.get_report()
        return {
            'predictions_count': report.predictions_analyzed,
            'brier_score': report.brier_score,
            'calibration_error': report.calibration_error,
            'is_calibrated': report.is_calibrated,
            'grade': report.calibration_grade,
        }

    def get_reliability_data(self) -> List[Tuple[float, float, int]]:
        """
        Get data for reliability diagram.

        Returns:
            List of (predicted_mean, actual_mean, count) for each bucket
        """
        if len(self._predictions) < 20:
            return []

        report = self.get_report()
        data = []

        for bucket_name, bucket_data in sorted(report.bucket_analysis.items()):
            if bucket_data['count'] >= 3:
                data.append((
                    bucket_data['predicted_mean'],
                    bucket_data['actual_mean'],
                    bucket_data['count'],
                ))

        return data

    def reset(self) -> None:
        """Reset all tracked predictions."""
        self._predictions.clear()
        self._last_report = None

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            state_file = self.state_dir / "calibration_state.json"

            state = {
                'predictions': [
                    {
                        'timestamp': p.timestamp.isoformat(),
                        'predicted_prob': p.predicted_prob,
                        'actual_outcome': p.actual_outcome,
                        'strategy': p.strategy,
                        'context': p.context,
                    }
                    for p in list(self._predictions)[-500:]
                ],
                'saved_at': datetime.now().isoformat(),
            }

            state_file.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.debug(f"Failed to save calibration state: {e}")

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        try:
            state_file = self.state_dir / "calibration_state.json"
            if not state_file.exists():
                return

            state = json.loads(state_file.read_text())

            for p in state.get('predictions', []):
                self._predictions.append(PredictionRecord(
                    timestamp=datetime.fromisoformat(p['timestamp']),
                    predicted_prob=p['predicted_prob'],
                    actual_outcome=p['actual_outcome'],
                    strategy=p.get('strategy', 'unknown'),
                    context=p.get('context', {}),
                ))

            logger.info(f"Loaded calibration state: {len(self._predictions)} predictions")
        except Exception as e:
            logger.debug(f"Failed to load calibration state: {e}")


# Singleton instance
_calibration_tracker: Optional[CalibrationTracker] = None


def get_calibration_tracker() -> CalibrationTracker:
    """Get the singleton calibration tracker instance."""
    global _calibration_tracker
    if _calibration_tracker is None:
        _calibration_tracker = CalibrationTracker()
    return _calibration_tracker


def record_prediction(
    predicted_prob: float,
    actual_outcome: bool,
    strategy: str = "unknown",
) -> None:
    """Convenience function to record a prediction."""
    get_calibration_tracker().record(
        predicted_prob=predicted_prob,
        actual_outcome=actual_outcome,
        strategy=strategy,
    )


def get_calibration_report() -> CalibrationReport:
    """Convenience function to get calibration report."""
    return get_calibration_tracker().get_report()


def calculate_brier_score(predictions: List[Tuple[float, bool]]) -> float:
    """
    Calculate Brier score from a list of (predicted_prob, actual_outcome) tuples.

    Brier Score = mean((predicted - actual)^2)
    - Perfect score: 0.0
    - Random guessing at 0.5: 0.25
    - Always wrong: 1.0

    Args:
        predictions: List of (predicted_prob, actual_outcome) tuples

    Returns:
        Brier score (0.0 to 1.0, lower is better)
    """
    if not predictions:
        return 1.0

    scores = []
    for predicted, actual in predictions:
        outcome = 1.0 if actual else 0.0
        scores.append((predicted - outcome) ** 2)

    return statistics.mean(scores)
