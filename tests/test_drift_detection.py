"""
Tests for Drift Detection and Calibration.

Tests performance drift detection, Brier score calculation,
and calibration tracking.
"""
import pytest
import statistics
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

from monitor.drift_detector import (
    DriftDetector,
    DriftReport,
    DriftType,
    DriftSeverity,
    DriftThresholds,
    TradeRecord,
    get_drift_detector,
    record_trade,
    check_drift,
)

from monitor.calibration import (
    CalibrationTracker,
    CalibrationReport,
    get_calibration_tracker,
    record_prediction,
    get_calibration_report,
    calculate_brier_score,
)


class TestDriftDetector:
    """Tests for DriftDetector."""

    @pytest.fixture
    def detector(self):
        """Create fresh detector for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = DriftDetector(
                state_dir=Path(tmpdir),
                auto_persist=False,
            )
            yield detector

    def test_initialization(self, detector):
        """Should initialize with defaults."""
        assert detector.thresholds.win_rate_min == 0.45
        assert detector.thresholds.profit_factor_min == 1.0
        assert len(detector._trades) == 0

    def test_record_trade(self, detector):
        """Should record trades."""
        detector.record_trade(won=True, pnl=100.0)
        detector.record_trade(won=False, pnl=-50.0)

        assert len(detector._trades) == 2
        assert detector._trades[0].won == True
        assert detector._trades[1].pnl == -50.0

    def test_consecutive_losses_tracking(self, detector):
        """Should track consecutive losses."""
        detector.record_trade(won=True, pnl=100.0)
        assert detector._consecutive_losses == 0

        detector.record_trade(won=False, pnl=-50.0)
        assert detector._consecutive_losses == 1

        detector.record_trade(won=False, pnl=-50.0)
        assert detector._consecutive_losses == 2

        detector.record_trade(won=True, pnl=100.0)
        assert detector._consecutive_losses == 0

    def test_equity_tracking(self, detector):
        """Should track equity and peak."""
        detector.record_trade(won=True, pnl=100.0)
        assert detector._current_equity == 100.0
        assert detector._peak_equity == 100.0

        detector.record_trade(won=True, pnl=50.0)
        assert detector._current_equity == 150.0
        assert detector._peak_equity == 150.0

        detector.record_trade(won=False, pnl=-30.0)
        assert detector._current_equity == 120.0
        assert detector._peak_equity == 150.0  # Peak unchanged

    def test_no_drift_with_good_performance(self, detector):
        """Should detect no drift with good performance."""
        # Record 50 winning trades
        for _ in range(50):
            detector.record_trade(won=True, pnl=100.0, predicted_prob=0.7)

        report = detector.check_drift()

        assert not report.drift_detected
        assert report.severity == DriftSeverity.NONE

    def test_drift_detected_low_win_rate(self, detector):
        """Should detect drift when win rate drops."""
        # Set low thresholds for test
        detector.thresholds.win_rate_min = 0.50
        detector.thresholds.min_trades_for_check = 10

        # Record mostly losses
        for i in range(20):
            detector.record_trade(won=(i < 5), pnl=100.0 if i < 5 else -50.0)

        report = detector.check_drift()

        assert report.drift_detected
        assert DriftType.WIN_RATE in report.drift_types

    def test_drift_detected_consecutive_losses(self, detector):
        """Should detect drift on consecutive losses."""
        detector.thresholds.consecutive_losses_max = 5
        detector.thresholds.min_trades_for_check = 5

        # Record 6 consecutive losses
        for _ in range(6):
            detector.record_trade(won=False, pnl=-50.0)

        report = detector.check_drift()

        assert report.drift_detected
        assert report.severity == DriftSeverity.CRITICAL

    def test_baseline_calculation(self, detector):
        """Should calculate baseline from initial trades."""
        detector.thresholds.baseline_window = 20

        # Record baseline period trades (70% win rate)
        for i in range(25):
            detector.record_trade(won=(i % 10 < 7), pnl=100.0 if i % 10 < 7 else -50.0)

        assert detector._baseline_set
        assert 0.6 <= detector._baseline['win_rate'] <= 0.8

    def test_performance_degradation_detection(self, detector):
        """Should detect degradation from baseline."""
        detector.thresholds.baseline_window = 20
        detector.thresholds.rolling_window = 10
        detector.thresholds.win_rate_degradation = 0.15
        detector.thresholds.min_trades_for_check = 5

        # Baseline period: 80% win rate
        for i in range(25):
            detector.record_trade(won=(i % 10 < 8), pnl=100.0 if i % 10 < 8 else -50.0)

        # Recent period: 40% win rate (significant drop)
        for i in range(15):
            detector.record_trade(won=(i % 10 < 4), pnl=100.0 if i % 10 < 4 else -50.0)

        report = detector.check_drift()

        assert report.drift_detected
        assert DriftType.PERFORMANCE in report.drift_types or DriftType.WIN_RATE in report.drift_types

    def test_get_status(self, detector):
        """Should return status dict."""
        for i in range(10):
            detector.record_trade(won=True, pnl=100.0)

        status = detector.get_status()

        assert status['trades_recorded'] == 10
        assert 'rolling_metrics' in status
        assert 'consecutive_losses' in status

    def test_reset(self, detector):
        """Should reset all state."""
        for _ in range(10):
            detector.record_trade(won=True, pnl=100.0)

        detector.reset()

        assert len(detector._trades) == 0
        assert detector._consecutive_losses == 0
        assert detector._current_equity == 0.0


class TestDriftThresholds:
    """Tests for DriftThresholds."""

    def test_default_thresholds(self):
        """Should have sensible defaults."""
        t = DriftThresholds()

        assert t.win_rate_min == 0.45
        assert t.profit_factor_min == 1.0
        assert t.consecutive_losses_max == 10
        assert t.rolling_window == 50

    def test_custom_thresholds(self):
        """Should accept custom values."""
        t = DriftThresholds(
            win_rate_min=0.55,
            profit_factor_min=1.2,
            consecutive_losses_max=5,
        )

        assert t.win_rate_min == 0.55
        assert t.profit_factor_min == 1.2
        assert t.consecutive_losses_max == 5


class TestDriftReport:
    """Tests for DriftReport."""

    def test_to_dict(self):
        """Should convert to dict."""
        report = DriftReport(
            checked_at=datetime.now(),
            drift_detected=True,
            drift_types=[DriftType.WIN_RATE],
            severity=DriftSeverity.MEDIUM,
            should_stand_down=False,
            reason="Win rate below minimum",
        )

        d = report.to_dict()

        assert d['drift_detected'] == True
        assert 'win_rate' in d['drift_types']
        assert d['severity'] == 'medium'


class TestCalibrationTracker:
    """Tests for CalibrationTracker."""

    @pytest.fixture
    def tracker(self):
        """Create fresh tracker for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = CalibrationTracker(
                state_dir=Path(tmpdir),
                auto_persist=False,
            )
            yield tracker

    def test_initialization(self, tracker):
        """Should initialize with defaults."""
        assert tracker.brier_threshold == 0.25
        assert tracker.calibration_threshold == 0.10
        assert len(tracker._predictions) == 0

    def test_record_prediction(self, tracker):
        """Should record predictions."""
        tracker.record(predicted_prob=0.7, actual_outcome=True)
        tracker.record(predicted_prob=0.3, actual_outcome=False)

        assert len(tracker._predictions) == 2

    def test_perfect_calibration(self, tracker):
        """Should report good calibration for perfect predictions."""
        # Perfect calibration: 70% confidence always wins
        for _ in range(50):
            tracker.record(predicted_prob=0.7, actual_outcome=True)
        for _ in range(21):  # ~30% at 70% confidence lose
            tracker.record(predicted_prob=0.7, actual_outcome=False)

        report = tracker.get_report()

        # Should have good Brier score (predicting 0.7 and getting ~70% wins)
        assert report.brier_score < 0.3
        assert report.predictions_analyzed == 71

    def test_poor_calibration(self, tracker):
        """Should detect poor calibration."""
        # Always predict 90% but only win 50%
        for i in range(100):
            tracker.record(
                predicted_prob=0.9,
                actual_outcome=(i % 2 == 0),  # 50% win rate
            )

        report = tracker.get_report()

        # Brier score should be poor
        # Predicting 0.9 but getting 0.5 outcome = (0.9-0.5)^2 or (0.9-1)^2
        assert report.brier_score > 0.15
        assert not report.is_calibrated

    def test_calibration_grade(self, tracker):
        """Should assign correct grades."""
        # Good calibration
        for i in range(100):
            prob = 0.6
            outcome = i < 60  # 60% win rate for 60% predictions
            tracker.record(predicted_prob=prob, actual_outcome=outcome)

        report = tracker.get_report()

        # Should get decent grade
        assert report.calibration_grade in ['A', 'B', 'C']

    def test_bucket_analysis(self, tracker):
        """Should analyze by probability bucket."""
        # Mix of predictions
        for i in range(30):
            tracker.record(predicted_prob=0.3, actual_outcome=(i < 9))  # 30% at 30%
        for i in range(30):
            tracker.record(predicted_prob=0.7, actual_outcome=(i < 21))  # 70% at 70%

        report = tracker.get_report()

        assert len(report.bucket_analysis) >= 2
        assert '0.2-0.3' in report.bucket_analysis or '0.3-0.4' in report.bucket_analysis
        assert '0.6-0.7' in report.bucket_analysis or '0.7-0.8' in report.bucket_analysis

    def test_get_stats(self, tracker):
        """Should return summary stats."""
        for i in range(30):
            tracker.record(predicted_prob=0.6, actual_outcome=(i < 18))

        stats = tracker.get_stats()

        assert stats['predictions_count'] == 30
        assert 'brier_score' in stats
        assert 'is_calibrated' in stats

    def test_reset(self, tracker):
        """Should reset state."""
        for _ in range(10):
            tracker.record(predicted_prob=0.5, actual_outcome=True)

        tracker.reset()

        assert len(tracker._predictions) == 0


class TestBrierScore:
    """Tests for Brier score calculation."""

    def test_perfect_predictions(self):
        """Perfect predictions should have Brier score of 0."""
        predictions = [
            (1.0, True),
            (0.0, False),
            (1.0, True),
            (0.0, False),
        ]

        score = calculate_brier_score(predictions)

        assert score == 0.0

    def test_worst_predictions(self):
        """Completely wrong predictions should have high Brier score."""
        predictions = [
            (0.0, True),  # Predicted 0%, was True -> error = 1.0
            (1.0, False),  # Predicted 100%, was False -> error = 1.0
        ]

        score = calculate_brier_score(predictions)

        assert score == 1.0

    def test_random_predictions(self):
        """Random 50% predictions should have Brier ~0.25."""
        predictions = [
            (0.5, True),
            (0.5, False),
            (0.5, True),
            (0.5, False),
        ]

        score = calculate_brier_score(predictions)

        assert score == 0.25  # (0.5 - 1)^2 = 0.25 and (0.5 - 0)^2 = 0.25

    def test_empty_predictions(self):
        """Empty predictions should return 1.0."""
        score = calculate_brier_score([])
        assert score == 1.0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_drift_detector(self):
        """Should return singleton."""
        d1 = get_drift_detector()
        d2 = get_drift_detector()
        assert d1 is d2

    def test_get_calibration_tracker(self):
        """Should return singleton."""
        t1 = get_calibration_tracker()
        t2 = get_calibration_tracker()
        assert t1 is t2


# Run with: pytest tests/test_drift_detection.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
