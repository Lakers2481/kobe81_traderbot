"""
Monitor module for Kobe Trading System.
Provides health check endpoints, metrics collection, and drift detection.

Components:
- health_endpoints: HTTP health checks (/health, /readiness, /liveness, /metrics)
- drift_detector: Performance drift detection (win rate, PF, Sharpe degradation)
- calibration: Probability calibration tracking (Brier score, calibration error)
"""
from __future__ import annotations

from .health_endpoints import (
    start_health_server,
    get_metrics,
    update_request_counter,
    update_performance_metrics,
    load_performance_from_summary,
    reset_metrics,
)

from .drift_detector import (
    DriftDetector,
    DriftReport,
    DriftType,
    DriftSeverity,
    DriftThresholds,
    get_drift_detector,
    record_trade,
    check_drift,
)

from .calibration import (
    CalibrationTracker,
    CalibrationReport,
    get_calibration_tracker,
    record_prediction,
    get_calibration_report,
    calculate_brier_score,
)

__all__ = [
    # Health endpoints
    "start_health_server",
    "get_metrics",
    "update_request_counter",
    "update_performance_metrics",
    "load_performance_from_summary",
    "reset_metrics",
    # Drift detection
    "DriftDetector",
    "DriftReport",
    "DriftType",
    "DriftSeverity",
    "DriftThresholds",
    "get_drift_detector",
    "record_trade",
    "check_drift",
    # Calibration
    "CalibrationTracker",
    "CalibrationReport",
    "get_calibration_tracker",
    "record_prediction",
    "get_calibration_report",
    "calculate_brier_score",
]
