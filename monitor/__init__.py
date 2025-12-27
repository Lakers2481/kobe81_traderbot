"""
Monitor module for Kobe Trading System.
Provides health check endpoints, metrics collection, and drift detection.

Components:
- health_endpoints: HTTP health checks (/health, /readiness, /liveness, /metrics)
- drift_detector: Performance drift detection (win rate, PF, Sharpe degradation)
- calibration: Probability calibration tracking (Brier score, calibration error)
- heartbeat: Process heartbeat tracking for daemon monitoring
"""
from __future__ import annotations

from .health_endpoints import (
    start_health_server,
    get_metrics,
    update_request_counter,
    update_trade_event,
    update_performance_metrics,
    load_performance_from_summary,
    reset_metrics,
)

# Use lazy imports for optional components to avoid import errors
def __getattr__(name: str):
    """Lazy import for optional monitor components."""
    if name in ("DriftDetector", "DriftReport", "DriftType", "DriftSeverity",
                "DriftThresholds", "get_drift_detector", "record_trade", "check_drift"):
        from . import drift_detector
        return getattr(drift_detector, name)
    elif name in ("brier_score", "reliability_table"):
        from . import calibration
        return getattr(calibration, name)
    elif name in ("HeartbeatWriter", "read_heartbeat", "is_heartbeat_stale",
                  "get_heartbeat_age", "init_global_heartbeat", "get_global_heartbeat",
                  "update_global_heartbeat", "stop_global_heartbeat"):
        from . import heartbeat
        return getattr(heartbeat, name)
    raise AttributeError(f"module 'monitor' has no attribute '{name}'")

__all__ = [
    # Health endpoints
    "start_health_server",
    "get_metrics",
    "update_request_counter",
    "update_trade_event",
    "update_performance_metrics",
    "load_performance_from_summary",
    "reset_metrics",
    # Drift detection (lazy)
    "DriftDetector",
    "DriftReport",
    "DriftType",
    "DriftSeverity",
    "DriftThresholds",
    "get_drift_detector",
    "record_trade",
    "check_drift",
    # Calibration (lazy)
    "brier_score",
    "reliability_table",
    # Heartbeat (lazy)
    "HeartbeatWriter",
    "read_heartbeat",
    "is_heartbeat_stale",
    "get_heartbeat_age",
    "init_global_heartbeat",
    "get_global_heartbeat",
    "update_global_heartbeat",
    "stop_global_heartbeat",
]
