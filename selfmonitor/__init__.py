"""
Self-Monitoring & Failure Detection Module
===========================================

Automated monitoring, circuit breakers, and anomaly detection
for trading system health and safety.

Components:
- CircuitBreaker: Automatic trading halt on failures
- AnomalyDetector: Detect unusual patterns
- HealthMonitor: Overall system health tracking
"""

from .circuit_breaker import (
    CircuitBreaker,
    BreakerState,
    BreakerConfig,
    get_breaker,
    check_breaker,
    trip_breaker,
)

from .anomaly_detector import (
    AnomalyDetector,
    AnomalyType,
    AnomalyAlert,
    detect_anomalies,
    is_anomalous,
)

__all__ = [
    'CircuitBreaker',
    'BreakerState',
    'BreakerConfig',
    'get_breaker',
    'check_breaker',
    'trip_breaker',
    'AnomalyDetector',
    'AnomalyType',
    'AnomalyAlert',
    'detect_anomalies',
    'is_anomalous',
]
