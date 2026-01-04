"""
Circuit Breaker Pattern for API Resilience

Implements the Circuit Breaker pattern to prevent cascading failures
when external services (Polygon, Alpaca, etc.) are degraded or unavailable.

Based on: Codex & Gemini reliability recommendations (2026-01-04)

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service unavailable, requests fail immediately
- HALF_OPEN: Testing if service recovered

Usage:
    from core.circuit_breaker import CircuitBreaker, circuit_protected

    # Decorator usage
    @circuit_protected("polygon_api")
    def fetch_data(symbol):
        return polygon_client.get_bars(symbol)

    # Context manager usage
    breaker = CircuitBreaker("alpaca_api")
    with breaker:
        order = alpaca_client.submit_order(...)

    # Manual usage
    breaker = get_breaker("polygon_api")
    if breaker.allow_request():
        try:
            result = api_call()
            breaker.record_success()
        except Exception as e:
            breaker.record_failure(e)
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    current_state: CircuitState = CircuitState.CLOSED
    recent_errors: List[str] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rejected_requests": self.rejected_requests,
            "consecutive_failures": self.consecutive_failures,
            "failure_rate": round(self.failure_rate, 4),
            "success_rate": round(self.success_rate, 4),
            "current_state": self.current_state.value,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success": self.last_success_time.isoformat() if self.last_success_time else None,
            "recent_errors": self.recent_errors[-5:],
        }


class CircuitOpenError(Exception):
    """Raised when circuit is open and request is rejected."""

    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"Circuit '{name}' is OPEN. Retry after {retry_after:.1f}s")


class CircuitBreaker:
    """
    Circuit breaker for external service calls.

    Protects against cascading failures by:
    1. Tracking consecutive failures
    2. Opening circuit after threshold exceeded
    3. Periodically testing if service recovered
    4. Closing circuit after successful recovery
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 3,
        success_threshold: int = 2,
        excluded_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Unique name for this circuit
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            half_open_requests: Max requests allowed in half-open state
            success_threshold: Successes needed to close circuit
            excluded_exceptions: Exceptions that don't count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        self.success_threshold = success_threshold
        self.excluded_exceptions = excluded_exceptions or []

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._lock = threading.RLock()
        self._half_open_count = 0
        self._half_open_successes = 0
        self._opened_at: Optional[datetime] = None

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        with self._lock:
            self._stats.current_state = self._state
            return self._stats

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self.state == CircuitState.HALF_OPEN

    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout elapsed
                if self._opened_at and datetime.now() > self._opened_at + timedelta(seconds=self.recovery_timeout):
                    self._transition_to(CircuitState.HALF_OPEN)
                    self._half_open_count = 0
                    self._half_open_successes = 0
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open
                if self._half_open_count < self.half_open_requests:
                    self._half_open_count += 1
                    return True
                return False

            return False

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            self._stats.total_requests += 1
            self._stats.successful_requests += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info(f"Circuit '{self.name}' CLOSED after recovery")

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed request."""
        with self._lock:
            # Check if this exception type is excluded
            if error and any(isinstance(error, exc_type) for exc_type in self.excluded_exceptions):
                logger.debug(f"Circuit '{self.name}': Excluded exception {type(error).__name__}")
                return

            self._stats.total_requests += 1
            self._stats.failed_requests += 1
            self._stats.consecutive_failures += 1
            self._stats.last_failure_time = datetime.now()

            if error:
                error_msg = f"{type(error).__name__}: {str(error)[:100]}"
                self._stats.recent_errors.append(error_msg)
                if len(self._stats.recent_errors) > 10:
                    self._stats.recent_errors = self._stats.recent_errors[-10:]

            if self._state == CircuitState.HALF_OPEN:
                # Single failure in half-open reopens circuit
                self._transition_to(CircuitState.OPEN)
                logger.warning(f"Circuit '{self.name}' reopened after failure in half-open")
                return

            if self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    logger.error(
                        f"Circuit '{self.name}' OPENED after {self.failure_threshold} failures. "
                        f"Last error: {error}"
                    )

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._stats.last_state_change = datetime.now()

        if new_state == CircuitState.OPEN:
            self._opened_at = datetime.now()

        logger.info(f"Circuit '{self.name}': {old_state.value} -> {new_state.value}")

    def reset(self) -> None:
        """Reset circuit to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._stats.consecutive_failures = 0
            self._half_open_count = 0
            self._half_open_successes = 0
            self._opened_at = None
            logger.info(f"Circuit '{self.name}' manually reset to CLOSED")

    def force_open(self) -> None:
        """Force circuit to open state."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)
            logger.warning(f"Circuit '{self.name}' forced OPEN")

    def get_retry_after(self) -> float:
        """Get seconds until next retry is allowed."""
        with self._lock:
            if self._state != CircuitState.OPEN or self._opened_at is None:
                return 0.0
            elapsed = (datetime.now() - self._opened_at).total_seconds()
            return max(0.0, self.recovery_timeout - elapsed)

    def __enter__(self) -> "CircuitBreaker":
        """Context manager entry."""
        if not self.allow_request():
            raise CircuitOpenError(self.name, self.get_retry_after())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure(exc_val)
        return False  # Don't suppress exceptions


# ============================================================================
# Circuit Breaker Registry
# ============================================================================

_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.

    Args:
        name: Unique name for the circuit
        failure_threshold: Failures before opening (only used on creation)
        recovery_timeout: Seconds to wait before testing (only used on creation)

    Returns:
        CircuitBreaker instance
    """
    with _registry_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return _breakers[name]


def get_all_breakers() -> Dict[str, CircuitBreaker]:
    """Get all registered circuit breakers."""
    with _registry_lock:
        return dict(_breakers)


def reset_all_breakers() -> None:
    """Reset all circuit breakers to closed state."""
    with _registry_lock:
        for breaker in _breakers.values():
            breaker.reset()


def get_circuit_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers."""
    with _registry_lock:
        return {name: breaker.stats.to_dict() for name, breaker in _breakers.items()}


# ============================================================================
# Decorator
# ============================================================================

def circuit_protected(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    fallback: Optional[Callable] = None,
) -> Callable:
    """
    Decorator to protect a function with a circuit breaker.

    Args:
        name: Circuit breaker name
        failure_threshold: Failures before opening
        recovery_timeout: Seconds before testing recovery
        fallback: Optional fallback function when circuit is open

    Usage:
        @circuit_protected("polygon_api")
        def fetch_bars(symbol):
            return polygon.get_bars(symbol)

        @circuit_protected("alpaca_api", fallback=lambda: None)
        def get_positions():
            return alpaca.get_positions()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            breaker = get_breaker(name, failure_threshold, recovery_timeout)

            if not breaker.allow_request():
                if fallback:
                    logger.warning(f"Circuit '{name}' is OPEN, using fallback")
                    return fallback(*args, **kwargs)
                raise CircuitOpenError(name, breaker.get_retry_after())

            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure(e)
                raise

        return wrapper
    return decorator


# ============================================================================
# Pre-configured Breakers for Kobe
# ============================================================================

def get_polygon_breaker() -> CircuitBreaker:
    """Get circuit breaker for Polygon API."""
    return get_breaker(
        "polygon_api",
        failure_threshold=5,
        recovery_timeout=60.0,  # 1 minute before retry
    )


def get_alpaca_breaker() -> CircuitBreaker:
    """Get circuit breaker for Alpaca API."""
    return get_breaker(
        "alpaca_api",
        failure_threshold=3,  # Lower threshold for broker
        recovery_timeout=30.0,
    )


def get_stooq_breaker() -> CircuitBreaker:
    """Get circuit breaker for Stooq API."""
    return get_breaker(
        "stooq_api",
        failure_threshold=5,
        recovery_timeout=120.0,  # 2 minutes (free tier)
    )


def get_yfinance_breaker() -> CircuitBreaker:
    """Get circuit breaker for Yahoo Finance API."""
    return get_breaker(
        "yfinance_api",
        failure_threshold=5,
        recovery_timeout=120.0,
    )


# ============================================================================
# Utility Functions
# ============================================================================

def is_service_available(name: str) -> bool:
    """Check if a service circuit is available (not open)."""
    with _registry_lock:
        if name not in _breakers:
            return True
        return _breakers[name].state != CircuitState.OPEN


def get_healthy_services() -> List[str]:
    """Get list of services with closed circuits."""
    with _registry_lock:
        return [
            name for name, breaker in _breakers.items()
            if breaker.state == CircuitState.CLOSED
        ]


def get_degraded_services() -> List[str]:
    """Get list of services with open or half-open circuits."""
    with _registry_lock:
        return [
            name for name, breaker in _breakers.items()
            if breaker.state in (CircuitState.OPEN, CircuitState.HALF_OPEN)
        ]


def should_halt_trading() -> bool:
    """
    Check if trading should halt due to critical service failures.

    Returns True if broker API circuit is open.
    """
    with _registry_lock:
        if "alpaca_api" in _breakers:
            return _breakers["alpaca_api"].state == CircuitState.OPEN
        return False
