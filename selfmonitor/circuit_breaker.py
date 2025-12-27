"""
Circuit Breaker for Trading Systems
=====================================

Automatically halts trading when failure conditions are detected.
Protects against cascading failures and excessive losses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class BreakerState(Enum):
    """State of the circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Trading halted
    HALF_OPEN = "half_open"  # Testing if safe to resume


@dataclass
class BreakerConfig:
    """Configuration for circuit breaker."""
    # Loss thresholds
    max_daily_loss: float = 1000.0
    max_consecutive_losses: int = 5
    max_drawdown: float = 0.10  # 10%

    # Error thresholds
    max_errors_per_hour: int = 10
    max_api_failures: int = 5

    # Recovery
    cooldown_minutes: int = 30
    half_open_trades: int = 3  # Test trades before full recovery

    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_daily_loss': self.max_daily_loss,
            'max_consecutive_losses': self.max_consecutive_losses,
            'max_drawdown': self.max_drawdown,
            'cooldown_minutes': self.cooldown_minutes,
        }


@dataclass
class TripRecord:
    """Record of a circuit breaker trip."""
    tripped_at: datetime
    reason: str
    state: BreakerState
    details: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """
    Circuit breaker for trading system protection.

    Monitors for failure conditions and automatically halts
    trading when thresholds are exceeded.
    """

    def __init__(
        self,
        config: Optional[BreakerConfig] = None,
        state_file: Optional[Path] = None,
    ):
        self.config = config or BreakerConfig()
        self.state_file = state_file or Path("state/circuit_breaker.json")

        self._state = BreakerState.CLOSED
        self._tripped_at: Optional[datetime] = None
        self._trip_reason: str = ""
        self._trip_history: List[TripRecord] = []

        # Counters
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
        self._errors_this_hour = 0
        self._api_failures = 0
        self._half_open_trades = 0

        logger.info("CircuitBreaker initialized")

    @property
    def state(self) -> BreakerState:
        """Get current breaker state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Whether trading is halted."""
        return self._state == BreakerState.OPEN

    @property
    def can_trade(self) -> bool:
        """Whether trading is allowed."""
        if self._state == BreakerState.CLOSED:
            return True
        if self._state == BreakerState.HALF_OPEN:
            return self._half_open_trades < self.config.half_open_trades
        return False

    def check(self) -> bool:
        """Check if trading is allowed. Returns True if OK."""
        if self._state == BreakerState.OPEN:
            # Check if cooldown has passed
            if self._tripped_at:
                cooldown = timedelta(minutes=self.config.cooldown_minutes)
                if datetime.now() - self._tripped_at > cooldown:
                    self._transition_to_half_open()
            return False

        if self._state == BreakerState.HALF_OPEN:
            return self._half_open_trades < self.config.half_open_trades

        return True

    def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN."""
        self._state = BreakerState.HALF_OPEN
        self._half_open_trades = 0
        logger.info("Circuit breaker transitioning to HALF_OPEN")

    def trip(self, reason: str, details: Optional[Dict] = None):
        """Trip the circuit breaker."""
        self._state = BreakerState.OPEN
        self._tripped_at = datetime.now()
        self._trip_reason = reason

        record = TripRecord(
            tripped_at=datetime.now(),
            reason=reason,
            state=BreakerState.OPEN,
            details=details or {},
        )
        self._trip_history.append(record)

        logger.warning(f"Circuit breaker TRIPPED: {reason}")

    def reset(self):
        """Reset the circuit breaker."""
        self._state = BreakerState.CLOSED
        self._tripped_at = None
        self._trip_reason = ""
        self._consecutive_losses = 0
        self._errors_this_hour = 0
        self._api_failures = 0
        logger.info("Circuit breaker RESET")

    def record_trade(self, pnl: float):
        """Record a trade outcome."""
        self._daily_pnl += pnl

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        if self._state == BreakerState.HALF_OPEN:
            self._half_open_trades += 1
            if pnl > 0 and self._half_open_trades >= self.config.half_open_trades:
                self.reset()

        # Check thresholds
        self._check_thresholds()

    def record_error(self, is_api: bool = False):
        """Record an error."""
        self._errors_this_hour += 1
        if is_api:
            self._api_failures += 1
        self._check_thresholds()

    def _check_thresholds(self):
        """Check if any threshold is exceeded."""
        if self._state == BreakerState.OPEN:
            return

        if self._daily_pnl <= -self.config.max_daily_loss:
            self.trip(f"Daily loss limit exceeded: ${self._daily_pnl:.2f}")
            return

        if self._consecutive_losses >= self.config.max_consecutive_losses:
            self.trip(f"Consecutive losses: {self._consecutive_losses}")
            return

        if self._errors_this_hour >= self.config.max_errors_per_hour:
            self.trip(f"Error rate exceeded: {self._errors_this_hour}/hour")
            return

        if self._api_failures >= self.config.max_api_failures:
            self.trip(f"API failures: {self._api_failures}")
            return

    def reset_daily(self):
        """Reset daily counters (call at start of day)."""
        self._daily_pnl = 0.0
        self._errors_this_hour = 0
        self._api_failures = 0

    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            'state': self._state.value,
            'can_trade': self.can_trade,
            'daily_pnl': self._daily_pnl,
            'consecutive_losses': self._consecutive_losses,
            'errors_this_hour': self._errors_this_hour,
            'trip_reason': self._trip_reason,
            'trips_today': len([t for t in self._trip_history
                               if t.tripped_at.date() == datetime.now().date()]),
        }


# Global instance
_breaker: Optional[CircuitBreaker] = None


def get_breaker() -> CircuitBreaker:
    """Get or create global circuit breaker."""
    global _breaker
    if _breaker is None:
        _breaker = CircuitBreaker()
    return _breaker


def check_breaker() -> bool:
    """Check if trading is allowed."""
    return get_breaker().check()


def trip_breaker(reason: str):
    """Trip the circuit breaker."""
    get_breaker().trip(reason)
