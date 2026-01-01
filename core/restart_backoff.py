"""
Exponential Backoff for Process Restarts.

Prevents restart storms by enforcing exponential delays between restart attempts.
Mirrors the rate_limiter.py pattern but for process-level operations.

Usage:
    from core.restart_backoff import RestartBackoff, get_restart_backoff

    backoff = get_restart_backoff()
    allowed, delay, reason = backoff.should_restart()

    if not allowed:
        print(f"Restart blocked: {reason}")
        return

    if delay > 0:
        print(f"Waiting {delay:.1f}s before restart")
        time.sleep(delay)

    # ... perform restart ...
    backoff.record_restart()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# Global singleton
_restart_backoff: Optional["RestartBackoff"] = None

DEFAULT_STATE_FILE = Path("state/restart_backoff.json")


@dataclass
class RestartBackoffConfig:
    """Configuration for restart backoff."""

    enabled: bool = True
    base_delay_seconds: float = 30.0
    max_delay_seconds: float = 3600.0  # 1 hour max
    backoff_multiplier: float = 2.0
    max_attempts_per_hour: int = 5
    jitter_enabled: bool = True
    jitter_factor: float = 0.1  # +/- 10% jitter
    state_file: Path = field(default_factory=lambda: DEFAULT_STATE_FILE)
    cooldown_hours: float = 1.0  # Reset attempt counter after this period


@dataclass
class RestartState:
    """Persistent state for restart tracking."""

    component: str = "scheduler"
    attempt_count: int = 0
    last_restart_time: Optional[str] = None  # ISO format
    next_restart_allowed_at: Optional[str] = None  # ISO format
    failed_attempts: List[dict] = field(default_factory=list)
    total_restarts: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RestartState":
        """Create from dictionary."""
        return cls(
            component=data.get("component", "scheduler"),
            attempt_count=data.get("attempt_count", 0),
            last_restart_time=data.get("last_restart_time"),
            next_restart_allowed_at=data.get("next_restart_allowed_at"),
            failed_attempts=data.get("failed_attempts", []),
            total_restarts=data.get("total_restarts", 0),
        )


@dataclass
class RestartBackoff:
    """
    Exponential backoff manager for process restarts.

    Prevents restart storms by:
    1. Enforcing exponential delays between attempts
    2. Limiting max attempts per hour
    3. Persisting state across process restarts
    4. Adding jitter to prevent thundering herd
    """

    config: RestartBackoffConfig = field(default_factory=RestartBackoffConfig)
    _state: RestartState = field(default_factory=RestartState)

    def __post_init__(self):
        """Load existing state from disk."""
        self._load_state()

    def _load_state(self) -> None:
        """Load state from disk if it exists."""
        try:
            if self.config.state_file.exists():
                data = json.loads(self.config.state_file.read_text())
                self._state = RestartState.from_dict(data)
                logger.debug(f"Loaded restart state: {self._state.attempt_count} attempts")
        except Exception as e:
            logger.warning(f"Could not load restart state: {e}")
            self._state = RestartState()

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.config.state_file.write_text(
                json.dumps(self._state.to_dict(), indent=2)
            )
        except Exception as e:
            logger.error(f"Could not save restart state: {e}")

    def _check_cooldown_reset(self) -> None:
        """Reset attempt counter if cooldown period has passed."""
        if self._state.last_restart_time:
            try:
                last_restart = datetime.fromisoformat(self._state.last_restart_time)
                cooldown_delta = timedelta(hours=self.config.cooldown_hours)
                if datetime.now() - last_restart > cooldown_delta:
                    logger.info(
                        f"Cooldown period ({self.config.cooldown_hours}h) passed. "
                        f"Resetting attempt counter from {self._state.attempt_count} to 0."
                    )
                    self._state.attempt_count = 0
                    self._state.failed_attempts = []
                    self._save_state()
            except (ValueError, TypeError):
                pass

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number.

        Formula: delay = base * (multiplier ^ attempt)
        With optional jitter: delay * (1 +/- jitter_factor)

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds
        """
        delay = self.config.base_delay_seconds * (
            self.config.backoff_multiplier ** attempt
        )
        delay = min(delay, self.config.max_delay_seconds)

        # Add jitter if enabled
        if self.config.jitter_enabled:
            import random

            jitter = delay * self.config.jitter_factor
            delay = delay + random.uniform(-jitter, jitter)

        return max(0, delay)

    def should_restart(self) -> Tuple[bool, float, str]:
        """
        Determine if restart is allowed.

        Returns:
            Tuple of (allowed, delay_seconds, reason)
            - allowed: True if restart is permitted
            - delay_seconds: How long to wait before restarting
            - reason: Human-readable explanation
        """
        if not self.config.enabled:
            return True, 0.0, "backoff_disabled"

        # Check if cooldown has passed (reset counter)
        self._check_cooldown_reset()

        # Check max attempts per hour
        if self._state.attempt_count >= self.config.max_attempts_per_hour:
            return (
                False,
                0.0,
                f"max_attempts_exceeded: {self._state.attempt_count} >= {self.config.max_attempts_per_hour}",
            )

        # Check if we need to wait
        if self._state.next_restart_allowed_at:
            try:
                next_allowed = datetime.fromisoformat(
                    self._state.next_restart_allowed_at
                )
                now = datetime.now()
                if now < next_allowed:
                    wait_seconds = (next_allowed - now).total_seconds()
                    return (
                        True,
                        wait_seconds,
                        f"waiting_for_backoff: {wait_seconds:.1f}s remaining",
                    )
            except (ValueError, TypeError):
                pass

        # First attempt or within allowed window
        delay = self.get_delay(self._state.attempt_count)
        return True, delay, f"restart_allowed: attempt {self._state.attempt_count + 1}"

    def record_restart(self, success: bool = True, error: Optional[str] = None) -> None:
        """
        Record a restart attempt.

        Args:
            success: Whether the restart succeeded
            error: Error message if restart failed
        """
        now = datetime.now()
        self._state.attempt_count += 1
        self._state.last_restart_time = now.isoformat()
        self._state.total_restarts += 1

        # Calculate next allowed time
        next_delay = self.get_delay(self._state.attempt_count)
        self._state.next_restart_allowed_at = (
            now + timedelta(seconds=next_delay)
        ).isoformat()

        # Record failed attempt
        if not success:
            self._state.failed_attempts.append(
                {
                    "time": now.isoformat(),
                    "error": error or "unknown",
                    "attempt": self._state.attempt_count,
                }
            )
            # Keep only last 10 failures
            self._state.failed_attempts = self._state.failed_attempts[-10:]

        self._save_state()

        logger.info(
            f"Restart recorded: attempt={self._state.attempt_count}, "
            f"next_allowed_at={self._state.next_restart_allowed_at}"
        )

    def record_success(self) -> None:
        """Record a successful restart (convenience method)."""
        self.record_restart(success=True)

    def record_failure(self, error: str) -> None:
        """Record a failed restart (convenience method)."""
        self.record_restart(success=False, error=error)

    def reset(self) -> None:
        """Reset the backoff state (for manual recovery)."""
        self._state = RestartState()
        self._save_state()
        logger.info("Restart backoff state reset")

    def get_status(self) -> dict:
        """Get current backoff status for monitoring."""
        self._check_cooldown_reset()

        return {
            "enabled": self.config.enabled,
            "attempt_count": self._state.attempt_count,
            "max_attempts_per_hour": self.config.max_attempts_per_hour,
            "last_restart_time": self._state.last_restart_time,
            "next_restart_allowed_at": self._state.next_restart_allowed_at,
            "total_restarts": self._state.total_restarts,
            "recent_failures": len(self._state.failed_attempts),
            "base_delay_seconds": self.config.base_delay_seconds,
            "max_delay_seconds": self.config.max_delay_seconds,
            "backoff_multiplier": self.config.backoff_multiplier,
        }


def get_restart_backoff(config: Optional[RestartBackoffConfig] = None) -> RestartBackoff:
    """
    Get singleton RestartBackoff instance.

    Args:
        config: Optional config (only applied on first call)

    Returns:
        Shared RestartBackoff instance
    """
    global _restart_backoff

    if _restart_backoff is None:
        _restart_backoff = RestartBackoff(config=config or RestartBackoffConfig())
        logger.info("Restart backoff initialized")

    return _restart_backoff


def reset_restart_backoff() -> None:
    """Reset singleton for testing."""
    global _restart_backoff
    _restart_backoff = None


# Convenience functions
def should_restart() -> Tuple[bool, float, str]:
    """Check if restart is allowed (convenience function)."""
    return get_restart_backoff().should_restart()


def record_restart_attempt(success: bool = True, error: Optional[str] = None) -> None:
    """Record restart attempt (convenience function)."""
    get_restart_backoff().record_restart(success=success, error=error)
