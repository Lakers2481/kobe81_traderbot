"""
Unit tests for the Restart Backoff module.

Tests:
1. Exponential delay calculation
2. Max attempts enforcement
3. State persistence
4. Cooldown reset
5. Jitter behavior
"""

import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.restart_backoff import (
    RestartBackoff,
    RestartBackoffConfig,
    RestartState,
    get_restart_backoff,
    reset_restart_backoff,
    should_restart,
    record_restart_attempt,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_state_dir():
    """Create a temporary directory for state files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def config(temp_state_dir):
    """Default config with temp state file."""
    return RestartBackoffConfig(
        state_file=temp_state_dir / "restart_backoff.json",
        base_delay_seconds=10.0,
        max_delay_seconds=60.0,
        max_attempts_per_hour=3,
        jitter_enabled=False,  # Disable for deterministic tests
    )


@pytest.fixture
def backoff(config):
    """Fresh RestartBackoff for each test."""
    reset_restart_backoff()
    return RestartBackoff(config=config)


# ============================================================================
# RestartBackoffConfig Tests
# ============================================================================


class TestRestartBackoffConfig:
    """Tests for RestartBackoffConfig dataclass."""

    def test_default_values(self):
        """Default config should have sensible defaults."""
        config = RestartBackoffConfig()
        assert config.enabled is True
        assert config.base_delay_seconds == 30.0
        assert config.max_delay_seconds == 3600.0
        assert config.backoff_multiplier == 2.0
        assert config.max_attempts_per_hour == 5
        assert config.jitter_enabled is True
        assert config.cooldown_hours == 1.0

    def test_custom_values(self):
        """Custom config values should be applied."""
        config = RestartBackoffConfig(
            base_delay_seconds=60.0,
            max_attempts_per_hour=10,
        )
        assert config.base_delay_seconds == 60.0
        assert config.max_attempts_per_hour == 10


# ============================================================================
# RestartState Tests
# ============================================================================


class TestRestartState:
    """Tests for RestartState dataclass."""

    def test_to_dict(self):
        """State should serialize to dictionary."""
        state = RestartState(
            component="scheduler",
            attempt_count=2,
            last_restart_time="2025-01-01T12:00:00",
        )
        d = state.to_dict()

        assert d["component"] == "scheduler"
        assert d["attempt_count"] == 2
        assert d["last_restart_time"] == "2025-01-01T12:00:00"

    def test_from_dict(self):
        """State should deserialize from dictionary."""
        data = {
            "component": "runner",
            "attempt_count": 3,
            "last_restart_time": "2025-01-01T13:00:00",
            "total_restarts": 10,
        }
        state = RestartState.from_dict(data)

        assert state.component == "runner"
        assert state.attempt_count == 3
        assert state.total_restarts == 10


# ============================================================================
# RestartBackoff Tests
# ============================================================================


class TestRestartBackoff:
    """Tests for RestartBackoff class."""

    def test_get_delay_exponential(self, backoff):
        """Delay should increase exponentially."""
        # base = 10, multiplier = 2
        assert backoff.get_delay(0) == 10.0  # 10 * 2^0
        assert backoff.get_delay(1) == 20.0  # 10 * 2^1
        assert backoff.get_delay(2) == 40.0  # 10 * 2^2

    def test_get_delay_capped(self, backoff):
        """Delay should be capped at max."""
        # max = 60
        assert backoff.get_delay(10) == 60.0  # Would be 10240, capped to 60

    def test_should_restart_first_attempt(self, backoff):
        """First restart should be allowed immediately."""
        allowed, delay, reason = backoff.should_restart()

        assert allowed is True
        assert delay >= 0  # May be the initial delay
        assert "restart_allowed" in reason

    def test_should_restart_blocked_after_max(self, backoff):
        """Restarts should be blocked after max attempts."""
        # Record max attempts
        for _ in range(backoff.config.max_attempts_per_hour):
            backoff.record_restart()

        allowed, delay, reason = backoff.should_restart()

        assert allowed is False
        assert "max_attempts_exceeded" in reason

    def test_record_restart_increments_counter(self, backoff):
        """Recording restart should increment counter."""
        assert backoff._state.attempt_count == 0

        backoff.record_restart()
        assert backoff._state.attempt_count == 1

        backoff.record_restart()
        assert backoff._state.attempt_count == 2

    def test_record_restart_persists_state(self, backoff, config):
        """State should be persisted to disk."""
        backoff.record_restart()

        # Read state file directly
        assert config.state_file.exists()
        data = json.loads(config.state_file.read_text())
        assert data["attempt_count"] == 1

    def test_state_loads_on_init(self, config):
        """State should be loaded from disk on init."""
        # Create pre-existing state file
        state_data = {
            "component": "scheduler",
            "attempt_count": 2,
            "last_restart_time": "2025-01-01T12:00:00",
            "total_restarts": 5,
        }
        config.state_file.parent.mkdir(parents=True, exist_ok=True)
        config.state_file.write_text(json.dumps(state_data))

        # Create new backoff instance
        backoff = RestartBackoff(config=config)

        assert backoff._state.attempt_count == 2
        assert backoff._state.total_restarts == 5


class TestCooldownReset:
    """Tests for cooldown-based counter reset."""

    def test_reset_after_cooldown(self, temp_state_dir):
        """Attempt counter should reset after cooldown period."""
        config = RestartBackoffConfig(
            state_file=temp_state_dir / "restart_backoff.json",
            cooldown_hours=0.001,  # ~3.6 seconds for fast test
            max_attempts_per_hour=5,
        )

        backoff = RestartBackoff(config=config)

        # Record some attempts
        backoff.record_restart()
        backoff.record_restart()
        assert backoff._state.attempt_count == 2

        # Manually set last restart to be old enough
        old_time = datetime.now() - timedelta(hours=1)
        backoff._state.last_restart_time = old_time.isoformat()
        backoff._save_state()

        # Check should_restart - should trigger cooldown reset
        allowed, _, _ = backoff.should_restart()

        assert allowed is True
        assert backoff._state.attempt_count == 0  # Reset!


class TestJitter:
    """Tests for jitter behavior."""

    def test_jitter_adds_randomness(self, temp_state_dir):
        """Jitter should add randomness to delays."""
        config = RestartBackoffConfig(
            state_file=temp_state_dir / "restart_backoff.json",
            base_delay_seconds=100.0,
            jitter_enabled=True,
            jitter_factor=0.5,  # +/- 50%
        )

        backoff = RestartBackoff(config=config)

        # Get multiple delays
        delays = [backoff.get_delay(0) for _ in range(10)]

        # Should have some variation
        unique_delays = set(delays)
        assert len(unique_delays) > 1  # Not all the same

        # Should be within jitter range (50-150)
        for d in delays:
            assert 50 <= d <= 150


class TestDisabled:
    """Tests for disabled backoff."""

    def test_always_allowed_when_disabled(self, temp_state_dir):
        """Restarts should always be allowed when disabled."""
        config = RestartBackoffConfig(
            state_file=temp_state_dir / "restart_backoff.json",
            enabled=False,
        )

        backoff = RestartBackoff(config=config)

        # Record many attempts
        for _ in range(10):
            backoff.record_restart()

        # Should still be allowed
        allowed, delay, reason = backoff.should_restart()

        assert allowed is True
        assert delay == 0.0
        assert "disabled" in reason.lower()


class TestGetStatus:
    """Tests for status dictionary."""

    def test_status_contains_required_fields(self, backoff):
        """Status dict should have all required fields."""
        backoff.record_restart()
        status = backoff.get_status()

        assert "enabled" in status
        assert "attempt_count" in status
        assert "max_attempts_per_hour" in status
        assert "last_restart_time" in status
        assert "total_restarts" in status
        assert "base_delay_seconds" in status


# ============================================================================
# Singleton Tests
# ============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_restart_backoff_returns_singleton(self, temp_state_dir):
        """get_restart_backoff should return same instance."""
        reset_restart_backoff()

        config = RestartBackoffConfig(
            state_file=temp_state_dir / "restart_backoff.json"
        )

        b1 = get_restart_backoff(config)
        b2 = get_restart_backoff()

        assert b1 is b2

    def test_reset_clears_singleton(self, temp_state_dir):
        """reset_restart_backoff should clear singleton."""
        reset_restart_backoff()

        config = RestartBackoffConfig(
            state_file=temp_state_dir / "restart_backoff.json"
        )

        b1 = get_restart_backoff(config)
        reset_restart_backoff()
        b2 = get_restart_backoff(config)

        assert b1 is not b2


# ============================================================================
# Convenience Functions Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_should_restart_function(self, temp_state_dir):
        """should_restart should work."""
        reset_restart_backoff()

        config = RestartBackoffConfig(
            state_file=temp_state_dir / "restart_backoff.json",
            jitter_enabled=False,
        )
        get_restart_backoff(config)

        allowed, delay, reason = should_restart()

        assert allowed is True
        assert isinstance(delay, float)
        assert isinstance(reason, str)

    def test_record_restart_attempt_function(self, temp_state_dir):
        """record_restart_attempt should work."""
        reset_restart_backoff()

        config = RestartBackoffConfig(
            state_file=temp_state_dir / "restart_backoff.json"
        )
        backoff = get_restart_backoff(config)

        record_restart_attempt(success=True)

        assert backoff._state.attempt_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
