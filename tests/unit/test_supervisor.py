"""
Unit tests for self-healing supervisor.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import Mock, patch

from ops.supervisor import (
    Supervisor,
    SupervisorMode,
    HealthCheck,
    HealthStatus,
    ComponentHealth,
)

ET = ZoneInfo("America/New_York")


class TestHealthStatus:
    """Tests for health status enum."""

    def test_status_values(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestHealthCheck:
    """Tests for health check dataclass."""

    def test_healthy_check(self):
        check = HealthCheck(
            component="broker",
            status=HealthStatus.HEALTHY,
            message="Connected",
            timestamp=datetime.now(ET),
        )
        assert check.status == HealthStatus.HEALTHY
        assert check.component == "broker"

    def test_unhealthy_check(self):
        check = HealthCheck(
            component="data_provider",
            status=HealthStatus.UNHEALTHY,
            message="API timeout",
            timestamp=datetime.now(ET),
            last_success=datetime.now(ET) - timedelta(minutes=30),
        )
        assert check.status == HealthStatus.UNHEALTHY
        assert check.last_success is not None

    def test_to_dict(self):
        check = HealthCheck(
            component="broker",
            status=HealthStatus.HEALTHY,
            message="OK",
            timestamp=datetime.now(ET),
        )
        d = check.to_dict()
        assert d["component"] == "broker"
        assert d["status"] == "healthy"


class TestComponentHealth:
    """Tests for component health tracking."""

    def test_record_success(self):
        health = ComponentHealth("test_component")
        health.record_success()

        assert health.consecutive_failures == 0
        assert health.last_success is not None

    def test_record_failure(self):
        health = ComponentHealth("test_component")
        health.record_failure("Connection error")

        assert health.consecutive_failures == 1
        assert health.last_error == "Connection error"

    def test_consecutive_failures(self):
        health = ComponentHealth("test_component")

        for i in range(5):
            health.record_failure(f"Error {i}")

        assert health.consecutive_failures == 5

    def test_failure_reset_on_success(self):
        health = ComponentHealth("test_component")

        health.record_failure("Error")
        health.record_failure("Error")
        assert health.consecutive_failures == 2

        health.record_success()
        assert health.consecutive_failures == 0

    def test_error_rate(self):
        health = ComponentHealth("test_component")

        # 3 successes, 2 failures
        for _ in range(3):
            health.record_success()
        for _ in range(2):
            health.record_failure("Error")

        rate = health.get_error_rate(window_minutes=60)
        assert rate > 0  # Should have some error rate


class TestSupervisor:
    """Tests for supervisor."""

    def test_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            supervisor = Supervisor(state_dir=Path(tmpdir))
            assert supervisor.mode == SupervisorMode.NORMAL

    def test_check_health_all_healthy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            supervisor = Supervisor(state_dir=Path(tmpdir))

            # Mock all components as healthy
            supervisor._components["broker"].record_success()
            supervisor._components["data_provider"].record_success()
            supervisor._components["risk_gate"].record_success()

            health = supervisor.check_health()

            assert all(h.status == HealthStatus.HEALTHY for h in health.values())

    def test_check_degraded_on_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            supervisor = Supervisor(
                state_dir=Path(tmpdir),
                degraded_threshold=3,
            )

            # Simulate failures
            for _ in range(3):
                supervisor._components["broker"].record_failure("Connection timeout")

            health = supervisor.check_health()

            broker_health = health.get("broker")
            assert broker_health.status == HealthStatus.DEGRADED

    def test_activate_safe_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            supervisor = Supervisor(state_dir=Path(tmpdir))

            supervisor.activate_safe_mode("Critical broker failure")

            assert supervisor.mode == SupervisorMode.SAFE
            # Check kill switch file exists
            kill_switch = Path(tmpdir) / "KILL_SWITCH"
            assert kill_switch.exists()

    def test_deactivate_safe_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            supervisor = Supervisor(state_dir=Path(tmpdir))

            supervisor.activate_safe_mode("Test")
            assert supervisor.mode == SupervisorMode.SAFE

            supervisor.deactivate_safe_mode()
            assert supervisor.mode == SupervisorMode.NORMAL
            # Kill switch should be removed
            kill_switch = Path(tmpdir) / "KILL_SWITCH"
            assert not kill_switch.exists()

    def test_auto_safe_mode_on_critical(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            supervisor = Supervisor(
                state_dir=Path(tmpdir),
                safe_mode_threshold=5,
            )

            # Trigger critical failures
            for _ in range(6):
                supervisor._components["broker"].record_failure("Critical error")

            supervisor._evaluate_mode()

            assert supervisor.mode == SupervisorMode.SAFE

    def test_should_restart_component(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            supervisor = Supervisor(
                state_dir=Path(tmpdir),
                restart_threshold=3,
            )

            # Not enough failures for restart
            supervisor._components["data_provider"].record_failure("Error")
            assert not supervisor.should_restart("data_provider")

            # Enough failures for restart
            for _ in range(3):
                supervisor._components["data_provider"].record_failure("Error")
            assert supervisor.should_restart("data_provider")

    def test_get_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            supervisor = Supervisor(state_dir=Path(tmpdir))

            status = supervisor.get_status()

            assert "mode" in status
            assert "components" in status
            assert status["mode"] == "NORMAL"

    def test_from_config(self):
        config = {
            "check_interval": 30,
            "degraded_threshold": 5,
            "safe_mode_threshold": 10,
            "restart_threshold": 3,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            supervisor = Supervisor.from_config(config, state_dir=Path(tmpdir))

            assert supervisor.check_interval == 30
            assert supervisor.degraded_threshold == 5


class TestSupervisorPersistence:
    """Tests for supervisor state persistence."""

    def test_save_and_load_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)

            # Create and record some state
            supervisor1 = Supervisor(state_dir=state_dir)
            supervisor1._components["broker"].record_failure("Error")
            supervisor1._components["broker"].record_failure("Error")
            supervisor1.save_state()

            # Load in new supervisor
            supervisor2 = Supervisor(state_dir=state_dir)
            supervisor2.load_state()

            # State should be preserved
            assert supervisor2._components["broker"].consecutive_failures == 2


class TestSupervisorModes:
    """Tests for supervisor mode transitions."""

    def test_normal_to_degraded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            supervisor = Supervisor(
                state_dir=Path(tmpdir),
                degraded_threshold=3,
            )

            assert supervisor.mode == SupervisorMode.NORMAL

            # Trigger degraded
            for _ in range(4):
                supervisor._components["broker"].record_failure("Error")

            supervisor._evaluate_mode()

            assert supervisor.mode == SupervisorMode.DEGRADED

    def test_degraded_to_safe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            supervisor = Supervisor(
                state_dir=Path(tmpdir),
                degraded_threshold=3,
                safe_mode_threshold=6,
            )

            # First go to degraded
            for _ in range(4):
                supervisor._components["broker"].record_failure("Error")
            supervisor._evaluate_mode()
            assert supervisor.mode == SupervisorMode.DEGRADED

            # Then to safe
            for _ in range(3):
                supervisor._components["broker"].record_failure("Error")
            supervisor._evaluate_mode()
            assert supervisor.mode == SupervisorMode.SAFE

    def test_recovery_from_degraded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            supervisor = Supervisor(
                state_dir=Path(tmpdir),
                degraded_threshold=3,
            )

            # Go to degraded
            for _ in range(4):
                supervisor._components["broker"].record_failure("Error")
            supervisor._evaluate_mode()
            assert supervisor.mode == SupervisorMode.DEGRADED

            # Recover with successes
            for _ in range(5):
                supervisor._components["broker"].record_success()
            supervisor._evaluate_mode()

            assert supervisor.mode == SupervisorMode.NORMAL
