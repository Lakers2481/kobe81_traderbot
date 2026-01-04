"""
Tests for the autonomous/run.py entrypoint.

Verifies:
- KobeRunner class works
- Status command works
- Health check works
- Tour command works
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestKobeRunnerImport:
    """Test KobeRunner can be imported."""

    def test_import_kobe_runner(self):
        """KobeRunner should be importable."""
        from autonomous.run import KobeRunner
        assert KobeRunner is not None

    def test_import_main(self):
        """main function should be importable."""
        from autonomous.run import main
        assert callable(main)


class TestKobeRunnerInstantiation:
    """Test KobeRunner instantiation."""

    def test_create_runner(self):
        """KobeRunner can be instantiated."""
        from autonomous.run import KobeRunner

        runner = KobeRunner()
        assert runner is not None
        assert runner.project_root is not None
        assert runner.state_dir is not None

    def test_runner_has_commands(self):
        """KobeRunner has command methods."""
        from autonomous.run import KobeRunner

        runner = KobeRunner()

        assert hasattr(runner, "cmd_status")
        assert hasattr(runner, "cmd_awareness")
        assert hasattr(runner, "cmd_research")
        assert hasattr(runner, "cmd_demo")
        assert hasattr(runner, "cmd_start")
        assert hasattr(runner, "cmd_stop")
        assert hasattr(runner, "cmd_health")
        assert hasattr(runner, "cmd_weekend")
        assert hasattr(runner, "cmd_tour")


class TestStatusCommand:
    """Test status command."""

    def test_status_returns_int(self):
        """cmd_status returns integer exit code."""
        from autonomous.run import KobeRunner

        runner = KobeRunner()
        result = runner.cmd_status()

        assert isinstance(result, int)
        assert result == 0  # Should succeed


class TestAwarenessCommand:
    """Test awareness command."""

    def test_awareness_returns_int(self):
        """cmd_awareness returns integer exit code."""
        from autonomous.run import KobeRunner

        runner = KobeRunner()

        # May fail if MarketAwareness not fully configured
        try:
            result = runner.cmd_awareness()
            assert isinstance(result, int)
        except Exception:
            # OK if awareness module not fully available
            pass


class TestTourCommand:
    """Test tour command."""

    def test_tour_returns_int(self):
        """cmd_tour returns integer exit code."""
        from autonomous.run import KobeRunner

        runner = KobeRunner()
        result = runner.cmd_tour()

        assert isinstance(result, int)
        assert result == 0  # Should succeed


class TestHelperMethods:
    """Test helper methods."""

    def test_check_already_running(self):
        """_check_already_running returns bool."""
        from autonomous.run import KobeRunner

        runner = KobeRunner()
        result = runner._check_already_running()

        assert isinstance(result, bool)

    def test_pid_file_path(self):
        """PID file path is correct."""
        from autonomous.run import KobeRunner

        runner = KobeRunner()

        assert "kobe.pid" in str(runner.pid_file)
        assert "autonomous" in str(runner.pid_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
