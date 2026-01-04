"""
Integration test for demo run.

Verifies:
- Demo command runs without errors
- Status command works
- Tour command works
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestDemoRun:
    """Test demo functionality."""

    def test_status_runs(self):
        """Status command should run without error."""
        from autonomous.run import KobeRunner

        runner = KobeRunner()
        exit_code = runner.cmd_status()

        assert exit_code == 0

    def test_tour_runs(self):
        """Tour command should run without error."""
        from autonomous.run import KobeRunner

        runner = KobeRunner()
        exit_code = runner.cmd_tour()

        assert exit_code == 0

    def test_research_runs(self):
        """Research command should run without error."""
        from autonomous.run import KobeRunner

        runner = KobeRunner()
        exit_code = runner.cmd_research()

        assert exit_code == 0


class TestHealthCheck:
    """Test health check functionality."""

    def test_health_check_returns_int(self):
        """Health check should return integer."""
        from autonomous.run import KobeRunner

        runner = KobeRunner()
        exit_code = runner.cmd_health()

        # May return 0 (healthy) or 1 (unhealthy)
        # Both are valid results
        assert isinstance(exit_code, int)
        assert exit_code in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
