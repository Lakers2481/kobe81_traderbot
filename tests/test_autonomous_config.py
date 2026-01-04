"""
Tests for the autonomous configuration.

Verifies:
- config/autonomous.yaml loads correctly
- Required sections exist
- Key values are properly set
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestConfigExists:
    """Test config file exists."""

    def test_autonomous_yaml_exists(self):
        """config/autonomous.yaml should exist."""
        config_path = PROJECT_ROOT / "config" / "autonomous.yaml"
        assert config_path.exists(), "autonomous.yaml not found"


class TestConfigLoading:
    """Test config loading."""

    def test_load_autonomous_config(self):
        """Config should load without errors."""
        import yaml

        config_path = PROJECT_ROOT / "config" / "autonomous.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict)

    def test_brain_section_exists(self):
        """Config should have brain section."""
        import yaml

        config_path = PROJECT_ROOT / "config" / "autonomous.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "brain" in config
        assert "cycle_interval" in config["brain"]

    def test_scheduler_section_exists(self):
        """Config should have scheduler section."""
        import yaml

        config_path = PROJECT_ROOT / "config" / "autonomous.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "scheduler" in config

    def test_awareness_section_exists(self):
        """Config should have awareness section."""
        import yaml

        config_path = PROJECT_ROOT / "config" / "autonomous.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "awareness" in config
        assert "phases" in config["awareness"]
        assert "weekend" in config["awareness"]

    def test_pipelines_section_exists(self):
        """Config should have pipelines section."""
        import yaml

        config_path = PROJECT_ROOT / "config" / "autonomous.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "pipelines" in config

    def test_safety_section_exists(self):
        """Config should have safety section."""
        import yaml

        config_path = PROJECT_ROOT / "config" / "autonomous.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "safety" in config
        assert "paper_only" in config["safety"]
        assert config["safety"]["paper_only"] is True


class TestConfigValues:
    """Test specific config values."""

    def test_cycle_interval_is_60(self):
        """Brain cycle interval should be 60 seconds."""
        import yaml

        config_path = PROJECT_ROOT / "config" / "autonomous.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["brain"]["cycle_interval"] == 60

    def test_paper_only_is_true(self):
        """Paper only should be True."""
        import yaml

        config_path = PROJECT_ROOT / "config" / "autonomous.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["safety"]["paper_only"] is True

    def test_all_phases_defined(self):
        """All 9 phases should be defined."""
        import yaml

        config_path = PROJECT_ROOT / "config" / "autonomous.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        phases = config["awareness"]["phases"]
        expected_phases = [
            "pre_market_early",
            "pre_market_active",
            "market_opening",
            "market_morning",
            "market_lunch",
            "market_afternoon",
            "market_close",
            "after_hours",
            "night",
        ]

        for phase in expected_phases:
            assert phase in phases, f"Missing phase: {phase}"

    def test_weekend_mode_enabled(self):
        """Weekend mode should be enabled."""
        import yaml

        config_path = PROJECT_ROOT / "config" / "autonomous.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["awareness"]["weekend"]["enabled"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
