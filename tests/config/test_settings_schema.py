"""
Tests for config/settings_schema.py - Typed config validation.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings_schema import (
    load_validated_settings,
    validate_settings_for_mode,
    get_validated_mode_config,
    require_valid_settings,
    SettingsValidationError,
)


class TestLoadValidatedSettings:
    """Tests for loading and validating settings."""

    def test_loads_default_settings(self):
        """Loads settings with defaults when no file exists."""
        with patch('config.settings_schema._load_yaml_config', return_value={}):
            settings = load_validated_settings()
            # Should have default system settings
            assert hasattr(settings, 'system')

    def test_loads_custom_settings(self):
        """Loads settings from YAML config."""
        mock_config = {
            "system": {
                "name": "TestKobe",
                "mode": "paper",
            },
            "trading_mode": "paper",
        }
        with patch('config.settings_schema._load_yaml_config', return_value=mock_config):
            settings = load_validated_settings()
            assert settings.system.name == "TestKobe"
            assert settings.system.mode == "paper"


class TestValidateSettingsForMode:
    """Tests for mode-specific validation."""

    def test_paper_mode_is_lenient(self):
        """Paper mode allows more lenient settings."""
        mock_config = {
            "system": {"mode": "paper"},
            "modes": {
                "paper": {
                    "max_notional_per_order": 50000,
                    "max_daily_notional": 100000,
                    "max_positions": 20,  # High but OK for paper
                    "risk_per_trade_pct": 0.05,
                }
            },
        }
        with patch('config.settings_schema._load_yaml_config', return_value=mock_config):
            with patch.dict(os.environ, {'ALPACA_API_KEY_ID': 'test', 'ALPACA_API_SECRET_KEY': 'test'}):
                is_valid, errors = validate_settings_for_mode("paper")
                # Paper mode should be more lenient
                # Just checking it doesn't crash

    def test_live_mode_requires_broker_keys(self):
        """Live mode requires Alpaca API keys."""
        mock_config = {
            "system": {"mode": "live"},
            "modes": {
                "real": {
                    "max_notional_per_order": 10000,
                    "max_daily_notional": 20000,
                    "max_positions": 2,
                    "risk_per_trade_pct": 0.02,
                    "max_notional_pct": 0.10,
                }
            },
        }
        with patch('config.settings_schema._load_yaml_config', return_value=mock_config):
            with patch.dict(os.environ, {}, clear=True):
                is_valid, errors = validate_settings_for_mode("live")
                assert "ALPACA_API_KEY_ID" in str(errors)

    def test_live_mode_valid_with_keys(self):
        """Live mode passes with proper configuration."""
        mock_config = {
            "system": {"mode": "live"},
            "modes": {
                "real": {
                    "max_notional_per_order": 10000,
                    "max_daily_notional": 20000,
                    "max_positions": 2,
                    "risk_per_trade_pct": 0.02,
                    "max_notional_pct": 0.10,
                }
            },
        }
        with patch('config.settings_schema._load_yaml_config', return_value=mock_config):
            with patch.dict(os.environ, {
                'ALPACA_API_KEY_ID': 'test_key',
                'ALPACA_API_SECRET_KEY': 'test_secret',
            }):
                is_valid, errors = validate_settings_for_mode("live")
                # Should be valid with proper keys


class TestGetValidatedModeConfig:
    """Tests for getting mode-specific configuration."""

    def test_gets_paper_mode_config(self):
        """Gets paper mode configuration."""
        mock_config = {
            "modes": {
                "paper": {
                    "max_notional_per_order": 15000,
                    "max_daily_notional": 45000,
                    "max_positions": 5,
                    "risk_per_trade_pct": 0.01,
                }
            },
        }
        with patch('config.settings_schema._load_yaml_config', return_value=mock_config):
            config = get_validated_mode_config("paper")
            assert config.get("max_notional_per_order") == 15000

    def test_gets_live_mode_config_as_real(self):
        """Gets 'real' mode config when 'live' is requested."""
        mock_config = {
            "modes": {
                "real": {
                    "max_notional_per_order": 10000,
                    "max_positions": 2,
                    "max_daily_notional": 20000,
                    "risk_per_trade_pct": 0.02,
                }
            },
        }
        with patch('config.settings_schema._load_yaml_config', return_value=mock_config):
            config = get_validated_mode_config("live")
            assert config.get("max_positions") == 2


class TestRequireValidSettings:
    """Tests for strict validation that raises on error."""

    def test_raises_in_live_mode_with_invalid_settings(self):
        """Raises SettingsValidationError in live mode with invalid config."""
        mock_config = {
            "system": {"mode": "paper"},  # Mismatch!
        }
        with patch('config.settings_schema._load_yaml_config', return_value=mock_config):
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(SettingsValidationError):
                    require_valid_settings("live")

    def test_warns_but_continues_in_paper_mode(self):
        """Only warns in paper mode with invalid config."""
        mock_config = {
            "system": {"mode": "live"},  # Mismatch but paper mode is lenient
        }
        with patch('config.settings_schema._load_yaml_config', return_value=mock_config):
            with patch.dict(os.environ, {
                'ALPACA_API_KEY_ID': 'test',
                'ALPACA_API_SECRET_KEY': 'test',
            }):
                # Should not raise in paper mode
                settings = require_valid_settings("paper")
                assert settings is not None


class TestSchemaDefaults:
    """Tests for schema default values."""

    def test_system_config_defaults(self):
        """SystemConfig has reasonable defaults."""
        with patch('config.settings_schema._load_yaml_config', return_value={}):
            settings = load_validated_settings()
            assert settings.system.mode == "paper"
            assert settings.system.timezone == "America/New_York"

    def test_risk_config_defaults(self):
        """RiskConfig has safe defaults."""
        with patch('config.settings_schema._load_yaml_config', return_value={}):
            settings = load_validated_settings()
            assert settings.risk.max_position_size == 0.10
            assert settings.risk.max_open_positions == 10
