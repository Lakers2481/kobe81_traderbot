"""
Unit tests for the VIX Monitor module.

Tests:
1. VIX fetching with cache
2. Pause trading logic
3. Regime adjustment multipliers
4. Graceful degradation on fetch failures
5. Config validation
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.vix_monitor import (
    VIXMonitor,
    VIXConfig,
    VIXReading,
    get_vix_monitor,
    reset_vix_monitor,
    get_vix_level,
    should_pause_for_vix,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_config():
    """Default VIX configuration."""
    return VIXConfig()


@pytest.fixture
def strict_config():
    """Strict VIX configuration with lower thresholds."""
    return VIXConfig(
        pause_threshold=20.0,
        elevated_threshold=15.0,
        extreme_threshold=25.0,
    )


@pytest.fixture
def disabled_config():
    """VIX monitoring disabled."""
    return VIXConfig(pause_enabled=False)


@pytest.fixture
def monitor(default_config):
    """Fresh VIX monitor for each test."""
    reset_vix_monitor()
    return VIXMonitor(config=default_config)


# ============================================================================
# VIXConfig Tests
# ============================================================================


class TestVIXConfig:
    """Tests for VIXConfig dataclass."""

    def test_default_values(self):
        """Default config should have sensible defaults."""
        config = VIXConfig()
        assert config.pause_enabled is True
        assert config.pause_threshold == 30.0
        assert config.elevated_threshold == 25.0
        assert config.extreme_threshold == 40.0
        assert config.cache_ttl_seconds == 3600
        assert config.data_source == "yfinance"
        assert config.fallback_vix == 20.0

    def test_custom_values(self):
        """Custom config values should be applied."""
        config = VIXConfig(
            pause_threshold=25.0,
            data_source="polygon",
        )
        assert config.pause_threshold == 25.0
        assert config.data_source == "polygon"


# ============================================================================
# VIXReading Tests
# ============================================================================


class TestVIXReading:
    """Tests for VIXReading dataclass."""

    def test_age_seconds(self):
        """Age should be calculated correctly."""
        reading = VIXReading(
            level=20.0,
            timestamp=datetime.now() - timedelta(seconds=60),
            source="test",
        )
        assert 59 <= reading.age_seconds <= 61

    def test_is_elevated(self):
        """Elevated check should work."""
        low = VIXReading(level=15.0, timestamp=datetime.now(), source="test")
        high = VIXReading(level=30.0, timestamp=datetime.now(), source="test")

        assert low.is_elevated(threshold=25.0) is False
        assert high.is_elevated(threshold=25.0) is True

    def test_is_extreme(self):
        """Extreme check should work."""
        normal = VIXReading(level=25.0, timestamp=datetime.now(), source="test")
        extreme = VIXReading(level=45.0, timestamp=datetime.now(), source="test")

        assert normal.is_extreme(threshold=40.0) is False
        assert extreme.is_extreme(threshold=40.0) is True


# ============================================================================
# VIXMonitor Tests
# ============================================================================


class TestVIXMonitor:
    """Tests for VIXMonitor class."""

    @patch("core.vix_monitor.VIXMonitor._fetch_from_yfinance")
    def test_fetch_vix_caches_result(self, mock_fetch, monitor):
        """VIX fetch should cache results."""
        mock_fetch.return_value = 18.5

        # First call should fetch
        reading1 = monitor.fetch_vix()
        assert mock_fetch.call_count == 1
        assert reading1.level == 18.5

        # Second call should use cache
        reading2 = monitor.fetch_vix()
        assert mock_fetch.call_count == 1  # No new fetch
        assert reading2.level == 18.5

    @patch("core.vix_monitor.VIXMonitor._fetch_from_yfinance")
    def test_fetch_vix_force_refresh(self, mock_fetch, monitor):
        """Force refresh should bypass cache."""
        mock_fetch.return_value = 18.5

        monitor.fetch_vix()
        mock_fetch.return_value = 22.0
        reading = monitor.fetch_vix(force_refresh=True)

        assert mock_fetch.call_count == 2
        assert reading.level == 22.0

    @patch("core.vix_monitor.VIXMonitor._fetch_from_yfinance")
    def test_fetch_vix_fallback_on_error(self, mock_fetch, monitor):
        """Should use fallback VIX when fetch fails."""
        mock_fetch.side_effect = Exception("Network error")

        reading = monitor.fetch_vix()

        assert reading.level == monitor.config.fallback_vix
        assert reading.source == "fallback"
        assert reading.is_stale is True

    @patch("core.vix_monitor.VIXMonitor._fetch_from_yfinance")
    def test_fetch_vix_stale_cache_on_error(self, mock_fetch, monitor):
        """Should use stale cache when fresh fetch fails."""
        # First successful fetch
        mock_fetch.return_value = 21.0
        monitor.fetch_vix()

        # Subsequent fetch fails
        mock_fetch.side_effect = Exception("Network error")
        reading = monitor.fetch_vix(force_refresh=True)

        # Should return stale cache
        assert reading.level == 21.0
        assert reading.is_stale is True


class TestShouldPauseTrading:
    """Tests for pause trading logic."""

    @patch("core.vix_monitor.VIXMonitor._fetch_from_yfinance")
    def test_pause_when_vix_high(self, mock_fetch, monitor):
        """Should pause when VIX >= threshold."""
        mock_fetch.return_value = 35.0

        should_pause, vix_level, reason = monitor.should_pause_trading()

        assert should_pause is True
        assert vix_level == 35.0
        assert "VIX HIGH" in reason

    @patch("core.vix_monitor.VIXMonitor._fetch_from_yfinance")
    def test_pause_when_vix_extreme(self, mock_fetch, monitor):
        """Should pause when VIX >= extreme threshold."""
        mock_fetch.return_value = 45.0

        should_pause, vix_level, reason = monitor.should_pause_trading()

        assert should_pause is True
        assert vix_level == 45.0
        assert "VIX EXTREME" in reason

    @patch("core.vix_monitor.VIXMonitor._fetch_from_yfinance")
    def test_no_pause_when_vix_normal(self, mock_fetch, monitor):
        """Should not pause when VIX < threshold."""
        mock_fetch.return_value = 18.0

        should_pause, vix_level, reason = monitor.should_pause_trading()

        assert should_pause is False
        assert vix_level == 18.0
        assert "VIX OK" in reason

    def test_no_pause_when_disabled(self, disabled_config):
        """Should never pause when VIX monitoring is disabled."""
        monitor = VIXMonitor(config=disabled_config)

        should_pause, _, reason = monitor.should_pause_trading()

        assert should_pause is False
        assert "disabled" in reason.lower()


class TestRegimeAdjustment:
    """Tests for position sizing based on VIX."""

    @patch("core.vix_monitor.VIXMonitor._fetch_from_yfinance")
    def test_full_size_when_normal(self, mock_fetch, monitor):
        """Full position when VIX is normal."""
        mock_fetch.return_value = 15.0

        multiplier = monitor.get_regime_adjustment()

        assert multiplier == 1.0

    @patch("core.vix_monitor.VIXMonitor._fetch_from_yfinance")
    def test_half_size_when_elevated(self, mock_fetch, monitor):
        """Half position when VIX is elevated."""
        mock_fetch.return_value = 27.0

        multiplier = monitor.get_regime_adjustment()

        assert multiplier == 0.5

    @patch("core.vix_monitor.VIXMonitor._fetch_from_yfinance")
    def test_quarter_size_when_high(self, mock_fetch, monitor):
        """Quarter position when VIX is high."""
        mock_fetch.return_value = 32.0

        multiplier = monitor.get_regime_adjustment()

        assert multiplier == 0.25

    @patch("core.vix_monitor.VIXMonitor._fetch_from_yfinance")
    def test_zero_size_when_extreme(self, mock_fetch, monitor):
        """Zero position when VIX is extreme."""
        mock_fetch.return_value = 45.0

        multiplier = monitor.get_regime_adjustment()

        assert multiplier == 0.0


class TestGetStatus:
    """Tests for status dictionary."""

    @patch("core.vix_monitor.VIXMonitor._fetch_from_yfinance")
    def test_status_contains_required_fields(self, mock_fetch, monitor):
        """Status dict should have all required fields."""
        mock_fetch.return_value = 20.0

        status = monitor.get_status()

        assert "vix_level" in status
        assert "vix_source" in status
        assert "trading_paused" in status
        assert "pause_threshold" in status
        assert "position_multiplier" in status


# ============================================================================
# Singleton Tests
# ============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_vix_monitor_returns_singleton(self):
        """get_vix_monitor should return same instance."""
        reset_vix_monitor()

        m1 = get_vix_monitor()
        m2 = get_vix_monitor()

        assert m1 is m2

    def test_reset_clears_singleton(self):
        """reset_vix_monitor should clear singleton."""
        reset_vix_monitor()
        m1 = get_vix_monitor()
        reset_vix_monitor()
        m2 = get_vix_monitor()

        assert m1 is not m2


# ============================================================================
# Convenience Functions Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @patch("core.vix_monitor.VIXMonitor._fetch_from_yfinance")
    def test_get_vix_level(self, mock_fetch):
        """get_vix_level should return float."""
        reset_vix_monitor()
        mock_fetch.return_value = 19.5

        level = get_vix_level()

        assert level == 19.5

    @patch("core.vix_monitor.VIXMonitor._fetch_from_yfinance")
    def test_should_pause_for_vix(self, mock_fetch):
        """should_pause_for_vix should return tuple."""
        reset_vix_monitor()
        mock_fetch.return_value = 35.0

        should_pause, level, reason = should_pause_for_vix()

        assert should_pause is True
        assert level == 35.0
        assert isinstance(reason, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
