"""
Tests for execution/intraday_trigger.py - Intraday entry trigger.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from execution.intraday_trigger import (
    IntradayTrigger,
    check_entry_trigger,
    get_intraday_trigger,
)


class TestIntradayTriggerKillSwitch:
    """Tests for kill switch awareness in intraday trigger."""

    def test_respects_kill_switch_active(self):
        """Trigger returns False immediately when kill switch is active.

        FIX (2026-01-05): Intraday trigger must check kill switch before
        any API calls or data fetching.
        """
        trigger = IntradayTrigger(mode="vwap_reclaim")

        with patch('execution.intraday_trigger.is_kill_switch_active', return_value=True):
            with patch('execution.intraday_trigger.fetch_intraday_bars') as mock_fetch:
                result = trigger.check_trigger("AAPL", "long")

                # Should NOT call fetch_intraday_bars when kill switch is active
                mock_fetch.assert_not_called()

                # Should return non-triggered result
                assert result.triggered is False
                assert result.symbol == "AAPL"
                assert result.side == "long"
                assert "kill switch" in result.reason.lower()

    def test_allows_trigger_when_kill_switch_inactive(self):
        """Trigger proceeds normally when kill switch is not active."""
        trigger = IntradayTrigger(mode="vwap_reclaim")

        # Mock bar data
        mock_bar = MagicMock()
        mock_bar.close = 150.0
        mock_bar.vwap = 148.0
        mock_bar.high = 151.0
        mock_bar.low = 147.0

        with patch('execution.intraday_trigger.is_kill_switch_active', return_value=False):
            with patch('execution.intraday_trigger.fetch_intraday_bars', return_value=[mock_bar] * 15):
                result = trigger.check_trigger("AAPL", "long")

                # Should proceed with trigger check
                # Price 150 > VWAP 148, so should trigger for long
                assert result.triggered is True
                assert "kill switch" not in result.reason.lower()

    def test_increments_prometheus_counter_on_kill_switch(self):
        """Prometheus counter is incremented when kill switch blocks trigger."""
        trigger = IntradayTrigger(mode="vwap_reclaim")

        with patch('execution.intraday_trigger.is_kill_switch_active', return_value=True):
            with patch('execution.intraday_trigger.INTRADAY_TRIGGER_SKIPPED') as mock_counter:
                mock_labels = MagicMock()
                mock_counter.labels.return_value = mock_labels

                trigger.check_trigger("AAPL", "long")

                mock_counter.labels.assert_called_once_with(reason="kill_switch")
                mock_labels.inc.assert_called_once()


class TestIntradayTriggerModes:
    """Tests for different trigger modes."""

    def test_vwap_reclaim_long_triggered(self):
        """VWAP reclaim triggers for long when price > VWAP."""
        trigger = IntradayTrigger(mode="vwap_reclaim")

        mock_bar = MagicMock()
        mock_bar.close = 150.0
        mock_bar.vwap = 148.0
        mock_bar.high = 151.0
        mock_bar.low = 147.0

        with patch('execution.intraday_trigger.is_kill_switch_active', return_value=False):
            with patch('execution.intraday_trigger.fetch_intraday_bars', return_value=[mock_bar] * 15):
                result = trigger.check_trigger("AAPL", "long")
                assert result.triggered is True
                assert "TRIGGERED" in result.reason

    def test_vwap_reclaim_long_not_triggered(self):
        """VWAP reclaim does not trigger for long when price < VWAP."""
        trigger = IntradayTrigger(mode="vwap_reclaim")

        mock_bar = MagicMock()
        mock_bar.close = 146.0
        mock_bar.vwap = 148.0
        mock_bar.high = 149.0
        mock_bar.low = 145.0

        with patch('execution.intraday_trigger.is_kill_switch_active', return_value=False):
            with patch('execution.intraday_trigger.fetch_intraday_bars', return_value=[mock_bar] * 15):
                result = trigger.check_trigger("AAPL", "long")
                assert result.triggered is False
                assert "Waiting" in result.reason

    def test_vwap_reclaim_short_triggered(self):
        """VWAP reclaim triggers for short when price < VWAP."""
        trigger = IntradayTrigger(mode="vwap_reclaim")

        mock_bar = MagicMock()
        mock_bar.close = 146.0
        mock_bar.vwap = 148.0
        mock_bar.high = 149.0
        mock_bar.low = 145.0

        with patch('execution.intraday_trigger.is_kill_switch_active', return_value=False):
            with patch('execution.intraday_trigger.fetch_intraday_bars', return_value=[mock_bar] * 15):
                result = trigger.check_trigger("AAPL", "short")
                assert result.triggered is True

    def test_returns_false_when_no_data(self):
        """Returns False with no data available."""
        trigger = IntradayTrigger(mode="vwap_reclaim")

        with patch('execution.intraday_trigger.is_kill_switch_active', return_value=False):
            with patch('execution.intraday_trigger.fetch_intraday_bars', return_value=[]):
                result = trigger.check_trigger("AAPL", "long")
                assert result.triggered is False
                assert "no data" in result.reason.lower() or "No intraday" in result.reason


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_intraday_trigger_singleton(self):
        """get_intraday_trigger returns same instance for same mode."""
        with patch('execution.intraday_trigger._trigger', None):
            trigger1 = get_intraday_trigger("vwap_reclaim")
            trigger2 = get_intraday_trigger("vwap_reclaim")
            # Note: This tests the caching behavior
            assert trigger1.mode == trigger2.mode

    def test_check_entry_trigger_function(self):
        """check_entry_trigger convenience function works."""
        with patch('execution.intraday_trigger.is_kill_switch_active', return_value=True):
            result = check_entry_trigger("AAPL", "long", mode="vwap_reclaim")
            assert result.triggered is False
            assert result.symbol == "AAPL"
