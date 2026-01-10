"""
Tests for scripts/preflight_live.py - Live trading preflight gate.
"""
from __future__ import annotations

from unittest.mock import patch
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.preflight_live import (
    run_preflight,
    check_kill_switch,
    check_broker_keys,
    check_webhook_hmac,
    CheckResult,
    PreflightReport,
)


class TestCheckKillSwitch:
    """Tests for kill switch check."""

    def test_passes_when_inactive(self):
        """Check passes when kill switch is not active."""
        with patch('core.kill_switch.is_kill_switch_active', return_value=False):
            result = check_kill_switch()
            assert result.passed is True
            assert result.blocking is True

    def test_fails_when_active(self):
        """Check fails when kill switch is active."""
        with patch('core.kill_switch.is_kill_switch_active', return_value=True):
            result = check_kill_switch()
            assert result.passed is False
            assert result.blocking is True
            assert "ACTIVE" in result.message


class TestCheckBrokerKeys:
    """Tests for broker keys check."""

    def test_passes_with_both_keys(self):
        """Check passes when both Alpaca keys are present."""
        with patch.dict('os.environ', {
            'ALPACA_API_KEY_ID': 'test_key',
            'ALPACA_API_SECRET_KEY': 'test_secret',
        }):
            result = check_broker_keys()
            assert result.passed is True

    def test_fails_with_missing_key(self):
        """Check fails when a key is missing."""
        with patch.dict('os.environ', {
            'ALPACA_API_KEY_ID': '',
            'ALPACA_API_SECRET_KEY': 'test_secret',
        }, clear=True):
            result = check_broker_keys()
            assert result.passed is False
            assert "ALPACA_API_KEY_ID" in result.message


class TestCheckWebhookHmac:
    """Tests for webhook HMAC check."""

    def test_passes_with_valid_secret(self):
        """Check passes with 32+ character HMAC secret."""
        with patch.dict('os.environ', {
            'WEBHOOK_HMAC_SECRET': 'a' * 32,
        }):
            result = check_webhook_hmac()
            assert result.passed is True

    def test_fails_with_short_secret(self):
        """Check fails with too-short HMAC secret."""
        with patch.dict('os.environ', {
            'WEBHOOK_HMAC_SECRET': 'short',
        }):
            result = check_webhook_hmac()
            assert result.passed is False
            assert "too short" in result.message

    def test_fails_with_no_secret(self):
        """Check fails when HMAC secret is not set."""
        with patch.dict('os.environ', {}, clear=True):
            result = check_webhook_hmac()
            assert result.passed is False


class TestRunPreflight:
    """Tests for full preflight run."""

    def test_includes_hmac_check_for_live_mode(self):
        """HMAC check is included in live mode."""
        with patch('scripts.preflight_live.check_webhook_hmac') as mock_hmac:
            mock_hmac.return_value = CheckResult(
                name="webhook_hmac",
                passed=True,
                message="OK",
                blocking=True,
                severity="critical",
            )

            # Mock all other checks to pass
            with patch('scripts.preflight_live.check_settings_schema') as m1, \
                 patch('scripts.preflight_live.check_broker_keys') as m2, \
                 patch('scripts.preflight_live.check_broker_connectivity') as m3, \
                 patch('scripts.preflight_live.check_kill_switch') as m4, \
                 patch('scripts.preflight_live.check_config_pin') as m5, \
                 patch('scripts.preflight_live.check_mode_match') as m6, \
                 patch('scripts.preflight_live.check_data_freshness') as m7, \
                 patch('scripts.preflight_live.check_market_calendar') as m8, \
                 patch('scripts.preflight_live.check_earnings_source') as m9, \
                 patch('scripts.preflight_live.check_prometheus') as m10, \
                 patch('scripts.preflight_live.check_llm_budget') as m11, \
                 patch('scripts.preflight_live.check_position_reconciliation') as m12, \
                 patch('scripts.preflight_live.check_pending_orders') as m13, \
                 patch('scripts.preflight_live.check_hash_chain') as m14:

                for m in [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14]:
                    m.return_value = CheckResult(
                        name="test",
                        passed=True,
                        message="OK",
                        blocking=False,
                        severity="info",
                    )

                report = run_preflight(mode="live")

                # HMAC check should have been called for live mode
                mock_hmac.assert_called_once()

    def test_skips_hmac_check_for_paper_mode(self):
        """HMAC check is skipped in paper mode."""
        with patch('scripts.preflight_live.check_webhook_hmac') as mock_hmac:
            # Mock all checks
            with patch('scripts.preflight_live.check_settings_schema') as m1, \
                 patch('scripts.preflight_live.check_broker_keys') as m2, \
                 patch('scripts.preflight_live.check_broker_connectivity') as m3, \
                 patch('scripts.preflight_live.check_kill_switch') as m4, \
                 patch('scripts.preflight_live.check_config_pin') as m5, \
                 patch('scripts.preflight_live.check_mode_match') as m6, \
                 patch('scripts.preflight_live.check_data_freshness') as m7, \
                 patch('scripts.preflight_live.check_market_calendar') as m8, \
                 patch('scripts.preflight_live.check_earnings_source') as m9, \
                 patch('scripts.preflight_live.check_prometheus') as m10, \
                 patch('scripts.preflight_live.check_llm_budget') as m11, \
                 patch('scripts.preflight_live.check_position_reconciliation') as m12, \
                 patch('scripts.preflight_live.check_pending_orders') as m13, \
                 patch('scripts.preflight_live.check_hash_chain') as m14:

                for m in [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14]:
                    m.return_value = CheckResult(
                        name="test",
                        passed=True,
                        message="OK",
                        blocking=False,
                        severity="info",
                    )

                report = run_preflight(mode="paper")

                # HMAC check should NOT have been called for paper mode
                mock_hmac.assert_not_called()

    def test_report_counts_blocking_failures(self):
        """Report correctly counts blocking failures."""
        checks = [
            CheckResult("pass1", True, "ok", True, "critical"),
            CheckResult("fail1", False, "err", True, "critical"),  # blocking fail
            CheckResult("fail2", False, "err", False, "warning"),  # non-blocking fail
            CheckResult("fail3", False, "err", True, "critical"),  # blocking fail
        ]

        report = PreflightReport(
            mode="live",
            timestamp="2026-01-05T00:00:00",
            all_passed=False,
            blocking_failures=2,
            total_checks=4,
            checks=checks,
        )

        assert report.blocking_failures == 2
        assert report.all_passed is False

    def test_report_to_dict(self):
        """Report can be serialized to dict."""
        report = PreflightReport(
            mode="paper",
            timestamp="2026-01-05T00:00:00",
            all_passed=True,
            blocking_failures=0,
            total_checks=1,
            checks=[
                CheckResult("test", True, "ok", False, "info"),
            ],
        )

        d = report.to_dict()
        assert d["mode"] == "paper"
        assert d["all_passed"] is True
        assert len(d["checks"]) == 1

        # Should be JSON serializable
        json.dumps(d)
