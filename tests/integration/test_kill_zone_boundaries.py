"""
INTEGRATION TESTS: Kill Zone Time Boundaries (CRITICAL)

Tests ICT-style time-based trade blocking to ensure:
- No trades during opening range (9:30-10:00 AM)
- No trades during lunch chop (11:30 AM - 2:00 PM)
- No trades during close (3:30-4:00 PM)
- Proper handling of DST transitions
- Correct zone identification

This is CRITICAL because trading outside valid windows is a major
source of losses for algorithmic systems.

Author: Kobe Trading System
Version: 1.0.0
Date: 2026-01-06
"""

import pytest
from datetime import datetime, time as dtime
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Try to import freezegun for time manipulation
try:
    from freezegun import freeze_time
    HAS_FREEZEGUN = True
except ImportError:
    HAS_FREEZEGUN = False
    freeze_time = lambda x: lambda f: f  # No-op decorator


def skip_if_no_freezegun():
    """Skip test if freezegun not available."""
    if not HAS_FREEZEGUN:
        pytest.skip("freezegun not installed")


@pytest.mark.integration
@pytest.mark.kill_zone
class TestOpeningRangeBlocked:
    """Test that opening range (9:30-10:00 AM) is blocked."""

    def test_935_am_blocked(self):
        """Trading at 9:35 AM should be blocked."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        # Mock current time to 9:35 AM ET
        with freeze_time("2026-01-06 09:35:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is False
            assert status.current_zone == KillZone.OPENING_RANGE
            assert "opening" in status.reason.lower()

    def test_930_am_blocked(self):
        """Trading exactly at market open (9:30 AM) should be blocked."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        with freeze_time("2026-01-06 09:30:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is False
            assert status.current_zone == KillZone.OPENING_RANGE

    def test_959_am_blocked(self):
        """Trading at 9:59 AM should still be blocked."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        with freeze_time("2026-01-06 09:59:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is False
            assert status.current_zone == KillZone.OPENING_RANGE


@pytest.mark.integration
@pytest.mark.kill_zone
class TestPrimaryWindowAllowed:
    """Test that primary window (10:00-11:30 AM) allows trading."""

    def test_1000_am_allowed(self):
        """Trading at exactly 10:00 AM should be allowed."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        with freeze_time("2026-01-06 10:00:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is True
            assert status.current_zone == KillZone.LONDON_CLOSE

    def test_1005_am_allowed(self):
        """Trading at 10:05 AM should be allowed."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        with freeze_time("2026-01-06 10:05:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is True
            assert status.current_zone == KillZone.LONDON_CLOSE

    def test_1100_am_allowed(self):
        """Trading at 11:00 AM should be allowed."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        with freeze_time("2026-01-06 11:00:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is True
            assert status.current_zone == KillZone.LONDON_CLOSE


@pytest.mark.integration
@pytest.mark.kill_zone
class TestLunchChopBlocked:
    """Test that lunch chop (11:30 AM - 2:00 PM) is blocked."""

    def test_1200_pm_blocked(self):
        """Trading at 12:00 PM should be blocked (lunch chop)."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        with freeze_time("2026-01-06 12:00:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is False
            assert status.current_zone == KillZone.LUNCH_CHOP

    def test_1300_pm_blocked(self):
        """Trading at 1:00 PM should be blocked (lunch chop)."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        with freeze_time("2026-01-06 13:00:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is False
            assert status.current_zone == KillZone.LUNCH_CHOP

    def test_lunch_can_be_disabled(self):
        """Lunch blocking can be disabled for aggressive mode."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZoneConfig

        config = KillZoneConfig(block_lunch=False)
        gate = KillZoneGate(config=config)

        with freeze_time("2026-01-06 12:00:00-05:00"):
            status = gate.check_can_trade()

            # Should be allowed when lunch blocking disabled
            assert status.can_trade is True


@pytest.mark.integration
@pytest.mark.kill_zone
class TestPowerHourAllowed:
    """Test that power hour (2:30-3:30 PM) allows trading."""

    def test_1430_pm_allowed(self):
        """Trading at 2:30 PM should be allowed (power hour start)."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        with freeze_time("2026-01-06 14:30:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is True
            assert status.current_zone == KillZone.POWER_HOUR

    def test_1500_pm_allowed(self):
        """Trading at 3:00 PM should be allowed (mid power hour)."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        with freeze_time("2026-01-06 15:00:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is True
            assert status.current_zone == KillZone.POWER_HOUR


@pytest.mark.integration
@pytest.mark.kill_zone
class TestCloseBlocked:
    """Test that close period (3:30-4:00 PM) is blocked."""

    def test_1545_pm_blocked(self):
        """Trading at 3:45 PM should be blocked (close period)."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        with freeze_time("2026-01-06 15:45:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is False
            assert status.current_zone == KillZone.CLOSE
            assert "close" in status.reason.lower()

    def test_1530_pm_blocked(self):
        """Trading at exactly 3:30 PM should be blocked."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        with freeze_time("2026-01-06 15:30:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is False
            assert status.current_zone == KillZone.CLOSE


@pytest.mark.integration
@pytest.mark.kill_zone
class TestPreMarketBlocked:
    """Test that pre-market is blocked."""

    def test_0800_am_blocked(self):
        """Trading at 8:00 AM should be blocked (pre-market)."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        with freeze_time("2026-01-06 08:00:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is False
            assert status.current_zone == KillZone.PRE_MARKET

    def test_0929_am_blocked(self):
        """Trading at 9:29 AM should be blocked (pre-market)."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        with freeze_time("2026-01-06 09:29:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is False
            assert status.current_zone == KillZone.PRE_MARKET


@pytest.mark.integration
@pytest.mark.kill_zone
class TestAfterHoursBlocked:
    """Test that after hours is blocked."""

    def test_1700_pm_blocked(self):
        """Trading at 5:00 PM should be blocked (after hours)."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        with freeze_time("2026-01-06 17:00:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is False
            assert status.current_zone == KillZone.AFTER_HOURS

    def test_1600_pm_blocked(self):
        """Trading at exactly 4:00 PM should be blocked (market closed)."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        with freeze_time("2026-01-06 16:00:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is False
            assert status.current_zone == KillZone.AFTER_HOURS


@pytest.mark.integration
@pytest.mark.kill_zone
class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_can_trade_now_function(self):
        """Test can_trade_now() convenience function."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import can_trade_now

        with freeze_time("2026-01-06 10:30:00-05:00"):
            assert can_trade_now() is True

        with freeze_time("2026-01-06 09:35:00-05:00"):
            assert can_trade_now() is False

    def test_get_current_zone_function(self):
        """Test get_current_zone() convenience function."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import get_current_zone, KillZone

        with freeze_time("2026-01-06 10:30:00-05:00"):
            assert get_current_zone() == KillZone.LONDON_CLOSE

        with freeze_time("2026-01-06 12:00:00-05:00"):
            assert get_current_zone() == KillZone.LUNCH_CHOP

    def test_check_trade_allowed_function(self):
        """Test check_trade_allowed() convenience function."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import check_trade_allowed

        with freeze_time("2026-01-06 10:30:00-05:00"):
            allowed, reason = check_trade_allowed()
            assert allowed is True

        with freeze_time("2026-01-06 09:35:00-05:00"):
            allowed, reason = check_trade_allowed()
            assert allowed is False
            assert len(reason) > 0


@pytest.mark.integration
@pytest.mark.kill_zone
class TestZoneTransitions:
    """Test behavior at zone transition boundaries."""

    def test_opening_to_primary_transition(self):
        """Test transition from opening range to primary window."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        # Just before transition (9:59:59)
        with freeze_time("2026-01-06 09:59:59-05:00"):
            status = gate.check_can_trade()
            assert status.can_trade is False
            assert status.current_zone == KillZone.OPENING_RANGE

        # Just after transition (10:00:00)
        with freeze_time("2026-01-06 10:00:00-05:00"):
            status = gate.check_can_trade()
            assert status.can_trade is True
            assert status.current_zone == KillZone.LONDON_CLOSE

    def test_primary_to_lunch_transition(self):
        """Test transition from primary window to lunch chop."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        # Just before transition (11:29:59)
        with freeze_time("2026-01-06 11:29:59-05:00"):
            status = gate.check_can_trade()
            assert status.can_trade is True
            assert status.current_zone == KillZone.LONDON_CLOSE

        # Just after transition (11:30:00)
        with freeze_time("2026-01-06 11:30:00-05:00"):
            status = gate.check_can_trade()
            assert status.can_trade is False
            assert status.current_zone == KillZone.LUNCH_CHOP


@pytest.mark.integration
@pytest.mark.kill_zone
class TestZoneDetectionWithoutMocking:
    """Test zone detection without freezegun (uses actual time)."""

    def test_zone_enum_values(self):
        """Verify all expected zones exist."""
        from risk.kill_zone_gate import KillZone

        expected_zones = [
            "PRE_MARKET",
            "OPENING_RANGE",
            "LONDON_CLOSE",
            "LUNCH_CHOP",
            "POWER_HOUR",
            "CLOSE",
            "AFTER_HOURS",
        ]

        for zone_name in expected_zones:
            assert hasattr(KillZone, zone_name), f"Missing zone: {zone_name}"

    def test_gate_initialization(self):
        """Verify gate initializes correctly."""
        from risk.kill_zone_gate import KillZoneGate, KillZoneConfig

        # Default config
        gate = KillZoneGate()
        assert gate.config.block_opening_range is True
        assert gate.config.block_lunch is True
        assert gate.config.block_close is True

        # Custom config
        custom_config = KillZoneConfig(block_lunch=False)
        gate = KillZoneGate(config=custom_config)
        assert gate.config.block_lunch is False

    def test_status_has_next_window_info(self):
        """Verify status includes next window information when blocked."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate

        gate = KillZoneGate()

        # During opening range
        with freeze_time("2026-01-06 09:35:00-05:00"):
            status = gate.check_can_trade()

            assert status.can_trade is False
            assert status.next_window_opens is not None
            # Next window should be primary (10:00)
            assert status.next_window_opens == dtime(10, 0)


@pytest.mark.integration
@pytest.mark.kill_zone
@pytest.mark.slow
class TestDSTTransitions:
    """Test Daylight Saving Time transition handling."""

    def test_spring_forward_transition(self):
        """Test DST spring forward (March)."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        # Day before DST (first Sunday of March at 10:00 AM)
        # In 2026, DST starts March 8
        with freeze_time("2026-03-07 10:00:00-05:00"):  # Still EST
            status = gate.check_can_trade()
            assert status.can_trade is True
            assert status.current_zone == KillZone.LONDON_CLOSE

        # Day after DST at same "wall clock" time
        with freeze_time("2026-03-09 10:00:00-04:00"):  # Now EDT
            status = gate.check_can_trade()
            assert status.can_trade is True
            assert status.current_zone == KillZone.LONDON_CLOSE

    def test_fall_back_transition(self):
        """Test DST fall back (November)."""
        skip_if_no_freezegun()

        from risk.kill_zone_gate import KillZoneGate, KillZone

        gate = KillZoneGate()

        # Day before DST ends (first Sunday of November)
        # In 2026, DST ends November 1
        with freeze_time("2026-10-31 10:00:00-04:00"):  # Still EDT
            status = gate.check_can_trade()
            assert status.can_trade is True

        # Day after DST at same "wall clock" time
        with freeze_time("2026-11-02 10:00:00-05:00"):  # Now EST
            status = gate.check_can_trade()
            assert status.can_trade is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
