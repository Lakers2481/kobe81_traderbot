"""
Unit tests for multi-asset market clock.
"""

import pytest
from datetime import datetime, date, time
from zoneinfo import ZoneInfo

from core.clock import (
    MarketClock,
    AssetType,
    SessionType,
    MarketEvent,
)
from core.clock.equities_calendar import EquitiesCalendar
from core.clock.crypto_clock import CryptoClock
from core.clock.options_event_clock import OptionsEventClock
from core.clock.macro_events import MacroEventCalendar, MacroEventType

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


class TestEquitiesCalendar:
    """Tests for equities calendar."""

    def test_weekend_detection(self):
        cal = EquitiesCalendar()
        # Saturday
        sat = datetime(2025, 12, 27, 10, 0, tzinfo=ET)
        assert cal.is_weekend(sat)
        # Friday
        fri = datetime(2025, 12, 26, 10, 0, tzinfo=ET)
        assert not cal.is_weekend(fri)

    def test_holiday_detection(self):
        cal = EquitiesCalendar()
        # Christmas 2025
        xmas = date(2025, 12, 25)
        assert cal.is_full_holiday(xmas)
        # Regular day
        regular = date(2025, 12, 22)
        assert not cal.is_full_holiday(regular)

    def test_early_close_detection(self):
        cal = EquitiesCalendar()
        # Christmas Eve 2025
        xmas_eve = date(2025, 12, 24)
        is_early, close_time = cal.is_early_close(xmas_eve)
        assert is_early
        assert close_time == time(13, 0)

    def test_trading_day(self):
        cal = EquitiesCalendar()
        # Friday Dec 26, 2025 is a trading day
        assert cal.is_trading_day(date(2025, 12, 26))
        # Saturday is not
        assert not cal.is_trading_day(date(2025, 12, 27))
        # Christmas is not
        assert not cal.is_trading_day(date(2025, 12, 25))

    def test_market_open(self):
        cal = EquitiesCalendar()
        # 10:30 AM on a trading day
        dt = datetime(2025, 12, 22, 10, 30, tzinfo=ET)
        assert cal.is_market_open(dt)
        # 4:30 PM (after close)
        dt_after = datetime(2025, 12, 22, 16, 30, tzinfo=ET)
        assert not cal.is_market_open(dt_after)

    def test_session_info(self):
        cal = EquitiesCalendar()
        # Pre-market
        pre = datetime(2025, 12, 22, 5, 0, tzinfo=ET)
        session = cal.get_session_info(pre)
        assert session.session_type == SessionType.PRE_MARKET
        assert not session.is_open

        # Regular hours
        regular = datetime(2025, 12, 22, 11, 0, tzinfo=ET)
        session = cal.get_session_info(regular)
        assert session.session_type == SessionType.REGULAR
        assert session.is_open

    def test_next_trading_day(self):
        cal = EquitiesCalendar()
        # Friday -> Monday
        friday = date(2025, 12, 26)
        next_day = cal.next_trading_day(friday)
        assert next_day == date(2025, 12, 29)  # Monday


class TestCryptoClock:
    """Tests for 24/7 crypto clock."""

    def test_always_open(self):
        clock = CryptoClock()
        # Any time should be open
        dt = datetime(2025, 12, 27, 3, 0, tzinfo=UTC)
        assert clock.is_market_open(dt)

    def test_continuous_session(self):
        clock = CryptoClock()
        session = clock.get_session_info()
        assert session.session_type == SessionType.CONTINUOUS
        assert session.is_open

    def test_next_scan_aligned(self):
        clock = CryptoClock(cadence_hours=4, align_to_utc=True)
        # At 1:30 UTC, next scan should be 4:00 UTC
        dt = datetime(2025, 12, 27, 1, 30, tzinfo=UTC)
        event = clock.next_scan_event(dt)
        assert event.scheduled_time.hour == 4

    def test_scan_times_for_day(self):
        clock = CryptoClock(cadence_hours=4)
        scans = clock.get_scan_times_for_day()
        # Should have 6 scans: 0, 4, 8, 12, 16, 20
        assert len(scans) == 6


class TestOptionsEventClock:
    """Tests for options event clock."""

    def test_monthly_expiry(self):
        clock = OptionsEventClock()
        # December 2025 monthly expiry should be 3rd Friday (Dec 19)
        expiry = clock.get_monthly_expiry(2025, 12)
        assert expiry == date(2025, 12, 19)

    def test_is_expiry_day(self):
        clock = OptionsEventClock()
        # Any Friday is an expiry day
        friday = date(2025, 12, 19)
        assert clock.is_expiry_day(friday)
        # Monday is not
        monday = date(2025, 12, 22)
        assert not clock.is_expiry_day(monday)

    def test_weekly_expiries(self):
        clock = OptionsEventClock()
        start = date(2025, 12, 1)
        end = date(2025, 12, 31)
        expiries = clock.get_weekly_expiries(start, end)
        # December 2025 has 4 Fridays (5, 12, 19, 26) - some may be holidays
        assert len(expiries) >= 3


class TestMacroEventCalendar:
    """Tests for macro event calendar."""

    def test_fomc_day(self):
        cal = MacroEventCalendar()
        # January 29, 2025 is an FOMC day
        assert cal.is_fomc_day(date(2025, 1, 29))
        # January 30 is not
        assert not cal.is_fomc_day(date(2025, 1, 30))

    def test_fomc_week(self):
        cal = MacroEventCalendar()
        # Day before FOMC
        assert cal.is_fomc_week(date(2025, 1, 28))
        # Day after FOMC
        assert cal.is_fomc_week(date(2025, 1, 30))

    def test_next_fomc(self):
        cal = MacroEventCalendar()
        next_event = cal.next_fomc(date(2025, 1, 1))
        assert next_event is not None
        assert next_event.event_date == date(2025, 1, 29)

    def test_high_volatility_period(self):
        cal = MacroEventCalendar()
        # FOMC day
        assert cal.is_high_volatility_period(date(2025, 1, 29))
        # NFP Friday (first Friday)
        assert cal.is_high_volatility_period(date(2025, 1, 3))


class TestMarketClock:
    """Tests for unified market clock."""

    def test_create_from_config(self):
        config = {
            "equities": {"enabled": True, "scan_times": "09:35,10:30"},
            "crypto": {"enabled": False},
            "options": {"enabled": False},
        }
        clock = MarketClock.from_config(config)
        assert clock.equities_enabled
        assert not clock.crypto_enabled
        assert clock.equities_scan_times == ["09:35", "10:30"]

    def test_market_open_equities(self):
        clock = MarketClock()
        # During regular hours
        dt = datetime(2025, 12, 22, 11, 0, tzinfo=ET)
        assert clock.is_market_open(AssetType.EQUITIES, dt)

    def test_market_open_crypto(self):
        clock = MarketClock(crypto_enabled=True)
        # Crypto is always open
        dt = datetime(2025, 12, 27, 3, 0, tzinfo=ET)
        assert clock.is_market_open(AssetType.CRYPTO, dt)

    def test_next_event_equities(self):
        clock = MarketClock(equities_scan_times=["09:35", "10:30", "15:55"])
        # At 9:00 AM, next event should be 9:35 scan
        dt = datetime(2025, 12, 22, 9, 0, tzinfo=ET)
        event = clock.next_event(AssetType.EQUITIES, dt)
        assert event is not None
        assert event.event_type == "scan"
        assert event.scheduled_time.hour == 9
        assert event.scheduled_time.minute == 35

    def test_next_event_any(self):
        clock = MarketClock(
            equities_enabled=True,
            crypto_enabled=True,
            crypto_cadence_hours=1,
        )
        # Should return soonest event across all asset types
        event = clock.next_event_any()
        assert event is not None


class TestMarketEvent:
    """Tests for market event dataclass."""

    def test_event_ordering(self):
        now = datetime.now(ET)
        event1 = MarketEvent(
            asset_type=AssetType.EQUITIES,
            event_type="scan",
            scheduled_time=now,
        )
        event2 = MarketEvent(
            asset_type=AssetType.CRYPTO,
            event_type="scan",
            scheduled_time=now + timedelta(hours=1),
        )
        assert event1 < event2


# Import timedelta for test
from datetime import timedelta
