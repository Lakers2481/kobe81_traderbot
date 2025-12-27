"""
US equities market calendar for NYSE/NASDAQ.

Handles holidays, early closes, and DST-aware session times.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from .market_clock import AssetType, SessionType, SessionInfo, MarketEvent

# Eastern Time
ET = ZoneInfo("America/New_York")

# Regular market hours (Eastern Time)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
PRE_MARKET_OPEN = time(4, 0)
AFTER_HOURS_CLOSE = time(20, 0)

# US Market holidays (2024-2026)
# Format: (month, day, name, early_close_time or None)
US_MARKET_HOLIDAYS: Dict[int, List[Tuple[int, int, str, Optional[time]]]] = {
    2024: [
        (1, 1, "New Year's Day", None),
        (1, 15, "Martin Luther King Jr. Day", None),
        (2, 19, "Presidents Day", None),
        (3, 29, "Good Friday", None),
        (5, 27, "Memorial Day", None),
        (6, 19, "Juneteenth", None),
        (7, 3, "Independence Day (Early Close)", time(13, 0)),
        (7, 4, "Independence Day", None),
        (9, 2, "Labor Day", None),
        (11, 28, "Thanksgiving", None),
        (11, 29, "Black Friday (Early Close)", time(13, 0)),
        (12, 24, "Christmas Eve (Early Close)", time(13, 0)),
        (12, 25, "Christmas Day", None),
    ],
    2025: [
        (1, 1, "New Year's Day", None),
        (1, 20, "Martin Luther King Jr. Day", None),
        (2, 17, "Presidents Day", None),
        (4, 18, "Good Friday", None),
        (5, 26, "Memorial Day", None),
        (6, 19, "Juneteenth", None),
        (7, 3, "Independence Day (Early Close)", time(13, 0)),
        (7, 4, "Independence Day", None),
        (9, 1, "Labor Day", None),
        (11, 27, "Thanksgiving", None),
        (11, 28, "Black Friday (Early Close)", time(13, 0)),
        (12, 24, "Christmas Eve (Early Close)", time(13, 0)),
        (12, 25, "Christmas Day", None),
    ],
    2026: [
        (1, 1, "New Year's Day", None),
        (1, 19, "Martin Luther King Jr. Day", None),
        (2, 16, "Presidents Day", None),
        (4, 3, "Good Friday", None),
        (5, 25, "Memorial Day", None),
        (6, 19, "Juneteenth", None),
        (7, 3, "Independence Day (Early Close)", time(13, 0)),
        (7, 4, "Independence Day", None),  # Falls on Saturday
        (9, 7, "Labor Day", None),
        (11, 26, "Thanksgiving", None),
        (11, 27, "Black Friday (Early Close)", time(13, 0)),
        (12, 24, "Christmas Eve (Early Close)", time(13, 0)),
        (12, 25, "Christmas Day", None),
    ],
}


class EquitiesCalendar:
    """
    US equities market calendar.

    Provides timezone-aware market hours, holiday detection, and
    session information for NYSE/NASDAQ trading.
    """

    def __init__(self):
        self._holiday_cache: Dict[date, Tuple[str, Optional[time]]] = {}
        self._build_holiday_cache()

    def _build_holiday_cache(self) -> None:
        """Pre-compute holiday dates for faster lookup."""
        for year, holidays in US_MARKET_HOLIDAYS.items():
            for month, day, name, early_close in holidays:
                d = date(year, month, day)
                self._holiday_cache[d] = (name, early_close)

    def is_weekend(self, dt: datetime | date) -> bool:
        """Check if date is weekend."""
        if isinstance(dt, datetime):
            dt = dt.date()
        return dt.weekday() >= 5

    def get_holiday_info(self, dt: datetime | date) -> Optional[Tuple[str, Optional[time]]]:
        """Get holiday info for a date if it's a holiday."""
        if isinstance(dt, datetime):
            dt = dt.date()
        return self._holiday_cache.get(dt)

    def is_full_holiday(self, dt: datetime | date) -> bool:
        """Check if date is a full market holiday (closed all day)."""
        info = self.get_holiday_info(dt)
        if info is None:
            return False
        _, early_close = info
        return early_close is None

    def is_early_close(self, dt: datetime | date) -> Tuple[bool, Optional[time]]:
        """Check if date is an early close day and return close time."""
        info = self.get_holiday_info(dt)
        if info is None:
            return False, None
        _, early_close = info
        if early_close is not None:
            return True, early_close
        return False, None

    def is_trading_day(self, dt: datetime | date) -> bool:
        """Check if date is a trading day (market open at least part of day)."""
        if isinstance(dt, datetime):
            dt = dt.date()
        if self.is_weekend(dt):
            return False
        if self.is_full_holiday(dt):
            return False
        return True

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Check if market is currently open for regular trading."""
        if dt is None:
            dt = datetime.now(ET)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)

        if not self.is_trading_day(dt):
            return False

        current_time = dt.time()
        _, close_time = self.get_market_hours(dt)

        return MARKET_OPEN <= current_time < close_time

    def get_market_hours(self, dt: datetime | date) -> Tuple[time, time]:
        """Get market hours (open, close) for a date."""
        is_early, early_close = self.is_early_close(dt)
        if is_early and early_close:
            return MARKET_OPEN, early_close
        return MARKET_OPEN, MARKET_CLOSE

    def get_session_info(self, dt: Optional[datetime] = None) -> SessionInfo:
        """Get detailed session information for a datetime."""
        if dt is None:
            dt = datetime.now(ET)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)

        if self.is_weekend(dt):
            return SessionInfo(
                asset_type=AssetType.EQUITIES,
                session_type=SessionType.CLOSED,
                is_open=False,
                session_start=None,
                session_end=None,
                reason="Weekend",
            )

        holiday_info = self.get_holiday_info(dt)
        if holiday_info:
            name, early_close = holiday_info
            if early_close is None:
                return SessionInfo(
                    asset_type=AssetType.EQUITIES,
                    session_type=SessionType.CLOSED,
                    is_open=False,
                    session_start=None,
                    session_end=None,
                    reason=f"Holiday: {name}",
                )

        open_time, close_time = self.get_market_hours(dt)
        current_time = dt.time()
        base_date = dt.date()

        if current_time < PRE_MARKET_OPEN:
            return SessionInfo(
                asset_type=AssetType.EQUITIES,
                session_type=SessionType.CLOSED,
                is_open=False,
                session_start=None,
                session_end=datetime.combine(base_date, PRE_MARKET_OPEN, ET),
                reason="Pre-pre-market (overnight)",
            )
        elif current_time < open_time:
            return SessionInfo(
                asset_type=AssetType.EQUITIES,
                session_type=SessionType.PRE_MARKET,
                is_open=False,
                session_start=datetime.combine(base_date, PRE_MARKET_OPEN, ET),
                session_end=datetime.combine(base_date, open_time, ET),
                reason="Pre-market trading",
            )
        elif current_time < close_time:
            return SessionInfo(
                asset_type=AssetType.EQUITIES,
                session_type=SessionType.REGULAR,
                is_open=True,
                session_start=datetime.combine(base_date, open_time, ET),
                session_end=datetime.combine(base_date, close_time, ET),
                reason="Regular trading hours",
            )
        elif current_time < AFTER_HOURS_CLOSE:
            return SessionInfo(
                asset_type=AssetType.EQUITIES,
                session_type=SessionType.AFTER_HOURS,
                is_open=False,
                session_start=datetime.combine(base_date, close_time, ET),
                session_end=datetime.combine(base_date, AFTER_HOURS_CLOSE, ET),
                reason="After-hours trading",
            )
        else:
            return SessionInfo(
                asset_type=AssetType.EQUITIES,
                session_type=SessionType.CLOSED,
                is_open=False,
                session_start=None,
                session_end=None,
                reason="Market closed for the day",
            )

    def next_trading_day(self, from_date: Optional[date] = None) -> date:
        """Get the next trading day after the given date."""
        if from_date is None:
            from_date = datetime.now(ET).date()

        next_day = from_date + timedelta(days=1)
        max_days = 10  # Safety limit

        for _ in range(max_days):
            if self.is_trading_day(next_day):
                return next_day
            next_day += timedelta(days=1)

        # Fallback: return next weekday
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day

    def previous_trading_day(self, from_date: Optional[date] = None) -> date:
        """Get the previous trading day before the given date."""
        if from_date is None:
            from_date = datetime.now(ET).date()

        prev_day = from_date - timedelta(days=1)
        max_days = 10

        for _ in range(max_days):
            if self.is_trading_day(prev_day):
                return prev_day
            prev_day -= timedelta(days=1)

        # Fallback: return previous weekday
        while prev_day.weekday() >= 5:
            prev_day -= timedelta(days=1)
        return prev_day

    def trading_days_between(self, start: date, end: date) -> int:
        """Count trading days between two dates (exclusive of end)."""
        count = 0
        current = start
        while current < end:
            if self.is_trading_day(current):
                count += 1
            current += timedelta(days=1)
        return count

    def get_trading_days_in_range(self, start: date, end: date) -> List[date]:
        """Get list of trading days in range (inclusive of both)."""
        days = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days
