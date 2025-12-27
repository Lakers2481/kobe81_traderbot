"""
Options event-driven clock.

Tracks expiry dates, earnings proximity, and macro events.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

from .market_clock import AssetType, SessionType, SessionInfo, MarketEvent
from .equities_calendar import EquitiesCalendar

ET = ZoneInfo("America/New_York")


@dataclass
class OptionsEvent:
    """Represents an options-related event."""
    event_type: str  # "monthly_expiry", "weekly_expiry", "earnings", "fomc", "cpi"
    event_date: date
    symbol: Optional[str] = None  # For earnings
    description: str = ""


class OptionsEventClock:
    """
    Options event calendar for event-driven trading.

    Tracks:
    - Monthly expiry (3rd Friday of each month)
    - Weekly expiry (every Friday)
    - Earnings proximity
    - FOMC/CPI macro events
    """

    def __init__(self):
        self._equities_cal = EquitiesCalendar()
        self._cached_events: List[OptionsEvent] = []
        self._cache_year: Optional[int] = None

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Options follow equities hours."""
        return self._equities_cal.is_market_open(dt)

    def get_session_info(self, dt: Optional[datetime] = None) -> SessionInfo:
        """Get session info - follows equities."""
        session = self._equities_cal.get_session_info(dt)
        # Override asset type
        return SessionInfo(
            asset_type=AssetType.OPTIONS,
            session_type=session.session_type,
            is_open=session.is_open,
            session_start=session.session_start,
            session_end=session.session_end,
            reason=session.reason,
        )

    def get_monthly_expiry(self, year: int, month: int) -> date:
        """Get the 3rd Friday of the month (monthly options expiry)."""
        # Find first day of month
        first_day = date(year, month, 1)

        # Find first Friday
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)

        # Third Friday is 2 weeks later
        third_friday = first_friday + timedelta(weeks=2)

        # Adjust if it's a holiday (rare but possible)
        if not self._equities_cal.is_trading_day(third_friday):
            # Move to Thursday
            third_friday = third_friday - timedelta(days=1)

        return third_friday

    def get_weekly_expiries(self, start_date: date, end_date: date) -> List[date]:
        """Get all weekly expiry dates in range."""
        expiries = []
        current = start_date

        while current <= end_date:
            # Find next Friday
            days_until_friday = (4 - current.weekday()) % 7
            if days_until_friday == 0 and current.weekday() == 4:
                friday = current
            else:
                friday = current + timedelta(days=days_until_friday)

            if friday <= end_date and self._equities_cal.is_trading_day(friday):
                expiries.append(friday)

            current = friday + timedelta(days=1)

        return expiries

    def is_expiry_day(self, dt: Optional[date] = None) -> bool:
        """Check if date is an options expiry day."""
        if dt is None:
            dt = datetime.now(ET).date()
        elif isinstance(dt, datetime):
            dt = dt.date()

        # Friday is expiry day (weekly or monthly)
        return dt.weekday() == 4 and self._equities_cal.is_trading_day(dt)

    def is_monthly_expiry(self, dt: Optional[date] = None) -> bool:
        """Check if date is a monthly expiry (3rd Friday)."""
        if dt is None:
            dt = datetime.now(ET).date()
        elif isinstance(dt, datetime):
            dt = dt.date()

        if not self.is_expiry_day(dt):
            return False

        monthly = self.get_monthly_expiry(dt.year, dt.month)
        return dt == monthly

    def days_to_expiry(self, from_date: Optional[date] = None) -> int:
        """Get trading days until next expiry."""
        if from_date is None:
            from_date = datetime.now(ET).date()
        elif isinstance(from_date, datetime):
            from_date = from_date.date()

        # Find next Friday
        days_until_friday = (4 - from_date.weekday()) % 7
        if days_until_friday == 0:
            days_until_friday = 7  # Next Friday, not today

        next_expiry = from_date + timedelta(days=days_until_friday)

        # Adjust if not a trading day
        while not self._equities_cal.is_trading_day(next_expiry):
            next_expiry -= timedelta(days=1)

        return self._equities_cal.trading_days_between(from_date, next_expiry)

    def is_near_earnings(self, symbol: str, check_date: Optional[date] = None) -> bool:
        """
        Check if symbol is near earnings date.

        Delegates to core earnings filter if available.
        """
        try:
            from core.earnings_filter import is_near_earnings as core_check
            return core_check(symbol, check_date)
        except ImportError:
            # No earnings filter available, assume safe
            return False

    def next_event(self, dt: Optional[datetime] = None) -> Optional[MarketEvent]:
        """Get next options-related event."""
        if dt is None:
            dt = datetime.now(ET)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)

        current_date = dt.date()

        # Check for next expiry
        days_until_friday = (4 - current_date.weekday()) % 7
        if days_until_friday == 0 and dt.hour >= 16:
            # Past market close on Friday, next expiry is next week
            days_until_friday = 7

        next_expiry = current_date + timedelta(days=days_until_friday)

        # Adjust for holidays
        while not self._equities_cal.is_trading_day(next_expiry):
            next_expiry -= timedelta(days=1)

        is_monthly = self.is_monthly_expiry(next_expiry)

        return MarketEvent(
            asset_type=AssetType.OPTIONS,
            event_type="monthly_expiry" if is_monthly else "weekly_expiry",
            scheduled_time=datetime.combine(next_expiry, datetime.min.time()).replace(
                hour=16, minute=0, tzinfo=ET
            ),
            metadata={
                "expiry_date": next_expiry.isoformat(),
                "is_monthly": is_monthly,
            },
        )

    def get_events_in_range(
        self,
        start_date: date,
        end_date: date,
        include_weekly: bool = True,
    ) -> List[OptionsEvent]:
        """Get all options events in date range."""
        events: List[OptionsEvent] = []

        # Monthly expiries
        current_date = start_date
        while current_date <= end_date:
            monthly = self.get_monthly_expiry(current_date.year, current_date.month)
            if start_date <= monthly <= end_date:
                events.append(OptionsEvent(
                    event_type="monthly_expiry",
                    event_date=monthly,
                    description=f"Monthly options expiry ({monthly.strftime('%B %Y')})",
                ))

            # Move to next month
            if current_date.month == 12:
                current_date = date(current_date.year + 1, 1, 1)
            else:
                current_date = date(current_date.year, current_date.month + 1, 1)

        # Weekly expiries (if requested)
        if include_weekly:
            for friday in self.get_weekly_expiries(start_date, end_date):
                # Skip if it's a monthly (already added)
                if not self.is_monthly_expiry(friday):
                    events.append(OptionsEvent(
                        event_type="weekly_expiry",
                        event_date=friday,
                        description=f"Weekly options expiry ({friday.strftime('%Y-%m-%d')})",
                    ))

        # Sort by date
        events.sort(key=lambda e: e.event_date)
        return events
