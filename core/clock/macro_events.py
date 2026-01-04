"""
Macro event calendar for FOMC, CPI, and other market-moving events.

Provides known event dates that may require adjusted trading behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


class MacroEventType(Enum):
    """Types of macro events."""
    FOMC = auto()          # Federal Reserve meeting
    CPI = auto()           # Consumer Price Index
    PPI = auto()           # Producer Price Index
    NFP = auto()           # Non-Farm Payrolls
    GDP = auto()           # GDP release
    PCE = auto()           # Personal Consumption Expenditures
    RETAIL_SALES = auto()  # Retail sales data


@dataclass
class MacroEvent:
    """Represents a scheduled macro event."""
    event_type: MacroEventType
    event_date: date
    event_time: Optional[time]  # Release time in ET
    description: str
    expected_volatility: str = "high"  # low, medium, high


# Known FOMC meeting dates (announcement day)
FOMC_DATES: Dict[int, List[date]] = {
    2024: [
        date(2024, 1, 31),
        date(2024, 3, 20),
        date(2024, 5, 1),
        date(2024, 6, 12),
        date(2024, 7, 31),
        date(2024, 9, 18),
        date(2024, 11, 7),
        date(2024, 12, 18),
    ],
    2025: [
        date(2025, 1, 29),
        date(2025, 3, 19),
        date(2025, 5, 7),
        date(2025, 6, 18),
        date(2025, 7, 30),
        date(2025, 9, 17),
        date(2025, 11, 5),
        date(2025, 12, 17),
    ],
    2026: [
        date(2026, 1, 28),
        date(2026, 3, 18),
        date(2026, 4, 29),
        date(2026, 6, 17),
        date(2026, 7, 29),
        date(2026, 9, 16),
        date(2026, 11, 4),
        date(2026, 12, 16),
    ],
}

# CPI release dates are typically the 2nd or 3rd week of each month
# These are approximate and should be verified annually
# Times are typically 8:30 AM ET


class MacroEventCalendar:
    """
    Calendar for macro economic events.

    Provides information about FOMC, CPI, and other market-moving
    events that may require adjusted trading behavior.
    """

    def __init__(self):
        self._event_cache: Dict[int, List[MacroEvent]] = {}

    def is_fomc_day(self, dt: Optional[date] = None) -> bool:
        """Check if date is an FOMC announcement day."""
        if dt is None:
            dt = datetime.now(ET).date()
        elif isinstance(dt, datetime):
            dt = dt.date()

        year = dt.year
        if year in FOMC_DATES:
            return dt in FOMC_DATES[year]
        return False

    def is_fomc_week(self, dt: Optional[date] = None) -> bool:
        """Check if date is in an FOMC week (day before/after meeting)."""
        if dt is None:
            dt = datetime.now(ET).date()
        elif isinstance(dt, datetime):
            dt = dt.date()

        year = dt.year
        if year not in FOMC_DATES:
            return False

        for fomc_date in FOMC_DATES[year]:
            # Check if within 1 day of FOMC
            delta = abs((dt - fomc_date).days)
            if delta <= 1:
                return True
        return False

    def next_fomc(self, from_date: Optional[date] = None) -> Optional[MacroEvent]:
        """Get next FOMC meeting date."""
        if from_date is None:
            from_date = datetime.now(ET).date()
        elif isinstance(from_date, datetime):
            from_date = from_date.date()

        # Check current year and next
        for year in [from_date.year, from_date.year + 1]:
            if year not in FOMC_DATES:
                continue
            for fomc_date in FOMC_DATES[year]:
                if fomc_date >= from_date:
                    return MacroEvent(
                        event_type=MacroEventType.FOMC,
                        event_date=fomc_date,
                        event_time=time(14, 0),  # FOMC typically announces at 2 PM ET
                        description="FOMC Meeting Decision",
                        expected_volatility="high",
                    )
        return None

    def days_to_fomc(self, from_date: Optional[date] = None) -> Optional[int]:
        """Get calendar days until next FOMC meeting."""
        next_event = self.next_fomc(from_date)
        if next_event is None:
            return None

        if from_date is None:
            from_date = datetime.now(ET).date()
        elif isinstance(from_date, datetime):
            from_date = from_date.date()

        return (next_event.event_date - from_date).days

    def get_cpi_dates(self, year: int) -> List[date]:
        """
        Get approximate CPI release dates for a year.

        CPI is typically released around the 10th-14th of each month.
        This is approximate; verify exact dates from BLS calendar.
        """
        dates = []
        for month in range(1, 13):
            # Approximate: 2nd week of the month (10th-14th)
            approx_date = date(year, month, 12)
            # Adjust to weekday if falls on weekend
            while approx_date.weekday() >= 5:
                approx_date -= timedelta(days=1)
            dates.append(approx_date)
        return dates

    def is_high_volatility_period(self, dt: Optional[date] = None) -> bool:
        """
        Check if date is in a known high-volatility period.

        This includes:
        - FOMC days
        - Day after FOMC
        - NFP Friday (first Friday of month)
        - CPI release day (approximate)
        """
        if dt is None:
            dt = datetime.now(ET).date()
        elif isinstance(dt, datetime):
            dt = dt.date()

        # FOMC week
        if self.is_fomc_week(dt):
            return True

        # NFP Friday (first Friday of month)
        if dt.weekday() == 4 and dt.day <= 7:
            return True

        # CPI week (approximate - around 10th-14th)
        if 9 <= dt.day <= 15:
            return True

        return False

    def get_events_in_range(
        self,
        start_date: date,
        end_date: date,
        event_types: Optional[List[MacroEventType]] = None,
    ) -> List[MacroEvent]:
        """Get all macro events in date range."""
        events: List[MacroEvent] = []

        if event_types is None:
            event_types = [MacroEventType.FOMC]  # Default to FOMC only

        # FOMC events
        if MacroEventType.FOMC in event_types:
            for year in range(start_date.year, end_date.year + 1):
                if year not in FOMC_DATES:
                    continue
                for fomc_date in FOMC_DATES[year]:
                    if start_date <= fomc_date <= end_date:
                        events.append(MacroEvent(
                            event_type=MacroEventType.FOMC,
                            event_date=fomc_date,
                            event_time=time(14, 0),
                            description="FOMC Meeting Decision",
                            expected_volatility="high",
                        ))

        # NFP events (first Friday of each month)
        if MacroEventType.NFP in event_types:
            current = start_date.replace(day=1)
            while current <= end_date:
                # Find first Friday
                first_day = current.replace(day=1)
                days_to_friday = (4 - first_day.weekday()) % 7
                nfp_date = first_day + timedelta(days=days_to_friday)

                if start_date <= nfp_date <= end_date:
                    events.append(MacroEvent(
                        event_type=MacroEventType.NFP,
                        event_date=nfp_date,
                        event_time=time(8, 30),
                        description="Non-Farm Payrolls Release",
                        expected_volatility="high",
                    ))

                # Next month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)

        events.sort(key=lambda e: e.event_date)
        return events

    def should_reduce_exposure(self, dt: Optional[date] = None) -> tuple[bool, str]:
        """
        Check if exposure should be reduced due to upcoming events.

        Returns (should_reduce, reason).
        """
        if dt is None:
            dt = datetime.now(ET).date()
        elif isinstance(dt, datetime):
            dt = dt.date()

        # FOMC day - highest priority
        if self.is_fomc_day(dt):
            return True, "FOMC announcement day"

        # Day before FOMC
        days = self.days_to_fomc(dt)
        if days == 1:
            return True, "Day before FOMC"

        # NFP Friday
        if dt.weekday() == 4 and dt.day <= 7:
            return True, "NFP release day (first Friday)"

        return False, ""
