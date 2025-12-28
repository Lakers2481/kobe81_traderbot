from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


# Canonical internal and display timezones
ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")


def now_et() -> datetime:
    """Return timezone-aware datetime in America/New_York (ET)."""
    return datetime.now(ET)


def to_ct(dt: datetime) -> datetime:
    """Convert a datetime to America/Chicago, assuming ET if naive.

    If `dt` is naive, treat it as ET (internal canonical) before conversion.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    return dt.astimezone(CT)


def fmt_ct(dt: datetime, include_tz: bool = True) -> str:
    """Format a datetime as 12-hour Central Time string for display.

    Example: "09:45 AM CT" (or without suffix if include_tz=False)
    """
    dt_ct = to_ct(dt)
    s = dt_ct.strftime("%I:%M %p")
    # Drop leading zero for human readability
    if s.startswith("0"):
        s = s[1:]
    return f"{s} CT" if include_tz else s


def fmt_ct_datetime(dt: datetime) -> str:
    """Format datetime as date + 12-hour time in Central Time.

    Example: "2025-12-27 9:45 AM CT"
    """
    dt_ct = to_ct(dt)
    date = dt_ct.strftime("%Y-%m-%d")
    time_str = dt_ct.strftime("%I:%M %p").lstrip("0")
    return f"{date} {time_str} CT"


def fmt_et(dt: datetime, include_tz: bool = True) -> str:
    """Format a datetime as 12-hour Eastern Time string for display.

    Example: "05:00 PM ET" (or without suffix if include_tz=False)
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    dt_et = dt.astimezone(ET)
    s = dt_et.strftime("%I:%M %p")
    if s.startswith("0"):
        s = s[1:]
    return f"{s} ET" if include_tz else s


def fmt_et_datetime(dt: datetime) -> str:
    """Format datetime as date + 12-hour time in Eastern Time.

    Example: "2025-12-27 5:00 PM ET"
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    dt_et = dt.astimezone(ET)
    date = dt_et.strftime("%Y-%m-%d")
    time_str = dt_et.strftime("%I:%M %p").lstrip("0")
    return f"{date} {time_str} ET"
