"""
Central Date Utilities for Kobe Trading System.
================================================

ALL scripts should use these functions - no hardcoded dates anywhere.

Created: 2026-01-07
Purpose: Prevent stale data issues caused by hardcoded year-specific dates.

Usage:
    from core.clock.date_utils import today_str, trading_start_date, is_cache_stale

    # Get today's date
    end_date = today_str()  # '2026-01-07'

    # Get start date for data fetching (1 year ago)
    start_date = trading_start_date()  # '2025-01-07'

    # Check if cached data is stale
    if is_cache_stale('2025-12-01'):
        refresh_cache()
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional

# Canonical timezone - ALL date/time operations use Eastern Time
ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    """
    Get current datetime in Eastern Time.

    Returns:
        datetime: Current time with ET timezone
    """
    return datetime.now(ET)


def today_str() -> str:
    """
    Get today's date as YYYY-MM-DD string.

    Returns:
        str: Today's date (e.g., '2026-01-07')
    """
    return now_et().strftime('%Y-%m-%d')


def trading_start_date(years_back: int = 1) -> str:
    """
    Get start date for data fetching.

    Args:
        years_back: Number of years to look back (default: 1)

    Returns:
        str: Start date as YYYY-MM-DD (e.g., '2025-01-07' for 1 year back)
    """
    return (now_et() - timedelta(days=365 * years_back)).strftime('%Y-%m-%d')


def days_ago(days: int) -> str:
    """
    Get date N days ago.

    Args:
        days: Number of days to subtract

    Returns:
        str: Date as YYYY-MM-DD
    """
    return (now_et() - timedelta(days=days)).strftime('%Y-%m-%d')


def is_cache_stale(last_date: str, max_days: int = 3) -> bool:
    """
    Check if cached data is stale.

    Args:
        last_date: Last date in cache (YYYY-MM-DD string or datetime-like)
        max_days: Maximum allowed age in days (default: 3 for weekend buffer)

    Returns:
        bool: True if cache is stale and needs refresh
    """
    try:
        # Handle various date formats
        date_str = str(last_date)[:10]
        last = datetime.strptime(date_str, '%Y-%m-%d')
        age_days = (now_et().date() - last.date()).days
        return age_days > max_days
    except (ValueError, TypeError):
        return True  # If can't parse, assume stale


def current_year() -> int:
    """
    Get current year.

    Returns:
        int: Current year (e.g., 2026)
    """
    return now_et().year


def current_month() -> int:
    """
    Get current month.

    Returns:
        int: Current month (1-12)
    """
    return now_et().month


def current_week() -> int:
    """
    Get current ISO week number.

    Returns:
        int: Week number (1-53)
    """
    return now_et().isocalendar()[1]


def current_day_of_week() -> int:
    """
    Get current day of week.

    Returns:
        int: Day of week (0=Monday, 6=Sunday)
    """
    return now_et().weekday()


def is_weekend() -> bool:
    """
    Check if today is a weekend.

    Returns:
        bool: True if Saturday or Sunday
    """
    return now_et().weekday() >= 5


def market_date_range(days: int = 60) -> tuple:
    """
    Get standard date range for market data fetching.

    Args:
        days: Number of days to look back (default: 60)

    Returns:
        tuple: (start_date, end_date) as YYYY-MM-DD strings
    """
    end = today_str()
    start = days_ago(days)
    return start, end


def backtest_date_range(years: int = 2) -> tuple:
    """
    Get standard date range for backtesting.

    Args:
        years: Number of years to look back (default: 2)

    Returns:
        tuple: (start_date, end_date) as YYYY-MM-DD strings
    """
    end = today_str()
    start = trading_start_date(years)
    return start, end


# Convenience exports
__all__ = [
    'ET',
    'now_et',
    'today_str',
    'trading_start_date',
    'days_ago',
    'is_cache_stale',
    'current_year',
    'current_month',
    'current_week',
    'current_day_of_week',
    'is_weekend',
    'market_date_range',
    'backtest_date_range',
]
