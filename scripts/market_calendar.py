#!/usr/bin/env python3
"""
Market calendar for Kobe trading system.
Shows market hours, holidays, and early closes.
Usage: python scripts/calendar.py [--today|--week|--month|--holidays YEAR]
"""

import argparse
from datetime import datetime, timedelta, time
from core.clock.tz_utils import fmt_ct
from typing import List, Tuple, Optional
import json

# US Market holidays (2024-2025)
# Format: (month, day, name, early_close_time or None)
US_MARKET_HOLIDAYS = {
    2024: [
        (1, 1, "New Year's Day", None),
        (1, 15, "Martin Luther King Jr. Day", None),
        (2, 19, "Presidents Day", None),
        (3, 29, "Good Friday", None),
        (5, 27, "Memorial Day", None),
        (6, 19, "Juneteenth", None),
        (7, 4, "Independence Day", None),
        (7, 3, "Independence Day (Early Close)", time(13, 0)),
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
        (7, 4, "Independence Day (Observed)", None),
        (9, 7, "Labor Day", None),
        (11, 26, "Thanksgiving", None),
        (11, 27, "Black Friday (Early Close)", time(13, 0)),
        (12, 24, "Christmas Eve (Early Close)", time(13, 0)),
        (12, 25, "Christmas Day", None),
    ],
}

# Regular market hours (Eastern Time)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
PRE_MARKET_OPEN = time(4, 0)
AFTER_HOURS_CLOSE = time(20, 0)


def is_weekend(dt: datetime) -> bool:
    """Check if date is weekend."""
    return dt.weekday() >= 5


def get_holiday_info(dt: datetime) -> Optional[Tuple[str, Optional[time]]]:
    """Get holiday info for a date if it's a holiday."""
    year = dt.year
    if year not in US_MARKET_HOLIDAYS:
        return None

    for month, day, name, early_close in US_MARKET_HOLIDAYS[year]:
        if dt.month == month and dt.day == day:
            return (name, early_close)
    return None


def is_market_closed(dt: datetime) -> Tuple[bool, str]:
    """Check if market is closed on given date."""
    if is_weekend(dt):
        return True, "Weekend"

    holiday = get_holiday_info(dt)
    if holiday:
        name, early_close = holiday
        if early_close is None:  # Full holiday
            return True, f"Holiday: {name}"
        else:
            return False, f"Early Close: {name} (closes at {early_close.strftime('%H:%M')} ET)"

    return False, "Market Open"


def get_market_hours(dt: datetime) -> Tuple[time, time]:
    """Get market hours for a date."""
    holiday = get_holiday_info(dt)
    if holiday:
        name, early_close = holiday
        if early_close:
            return MARKET_OPEN, early_close
    return MARKET_OPEN, MARKET_CLOSE


def show_today():
    """Show today's market status."""
    now = datetime.now()
    closed, reason = is_market_closed(now)
    market_open, market_close = get_market_hours(now)

    print(f"\n=== Market Status: {now.strftime('%A, %B %d, %Y')} ===\n")

    if closed:
        print(f"  Status: CLOSED")
        print(f"  Reason: {reason}")
        # Find next open day
        next_day = now + timedelta(days=1)
        while is_market_closed(next_day)[0]:
            next_day += timedelta(days=1)
        print(f"  Next Open: {next_day.strftime('%A, %B %d')}")
    else:
        print(f"  Status: {reason}")
        print(f"  Pre-Market: {fmt_ct(datetime.combine(now.date(), PRE_MARKET_OPEN))} to {fmt_ct(datetime.combine(now.date(), market_open))}")
        print(f"  Regular:    {fmt_ct(datetime.combine(now.date(), market_open))} to {fmt_ct(datetime.combine(now.date(), market_close))}")
        print(f"  After Hours: {fmt_ct(datetime.combine(now.date(), MARKET_CLOSE))} to {fmt_ct(datetime.combine(now.date(), AFTER_HOURS_CLOSE))}")

        # Current session
        current_time = now.time()
        if current_time < PRE_MARKET_OPEN:
            print(f"\n  Current Session: Pre-Pre-Market (closed)")
        elif current_time < MARKET_OPEN:
            print(f"\n  Current Session: Pre-Market")
        elif current_time < market_close:
            print(f"\n  Current Session: Regular Trading")
        elif current_time < AFTER_HOURS_CLOSE:
            print(f"\n  Current Session: After Hours")
        else:
            print(f"\n  Current Session: Closed")


def show_week():
    """Show this week's market schedule."""
    today = datetime.now().date()
    # Find Monday of current week
    monday = today - timedelta(days=today.weekday())

    print(f"\n=== This Week's Market Schedule ===\n")

    for i in range(7):
        day = datetime.combine(monday + timedelta(days=i), time(0, 0))
        closed, reason = is_market_closed(day)
        market_open, market_close = get_market_hours(day)

        day_name = day.strftime("%a %m/%d")
        is_today = " (TODAY)" if day.date() == today else ""

        if closed:
            print(f"  {day_name}: CLOSED - {reason}{is_today}")
        else:
            hours = f"{fmt_ct(datetime.combine(day.date(), market_open), include_tz=False)}-{fmt_ct(datetime.combine(day.date(), market_close), include_tz=False)} CT"
            if reason != "Market Open":
                print(f"  {day_name}: {hours} - {reason}{is_today}")
            else:
                print(f"  {day_name}: {hours}{is_today}")


def show_month():
    """Show this month's market schedule with holidays."""
    today = datetime.now()
    year, month = today.year, today.month

    print(f"\n=== {today.strftime('%B %Y')} Market Calendar ===\n")

    # Get all days in month
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)

    current = datetime(year, month, 1)
    closed_days = []
    early_closes = []

    while current < next_month:
        closed, reason = is_market_closed(current)
        if closed:
            closed_days.append((current, reason))
        elif "Early Close" in reason:
            early_closes.append((current, reason))
        current += timedelta(days=1)

    if closed_days:
        print("Market Closed Days:")
        for day, reason in closed_days:
            marker = " <-- TODAY" if day.date() == today.date() else ""
            print(f"  {day.strftime('%a %m/%d')}: {reason}{marker}")

    if early_closes:
        print("\nEarly Close Days:")
        for day, reason in early_closes:
            print(f"  {day.strftime('%a %m/%d')}: {reason}")

    # Trading days count
    trading_days = 0
    current = datetime(year, month, 1)
    while current < next_month:
        if not is_market_closed(current)[0]:
            trading_days += 1
        current += timedelta(days=1)

    print(f"\nTotal Trading Days: {trading_days}")


def show_holidays(year: int):
    """Show all holidays for a year."""
    print(f"\n=== {year} US Market Holidays ===\n")

    if year not in US_MARKET_HOLIDAYS:
        print(f"Holiday data not available for {year}")
        print(f"Available years: {list(US_MARKET_HOLIDAYS.keys())}")
        return

    for month, day, name, early_close in US_MARKET_HOLIDAYS[year]:
        date = datetime(year, month, day)
        day_str = date.strftime("%a %m/%d")

        if early_close:
            print(f"  {day_str}: {name} - Market closes at {early_close.strftime('%H:%M')} ET")
        else:
            print(f"  {day_str}: {name} - MARKET CLOSED")


def get_next_trading_day(from_date: datetime = None) -> datetime:
    """Get the next trading day."""
    if from_date is None:
        from_date = datetime.now()

    next_day = from_date + timedelta(days=1)
    while is_market_closed(next_day)[0]:
        next_day += timedelta(days=1)

    return next_day


def main():
    parser = argparse.ArgumentParser(description="Market calendar information")
    parser.add_argument("--today", action="store_true", help="Show today's market status")
    parser.add_argument("--week", action="store_true", help="Show this week's schedule")
    parser.add_argument("--month", action="store_true", help="Show this month's calendar")
    parser.add_argument("--holidays", type=int, metavar="YEAR", help="Show holidays for year")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--next", action="store_true", help="Show next trading day")

    args = parser.parse_args()

    if args.json:
        now = datetime.now()
        closed, reason = is_market_closed(now)
        market_open, market_close = get_market_hours(now)
        next_trading = get_next_trading_day(now)

        data = {
            "date": now.strftime("%Y-%m-%d"),
            "market_closed": closed,
            "reason": reason,
            "market_open": market_open.strftime("%H:%M"),
            "market_close": market_close.strftime("%H:%M"),
            "next_trading_day": next_trading.strftime("%Y-%m-%d"),
        }
        print(json.dumps(data, indent=2))
    elif args.holidays:
        show_holidays(args.holidays)
    elif args.month:
        show_month()
    elif args.week:
        show_week()
    elif args.next:
        next_day = get_next_trading_day()
        print(f"Next trading day: {next_day.strftime('%A, %B %d, %Y')}")
    else:
        # Default: show today
        show_today()


if __name__ == "__main__":
    main()
