"""
Signal Freshness Validator

Ensures signals are fresh (from current trading day) before submission.
Prevents execution of stale signals from previous trading sessions.

Critical safety feature - signals older than 1 trading day should NOT be traded.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Tuple
from zoneinfo import ZoneInfo
import pandas as pd

ET = ZoneInfo("America/New_York")


@dataclass
class FreshnessResult:
    """Result of signal freshness check."""
    is_fresh: bool
    signal_date: Optional[date]
    expected_date: date
    days_old: int
    reason: str

    @property
    def is_stale(self) -> bool:
        return not self.is_fresh


def get_last_trading_day(reference_date: Optional[date] = None) -> date:
    """
    Get the most recent trading day.

    Uses NYSE calendar when available, fallback to weekday logic.
    """
    if reference_date is None:
        reference_date = datetime.now(ET).date()

    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar('NYSE')

        start_check = reference_date - timedelta(days=10)
        schedule = nyse.schedule(
            start_date=start_check.isoformat(),
            end_date=reference_date.isoformat()
        )

        if len(schedule) > 0:
            return schedule.index[-1].date()
    except ImportError:
        pass

    # Fallback: adjust for weekends
    result = reference_date
    while result.weekday() >= 5:  # Sat=5, Sun=6
        result -= timedelta(days=1)
    return result


def get_expected_signal_date() -> date:
    """
    Get the expected date for fresh signals.

    During market hours: signals should be from today or previous trading day
    Before market open: signals from previous trading day are valid
    """
    now = datetime.now(ET)
    today = now.date()
    current_hour = now.hour

    # Before 9:30 AM ET, previous day's signals are valid
    if current_hour < 10:
        return get_last_trading_day(today - timedelta(days=1))

    # During and after market hours, today's signals (or last trading day if weekend)
    return get_last_trading_day(today)


def parse_signal_timestamp(timestamp) -> Optional[date]:
    """Parse signal timestamp from various formats."""
    if timestamp is None:
        return None

    if isinstance(timestamp, date):
        return timestamp

    if isinstance(timestamp, datetime):
        return timestamp.date()

    if isinstance(timestamp, pd.Timestamp):
        return timestamp.date()

    if isinstance(timestamp, str):
        try:
            # Try ISO format
            if 'T' in timestamp:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
            else:
                return datetime.fromisoformat(timestamp).date()
        except (ValueError, TypeError):
            pass

        try:
            # Try pandas parsing
            return pd.to_datetime(timestamp).date()
        except Exception:
            pass

    return None


def check_signal_freshness(
    signal_timestamp,
    max_age_days: int = 1,
    reference_date: Optional[date] = None,
) -> FreshnessResult:
    """
    Check if a signal is fresh enough to trade.

    Args:
        signal_timestamp: The timestamp of the signal (str, datetime, date, or pd.Timestamp)
        max_age_days: Maximum trading days old (default: 1)
        reference_date: Reference date for comparison (default: today)

    Returns:
        FreshnessResult with freshness status and details
    """
    if reference_date is None:
        reference_date = datetime.now(ET).date()

    signal_date = parse_signal_timestamp(signal_timestamp)
    expected_date = get_expected_signal_date()

    if signal_date is None:
        return FreshnessResult(
            is_fresh=False,
            signal_date=None,
            expected_date=expected_date,
            days_old=-1,
            reason="Cannot parse signal timestamp"
        )

    # Calculate trading days difference
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar('NYSE')

        start = min(signal_date, expected_date)
        end = max(signal_date, expected_date)

        schedule = nyse.schedule(
            start_date=start.isoformat(),
            end_date=end.isoformat()
        )
        trading_days = len(schedule) - 1 if len(schedule) > 0 else 0

    except ImportError:
        # Fallback to calendar days (approximate)
        trading_days = (expected_date - signal_date).days

    # Signal is fresh if within max_age_days trading days
    is_fresh = signal_date >= expected_date or trading_days <= max_age_days

    if is_fresh:
        reason = f"Signal is fresh (from {signal_date})"
    else:
        reason = f"STALE: Signal from {signal_date} is {trading_days} trading day(s) old (expected {expected_date})"

    return FreshnessResult(
        is_fresh=is_fresh,
        signal_date=signal_date,
        expected_date=expected_date,
        days_old=trading_days,
        reason=reason
    )


def validate_signal_file(
    file_path: Path,
    timestamp_column: str = "timestamp",
    max_age_days: int = 1,
) -> Tuple[bool, FreshnessResult, Optional[pd.DataFrame]]:
    """
    Validate all signals in a CSV file for freshness.

    Args:
        file_path: Path to signal CSV file
        timestamp_column: Name of timestamp column
        max_age_days: Maximum trading days old

    Returns:
        Tuple of (all_fresh, worst_result, dataframe)
    """
    if not Path(file_path).exists():
        return False, FreshnessResult(
            is_fresh=False,
            signal_date=None,
            expected_date=get_expected_signal_date(),
            days_old=-1,
            reason=f"Signal file not found: {file_path}"
        ), None

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return False, FreshnessResult(
            is_fresh=False,
            signal_date=None,
            expected_date=get_expected_signal_date(),
            days_old=-1,
            reason=f"Cannot read signal file: {e}"
        ), None

    if df.empty:
        return False, FreshnessResult(
            is_fresh=False,
            signal_date=None,
            expected_date=get_expected_signal_date(),
            days_old=-1,
            reason="Signal file is empty"
        ), df

    if timestamp_column not in df.columns:
        return False, FreshnessResult(
            is_fresh=False,
            signal_date=None,
            expected_date=get_expected_signal_date(),
            days_old=-1,
            reason=f"Missing '{timestamp_column}' column in signal file"
        ), df

    # Check all signals
    all_fresh = True
    worst_result = None

    for _, row in df.iterrows():
        result = check_signal_freshness(row.get(timestamp_column), max_age_days)
        if not result.is_fresh:
            all_fresh = False
            if worst_result is None or result.days_old > worst_result.days_old:
                worst_result = result

    if worst_result is None:
        # All signals are fresh
        first_ts = df[timestamp_column].iloc[0] if len(df) > 0 else None
        worst_result = check_signal_freshness(first_ts, max_age_days)

    return all_fresh, worst_result, df


def get_signal_date_summary(df: pd.DataFrame, timestamp_column: str = "timestamp") -> str:
    """Get a summary of signal dates in a dataframe."""
    if df.empty or timestamp_column not in df.columns:
        return "No signals"

    dates = df[timestamp_column].apply(parse_signal_timestamp)
    unique_dates = dates.dropna().unique()

    if len(unique_dates) == 0:
        return "No valid dates found"
    elif len(unique_dates) == 1:
        return f"All signals from {unique_dates[0]}"
    else:
        return f"Mixed dates: {sorted(unique_dates)}"


# Convenience function for quick checks
def is_signal_fresh(signal_timestamp, max_age_days: int = 1) -> bool:
    """Quick check if a signal is fresh."""
    result = check_signal_freshness(signal_timestamp, max_age_days)
    return result.is_fresh
