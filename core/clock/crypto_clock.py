"""
24/7 crypto market clock.

Provides configurable scan cadence for continuous crypto trading.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from .market_clock import AssetType, SessionType, SessionInfo, MarketEvent

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")


class CryptoClock:
    """
    Crypto market clock for 24/7 trading.

    Since crypto markets never close, this clock manages scan
    cadence and provides UTC-aligned scheduling.
    """

    def __init__(
        self,
        cadence_hours: int = 4,
        align_to_utc: bool = True,
    ):
        """
        Initialize crypto clock.

        Args:
            cadence_hours: Hours between scans (1, 4, 8, 24)
            align_to_utc: If True, align scans to UTC hours (0, 4, 8, etc.)
        """
        self.cadence_hours = cadence_hours
        self.align_to_utc = align_to_utc

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Crypto market is always open."""
        return True

    def get_session_info(self, dt: Optional[datetime] = None) -> SessionInfo:
        """Get session info - crypto is continuous."""
        if dt is None:
            dt = datetime.now(UTC)

        return SessionInfo(
            asset_type=AssetType.CRYPTO,
            session_type=SessionType.CONTINUOUS,
            is_open=True,
            session_start=None,  # No defined start
            session_end=None,    # No defined end
            reason="24/7 crypto market",
        )

    def next_scan_event(self, dt: Optional[datetime] = None) -> MarketEvent:
        """Get next scheduled scan event."""
        if dt is None:
            dt = datetime.now(UTC)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)

        if self.align_to_utc:
            # Align to UTC hours based on cadence
            next_scan = self._next_aligned_time(dt)
        else:
            # Next scan is cadence_hours from now
            next_scan = dt + timedelta(hours=self.cadence_hours)

        return MarketEvent(
            asset_type=AssetType.CRYPTO,
            event_type="scan",
            scheduled_time=next_scan,
            metadata={
                "cadence_hours": self.cadence_hours,
                "aligned": self.align_to_utc,
            },
        )

    def _next_aligned_time(self, dt: datetime) -> datetime:
        """Get next UTC-aligned scan time."""
        # Convert to UTC for alignment
        dt_utc = dt.astimezone(UTC)

        # Find next aligned hour
        current_hour = dt_utc.hour
        aligned_hours = list(range(0, 24, self.cadence_hours))

        for h in aligned_hours:
            if h > current_hour:
                next_hour = h
                break
        else:
            # Wrap to next day
            next_hour = aligned_hours[0]
            dt_utc += timedelta(days=1)

        # Create aligned datetime
        next_scan = dt_utc.replace(
            hour=next_hour,
            minute=0,
            second=0,
            microsecond=0,
        )

        return next_scan

    def get_scan_times_for_day(self, dt: Optional[datetime] = None) -> list[datetime]:
        """Get all scan times for a given day."""
        if dt is None:
            dt = datetime.now(UTC)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)

        base_date = dt.date()
        scans = []

        for hour in range(0, 24, self.cadence_hours):
            scan_dt = datetime.combine(
                base_date,
                datetime.min.time(),
            ).replace(hour=hour, tzinfo=UTC)
            scans.append(scan_dt)

        return scans

    def is_scan_time(self, dt: Optional[datetime] = None, tolerance_minutes: int = 5) -> bool:
        """Check if current time is within tolerance of a scan time."""
        if dt is None:
            dt = datetime.now(UTC)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)

        if not self.align_to_utc:
            return True  # Any time is valid if not aligned

        current_hour = dt.hour
        aligned_hours = list(range(0, 24, self.cadence_hours))

        if current_hour not in aligned_hours:
            return False

        # Check if within tolerance of the hour start
        minutes_past = dt.minute + dt.second / 60
        return minutes_past <= tolerance_minutes
