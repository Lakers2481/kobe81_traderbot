"""
Unified market clock abstraction for multi-asset scheduling.

Supports equities (NYSE/NASDAQ), crypto (24/7), and options (event-driven).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo

# Timezone constants
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


class AssetType(Enum):
    """Asset types supported by the scheduler."""
    EQUITIES = auto()
    CRYPTO = auto()
    OPTIONS = auto()


class SessionType(Enum):
    """Market session types."""
    PRE_MARKET = auto()
    REGULAR = auto()
    AFTER_HOURS = auto()
    CLOSED = auto()
    CONTINUOUS = auto()  # For crypto


@dataclass
class MarketEvent:
    """Represents a scheduled market event."""
    asset_type: AssetType
    event_type: str  # scan, expiry, fomc, etc.
    scheduled_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: MarketEvent) -> bool:
        return self.scheduled_time < other.scheduled_time


@dataclass
class SessionInfo:
    """Current session information."""
    asset_type: AssetType
    session_type: SessionType
    is_open: bool
    session_start: Optional[datetime]
    session_end: Optional[datetime]
    reason: str


class MarketClock:
    """
    Unified clock for multi-asset trading.

    Aggregates equities, crypto, and options calendars to provide
    a single interface for determining what to trade and when.
    """

    def __init__(
        self,
        equities_enabled: bool = True,
        crypto_enabled: bool = False,
        options_enabled: bool = False,
        equities_scan_times: Optional[List[str]] = None,
        crypto_cadence_hours: int = 4,
    ):
        self.equities_enabled = equities_enabled
        self.crypto_enabled = crypto_enabled
        self.options_enabled = options_enabled
        self.equities_scan_times = equities_scan_times or ["09:35", "10:30", "15:55"]
        self.crypto_cadence_hours = crypto_cadence_hours

        # Lazy imports to avoid circular dependencies
        self._equities_cal: Optional[Any] = None
        self._crypto_clock: Optional[Any] = None
        self._options_clock: Optional[Any] = None

    @property
    def equities_calendar(self):
        if self._equities_cal is None:
            from .equities_calendar import EquitiesCalendar
            self._equities_cal = EquitiesCalendar()
        return self._equities_cal

    @property
    def crypto_clock(self):
        if self._crypto_clock is None:
            from .crypto_clock import CryptoClock
            self._crypto_clock = CryptoClock(cadence_hours=self.crypto_cadence_hours)
        return self._crypto_clock

    @property
    def options_clock(self):
        if self._options_clock is None:
            from .options_event_clock import OptionsEventClock
            self._options_clock = OptionsEventClock()
        return self._options_clock

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> MarketClock:
        """Create clock from configuration dictionary."""
        if config is None:
            # Try loading from base.yaml
            try:
                from config.settings_loader import load_settings
                settings = load_settings()
                config = settings.get("scheduler", {})
            except Exception:
                config = {}

        equities_cfg = config.get("equities", {})
        crypto_cfg = config.get("crypto", {})
        options_cfg = config.get("options", {})

        scan_times_str = equities_cfg.get("scan_times", "09:35,10:30,15:55")
        if isinstance(scan_times_str, str):
            scan_times = [t.strip() for t in scan_times_str.split(",")]
        else:
            scan_times = scan_times_str

        return cls(
            equities_enabled=equities_cfg.get("enabled", True),
            crypto_enabled=crypto_cfg.get("enabled", False),
            options_enabled=options_cfg.get("enabled", False),
            equities_scan_times=scan_times,
            crypto_cadence_hours=crypto_cfg.get("cadence_hours", 4),
        )

    def is_market_open(self, asset_type: AssetType, dt: Optional[datetime] = None) -> bool:
        """Check if market is open for given asset type."""
        if dt is None:
            dt = datetime.now(ET)

        if asset_type == AssetType.EQUITIES:
            return self.equities_calendar.is_market_open(dt)
        elif asset_type == AssetType.CRYPTO:
            return True  # Crypto is always open
        elif asset_type == AssetType.OPTIONS:
            # Options follow equities hours
            return self.equities_calendar.is_market_open(dt)
        return False

    def get_session_info(self, asset_type: AssetType, dt: Optional[datetime] = None) -> SessionInfo:
        """Get current session information for asset type."""
        if dt is None:
            dt = datetime.now(ET)

        if asset_type == AssetType.EQUITIES:
            return self.equities_calendar.get_session_info(dt)
        elif asset_type == AssetType.CRYPTO:
            return self.crypto_clock.get_session_info(dt)
        elif asset_type == AssetType.OPTIONS:
            return self.options_clock.get_session_info(dt)

        return SessionInfo(
            asset_type=asset_type,
            session_type=SessionType.CLOSED,
            is_open=False,
            session_start=None,
            session_end=None,
            reason="Unknown asset type",
        )

    def next_event(self, asset_type: AssetType, dt: Optional[datetime] = None) -> Optional[MarketEvent]:
        """Get next scheduled event for asset type."""
        if dt is None:
            dt = datetime.now(ET)

        if asset_type == AssetType.EQUITIES:
            return self._next_equities_event(dt)
        elif asset_type == AssetType.CRYPTO:
            return self._next_crypto_event(dt)
        elif asset_type == AssetType.OPTIONS:
            return self._next_options_event(dt)
        return None

    def next_event_any(self, dt: Optional[datetime] = None) -> Optional[MarketEvent]:
        """Get next event across all enabled asset types."""
        if dt is None:
            dt = datetime.now(ET)

        events: List[MarketEvent] = []

        if self.equities_enabled:
            eq_event = self.next_event(AssetType.EQUITIES, dt)
            if eq_event:
                events.append(eq_event)

        if self.crypto_enabled:
            crypto_event = self.next_event(AssetType.CRYPTO, dt)
            if crypto_event:
                events.append(crypto_event)

        if self.options_enabled:
            opt_event = self.next_event(AssetType.OPTIONS, dt)
            if opt_event:
                events.append(opt_event)

        if not events:
            return None

        return min(events)

    def _next_equities_event(self, dt: datetime) -> Optional[MarketEvent]:
        """Get next equities scan event."""
        # Find next trading day and scan time
        current_date = dt.date()
        dt.time()

        # Check if today is a trading day
        if self.equities_calendar.is_trading_day(current_date):
            for scan_time in self.equities_scan_times:
                hour, minute = map(int, scan_time.split(":"))
                scan_dt = datetime.combine(current_date, datetime.min.time()).replace(
                    hour=hour, minute=minute, tzinfo=ET
                )
                if scan_dt > dt:
                    return MarketEvent(
                        asset_type=AssetType.EQUITIES,
                        event_type="scan",
                        scheduled_time=scan_dt,
                        metadata={"scan_time": scan_time},
                    )

        # Find next trading day
        next_day = self.equities_calendar.next_trading_day(current_date)
        if next_day and self.equities_scan_times:
            first_scan = self.equities_scan_times[0]
            hour, minute = map(int, first_scan.split(":"))
            scan_dt = datetime.combine(next_day, datetime.min.time()).replace(
                hour=hour, minute=minute, tzinfo=ET
            )
            return MarketEvent(
                asset_type=AssetType.EQUITIES,
                event_type="scan",
                scheduled_time=scan_dt,
                metadata={"scan_time": first_scan},
            )

        return None

    def _next_crypto_event(self, dt: datetime) -> Optional[MarketEvent]:
        """Get next crypto scan event."""
        return self.crypto_clock.next_scan_event(dt)

    def _next_options_event(self, dt: datetime) -> Optional[MarketEvent]:
        """Get next options event (expiry, earnings, etc.)."""
        return self.options_clock.next_event(dt)

    def sleep_until_next_event(self, max_sleep_seconds: int = 3600) -> Optional[MarketEvent]:
        """
        Sleep until the next event or max sleep time.

        Returns the event that triggered wake-up, or None if max_sleep reached.
        """
        import time

        next_event = self.next_event_any()
        if next_event is None:
            time.sleep(min(60, max_sleep_seconds))
            return None

        now = datetime.now(ET)
        seconds_until = (next_event.scheduled_time - now).total_seconds()

        if seconds_until <= 0:
            return next_event

        sleep_time = min(seconds_until, max_sleep_seconds)
        time.sleep(sleep_time)

        if sleep_time >= seconds_until:
            return next_event
        return None
