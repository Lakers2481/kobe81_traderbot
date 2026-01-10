"""
Kill Zone Gate - Time-based trade blocking for professional execution.

Implements ICT-style kill zones to prevent trading during:
- Opening range chaos (9:30-10:00 AM)
- Lunch chop (11:30 AM - 2:00 PM) - optional, configurable
- After hours

Usage:
    from risk.kill_zone_gate import KillZoneGate, can_trade_now

    gate = KillZoneGate()
    allowed, reason = gate.check_can_trade()

    # Or simple check
    if can_trade_now():
        execute_trade()
"""

from __future__ import annotations

import logging
from datetime import datetime, time as dtime
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class KillZone(Enum):
    """Trading zones throughout the day."""
    PRE_MARKET = "pre_market"           # Before 9:30 - no trading
    OPENING_RANGE = "opening_range"     # 9:30-10:00 - NO TRADING (amateur hour)
    LONDON_CLOSE = "london_close"       # 10:00-11:30 - PRIMARY WINDOW
    LUNCH_CHOP = "lunch_chop"           # 11:30-14:00 - Avoid new entries
    POWER_HOUR = "power_hour"           # 14:30-15:30 - SECONDARY WINDOW
    CLOSE = "close"                     # 15:30-16:00 - Manage only
    AFTER_HOURS = "after_hours"         # After 16:00 - no trading


@dataclass
class KillZoneConfig:
    """Configuration for kill zone behavior."""
    # Opening range - ALWAYS blocked
    opening_range_start: dtime = dtime(9, 30)
    opening_range_end: dtime = dtime(10, 0)
    block_opening_range: bool = True  # Cannot be disabled

    # Primary window
    primary_window_start: dtime = dtime(10, 0)
    primary_window_end: dtime = dtime(11, 30)

    # Lunch chop - configurable
    lunch_start: dtime = dtime(11, 30)
    lunch_end: dtime = dtime(14, 0)
    block_lunch: bool = True  # Can be disabled for aggressive mode

    # Power hour
    power_hour_start: dtime = dtime(14, 30)
    power_hour_end: dtime = dtime(15, 30)

    # Close
    close_start: dtime = dtime(15, 30)
    close_end: dtime = dtime(16, 0)
    block_close: bool = True  # No new entries in final 30 min

    # Market hours
    market_open: dtime = dtime(9, 30)
    market_close: dtime = dtime(16, 0)


@dataclass
class KillZoneStatus:
    """Current status of kill zone check."""
    can_trade: bool
    current_zone: KillZone
    reason: str
    next_window_opens: Optional[dtime] = None
    minutes_until_window: Optional[int] = None


class KillZoneGate:
    """
    Gate that blocks trades outside of valid kill zones.

    This implements professional trading discipline:
    - No trades in first 30 minutes (let dust settle)
    - No trades during lunch (low volume, fake moves)
    - Only trade in primary (10:00-11:30) and power hour (2:30-3:30) windows
    """

    def __init__(self, config: Optional[KillZoneConfig] = None):
        self.config = config or KillZoneConfig()
        self.ET = ZoneInfo('America/New_York')

    def get_current_zone(self, now: Optional[datetime] = None) -> KillZone:
        """Determine which kill zone we're currently in."""
        if now is None:
            now = datetime.now(self.ET)

        current_time = now.time()
        cfg = self.config

        if current_time < cfg.market_open:
            return KillZone.PRE_MARKET
        elif current_time < cfg.opening_range_end:
            return KillZone.OPENING_RANGE
        elif current_time < cfg.lunch_start:
            return KillZone.LONDON_CLOSE
        elif current_time < cfg.lunch_end:
            return KillZone.LUNCH_CHOP
        elif current_time < cfg.power_hour_start:
            return KillZone.LUNCH_CHOP  # Extended lunch
        elif current_time < cfg.close_start:
            return KillZone.POWER_HOUR
        elif current_time < cfg.market_close:
            return KillZone.CLOSE
        else:
            return KillZone.AFTER_HOURS

    def check_can_trade(self, now: Optional[datetime] = None) -> KillZoneStatus:
        """
        Check if trading is allowed right now.

        Returns:
            KillZoneStatus with can_trade, reason, and next window info
        """
        if now is None:
            now = datetime.now(self.ET)

        current_time = now.time()
        zone = self.get_current_zone(now)
        cfg = self.config

        # Calculate minutes until next window
        def minutes_until(target: dtime) -> int:
            now_mins = current_time.hour * 60 + current_time.minute
            target_mins = target.hour * 60 + target.minute
            return max(0, target_mins - now_mins)

        # Pre-market
        if zone == KillZone.PRE_MARKET:
            return KillZoneStatus(
                can_trade=False,
                current_zone=zone,
                reason="Pre-market: Market not open yet",
                next_window_opens=cfg.primary_window_start,
                minutes_until_window=minutes_until(cfg.primary_window_start)
            )

        # Opening range - ALWAYS BLOCKED
        if zone == KillZone.OPENING_RANGE:
            return KillZoneStatus(
                can_trade=False,
                current_zone=zone,
                reason="Opening Range (9:30-10:00): Amateur hour - let volatility settle",
                next_window_opens=cfg.primary_window_start,
                minutes_until_window=minutes_until(cfg.primary_window_start)
            )

        # Primary window - ALLOWED
        if zone == KillZone.LONDON_CLOSE:
            return KillZoneStatus(
                can_trade=True,
                current_zone=zone,
                reason="Primary Window (10:00-11:30): Optimal entry zone"
            )

        # Lunch chop - configurable
        if zone == KillZone.LUNCH_CHOP:
            if cfg.block_lunch:
                return KillZoneStatus(
                    can_trade=False,
                    current_zone=zone,
                    reason="Lunch Chop (11:30-14:30): Low volume, avoid new entries",
                    next_window_opens=cfg.power_hour_start,
                    minutes_until_window=minutes_until(cfg.power_hour_start)
                )
            else:
                return KillZoneStatus(
                    can_trade=True,
                    current_zone=zone,
                    reason="Lunch period: Trading allowed (aggressive mode)"
                )

        # Power hour - ALLOWED
        if zone == KillZone.POWER_HOUR:
            return KillZoneStatus(
                can_trade=True,
                current_zone=zone,
                reason="Power Hour (14:30-15:30): Secondary entry window"
            )

        # Close - blocked for new entries
        if zone == KillZone.CLOSE:
            if cfg.block_close:
                return KillZoneStatus(
                    can_trade=False,
                    current_zone=zone,
                    reason="Close (15:30-16:00): No new entries, manage existing positions"
                )
            else:
                return KillZoneStatus(
                    can_trade=True,
                    current_zone=zone,
                    reason="Close period: Trading allowed (aggressive mode)"
                )

        # After hours
        return KillZoneStatus(
            can_trade=False,
            current_zone=zone,
            reason="After Hours: Market closed",
            next_window_opens=cfg.primary_window_start,
            minutes_until_window=None  # Next trading day
        )

    def is_primary_window(self, now: Optional[datetime] = None) -> bool:
        """Check if we're in the primary trading window (10:00-11:30)."""
        zone = self.get_current_zone(now)
        return zone == KillZone.LONDON_CLOSE

    def is_power_hour(self, now: Optional[datetime] = None) -> bool:
        """Check if we're in power hour (2:30-3:30)."""
        zone = self.get_current_zone(now)
        return zone == KillZone.POWER_HOUR

    def get_next_window(self, now: Optional[datetime] = None) -> Tuple[str, dtime, int]:
        """
        Get info about the next trading window.

        Returns:
            Tuple of (window_name, start_time, minutes_until)
        """
        if now is None:
            now = datetime.now(self.ET)

        current_time = now.time()
        cfg = self.config

        windows = [
            ("Primary Window", cfg.primary_window_start),
            ("Power Hour", cfg.power_hour_start),
        ]

        for name, start in windows:
            if current_time < start:
                mins = (start.hour * 60 + start.minute) - (current_time.hour * 60 + current_time.minute)
                return (name, start, mins)

        # All windows passed, next is tomorrow
        return ("Primary Window (Tomorrow)", cfg.primary_window_start, None)


# Singleton instance for easy access
_default_gate: Optional[KillZoneGate] = None


def get_kill_zone_gate() -> KillZoneGate:
    """Get the default kill zone gate instance."""
    global _default_gate
    if _default_gate is None:
        _default_gate = KillZoneGate()
    return _default_gate


def can_trade_now() -> bool:
    """Simple check if trading is allowed right now."""
    gate = get_kill_zone_gate()
    status = gate.check_can_trade()
    return status.can_trade


def get_current_zone() -> KillZone:
    """Get the current kill zone."""
    gate = get_kill_zone_gate()
    return gate.get_current_zone()


def check_trade_allowed() -> Tuple[bool, str]:
    """
    Check if trading is allowed and return reason.

    Returns:
        Tuple of (allowed, reason)
    """
    gate = get_kill_zone_gate()
    status = gate.check_can_trade()
    return (status.can_trade, status.reason)


if __name__ == "__main__":
    # Test the kill zone gate
    gate = KillZoneGate()
    status = gate.check_can_trade()

    print(f"Current Zone: {status.current_zone.value}")
    print(f"Can Trade: {status.can_trade}")
    print(f"Reason: {status.reason}")

    if status.next_window_opens:
        print(f"Next Window: {status.next_window_opens}")
        print(f"Minutes Until: {status.minutes_until_window}")
