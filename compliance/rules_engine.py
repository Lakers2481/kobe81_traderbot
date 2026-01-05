from __future__ import annotations

"""
Compliance Rules Engine (optional helper).

Centralizes basic trading rules:
- Max position size (as fraction of equity)
- PDT rule placeholder (pattern day trader)
- No penny stocks (min price)
- Trading hours only (RTH)

This module is not wired into the production path by default; it can be used by
submitters or scanners for extra guardrails where needed.

FIX (2026-01-04): Updated RTH check to use EquitiesCalendar for proper timezone
handling, holiday awareness, and early close detection.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from core.clock.equities_calendar import EquitiesCalendar

# Singleton calendar instance
_calendar: Optional[EquitiesCalendar] = None


def _get_calendar() -> EquitiesCalendar:
    """Get or create singleton EquitiesCalendar."""
    global _calendar
    if _calendar is None:
        _calendar = EquitiesCalendar()
    return _calendar


@dataclass
class RuleConfig:
    max_position_size_frac: float = 0.10  # 10% of equity
    min_price: float = 3.0                # avoid penny stocks
    pdt_enabled: bool = False
    rth_only: bool = True
    rth_start: dtime = dtime(9, 30)       # Legacy: kept for compatibility
    rth_end: dtime = dtime(16, 0)         # Legacy: kept for compatibility
    use_calendar: bool = True             # Use EquitiesCalendar for RTH checks


def _within_rth(ts: datetime, start: dtime, end: dtime, use_calendar: bool = True) -> bool:
    """
    Check if timestamp is within Regular Trading Hours.

    FIX (2026-01-04): Now uses EquitiesCalendar for proper timezone handling,
    holiday awareness, and early close detection.

    Args:
        ts: Timestamp to check (assumed ET if naive, or will be converted)
        start: Legacy RTH start time (only used if use_calendar=False)
        end: Legacy RTH end time (only used if use_calendar=False)
        use_calendar: If True, use EquitiesCalendar; if False, use naive time check

    Returns:
        True if within RTH, False otherwise
    """
    if use_calendar:
        cal = _get_calendar()

        # Ensure timezone-aware datetime in ET
        ET = ZoneInfo("America/New_York")
        if ts.tzinfo is None:
            # Assume naive datetime is already in ET
            ts = ts.replace(tzinfo=ET)
        else:
            # Convert to ET
            ts = ts.astimezone(ET)

        # Check if it's a trading day first (handles weekends and holidays)
        if not cal.is_trading_day(ts):
            return False

        # Use calendar's is_market_open for full validation (handles early closes)
        return cal.is_market_open(ts)
    else:
        # Legacy naive time check (for backwards compatibility)
        t = dtime(ts.hour, ts.minute)
        return (t >= start) and (t <= end)


def evaluate(
    price: float,
    qty: int,
    account_equity: float,
    is_day_trade: bool = False,
    ts: datetime | None = None,
    cfg: RuleConfig | None = None,
) -> Tuple[bool, str]:
    cfg = cfg or RuleConfig()
    # Price floor (no penny stocks)
    if price < cfg.min_price:
        return False, f"min_price:{price:.2f} < {cfg.min_price:.2f}"
    # Max position size fraction
    notional = float(price) * int(qty)
    max_notional = float(cfg.max_position_size_frac) * float(account_equity)
    if notional > max_notional:
        return False, f"max_position_size_exceeded:{notional:.2f} > {max_notional:.2f}"
    # PDT placeholder
    if cfg.pdt_enabled and is_day_trade:
        # Enforcement of PDT capital rules would query broker status and day-trade counts
        # Here we flag as a warning condition instead of a hard veto.
        return False, "pdt_day_trade_restricted"
    # RTH only (with timezone awareness and holiday checking)
    if cfg.rth_only and ts is not None:
        if not _within_rth(ts, cfg.rth_start, cfg.rth_end, use_calendar=cfg.use_calendar):
            return False, "outside_regular_trading_hours"
    return True, "ok"

