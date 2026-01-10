"""
Intraday Entry Trigger
======================

Delays entry until price confirms intraday (VWAP reclaim or first-hour retest).
This helps avoid entering trades that immediately go against you.

Trigger Modes:
- vwap_reclaim: For LONG, price must be above session VWAP
                For SHORT, price must be below session VWAP
- first_hour_high: For LONG, price must break first hour's high
- first_hour_low: For SHORT, price must break first hour's low
- combined: VWAP + first hour confirmation

FIX (2026-01-04): Added Prometheus counter for trigger skips (when no data
available or credentials missing). Provides observability for debugging.

Usage:
    from execution.intraday_trigger import IntradayTrigger, TriggerResult

    trigger = IntradayTrigger(mode="vwap_reclaim")
    result = trigger.check_trigger("AAPL", "long")

    if result.triggered:
        # Safe to enter
        place_order(...)
    else:
        # Wait for trigger
        print(f"Waiting: {result.reason}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

# Using Polygon for intraday data (paid subscription)
# Switched from Alpaca which requires separate paid data subscription
from data.providers.polygon_intraday import (
    fetch_intraday_bars,
)

# FIX (2026-01-04): Import Prometheus counter for skip observability
from trade_logging.prometheus_metrics import INTRADAY_TRIGGER_SKIPPED

# FIX (2026-01-05): Import kill switch for safety check
from core.kill_switch import is_kill_switch_active

logger = logging.getLogger(__name__)


class TriggerMode(str, Enum):
    """Intraday entry trigger modes."""
    VWAP_RECLAIM = "vwap_reclaim"
    FIRST_HOUR_HIGH = "first_hour_high"
    FIRST_HOUR_LOW = "first_hour_low"
    COMBINED = "combined"


@dataclass
class TriggerResult:
    """Result of a trigger check."""
    triggered: bool
    symbol: str
    side: str
    price: Optional[float]
    vwap: Optional[float]
    first_hour_high: Optional[float]
    first_hour_low: Optional[float]
    reason: str
    checked_at: str


class IntradayTrigger:
    """
    Intraday entry trigger for confirming trade entries.

    Delays order submission until intraday price action confirms
    the trade direction (e.g., VWAP reclaim, first hour breakout).
    """

    def __init__(self, mode: str = "vwap_reclaim"):
        """
        Initialize the intraday trigger.

        Args:
            mode: Trigger mode - one of:
                  "vwap_reclaim" - Price above/below VWAP for long/short
                  "first_hour_high" - Price above first hour high for long
                  "first_hour_low" - Price below first hour low for short
                  "combined" - Both VWAP and first hour confirmation
        """
        self.mode = TriggerMode(mode)
        logger.info(f"IntradayTrigger initialized with mode: {self.mode.value}")

    def check_trigger(self, symbol: str, side: str) -> TriggerResult:
        """
        Check if the intraday trigger condition is met.

        Args:
            symbol: Stock ticker symbol
            side: Trade side ("long" or "short")

        Returns:
            TriggerResult with triggered status and details
        """
        side = side.lower()
        now = datetime.utcnow().isoformat()

        # FIX (2026-01-05): Check kill switch FIRST - never poll while kill-switched
        if is_kill_switch_active():
            INTRADAY_TRIGGER_SKIPPED.labels(reason="kill_switch").inc()
            logger.warning(f"Intraday trigger blocked for {symbol}: kill switch active")
            return TriggerResult(
                triggered=False,
                symbol=symbol,
                side=side,
                price=None,
                vwap=None,
                first_hour_high=None,
                first_hour_low=None,
                reason="Kill switch active - trading halted",
                checked_at=now,
            )

        # Fetch current data
        bars = fetch_intraday_bars(symbol, timeframe="5Min", limit=78)

        if not bars:
            # FIX (2026-01-04): Increment Prometheus counter for observability
            INTRADAY_TRIGGER_SKIPPED.labels(reason="no_data").inc()
            logger.warning(f"Intraday trigger skipped for {symbol}: no data available")
            return TriggerResult(
                triggered=False,
                symbol=symbol,
                side=side,
                price=None,
                vwap=None,
                first_hour_high=None,
                first_hour_low=None,
                reason="No intraday data available",
                checked_at=now,
            )

        current_price = bars[-1].close
        current_vwap = bars[-1].vwap

        # Calculate first hour range if enough bars
        first_hour_high = None
        first_hour_low = None
        if len(bars) >= 12:
            first_hour_high = max(b.high for b in bars[:12])
            first_hour_low = min(b.low for b in bars[:12])

        # Check trigger based on mode
        triggered = False
        reason = ""

        if self.mode == TriggerMode.VWAP_RECLAIM:
            triggered, reason = self._check_vwap_trigger(
                side, current_price, current_vwap
            )

        elif self.mode == TriggerMode.FIRST_HOUR_HIGH:
            triggered, reason = self._check_first_hour_high_trigger(
                side, current_price, first_hour_high, first_hour_low
            )

        elif self.mode == TriggerMode.FIRST_HOUR_LOW:
            triggered, reason = self._check_first_hour_low_trigger(
                side, current_price, first_hour_high, first_hour_low
            )

        elif self.mode == TriggerMode.COMBINED:
            triggered, reason = self._check_combined_trigger(
                side, current_price, current_vwap, first_hour_high, first_hour_low
            )

        return TriggerResult(
            triggered=triggered,
            symbol=symbol,
            side=side,
            price=current_price,
            vwap=current_vwap,
            first_hour_high=first_hour_high,
            first_hour_low=first_hour_low,
            reason=reason,
            checked_at=now,
        )

    def _check_vwap_trigger(
        self, side: str, price: float, vwap: float
    ) -> tuple[bool, str]:
        """Check VWAP reclaim trigger."""
        if vwap is None or vwap <= 0:
            return False, "VWAP data unavailable"

        if side == "long":
            if price > vwap:
                return True, f"TRIGGERED: Price ${price:.2f} > VWAP ${vwap:.2f}"
            else:
                return False, f"Waiting: Price ${price:.2f} < VWAP ${vwap:.2f}"
        else:  # short
            if price < vwap:
                return True, f"TRIGGERED: Price ${price:.2f} < VWAP ${vwap:.2f}"
            else:
                return False, f"Waiting: Price ${price:.2f} > VWAP ${vwap:.2f}"

    def _check_first_hour_high_trigger(
        self, side: str, price: float, fh_high: Optional[float], fh_low: Optional[float]
    ) -> tuple[bool, str]:
        """Check first hour high breakout trigger (for longs)."""
        if fh_high is None:
            return False, "First hour not complete"

        if side == "long":
            if price > fh_high:
                return True, f"TRIGGERED: Price ${price:.2f} > FH High ${fh_high:.2f}"
            else:
                return False, f"Waiting: Price ${price:.2f} < FH High ${fh_high:.2f}"
        else:  # short uses first hour low
            if fh_low and price < fh_low:
                return True, f"TRIGGERED: Price ${price:.2f} < FH Low ${fh_low:.2f}"
            else:
                return False, f"Waiting: Price ${price:.2f} > FH Low ${fh_low:.2f}"

    def _check_first_hour_low_trigger(
        self, side: str, price: float, fh_high: Optional[float], fh_low: Optional[float]
    ) -> tuple[bool, str]:
        """Check first hour low breakdown trigger (for shorts)."""
        if fh_low is None:
            return False, "First hour not complete"

        if side == "short":
            if price < fh_low:
                return True, f"TRIGGERED: Price ${price:.2f} < FH Low ${fh_low:.2f}"
            else:
                return False, f"Waiting: Price ${price:.2f} > FH Low ${fh_low:.2f}"
        else:  # long uses first hour high
            if fh_high and price > fh_high:
                return True, f"TRIGGERED: Price ${price:.2f} > FH High ${fh_high:.2f}"
            else:
                return False, f"Waiting: Price ${price:.2f} < FH High ${fh_high:.2f}"

    def _check_combined_trigger(
        self,
        side: str,
        price: float,
        vwap: float,
        fh_high: Optional[float],
        fh_low: Optional[float],
    ) -> tuple[bool, str]:
        """Check combined VWAP + first hour trigger."""
        vwap_ok, vwap_reason = self._check_vwap_trigger(side, price, vwap)

        if side == "long":
            fh_ok, fh_reason = self._check_first_hour_high_trigger(
                side, price, fh_high, fh_low
            )
        else:
            fh_ok, fh_reason = self._check_first_hour_low_trigger(
                side, price, fh_high, fh_low
            )

        if vwap_ok and fh_ok:
            return True, "TRIGGERED: VWAP + First Hour confirmed"
        elif vwap_ok and not fh_ok:
            return False, f"VWAP OK, but {fh_reason}"
        elif not vwap_ok and fh_ok:
            return False, f"FH OK, but {vwap_reason}"
        else:
            return False, f"Waiting: {vwap_reason}; {fh_reason}"


# Global instance
_trigger: Optional[IntradayTrigger] = None


def get_intraday_trigger(mode: str = "vwap_reclaim") -> IntradayTrigger:
    """Get or create global IntradayTrigger instance."""
    global _trigger
    if _trigger is None or _trigger.mode.value != mode:
        _trigger = IntradayTrigger(mode=mode)
    return _trigger


def check_entry_trigger(symbol: str, side: str, mode: str = "vwap_reclaim") -> TriggerResult:
    """
    Convenience function to check entry trigger.

    Args:
        symbol: Stock ticker symbol
        side: Trade side ("long" or "short")
        mode: Trigger mode

    Returns:
        TriggerResult
    """
    trigger = get_intraday_trigger(mode)
    return trigger.check_trigger(symbol, side)
