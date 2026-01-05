"""
Drawdown Circuit Breaker - Capital Protection

Protects capital by halting trading when drawdown thresholds are exceeded.
This is the most critical breaker - it prevents catastrophic losses.

Thresholds (configurable):
- Daily: -2% → HALT_ALL
- Weekly: -5% → HALT_ALL
- Max (from peak): -10% → HALT_ALL

Warning Levels:
- Daily: -1% → REDUCE_SIZE
- Weekly: -3% → REDUCE_SIZE
- Max: -5% → REDUCE_SIZE

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

from core.structured_log import get_logger
from .breaker_manager import BreakerAction, BreakerStatus

logger = get_logger(__name__)


@dataclass
class DrawdownThresholds:
    """Drawdown thresholds for circuit breaker."""
    # HALT thresholds (RED - stop all trading)
    daily_halt: float = 0.02       # -2% daily
    weekly_halt: float = 0.05      # -5% weekly
    max_halt: float = 0.10         # -10% from peak

    # WARNING thresholds (YELLOW - reduce size)
    daily_warning: float = 0.01    # -1% daily
    weekly_warning: float = 0.03   # -3% weekly
    max_warning: float = 0.05      # -5% from peak


class DrawdownBreaker:
    """
    Circuit breaker that monitors drawdown levels.

    Solo Trader Features:
    - Tracks daily, weekly, and max drawdown
    - Auto-reduces position size at warning levels
    - Full halt at critical levels
    - State persistence for accurate peak tracking
    """

    def __init__(self, thresholds: Optional[DrawdownThresholds] = None):
        """
        Initialize drawdown breaker.

        Args:
            thresholds: Custom thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or DrawdownThresholds()
        self._peak_equity: Optional[float] = None
        self._week_start_equity: Optional[float] = None
        self._day_start_equity: Optional[float] = None
        self._last_reset_date: Optional[datetime] = None

    def _update_peaks(self, equity: float, peak_equity: float) -> None:
        """Update internal peak tracking."""
        self._peak_equity = max(self._peak_equity or peak_equity, peak_equity)

        now = datetime.now()

        # Reset daily tracking at market open
        if self._last_reset_date is None or self._last_reset_date.date() != now.date():
            self._day_start_equity = equity
            self._last_reset_date = now

        # Reset weekly tracking on Monday
        if self._week_start_equity is None or now.weekday() == 0:
            if self._last_reset_date is None or self._last_reset_date.weekday() != 0:
                self._week_start_equity = equity

    def check(
        self,
        equity: float,
        peak_equity: float,
        daily_pnl: float,
        weekly_pnl: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Check drawdown levels against thresholds.

        Args:
            equity: Current account equity
            peak_equity: Peak equity value
            daily_pnl: Today's P&L as fraction of equity (negative = loss)
            weekly_pnl: This week's P&L as fraction of equity

        Returns:
            Dict with status, action, message, and details
        """
        self._update_peaks(equity, peak_equity)

        # Calculate drawdowns (as positive numbers for comparison)
        daily_dd = abs(min(daily_pnl, 0))
        weekly_dd = abs(min(weekly_pnl, 0))
        max_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

        # Track which threshold triggered
        triggered_by = None
        status = BreakerStatus.GREEN
        action = BreakerAction.CONTINUE
        threshold_hit = 0
        current_value = 0

        # Check HALT conditions (highest priority)
        if daily_dd >= self.thresholds.daily_halt:
            status = BreakerStatus.RED
            action = BreakerAction.HALT_ALL
            triggered_by = "daily"
            threshold_hit = self.thresholds.daily_halt
            current_value = daily_dd
            logger.warning(f"HALT: Daily drawdown {daily_dd:.2%} >= {self.thresholds.daily_halt:.2%}")

        elif weekly_dd >= self.thresholds.weekly_halt:
            status = BreakerStatus.RED
            action = BreakerAction.HALT_ALL
            triggered_by = "weekly"
            threshold_hit = self.thresholds.weekly_halt
            current_value = weekly_dd
            logger.warning(f"HALT: Weekly drawdown {weekly_dd:.2%} >= {self.thresholds.weekly_halt:.2%}")

        elif max_dd >= self.thresholds.max_halt:
            status = BreakerStatus.RED
            action = BreakerAction.HALT_ALL
            triggered_by = "max"
            threshold_hit = self.thresholds.max_halt
            current_value = max_dd
            logger.warning(f"HALT: Max drawdown {max_dd:.2%} >= {self.thresholds.max_halt:.2%}")

        # Check WARNING conditions
        elif daily_dd >= self.thresholds.daily_warning:
            status = BreakerStatus.YELLOW
            action = BreakerAction.REDUCE_SIZE
            triggered_by = "daily"
            threshold_hit = self.thresholds.daily_warning
            current_value = daily_dd
            logger.info(f"WARNING: Daily drawdown {daily_dd:.2%} >= {self.thresholds.daily_warning:.2%}")

        elif weekly_dd >= self.thresholds.weekly_warning:
            status = BreakerStatus.YELLOW
            action = BreakerAction.REDUCE_SIZE
            triggered_by = "weekly"
            threshold_hit = self.thresholds.weekly_warning
            current_value = weekly_dd
            logger.info(f"WARNING: Weekly drawdown {weekly_dd:.2%} >= {self.thresholds.weekly_warning:.2%}")

        elif max_dd >= self.thresholds.max_warning:
            status = BreakerStatus.YELLOW
            action = BreakerAction.REDUCE_SIZE
            triggered_by = "max"
            threshold_hit = self.thresholds.max_warning
            current_value = max_dd
            logger.info(f"WARNING: Max drawdown {max_dd:.2%} >= {self.thresholds.max_warning:.2%}")

        # Build message
        if triggered_by:
            message = f"{triggered_by.capitalize()} drawdown {current_value:.2%} exceeds threshold {threshold_hit:.2%}"
        else:
            message = f"Drawdown levels normal: daily={daily_dd:.2%}, weekly={weekly_dd:.2%}, max={max_dd:.2%}"

        return {
            "status": status,
            "action": action,
            "message": message,
            "triggered_by": triggered_by,
            "threshold": threshold_hit,
            "current_value": current_value,
            "details": {
                "daily_dd": daily_dd,
                "weekly_dd": weekly_dd,
                "max_dd": max_dd,
                "equity": equity,
                "peak_equity": peak_equity,
                "thresholds": {
                    "daily_halt": self.thresholds.daily_halt,
                    "weekly_halt": self.thresholds.weekly_halt,
                    "max_halt": self.thresholds.max_halt,
                    "daily_warning": self.thresholds.daily_warning,
                    "weekly_warning": self.thresholds.weekly_warning,
                    "max_warning": self.thresholds.max_warning,
                },
            },
        }

    def get_thresholds(self) -> DrawdownThresholds:
        """Get current thresholds."""
        return self.thresholds

    def set_thresholds(self, thresholds: DrawdownThresholds) -> None:
        """Update thresholds."""
        self.thresholds = thresholds
        logger.info(f"Drawdown thresholds updated: {thresholds}")


if __name__ == "__main__":
    # Demo
    breaker = DrawdownBreaker()

    print("=== Drawdown Breaker Demo ===\n")

    # Test scenarios
    scenarios = [
        {"equity": 50000, "peak": 50000, "daily": -0.005, "weekly": -0.01, "desc": "Normal day"},
        {"equity": 49000, "peak": 50000, "daily": -0.015, "weekly": -0.02, "desc": "Yellow - daily warning"},
        {"equity": 48500, "peak": 50000, "daily": -0.025, "weekly": -0.03, "desc": "RED - daily halt"},
        {"equity": 47000, "peak": 52000, "daily": -0.01, "weekly": -0.06, "desc": "RED - weekly halt"},
        {"equity": 45000, "peak": 50000, "daily": -0.01, "weekly": -0.03, "desc": "RED - max halt"},
    ]

    for s in scenarios:
        result = breaker.check(
            equity=s["equity"],
            peak_equity=s["peak"],
            daily_pnl=s["daily"],
            weekly_pnl=s["weekly"],
        )
        print(f"{s['desc']}:")
        print(f"  Status: {result['status'].value}")
        print(f"  Action: {result['action'].value}")
        print(f"  Message: {result['message']}")
        print()
