from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Any
import os


@dataclass
class RiskLimits:
    max_notional_per_order: float = 75.0  # canary budget per order
    max_daily_notional: float = 1_000.0   # daily cap
    min_price: float = 3.0
    max_price: float = 1000.0
    allow_shorts: bool = False  # default off


class PolicyGate:
    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self._daily_notional = 0.0
        self._last_reset_date: date = date.today()

    def reset_daily(self):
        """Reset daily notional counter."""
        self._daily_notional = 0.0
        self._last_reset_date = date.today()

    def _auto_reset_if_new_day(self):
        """Automatically reset daily budget if we're on a new trading day."""
        today = date.today()
        if today > self._last_reset_date:
            self.reset_daily()

    def check(self, symbol: str, side: str, price: float, qty: int) -> tuple[bool, str]:
        # Auto-reset daily budget at start of each new day
        self._auto_reset_if_new_day()

        if price <= 0 or qty <= 0:
            return False, "invalid_price_or_qty"
        if price < self.limits.min_price or price > self.limits.max_price:
            return False, "price_out_of_bounds"
        if side.lower() == 'short' and not self.limits.allow_shorts:
            return False, "shorts_disabled"
        notional = price * qty
        if notional > self.limits.max_notional_per_order:
            return False, "exceeds_per_order_budget"
        if self._daily_notional + notional > self.limits.max_daily_notional:
            return False, "exceeds_daily_budget"
        # Passed; update budget
        self._daily_notional += notional
        return True, "ok"

    def get_remaining_daily_budget(self) -> float:
        """Get remaining daily budget in USD."""
        self._auto_reset_if_new_day()
        return self.limits.max_daily_notional - self._daily_notional

    def get_status(self) -> Dict[str, Any]:
        """Get current PolicyGate status."""
        self._auto_reset_if_new_day()
        return {
            "daily_used": round(self._daily_notional, 2),
            "daily_limit": self.limits.max_daily_notional,
            "daily_remaining": round(self.get_remaining_daily_budget(), 2),
            "per_order_limit": self.limits.max_notional_per_order,
            "last_reset": self._last_reset_date.isoformat(),
        }

