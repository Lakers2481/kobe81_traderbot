from __future__ import annotations

from dataclasses import dataclass
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

    def reset_daily(self):
        self._daily_notional = 0.0

    def check(self, symbol: str, side: str, price: float, qty: int) -> tuple[bool, str]:
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

