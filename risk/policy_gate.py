from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Any, Optional
import os
import yaml


@dataclass
class RiskLimits:
    max_notional_per_order: float = 75.0  # canary budget per order
    max_daily_notional: float = 1_000.0   # daily cap
    min_price: float = 3.0
    max_price: float = 1000.0
    allow_shorts: bool = False  # default off
    max_positions: int = 3  # maximum concurrent positions
    risk_per_trade_pct: float = 0.005  # 0.5% per trade
    mode_name: str = "micro"  # current trading mode


def load_limits_from_config(config_path: Optional[str] = None) -> RiskLimits:
    """Load RiskLimits from config/base.yaml based on trading_mode.

    Args:
        config_path: Optional path to config file. Defaults to config/base.yaml.

    Returns:
        RiskLimits configured for the current trading mode.
    """
    if config_path is None:
        # Find config relative to project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "base.yaml"
    else:
        config_path = Path(config_path)

    # Default limits (micro mode)
    if not config_path.exists():
        return RiskLimits()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get trading mode (default: micro)
    trading_mode = config.get('trading_mode', 'micro')
    modes = config.get('modes', {})
    mode_config = modes.get(trading_mode, {})

    # Also pull from legacy risk section for backwards compatibility
    risk_config = config.get('risk', {})

    return RiskLimits(
        max_notional_per_order=mode_config.get('max_notional_per_order',
                                               risk_config.get('max_order_value', 75.0)),
        max_daily_notional=mode_config.get('max_daily_notional',
                                           risk_config.get('max_daily_loss', 1000.0)),
        min_price=risk_config.get('min_price', 3.0),
        max_price=risk_config.get('max_price', 1000.0),
        allow_shorts=risk_config.get('allow_shorts', False),
        max_positions=mode_config.get('max_positions',
                                      risk_config.get('max_open_positions', 3)),
        risk_per_trade_pct=mode_config.get('risk_per_trade_pct', 0.005),
        mode_name=trading_mode,
    )


class PolicyGate:
    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self._daily_notional = 0.0
        self._last_reset_date: date = date.today()
        self._position_count = 0

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> 'PolicyGate':
        """Create PolicyGate with limits loaded from config file.

        Args:
            config_path: Optional path to config file. Defaults to config/base.yaml.

        Returns:
            PolicyGate configured for the current trading mode.
        """
        limits = load_limits_from_config(config_path)
        return cls(limits=limits)

    def reset_daily(self):
        """Reset daily notional counter."""
        self._daily_notional = 0.0
        self._last_reset_date = date.today()

    def _auto_reset_if_new_day(self):
        """Automatically reset daily budget if we're on a new trading day."""
        today = date.today()
        if today > self._last_reset_date:
            self.reset_daily()

    def check(self, symbol: str, side: str, price: float, qty: int,
              stop_loss: Optional[float] = None) -> tuple[bool, str]:
        """Check if order passes all risk gates.

        Args:
            symbol: Stock symbol
            side: 'long' or 'short'
            price: Entry price
            qty: Number of shares
            stop_loss: Optional stop loss price for risk validation

        Returns:
            Tuple of (passed, reason)
        """
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

        # Note: Risk-based sizing is now handled by equity_sizer.calculate_position_size()
        # which enforces 2% of account equity per trade. PolicyGate only checks notional caps.

        # Passed; update budget
        self._daily_notional += notional
        return True, "ok"

    def get_remaining_daily_budget(self) -> float:
        """Get remaining daily budget in USD."""
        self._auto_reset_if_new_day()
        return self.limits.max_daily_notional - self._daily_notional

    def check_position_limit(self, current_positions: int) -> tuple[bool, str]:
        """Check if adding a new position would exceed the limit.

        Args:
            current_positions: Current number of open positions.

        Returns:
            Tuple of (passed, reason).
        """
        if current_positions >= self.limits.max_positions:
            return False, f"max_positions_reached ({self.limits.max_positions})"
        return True, "ok"

    def update_position_count(self, count: int):
        """Update the current position count."""
        self._position_count = count

    def get_status(self) -> Dict[str, Any]:
        """Get current PolicyGate status."""
        self._auto_reset_if_new_day()
        return {
            "trading_mode": self.limits.mode_name,
            "daily_used": round(self._daily_notional, 2),
            "daily_limit": self.limits.max_daily_notional,
            "daily_remaining": round(self.get_remaining_daily_budget(), 2),
            "per_order_limit": self.limits.max_notional_per_order,
            "max_positions": self.limits.max_positions,
            "current_positions": self._position_count,
            "risk_per_trade_pct": self.limits.risk_per_trade_pct,
            "last_reset": self._last_reset_date.isoformat(),
        }

