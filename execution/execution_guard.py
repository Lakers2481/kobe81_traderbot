"""
Execution guard for pre-trade validation.

Validates quotes, spreads, and trading status before order placement.
CRITICAL: Stand down on ANY uncertainty - reject order if in doubt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


class GuardStatus(Enum):
    """Guard check status."""
    PASSED = auto()
    REJECTED = auto()
    STAND_DOWN = auto()  # Uncertainty - do not trade


@dataclass
class GuardCheckResult:
    """Result of a full guard check."""
    status: GuardStatus
    approved: bool
    symbol: str
    side: str
    qty: int
    price: float
    rejection_reasons: List[str] = field(default_factory=list)
    stand_down_reasons: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(ET).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.name,
            "approved": self.approved,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "price": self.price,
            "rejection_reasons": self.rejection_reasons,
            "stand_down_reasons": self.stand_down_reasons,
            "checks": self.checks,
            "timestamp": self.timestamp,
        }


@dataclass
class QuoteData:
    """Quote data for validation."""
    symbol: str
    bid: Optional[float]
    ask: Optional[float]
    bid_size: Optional[int]
    ask_size: Optional[int]
    timestamp: datetime
    source: str = "unknown"


class ExecutionGuard:
    """
    Pre-execution validation gate.

    Validates:
    - Quote freshness (not stale)
    - Spread within limits
    - Quote validity (bid < ask, both present)
    - Trading status (not halted)

    CRITICAL RULE: Stand down on ANY uncertainty.
    If we cannot validate, we do not trade.
    """

    def __init__(
        self,
        max_quote_age_seconds: float = 5.0,
        max_spread_pct: float = 0.50,
        stand_down_on_uncertainty: bool = True,
        enabled: bool = True,
    ):
        self.max_quote_age_seconds = max_quote_age_seconds
        self.max_spread_pct = max_spread_pct
        self.stand_down_on_uncertainty = stand_down_on_uncertainty
        self.enabled = enabled

    def check_quote_freshness(
        self,
        quote_ts: datetime,
        max_age: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Check if quote is fresh enough for trading.

        Args:
            quote_ts: Timestamp of the quote
            max_age: Maximum age in seconds (uses default if None)

        Returns:
            (passed, reason)
        """
        if max_age is None:
            max_age = self.max_quote_age_seconds

        now = datetime.now(ET)

        # Ensure quote_ts is timezone-aware
        if quote_ts.tzinfo is None:
            quote_ts = quote_ts.replace(tzinfo=ET)

        age_seconds = (now - quote_ts).total_seconds()

        if age_seconds < 0:
            return False, f"Quote timestamp is in the future by {-age_seconds:.1f}s"

        if age_seconds > max_age:
            return False, f"Quote is stale ({age_seconds:.1f}s old, max {max_age}s)"

        return True, f"Quote is fresh ({age_seconds:.1f}s old)"

    def check_spread(
        self,
        bid: float,
        ask: float,
        max_pct: Optional[float] = None,
    ) -> Tuple[bool, float, str]:
        """
        Check if spread is within acceptable limits.

        Args:
            bid: Bid price
            ask: Ask price
            max_pct: Maximum spread as percentage of mid (uses default if None)

        Returns:
            (passed, spread_pct, reason)
        """
        if max_pct is None:
            max_pct = self.max_spread_pct

        if bid <= 0 or ask <= 0:
            return False, 0.0, "Invalid bid/ask prices (zero or negative)"

        if bid >= ask:
            return False, 0.0, f"Crossed market: bid {bid} >= ask {ask}"

        mid = (bid + ask) / 2
        spread = ask - bid
        spread_pct = (spread / mid) * 100

        if spread_pct > max_pct:
            return False, spread_pct, f"Spread {spread_pct:.2f}% exceeds max {max_pct}%"

        return True, spread_pct, f"Spread {spread_pct:.2f}% within limit"

    def check_quote_validity(
        self,
        bid: Optional[float],
        ask: Optional[float],
    ) -> Tuple[bool, str]:
        """
        Check basic quote validity.

        Args:
            bid: Bid price (may be None)
            ask: Ask price (may be None)

        Returns:
            (passed, reason)
        """
        if bid is None and ask is None:
            return False, "No quote data available"

        if bid is None:
            return False, "No bid price available"

        if ask is None:
            return False, "No ask price available"

        if bid <= 0:
            return False, f"Invalid bid price: {bid}"

        if ask <= 0:
            return False, f"Invalid ask price: {ask}"

        if bid >= ask:
            return False, f"Crossed or locked market: bid={bid}, ask={ask}"

        return True, "Quote is valid"

    def check_trading_status(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if symbol is tradable (not halted).

        This is a best-effort check using Alpaca's asset.tradable flag.
        If we cannot determine status, we stand down.

        Returns:
            (passed, reason)
        """
        try:
            # Try to get trading status from Alpaca
            from execution.broker_alpaca import get_asset_info

            asset_info = get_asset_info(symbol)

            if asset_info is None:
                if self.stand_down_on_uncertainty:
                    return False, f"Cannot determine trading status for {symbol}"
                return True, "Trading status unknown, proceeding with caution"

            if not asset_info.get("tradable", False):
                return False, f"Symbol {symbol} is not tradable"

            if asset_info.get("status") == "inactive":
                return False, f"Symbol {symbol} is inactive"

            return True, f"Symbol {symbol} is tradable"

        except ImportError:
            # Broker module not available - stand down if configured
            if self.stand_down_on_uncertainty:
                return False, "Cannot verify trading status - broker module unavailable"
            return True, "Trading status check skipped"

        except Exception as e:
            if self.stand_down_on_uncertainty:
                return False, f"Error checking trading status: {e}"
            return True, f"Trading status check failed, proceeding: {e}"

    def check_size_limits(
        self,
        qty: int,
        price: float,
        min_notional: float = 1.0,
        max_notional: float = 100000.0,
    ) -> Tuple[bool, str]:
        """
        Check order size is within reasonable limits.

        Returns:
            (passed, reason)
        """
        if qty <= 0:
            return False, f"Invalid quantity: {qty}"

        if price <= 0:
            return False, f"Invalid price: {price}"

        notional = qty * price

        if notional < min_notional:
            return False, f"Order too small: ${notional:.2f} (min ${min_notional})"

        if notional > max_notional:
            return False, f"Order too large: ${notional:.2f} (max ${max_notional})"

        return True, f"Order size ${notional:.2f} within limits"

    def full_check(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        quote: Optional[QuoteData] = None,
        skip_trading_status: bool = False,
    ) -> GuardCheckResult:
        """
        Run all guard checks.

        CRITICAL: If any check fails or is uncertain, order is rejected.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            qty: Order quantity
            price: Limit price
            quote: Optional quote data for validation
            skip_trading_status: Skip trading status check (for testing)

        Returns:
            GuardCheckResult with overall status
        """
        if not self.enabled:
            return GuardCheckResult(
                status=GuardStatus.PASSED,
                approved=True,
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                checks={"guard_enabled": False},
            )

        rejection_reasons = []
        stand_down_reasons = []
        checks = {}

        # 1. Check size limits
        size_passed, size_reason = self.check_size_limits(qty, price)
        checks["size_limits"] = {"passed": size_passed, "reason": size_reason}
        if not size_passed:
            rejection_reasons.append(size_reason)

        # 2. Check trading status (if not skipped)
        if not skip_trading_status:
            status_passed, status_reason = self.check_trading_status(symbol)
            checks["trading_status"] = {"passed": status_passed, "reason": status_reason}
            if not status_passed:
                if "cannot" in status_reason.lower() or "error" in status_reason.lower():
                    stand_down_reasons.append(status_reason)
                else:
                    rejection_reasons.append(status_reason)

        # 3. Quote validation (if quote provided)
        if quote is not None:
            # 3a. Quote validity
            validity_passed, validity_reason = self.check_quote_validity(
                quote.bid, quote.ask
            )
            checks["quote_validity"] = {"passed": validity_passed, "reason": validity_reason}
            if not validity_passed:
                if "no quote" in validity_reason.lower():
                    stand_down_reasons.append(validity_reason)
                else:
                    rejection_reasons.append(validity_reason)

            # 3b. Quote freshness
            if quote.timestamp:
                fresh_passed, fresh_reason = self.check_quote_freshness(quote.timestamp)
                checks["quote_freshness"] = {"passed": fresh_passed, "reason": fresh_reason}
                if not fresh_passed:
                    stand_down_reasons.append(fresh_reason)

            # 3c. Spread check (if we have valid bid/ask)
            if quote.bid and quote.ask and quote.bid < quote.ask:
                spread_passed, spread_pct, spread_reason = self.check_spread(
                    quote.bid, quote.ask
                )
                checks["spread"] = {
                    "passed": spread_passed,
                    "spread_pct": spread_pct,
                    "reason": spread_reason,
                }
                if not spread_passed:
                    rejection_reasons.append(spread_reason)

            # 3d. Price sanity check (is our price reasonable vs quote?)
            if quote.bid and quote.ask:
                mid = (quote.bid + quote.ask) / 2
                price_diff_pct = abs(price - mid) / mid * 100
                checks["price_sanity"] = {
                    "mid": mid,
                    "proposed_price": price,
                    "diff_pct": price_diff_pct,
                }
                if price_diff_pct > 5.0:  # More than 5% from mid
                    stand_down_reasons.append(
                        f"Proposed price ${price:.2f} differs {price_diff_pct:.1f}% from mid ${mid:.2f}"
                    )
        else:
            # No quote provided - stand down if configured
            if self.stand_down_on_uncertainty:
                stand_down_reasons.append("No quote data provided for validation")
            checks["quote"] = {"provided": False}

        # Determine final status
        if stand_down_reasons:
            status = GuardStatus.STAND_DOWN
            approved = False
        elif rejection_reasons:
            status = GuardStatus.REJECTED
            approved = False
        else:
            status = GuardStatus.PASSED
            approved = True

        return GuardCheckResult(
            status=status,
            approved=approved,
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            rejection_reasons=rejection_reasons,
            stand_down_reasons=stand_down_reasons,
            checks=checks,
        )

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> ExecutionGuard:
        """Create guard from configuration dictionary."""
        if config is None:
            try:
                from config.settings_loader import load_settings
                settings = load_settings()
                config = settings.get("execution", {}).get("guard", {})
            except Exception:
                config = {}

        return cls(
            max_quote_age_seconds=config.get("max_quote_age_seconds", 5.0),
            max_spread_pct=config.get("max_spread_pct", 0.50),
            stand_down_on_uncertainty=config.get("stand_down_on_uncertainty", True),
            enabled=config.get("enabled", True),
        )


def get_asset_info(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Stub function for getting asset info.

    This will be called by check_trading_status. The actual implementation
    should be in broker_alpaca.py.
    """
    # This is a stub - the actual implementation imports from broker_alpaca
    return None
