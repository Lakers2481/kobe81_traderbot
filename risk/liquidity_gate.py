"""
Liquidity Gate
==============

Pre-execution checks for liquidity and spread requirements.
Enforces ADV (Average Daily Volume) and bid-ask spread thresholds
to avoid slippage and execution issues.

Usage:
    from risk.liquidity_gate import LiquidityGate, LiquidityCheck

    gate = LiquidityGate(min_adv_usd=100_000, max_spread_pct=0.50)

    # Check before placing order
    check = gate.check_liquidity(
        symbol='AAPL',
        price=150.0,
        shares=100,
        bid=149.95,
        ask=150.05,
        avg_volume=50_000_000,
    )

    if check.passed:
        broker.place_order(...)
    else:
        logger.warning(f"Liquidity check failed: {check.reason}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class LiquidityIssue(Enum):
    """Types of liquidity issues."""
    INSUFFICIENT_ADV = "insufficient_adv"
    WIDE_SPREAD = "wide_spread"
    LOW_VOLUME = "low_volume"
    LARGE_ORDER_IMPACT = "large_order_impact"
    MISSING_QUOTE = "missing_quote"


@dataclass
class LiquidityCheck:
    """Result of a liquidity check."""
    symbol: str
    passed: bool
    reason: str = ""
    issues: List[LiquidityIssue] = field(default_factory=list)

    # Computed metrics
    adv_usd: float = 0.0
    spread_pct: float = 0.0
    order_pct_of_adv: float = 0.0

    # Thresholds used
    min_adv_usd: float = 0.0
    max_spread_pct: float = 0.0

    checked_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'passed': self.passed,
            'reason': self.reason,
            'issues': [i.value for i in self.issues],
            'adv_usd': self.adv_usd,
            'spread_pct': self.spread_pct,
            'order_pct_of_adv': self.order_pct_of_adv,
            'min_adv_usd': self.min_adv_usd,
            'max_spread_pct': self.max_spread_pct,
            'checked_at': self.checked_at,
        }


class LiquidityGate:
    """
    Pre-execution liquidity gating.

    Checks:
    1. Average Daily Volume (ADV) in USD >= min_adv_usd
    2. Bid-ask spread <= max_spread_pct
    3. Order size <= max_pct_of_adv of ADV (avoid market impact)
    """

    def __init__(
        self,
        min_adv_usd: float = 100_000,
        max_spread_pct: float = 0.50,
        max_pct_of_adv: float = 1.0,
    ):
        """
        Initialize liquidity gate.

        Args:
            min_adv_usd: Minimum average daily volume in USD (default: $100k)
            max_spread_pct: Maximum bid-ask spread as percentage (default: 0.50%)
            max_pct_of_adv: Max order size as % of ADV (default: 1%)
        """
        self.min_adv_usd = min_adv_usd
        self.max_spread_pct = max_spread_pct
        self.max_pct_of_adv = max_pct_of_adv

        self._check_history: List[LiquidityCheck] = []

    def check_liquidity(
        self,
        symbol: str,
        price: float,
        shares: int,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        avg_volume: Optional[int] = None,
        strict: bool = True,
    ) -> LiquidityCheck:
        """
        Check if order passes liquidity requirements.

        Args:
            symbol: Stock symbol
            price: Current price (or mid-price)
            shares: Number of shares to trade
            bid: Best bid price (optional)
            ask: Best ask price (optional)
            avg_volume: Average daily volume in shares (optional)
            strict: If True, any issue fails the check. If False, only critical issues fail.

        Returns:
            LiquidityCheck with pass/fail result and details
        """
        issues = []
        reasons = []

        # Calculate metrics
        order_value = price * shares
        adv_usd = (avg_volume * price) if avg_volume else 0.0
        spread_pct = 0.0

        if bid is not None and ask is not None and ask > 0:
            spread_pct = ((ask - bid) / ask) * 100

        order_pct_of_adv = (order_value / adv_usd * 100) if adv_usd > 0 else 100.0

        # Check ADV
        if avg_volume is None:
            issues.append(LiquidityIssue.MISSING_QUOTE)
            reasons.append("Missing volume data")
        elif adv_usd < self.min_adv_usd:
            issues.append(LiquidityIssue.INSUFFICIENT_ADV)
            reasons.append(
                f"ADV ${adv_usd:,.0f} < min ${self.min_adv_usd:,.0f}"
            )

        # Check spread
        if bid is None or ask is None:
            issues.append(LiquidityIssue.MISSING_QUOTE)
            reasons.append("Missing bid/ask quote")
        elif spread_pct > self.max_spread_pct:
            issues.append(LiquidityIssue.WIDE_SPREAD)
            reasons.append(
                f"Spread {spread_pct:.2f}% > max {self.max_spread_pct:.2f}%"
            )

        # Check order impact
        if order_pct_of_adv > self.max_pct_of_adv:
            issues.append(LiquidityIssue.LARGE_ORDER_IMPACT)
            reasons.append(
                f"Order is {order_pct_of_adv:.2f}% of ADV (max {self.max_pct_of_adv:.2f}%)"
            )

        # Determine pass/fail
        if strict:
            passed = len(issues) == 0
        else:
            # Only fail on critical issues (insufficient ADV, wide spread)
            critical_issues = {
                LiquidityIssue.INSUFFICIENT_ADV,
                LiquidityIssue.WIDE_SPREAD,
            }
            passed = len(set(issues) & critical_issues) == 0

        reason = "; ".join(reasons) if reasons else "All liquidity checks passed"

        check = LiquidityCheck(
            symbol=symbol,
            passed=passed,
            reason=reason,
            issues=issues,
            adv_usd=adv_usd,
            spread_pct=spread_pct,
            order_pct_of_adv=order_pct_of_adv,
            min_adv_usd=self.min_adv_usd,
            max_spread_pct=self.max_spread_pct,
        )

        self._check_history.append(check)
        return check

    def get_stats(self) -> Dict:
        """Get statistics on liquidity checks."""
        if not self._check_history:
            return {
                'total_checks': 0,
                'passed': 0,
                'failed': 0,
                'pass_rate': 0.0,
                'common_issues': {},
            }

        passed = sum(1 for c in self._check_history if c.passed)
        failed = len(self._check_history) - passed

        # Count issues
        issue_counts: Dict[str, int] = {}
        for check in self._check_history:
            for issue in check.issues:
                issue_counts[issue.value] = issue_counts.get(issue.value, 0) + 1

        return {
            'total_checks': len(self._check_history),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(self._check_history) * 100,
            'common_issues': issue_counts,
        }

    def reset_history(self):
        """Clear check history."""
        self._check_history = []


# Singleton instance with production defaults
_default_gate: Optional[LiquidityGate] = None


def get_liquidity_gate() -> LiquidityGate:
    """Get the default liquidity gate instance."""
    global _default_gate
    if _default_gate is None:
        _default_gate = LiquidityGate(
            min_adv_usd=100_000,
            max_spread_pct=0.50,
            max_pct_of_adv=1.0,
        )
    return _default_gate


def check_liquidity(
    symbol: str,
    price: float,
    shares: int,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    avg_volume: Optional[int] = None,
) -> LiquidityCheck:
    """Convenience function to check liquidity using default gate."""
    return get_liquidity_gate().check_liquidity(
        symbol=symbol,
        price=price,
        shares=shares,
        bid=bid,
        ask=ask,
        avg_volume=avg_volume,
    )
