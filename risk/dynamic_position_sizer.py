"""
Dynamic Position Sizer - Professional Portfolio Allocation

Implements adaptive position sizing based on signal quality and count:
- 1 A+ signal  = 10% allocation
- 2 A+ signals = 10% each (20% total)
- 3 A+ signals = 6.67% each (20% total)
- 4+ A+ signals = split 20% evenly, floor at 5% each

Daily budget cap: 20%
Per-position cap: 10%
Per-position floor: 5%

Quant Interview Ready: Implements professional capital allocation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from core.structured_log import jlog


@dataclass
class AllocationResult:
    """Result of dynamic allocation calculation."""
    symbol: str
    allocated_notional: float
    allocated_pct: float
    shares: int
    entry_price: float
    stop_loss: float
    risk_dollars: float
    risk_pct: float
    capped: bool = False
    cap_reason: Optional[str] = None


def calculate_dynamic_allocations(
    signals: List[Dict],
    account_equity: float,
    max_daily_pct: float = 0.20,
    max_per_position_pct: float = 0.10,
    min_per_position_pct: float = 0.05,
    risk_pct: float = 0.02,
) -> List[AllocationResult]:
    """
    Calculate dynamic position sizes based on signal count.

    Professional allocation logic:
    - Daily budget = 20% of account
    - Split evenly among qualified signals
    - Each capped at 10% max, floored at 5% min
    - Final shares based on risk (2%) but capped by allocation

    Args:
        signals: List of signal dicts with symbol, entry_price, stop_loss
        account_equity: Current account equity
        max_daily_pct: Max daily exposure (default 20%)
        max_per_position_pct: Max per position (default 10%)
        min_per_position_pct: Min per position (default 5%)
        risk_pct: Risk per trade (default 2%)

    Returns:
        List of AllocationResult with sized positions
    """
    if not signals:
        return []

    n_signals = len(signals)
    daily_budget = account_equity * max_daily_pct
    max_per_position = account_equity * max_per_position_pct
    min_per_position = account_equity * min_per_position_pct

    # Calculate per-signal allocation
    per_signal_notional = daily_budget / n_signals

    # Apply caps and floors
    per_signal_notional = min(per_signal_notional, max_per_position)
    per_signal_notional = max(per_signal_notional, min_per_position)

    # If we can't fit all signals within budget after floor, take fewer
    if per_signal_notional * n_signals > daily_budget:
        # Can only take floor(daily_budget / min_per_position) signals
        max_signals = int(daily_budget / min_per_position)
        signals = signals[:max_signals]
        n_signals = len(signals)
        per_signal_notional = daily_budget / n_signals if n_signals > 0 else 0

    jlog('dynamic_allocation_calculated',
         n_signals=n_signals,
         daily_budget=daily_budget,
         per_signal=per_signal_notional,
         equity=account_equity)

    results = []
    for sig in signals:
        symbol = sig.get('symbol', '')
        entry_price = float(sig.get('entry_price', 0))
        stop_loss = float(sig.get('stop_loss', 0))

        if entry_price <= 0:
            continue

        # Calculate risk-based shares
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            risk_per_share = entry_price * 0.05  # Default 5% stop

        risk_dollars = account_equity * risk_pct
        shares_by_risk = int(risk_dollars / risk_per_share)

        # Calculate allocation-based shares
        shares_by_allocation = int(per_signal_notional / entry_price)

        # Take the lesser (enforce both risk and allocation caps)
        final_shares = min(shares_by_risk, shares_by_allocation)
        final_shares = max(1, final_shares)  # At least 1 share

        final_notional = final_shares * entry_price
        final_pct = final_notional / account_equity
        final_risk = final_shares * risk_per_share

        # Determine if capped and why
        capped = False
        cap_reason = None
        if shares_by_allocation < shares_by_risk:
            capped = True
            cap_reason = 'allocation_cap'
        elif final_pct > max_per_position_pct:
            capped = True
            cap_reason = 'position_cap'

        result = AllocationResult(
            symbol=symbol,
            allocated_notional=final_notional,
            allocated_pct=final_pct,
            shares=final_shares,
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_dollars=final_risk,
            risk_pct=final_risk / account_equity,
            capped=capped,
            cap_reason=cap_reason,
        )

        results.append(result)

        jlog('dynamic_allocation_result',
             symbol=symbol,
             shares=final_shares,
             notional=final_notional,
             pct=f"{final_pct:.1%}",
             risk=final_risk,
             capped=capped,
             cap_reason=cap_reason)

    return results


def calculate_single_allocation(
    entry_price: float,
    stop_loss: float,
    account_equity: float,
    available_budget: float,
    max_per_position_pct: float = 0.10,
    risk_pct: float = 0.02,
) -> AllocationResult:
    """
    Calculate allocation for a single signal given available budget.

    Used when processing signals one at a time.

    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        account_equity: Current account equity
        available_budget: Remaining daily/weekly budget in dollars
        max_per_position_pct: Max per position (default 10%)
        risk_pct: Risk per trade (default 2%)

    Returns:
        AllocationResult with sized position
    """
    # Calculate caps
    max_per_position = account_equity * max_per_position_pct
    allocation = min(available_budget, max_per_position)

    # Calculate risk-based shares
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share <= 0:
        risk_per_share = entry_price * 0.05

    risk_dollars = account_equity * risk_pct
    shares_by_risk = int(risk_dollars / risk_per_share)

    # Calculate allocation-based shares
    shares_by_allocation = int(allocation / entry_price)

    # Take the lesser
    final_shares = min(shares_by_risk, shares_by_allocation)
    final_shares = max(1, final_shares)

    final_notional = final_shares * entry_price
    final_pct = final_notional / account_equity
    final_risk = final_shares * risk_per_share

    capped = shares_by_allocation < shares_by_risk
    cap_reason = 'allocation_cap' if capped else None

    return AllocationResult(
        symbol='',  # Caller should set
        allocated_notional=final_notional,
        allocated_pct=final_pct,
        shares=final_shares,
        entry_price=entry_price,
        stop_loss=stop_loss,
        risk_dollars=final_risk,
        risk_pct=final_risk / account_equity,
        capped=capped,
        cap_reason=cap_reason,
    )


def format_allocation_summary(results: List[AllocationResult]) -> str:
    """Format allocation results for display."""
    lines = ["DYNAMIC ALLOCATION SUMMARY", "=" * 50]

    total_notional = 0
    total_risk = 0

    for r in results:
        total_notional += r.allocated_notional
        total_risk += r.risk_dollars

        cap_str = f" [{r.cap_reason}]" if r.capped else ""
        lines.append(
            f"{r.symbol}: {r.shares} shares @ ${r.entry_price:.2f} = "
            f"${r.allocated_notional:,.0f} ({r.allocated_pct:.1%})"
            f"{cap_str}"
        )
        lines.append(f"  Risk: ${r.risk_dollars:.0f} ({r.risk_pct:.2%}) | Stop: ${r.stop_loss:.2f}")

    lines.append("-" * 50)
    lines.append(f"TOTAL: ${total_notional:,.0f} | Risk: ${total_risk:,.0f}")
    lines.append("=" * 50)

    return "\n".join(lines)
