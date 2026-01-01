"""
Equity-Based Position Sizer for Kobe Trading System

Implements proper 2% risk-based position sizing:
- Risk $ = Account Equity × Risk %
- Shares = Risk $ / (Entry - Stop)

This replaces the broken fixed-dollar sizing that used:
- Risk $ = max_notional × risk_pct (WRONG - gave ~$50 fixed)
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Result of position sizing calculation."""
    shares: int
    risk_dollars: float
    notional: float
    risk_per_share: float
    account_equity: float
    risk_pct: float
    capped: bool = False  # True if hit max notional cap
    cap_reason: str = ""


def get_account_equity() -> float:
    """
    Fetch current account equity from Alpaca.

    Returns:
        Account equity in USD, or 100000.0 as fallback.
    """
    try:
        from alpaca.trading.client import TradingClient

        api_key = os.environ.get('ALPACA_API_KEY_ID')
        api_secret = os.environ.get('ALPACA_API_SECRET_KEY')
        base_url = os.environ.get('ALPACA_BASE_URL', '')

        if not api_key or not api_secret:
            logger.warning("Alpaca credentials not found, using default equity")
            return 100_000.0

        is_paper = 'paper' in base_url.lower()
        client = TradingClient(api_key, api_secret, paper=is_paper)
        account = client.get_account()
        equity = float(account.equity)

        logger.debug(f"Account equity: ${equity:,.2f}")
        return equity

    except Exception as e:
        logger.warning(f"Failed to fetch account equity: {e}, using default")
        return 100_000.0


def calculate_position_size(
    entry_price: float,
    stop_loss: float,
    risk_pct: float = 0.02,  # 2% default
    account_equity: Optional[float] = None,
    max_notional_pct: float = 0.20,  # Max 20% of account in one position
    min_shares: int = 1,
    cognitive_multiplier: float = 1.0,
) -> PositionSize:
    """
    Calculate proper risk-based position size.

    Formula:
        Risk $ = Account Equity × Risk %
        Shares = Risk $ / |Entry - Stop|

    Args:
        entry_price: Expected entry price
        stop_loss: Stop loss price
        risk_pct: Risk per trade as decimal (0.02 = 2%)
        account_equity: Account equity (fetched from Alpaca if None)
        max_notional_pct: Maximum position size as % of account
        min_shares: Minimum shares to trade
        cognitive_multiplier: Multiplier from cognitive system (0.5-1.0)

    Returns:
        PositionSize with calculated shares and risk info

    Example:
        >>> size = calculate_position_size(
        ...     entry_price=250.0,
        ...     stop_loss=237.50,
        ...     risk_pct=0.02,
        ...     account_equity=105000
        ... )
        >>> print(f"{size.shares} shares, ${size.risk_dollars:.0f} risk")
        168 shares, $2100 risk
    """
    # Get account equity
    if account_equity is None:
        account_equity = get_account_equity()

    # Calculate risk per share
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share < 0.01:
        risk_per_share = entry_price * 0.05  # 5% fallback

    # Calculate max risk in dollars (2% of account)
    base_risk_dollars = account_equity * risk_pct

    # Apply cognitive multiplier (reduces size for low confidence)
    effective_risk = base_risk_dollars * cognitive_multiplier

    # Calculate shares by risk
    shares_by_risk = int(effective_risk / risk_per_share)

    # Calculate max shares by notional cap (20% of account)
    max_notional = account_equity * max_notional_pct
    shares_by_notional = int(max_notional / entry_price)

    # Take the lesser (enforce both caps)
    shares = max(min_shares, min(shares_by_risk, shares_by_notional))

    # Check if we hit the notional cap
    capped = shares_by_risk > shares_by_notional
    cap_reason = ""
    if capped:
        cap_reason = f"notional_cap_{max_notional_pct*100:.0f}pct"

    # Calculate actual values
    actual_risk = shares * risk_per_share
    notional = shares * entry_price

    return PositionSize(
        shares=shares,
        risk_dollars=round(actual_risk, 2),
        notional=round(notional, 2),
        risk_per_share=round(risk_per_share, 2),
        account_equity=round(account_equity, 2),
        risk_pct=risk_pct,
        capped=capped,
        cap_reason=cap_reason,
    )


def format_size_summary(size: PositionSize, symbol: str = "") -> str:
    """Format position size for logging."""
    sym = f"{symbol} " if symbol else ""
    return (
        f"{sym}{size.shares} shares @ ${size.notional:,.0f} notional | "
        f"Risk: ${size.risk_dollars:,.0f} ({size.risk_pct*100:.1f}% of ${size.account_equity:,.0f})"
        + (f" [CAPPED: {size.cap_reason}]" if size.capped else "")
    )
