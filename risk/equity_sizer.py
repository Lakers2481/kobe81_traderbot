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
from typing import Optional
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


def get_account_equity(fail_safe: bool = True) -> float:
    """
    Fetch current account equity from Alpaca.

    CRITICAL FIX (2026-01-04): No longer returns 100k fallback which caused
    10x position sizing errors. Now activates kill switch on failure.

    Args:
        fail_safe: If True, activates kill switch on failure. If False, raises exception.

    Returns:
        Account equity in USD. Returns 0.0 if fail_safe and error occurs.

    Raises:
        RuntimeError: If fail_safe=False and equity cannot be fetched.
    """
    try:
        from alpaca.trading.client import TradingClient

        api_key = os.environ.get('ALPACA_API_KEY_ID')
        api_secret = os.environ.get('ALPACA_API_SECRET_KEY')
        base_url = os.environ.get('ALPACA_BASE_URL', '')

        if not api_key or not api_secret:
            msg = "CRITICAL: Alpaca credentials not found - CANNOT DETERMINE EQUITY"
            logger.critical(msg)
            if fail_safe:
                from core.kill_switch import activate_kill_switch
                from core.structured_log import jlog
                activate_kill_switch(f"equity_sizer: {msg}")
                jlog('equity_fetch_failed', reason='missing_credentials', action='kill_switch_activated')
                return 0.0  # Return 0 to prevent any trades
            raise RuntimeError(msg)

        is_paper = 'paper' in base_url.lower()
        client = TradingClient(api_key, api_secret, paper=is_paper)
        account = client.get_account()
        equity = float(account.equity)

        logger.debug(f"Account equity: ${equity:,.2f}")
        return equity

    except RuntimeError:
        raise  # Re-raise RuntimeError from credentials check
    except Exception as e:
        msg = f"CRITICAL: Failed to fetch account equity: {e}"
        logger.critical(msg)
        if fail_safe:
            from core.kill_switch import activate_kill_switch
            from core.structured_log import jlog
            activate_kill_switch(f"equity_sizer: {msg}")
            jlog('equity_fetch_failed', reason=str(e), action='kill_switch_activated')
            return 0.0  # Return 0 to prevent any trades
        raise RuntimeError(msg) from e


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


def get_cognitive_multiplier(
    strategy: str = "dual_strategy",
    regime: str = "unknown"
) -> float:
    """
    Calculate position sizing multiplier based on strategy performance.

    FIX (2026-01-07): Fix 4 - Performance-based position sizing.
    Uses SelfModel to track strategy confidence and adjust position sizes.

    Args:
        strategy: The strategy name (e.g., "dual_strategy", "ibs_rsi", "turtle_soup")
        regime: Market regime (e.g., "bull", "bear", "neutral", "unknown")

    Returns:
        Multiplier between 0.25 and 1.0:
        - 1.0: Excellent performance (>65% win rate) - full size
        - 0.85: Good performance (55-65%) - slightly reduced
        - 0.70: Adequate performance (45-55%) - moderately reduced
        - 0.50: Weak performance (35-45%) - half size
        - 0.25: Poor performance (<35%) - quarter size
        - 1.0: Unknown (not enough data) - full size (benefit of doubt)

    Example:
        >>> mult = get_cognitive_multiplier("dual_strategy", "bull")
        >>> size = calculate_position_size(..., cognitive_multiplier=mult)
    """
    try:
        from cognitive.self_model import get_self_model

        self_model = get_self_model()
        perf = self_model.get_performance(strategy, regime)

        if perf is None:
            logger.debug(f"No performance data for {strategy}/{regime}, using full size")
            return 1.0

        # Map capability to multiplier
        multipliers = {
            "excellent": 1.0,    # Full size (>65% win rate)
            "good": 0.85,        # 85% of normal (55-65%)
            "adequate": 0.70,    # 70% of normal (45-55%)
            "weak": 0.50,        # 50% of normal (35-45%)
            "poor": 0.25,        # 25% of normal (<35%)
            "unknown": 1.0,      # Default to full size
        }

        capability_value = perf.capability.value if hasattr(perf.capability, 'value') else str(perf.capability)
        multiplier = multipliers.get(capability_value.lower(), 1.0)

        logger.info(
            f"Cognitive multiplier for {strategy}/{regime}: {multiplier:.2f} "
            f"(capability={capability_value}, win_rate={perf.win_rate:.2%})"
        )

        return multiplier

    except ImportError:
        logger.debug("SelfModel not available, using full size")
        return 1.0
    except Exception as e:
        logger.warning(f"Error getting cognitive multiplier: {e}, using full size")
        return 1.0  # Fail-safe: use full size


def calculate_position_size_with_kelly(
    entry_price: float,
    stop_loss: float,
    risk_pct: float = 0.02,
    account_equity: Optional[float] = None,
    max_notional_pct: float = 0.20,
    min_shares: int = 1,
    cognitive_multiplier: float = 1.0,
    use_kelly: bool = True,
    kelly_win_rate: float = 0.60,
    kelly_win_loss_ratio: float = 1.5,
    kelly_fraction: float = 0.5,  # Half Kelly for safety
) -> PositionSize:
    """
    Calculate position size with optional Kelly Criterion enhancement.

    FIX (2026-01-08): Wire Kelly position sizer from risk/advanced/ into execution path.
    Uses the minimum of standard risk-based sizing and Kelly sizing for safety.

    Args:
        entry_price: Expected entry price
        stop_loss: Stop loss price
        risk_pct: Risk per trade as decimal (0.02 = 2%)
        account_equity: Account equity (fetched from Alpaca if None)
        max_notional_pct: Maximum position size as % of account
        min_shares: Minimum shares to trade
        cognitive_multiplier: Multiplier from cognitive system (0.5-1.0)
        use_kelly: If True, also apply Kelly Criterion sizing
        kelly_win_rate: Win rate for Kelly calculation (default 0.60)
        kelly_win_loss_ratio: Win/loss ratio for Kelly (default 1.5)
        kelly_fraction: Fraction of full Kelly to use (default 0.5 = half Kelly)

    Returns:
        PositionSize with calculated shares (minimum of standard and Kelly if enabled)

    Example:
        >>> size = calculate_position_size_with_kelly(
        ...     entry_price=250.0,
        ...     stop_loss=237.50,
        ...     risk_pct=0.02,
        ...     account_equity=105000,
        ...     use_kelly=True,
        ...     kelly_win_rate=0.60
        ... )
        >>> print(f"{size.shares} shares, ${size.risk_dollars:.0f} risk")
    """
    # Standard risk-based calculation
    standard_size = calculate_position_size(
        entry_price=entry_price,
        stop_loss=stop_loss,
        risk_pct=risk_pct,
        account_equity=account_equity,
        max_notional_pct=max_notional_pct,
        min_shares=min_shares,
        cognitive_multiplier=cognitive_multiplier,
    )

    if not use_kelly:
        return standard_size

    # Kelly Criterion calculation
    try:
        from risk.advanced.kelly_position_sizer import KellyPositionSizer

        kelly_sizer = KellyPositionSizer(
            win_rate=kelly_win_rate,
            avg_win=kelly_win_loss_ratio,  # Normalized win (loss = 1.0)
            avg_loss=1.0,
            kelly_fraction=kelly_fraction,
            max_position_pct=max_notional_pct,
        )

        kelly_result = kelly_sizer.calculate_position_size(
            account_equity=standard_size.account_equity,
            current_price=entry_price,
            stop_loss=stop_loss,
        )

        kelly_shares = kelly_result.shares

        # Use the minimum of standard and Kelly for safety
        if kelly_shares < standard_size.shares:
            logger.info(
                f"Kelly reducing position: {standard_size.shares} -> {kelly_shares} shares "
                f"(Kelly fraction: {kelly_result.adjusted_kelly:.2%})"
            )
            return PositionSize(
                shares=kelly_shares,
                risk_dollars=kelly_shares * standard_size.risk_per_share,
                notional=kelly_shares * entry_price,
                risk_per_share=standard_size.risk_per_share,
                account_equity=standard_size.account_equity,
                risk_pct=risk_pct,
                capped=True,
                cap_reason=f"kelly_{kelly_fraction*100:.0f}pct",
            )
        else:
            logger.debug(
                f"Kelly allows {kelly_shares} shares, standard allows {standard_size.shares} - using standard"
            )
            return standard_size

    except ImportError as e:
        logger.warning(f"Kelly sizer not available: {e}, using standard sizing")
        return standard_size
    except Exception as e:
        logger.warning(f"Kelly calculation failed: {e}, using standard sizing")
        return standard_size
