"""
Kelly Criterion Position Sizing Module
=======================================

Dynamic position sizing using the Kelly Criterion for optimal capital allocation.

The Kelly Criterion calculates the optimal position size to maximize long-term growth
while managing risk. This implementation includes several safety features:

1. Fractional Kelly (default 0.5) - Uses half Kelly for safety
2. Volatility adjustment - Reduces size when volatility is high
3. Position caps - Maximum position size limit (default 25%)
4. Dynamic updates - Can update statistics from recent trade results

Mathematical Foundation:
    f* = (p*b - q) / b
    where:
        f* = optimal fraction of capital to risk
        p = win rate (probability of winning)
        q = 1 - p (probability of losing)
        b = win/loss ratio (avg_win / avg_loss)

MERGED FROM GAME_PLAN_2K28 - Production Ready
Validated System Performance:
- Win Rate: 66.96%
- Profit Factor: 1.53
- Total Trades: 1,501 (2021-2025)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class KellyPositionResult:
    """Result of Kelly Criterion position sizing calculation."""
    # Basic position info
    shares: int
    position_value: float
    risk_amount: float

    # Kelly calculations
    kelly_fraction: float
    fractional_kelly: float
    adjusted_kelly: float

    # Statistics used
    win_rate: float
    win_loss_ratio: float

    # Adjustments applied
    volatility_adjustment: float
    capped: bool
    original_shares: int

    # Metadata
    account_equity: float
    current_price: float
    stop_loss: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'shares': self.shares,
            'position_value': round(self.position_value, 2),
            'risk_amount': round(self.risk_amount, 2),
            'kelly_fraction': round(self.kelly_fraction, 4),
            'fractional_kelly': round(self.fractional_kelly, 4),
            'adjusted_kelly': round(self.adjusted_kelly, 4),
            'win_rate': round(self.win_rate, 4),
            'win_loss_ratio': round(self.win_loss_ratio, 4),
            'volatility_adjustment': round(self.volatility_adjustment, 4),
            'capped': self.capped,
            'original_shares': self.original_shares,
            'account_equity': round(self.account_equity, 2),
            'current_price': round(self.current_price, 2),
            'stop_loss': round(self.stop_loss, 2) if self.stop_loss else None,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Kelly Position: {self.shares} shares @ ${self.current_price:.2f} "
            f"(${self.position_value:,.0f}) | "
            f"Kelly: {self.kelly_fraction:.1%} -> {self.adjusted_kelly:.1%} | "
            f"Risk: ${self.risk_amount:,.0f}"
        )


def optimal_kelly(win_rate: float, win_loss_ratio: float) -> float:
    """
    Calculate the optimal Kelly fraction.

    Formula: f* = (p*b - q) / b
    where: p = win_rate, q = 1 - win_rate, b = win_loss_ratio

    Args:
        win_rate: Probability of winning (0-1)
        win_loss_ratio: Average win / average loss

    Returns:
        Optimal Kelly fraction (0-1). Can be negative if edge is negative.
    """
    if win_loss_ratio <= 0:
        logger.warning(f"Invalid win_loss_ratio: {win_loss_ratio}. Returning 0.")
        return 0.0

    if not (0 <= win_rate <= 1):
        logger.warning(f"Win rate {win_rate} outside [0,1]. Clamping.")
        win_rate = max(0.0, min(1.0, win_rate))

    p = win_rate
    q = 1.0 - win_rate
    b = win_loss_ratio

    kelly = (p * b - q) / b
    return kelly


def fractional_kelly(kelly: float, fraction: float = 0.5) -> float:
    """
    Apply fractional Kelly for safer position sizing.

    Full Kelly maximizes growth but can be too aggressive.
    Fractional Kelly (typically 0.25 to 0.5) provides better risk/reward.

    Args:
        kelly: Full Kelly fraction from optimal_kelly()
        fraction: Fraction to use (default: 0.5 = half Kelly)

    Returns:
        Fractional Kelly value (kelly * fraction)
    """
    if not (0 <= fraction <= 2):
        logger.warning(f"Unusual fraction {fraction}. Expected [0, 1].")

    result = kelly * fraction
    return max(0.0, result)


def volatility_adjusted_kelly(
    kelly: float,
    current_volatility: float,
    baseline_volatility: float = 0.02,
    max_adjustment: float = 0.5
) -> float:
    """
    Adjust Kelly fraction based on market volatility.

    When volatility increases, reduce position size.
    When volatility decreases, can use larger positions.

    Args:
        kelly: Kelly fraction to adjust
        current_volatility: Current market volatility (ATR / price)
        baseline_volatility: Normal/expected volatility (default: 0.02 = 2%)
        max_adjustment: Maximum adjustment factor (default: 0.5)

    Returns:
        Volatility-adjusted Kelly fraction
    """
    if current_volatility <= 0:
        logger.warning(f"Invalid volatility {current_volatility}. Using kelly unchanged.")
        return kelly

    vol_ratio = baseline_volatility / current_volatility
    vol_ratio = max(max_adjustment, min(vol_ratio, 2.0))

    adjusted = kelly * vol_ratio

    logger.debug(
        f"Volatility adjustment: kelly={kelly:.3f}, "
        f"current_vol={current_volatility:.3%}, "
        f"baseline_vol={baseline_volatility:.3%}, "
        f"ratio={vol_ratio:.3f}, adjusted={adjusted:.3f}"
    )

    return adjusted


class KellyPositionSizer:
    """
    Kelly Criterion-based position sizer with safety features.

    Key Features:
    - Calculates optimal position size based on win rate and win/loss ratio
    - Uses fractional Kelly (default 0.5) to reduce risk
    - Adjusts for volatility (reduce size when volatility is high)
    - Caps maximum position size (default 25% of equity)
    - Updates statistics from recent trades for adaptive sizing
    """

    def __init__(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.5,
        max_position_pct: float = 0.25,
        baseline_volatility: float = 0.02
    ):
        """
        Initialize Kelly Position Sizer.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount (% or $, must be positive)
            avg_loss: Average loss amount (% or $, must be positive)
            kelly_fraction: Fraction of Kelly to use (default: 0.5 = half Kelly)
            max_position_pct: Max position as % of equity (default: 0.25 = 25%)
            baseline_volatility: Expected normal volatility (default: 0.02 = 2%)
        """
        # Validate inputs
        if not (0 <= win_rate <= 1):
            raise ValueError(f"win_rate must be between 0 and 1, got {win_rate}")
        if avg_win <= 0:
            raise ValueError(f"avg_win must be positive, got {avg_win}")
        if avg_loss <= 0:
            raise ValueError(f"avg_loss must be positive, got {avg_loss}")
        if not (0 < kelly_fraction <= 1):
            raise ValueError(f"kelly_fraction must be between 0 and 1, got {kelly_fraction}")
        if not (0 < max_position_pct <= 1):
            raise ValueError(f"max_position_pct must be between 0 and 1, got {max_position_pct}")

        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.baseline_volatility = baseline_volatility

        # Calculate derived values
        self.win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        self.base_kelly = optimal_kelly(win_rate, self.win_loss_ratio)

        logger.info(
            f"Kelly Sizer initialized: WR={win_rate:.1%}, "
            f"W/L={self.win_loss_ratio:.2f}, "
            f"Kelly={self.base_kelly:.1%}, "
            f"Fractional={kelly_fraction}x"
        )

    def calculate_position_size(
        self,
        account_equity: float,
        current_price: float,
        stop_loss: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> KellyPositionResult:
        """
        Calculate position size using Kelly Criterion.

        Args:
            account_equity: Current account equity/balance
            current_price: Current stock price
            stop_loss: Optional stop loss price (for risk calculation)
            volatility: Optional current volatility (ATR/price ratio)

        Returns:
            KellyPositionResult with full details
        """
        # Validate inputs
        if account_equity <= 0:
            raise ValueError(f"account_equity must be positive, got {account_equity}")
        if current_price <= 0:
            raise ValueError(f"current_price must be positive, got {current_price}")
        if stop_loss is not None and stop_loss <= 0:
            raise ValueError(f"stop_loss must be positive, got {stop_loss}")

        # Start with base Kelly
        kelly = self.base_kelly

        # Apply fractional Kelly
        frac_kelly = fractional_kelly(kelly, self.kelly_fraction)

        # Apply volatility adjustment if provided
        if volatility is not None:
            adj_kelly = volatility_adjusted_kelly(
                frac_kelly, volatility, self.baseline_volatility
            )
        else:
            adj_kelly = frac_kelly

        # Calculate position value
        position_value = account_equity * adj_kelly

        # Calculate shares (before capping)
        shares_uncapped = int(position_value / current_price)

        # Apply maximum position cap
        max_value = account_equity * self.max_position_pct
        max_shares = int(max_value / current_price)

        shares = min(shares_uncapped, max_shares)
        capped = (shares < shares_uncapped)
        shares = max(0, shares)

        # Calculate actual position value
        actual_position_value = shares * current_price

        # Calculate risk amount
        if stop_loss is not None:
            risk_per_share = abs(current_price - stop_loss)
            risk_amount = shares * risk_per_share
        else:
            risk_amount = actual_position_value * (self.avg_loss / self.avg_win if self.avg_win > 0 else 0.5)

        # Calculate volatility adjustment factor
        if volatility is not None:
            vol_adjustment = min(self.baseline_volatility / volatility if volatility > 0 else 1.0, 2.0)
        else:
            vol_adjustment = 1.0

        result = KellyPositionResult(
            shares=shares,
            position_value=actual_position_value,
            risk_amount=risk_amount,
            kelly_fraction=kelly,
            fractional_kelly=frac_kelly,
            adjusted_kelly=adj_kelly,
            win_rate=self.win_rate,
            win_loss_ratio=self.win_loss_ratio,
            volatility_adjustment=vol_adjustment,
            capped=capped,
            original_shares=shares_uncapped,
            account_equity=account_equity,
            current_price=current_price,
            stop_loss=stop_loss
        )

        logger.info(f"Position sized: {result}")
        return result

    def update_statistics(
        self,
        recent_trades: List[Dict],
        lookback: Optional[int] = None
    ) -> Tuple[float, float, float]:
        """
        Update win rate and average win/loss from recent trades.

        Args:
            recent_trades: List of trade dicts with 'pnl' or 'return' keys
            lookback: Optional number of recent trades to use (default: all)

        Returns:
            Tuple of (new_win_rate, new_avg_win, new_avg_loss)
        """
        if not recent_trades:
            raise ValueError("recent_trades cannot be empty")

        # Use only recent N trades if lookback specified
        trades_to_use = recent_trades[-lookback:] if lookback else recent_trades

        # Extract P&L values
        pnl_values = []
        for trade in trades_to_use:
            if 'pnl' in trade:
                pnl_values.append(trade['pnl'])
            elif 'return' in trade:
                pnl_values.append(trade['return'])
            else:
                raise ValueError("Trade dict must have 'pnl' or 'return' key")

        if not pnl_values:
            raise ValueError("No valid P&L values found in trades")

        # Separate winners and losers
        wins = [pnl for pnl in pnl_values if pnl > 0]
        losses = [abs(pnl) for pnl in pnl_values if pnl < 0]

        # Calculate new statistics
        total_trades = len(pnl_values)
        win_count = len(wins)

        new_win_rate = win_count / total_trades if total_trades > 0 else 0
        new_avg_win = sum(wins) / len(wins) if wins else 0
        new_avg_loss = sum(losses) / len(losses) if losses else 0

        # Update instance variables
        old_stats = (self.win_rate, self.avg_win, self.avg_loss)

        self.win_rate = new_win_rate
        self.avg_win = new_avg_win if new_avg_win > 0 else self.avg_win
        self.avg_loss = new_avg_loss if new_avg_loss > 0 else self.avg_loss

        # Recalculate derived values
        self.win_loss_ratio = self.avg_win / self.avg_loss if self.avg_loss > 0 else 0
        self.base_kelly = optimal_kelly(self.win_rate, self.win_loss_ratio)

        logger.info(
            f"Statistics updated from {len(trades_to_use)} trades: "
            f"WR: {old_stats[0]:.1%} -> {new_win_rate:.1%}, "
            f"Avg W: {old_stats[1]:.3f} -> {new_avg_win:.3f}, "
            f"Avg L: {old_stats[2]:.3f} -> {new_avg_loss:.3f}, "
            f"Kelly: {self.base_kelly:.1%}"
        )

        return new_win_rate, new_avg_win, new_avg_loss

    def get_kelly_info(self) -> Dict:
        """Get current Kelly sizer configuration and statistics."""
        return {
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'win_loss_ratio': self.win_loss_ratio,
            'base_kelly': self.base_kelly,
            'kelly_fraction': self.kelly_fraction,
            'max_position_pct': self.max_position_pct,
            'baseline_volatility': self.baseline_volatility,
            'effective_kelly': self.base_kelly * self.kelly_fraction,
        }


def quick_kelly_position(
    account_equity: float,
    current_price: float,
    win_rate: float = 0.6696,
    profit_factor: float = 1.53,
    kelly_fraction: float = 0.5,
    max_position_pct: float = 0.25
) -> int:
    """
    Quick Kelly position calculation using profit factor.

    Convenience function for rapid position sizing.

    Args:
        account_equity: Account balance
        current_price: Stock price
        win_rate: Win rate (default: 0.6696 from validated stats)
        profit_factor: Profit factor (default: 1.53 from validated stats)
        kelly_fraction: Fraction of Kelly (default: 0.5)
        max_position_pct: Max position % (default: 0.25)

    Returns:
        Number of shares to trade
    """
    loss_rate = 1 - win_rate

    if loss_rate == 0:
        win_loss_ratio = float('inf')
    else:
        avg_win = 1.0
        avg_loss = (win_rate) / (profit_factor * loss_rate) if profit_factor * loss_rate > 0 else 0.01
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

    sizer = KellyPositionSizer(
        win_rate=win_rate,
        avg_win=1.0,
        avg_loss=1.0 / win_loss_ratio if win_loss_ratio > 0 else 1.0,
        kelly_fraction=kelly_fraction,
        max_position_pct=max_position_pct
    )

    result = sizer.calculate_position_size(
        account_equity=account_equity,
        current_price=current_price
    )

    return result.shares
