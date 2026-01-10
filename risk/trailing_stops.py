"""
Trailing Stop Manager
=====================

Dynamically manages stop losses to lock in profits.

Features:
- Move stop to breakeven after 1R profit
- Trail stop at 1R behind price after 2R profit
- Time-based tightening (reduce risk as trade ages)
- Volatility adjustment (wider stops in high VIX)

Usage:
    from risk.trailing_stops import TrailingStopManager

    tsm = TrailingStopManager()

    # On each bar update:
    new_stop = tsm.update_stop(position, current_price, current_bar)
    if new_stop != position['stop_loss']:
        update_broker_stop(position['symbol'], new_stop)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class StopState(Enum):
    """Current state of the trailing stop."""
    INITIAL = "initial"           # Original stop, no adjustment
    BREAKEVEN = "breakeven"       # Moved to breakeven (1R profit)
    TRAILING_1R = "trailing_1r"   # Trailing at 1R behind (2R profit)
    TRAILING_2R = "trailing_2r"   # Trailing at 2R behind (3R+ profit)
    TIGHT = "tight"               # Tightened due to time decay


@dataclass
class StopUpdate:
    """Result of stop update calculation."""
    symbol: str
    old_stop: float
    new_stop: float
    state: StopState
    reason: str
    r_multiple: float  # Current profit in R multiples
    should_update: bool

    def __str__(self):
        if self.should_update:
            return f"{self.symbol}: Stop {self.old_stop:.2f} -> {self.new_stop:.2f} ({self.state.value}, {self.r_multiple:.1f}R)"
        return f"{self.symbol}: No change ({self.state.value}, {self.r_multiple:.1f}R)"


class TrailingStopManager:
    """
    Manages trailing stops for open positions.

    Logic:
    - 0R to 1R: Keep original stop
    - 1R to 2R: Move stop to breakeven (entry price)
    - 2R to 3R: Trail at 1R behind current price
    - 3R+: Trail at 2R behind current price
    - After N bars: Tighten stop regardless
    """

    def __init__(
        self,
        breakeven_threshold: float = 1.0,    # Move to breakeven at 1R
        trail_1r_threshold: float = 2.0,     # Start trailing at 2R
        trail_2r_threshold: float = 3.0,     # Tighter trail at 3R
        time_decay_bars: int = 10,           # Tighten after N bars
        time_decay_factor: float = 0.5,      # Reduce stop distance by this factor
        buffer_pct: float = 0.002,           # 0.2% buffer below breakeven
    ):
        self.breakeven_threshold = breakeven_threshold
        self.trail_1r_threshold = trail_1r_threshold
        self.trail_2r_threshold = trail_2r_threshold
        self.time_decay_bars = time_decay_bars
        self.time_decay_factor = time_decay_factor
        self.buffer_pct = buffer_pct

        # Track position states
        self._position_states: Dict[str, StopState] = {}

        logger.info("TrailingStopManager initialized")

    def calculate_r_multiple(
        self,
        entry_price: float,
        current_price: float,
        initial_stop: float,
        side: str = 'long'
    ) -> float:
        """
        Calculate current profit in R multiples.

        R = risk per share = |entry - initial_stop|
        Profit in R = (current - entry) / R for longs
        """
        if side == 'long':
            risk_per_share = entry_price - initial_stop
            if risk_per_share <= 0:
                return 0.0
            profit_per_share = current_price - entry_price
            return profit_per_share / risk_per_share
        else:  # short
            risk_per_share = initial_stop - entry_price
            if risk_per_share <= 0:
                return 0.0
            profit_per_share = entry_price - current_price
            return profit_per_share / risk_per_share

    def update_stop(
        self,
        position: Dict,
        current_price: float,
        bars_held: int = 0,
        vix_level: Optional[float] = None,
    ) -> StopUpdate:
        """
        Calculate new stop loss for a position.

        Args:
            position: Dict with keys: symbol, entry_price, stop_loss, initial_stop, side
            current_price: Current market price
            bars_held: Number of bars since entry
            vix_level: Optional VIX for volatility adjustment

        Returns:
            StopUpdate with new stop and reasoning
        """
        symbol = position.get('symbol', 'UNKNOWN')
        entry_price = position.get('entry_price', 0)
        current_stop = position.get('stop_loss', 0)
        initial_stop = position.get('initial_stop', current_stop)
        side = position.get('side', 'long')

        if entry_price <= 0 or initial_stop <= 0:
            return StopUpdate(
                symbol=symbol,
                old_stop=current_stop,
                new_stop=current_stop,
                state=StopState.INITIAL,
                reason="Invalid position data",
                r_multiple=0,
                should_update=False
            )

        # Calculate current R multiple
        r_multiple = self.calculate_r_multiple(entry_price, current_price, initial_stop, side)

        # Calculate R (risk per share)
        r_value = abs(entry_price - initial_stop)

        # Get current state
        current_state = self._position_states.get(symbol, StopState.INITIAL)
        new_stop = current_stop
        new_state = current_state
        reason = "No change"

        if side == 'long':
            # === LONG POSITION LOGIC ===

            if r_multiple >= self.trail_2r_threshold:
                # 3R+ profit: Trail at 2R behind price
                trail_stop = current_price - (2 * r_value)
                if trail_stop > current_stop:
                    new_stop = trail_stop
                    new_state = StopState.TRAILING_2R
                    reason = f"Trailing 2R at {r_multiple:.1f}R profit"

            elif r_multiple >= self.trail_1r_threshold:
                # 2R-3R profit: Trail at 1R behind price
                trail_stop = current_price - r_value
                if trail_stop > current_stop:
                    new_stop = trail_stop
                    new_state = StopState.TRAILING_1R
                    reason = f"Trailing 1R at {r_multiple:.1f}R profit"

            elif r_multiple >= self.breakeven_threshold:
                # 1R-2R profit: Move to breakeven
                breakeven_stop = entry_price * (1 - self.buffer_pct)
                if breakeven_stop > current_stop:
                    new_stop = breakeven_stop
                    new_state = StopState.BREAKEVEN
                    reason = f"Breakeven at {r_multiple:.1f}R profit"

        else:
            # === SHORT POSITION LOGIC ===

            if r_multiple >= self.trail_2r_threshold:
                trail_stop = current_price + (2 * r_value)
                if trail_stop < current_stop:
                    new_stop = trail_stop
                    new_state = StopState.TRAILING_2R
                    reason = f"Trailing 2R at {r_multiple:.1f}R profit"

            elif r_multiple >= self.trail_1r_threshold:
                trail_stop = current_price + r_value
                if trail_stop < current_stop:
                    new_stop = trail_stop
                    new_state = StopState.TRAILING_1R
                    reason = f"Trailing 1R at {r_multiple:.1f}R profit"

            elif r_multiple >= self.breakeven_threshold:
                breakeven_stop = entry_price * (1 + self.buffer_pct)
                if breakeven_stop < current_stop:
                    new_stop = breakeven_stop
                    new_state = StopState.BREAKEVEN
                    reason = f"Breakeven at {r_multiple:.1f}R profit"

        # === TIME DECAY: Tighten after N bars ===
        if bars_held >= self.time_decay_bars and new_state == StopState.INITIAL:
            # Tighten stop if we've held too long without profit
            if side == 'long':
                time_stop = entry_price - (r_value * self.time_decay_factor)
                if time_stop > current_stop:
                    new_stop = time_stop
                    new_state = StopState.TIGHT
                    reason = f"Time decay after {bars_held} bars"
            else:
                time_stop = entry_price + (r_value * self.time_decay_factor)
                if time_stop < current_stop:
                    new_stop = time_stop
                    new_state = StopState.TIGHT
                    reason = f"Time decay after {bars_held} bars"

        # === VIX ADJUSTMENT: Widen in high volatility ===
        if vix_level is not None and vix_level > 30:
            # In high VIX, be more conservative with trailing
            if new_state in [StopState.TRAILING_1R, StopState.TRAILING_2R]:
                vix_buffer = (vix_level - 30) / 100  # e.g., VIX 40 = 10% wider
                if side == 'long':
                    new_stop = new_stop * (1 - vix_buffer)
                else:
                    new_stop = new_stop * (1 + vix_buffer)
                reason += f" (VIX adjusted: {vix_level:.0f})"

        # Update state tracking
        self._position_states[symbol] = new_state

        should_update = abs(new_stop - current_stop) > 0.01

        return StopUpdate(
            symbol=symbol,
            old_stop=current_stop,
            new_stop=round(new_stop, 2),
            state=new_state,
            reason=reason,
            r_multiple=r_multiple,
            should_update=should_update
        )

    def update_all_stops(
        self,
        positions: List[Dict],
        current_prices: Dict[str, float],
        bars_held: Optional[Dict[str, int]] = None,
        vix_level: Optional[float] = None,
    ) -> List[StopUpdate]:
        """
        Update stops for all positions.

        Args:
            positions: List of position dicts
            current_prices: Dict of symbol -> current price
            bars_held: Optional dict of symbol -> bars since entry
            vix_level: Optional current VIX level

        Returns:
            List of StopUpdate objects
        """
        updates = []

        for position in positions:
            symbol = position.get('symbol')
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            bars = bars_held.get(symbol, 0) if bars_held else 0

            update = self.update_stop(position, current_price, bars, vix_level)
            updates.append(update)

            if update.should_update:
                logger.info(str(update))

        return updates

    def reset_position(self, symbol: str):
        """Reset tracking for a closed position."""
        if symbol in self._position_states:
            del self._position_states[symbol]

    def get_position_state(self, symbol: str) -> StopState:
        """Get current stop state for a position."""
        return self._position_states.get(symbol, StopState.INITIAL)

    def get_stats(self) -> Dict:
        """Get statistics about managed positions."""
        states = list(self._position_states.values())
        return {
            'positions_tracked': len(states),
            'at_breakeven': sum(1 for s in states if s == StopState.BREAKEVEN),
            'trailing': sum(1 for s in states if s in [StopState.TRAILING_1R, StopState.TRAILING_2R]),
            'tightened': sum(1 for s in states if s == StopState.TIGHT),
        }


# Singleton instance
_trailing_stop_manager: Optional[TrailingStopManager] = None


def get_trailing_stop_manager() -> TrailingStopManager:
    """Get or create singleton TrailingStopManager."""
    global _trailing_stop_manager
    if _trailing_stop_manager is None:
        _trailing_stop_manager = TrailingStopManager()
    return _trailing_stop_manager
