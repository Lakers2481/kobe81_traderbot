"""
Intelligent Exit Manager - Adaptive Exit Strategies for Kobe

Mission 1: The 'Why'
-------------------
The current exit logic is mostly static (e.g., "exit after 7 days"). This is a major
weakness as it completely ignores the price action of a trade after entry. A winning
trade might reverse and give back all its profit before the time stop is hit, or a
losing trade might be stopped out unnecessarily on minor volatility.

This intelligent, adaptive exit system will dramatically improve risk management
and profit capture through:
1. ATR-Based Trailing Stops - Stops that adapt to volatility
2. Price Structure Stops - Context-aware stop placement
3. Partial Profit Taking - Intelligent profit capture

Author: Kobe Trading System
Date: 2026-01-07
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ExitAction(Enum):
    """Types of exit actions the manager can recommend."""
    HOLD = "hold"                    # No action needed
    TRAIL_STOP = "trail_stop"        # Update stop loss (trailing)
    STRUCTURE_STOP = "structure_stop" # Update stop to price structure
    SELL_PARTIAL = "sell_partial"    # Sell part of position
    SELL_FULL = "sell_full"          # Sell entire position
    STOP_HIT = "stop_hit"            # Stop loss triggered


@dataclass
class ExitStrategyConfig:
    """Configuration for exit strategies."""
    # ATR Trailing Stop
    atr_period: int = 14
    atr_multiplier: float = 2.0       # Stop distance in ATR units
    atr_trail_enabled: bool = True

    # Partial Profit Taking
    profit_target_atr: float = 3.0    # Take partial at 3x ATR
    partial_exit_pct: float = 0.5     # Exit 50% of position
    partial_profit_enabled: bool = True

    # Price Structure Stop
    swing_lookback: int = 10          # Bars to look back for swing points
    structure_buffer_pct: float = 0.005  # Buffer below swing low (0.5%)
    structure_stop_enabled: bool = True

    # Time-based fallback
    max_hold_days: int = 7            # Original time stop (fallback)
    time_stop_enabled: bool = True


@dataclass
class ExitRecommendation:
    """Recommendation from the exit manager."""
    action: ExitAction
    new_stop_price: Optional[float] = None
    exit_qty_pct: Optional[float] = None  # Percentage of position to exit
    reason: str = ""
    atr_value: Optional[float] = None
    swing_low: Optional[float] = None


@dataclass
class PositionContext:
    """Context about an open position."""
    symbol: str
    entry_price: float
    entry_time: datetime
    current_price: float
    current_stop: float
    qty: int
    side: str = "long"  # "long" or "short"
    highest_price: float = 0.0  # Track high water mark for trailing
    partial_taken: bool = False  # Track if partial profit was taken


class IntelligentExitManager:
    """
    Intelligent Exit Manager for adaptive position management.

    This class analyzes price action and volatility to recommend:
    1. Trailing stop updates based on ATR
    2. Stop placement based on price structure (swing lows)
    3. Partial profit taking on fast moves
    """

    def __init__(self, config: Optional[ExitStrategyConfig] = None):
        """Initialize with configuration."""
        self.config = config or ExitStrategyConfig()
        logger.info(f"IntelligentExitManager initialized: ATR={self.config.atr_multiplier}x, "
                   f"profit_target={self.config.profit_target_atr}x ATR")

    def calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range."""
        if len(df) < self.config.atr_period + 1:
            return 0.0

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr_list = []
        for i in range(1, len(df)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)

        # Simple moving average of TR
        if len(tr_list) >= self.config.atr_period:
            atr = np.mean(tr_list[-self.config.atr_period:])
            return float(atr)
        return 0.0

    def find_swing_low(self, df: pd.DataFrame, lookback: int = None) -> Optional[float]:
        """
        Find the most recent significant swing low.

        A swing low is a bar where the low is lower than the lows
        of the bars immediately before and after it.
        """
        lookback = lookback or self.config.swing_lookback
        if len(df) < lookback:
            return None

        recent = df.tail(lookback)
        lows = recent['low'].values

        # Find swing lows (local minima)
        swing_lows = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                swing_lows.append(lows[i])

        if swing_lows:
            # Return the most recent swing low
            return float(min(swing_lows[-3:]))  # Consider last 3 swing lows
        return None

    def find_swing_high(self, df: pd.DataFrame, lookback: int = None) -> Optional[float]:
        """Find the most recent significant swing high (for short positions)."""
        lookback = lookback or self.config.swing_lookback
        if len(df) < lookback:
            return None

        recent = df.tail(lookback)
        highs = recent['high'].values

        swing_highs = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                swing_highs.append(highs[i])

        if swing_highs:
            return float(max(swing_highs[-3:]))
        return None

    def update_trade_parameters(
        self,
        position: PositionContext,
        market_data: pd.DataFrame
    ) -> ExitRecommendation:
        """
        Analyze position and market data to recommend exit actions.

        This is the core method that integrates all exit strategies.

        Args:
            position: Current position context
            market_data: Recent OHLCV data for the symbol

        Returns:
            ExitRecommendation with action and parameters
        """
        # Calculate ATR
        atr = self.calculate_atr(market_data)
        if atr == 0:
            logger.warning(f"{position.symbol}: Could not calculate ATR")
            return ExitRecommendation(action=ExitAction.HOLD, reason="Insufficient data for ATR")

        recommendations = []

        # For long positions
        if position.side == "long":
            recommendations.extend(
                self._analyze_long_position(position, market_data, atr)
            )
        else:  # Short positions
            recommendations.extend(
                self._analyze_short_position(position, market_data, atr)
            )

        # Check time stop as fallback
        if self.config.time_stop_enabled:
            days_held = (datetime.now() - position.entry_time).days
            if days_held >= self.config.max_hold_days:
                recommendations.append(ExitRecommendation(
                    action=ExitAction.SELL_FULL,
                    reason=f"Time stop: held {days_held} days >= {self.config.max_hold_days}",
                    atr_value=atr
                ))

        # Priority: STOP_HIT > SELL_FULL > SELL_PARTIAL > TRAIL_STOP > HOLD
        priority = {
            ExitAction.STOP_HIT: 5,
            ExitAction.SELL_FULL: 4,
            ExitAction.SELL_PARTIAL: 3,
            ExitAction.TRAIL_STOP: 2,
            ExitAction.STRUCTURE_STOP: 2,
            ExitAction.HOLD: 1
        }

        if recommendations:
            best = max(recommendations, key=lambda r: priority.get(r.action, 0))
            return best

        return ExitRecommendation(action=ExitAction.HOLD, reason="No action needed", atr_value=atr)

    def _analyze_long_position(
        self,
        position: PositionContext,
        market_data: pd.DataFrame,
        atr: float
    ) -> List[ExitRecommendation]:
        """Analyze exit conditions for a long position."""
        recommendations = []
        current = position.current_price
        entry = position.entry_price
        stop = position.current_stop

        # 1. Check if stop was hit
        if current <= stop:
            recommendations.append(ExitRecommendation(
                action=ExitAction.STOP_HIT,
                reason=f"Stop hit: current ${current:.2f} <= stop ${stop:.2f}",
                atr_value=atr
            ))
            return recommendations  # Immediate exit

        # 2. ATR-Based Trailing Stop
        if self.config.atr_trail_enabled:
            # New trailing stop = current price - (ATR * multiplier)
            atr_stop = current - (atr * self.config.atr_multiplier)

            # Only trail UP, never down
            if atr_stop > stop:
                recommendations.append(ExitRecommendation(
                    action=ExitAction.TRAIL_STOP,
                    new_stop_price=round(atr_stop, 2),
                    reason=f"ATR trail: ${stop:.2f} -> ${atr_stop:.2f} (current ${current:.2f})",
                    atr_value=atr
                ))

        # 3. Price Structure Stop
        if self.config.structure_stop_enabled:
            swing_low = self.find_swing_low(market_data)
            if swing_low:
                # Place stop just below swing low
                structure_stop = swing_low * (1 - self.config.structure_buffer_pct)

                # Only use if better than current stop and not too tight
                if structure_stop > stop and structure_stop < current * 0.97:
                    recommendations.append(ExitRecommendation(
                        action=ExitAction.STRUCTURE_STOP,
                        new_stop_price=round(structure_stop, 2),
                        reason=f"Structure stop: below swing low ${swing_low:.2f}",
                        atr_value=atr,
                        swing_low=swing_low
                    ))

        # 4. Partial Profit Taking
        if self.config.partial_profit_enabled and not position.partial_taken:
            profit_target = entry + (atr * self.config.profit_target_atr)
            if current >= profit_target:
                recommendations.append(ExitRecommendation(
                    action=ExitAction.SELL_PARTIAL,
                    exit_qty_pct=self.config.partial_exit_pct,
                    reason=f"Profit target hit: ${current:.2f} >= ${profit_target:.2f} ({self.config.profit_target_atr}x ATR)",
                    atr_value=atr
                ))

        return recommendations

    def _analyze_short_position(
        self,
        position: PositionContext,
        market_data: pd.DataFrame,
        atr: float
    ) -> List[ExitRecommendation]:
        """Analyze exit conditions for a short position."""
        recommendations = []
        current = position.current_price
        entry = position.entry_price
        stop = position.current_stop

        # 1. Check if stop was hit (for shorts, price going UP hits stop)
        if current >= stop:
            recommendations.append(ExitRecommendation(
                action=ExitAction.STOP_HIT,
                reason=f"Stop hit: current ${current:.2f} >= stop ${stop:.2f}",
                atr_value=atr
            ))
            return recommendations

        # 2. ATR-Based Trailing Stop (for shorts, trail DOWN)
        if self.config.atr_trail_enabled:
            atr_stop = current + (atr * self.config.atr_multiplier)

            # Only trail DOWN for shorts
            if atr_stop < stop:
                recommendations.append(ExitRecommendation(
                    action=ExitAction.TRAIL_STOP,
                    new_stop_price=round(atr_stop, 2),
                    reason=f"ATR trail: ${stop:.2f} -> ${atr_stop:.2f}",
                    atr_value=atr
                ))

        # 3. Price Structure Stop (for shorts, use swing high)
        if self.config.structure_stop_enabled:
            swing_high = self.find_swing_high(market_data)
            if swing_high:
                structure_stop = swing_high * (1 + self.config.structure_buffer_pct)

                if structure_stop < stop and structure_stop > current * 1.03:
                    recommendations.append(ExitRecommendation(
                        action=ExitAction.STRUCTURE_STOP,
                        new_stop_price=round(structure_stop, 2),
                        reason=f"Structure stop: above swing high ${swing_high:.2f}",
                        atr_value=atr,
                        swing_low=swing_high
                    ))

        # 4. Partial Profit Taking (for shorts, price going down is profit)
        if self.config.partial_profit_enabled and not position.partial_taken:
            profit_target = entry - (atr * self.config.profit_target_atr)
            if current <= profit_target:
                recommendations.append(ExitRecommendation(
                    action=ExitAction.SELL_PARTIAL,
                    exit_qty_pct=self.config.partial_exit_pct,
                    reason=f"Profit target hit: ${current:.2f} <= ${profit_target:.2f}",
                    atr_value=atr
                ))

        return recommendations

    def get_status(self) -> Dict[str, Any]:
        """Return current configuration status."""
        return {
            "atr_period": self.config.atr_period,
            "atr_multiplier": self.config.atr_multiplier,
            "atr_trail_enabled": self.config.atr_trail_enabled,
            "profit_target_atr": self.config.profit_target_atr,
            "partial_exit_pct": self.config.partial_exit_pct,
            "partial_profit_enabled": self.config.partial_profit_enabled,
            "swing_lookback": self.config.swing_lookback,
            "structure_stop_enabled": self.config.structure_stop_enabled,
            "max_hold_days": self.config.max_hold_days,
            "time_stop_enabled": self.config.time_stop_enabled,
        }


# Singleton instance
_exit_manager: Optional[IntelligentExitManager] = None


def get_exit_manager(config: Optional[ExitStrategyConfig] = None) -> IntelligentExitManager:
    """Get or create the singleton exit manager."""
    global _exit_manager
    if _exit_manager is None:
        _exit_manager = IntelligentExitManager(config)
    return _exit_manager


# Example usage
if __name__ == "__main__":
    # Demo with sample data
    import random

    # Create sample market data
    dates = pd.date_range(start="2026-01-01", periods=30, freq="D")
    base_price = 100.0
    data = {
        "timestamp": dates,
        "open": [base_price + random.uniform(-2, 2) for _ in range(30)],
        "high": [base_price + random.uniform(0, 3) for _ in range(30)],
        "low": [base_price + random.uniform(-3, 0) for _ in range(30)],
        "close": [base_price + random.uniform(-1, 1) for _ in range(30)],
        "volume": [random.randint(100000, 500000) for _ in range(30)],
    }
    df = pd.DataFrame(data)

    # Fix OHLC consistency
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    # Create position
    position = PositionContext(
        symbol="TEST",
        entry_price=98.0,
        entry_time=datetime(2026, 1, 5),
        current_price=102.0,
        current_stop=96.0,
        qty=100,
        side="long"
    )

    # Get recommendation
    manager = get_exit_manager()
    rec = manager.update_trade_parameters(position, df)

    print(f"Recommendation: {rec.action.value}")
    print(f"Reason: {rec.reason}")
    if rec.new_stop_price:
        print(f"New Stop: ${rec.new_stop_price:.2f}")
    if rec.atr_value:
        print(f"ATR: ${rec.atr_value:.2f}")
