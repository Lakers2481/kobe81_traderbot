"""
Regime-Adaptive Slippage Model.

Adjusts slippage estimates based on market regime and VIX level.
More realistic than fixed slippage for varying market conditions.

Key Features:
- VIX-based slippage scaling
- Regime detection integration
- Intraday volatility adjustment
- Volume-weighted adjustments
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum, auto
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtest.slippage import (
    SlippageModel,
    SlippageType,
    SlippageResult,
    FixedBpsSlippage,
)

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    LOW_VOL = auto()       # VIX < 15, calm markets
    NORMAL = auto()        # VIX 15-20, typical conditions
    ELEVATED = auto()      # VIX 20-30, heightened uncertainty
    HIGH_VOL = auto()      # VIX 30-50, fear/stress
    CRISIS = auto()        # VIX > 50, extreme conditions


class TimeOfDay(Enum):
    """Time of day for intraday adjustments."""
    PRE_MARKET = auto()    # Before 9:30 AM
    OPEN_AUCTION = auto()  # 9:30-10:00 AM
    MORNING = auto()       # 10:00 AM - 12:00 PM
    MIDDAY = auto()        # 12:00 PM - 2:00 PM
    AFTERNOON = auto()     # 2:00 PM - 3:30 PM
    CLOSE_AUCTION = auto() # 3:30-4:00 PM
    AFTER_HOURS = auto()   # After 4:00 PM


@dataclass
class RegimeSlippageResult(SlippageResult):
    """Extended slippage result with regime information."""
    regime: MarketRegime = MarketRegime.NORMAL
    vix_level: Optional[float] = None
    regime_multiplier: float = 1.0
    time_of_day: Optional[TimeOfDay] = None


class RegimeAdaptiveSlippage(SlippageModel):
    """
    Slippage model that adapts to market regime and VIX level.

    In high-volatility environments, slippage typically increases due to:
    - Wider bid-ask spreads
    - Lower market depth
    - Faster price movements
    - Reduced liquidity

    VIX Regime Multipliers (empirically calibrated):
    - VIX < 15:  0.6x base slippage (calm markets, tight spreads)
    - VIX 15-20: 1.0x base slippage (normal conditions)
    - VIX 20-30: 1.5x base slippage (elevated uncertainty)
    - VIX 30-50: 2.5x base slippage (fear/stress)
    - VIX > 50:  4.0x base slippage (crisis mode)
    """

    # VIX regime thresholds and multipliers
    VIX_REGIMES = {
        MarketRegime.LOW_VOL: (0, 15, 0.6),
        MarketRegime.NORMAL: (15, 20, 1.0),
        MarketRegime.ELEVATED: (20, 30, 1.5),
        MarketRegime.HIGH_VOL: (30, 50, 2.5),
        MarketRegime.CRISIS: (50, float("inf"), 4.0),
    }

    # Time of day multipliers (captures intraday patterns)
    TIME_MULTIPLIERS = {
        TimeOfDay.PRE_MARKET: 2.0,     # Thin liquidity
        TimeOfDay.OPEN_AUCTION: 1.5,   # Opening volatility
        TimeOfDay.MORNING: 1.0,        # Normal
        TimeOfDay.MIDDAY: 0.9,         # Quieter period
        TimeOfDay.AFTERNOON: 1.0,      # Normal
        TimeOfDay.CLOSE_AUCTION: 1.3,  # Closing volatility
        TimeOfDay.AFTER_HOURS: 2.5,    # Very thin liquidity
    }

    def __init__(
        self,
        base_bps: float = 5.0,
        use_vix_adjustment: bool = True,
        use_time_adjustment: bool = False,  # Disabled by default for EOD backtests
        min_bps: float = 1.0,
        max_bps: float = 100.0,
        fallback_vix: float = 18.0,  # Assume normal if VIX unknown
    ):
        """
        Initialize regime-adaptive slippage model.

        Args:
            base_bps: Base slippage in basis points (at VIX ~18)
            use_vix_adjustment: Whether to scale by VIX
            use_time_adjustment: Whether to scale by time of day
            min_bps: Minimum slippage floor
            max_bps: Maximum slippage cap
            fallback_vix: VIX to assume if not provided
        """
        self.base_bps = base_bps
        self.use_vix_adjustment = use_vix_adjustment
        self.use_time_adjustment = use_time_adjustment
        self.min_bps = min_bps
        self.max_bps = max_bps
        self.fallback_vix = fallback_vix

        # Keep a simple model as fallback
        self._fallback_model = FixedBpsSlippage(bps=base_bps)

    @property
    def model_type(self) -> SlippageType:
        return SlippageType.FIXED_BPS  # Closest base type

    def get_regime(self, vix: float) -> MarketRegime:
        """Classify VIX level into regime."""
        for regime, (low, high, _) in self.VIX_REGIMES.items():
            if low <= vix < high:
                return regime
        return MarketRegime.NORMAL

    def get_vix_multiplier(self, vix: float) -> float:
        """Get slippage multiplier for VIX level."""
        regime = self.get_regime(vix)
        return self.VIX_REGIMES[regime][2]

    def get_time_of_day(self, timestamp: datetime) -> TimeOfDay:
        """Classify timestamp into time of day bucket."""
        t = timestamp.time()

        if t < time(9, 30):
            return TimeOfDay.PRE_MARKET
        elif t < time(10, 0):
            return TimeOfDay.OPEN_AUCTION
        elif t < time(12, 0):
            return TimeOfDay.MORNING
        elif t < time(14, 0):
            return TimeOfDay.MIDDAY
        elif t < time(15, 30):
            return TimeOfDay.AFTERNOON
        elif t < time(16, 0):
            return TimeOfDay.CLOSE_AUCTION
        else:
            return TimeOfDay.AFTER_HOURS

    def get_time_multiplier(self, timestamp: datetime) -> float:
        """Get slippage multiplier for time of day."""
        tod = self.get_time_of_day(timestamp)
        return self.TIME_MULTIPLIERS[tod]

    def calculate(
        self,
        price: float,
        side: str,
        qty: int,
        vix: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        **kwargs,
    ) -> RegimeSlippageResult:
        """
        Calculate regime-adaptive slippage.

        Args:
            price: Base price
            side: "buy" or "sell"
            qty: Order quantity
            vix: VIX level (uses fallback if not provided)
            timestamp: Trade timestamp (for time-of-day adjustment)
            **kwargs: Additional parameters

        Returns:
            RegimeSlippageResult with full details
        """
        # Get VIX
        effective_vix = vix if vix is not None else self.fallback_vix
        regime = self.get_regime(effective_vix)

        # Calculate multipliers
        total_multiplier = 1.0

        if self.use_vix_adjustment:
            vix_mult = self.get_vix_multiplier(effective_vix)
            total_multiplier *= vix_mult

        time_of_day = None
        if self.use_time_adjustment and timestamp is not None:
            time_of_day = self.get_time_of_day(timestamp)
            time_mult = self.TIME_MULTIPLIERS[time_of_day]
            total_multiplier *= time_mult

        # Calculate slippage
        adjusted_bps = self.base_bps * total_multiplier

        # Apply min/max bounds
        adjusted_bps = max(self.min_bps, min(self.max_bps, adjusted_bps))

        # Convert to price adjustment
        slippage_pct = adjusted_bps / 10000
        slippage_amount = price * slippage_pct

        # Adjust price
        adjusted_price = self.adjust_price(price, slippage_amount, side)

        return RegimeSlippageResult(
            model_type=self.model_type,
            base_price=price,
            slippage_amount=slippage_amount,
            adjusted_price=adjusted_price,
            slippage_bps=adjusted_bps,
            metadata={
                "model": "regime_adaptive",
                "base_bps": self.base_bps,
                "total_multiplier": total_multiplier,
                "vix": effective_vix,
                "vix_fallback_used": vix is None,
            },
            regime=regime,
            vix_level=effective_vix,
            regime_multiplier=total_multiplier,
            time_of_day=time_of_day,
        )


class VIXIntegratedSlippage(RegimeAdaptiveSlippage):
    """
    Slippage model that integrates with VIX data source.

    Automatically fetches VIX from data provider if available.
    """

    def __init__(
        self,
        base_bps: float = 5.0,
        vix_data: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        """
        Initialize VIX-integrated slippage model.

        Args:
            base_bps: Base slippage in basis points
            vix_data: DataFrame with VIX data (index=date, columns include 'close' or 'VIX')
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(base_bps=base_bps, **kwargs)
        self.vix_data = vix_data
        self._vix_cache: Dict[str, float] = {}

    def set_vix_data(self, vix_data: pd.DataFrame) -> None:
        """Set or update VIX data."""
        self.vix_data = vix_data
        self._vix_cache.clear()

    def get_vix_for_date(self, date: datetime) -> Optional[float]:
        """Lookup VIX for a specific date."""
        if self.vix_data is None:
            return None

        # Check cache
        date_key = date.strftime("%Y-%m-%d")
        if date_key in self._vix_cache:
            return self._vix_cache[date_key]

        # Look up in data
        try:
            # Handle different index types
            if isinstance(self.vix_data.index, pd.DatetimeIndex):
                # Get closest date on or before
                mask = self.vix_data.index.date <= date.date()
                if mask.any():
                    vix_row = self.vix_data.loc[mask].iloc[-1]
                    vix = vix_row.get("close", vix_row.get("VIX", vix_row.get("Close", None)))
                    if vix is not None:
                        self._vix_cache[date_key] = float(vix)
                        return float(vix)
        except Exception as e:
            logger.debug(f"VIX lookup failed for {date}: {e}")

        return None

    def calculate(
        self,
        price: float,
        side: str,
        qty: int,
        vix: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        trade_date: Optional[datetime] = None,
        **kwargs,
    ) -> RegimeSlippageResult:
        """
        Calculate slippage with automatic VIX lookup.

        Args:
            price: Base price
            side: "buy" or "sell"
            qty: Order quantity
            vix: VIX level (overrides lookup if provided)
            timestamp: Trade timestamp
            trade_date: Trade date (for VIX lookup if timestamp not provided)
            **kwargs: Additional parameters

        Returns:
            RegimeSlippageResult
        """
        # Auto-lookup VIX if not provided
        if vix is None:
            lookup_date = timestamp or trade_date
            if lookup_date is not None:
                vix = self.get_vix_for_date(lookup_date)

        return super().calculate(
            price=price,
            side=side,
            qty=qty,
            vix=vix,
            timestamp=timestamp,
            **kwargs,
        )


class HybridSlippage(SlippageModel):
    """
    Hybrid slippage model combining multiple approaches.

    Takes the maximum of:
    - Fixed minimum slippage
    - Regime-adaptive slippage
    - Volume-based impact

    Ensures conservative (higher) slippage estimates.
    """

    def __init__(
        self,
        min_bps: float = 2.0,
        base_bps: float = 5.0,
        volume_impact_coef: float = 0.1,
        use_vix: bool = True,
        fallback_vix: float = 18.0,
    ):
        """
        Initialize hybrid slippage model.

        Args:
            min_bps: Absolute minimum slippage
            base_bps: Base slippage for regime model
            volume_impact_coef: Coefficient for volume impact
            use_vix: Whether to use VIX adjustment
            fallback_vix: Fallback VIX if not provided
        """
        self.min_bps = min_bps
        self.regime_model = RegimeAdaptiveSlippage(
            base_bps=base_bps,
            use_vix_adjustment=use_vix,
            fallback_vix=fallback_vix,
        )
        self.volume_impact_coef = volume_impact_coef

    @property
    def model_type(self) -> SlippageType:
        return SlippageType.VOLUME_IMPACT

    def calculate(
        self,
        price: float,
        side: str,
        qty: int,
        vix: Optional[float] = None,
        daily_volume: Optional[float] = None,
        volatility: Optional[float] = None,
        **kwargs,
    ) -> SlippageResult:
        """
        Calculate hybrid slippage as max of multiple models.

        Args:
            price: Base price
            side: "buy" or "sell"
            qty: Order quantity
            vix: VIX level
            daily_volume: Average daily volume
            volatility: Stock volatility (annualized)
            **kwargs: Additional parameters

        Returns:
            SlippageResult with conservative estimate
        """
        # 1. Minimum slippage
        min_slippage_bps = self.min_bps

        # 2. Regime-adaptive slippage
        regime_result = self.regime_model.calculate(
            price=price,
            side=side,
            qty=qty,
            vix=vix,
            **kwargs,
        )
        regime_bps = regime_result.slippage_bps

        # 3. Volume impact (if available)
        if daily_volume is not None and daily_volume > 0:
            participation = qty / daily_volume
            vol = volatility if volatility and volatility > 0 else 0.02
            impact_bps = self.volume_impact_coef * vol * np.sqrt(participation) * 10000
        else:
            impact_bps = 0.0

        # Take maximum
        final_bps = max(min_slippage_bps, regime_bps, impact_bps)
        final_bps = min(final_bps, 100.0)  # Cap at 1%

        slippage_pct = final_bps / 10000
        slippage_amount = price * slippage_pct
        adjusted_price = self.adjust_price(price, slippage_amount, side)

        return SlippageResult(
            model_type=self.model_type,
            base_price=price,
            slippage_amount=slippage_amount,
            adjusted_price=adjusted_price,
            slippage_bps=final_bps,
            metadata={
                "model": "hybrid",
                "min_bps": min_slippage_bps,
                "regime_bps": regime_bps,
                "volume_impact_bps": impact_bps,
                "final_bps": final_bps,
                "vix": vix,
                "regime": regime_result.regime.name if hasattr(regime_result, "regime") else None,
            },
        )


def create_regime_slippage_model(
    model_type: str = "regime_adaptive",
    base_bps: float = 5.0,
    vix_data: Optional[pd.DataFrame] = None,
    **kwargs,
) -> SlippageModel:
    """
    Factory function to create regime-aware slippage models.

    Args:
        model_type: Type of model ("regime_adaptive", "vix_integrated", "hybrid")
        base_bps: Base slippage in basis points
        vix_data: VIX data for integrated model
        **kwargs: Model-specific parameters

    Returns:
        SlippageModel instance
    """
    model_type = model_type.lower()

    if model_type == "regime_adaptive":
        return RegimeAdaptiveSlippage(base_bps=base_bps, **kwargs)
    elif model_type == "vix_integrated":
        return VIXIntegratedSlippage(base_bps=base_bps, vix_data=vix_data, **kwargs)
    elif model_type == "hybrid":
        return HybridSlippage(base_bps=base_bps, **kwargs)
    else:
        raise ValueError(f"Unknown regime slippage model type: {model_type}")


# Default model instance
DEFAULT_REGIME_SLIPPAGE = RegimeAdaptiveSlippage(base_bps=5.0)
