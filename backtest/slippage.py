"""
Configurable slippage models for backtesting.

Provides multiple slippage estimation methods for realistic backtests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional
import math


class SlippageType(Enum):
    """Available slippage model types."""
    FIXED_BPS = auto()
    ATR_FRACTION = auto()
    SPREAD_PERCENTILE = auto()
    VOLUME_IMPACT = auto()
    ZERO = auto()


@dataclass
class SlippageResult:
    """Result of slippage calculation."""
    model_type: SlippageType
    base_price: float
    slippage_amount: float
    adjusted_price: float
    slippage_bps: float
    metadata: Dict[str, Any]

    @property
    def slippage_pct(self) -> float:
        """Slippage as percentage."""
        return self.slippage_bps / 100


class SlippageModel(ABC):
    """Base class for slippage models."""

    @property
    @abstractmethod
    def model_type(self) -> SlippageType:
        """Return the model type."""
        pass

    @abstractmethod
    def calculate(
        self,
        price: float,
        side: str,
        qty: int,
        **kwargs,
    ) -> SlippageResult:
        """
        Calculate slippage for a trade.

        Args:
            price: Base price (typically close or limit price)
            side: "buy" or "sell"
            qty: Order quantity
            **kwargs: Model-specific parameters

        Returns:
            SlippageResult with adjusted price
        """
        pass

    def adjust_price(self, price: float, slippage: float, side: str) -> float:
        """
        Adjust price by slippage amount.

        For buys: add slippage (pay more)
        For sells: subtract slippage (receive less)
        """
        if side.lower() == "buy":
            return price + slippage
        else:
            return price - slippage


class ZeroSlippage(SlippageModel):
    """No slippage - fills at exact price."""

    @property
    def model_type(self) -> SlippageType:
        return SlippageType.ZERO

    def calculate(
        self,
        price: float,
        side: str,
        qty: int,
        **kwargs,
    ) -> SlippageResult:
        return SlippageResult(
            model_type=self.model_type,
            base_price=price,
            slippage_amount=0.0,
            adjusted_price=price,
            slippage_bps=0.0,
            metadata={"model": "zero"},
        )


class FixedBpsSlippage(SlippageModel):
    """
    Fixed basis points slippage.

    Simple and predictable - good default for liquid equities.
    """

    def __init__(self, bps: float = 5.0):
        """
        Initialize fixed BPS slippage.

        Args:
            bps: Basis points of slippage (default 5 = 0.05%)
        """
        self.bps = bps

    @property
    def model_type(self) -> SlippageType:
        return SlippageType.FIXED_BPS

    def calculate(
        self,
        price: float,
        side: str,
        qty: int,
        **kwargs,
    ) -> SlippageResult:
        slippage_pct = self.bps / 10000  # Convert bps to decimal
        slippage_amount = price * slippage_pct

        adjusted_price = self.adjust_price(price, slippage_amount, side)

        return SlippageResult(
            model_type=self.model_type,
            base_price=price,
            slippage_amount=slippage_amount,
            adjusted_price=adjusted_price,
            slippage_bps=self.bps,
            metadata={
                "model": "fixed_bps",
                "bps_setting": self.bps,
            },
        )


class ATRFractionSlippage(SlippageModel):
    """
    ATR-based slippage.

    Uses a fraction of Average True Range for more dynamic slippage
    that adapts to volatility.
    """

    def __init__(self, atr_fraction: float = 0.10):
        """
        Initialize ATR fraction slippage.

        Args:
            atr_fraction: Fraction of ATR to use as slippage (default 10%)
        """
        self.atr_fraction = atr_fraction

    @property
    def model_type(self) -> SlippageType:
        return SlippageType.ATR_FRACTION

    def calculate(
        self,
        price: float,
        side: str,
        qty: int,
        atr: Optional[float] = None,
        **kwargs,
    ) -> SlippageResult:
        if atr is None or atr <= 0:
            # Fall back to fixed 5 bps if no ATR
            slippage_amount = price * 0.0005
            slippage_bps = 5.0
            metadata = {
                "model": "atr_fraction",
                "fallback": True,
                "reason": "ATR not provided",
            }
        else:
            slippage_amount = atr * self.atr_fraction
            slippage_bps = (slippage_amount / price) * 10000
            metadata = {
                "model": "atr_fraction",
                "atr": atr,
                "atr_fraction": self.atr_fraction,
            }

        adjusted_price = self.adjust_price(price, slippage_amount, side)

        return SlippageResult(
            model_type=self.model_type,
            base_price=price,
            slippage_amount=slippage_amount,
            adjusted_price=adjusted_price,
            slippage_bps=slippage_bps,
            metadata=metadata,
        )


class SpreadPercentileSlippage(SlippageModel):
    """
    Spread-based slippage.

    Uses estimated or actual bid-ask spread to model slippage.
    Typically crosses half the spread.
    """

    def __init__(self, spread_fraction: float = 0.50):
        """
        Initialize spread percentile slippage.

        Args:
            spread_fraction: Fraction of spread to cross (default 50%)
        """
        self.spread_fraction = spread_fraction

    @property
    def model_type(self) -> SlippageType:
        return SlippageType.SPREAD_PERCENTILE

    def calculate(
        self,
        price: float,
        side: str,
        qty: int,
        spread: Optional[float] = None,
        spread_pct: Optional[float] = None,
        **kwargs,
    ) -> SlippageResult:
        # Determine spread
        if spread is not None and spread > 0:
            actual_spread = spread
        elif spread_pct is not None and spread_pct > 0:
            actual_spread = price * (spread_pct / 100)
        else:
            # Estimate spread based on price level
            # Higher-priced stocks tend to have tighter relative spreads
            if price >= 100:
                estimated_spread_pct = 0.02  # 2 bps
            elif price >= 50:
                estimated_spread_pct = 0.03  # 3 bps
            elif price >= 20:
                estimated_spread_pct = 0.05  # 5 bps
            else:
                estimated_spread_pct = 0.10  # 10 bps
            actual_spread = price * (estimated_spread_pct / 100)

        slippage_amount = actual_spread * self.spread_fraction
        slippage_bps = (slippage_amount / price) * 10000

        adjusted_price = self.adjust_price(price, slippage_amount, side)

        return SlippageResult(
            model_type=self.model_type,
            base_price=price,
            slippage_amount=slippage_amount,
            adjusted_price=adjusted_price,
            slippage_bps=slippage_bps,
            metadata={
                "model": "spread_percentile",
                "spread": actual_spread,
                "spread_fraction": self.spread_fraction,
            },
        )


class VolumeImpactSlippage(SlippageModel):
    """
    Volume-based market impact slippage.

    Uses square-root impact model: impact = sigma * sqrt(qty / ADV)
    where sigma is daily volatility and ADV is average daily volume.
    """

    def __init__(
        self,
        impact_coefficient: float = 0.1,
        min_bps: float = 2.0,
        max_bps: float = 100.0,
    ):
        """
        Initialize volume impact slippage.

        Args:
            impact_coefficient: Multiplier for impact calculation
            min_bps: Minimum slippage floor
            max_bps: Maximum slippage cap
        """
        self.impact_coefficient = impact_coefficient
        self.min_bps = min_bps
        self.max_bps = max_bps

    @property
    def model_type(self) -> SlippageType:
        return SlippageType.VOLUME_IMPACT

    def calculate(
        self,
        price: float,
        side: str,
        qty: int,
        daily_volume: Optional[float] = None,
        volatility: Optional[float] = None,
        **kwargs,
    ) -> SlippageResult:
        if daily_volume is None or daily_volume <= 0:
            # Fall back to fixed slippage
            slippage_bps = 10.0
            metadata = {
                "model": "volume_impact",
                "fallback": True,
                "reason": "Volume not provided",
            }
        elif volatility is None or volatility <= 0:
            # Estimate volatility as 2% if not provided
            volatility = 0.02
            participation = qty / daily_volume
            impact_pct = self.impact_coefficient * volatility * math.sqrt(participation)
            slippage_bps = impact_pct * 10000
            metadata = {
                "model": "volume_impact",
                "volatility_estimated": True,
                "volatility": volatility,
                "participation_rate": participation,
            }
        else:
            participation = qty / daily_volume
            impact_pct = self.impact_coefficient * volatility * math.sqrt(participation)
            slippage_bps = impact_pct * 10000
            metadata = {
                "model": "volume_impact",
                "volatility": volatility,
                "daily_volume": daily_volume,
                "participation_rate": participation,
            }

        # Apply min/max bounds
        slippage_bps = max(self.min_bps, min(self.max_bps, slippage_bps))
        slippage_amount = price * (slippage_bps / 10000)

        adjusted_price = self.adjust_price(price, slippage_amount, side)

        metadata["final_bps"] = slippage_bps
        metadata["min_bps"] = self.min_bps
        metadata["max_bps"] = self.max_bps

        return SlippageResult(
            model_type=self.model_type,
            base_price=price,
            slippage_amount=slippage_amount,
            adjusted_price=adjusted_price,
            slippage_bps=slippage_bps,
            metadata=metadata,
        )


def create_slippage_model(
    model_type: str = "fixed_bps",
    **kwargs,
) -> SlippageModel:
    """
    Factory function to create slippage models.

    Args:
        model_type: Type of model ("fixed_bps", "atr_fraction", "spread_percentile", "volume_impact", "zero")
        **kwargs: Model-specific parameters

    Returns:
        SlippageModel instance
    """
    model_type = model_type.lower()

    if model_type == "zero":
        return ZeroSlippage()
    elif model_type == "fixed_bps":
        bps = kwargs.get("bps", 5.0)
        return FixedBpsSlippage(bps=bps)
    elif model_type == "atr_fraction":
        atr_fraction = kwargs.get("atr_fraction", 0.10)
        return ATRFractionSlippage(atr_fraction=atr_fraction)
    elif model_type == "spread_percentile":
        spread_fraction = kwargs.get("spread_fraction", 0.50)
        return SpreadPercentileSlippage(spread_fraction=spread_fraction)
    elif model_type == "volume_impact":
        return VolumeImpactSlippage(
            impact_coefficient=kwargs.get("impact_coefficient", 0.1),
            min_bps=kwargs.get("min_bps", 2.0),
            max_bps=kwargs.get("max_bps", 100.0),
        )
    else:
        raise ValueError(f"Unknown slippage model type: {model_type}")


# Default model for convenience
DEFAULT_SLIPPAGE_MODEL = FixedBpsSlippage(bps=5.0)
