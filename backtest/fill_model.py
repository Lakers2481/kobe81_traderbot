"""
Fill probability models for backtesting.

Models the likelihood and extent of order fills.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional
import random


class FillModelType(Enum):
    """Types of fill models."""
    ALWAYS_FILL = auto()
    LIMIT_ORDER = auto()
    PROBABILISTIC = auto()
    PARTIAL_FILL = auto()


@dataclass
class FillResult:
    """Result of fill calculation."""
    model_type: FillModelType
    would_fill: bool
    fill_qty: int
    fill_price: float
    fill_probability: float
    metadata: Dict[str, Any]


class FillModel(ABC):
    """Base class for fill models."""

    @property
    @abstractmethod
    def model_type(self) -> FillModelType:
        """Return the model type."""
        pass

    @abstractmethod
    def calculate_fill(
        self,
        order_side: str,
        limit_price: float,
        qty: int,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_volume: Optional[float] = None,
        **kwargs,
    ) -> FillResult:
        """
        Calculate fill for an order given bar data.

        Args:
            order_side: "buy" or "sell"
            limit_price: Order limit price
            qty: Order quantity
            bar_open: Bar open price
            bar_high: Bar high price
            bar_low: Bar low price
            bar_close: Bar close price
            bar_volume: Bar volume (for partial fill models)
            **kwargs: Model-specific parameters

        Returns:
            FillResult with fill details
        """
        pass


class AlwaysFillModel(FillModel):
    """
    Always fill at close price.

    Simple model - assumes all orders fill at bar close.
    Default behavior for most backtests.
    """

    @property
    def model_type(self) -> FillModelType:
        return FillModelType.ALWAYS_FILL

    def calculate_fill(
        self,
        order_side: str,
        limit_price: float,
        qty: int,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_volume: Optional[float] = None,
        **kwargs,
    ) -> FillResult:
        return FillResult(
            model_type=self.model_type,
            would_fill=True,
            fill_qty=qty,
            fill_price=bar_close,
            fill_probability=1.0,
            metadata={"model": "always_fill"},
        )


class LimitOrderFillModel(FillModel):
    """
    Limit order fill model.

    Fills if price touches limit during the bar:
    - Buy limit fills if bar_low <= limit_price
    - Sell limit fills if bar_high >= limit_price

    Fill price is the limit price (best case) or bar open if
    already through the limit at open.
    """

    def __init__(self, use_open_for_gap: bool = True):
        """
        Initialize limit order fill model.

        Args:
            use_open_for_gap: If True, fill at open when price gaps through limit
        """
        self.use_open_for_gap = use_open_for_gap

    @property
    def model_type(self) -> FillModelType:
        return FillModelType.LIMIT_ORDER

    def calculate_fill(
        self,
        order_side: str,
        limit_price: float,
        qty: int,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_volume: Optional[float] = None,
        **kwargs,
    ) -> FillResult:
        side = order_side.lower()

        if side == "buy":
            # Buy limit fills if price touches or goes below limit
            would_fill = bar_low <= limit_price

            if would_fill:
                # Check if gapped below limit
                if self.use_open_for_gap and bar_open < limit_price:
                    fill_price = bar_open
                else:
                    fill_price = limit_price
            else:
                fill_price = 0.0

        else:  # sell
            # Sell limit fills if price touches or goes above limit
            would_fill = bar_high >= limit_price

            if would_fill:
                # Check if gapped above limit
                if self.use_open_for_gap and bar_open > limit_price:
                    fill_price = bar_open
                else:
                    fill_price = limit_price
            else:
                fill_price = 0.0

        return FillResult(
            model_type=self.model_type,
            would_fill=would_fill,
            fill_qty=qty if would_fill else 0,
            fill_price=fill_price,
            fill_probability=1.0 if would_fill else 0.0,
            metadata={
                "model": "limit_order",
                "limit_price": limit_price,
                "bar_range": (bar_low, bar_high),
            },
        )


class ProbabilisticFillModel(FillModel):
    """
    Probabilistic fill model.

    Assigns a fill probability based on how far the limit price
    is from the bar's range. Orders at the edge of the range
    have lower probability of filling.
    """

    def __init__(
        self,
        base_probability: float = 0.95,
        edge_penalty: float = 0.3,
    ):
        """
        Initialize probabilistic fill model.

        Args:
            base_probability: Base fill probability when limit is inside range
            edge_penalty: Probability reduction when limit is at edge of range
        """
        self.base_probability = base_probability
        self.edge_penalty = edge_penalty

    @property
    def model_type(self) -> FillModelType:
        return FillModelType.PROBABILISTIC

    def calculate_fill(
        self,
        order_side: str,
        limit_price: float,
        qty: int,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_volume: Optional[float] = None,
        **kwargs,
    ) -> FillResult:
        side = order_side.lower()
        bar_range = bar_high - bar_low

        if bar_range <= 0:
            # No range - degenerate case
            if side == "buy" and limit_price >= bar_close:
                probability = 1.0
            elif side == "sell" and limit_price <= bar_close:
                probability = 1.0
            else:
                probability = 0.0
        else:
            if side == "buy":
                if limit_price < bar_low:
                    # Limit below range - no fill
                    probability = 0.0
                elif limit_price > bar_high:
                    # Limit above range - guaranteed fill
                    probability = 1.0
                else:
                    # Limit within range - probability based on position
                    # Higher limit = higher probability for buys
                    position = (limit_price - bar_low) / bar_range
                    probability = self.base_probability * position
                    # Apply edge penalty near the low
                    if position < 0.2:
                        probability *= (1 - self.edge_penalty)
            else:  # sell
                if limit_price > bar_high:
                    # Limit above range - no fill
                    probability = 0.0
                elif limit_price < bar_low:
                    # Limit below range - guaranteed fill
                    probability = 1.0
                else:
                    # Limit within range
                    # Lower limit = higher probability for sells
                    position = (bar_high - limit_price) / bar_range
                    probability = self.base_probability * position
                    if position < 0.2:
                        probability *= (1 - self.edge_penalty)

        # Simulate the fill based on probability
        would_fill = random.random() < probability

        if would_fill:
            fill_price = limit_price
            fill_qty = qty
        else:
            fill_price = 0.0
            fill_qty = 0

        return FillResult(
            model_type=self.model_type,
            would_fill=would_fill,
            fill_qty=fill_qty,
            fill_price=fill_price,
            fill_probability=probability,
            metadata={
                "model": "probabilistic",
                "calculated_probability": probability,
                "bar_range": bar_range,
            },
        )


class PartialFillModel(FillModel):
    """
    Partial fill model based on volume.

    Models partial fills for larger orders that may not fully
    execute at the limit price. Uses participation rate to
    determine fill quantity.
    """

    def __init__(
        self,
        max_participation_rate: float = 0.10,
        min_fill_ratio: float = 0.25,
    ):
        """
        Initialize partial fill model.

        Args:
            max_participation_rate: Maximum % of bar volume we can capture
            min_fill_ratio: Minimum fill ratio if any fill occurs
        """
        self.max_participation_rate = max_participation_rate
        self.min_fill_ratio = min_fill_ratio

    @property
    def model_type(self) -> FillModelType:
        return FillModelType.PARTIAL_FILL

    def calculate_fill(
        self,
        order_side: str,
        limit_price: float,
        qty: int,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_volume: Optional[float] = None,
        **kwargs,
    ) -> FillResult:
        side = order_side.lower()

        # First check if limit would be touched
        if side == "buy":
            would_fill = bar_low <= limit_price
        else:
            would_fill = bar_high >= limit_price

        if not would_fill:
            return FillResult(
                model_type=self.model_type,
                would_fill=False,
                fill_qty=0,
                fill_price=0.0,
                fill_probability=0.0,
                metadata={"model": "partial_fill", "reason": "limit_not_touched"},
            )

        # Calculate max fill based on volume
        if bar_volume is None or bar_volume <= 0:
            # No volume data - assume full fill
            fill_qty = qty
            fill_ratio = 1.0
        else:
            max_shares = int(bar_volume * self.max_participation_rate)
            fill_qty = min(qty, max_shares)

            # Ensure minimum fill ratio
            if fill_qty < qty * self.min_fill_ratio:
                fill_qty = max(1, int(qty * self.min_fill_ratio))

            fill_ratio = fill_qty / qty

        # Determine fill price
        if side == "buy":
            if bar_open < limit_price:
                fill_price = bar_open
            else:
                fill_price = limit_price
        else:
            if bar_open > limit_price:
                fill_price = bar_open
            else:
                fill_price = limit_price

        return FillResult(
            model_type=self.model_type,
            would_fill=True,
            fill_qty=fill_qty,
            fill_price=fill_price,
            fill_probability=fill_ratio,
            metadata={
                "model": "partial_fill",
                "requested_qty": qty,
                "fill_ratio": fill_ratio,
                "bar_volume": bar_volume,
                "max_participation": self.max_participation_rate,
            },
        )


def create_fill_model(
    model_type: str = "always",
    **kwargs,
) -> FillModel:
    """
    Factory function to create fill models.

    Args:
        model_type: Type of model ("always", "limit", "probabilistic", "partial")
        **kwargs: Model-specific parameters

    Returns:
        FillModel instance
    """
    model_type = model_type.lower()

    if model_type in ("always", "always_fill"):
        return AlwaysFillModel()
    elif model_type in ("limit", "limit_order"):
        return LimitOrderFillModel(
            use_open_for_gap=kwargs.get("use_open_for_gap", True),
        )
    elif model_type == "probabilistic":
        return ProbabilisticFillModel(
            base_probability=kwargs.get("base_probability", 0.95),
            edge_penalty=kwargs.get("edge_penalty", 0.3),
        )
    elif model_type in ("partial", "partial_fill"):
        return PartialFillModel(
            max_participation_rate=kwargs.get("max_participation_rate", 0.10),
            min_fill_ratio=kwargs.get("min_fill_ratio", 0.25),
        )
    else:
        raise ValueError(f"Unknown fill model type: {model_type}")


# Default model
DEFAULT_FILL_MODEL = AlwaysFillModel()
