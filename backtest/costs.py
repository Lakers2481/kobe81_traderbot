"""
Cost models for backtesting.

Models commissions, fees, and other trading costs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional


class CostType(Enum):
    """Types of cost models."""
    ZERO = auto()
    EQUITY_COMMISSION = auto()
    CRYPTO_TAKER = auto()
    TIERED = auto()


@dataclass
class CostResult:
    """Result of cost calculation."""
    cost_type: CostType
    total_cost: float
    breakdown: Dict[str, float]
    metadata: Dict[str, Any]


class CostModel(ABC):
    """Base class for cost models."""

    @property
    @abstractmethod
    def cost_type(self) -> CostType:
        """Return the cost type."""
        pass

    @abstractmethod
    def calculate(
        self,
        price: float,
        qty: int,
        side: str,
        **kwargs,
    ) -> CostResult:
        """
        Calculate trading costs.

        Args:
            price: Execution price
            qty: Order quantity
            side: "buy" or "sell"
            **kwargs: Model-specific parameters

        Returns:
            CostResult with breakdown
        """
        pass


class ZeroCostModel(CostModel):
    """No costs - for comparison purposes."""

    @property
    def cost_type(self) -> CostType:
        return CostType.ZERO

    def calculate(
        self,
        price: float,
        qty: int,
        side: str,
        **kwargs,
    ) -> CostResult:
        return CostResult(
            cost_type=self.cost_type,
            total_cost=0.0,
            breakdown={},
            metadata={"model": "zero"},
        )


class EquityCommissionCost(CostModel):
    """
    Equity commission and regulatory fee model.

    For commission-free brokers (Alpaca), only regulatory fees apply:
    - SEC fee: $22.90 per $1M of sell proceeds (sells only)
    - FINRA TAF: $0.000119 per share (max $5.95 per trade)
    - Exchange fees: typically passed through

    For brokers with commissions:
    - Per-share: e.g., $0.005/share
    - Flat fee: e.g., $4.95/trade
    - BPS: e.g., 10 bps of notional
    """

    # Regulatory fee rates (as of 2024)
    SEC_FEE_RATE = 22.90 / 1_000_000  # Per dollar of sell proceeds
    FINRA_TAF_RATE = 0.000119  # Per share
    FINRA_TAF_MAX = 5.95  # Maximum per trade

    def __init__(
        self,
        per_share: float = 0.0,
        flat_fee: float = 0.0,
        bps: float = 0.0,
        include_sec_fee: bool = True,
        include_finra_taf: bool = True,
    ):
        """
        Initialize equity cost model.

        Args:
            per_share: Commission per share
            flat_fee: Flat commission per trade
            bps: Commission in basis points
            include_sec_fee: Include SEC fee on sells
            include_finra_taf: Include FINRA TAF
        """
        self.per_share = per_share
        self.flat_fee = flat_fee
        self.bps = bps
        self.include_sec_fee = include_sec_fee
        self.include_finra_taf = include_finra_taf

    @property
    def cost_type(self) -> CostType:
        return CostType.EQUITY_COMMISSION

    def calculate(
        self,
        price: float,
        qty: int,
        side: str,
        **kwargs,
    ) -> CostResult:
        notional = price * qty
        breakdown = {}
        total = 0.0

        # Per-share commission
        if self.per_share > 0:
            per_share_cost = self.per_share * qty
            breakdown["per_share"] = per_share_cost
            total += per_share_cost

        # Flat fee
        if self.flat_fee > 0:
            breakdown["flat_fee"] = self.flat_fee
            total += self.flat_fee

        # BPS commission
        if self.bps > 0:
            bps_cost = notional * (self.bps / 10000)
            breakdown["bps_commission"] = bps_cost
            total += bps_cost

        # SEC fee (sells only)
        if self.include_sec_fee and side.lower() == "sell":
            sec_fee = notional * self.SEC_FEE_RATE
            breakdown["sec_fee"] = sec_fee
            total += sec_fee

        # FINRA TAF
        if self.include_finra_taf:
            finra_taf = min(qty * self.FINRA_TAF_RATE, self.FINRA_TAF_MAX)
            breakdown["finra_taf"] = finra_taf
            total += finra_taf

        return CostResult(
            cost_type=self.cost_type,
            total_cost=total,
            breakdown=breakdown,
            metadata={
                "model": "equity_commission",
                "notional": notional,
                "qty": qty,
                "side": side,
            },
        )


class CryptoTakerCost(CostModel):
    """
    Crypto taker fee model.

    Most exchanges charge taker fees around 0.1% for market orders.
    Maker fees are typically lower but we assume taker for backtests.
    """

    def __init__(self, taker_fee_bps: float = 10.0):
        """
        Initialize crypto cost model.

        Args:
            taker_fee_bps: Taker fee in basis points (default 10 = 0.1%)
        """
        self.taker_fee_bps = taker_fee_bps

    @property
    def cost_type(self) -> CostType:
        return CostType.CRYPTO_TAKER

    def calculate(
        self,
        price: float,
        qty: int,
        side: str,
        **kwargs,
    ) -> CostResult:
        notional = price * qty
        taker_fee = notional * (self.taker_fee_bps / 10000)

        return CostResult(
            cost_type=self.cost_type,
            total_cost=taker_fee,
            breakdown={"taker_fee": taker_fee},
            metadata={
                "model": "crypto_taker",
                "taker_fee_bps": self.taker_fee_bps,
                "notional": notional,
            },
        )


class TieredCostModel(CostModel):
    """
    Tiered cost model based on monthly volume.

    Higher monthly volume = lower fees.
    """

    def __init__(
        self,
        tiers: Optional[Dict[float, float]] = None,
        current_monthly_volume: float = 0.0,
    ):
        """
        Initialize tiered cost model.

        Args:
            tiers: Dict mapping volume threshold to fee bps
                   e.g., {0: 10, 100000: 7, 500000: 5, 1000000: 3}
            current_monthly_volume: Current month's trading volume
        """
        self.tiers = tiers or {
            0: 10,        # $0+: 10 bps
            100000: 7,    # $100k+: 7 bps
            500000: 5,    # $500k+: 5 bps
            1000000: 3,   # $1M+: 3 bps
        }
        self.current_monthly_volume = current_monthly_volume

    @property
    def cost_type(self) -> CostType:
        return CostType.TIERED

    def get_current_tier_bps(self) -> float:
        """Get fee bps for current volume tier."""
        applicable_bps = 10.0  # Default
        for threshold, bps in sorted(self.tiers.items()):
            if self.current_monthly_volume >= threshold:
                applicable_bps = bps
        return applicable_bps

    def calculate(
        self,
        price: float,
        qty: int,
        side: str,
        **kwargs,
    ) -> CostResult:
        notional = price * qty
        tier_bps = self.get_current_tier_bps()
        fee = notional * (tier_bps / 10000)

        return CostResult(
            cost_type=self.cost_type,
            total_cost=fee,
            breakdown={"tiered_fee": fee},
            metadata={
                "model": "tiered",
                "tier_bps": tier_bps,
                "monthly_volume": self.current_monthly_volume,
                "notional": notional,
            },
        )


def create_cost_model(
    model_type: str = "equity",
    **kwargs,
) -> CostModel:
    """
    Factory function to create cost models.

    Args:
        model_type: Type of model ("zero", "equity", "crypto", "tiered")
        **kwargs: Model-specific parameters

    Returns:
        CostModel instance
    """
    model_type = model_type.lower()

    if model_type == "zero":
        return ZeroCostModel()
    elif model_type in ("equity", "equity_commission"):
        return EquityCommissionCost(
            per_share=kwargs.get("per_share", 0.0),
            flat_fee=kwargs.get("flat_fee", 0.0),
            bps=kwargs.get("bps", 0.0),
            include_sec_fee=kwargs.get("include_sec_fee", True),
            include_finra_taf=kwargs.get("include_finra_taf", True),
        )
    elif model_type in ("crypto", "crypto_taker"):
        return CryptoTakerCost(
            taker_fee_bps=kwargs.get("taker_fee_bps", 10.0),
        )
    elif model_type == "tiered":
        return TieredCostModel(
            tiers=kwargs.get("tiers"),
            current_monthly_volume=kwargs.get("current_monthly_volume", 0.0),
        )
    else:
        raise ValueError(f"Unknown cost model type: {model_type}")


# Default model for Alpaca (commission-free, regulatory fees only)
DEFAULT_COST_MODEL = EquityCommissionCost(
    per_share=0.0,
    flat_fee=0.0,
    bps=0.0,
    include_sec_fee=True,
    include_finra_taf=True,
)
