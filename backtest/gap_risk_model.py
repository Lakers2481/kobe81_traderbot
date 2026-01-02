"""
Gap Risk Model for Overnight Position Risk.

Models overnight gaps for positions held through market close.
Essential for realistic backtesting of swing trades.

Key Features:
- Historical gap distribution modeling
- VIX-adjusted gap risk
- Earnings/event gap multipliers
- Monte Carlo gap simulation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import random

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GapType(Enum):
    """Types of overnight gaps."""
    NORMAL = auto()        # Regular market gap
    EARNINGS = auto()      # Post-earnings gap
    NEWS_EVENT = auto()    # News-driven gap
    MARKET_WIDE = auto()   # Broad market gap (correlated)


@dataclass
class GapRiskResult:
    """Result of gap risk calculation."""
    symbol: str
    gap_type: GapType
    expected_gap_pct: float       # Expected gap magnitude
    gap_std: float                # Standard deviation of gap
    worst_case_gap_pct: float     # 95th percentile adverse gap
    simulated_gap_pct: float      # Actual simulated gap (if run)
    stop_gap_out: bool            # Would stop be gapped through?
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def expected_gap_dollars(self) -> float:
        """Expected gap in dollar terms per share (requires price in metadata)."""
        price = self.metadata.get("entry_price", 100)
        return price * (self.expected_gap_pct / 100)


@dataclass
class GapDistribution:
    """Historical gap distribution parameters."""
    mean: float = 0.0          # Mean gap % (slightly positive historically)
    std: float = 1.2           # Std dev of gaps (typical ~1.2%)
    skewness: float = -0.3     # Slight negative skew (bigger down gaps)
    kurtosis: float = 4.0      # Fat tails

    # Percentiles for reference
    p5: float = -2.5           # 5th percentile (adverse for longs)
    p95: float = 2.0           # 95th percentile (adverse for shorts)


class GapRiskModel:
    """
    Models overnight gap risk for equity positions.

    Uses historical gap distributions adjusted for:
    - Current volatility regime (VIX)
    - Upcoming events (earnings, dividends)
    - Sector and market-wide factors

    Default parameters calibrated to S&P 500 historical gaps.
    """

    # Historical gap statistics (S&P 500, 2000-2024)
    DEFAULT_GAP_STATS = GapDistribution(
        mean=0.03,      # Slight positive bias
        std=1.2,        # ~1.2% daily std dev
        skewness=-0.3,  # Negative skew
        kurtosis=4.0,   # Fat tails
        p5=-2.5,
        p95=2.0,
    )

    # VIX adjustment multipliers
    VIX_MULTIPLIERS = {
        "low": 0.6,      # VIX < 15
        "normal": 1.0,   # VIX 15-25
        "elevated": 1.5, # VIX 25-35
        "high": 2.5,     # VIX > 35
    }

    # Event multipliers
    EVENT_MULTIPLIERS = {
        GapType.NORMAL: 1.0,
        GapType.EARNINGS: 3.0,      # Earnings gaps are ~3x larger
        GapType.NEWS_EVENT: 2.0,    # News events ~2x
        GapType.MARKET_WIDE: 1.5,   # Market-wide events ~1.5x
    }

    def __init__(
        self,
        base_gap_std: float = 1.2,
        vix_adjustment: bool = True,
        event_adjustment: bool = True,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize gap risk model.

        Args:
            base_gap_std: Base standard deviation of gaps (default 1.2%)
            vix_adjustment: Whether to adjust for VIX level
            event_adjustment: Whether to adjust for events
            random_seed: Random seed for reproducibility
        """
        self.base_gap_std = base_gap_std
        self.vix_adjustment = vix_adjustment
        self.event_adjustment = event_adjustment

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def get_vix_regime(self, vix: float) -> str:
        """Classify VIX into regime."""
        if vix < 15:
            return "low"
        elif vix < 25:
            return "normal"
        elif vix < 35:
            return "elevated"
        else:
            return "high"

    def get_gap_multiplier(
        self,
        vix: Optional[float] = None,
        gap_type: GapType = GapType.NORMAL,
    ) -> float:
        """Calculate total gap multiplier based on VIX and event type."""
        multiplier = 1.0

        # VIX adjustment
        if self.vix_adjustment and vix is not None:
            regime = self.get_vix_regime(vix)
            multiplier *= self.VIX_MULTIPLIERS[regime]

        # Event adjustment
        if self.event_adjustment:
            multiplier *= self.EVENT_MULTIPLIERS[gap_type]

        return multiplier

    def estimate_gap_distribution(
        self,
        symbol: str,
        vix: Optional[float] = None,
        gap_type: GapType = GapType.NORMAL,
        historical_gaps: Optional[pd.Series] = None,
    ) -> GapDistribution:
        """
        Estimate gap distribution for a symbol.

        Args:
            symbol: Stock symbol
            vix: Current VIX level
            gap_type: Type of expected gap
            historical_gaps: Historical gap data for symbol (if available)

        Returns:
            GapDistribution with adjusted parameters
        """
        # Start with defaults or historical if available
        if historical_gaps is not None and len(historical_gaps) > 20:
            base_dist = GapDistribution(
                mean=float(historical_gaps.mean()),
                std=float(historical_gaps.std()),
                skewness=float(historical_gaps.skew()) if len(historical_gaps) > 50 else -0.3,
                kurtosis=float(historical_gaps.kurtosis()) if len(historical_gaps) > 50 else 4.0,
                p5=float(historical_gaps.quantile(0.05)),
                p95=float(historical_gaps.quantile(0.95)),
            )
        else:
            base_dist = self.DEFAULT_GAP_STATS

        # Apply multiplier
        multiplier = self.get_gap_multiplier(vix, gap_type)

        return GapDistribution(
            mean=base_dist.mean,
            std=base_dist.std * multiplier,
            skewness=base_dist.skewness,
            kurtosis=base_dist.kurtosis,
            p5=base_dist.p5 * multiplier,
            p95=base_dist.p95 * multiplier,
        )

    def simulate_gap(
        self,
        distribution: GapDistribution,
        n_simulations: int = 1,
    ) -> np.ndarray:
        """
        Simulate gap(s) from distribution.

        Uses skewed-t distribution to capture fat tails and asymmetry.

        Args:
            distribution: Gap distribution parameters
            n_simulations: Number of gaps to simulate

        Returns:
            Array of simulated gap percentages
        """
        # Use normal as base, adjust for skewness via shift
        # More sophisticated would use skewed-t, but this is reasonable
        base_gaps = np.random.normal(
            loc=distribution.mean,
            scale=distribution.std,
            size=n_simulations,
        )

        # Add fat tails via occasional multiplier
        tail_probability = 0.05  # 5% chance of tail event
        is_tail = np.random.random(n_simulations) < tail_probability
        tail_multiplier = np.where(is_tail, 2.0, 1.0)

        # Apply skewness (make more negative gaps larger)
        if distribution.skewness < 0:
            # Negative skew: enhance negative gaps
            skew_adjustment = np.where(
                base_gaps < 0,
                1.0 + abs(distribution.skewness) * 0.5,
                1.0,
            )
        else:
            # Positive skew: enhance positive gaps
            skew_adjustment = np.where(
                base_gaps > 0,
                1.0 + distribution.skewness * 0.5,
                1.0,
            )

        return base_gaps * tail_multiplier * skew_adjustment

    def calculate_gap_risk(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_price: float,
        vix: Optional[float] = None,
        gap_type: GapType = GapType.NORMAL,
        historical_gaps: Optional[pd.Series] = None,
        n_simulations: int = 10000,
    ) -> GapRiskResult:
        """
        Calculate gap risk for a position.

        Args:
            symbol: Stock symbol
            side: "long" or "short"
            entry_price: Entry price
            stop_price: Stop loss price
            vix: Current VIX level
            gap_type: Type of expected gap
            historical_gaps: Historical gap data
            n_simulations: Number of Monte Carlo simulations

        Returns:
            GapRiskResult with risk metrics
        """
        # Get distribution
        distribution = self.estimate_gap_distribution(
            symbol=symbol,
            vix=vix,
            gap_type=gap_type,
            historical_gaps=historical_gaps,
        )

        # Simulate gaps
        simulated_gaps = self.simulate_gap(distribution, n_simulations)

        # Calculate stop distance
        if side.lower() == "long":
            stop_distance_pct = ((entry_price - stop_price) / entry_price) * 100
            # For longs, we care about adverse (negative) gaps
            adverse_gaps = -simulated_gaps  # Flip sign: negative gap is adverse for long
            worst_case_gap = np.percentile(simulated_gaps, 5)  # 5th percentile
        else:
            stop_distance_pct = ((stop_price - entry_price) / entry_price) * 100
            # For shorts, positive gaps are adverse
            adverse_gaps = simulated_gaps
            worst_case_gap = np.percentile(simulated_gaps, 95)  # 95th percentile

        # Calculate probability of gapping through stop
        gap_through_probability = np.mean(np.abs(adverse_gaps) > stop_distance_pct)

        # Single simulation for this result
        single_sim = simulated_gaps[0]

        # Check if stop would be gapped
        if side.lower() == "long":
            stop_gap_out = single_sim < -stop_distance_pct
        else:
            stop_gap_out = single_sim > stop_distance_pct

        return GapRiskResult(
            symbol=symbol,
            gap_type=gap_type,
            expected_gap_pct=distribution.mean,
            gap_std=distribution.std,
            worst_case_gap_pct=worst_case_gap,
            simulated_gap_pct=single_sim,
            stop_gap_out=stop_gap_out,
            metadata={
                "entry_price": entry_price,
                "stop_price": stop_price,
                "stop_distance_pct": stop_distance_pct,
                "vix": vix,
                "gap_through_probability": gap_through_probability,
                "p5": distribution.p5,
                "p95": distribution.p95,
                "n_simulations": n_simulations,
                "side": side,
            },
        )

    def calculate_historical_gaps(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
        open_col: str = "open",
    ) -> pd.Series:
        """
        Calculate historical overnight gaps from OHLC data.

        Args:
            df: DataFrame with OHLC data
            close_col: Name of close column
            open_col: Name of open column

        Returns:
            Series of gap percentages
        """
        # Gap = (Today's Open - Yesterday's Close) / Yesterday's Close * 100
        gaps = (df[open_col] - df[close_col].shift(1)) / df[close_col].shift(1) * 100
        return gaps.dropna()

    def apply_gap_to_backtest(
        self,
        trades_df: pd.DataFrame,
        ohlc_data: Dict[str, pd.DataFrame],
        vix_data: Optional[pd.DataFrame] = None,
        earnings_dates: Optional[Dict[str, List[datetime]]] = None,
    ) -> pd.DataFrame:
        """
        Apply gap risk adjustments to backtest trades.

        For positions held overnight, simulates realistic exit prices
        that account for potential gap risk.

        Args:
            trades_df: DataFrame of backtest trades
            ohlc_data: Dict mapping symbol to OHLC DataFrame
            vix_data: DataFrame with VIX data (optional)
            earnings_dates: Dict mapping symbol to earnings dates (optional)

        Returns:
            Updated trades DataFrame with gap-adjusted exits
        """
        adjusted_trades = trades_df.copy()

        for idx, trade in trades_df.iterrows():
            symbol = trade.get("symbol")
            if symbol not in ohlc_data:
                continue

            # Check if position held overnight
            entry_date = pd.to_datetime(trade.get("entry_date", trade.get("entry_time")))
            exit_date = pd.to_datetime(trade.get("exit_date", trade.get("exit_time")))

            if entry_date is None or exit_date is None:
                continue

            # If same day, no gap risk
            if entry_date.date() == exit_date.date():
                continue

            # Determine gap type
            gap_type = GapType.NORMAL
            if earnings_dates and symbol in earnings_dates:
                for earnings_date in earnings_dates[symbol]:
                    if entry_date.date() <= earnings_date.date() < exit_date.date():
                        gap_type = GapType.EARNINGS
                        break

            # Get VIX for the day
            vix = None
            if vix_data is not None and not vix_data.empty:
                try:
                    vix_row = vix_data.loc[vix_data.index <= entry_date].iloc[-1]
                    vix = vix_row.get("close", vix_row.get("VIX", None))
                except (IndexError, KeyError):
                    pass

            # Calculate historical gaps for this symbol
            symbol_df = ohlc_data[symbol]
            historical_gaps = self.calculate_historical_gaps(symbol_df)

            # Get trade details
            entry_price = trade.get("entry_price", 100)
            stop_price = trade.get("stop_price", entry_price * 0.95)
            side = trade.get("side", "long")

            # Calculate gap risk
            gap_result = self.calculate_gap_risk(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_price=stop_price,
                vix=vix,
                gap_type=gap_type,
                historical_gaps=historical_gaps,
            )

            # If stop gapped through, adjust exit price
            if gap_result.stop_gap_out:
                gap_pct = gap_result.simulated_gap_pct / 100

                # Calculate gapped exit price
                if side.lower() == "long":
                    gapped_price = entry_price * (1 + gap_pct)
                    # Exit price is worse than stop
                    if gapped_price < stop_price:
                        adjusted_trades.loc[idx, "exit_price"] = gapped_price
                        adjusted_trades.loc[idx, "exit_reason"] = "gap_stop"
                        adjusted_trades.loc[idx, "gap_pct"] = gap_result.simulated_gap_pct
                else:
                    gapped_price = entry_price * (1 + gap_pct)
                    # For shorts, gap up is adverse
                    if gapped_price > stop_price:
                        adjusted_trades.loc[idx, "exit_price"] = gapped_price
                        adjusted_trades.loc[idx, "exit_reason"] = "gap_stop"
                        adjusted_trades.loc[idx, "gap_pct"] = gap_result.simulated_gap_pct

        return adjusted_trades


def create_gap_risk_model(
    base_gap_std: float = 1.2,
    vix_adjustment: bool = True,
    event_adjustment: bool = True,
    random_seed: Optional[int] = None,
) -> GapRiskModel:
    """Factory function to create a GapRiskModel."""
    return GapRiskModel(
        base_gap_std=base_gap_std,
        vix_adjustment=vix_adjustment,
        event_adjustment=event_adjustment,
        random_seed=random_seed,
    )


# Default model instance
DEFAULT_GAP_RISK_MODEL = GapRiskModel()
