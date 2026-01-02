"""
Edge Decomposition for Stratified Trade Analysis.

Analyzes trading performance across multiple dimensions to understand
where edge exists and where it may be degrading.

Dimensions Analyzed:
- Day of week
- Volatility regime (low/medium/high VIX)
- Market regime (bull/bear/neutral)
- Liquidity buckets
- Gap size buckets
- Earnings proximity
- Sector
- Strategy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DimensionType(Enum):
    """Dimension types for decomposition."""
    DAY_OF_WEEK = auto()
    VOLATILITY_BUCKET = auto()
    REGIME = auto()
    LIQUIDITY = auto()
    GAP_SIZE = auto()
    EARNINGS_PROXIMITY = auto()
    SECTOR = auto()
    STRATEGY = auto()
    MONTH = auto()
    HOLD_TIME = auto()


@dataclass
class DimensionStats:
    """Statistics for a single bucket within a dimension."""
    dimension: DimensionType
    bucket_name: str
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    avg_pnl: float
    avg_winner: float
    avg_loser: float
    expectancy: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None

    @property
    def edge_score(self) -> float:
        """
        Calculate edge score (0-100).

        Based on:
        - Win rate (40% weight)
        - Profit factor (40% weight)
        - Trade count confidence (20% weight)
        """
        # Win rate contribution (0-40 points)
        wr_score = min(40, max(0, (self.win_rate - 0.30) * 100))

        # Profit factor contribution (0-40 points)
        pf_score = min(40, max(0, (self.profit_factor - 0.5) * 20))

        # Trade count confidence (0-20 points, scales with log of trades)
        tc_score = min(20, np.log1p(self.trade_count) * 3)

        return wr_score + pf_score + tc_score


@dataclass
class DecompositionResult:
    """Complete edge decomposition result."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_trades: int = 0
    overall_win_rate: float = 0.0
    overall_profit_factor: float = 0.0
    overall_pnl: float = 0.0

    by_day_of_week: Dict[str, DimensionStats] = field(default_factory=dict)
    by_volatility: Dict[str, DimensionStats] = field(default_factory=dict)
    by_regime: Dict[str, DimensionStats] = field(default_factory=dict)
    by_liquidity: Dict[str, DimensionStats] = field(default_factory=dict)
    by_gap_size: Dict[str, DimensionStats] = field(default_factory=dict)
    by_sector: Dict[str, DimensionStats] = field(default_factory=dict)
    by_strategy: Dict[str, DimensionStats] = field(default_factory=dict)
    by_month: Dict[str, DimensionStats] = field(default_factory=dict)
    by_hold_time: Dict[str, DimensionStats] = field(default_factory=dict)

    weak_spots: List[Tuple[DimensionType, str, float]] = field(default_factory=list)
    strong_spots: List[Tuple[DimensionType, str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        def stats_to_dict(stats_dict: Dict[str, DimensionStats]) -> Dict:
            return {k: {
                "trade_count": v.trade_count,
                "win_rate": round(v.win_rate, 4),
                "profit_factor": round(v.profit_factor, 4),
                "total_pnl": round(v.total_pnl, 2),
                "edge_score": round(v.edge_score, 2),
            } for k, v in stats_dict.items()}

        return {
            "timestamp": self.timestamp.isoformat(),
            "total_trades": self.total_trades,
            "overall_win_rate": round(self.overall_win_rate, 4),
            "overall_profit_factor": round(self.overall_profit_factor, 4),
            "overall_pnl": round(self.overall_pnl, 2),
            "by_day_of_week": stats_to_dict(self.by_day_of_week),
            "by_volatility": stats_to_dict(self.by_volatility),
            "by_regime": stats_to_dict(self.by_regime),
            "by_sector": stats_to_dict(self.by_sector),
            "by_strategy": stats_to_dict(self.by_strategy),
            "weak_spots": [(d.name, b, round(s, 2)) for d, b, s in self.weak_spots],
            "strong_spots": [(d.name, b, round(s, 2)) for d, b, s in self.strong_spots],
        }


class EdgeDecomposition:
    """
    Analyze trading edge across multiple dimensions.

    Stratifies trade performance to identify:
    - Where edge is strongest
    - Where edge is weakest or negative
    - How edge varies with market conditions
    """

    # Day of week names
    DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    # Volatility buckets (VIX levels)
    VOL_BUCKETS = [
        ("Low", 0, 15),
        ("Normal", 15, 20),
        ("Elevated", 20, 30),
        ("High", 30, 100),
    ]

    # Regime labels
    REGIMES = ["BULLISH", "NEUTRAL", "BEARISH"]

    # Liquidity buckets (ADV in $M)
    LIQUIDITY_BUCKETS = [
        ("Very Low", 0, 1),
        ("Low", 1, 10),
        ("Medium", 10, 50),
        ("High", 50, 500),
        ("Very High", 500, float("inf")),
    ]

    # Gap size buckets (%)
    GAP_BUCKETS = [
        ("Small Down", -float("inf"), -2),
        ("Down", -2, 0),
        ("Flat", 0, 0.5),
        ("Up", 0.5, 2),
        ("Large Up", 2, float("inf")),
    ]

    # Hold time buckets (days)
    HOLD_BUCKETS = [
        ("Intraday", 0, 1),
        ("1-2 Days", 1, 3),
        ("3-5 Days", 3, 6),
        ("1-2 Weeks", 6, 15),
        ("2+ Weeks", 15, float("inf")),
    ]

    def __init__(
        self,
        min_trades_per_bucket: int = 10,
        weak_edge_threshold: float = 30.0,  # Edge score below this is weak
        strong_edge_threshold: float = 70.0,  # Edge score above this is strong
    ):
        """
        Initialize edge decomposition.

        Args:
            min_trades_per_bucket: Minimum trades for statistical significance
            weak_edge_threshold: Edge score threshold for weak spots
            strong_edge_threshold: Edge score threshold for strong spots
        """
        self.min_trades_per_bucket = min_trades_per_bucket
        self.weak_edge_threshold = weak_edge_threshold
        self.strong_edge_threshold = strong_edge_threshold

    def analyze(
        self,
        trades_df: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None,
        regime_data: Optional[pd.DataFrame] = None,
    ) -> DecompositionResult:
        """
        Perform full edge decomposition analysis.

        Args:
            trades_df: DataFrame of trades with columns:
                - entry_date/entry_time: Entry timestamp
                - exit_date/exit_time: Exit timestamp
                - symbol: Stock symbol
                - pnl: Trade P&L
                - side: long/short
                - sector (optional): Sector name
                - strategy (optional): Strategy name
                - vix (optional): VIX at entry
                - regime (optional): Market regime at entry
                - adv (optional): Average daily volume in USD
                - gap_pct (optional): Gap percentage
            vix_data: Optional VIX DataFrame for lookups
            regime_data: Optional regime DataFrame for lookups

        Returns:
            DecompositionResult with all dimension breakdowns
        """
        if trades_df.empty:
            return DecompositionResult()

        # Ensure required columns
        trades = trades_df.copy()
        trades = self._prepare_trades(trades, vix_data, regime_data)

        # Calculate overall stats
        result = DecompositionResult(
            total_trades=len(trades),
            overall_win_rate=self._win_rate(trades),
            overall_profit_factor=self._profit_factor(trades),
            overall_pnl=trades["pnl"].sum(),
        )

        # Decompose by each dimension
        result.by_day_of_week = self._decompose_by_dow(trades)
        result.by_volatility = self._decompose_by_volatility(trades)
        result.by_regime = self._decompose_by_regime(trades)
        result.by_liquidity = self._decompose_by_liquidity(trades)
        result.by_gap_size = self._decompose_by_gap(trades)
        result.by_sector = self._decompose_by_column(trades, "sector", DimensionType.SECTOR)
        result.by_strategy = self._decompose_by_column(trades, "strategy", DimensionType.STRATEGY)
        result.by_month = self._decompose_by_month(trades)
        result.by_hold_time = self._decompose_by_hold_time(trades)

        # Identify weak and strong spots
        result.weak_spots = self._find_weak_spots(result)
        result.strong_spots = self._find_strong_spots(result)

        return result

    def _prepare_trades(
        self,
        trades: pd.DataFrame,
        vix_data: Optional[pd.DataFrame],
        regime_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Prepare trades DataFrame with derived columns."""
        # Parse entry date
        if "entry_date" in trades.columns:
            trades["entry_dt"] = pd.to_datetime(trades["entry_date"])
        elif "entry_time" in trades.columns:
            trades["entry_dt"] = pd.to_datetime(trades["entry_time"])
        else:
            trades["entry_dt"] = pd.Timestamp.now()

        # Parse exit date
        if "exit_date" in trades.columns:
            trades["exit_dt"] = pd.to_datetime(trades["exit_date"])
        elif "exit_time" in trades.columns:
            trades["exit_dt"] = pd.to_datetime(trades["exit_time"])
        else:
            trades["exit_dt"] = trades["entry_dt"] + pd.Timedelta(days=1)

        # Day of week
        trades["dow"] = trades["entry_dt"].dt.dayofweek

        # Month
        trades["month"] = trades["entry_dt"].dt.to_period("M").astype(str)

        # Hold time in days
        trades["hold_days"] = (trades["exit_dt"] - trades["entry_dt"]).dt.total_seconds() / 86400

        # VIX lookup if not present
        if "vix" not in trades.columns and vix_data is not None:
            trades["vix"] = trades["entry_dt"].apply(
                lambda dt: self._lookup_vix(dt, vix_data)
            )

        # Regime lookup if not present
        if "regime" not in trades.columns and regime_data is not None:
            trades["regime"] = trades["entry_dt"].apply(
                lambda dt: self._lookup_regime(dt, regime_data)
            )

        # Fill missing columns with defaults
        trades["vix"] = trades.get("vix", pd.Series([18.0] * len(trades)))
        trades["regime"] = trades.get("regime", pd.Series(["NEUTRAL"] * len(trades)))
        trades["sector"] = trades.get("sector", pd.Series(["Unknown"] * len(trades)))
        trades["strategy"] = trades.get("strategy", pd.Series(["Unknown"] * len(trades)))
        trades["adv"] = trades.get("adv", pd.Series([50.0] * len(trades)))  # $50M default
        trades["gap_pct"] = trades.get("gap_pct", pd.Series([0.0] * len(trades)))

        return trades

    def _lookup_vix(self, dt: datetime, vix_data: pd.DataFrame) -> float:
        """Lookup VIX for a date."""
        try:
            mask = vix_data.index <= dt
            if mask.any():
                return float(vix_data.loc[mask].iloc[-1].get("close", 18))
        except Exception:
            pass
        return 18.0

    def _lookup_regime(self, dt: datetime, regime_data: pd.DataFrame) -> str:
        """Lookup regime for a date."""
        try:
            mask = regime_data.index <= dt
            if mask.any():
                return str(regime_data.loc[mask].iloc[-1].get("regime", "NEUTRAL"))
        except Exception:
            pass
        return "NEUTRAL"

    def _calculate_bucket_stats(
        self,
        bucket_trades: pd.DataFrame,
        dimension: DimensionType,
        bucket_name: str,
    ) -> DimensionStats:
        """Calculate statistics for a bucket of trades."""
        n = len(bucket_trades)
        if n == 0:
            return DimensionStats(
                dimension=dimension,
                bucket_name=bucket_name,
                trade_count=0,
                win_count=0,
                loss_count=0,
                win_rate=0.0,
                profit_factor=0.0,
                total_pnl=0.0,
                avg_pnl=0.0,
                avg_winner=0.0,
                avg_loser=0.0,
                expectancy=0.0,
            )

        winners = bucket_trades[bucket_trades["pnl"] > 0]
        losers = bucket_trades[bucket_trades["pnl"] <= 0]

        win_count = len(winners)
        loss_count = len(losers)
        win_rate = win_count / n if n > 0 else 0

        gross_profit = winners["pnl"].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers["pnl"].sum()) if len(losers) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

        total_pnl = bucket_trades["pnl"].sum()
        avg_pnl = total_pnl / n if n > 0 else 0
        avg_winner = winners["pnl"].mean() if len(winners) > 0 else 0
        avg_loser = losers["pnl"].mean() if len(losers) > 0 else 0

        expectancy = (win_rate * avg_winner) + ((1 - win_rate) * avg_loser)

        return DimensionStats(
            dimension=dimension,
            bucket_name=bucket_name,
            trade_count=n,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            expectancy=expectancy,
        )

    def _win_rate(self, trades: pd.DataFrame) -> float:
        """Calculate win rate."""
        if len(trades) == 0:
            return 0.0
        return (trades["pnl"] > 0).sum() / len(trades)

    def _profit_factor(self, trades: pd.DataFrame) -> float:
        """Calculate profit factor."""
        gross_profit = trades[trades["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades[trades["pnl"] <= 0]["pnl"].sum())
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0
        return gross_profit / gross_loss

    def _decompose_by_dow(self, trades: pd.DataFrame) -> Dict[str, DimensionStats]:
        """Decompose by day of week."""
        result = {}
        for dow in range(5):  # Monday to Friday
            day_name = self.DOW_NAMES[dow]
            bucket_trades = trades[trades["dow"] == dow]
            if len(bucket_trades) >= self.min_trades_per_bucket:
                result[day_name] = self._calculate_bucket_stats(
                    bucket_trades, DimensionType.DAY_OF_WEEK, day_name
                )
        return result

    def _decompose_by_volatility(self, trades: pd.DataFrame) -> Dict[str, DimensionStats]:
        """Decompose by VIX bucket."""
        result = {}
        for name, low, high in self.VOL_BUCKETS:
            bucket_trades = trades[(trades["vix"] >= low) & (trades["vix"] < high)]
            if len(bucket_trades) >= self.min_trades_per_bucket:
                result[name] = self._calculate_bucket_stats(
                    bucket_trades, DimensionType.VOLATILITY_BUCKET, name
                )
        return result

    def _decompose_by_regime(self, trades: pd.DataFrame) -> Dict[str, DimensionStats]:
        """Decompose by market regime."""
        result = {}
        for regime in trades["regime"].unique():
            bucket_trades = trades[trades["regime"] == regime]
            if len(bucket_trades) >= self.min_trades_per_bucket:
                result[regime] = self._calculate_bucket_stats(
                    bucket_trades, DimensionType.REGIME, regime
                )
        return result

    def _decompose_by_liquidity(self, trades: pd.DataFrame) -> Dict[str, DimensionStats]:
        """Decompose by liquidity bucket (ADV in $M)."""
        result = {}
        for name, low, high in self.LIQUIDITY_BUCKETS:
            bucket_trades = trades[(trades["adv"] >= low) & (trades["adv"] < high)]
            if len(bucket_trades) >= self.min_trades_per_bucket:
                result[name] = self._calculate_bucket_stats(
                    bucket_trades, DimensionType.LIQUIDITY, name
                )
        return result

    def _decompose_by_gap(self, trades: pd.DataFrame) -> Dict[str, DimensionStats]:
        """Decompose by gap size bucket."""
        result = {}
        for name, low, high in self.GAP_BUCKETS:
            bucket_trades = trades[(trades["gap_pct"] >= low) & (trades["gap_pct"] < high)]
            if len(bucket_trades) >= self.min_trades_per_bucket:
                result[name] = self._calculate_bucket_stats(
                    bucket_trades, DimensionType.GAP_SIZE, name
                )
        return result

    def _decompose_by_column(
        self,
        trades: pd.DataFrame,
        column: str,
        dimension: DimensionType,
    ) -> Dict[str, DimensionStats]:
        """Decompose by a categorical column."""
        if column not in trades.columns:
            return {}

        result = {}
        for value in trades[column].unique():
            if pd.isna(value):
                continue
            bucket_trades = trades[trades[column] == value]
            if len(bucket_trades) >= self.min_trades_per_bucket:
                result[str(value)] = self._calculate_bucket_stats(
                    bucket_trades, dimension, str(value)
                )
        return result

    def _decompose_by_month(self, trades: pd.DataFrame) -> Dict[str, DimensionStats]:
        """Decompose by month."""
        return self._decompose_by_column(trades, "month", DimensionType.MONTH)

    def _decompose_by_hold_time(self, trades: pd.DataFrame) -> Dict[str, DimensionStats]:
        """Decompose by hold time bucket."""
        result = {}
        for name, low, high in self.HOLD_BUCKETS:
            bucket_trades = trades[(trades["hold_days"] >= low) & (trades["hold_days"] < high)]
            if len(bucket_trades) >= self.min_trades_per_bucket:
                result[name] = self._calculate_bucket_stats(
                    bucket_trades, DimensionType.HOLD_TIME, name
                )
        return result

    def _find_weak_spots(self, result: DecompositionResult) -> List[Tuple[DimensionType, str, float]]:
        """Find buckets with weak edge."""
        weak = []

        for dim_type, stats_dict in [
            (DimensionType.DAY_OF_WEEK, result.by_day_of_week),
            (DimensionType.VOLATILITY_BUCKET, result.by_volatility),
            (DimensionType.REGIME, result.by_regime),
            (DimensionType.SECTOR, result.by_sector),
            (DimensionType.STRATEGY, result.by_strategy),
        ]:
            for bucket_name, stats in stats_dict.items():
                if stats.trade_count >= self.min_trades_per_bucket:
                    if stats.edge_score < self.weak_edge_threshold:
                        weak.append((dim_type, bucket_name, stats.edge_score))

        # Sort by edge score (worst first)
        weak.sort(key=lambda x: x[2])
        return weak

    def _find_strong_spots(self, result: DecompositionResult) -> List[Tuple[DimensionType, str, float]]:
        """Find buckets with strong edge."""
        strong = []

        for dim_type, stats_dict in [
            (DimensionType.DAY_OF_WEEK, result.by_day_of_week),
            (DimensionType.VOLATILITY_BUCKET, result.by_volatility),
            (DimensionType.REGIME, result.by_regime),
            (DimensionType.SECTOR, result.by_sector),
            (DimensionType.STRATEGY, result.by_strategy),
        ]:
            for bucket_name, stats in stats_dict.items():
                if stats.trade_count >= self.min_trades_per_bucket:
                    if stats.edge_score > self.strong_edge_threshold:
                        strong.append((dim_type, bucket_name, stats.edge_score))

        # Sort by edge score (best first)
        strong.sort(key=lambda x: x[2], reverse=True)
        return strong

    def detect_degradation(
        self,
        recent_trades: pd.DataFrame,
        historical_trades: pd.DataFrame,
        degradation_threshold: float = 0.15,  # 15% worse is degradation
    ) -> List[Tuple[DimensionType, str, float, float]]:
        """
        Detect dimensions where edge has degraded.

        Args:
            recent_trades: Recent trades (e.g., last 30 days)
            historical_trades: Historical trades (baseline)
            degradation_threshold: Relative decrease to flag as degradation

        Returns:
            List of (dimension, bucket, recent_score, historical_score) tuples
        """
        if len(recent_trades) < self.min_trades_per_bucket:
            return []

        recent_result = self.analyze(recent_trades)
        historical_result = self.analyze(historical_trades)

        degradations = []

        for dim_type, recent_stats, historical_stats in [
            (DimensionType.DAY_OF_WEEK, recent_result.by_day_of_week, historical_result.by_day_of_week),
            (DimensionType.VOLATILITY_BUCKET, recent_result.by_volatility, historical_result.by_volatility),
            (DimensionType.REGIME, recent_result.by_regime, historical_result.by_regime),
            (DimensionType.STRATEGY, recent_result.by_strategy, historical_result.by_strategy),
        ]:
            for bucket_name in recent_stats:
                if bucket_name not in historical_stats:
                    continue

                recent_score = recent_stats[bucket_name].edge_score
                historical_score = historical_stats[bucket_name].edge_score

                if historical_score > 0:
                    relative_change = (historical_score - recent_score) / historical_score
                    if relative_change > degradation_threshold:
                        degradations.append((dim_type, bucket_name, recent_score, historical_score))

        return degradations


# Default instance
DEFAULT_EDGE_DECOMPOSITION = EdgeDecomposition()
