"""
Multi-Source Data Quorum System

Implements Byzantine fault tolerance for market data by querying multiple sources
and using voting/consensus to detect data corruption or manipulation.

Based on: Codex & Gemini reliability recommendations (2026-01-04)

Sources:
- Polygon.io (PRIMARY - paid, highest quality)
- Stooq (FREE - good coverage, slight delays)
- Yahoo Finance (FREE - fallback, occasional gaps)

Usage:
    from data.quorum import DataQuorum, QuorumResult

    quorum = DataQuorum()
    result = quorum.get_verified_price("AAPL", "2024-01-15")
    if result.consensus_reached:
        print(f"Verified close: ${result.close:.2f}")
    else:
        print(f"CONFLICT: {result.discrepancies}")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources for quorum."""
    POLYGON = "polygon"
    STOOQ = "stooq"
    YFINANCE = "yfinance"
    ALPACA = "alpaca"


class ConsensusMethod(Enum):
    """Methods for reaching consensus."""
    MAJORITY = "majority"      # 2/3 sources agree
    WEIGHTED = "weighted"      # Weight by source reliability
    MEDIAN = "median"          # Use median value
    PRIMARY = "primary"        # Trust primary source unless flagged


@dataclass
class SourcePrice:
    """Price data from a single source."""
    source: DataSource
    symbol: str
    date: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    adjusted_close: Optional[float] = None
    fetch_time_ms: float = 0.0
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return (
            self.error is None
            and self.close is not None
            and self.close > 0
        )


@dataclass
class Discrepancy:
    """A discrepancy between data sources."""
    field: str
    source_a: DataSource
    source_b: DataSource
    value_a: float
    value_b: float
    pct_diff: float
    severity: str  # "minor", "major", "critical"

    def __str__(self) -> str:
        return (
            f"{self.field}: {self.source_a.value}={self.value_a:.2f} vs "
            f"{self.source_b.value}={self.value_b:.2f} ({self.pct_diff:.2%} diff) [{self.severity}]"
        )


@dataclass
class QuorumResult:
    """Result of a quorum vote on price data."""
    symbol: str
    date: str
    consensus_reached: bool
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    adjusted_close: Optional[float] = None

    # Voting details
    sources_queried: List[DataSource] = field(default_factory=list)
    sources_agreed: List[DataSource] = field(default_factory=list)
    sources_failed: List[DataSource] = field(default_factory=list)
    discrepancies: List[Discrepancy] = field(default_factory=list)

    # Metadata
    confidence: float = 0.0  # 0-1
    primary_source: Optional[DataSource] = None
    consensus_method: Optional[ConsensusMethod] = None
    fetch_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "date": self.date,
            "consensus_reached": self.consensus_reached,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "adjusted_close": self.adjusted_close,
            "sources_queried": [s.value for s in self.sources_queried],
            "sources_agreed": [s.value for s in self.sources_agreed],
            "sources_failed": [s.value for s in self.sources_failed],
            "discrepancies": [str(d) for d in self.discrepancies],
            "confidence": self.confidence,
            "fetch_time_ms": self.fetch_time_ms,
        }


class DataSourceAdapter(ABC):
    """Abstract adapter for data sources."""

    @abstractmethod
    def fetch_price(self, symbol: str, date_str: str) -> SourcePrice:
        """Fetch price for a single date."""
        pass

    @abstractmethod
    def fetch_range(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch price range."""
        pass


class PolygonAdapter(DataSourceAdapter):
    """Adapter for Polygon.io data."""

    def __init__(self):
        self.source = DataSource.POLYGON

    def fetch_price(self, symbol: str, date_str: str) -> SourcePrice:
        start_time = datetime.now()
        try:
            from data.providers.polygon_eod import PolygonEODProvider

            provider = PolygonEODProvider()
            df = provider.fetch(symbol, date_str, date_str)

            if df is None or df.empty:
                return SourcePrice(
                    source=self.source,
                    symbol=symbol,
                    date=date_str,
                    error="No data returned",
                )

            row = df.iloc[-1]
            return SourcePrice(
                source=self.source,
                symbol=symbol,
                date=date_str,
                open=float(row.get("open", row.get("Open", 0))),
                high=float(row.get("high", row.get("High", 0))),
                low=float(row.get("low", row.get("Low", 0))),
                close=float(row.get("close", row.get("Close", 0))),
                volume=int(row.get("volume", row.get("Volume", 0))),
                fetch_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

        except Exception as e:
            return SourcePrice(
                source=self.source,
                symbol=symbol,
                date=date_str,
                error=str(e),
                fetch_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

    def fetch_range(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            from data.providers.polygon_eod import PolygonEODProvider

            provider = PolygonEODProvider()
            return provider.fetch(symbol, start, end)
        except Exception:
            return pd.DataFrame()


class StooqAdapter(DataSourceAdapter):
    """Adapter for Stooq data."""

    def __init__(self):
        self.source = DataSource.STOOQ

    def fetch_price(self, symbol: str, date_str: str) -> SourcePrice:
        start_time = datetime.now()
        try:
            from data.providers.stooq_eod import StooqEODProvider

            provider = StooqEODProvider()
            df = provider.fetch(symbol, date_str, date_str)

            if df is None or df.empty:
                return SourcePrice(
                    source=self.source,
                    symbol=symbol,
                    date=date_str,
                    error="No data returned",
                )

            row = df.iloc[-1]
            return SourcePrice(
                source=self.source,
                symbol=symbol,
                date=date_str,
                open=float(row.get("open", row.get("Open", 0))),
                high=float(row.get("high", row.get("High", 0))),
                low=float(row.get("low", row.get("Low", 0))),
                close=float(row.get("close", row.get("Close", 0))),
                volume=int(row.get("volume", row.get("Volume", 0))),
                fetch_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

        except Exception as e:
            return SourcePrice(
                source=self.source,
                symbol=symbol,
                date=date_str,
                error=str(e),
                fetch_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

    def fetch_range(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            from data.providers.stooq_eod import StooqEODProvider

            provider = StooqEODProvider()
            return provider.fetch(symbol, start, end)
        except Exception:
            return pd.DataFrame()


class YFinanceAdapter(DataSourceAdapter):
    """Adapter for Yahoo Finance data."""

    def __init__(self):
        self.source = DataSource.YFINANCE

    def fetch_price(self, symbol: str, date_str: str) -> SourcePrice:
        start_time = datetime.now()
        try:
            from data.providers.yfinance_eod import YFinanceEODProvider

            provider = YFinanceEODProvider()
            df = provider.fetch(symbol, date_str, date_str)

            if df is None or df.empty:
                return SourcePrice(
                    source=self.source,
                    symbol=symbol,
                    date=date_str,
                    error="No data returned",
                )

            row = df.iloc[-1]
            return SourcePrice(
                source=self.source,
                symbol=symbol,
                date=date_str,
                open=float(row.get("open", row.get("Open", 0))),
                high=float(row.get("high", row.get("High", 0))),
                low=float(row.get("low", row.get("Low", 0))),
                close=float(row.get("close", row.get("Close", 0))),
                volume=int(row.get("volume", row.get("Volume", 0))),
                fetch_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

        except Exception as e:
            return SourcePrice(
                source=self.source,
                symbol=symbol,
                date=date_str,
                error=str(e),
                fetch_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

    def fetch_range(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            from data.providers.yfinance_eod import YFinanceEODProvider

            provider = YFinanceEODProvider()
            return provider.fetch(symbol, start, end)
        except Exception:
            return pd.DataFrame()


class DataQuorum:
    """
    Multi-source data quorum for Byzantine fault tolerance.

    Queries multiple data sources and uses voting to detect data
    corruption, manipulation, or corporate action discrepancies.
    """

    # Tolerance thresholds for price differences
    CLOSE_TOLERANCE_PCT = 0.005  # 0.5% - minor discrepancy
    MAJOR_TOLERANCE_PCT = 0.02   # 2% - major discrepancy (possible split)
    CRITICAL_TOLERANCE_PCT = 0.10  # 10% - critical (likely data error)

    # Source reliability weights (0-1)
    SOURCE_WEIGHTS = {
        DataSource.POLYGON: 1.0,   # Primary, highest quality
        DataSource.ALPACA: 0.9,    # High quality, real-time
        DataSource.STOOQ: 0.7,     # Good, but delayed
        DataSource.YFINANCE: 0.6,  # Fallback, occasional issues
    }

    def __init__(
        self,
        sources: Optional[List[DataSource]] = None,
        consensus_method: ConsensusMethod = ConsensusMethod.WEIGHTED,
        min_sources: int = 2,
    ):
        self.consensus_method = consensus_method
        self.min_sources = min_sources

        # Default sources
        if sources is None:
            sources = [DataSource.POLYGON, DataSource.STOOQ, DataSource.YFINANCE]

        self.sources = sources
        self.adapters = self._init_adapters()

    def _init_adapters(self) -> Dict[DataSource, DataSourceAdapter]:
        """Initialize adapters for each source."""
        adapters = {}
        for source in self.sources:
            if source == DataSource.POLYGON:
                adapters[source] = PolygonAdapter()
            elif source == DataSource.STOOQ:
                adapters[source] = StooqAdapter()
            elif source == DataSource.YFINANCE:
                adapters[source] = YFinanceAdapter()
        return adapters

    def get_verified_price(
        self,
        symbol: str,
        date_str: str,
    ) -> QuorumResult:
        """
        Get verified price data using quorum voting.

        Args:
            symbol: Stock symbol
            date_str: Date string (YYYY-MM-DD)

        Returns:
            QuorumResult with consensus data or discrepancies
        """
        start_time = datetime.now()

        # Fetch from all sources
        prices: Dict[DataSource, SourcePrice] = {}
        for source, adapter in self.adapters.items():
            price = adapter.fetch_price(symbol, date_str)
            prices[source] = price

        # Separate valid from failed
        valid_prices = {s: p for s, p in prices.items() if p.is_valid}
        failed_sources = [s for s, p in prices.items() if not p.is_valid]

        # Check minimum sources
        if len(valid_prices) < self.min_sources:
            return QuorumResult(
                symbol=symbol,
                date=date_str,
                consensus_reached=False,
                sources_queried=list(prices.keys()),
                sources_failed=failed_sources,
                confidence=0.0,
                fetch_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

        # Find discrepancies
        discrepancies = self._find_discrepancies(valid_prices)

        # Reach consensus
        consensus = self._reach_consensus(valid_prices, discrepancies)

        # Calculate confidence
        confidence = self._calculate_confidence(valid_prices, discrepancies)

        return QuorumResult(
            symbol=symbol,
            date=date_str,
            consensus_reached=len(discrepancies) == 0 or confidence > 0.7,
            open=consensus.get("open"),
            high=consensus.get("high"),
            low=consensus.get("low"),
            close=consensus.get("close"),
            volume=consensus.get("volume"),
            adjusted_close=consensus.get("adjusted_close"),
            sources_queried=list(prices.keys()),
            sources_agreed=list(valid_prices.keys()),
            sources_failed=failed_sources,
            discrepancies=discrepancies,
            confidence=confidence,
            primary_source=DataSource.POLYGON if DataSource.POLYGON in valid_prices else None,
            consensus_method=self.consensus_method,
            fetch_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
        )

    def _find_discrepancies(
        self,
        prices: Dict[DataSource, SourcePrice],
    ) -> List[Discrepancy]:
        """Find discrepancies between sources."""
        discrepancies = []
        sources = list(prices.keys())

        for i, source_a in enumerate(sources):
            for source_b in sources[i + 1:]:
                price_a = prices[source_a]
                price_b = prices[source_b]

                # Compare each field
                for field in ["open", "high", "low", "close"]:
                    val_a = getattr(price_a, field)
                    val_b = getattr(price_b, field)

                    if val_a is None or val_b is None:
                        continue
                    if val_a == 0 or val_b == 0:
                        continue

                    pct_diff = abs(val_a - val_b) / max(val_a, val_b)

                    if pct_diff > self.CLOSE_TOLERANCE_PCT:
                        severity = "minor"
                        if pct_diff > self.MAJOR_TOLERANCE_PCT:
                            severity = "major"
                        if pct_diff > self.CRITICAL_TOLERANCE_PCT:
                            severity = "critical"

                        discrepancies.append(Discrepancy(
                            field=field,
                            source_a=source_a,
                            source_b=source_b,
                            value_a=val_a,
                            value_b=val_b,
                            pct_diff=pct_diff,
                            severity=severity,
                        ))

        return discrepancies

    def _reach_consensus(
        self,
        prices: Dict[DataSource, SourcePrice],
        discrepancies: List[Discrepancy],
    ) -> Dict[str, Any]:
        """Reach consensus on price values."""

        if self.consensus_method == ConsensusMethod.WEIGHTED:
            return self._weighted_consensus(prices)
        elif self.consensus_method == ConsensusMethod.MEDIAN:
            return self._median_consensus(prices)
        elif self.consensus_method == ConsensusMethod.PRIMARY:
            return self._primary_consensus(prices)
        else:
            return self._weighted_consensus(prices)

    def _weighted_consensus(
        self,
        prices: Dict[DataSource, SourcePrice],
    ) -> Dict[str, Any]:
        """Weighted average based on source reliability."""
        consensus = {}

        for field in ["open", "high", "low", "close"]:
            values = []
            weights = []

            for source, price in prices.items():
                val = getattr(price, field)
                if val is not None and val > 0:
                    values.append(val)
                    weights.append(self.SOURCE_WEIGHTS.get(source, 0.5))

            if values:
                consensus[field] = np.average(values, weights=weights)

        # Volume: sum or max
        volumes = [p.volume for p in prices.values() if p.volume and p.volume > 0]
        if volumes:
            consensus["volume"] = int(np.median(volumes))

        return consensus

    def _median_consensus(
        self,
        prices: Dict[DataSource, SourcePrice],
    ) -> Dict[str, Any]:
        """Use median value across sources."""
        consensus = {}

        for field in ["open", "high", "low", "close"]:
            values = [
                getattr(p, field)
                for p in prices.values()
                if getattr(p, field) is not None and getattr(p, field) > 0
            ]
            if values:
                consensus[field] = float(np.median(values))

        volumes = [p.volume for p in prices.values() if p.volume and p.volume > 0]
        if volumes:
            consensus["volume"] = int(np.median(volumes))

        return consensus

    def _primary_consensus(
        self,
        prices: Dict[DataSource, SourcePrice],
    ) -> Dict[str, Any]:
        """Trust primary source (Polygon) if available."""
        if DataSource.POLYGON in prices:
            p = prices[DataSource.POLYGON]
            return {
                "open": p.open,
                "high": p.high,
                "low": p.low,
                "close": p.close,
                "volume": p.volume,
            }
        return self._weighted_consensus(prices)

    def _calculate_confidence(
        self,
        prices: Dict[DataSource, SourcePrice],
        discrepancies: List[Discrepancy],
    ) -> float:
        """Calculate confidence score (0-1)."""
        if not prices:
            return 0.0

        # Start with base confidence from number of sources
        base_conf = min(len(prices) / 3.0, 1.0)

        # Penalize for discrepancies
        penalty = 0.0
        for d in discrepancies:
            if d.severity == "minor":
                penalty += 0.05
            elif d.severity == "major":
                penalty += 0.15
            elif d.severity == "critical":
                penalty += 0.30

        # Boost for having primary source
        boost = 0.1 if DataSource.POLYGON in prices else 0.0

        confidence = base_conf + boost - penalty
        return max(0.0, min(1.0, confidence))

    def verify_range(
        self,
        symbol: str,
        start: str,
        end: str,
        sample_size: int = 10,
    ) -> List[QuorumResult]:
        """
        Verify data over a date range using sampling.

        Args:
            symbol: Stock symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            sample_size: Number of dates to sample

        Returns:
            List of QuorumResult for sampled dates
        """
        from pandas.tseries.offsets import BDay

        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)

        # Generate business days
        all_dates = pd.date_range(start=start_date, end=end_date, freq=BDay())

        # Sample dates
        if len(all_dates) > sample_size:
            indices = np.linspace(0, len(all_dates) - 1, sample_size, dtype=int)
            sample_dates = all_dates[indices]
        else:
            sample_dates = all_dates

        results = []
        for dt in sample_dates:
            result = self.get_verified_price(symbol, dt.strftime("%Y-%m-%d"))
            results.append(result)
            logger.debug(f"{symbol} {dt.date()}: consensus={result.consensus_reached}, conf={result.confidence:.2f}")

        return results


# ============================================================================
# Convenience Functions
# ============================================================================

_default_quorum: Optional[DataQuorum] = None


def get_quorum() -> DataQuorum:
    """Get or create default quorum instance."""
    global _default_quorum
    if _default_quorum is None:
        _default_quorum = DataQuorum()
    return _default_quorum


def verify_price(symbol: str, date_str: str) -> QuorumResult:
    """Verify a single price using the default quorum."""
    return get_quorum().get_verified_price(symbol, date_str)


def verify_range(
    symbol: str,
    start: str,
    end: str,
    sample_size: int = 10,
) -> List[QuorumResult]:
    """Verify a date range using the default quorum."""
    return get_quorum().verify_range(symbol, start, end, sample_size)


def check_data_integrity(
    symbol: str,
    df: pd.DataFrame,
    sample_size: int = 5,
) -> Tuple[bool, List[Discrepancy]]:
    """
    Check data integrity by comparing DataFrame against multiple sources.

    Args:
        symbol: Stock symbol
        df: DataFrame with OHLCV data to verify
        sample_size: Number of dates to sample

    Returns:
        Tuple of (passed, discrepancies)
    """
    if df is None or df.empty:
        return False, []

    quorum = get_quorum()
    all_discrepancies = []

    # Sample random dates from DataFrame
    if len(df) > sample_size:
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
    else:
        sample_indices = range(len(df))

    for idx in sample_indices:
        row = df.iloc[idx]

        # Get date
        if "timestamp" in df.columns:
            date_str = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d")
        elif "date" in df.columns:
            date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        elif isinstance(df.index, pd.DatetimeIndex):
            date_str = df.index[idx].strftime("%Y-%m-%d")
        else:
            continue

        result = quorum.get_verified_price(symbol, date_str)

        if result.discrepancies:
            all_discrepancies.extend(result.discrepancies)

    # Check for critical discrepancies
    critical = [d for d in all_discrepancies if d.severity == "critical"]

    passed = len(critical) == 0
    return passed, all_discrepancies
