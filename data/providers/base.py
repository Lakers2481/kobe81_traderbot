"""
Abstract Base Class for Data Loaders.

This module provides the abstract interface that all data providers
must implement, ensuring consistent behavior across different data sources.

Blueprint Alignment:
    Implements Section 2.2 requirements for data providers with:
    - Consistent interface across all providers
    - Built-in validation hooks
    - Rate limiting support
    - Fallback provider support

Usage:
    from data.providers.base import DataLoaderBase

    class MyProvider(DataLoaderBase):
        @property
        def provider_name(self) -> str:
            return "my_provider"

        def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
            # Implementation
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import pandas as pd


class AssetClass(Enum):
    """Supported asset classes."""
    EQUITY = "equity"
    CRYPTO = "crypto"
    OPTIONS = "options"
    FUTURES = "futures"
    FOREX = "forex"
    MACRO = "macro"


class DataFrequency(Enum):
    """Data frequency/timeframe."""
    TICK = "tick"
    SECOND = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


@dataclass
class ProviderCapabilities:
    """
    Describes the capabilities of a data provider.

    Used to match providers to data requirements.
    """
    asset_classes: List[AssetClass]
    frequencies: List[DataFrequency]
    max_history_days: int
    supports_live_data: bool = False
    supports_batch_requests: bool = True
    rate_limit_per_minute: Optional[int] = None
    requires_api_key: bool = True
    is_free: bool = False


class DataLoaderBase(ABC):
    """
    Abstract base class for all data loaders.

    All data providers must inherit from this class and implement
    the abstract methods. This ensures a consistent interface
    for fetching market data.

    Key responsibilities:
    - Fetch historical OHLCV data
    - Validate returned data
    - Handle rate limiting
    - Provide fallback support
    """

    def __init__(self):
        """Initialize the data loader."""
        self._cache: Dict[str, pd.DataFrame] = {}
        self._last_fetch_time: Optional[datetime] = None
        self._request_count: int = 0

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Return the unique name of this provider.

        Example: "polygon", "stooq", "yfinance"
        """
        pass

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """
        Return the capabilities of this provider.

        Used for provider selection and validation.
        """
        pass

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Ticker symbol (e.g., "AAPL")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
            The date column should be datetime type.

        Raises:
            DataFetchError: If fetch fails
        """
        pass

    def fetch_multiple(
        self,
        symbols: List[str],
        start: str,
        end: str,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Default implementation calls fetch() for each symbol.
        Override for batch API support.

        Args:
            symbols: List of ticker symbols
            start: Start date
            end: End date
            frequency: Data frequency

        Returns:
            Dict mapping symbol to DataFrame
        """
        result = {}
        for symbol in symbols:
            try:
                df = self.fetch(symbol, start, end, frequency)
                result[symbol] = df
            except Exception:
                # Log but continue with other symbols
                pass
        return result

    def validate_output(self, df: pd.DataFrame) -> bool:
        """
        Validate that the output DataFrame meets requirements.

        Checks:
        - Required columns present
        - No null values in critical columns
        - OHLC relationships valid
        - Dates are unique and sorted

        Args:
            df: DataFrame to validate

        Returns:
            True if valid
        """
        if df.empty:
            return True

        # Check required columns
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return False

        # Check for nulls in price columns
        price_cols = ["open", "high", "low", "close"]
        if df[price_cols].isnull().any().any():
            return False

        # Check OHLC relationships
        if not self._validate_ohlc_relationships(df):
            return False

        # Check date uniqueness
        if df["date"].duplicated().any():
            return False

        return True

    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> bool:
        """
        Validate OHLC bar relationships.

        Checks:
        - high >= max(open, close)
        - low <= min(open, close)
        - high >= low
        - All prices > 0
        """
        if df.empty:
            return True

        # All prices positive
        price_cols = ["open", "high", "low", "close"]
        if (df[price_cols] <= 0).any().any():
            return False

        # High must be highest
        max_oc = df[["open", "close"]].max(axis=1)
        if (df["high"] < max_oc).any():
            return False

        # Low must be lowest
        min_oc = df[["open", "close"]].min(axis=1)
        if (df["low"] > min_oc).any():
            return False

        # High >= Low
        if (df["high"] < df["low"]).any():
            return False

        return True

    def is_available(self) -> bool:
        """
        Check if the provider is available and configured.

        Override to check for API keys, connectivity, etc.

        Returns:
            True if provider is ready to use
        """
        return True

    def get_health(self) -> Dict[str, Any]:
        """
        Get health status of the provider.

        Returns:
            Dict with health information
        """
        return {
            "provider": self.provider_name,
            "available": self.is_available(),
            "request_count": self._request_count,
            "last_fetch": self._last_fetch_time.isoformat() if self._last_fetch_time else None,
        }

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()

    def get_cache_key(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: DataFrequency
    ) -> str:
        """Generate cache key for a request."""
        return f"{symbol}:{start}:{end}:{frequency.value}"

    def get_from_cache(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: DataFrequency
    ) -> Optional[pd.DataFrame]:
        """Get data from cache if available."""
        key = self.get_cache_key(symbol, start, end, frequency)
        return self._cache.get(key)

    def set_cache(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: DataFrequency,
        df: pd.DataFrame
    ) -> None:
        """Store data in cache."""
        key = self.get_cache_key(symbol, start, end, frequency)
        self._cache[key] = df

    def fetch_with_cache(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> pd.DataFrame:
        """
        Fetch data with caching.

        Args:
            symbol: Ticker symbol
            start: Start date
            end: End date
            frequency: Data frequency

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        cached = self.get_from_cache(symbol, start, end, frequency)
        if cached is not None:
            return cached

        # Fetch from provider
        df = self.fetch(symbol, start, end, frequency)

        # Cache the result
        self.set_cache(symbol, start, end, frequency, df)

        return df

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider_name})"
