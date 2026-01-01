"""
VIX Monitor for Trading Pause Logic.

Fetches VIX (CBOE Volatility Index) and determines if trading should
be paused due to elevated market volatility.

Usage:
    from core.vix_monitor import VIXMonitor, get_vix_monitor

    monitor = get_vix_monitor()
    should_pause, vix_level, reason = monitor.should_pause_trading()

    if should_pause:
        print(f"Trading paused: {reason}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Global singleton instance
_vix_monitor: Optional["VIXMonitor"] = None


@dataclass
class VIXConfig:
    """Configuration for VIX monitoring."""

    pause_enabled: bool = True
    pause_threshold: float = 30.0
    elevated_threshold: float = 25.0
    extreme_threshold: float = 40.0
    cache_ttl_seconds: int = 3600  # 1 hour cache
    data_source: str = "yfinance"  # "yfinance" or "polygon"
    fallback_vix: float = 20.0  # Default if fetch fails


@dataclass
class VIXReading:
    """A single VIX reading with metadata."""

    level: float
    timestamp: datetime
    source: str
    is_stale: bool = False

    @property
    def age_seconds(self) -> float:
        """Time since this reading was fetched."""
        return (datetime.now() - self.timestamp).total_seconds()

    def is_elevated(self, threshold: float = 25.0) -> bool:
        """Check if VIX is elevated."""
        return self.level >= threshold

    def is_extreme(self, threshold: float = 40.0) -> bool:
        """Check if VIX is at extreme levels."""
        return self.level >= threshold


@dataclass
class VIXMonitor:
    """
    Monitor VIX levels and determine trading pause status.

    Caches VIX readings to avoid excessive API calls.
    Provides graceful degradation if VIX data unavailable.
    """

    config: VIXConfig = field(default_factory=VIXConfig)
    _cache: Optional[VIXReading] = field(default=None, repr=False)
    _last_fetch_attempt: Optional[datetime] = field(default=None, repr=False)
    _consecutive_failures: int = field(default=0, repr=False)

    def fetch_vix(self, force_refresh: bool = False) -> VIXReading:
        """
        Fetch current VIX level.

        Uses caching to minimize API calls. Falls back to default
        value if fetch fails.

        Args:
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            VIXReading with current VIX level
        """
        # Check cache first
        if not force_refresh and self._cache is not None:
            cache_age = self._cache.age_seconds
            if cache_age < self.config.cache_ttl_seconds:
                logger.debug(
                    f"VIX cache hit: {self._cache.level:.2f} "
                    f"(age: {cache_age:.0f}s)"
                )
                return self._cache

        # Mark cache as stale if exists
        if self._cache is not None:
            self._cache.is_stale = True

        # Attempt to fetch fresh VIX
        self._last_fetch_attempt = datetime.now()

        try:
            if self.config.data_source == "yfinance":
                vix_level = self._fetch_from_yfinance()
            elif self.config.data_source == "polygon":
                vix_level = self._fetch_from_polygon()
            else:
                logger.warning(
                    f"Unknown VIX data source: {self.config.data_source}, "
                    f"using fallback"
                )
                vix_level = self.config.fallback_vix

            # Success - reset failure counter
            self._consecutive_failures = 0

            self._cache = VIXReading(
                level=vix_level,
                timestamp=datetime.now(),
                source=self.config.data_source,
                is_stale=False,
            )

            logger.info(f"VIX fetched: {vix_level:.2f} from {self.config.data_source}")
            return self._cache

        except Exception as e:
            self._consecutive_failures += 1
            logger.warning(
                f"VIX fetch failed (attempt {self._consecutive_failures}): {e}"
            )

            # Return stale cache if available
            if self._cache is not None:
                logger.info(
                    f"Using stale VIX cache: {self._cache.level:.2f} "
                    f"(age: {self._cache.age_seconds:.0f}s)"
                )
                return self._cache

            # Ultimate fallback
            logger.warning(
                f"No VIX data available, using fallback: {self.config.fallback_vix}"
            )
            return VIXReading(
                level=self.config.fallback_vix,
                timestamp=datetime.now(),
                source="fallback",
                is_stale=True,
            )

    def _fetch_from_yfinance(self) -> float:
        """Fetch VIX from Yahoo Finance."""
        try:
            import yfinance as yf

            ticker = yf.Ticker("^VIX")
            # Get today's data
            hist = ticker.history(period="1d")

            if hist.empty:
                # Try getting last 5 days
                hist = ticker.history(period="5d")

            if hist.empty:
                raise ValueError("No VIX data available from yfinance")

            # Get most recent close
            vix_level = float(hist["Close"].iloc[-1])

            if vix_level <= 0 or vix_level > 100:
                raise ValueError(f"Invalid VIX level: {vix_level}")

            return vix_level

        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            raise

    def _fetch_from_polygon(self) -> float:
        """Fetch VIX from Polygon.io."""
        try:
            import os

            import requests

            api_key = os.getenv("POLYGON_API_KEY")
            if not api_key:
                raise ValueError("POLYGON_API_KEY not set")

            # VIX ticker on Polygon
            url = f"https://api.polygon.io/v2/aggs/ticker/I:VIX/prev"
            params = {"apiKey": api_key}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            if not results:
                raise ValueError("No VIX data from Polygon")

            vix_level = float(results[0].get("c", 0))

            if vix_level <= 0 or vix_level > 100:
                raise ValueError(f"Invalid VIX level from Polygon: {vix_level}")

            return vix_level

        except ImportError:
            logger.error("requests not installed")
            raise

    def should_pause_trading(self) -> Tuple[bool, float, str]:
        """
        Determine if trading should be paused based on VIX.

        Returns:
            Tuple of (should_pause, vix_level, reason_message)
        """
        if not self.config.pause_enabled:
            return False, 0.0, "VIX pause disabled"

        reading = self.fetch_vix()

        if reading.level >= self.config.extreme_threshold:
            return (
                True,
                reading.level,
                f"VIX EXTREME: {reading.level:.1f} >= {self.config.extreme_threshold} "
                f"(source: {reading.source})",
            )

        if reading.level >= self.config.pause_threshold:
            return (
                True,
                reading.level,
                f"VIX HIGH: {reading.level:.1f} >= {self.config.pause_threshold} "
                f"(source: {reading.source})",
            )

        if reading.level >= self.config.elevated_threshold:
            # Warning but don't pause
            logger.warning(
                f"VIX elevated: {reading.level:.1f} >= {self.config.elevated_threshold}"
            )

        return (
            False,
            reading.level,
            f"VIX OK: {reading.level:.1f} < {self.config.pause_threshold} "
            f"(source: {reading.source})",
        )

    def get_regime_adjustment(self) -> float:
        """
        Get position sizing multiplier based on VIX level.

        Returns:
            Multiplier (0.0 to 1.0) for position sizing
        """
        reading = self.fetch_vix()

        if reading.level >= self.config.extreme_threshold:
            return 0.0  # No trading
        elif reading.level >= self.config.pause_threshold:
            return 0.25  # Quarter size
        elif reading.level >= self.config.elevated_threshold:
            return 0.5  # Half size
        else:
            return 1.0  # Full size

    def get_status(self) -> dict:
        """Get current VIX monitoring status for dashboards."""
        reading = self.fetch_vix()
        should_pause, _, reason = self.should_pause_trading()

        return {
            "vix_level": reading.level,
            "vix_source": reading.source,
            "vix_timestamp": reading.timestamp.isoformat(),
            "vix_age_seconds": reading.age_seconds,
            "vix_is_stale": reading.is_stale,
            "trading_paused": should_pause,
            "pause_reason": reason,
            "pause_threshold": self.config.pause_threshold,
            "elevated_threshold": self.config.elevated_threshold,
            "extreme_threshold": self.config.extreme_threshold,
            "position_multiplier": self.get_regime_adjustment(),
            "consecutive_failures": self._consecutive_failures,
        }


def get_vix_monitor(config: Optional[VIXConfig] = None) -> VIXMonitor:
    """
    Get singleton VIXMonitor instance.

    Args:
        config: Optional config to use (only applied on first call)

    Returns:
        Shared VIXMonitor instance
    """
    global _vix_monitor

    if _vix_monitor is None:
        _vix_monitor = VIXMonitor(config=config or VIXConfig())
        logger.info(
            f"VIX monitor initialized: pause_threshold={_vix_monitor.config.pause_threshold}"
        )

    return _vix_monitor


def reset_vix_monitor() -> None:
    """Reset singleton for testing."""
    global _vix_monitor
    _vix_monitor = None


# Convenience functions
def get_vix_level() -> float:
    """Get current VIX level (convenience function)."""
    return get_vix_monitor().fetch_vix().level


def should_pause_for_vix() -> Tuple[bool, float, str]:
    """Check if trading should pause (convenience function)."""
    return get_vix_monitor().should_pause_trading()
