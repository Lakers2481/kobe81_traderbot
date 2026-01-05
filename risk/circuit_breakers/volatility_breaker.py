"""
Volatility Circuit Breaker - VIX-Based Protection

Protects capital during high-volatility market regimes by reducing
position sizes or halting trading when VIX exceeds thresholds.

Thresholds:
- VIX > 40: HALT_ALL (extreme fear)
- VIX > 30: PAUSE_NEW (high fear)
- VIX > 25: REDUCE_SIZE (elevated concern)
- VIX > 20: ALERT_ONLY (watch mode)

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path
import json

from core.structured_log import get_logger
from .breaker_manager import BreakerAction, BreakerStatus

logger = get_logger(__name__)


@dataclass
class VolatilityThresholds:
    """VIX thresholds for circuit breaker."""
    # VIX levels (absolute values, not percentages)
    halt_threshold: float = 40.0      # HALT_ALL - extreme fear
    pause_threshold: float = 30.0     # PAUSE_NEW - high fear
    reduce_threshold: float = 25.0    # REDUCE_SIZE - elevated
    alert_threshold: float = 20.0     # ALERT_ONLY - watch

    # VIX spike detection (percentage change)
    spike_1h_threshold: float = 0.15  # 15% spike in 1 hour
    spike_1d_threshold: float = 0.25  # 25% spike in 1 day


class VolatilityBreaker:
    """
    Circuit breaker that monitors VIX and market volatility.

    Solo Trader Features:
    - Real-time VIX monitoring
    - Automatic position reduction during fear regimes
    - Spike detection for sudden volatility events
    - Historical VIX tracking for context
    """

    VIX_CACHE_FILE = Path("state/circuit_breakers/vix_history.json")
    VIX_CACHE_TTL = 60  # seconds

    def __init__(self, thresholds: Optional[VolatilityThresholds] = None):
        """
        Initialize volatility breaker.

        Args:
            thresholds: Custom thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or VolatilityThresholds()
        self._vix_history: list = []
        self._last_fetch: Optional[datetime] = None
        self._cached_vix: Optional[float] = None

        # Ensure cache directory exists
        self.VIX_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load historical VIX data
        self._load_vix_history()

    def _load_vix_history(self) -> None:
        """Load historical VIX from cache."""
        if self.VIX_CACHE_FILE.exists():
            try:
                with open(self.VIX_CACHE_FILE, "r") as f:
                    data = json.load(f)
                    self._vix_history = data.get("history", [])[-100:]  # Keep last 100
            except Exception as e:
                logger.warning(f"Failed to load VIX history: {e}")

    def _save_vix_history(self) -> None:
        """Save VIX history to cache."""
        try:
            with open(self.VIX_CACHE_FILE, "w") as f:
                json.dump({"history": self._vix_history[-100:]}, f)
        except Exception as e:
            logger.warning(f"Failed to save VIX history: {e}")

    def _get_current_vix(self) -> Optional[float]:
        """
        Get current VIX level from data provider.

        Returns:
            Current VIX value or None if unavailable
        """
        # Use cached value if fresh enough
        if (
            self._cached_vix is not None
            and self._last_fetch
            and (datetime.now() - self._last_fetch).total_seconds() < self.VIX_CACHE_TTL
        ):
            return self._cached_vix

        try:
            # Try FRED provider (has VIX as VIXCLS)
            from data.providers.fred_macro import get_fred_provider

            provider = get_fred_provider()
            df = provider.get_series("VIXCLS")

            if not df.empty:
                vix = float(df["value"].iloc[-1])
                self._cached_vix = vix
                self._last_fetch = datetime.now()

                # Add to history
                self._vix_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "value": vix,
                })
                self._save_vix_history()

                return vix

        except Exception as e:
            logger.debug(f"Failed to fetch VIX from FRED: {e}")

        try:
            # Fallback: Try to get from Polygon
            from data.providers.polygon_eod import PolygonEODProvider

            provider = PolygonEODProvider()
            df = provider.get("VIX", limit=1)

            if df is not None and not df.empty:
                vix = float(df["close"].iloc[-1])
                self._cached_vix = vix
                self._last_fetch = datetime.now()
                return vix

        except Exception as e:
            logger.debug(f"Failed to fetch VIX from Polygon: {e}")

        return self._cached_vix  # Return stale cache if available

    def _calculate_vix_spike(self) -> Dict[str, float]:
        """
        Calculate VIX spike percentages.

        Returns:
            Dict with 1h and 1d spike percentages
        """
        if len(self._vix_history) < 2:
            return {"spike_1h": 0.0, "spike_1d": 0.0}

        now = datetime.now()
        current_vix = self._vix_history[-1]["value"]

        spike_1h = 0.0
        spike_1d = 0.0

        # Find 1-hour and 1-day ago values
        for entry in reversed(self._vix_history[:-1]):
            entry_time = datetime.fromisoformat(entry["timestamp"])
            age = (now - entry_time).total_seconds() / 3600  # hours

            if age >= 1 and spike_1h == 0:
                spike_1h = (current_vix - entry["value"]) / entry["value"]

            if age >= 24:
                spike_1d = (current_vix - entry["value"]) / entry["value"]
                break

        return {"spike_1h": spike_1h, "spike_1d": spike_1d}

    def check(
        self,
        vix_level: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Check volatility levels against thresholds.

        Args:
            vix_level: Current VIX level (fetched if not provided)
            **kwargs: Ignored (for compatibility with BreakerManager)

        Returns:
            Dict with status, action, message, and details
        """
        # Get VIX level
        if vix_level is None:
            vix_level = self._get_current_vix()

        if vix_level is None:
            return {
                "status": BreakerStatus.GREEN,
                "action": BreakerAction.CONTINUE,
                "message": "VIX data unavailable - continuing with caution",
                "threshold": 0,
                "current_value": 0,
                "details": {"vix_available": False},
            }

        # Calculate spike
        spikes = self._calculate_vix_spike()

        # Determine status and action based on VIX level
        status = BreakerStatus.GREEN
        action = BreakerAction.CONTINUE
        triggered_by = None

        # Check HALT condition
        if vix_level >= self.thresholds.halt_threshold:
            status = BreakerStatus.RED
            action = BreakerAction.HALT_ALL
            triggered_by = "vix_extreme"
            logger.warning(f"HALT: VIX {vix_level:.1f} >= {self.thresholds.halt_threshold}")

        # Check PAUSE condition
        elif vix_level >= self.thresholds.pause_threshold:
            status = BreakerStatus.RED
            action = BreakerAction.PAUSE_NEW
            triggered_by = "vix_high"
            logger.warning(f"PAUSE: VIX {vix_level:.1f} >= {self.thresholds.pause_threshold}")

        # Check REDUCE condition
        elif vix_level >= self.thresholds.reduce_threshold:
            status = BreakerStatus.YELLOW
            action = BreakerAction.REDUCE_SIZE
            triggered_by = "vix_elevated"
            logger.info(f"REDUCE: VIX {vix_level:.1f} >= {self.thresholds.reduce_threshold}")

        # Check ALERT condition
        elif vix_level >= self.thresholds.alert_threshold:
            status = BreakerStatus.YELLOW
            action = BreakerAction.ALERT_ONLY
            triggered_by = "vix_watch"
            logger.info(f"ALERT: VIX {vix_level:.1f} >= {self.thresholds.alert_threshold}")

        # Check spike conditions (can upgrade severity)
        if spikes["spike_1h"] >= self.thresholds.spike_1h_threshold:
            if action.value == "continue":
                status = BreakerStatus.YELLOW
                action = BreakerAction.REDUCE_SIZE
                triggered_by = "vix_spike_1h"
            logger.warning(f"VIX 1h spike: {spikes['spike_1h']:.1%}")

        if spikes["spike_1d"] >= self.thresholds.spike_1d_threshold:
            if action.value in ["continue", "alert_only"]:
                status = BreakerStatus.YELLOW
                action = BreakerAction.REDUCE_SIZE
                triggered_by = "vix_spike_1d"
            logger.warning(f"VIX 1d spike: {spikes['spike_1d']:.1%}")

        # Build message
        if triggered_by:
            threshold = {
                "vix_extreme": self.thresholds.halt_threshold,
                "vix_high": self.thresholds.pause_threshold,
                "vix_elevated": self.thresholds.reduce_threshold,
                "vix_watch": self.thresholds.alert_threshold,
                "vix_spike_1h": self.thresholds.spike_1h_threshold,
                "vix_spike_1d": self.thresholds.spike_1d_threshold,
            }.get(triggered_by, 0)
            message = f"VIX at {vix_level:.1f} triggered {triggered_by} (threshold: {threshold})"
        else:
            message = f"VIX at {vix_level:.1f} - within normal range"

        return {
            "status": status,
            "action": action,
            "message": message,
            "triggered_by": triggered_by,
            "threshold": self.thresholds.reduce_threshold,  # Main threshold
            "current_value": vix_level,
            "details": {
                "vix_level": vix_level,
                "spike_1h": spikes["spike_1h"],
                "spike_1d": spikes["spike_1d"],
                "thresholds": {
                    "halt": self.thresholds.halt_threshold,
                    "pause": self.thresholds.pause_threshold,
                    "reduce": self.thresholds.reduce_threshold,
                    "alert": self.thresholds.alert_threshold,
                    "spike_1h": self.thresholds.spike_1h_threshold,
                    "spike_1d": self.thresholds.spike_1d_threshold,
                },
            },
        }

    def get_vix_regime(self) -> str:
        """
        Get current VIX regime description.

        Returns:
            Regime string: "LOW", "NORMAL", "ELEVATED", "HIGH", "EXTREME"
        """
        vix = self._get_current_vix()

        if vix is None:
            return "UNKNOWN"
        elif vix < 15:
            return "LOW"
        elif vix < 20:
            return "NORMAL"
        elif vix < 25:
            return "ELEVATED"
        elif vix < 30:
            return "HIGH"
        else:
            return "EXTREME"


if __name__ == "__main__":
    # Demo
    breaker = VolatilityBreaker()

    print("=== Volatility Breaker Demo ===\n")

    # Test scenarios
    scenarios = [
        {"vix": 15.0, "desc": "Low VIX"},
        {"vix": 18.0, "desc": "Normal VIX"},
        {"vix": 22.0, "desc": "Watch mode (VIX > 20)"},
        {"vix": 27.0, "desc": "Reduce size (VIX > 25)"},
        {"vix": 35.0, "desc": "Pause new trades (VIX > 30)"},
        {"vix": 45.0, "desc": "HALT ALL (VIX > 40)"},
    ]

    for s in scenarios:
        result = breaker.check(vix_level=s["vix"])
        print(f"{s['desc']} (VIX = {s['vix']}):")
        print(f"  Status: {result['status'].value}")
        print(f"  Action: {result['action'].value}")
        print(f"  Message: {result['message']}")
        print()

    # Show current regime
    print(f"Current VIX Regime: {breaker.get_vix_regime()}")
