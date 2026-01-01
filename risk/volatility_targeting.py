"""
Volatility Targeting Gate for Kobe Trading System.

Scales position sizes to maintain a target portfolio volatility (e.g., 15% annualized).
This is a SAFETY-CRITICAL component for real money trading.

When market volatility spikes, positions are scaled DOWN to maintain consistent risk.
When volatility drops, positions can be scaled UP (within limits).

Key Formula:
    scale_factor = target_vol / realized_vol
    adjusted_shares = base_shares * scale_factor * regime_adjustment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import yaml

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VolatilityTargetConfig:
    """Configuration for volatility targeting."""
    target_annual_vol: float = 0.15       # 15% annualized target
    vol_lookback_days: int = 21           # Realized vol calculation window
    max_leverage: float = 1.5             # Max scale-up factor
    min_scale: float = 0.25               # Min scale-down factor
    vol_update_frequency: str = "daily"   # daily | intraday
    use_ewm: bool = True                  # Use exponentially weighted vol
    ewm_halflife: int = 10                # Half-life for EWM in days
    vix_override_threshold: float = 30.0  # If VIX > 30, force min_scale
    regime_bearish_scale: float = 0.5     # Scale in bearish regime
    regime_neutral_scale: float = 0.75    # Scale in neutral regime
    regime_bullish_scale: float = 1.0     # Scale in bullish regime


@dataclass
class VolatilityTargetResult:
    """Result of volatility targeting calculation."""
    target_vol: float
    realized_vol: float
    vol_ratio: float                      # target / realized
    base_scale_factor: float              # Clamped vol ratio
    regime: str                           # BULLISH, NEUTRAL, BEARISH
    regime_adjustment: float              # Regime-based multiplier
    vix_level: Optional[float]            # Current VIX if available
    vix_override: bool                    # True if VIX forced min scale
    final_scale_factor: float             # Final position multiplier
    original_shares: int
    adjusted_shares: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_vol": round(self.target_vol, 4),
            "realized_vol": round(self.realized_vol, 4),
            "vol_ratio": round(self.vol_ratio, 4),
            "base_scale_factor": round(self.base_scale_factor, 4),
            "regime": self.regime,
            "regime_adjustment": round(self.regime_adjustment, 4),
            "vix_level": round(self.vix_level, 2) if self.vix_level else None,
            "vix_override": self.vix_override,
            "final_scale_factor": round(self.final_scale_factor, 4),
            "original_shares": self.original_shares,
            "adjusted_shares": self.adjusted_shares,
            "timestamp": self.timestamp.isoformat(),
        }


class VolatilityTargetingGate:
    """
    Scales position sizes to target a specific portfolio volatility.

    This gate ADJUSTS position sizes (doesn't block) to maintain consistent risk
    across different volatility regimes.

    Usage:
        gate = VolatilityTargetingGate.from_config()
        result = gate.adjust_position_size(
            base_shares=100,
            entry_price=150.0,
            spy_data=spy_df,
            regime="BULLISH"
        )
        adjusted_shares = result.adjusted_shares
    """

    def __init__(
        self,
        config: Optional[VolatilityTargetConfig] = None,
        enabled: bool = True,
    ):
        self.config = config or VolatilityTargetConfig()
        self.enabled = enabled
        self._vol_cache: Dict[str, float] = {}
        self._last_update: Optional[datetime] = None
        self._last_result: Optional[VolatilityTargetResult] = None

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> 'VolatilityTargetingGate':
        """Create VolatilityTargetingGate from config file."""
        if config_path is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "base.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Config not found at {config_path}, using defaults")
            return cls()

        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)

            vol_cfg = cfg.get('volatility_targeting', {})
            enabled = vol_cfg.get('enabled', True)

            config = VolatilityTargetConfig(
                target_annual_vol=vol_cfg.get('target_annual_vol', 0.15),
                vol_lookback_days=vol_cfg.get('vol_lookback_days', 21),
                max_leverage=vol_cfg.get('max_leverage', 1.5),
                min_scale=vol_cfg.get('min_scale', 0.25),
                use_ewm=vol_cfg.get('use_ewm', True),
                ewm_halflife=vol_cfg.get('ewm_halflife', 10),
                vix_override_threshold=vol_cfg.get('vix_override_threshold', 30.0),
                regime_bearish_scale=vol_cfg.get('regime_bearish_scale', 0.5),
                regime_neutral_scale=vol_cfg.get('regime_neutral_scale', 0.75),
                regime_bullish_scale=vol_cfg.get('regime_bullish_scale', 1.0),
            )

            return cls(config=config, enabled=enabled)

        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            return cls()

    def calculate_realized_vol(
        self,
        prices: pd.Series,
        method: str = "close_to_close",
    ) -> float:
        """
        Calculate annualized realized volatility.

        Args:
            prices: Price series (e.g., SPY close prices)
            method: Calculation method
                - "close_to_close": Standard close-to-close returns
                - "parkinson": High-low based (if OHLC available)
                - "yang_zhang": Open/high/low/close (most accurate)

        Returns:
            Annualized volatility as decimal (e.g., 0.15 for 15%)
        """
        if len(prices) < 5:
            logger.warning("Insufficient price data, using default vol 0.20")
            return 0.20

        # Use last N days based on config
        prices = prices.tail(self.config.vol_lookback_days + 1)

        # Calculate returns
        returns = prices.pct_change().dropna()

        if len(returns) < 3:
            logger.warning("Insufficient returns, using default vol 0.20")
            return 0.20

        if self.config.use_ewm:
            # Exponentially weighted volatility
            vol_daily = returns.ewm(halflife=self.config.ewm_halflife).std().iloc[-1]
        else:
            # Simple rolling volatility
            vol_daily = returns.std()

        # Annualize (assuming 252 trading days)
        vol_annual = vol_daily * np.sqrt(252)

        return float(vol_annual)

    def calculate_realized_vol_ohlc(
        self,
        ohlc_df: pd.DataFrame,
        method: str = "yang_zhang",
    ) -> float:
        """
        Calculate realized volatility using OHLC data (more accurate).

        Args:
            ohlc_df: DataFrame with Open, High, Low, Close columns
            method: "parkinson" or "yang_zhang"

        Returns:
            Annualized volatility
        """
        df = ohlc_df.tail(self.config.vol_lookback_days + 1).copy()

        if len(df) < 5:
            return 0.20

        if method == "parkinson":
            # Parkinson volatility (uses high-low range)
            log_hl = np.log(df['High'] / df['Low'])
            parkinson_var = (1 / (4 * np.log(2))) * (log_hl ** 2).mean()
            vol_daily = np.sqrt(parkinson_var)

        elif method == "yang_zhang":
            # Yang-Zhang volatility (most accurate, uses OHLC)
            log_ho = np.log(df['High'] / df['Open'])
            log_lo = np.log(df['Low'] / df['Open'])
            log_co = np.log(df['Close'] / df['Open'])
            log_oc = np.log(df['Open'] / df['Close'].shift(1))
            log_cc = np.log(df['Close'] / df['Close'].shift(1))

            # Rogers-Satchell volatility
            rs_var = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).mean()

            # Overnight variance
            overnight_var = log_oc.var()

            # Open-to-close variance
            open_close_var = log_co.var()

            # Combine (Yang-Zhang)
            k = 0.34 / (1.34 + (self.config.vol_lookback_days + 1) / (self.config.vol_lookback_days - 1))
            yz_var = overnight_var + k * open_close_var + (1 - k) * rs_var

            vol_daily = np.sqrt(max(0, yz_var))

        else:
            # Fall back to close-to-close
            returns = df['Close'].pct_change().dropna()
            vol_daily = returns.std()

        vol_annual = vol_daily * np.sqrt(252)
        return float(vol_annual)

    def get_scale_factor(
        self,
        spy_data: pd.DataFrame,
        regime: str = "NEUTRAL",
        vix_level: Optional[float] = None,
    ) -> VolatilityTargetResult:
        """
        Calculate position scale factor based on volatility targeting.

        Args:
            spy_data: SPY price data (DataFrame with at least 'Close' column)
            regime: Market regime from HMM detector (BULLISH, NEUTRAL, BEARISH)
            vix_level: Optional current VIX level

        Returns:
            VolatilityTargetResult with scale factor
        """
        # Calculate realized volatility
        if 'High' in spy_data.columns and 'Low' in spy_data.columns:
            realized_vol = self.calculate_realized_vol_ohlc(spy_data)
        else:
            realized_vol = self.calculate_realized_vol(spy_data['Close'])

        # Base scale factor: target / realized
        if realized_vol > 0.001:
            vol_ratio = self.config.target_annual_vol / realized_vol
        else:
            vol_ratio = 1.0

        # Clamp to [min_scale, max_leverage]
        base_scale = np.clip(vol_ratio, self.config.min_scale, self.config.max_leverage)

        # Regime adjustment
        regime_upper = regime.upper()
        if regime_upper == "BEARISH":
            regime_adj = self.config.regime_bearish_scale
        elif regime_upper == "NEUTRAL":
            regime_adj = self.config.regime_neutral_scale
        else:  # BULLISH or unknown
            regime_adj = self.config.regime_bullish_scale

        # VIX override: if VIX > threshold, force minimum scale
        vix_override = False
        if vix_level is not None and vix_level > self.config.vix_override_threshold:
            vix_override = True
            final_scale = self.config.min_scale
            logger.warning(
                f"VIX {vix_level:.1f} > {self.config.vix_override_threshold}, "
                f"forcing min_scale {self.config.min_scale}"
            )
        else:
            final_scale = base_scale * regime_adj
            # Re-clamp after regime adjustment
            final_scale = np.clip(final_scale, self.config.min_scale, self.config.max_leverage)

        result = VolatilityTargetResult(
            target_vol=self.config.target_annual_vol,
            realized_vol=realized_vol,
            vol_ratio=vol_ratio,
            base_scale_factor=base_scale,
            regime=regime_upper,
            regime_adjustment=regime_adj,
            vix_level=vix_level,
            vix_override=vix_override,
            final_scale_factor=final_scale,
            original_shares=0,  # Set in adjust_position_size
            adjusted_shares=0,  # Set in adjust_position_size
        )

        self._last_result = result
        self._last_update = datetime.now()

        return result

    def adjust_position_size(
        self,
        base_shares: int,
        entry_price: float,
        spy_data: pd.DataFrame,
        regime: str = "NEUTRAL",
        vix_level: Optional[float] = None,
        min_shares: int = 1,
    ) -> VolatilityTargetResult:
        """
        Adjust position size based on volatility targeting.

        Args:
            base_shares: Base position size (from equity sizer)
            entry_price: Entry price
            spy_data: SPY price data for vol calculation
            regime: Market regime (BULLISH, NEUTRAL, BEARISH)
            vix_level: Optional VIX level
            min_shares: Minimum shares to return

        Returns:
            VolatilityTargetResult with adjusted shares
        """
        if not self.enabled:
            return VolatilityTargetResult(
                target_vol=self.config.target_annual_vol,
                realized_vol=0.0,
                vol_ratio=1.0,
                base_scale_factor=1.0,
                regime=regime,
                regime_adjustment=1.0,
                vix_level=vix_level,
                vix_override=False,
                final_scale_factor=1.0,
                original_shares=base_shares,
                adjusted_shares=base_shares,
            )

        # Get scale factor
        result = self.get_scale_factor(spy_data, regime, vix_level)

        # Calculate adjusted shares
        adjusted = int(base_shares * result.final_scale_factor)
        adjusted = max(min_shares, adjusted)

        # Update result with share info
        result.original_shares = base_shares
        result.adjusted_shares = adjusted

        logger.info(
            f"Vol targeting: {base_shares} -> {adjusted} shares "
            f"(scale={result.final_scale_factor:.2f}, vol={result.realized_vol:.1%}, "
            f"regime={result.regime})"
        )

        self._last_result = result
        return result

    def get_status(self) -> Dict[str, Any]:
        """Get current gate status."""
        return {
            "enabled": self.enabled,
            "config": {
                "target_annual_vol": self.config.target_annual_vol,
                "vol_lookback_days": self.config.vol_lookback_days,
                "max_leverage": self.config.max_leverage,
                "min_scale": self.config.min_scale,
                "vix_override_threshold": self.config.vix_override_threshold,
            },
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "last_result": self._last_result.to_dict() if self._last_result else None,
        }


# Convenience function for integration
def get_volatility_scale_factor(
    spy_prices: pd.Series,
    regime: str = "NEUTRAL",
    vix_level: Optional[float] = None,
    config_path: Optional[str] = None,
) -> float:
    """
    Quick helper to get volatility scale factor.

    Args:
        spy_prices: SPY close prices
        regime: Market regime
        vix_level: Optional VIX level

    Returns:
        Scale factor (0.25 to 1.5)
    """
    gate = VolatilityTargetingGate.from_config(config_path)
    spy_df = pd.DataFrame({'Close': spy_prices})
    result = gate.get_scale_factor(spy_df, regime, vix_level)
    return result.final_scale_factor
