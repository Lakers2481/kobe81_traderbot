"""
Portfolio Governor - Dynamic Hedging & Correlation Control

Mission 2: The 'Why'
-------------------
The bot currently thinks trade-by-trade. To reach the next level, it must think
like a portfolio manager, understanding the risk of its entire book. If all its
positions are highly correlated long positions in tech stocks, it is exposed to
a single point of failure (a tech sector downturn).

This component will manage that macro risk through:
1. Portfolio Beta Calculation - Sensitivity to market moves
2. Dynamic Hedging - Automatic SPY put protection when beta exceeds threshold
3. Dynamic Exposure Multiplier - Risk adjustment based on confidence + regime

Author: Kobe Trading System
Date: 2026-01-07
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HedgeAction(Enum):
    """Types of hedging actions."""
    NONE = "none"
    BUY_SPY_PUTS = "buy_spy_puts"
    INCREASE_HEDGE = "increase_hedge"
    REDUCE_EXPOSURE = "reduce_exposure"


@dataclass
class PortfolioGovernorConfig:
    """Configuration for portfolio governor."""
    # Beta thresholds
    beta_threshold: float = 0.8        # Start hedging above this beta
    beta_critical: float = 1.2         # Aggressive hedging above this

    # Hedge sizing
    hedge_ratio_base: float = 0.3      # Base hedge ratio (30% of excess beta)
    max_hedge_pct: float = 0.20        # Max hedge size as % of portfolio

    # Exposure multipliers
    min_exposure_mult: float = 0.25    # Minimum exposure (25% of normal)
    max_exposure_mult: float = 1.25    # Maximum exposure (125% of normal)
    low_confidence_mult: float = 0.5   # Multiplier when confidence is low
    bearish_regime_mult: float = 0.5   # Multiplier in bear regime
    bullish_regime_mult: float = 1.2   # Multiplier in bull regime

    # Correlation limits
    max_sector_concentration: float = 0.40  # Max 40% in one sector
    max_single_stock_pct: float = 0.20      # Max 20% in one stock


@dataclass
class HedgeRecommendation:
    """Recommendation for portfolio hedge."""
    action: HedgeAction
    instrument: str = "SPY"
    instrument_type: str = "put"
    size_pct: float = 0.0        # As percentage of portfolio
    strike_pct: float = 0.95     # OTM percentage (95% = 5% OTM)
    dte: int = 21                # Days to expiration
    reason: str = ""
    portfolio_beta: float = 0.0
    excess_beta: float = 0.0


@dataclass
class Position:
    """Simple position representation."""
    symbol: str
    shares: int
    current_price: float
    entry_price: float
    beta: float = 1.0
    sector: str = "unknown"

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price


class PortfolioGovernor:
    """
    Portfolio Governor for dynamic risk management.

    This class monitors portfolio-level risk and provides:
    1. Portfolio beta calculation vs SPY
    2. Hedging recommendations when beta exceeds thresholds
    3. Dynamic exposure multipliers based on confidence and regime
    """

    def __init__(self, config: Optional[PortfolioGovernorConfig] = None):
        """Initialize with configuration."""
        self.config = config or PortfolioGovernorConfig()
        self._beta_cache: Dict[str, float] = {}
        self._sector_map: Dict[str, str] = self._load_sector_map()
        logger.info(f"PortfolioGovernor initialized: beta_threshold={self.config.beta_threshold}")

    def _load_sector_map(self) -> Dict[str, str]:
        """Load sector mapping for symbols."""
        # Default sector mapping (can be extended from file)
        return {
            # Tech
            "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
            "NVDA": "technology", "META": "technology", "AMZN": "technology",
            "TSLA": "technology",
            # Finance
            "JPM": "financial", "BAC": "financial", "GS": "financial",
            "MS": "financial", "WFC": "financial",
            # Healthcare
            "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
            "MRK": "healthcare", "ABBV": "healthcare",
            # Energy
            "XOM": "energy", "CVX": "energy", "COP": "energy",
            "OXY": "energy", "SLB": "energy",
            # Consumer
            "WMT": "consumer", "HD": "consumer", "MCD": "consumer",
            "NKE": "consumer", "SBUX": "consumer",
            # Industrial
            "CAT": "industrial", "BA": "industrial", "UPS": "industrial",
            "HON": "industrial", "GE": "industrial",
        }

    def calculate_portfolio_beta(
        self,
        positions: List[Position],
        spy_data: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Calculate weighted-average beta of all open positions relative to SPY.

        Args:
            positions: List of current positions
            spy_data: Optional SPY price data for correlation calculation

        Returns:
            Portfolio beta (1.0 = moves like market)
        """
        if not positions:
            return 0.0

        total_value = sum(p.market_value for p in positions)
        if total_value == 0:
            return 0.0

        weighted_beta = 0.0
        for position in positions:
            weight = position.market_value / total_value
            beta = self._get_beta(position.symbol)
            weighted_beta += weight * beta
            logger.debug(f"{position.symbol}: beta={beta:.2f}, weight={weight:.2%}")

        logger.info(f"Portfolio beta: {weighted_beta:.3f} ({len(positions)} positions)")
        return weighted_beta

    def _get_beta(self, symbol: str) -> float:
        """Get beta for a symbol (cached or estimated)."""
        if symbol in self._beta_cache:
            return self._beta_cache[symbol]

        # Default beta estimates by sector
        sector = self._sector_map.get(symbol, "unknown")
        sector_betas = {
            "technology": 1.3,
            "financial": 1.2,
            "healthcare": 0.8,
            "energy": 1.4,
            "consumer": 0.9,
            "industrial": 1.1,
            "utility": 0.5,
            "unknown": 1.0,
        }

        beta = sector_betas.get(sector, 1.0)
        self._beta_cache[symbol] = beta
        return beta

    def set_beta(self, symbol: str, beta: float) -> None:
        """Manually set beta for a symbol."""
        self._beta_cache[symbol] = beta

    def get_hedge_action(
        self,
        positions: List[Position],
        spy_data: Optional[pd.DataFrame] = None
    ) -> HedgeRecommendation:
        """
        Determine if hedging is needed and recommend action.

        Args:
            positions: Current portfolio positions
            spy_data: Optional SPY price data

        Returns:
            HedgeRecommendation with action and parameters
        """
        portfolio_beta = self.calculate_portfolio_beta(positions, spy_data)
        excess_beta = portfolio_beta - self.config.beta_threshold

        if excess_beta <= 0:
            return HedgeRecommendation(
                action=HedgeAction.NONE,
                reason=f"Portfolio beta {portfolio_beta:.2f} <= threshold {self.config.beta_threshold}",
                portfolio_beta=portfolio_beta,
                excess_beta=0
            )

        # Calculate hedge size proportional to excess beta
        total_value = sum(p.market_value for p in positions)
        hedge_ratio = min(
            self.config.hedge_ratio_base * (excess_beta / 0.2),  # Scale with excess
            self.config.max_hedge_pct
        )

        # Determine action severity
        if portfolio_beta >= self.config.beta_critical:
            action = HedgeAction.BUY_SPY_PUTS
            strike_pct = 0.93  # More aggressive (7% OTM)
            reason = f"CRITICAL: Portfolio beta {portfolio_beta:.2f} >= {self.config.beta_critical}"
        elif excess_beta > 0.2:
            action = HedgeAction.BUY_SPY_PUTS
            strike_pct = 0.95  # Standard (5% OTM)
            reason = f"HIGH: Portfolio beta {portfolio_beta:.2f}, excess {excess_beta:.2f}"
        else:
            action = HedgeAction.REDUCE_EXPOSURE
            strike_pct = 0.97  # Conservative (3% OTM)
            reason = f"MODERATE: Portfolio beta {portfolio_beta:.2f}, consider reducing exposure"

        return HedgeRecommendation(
            action=action,
            instrument="SPY",
            instrument_type="put",
            size_pct=hedge_ratio,
            strike_pct=strike_pct,
            dte=21,
            reason=reason,
            portfolio_beta=portfolio_beta,
            excess_beta=excess_beta
        )

    def get_exposure_multiplier(
        self,
        confidence: Optional[float] = None,
        regime: Optional[str] = None
    ) -> float:
        """
        Calculate dynamic exposure multiplier based on confidence and regime.

        This multiplier adjusts position sizing based on:
        1. Self-model confidence in current strategy
        2. HMM regime detector market state

        Args:
            confidence: Strategy confidence (0.0-1.0) or None to query self_model
            regime: Market regime ("BULLISH", "NEUTRAL", "BEARISH") or None to query HMM

        Returns:
            Exposure multiplier (0.25 to 1.25)
        """
        # Get confidence from self_model if not provided
        if confidence is None:
            try:
                from cognitive.self_model import get_self_model
                self_model = get_self_model()
                confidence = self_model.get_current_confidence()
            except Exception as e:
                logger.warning(f"Could not get self_model confidence: {e}")
                confidence = 0.6  # Default moderate confidence

        # Get regime from HMM if not provided
        if regime is None:
            try:
                from ml_advanced.hmm_regime_detector import get_regime_detector
                detector = get_regime_detector()
                regime_state = detector.get_current_regime()
                regime = regime_state.regime.value if regime_state else "NEUTRAL"
            except Exception as e:
                logger.warning(f"Could not get HMM regime: {e}")
                regime = "NEUTRAL"

        # Calculate multiplier
        multiplier = 1.0

        # Confidence adjustment
        if confidence < 0.4:
            multiplier *= self.config.low_confidence_mult
            logger.info(f"Low confidence ({confidence:.2f}): reducing exposure")
        elif confidence > 0.7:
            multiplier *= 1.1  # Slight boost for high confidence
            logger.info(f"High confidence ({confidence:.2f}): slight exposure boost")

        # Regime adjustment
        regime_upper = regime.upper() if regime else "NEUTRAL"
        if regime_upper == "BEARISH":
            multiplier *= self.config.bearish_regime_mult
            logger.info(f"Bearish regime: reducing exposure to {multiplier:.2f}x")
        elif regime_upper == "BULLISH":
            multiplier *= self.config.bullish_regime_mult
            logger.info(f"Bullish regime: increasing exposure to {multiplier:.2f}x")

        # Clamp to limits
        multiplier = max(self.config.min_exposure_mult, min(multiplier, self.config.max_exposure_mult))

        logger.info(f"Exposure multiplier: {multiplier:.2f} (confidence={confidence:.2f}, regime={regime})")
        return multiplier

    def check_concentration_limits(self, positions: List[Position]) -> Dict[str, Any]:
        """
        Check portfolio concentration limits.

        Returns:
            Dict with concentration analysis and any limit breaches
        """
        if not positions:
            return {"status": "ok", "positions": 0}

        total_value = sum(p.market_value for p in positions)
        if total_value == 0:
            return {"status": "ok", "positions": 0}

        # Single stock concentration
        stock_concentrations = {
            p.symbol: p.market_value / total_value
            for p in positions
        }

        # Sector concentration
        sector_values: Dict[str, float] = {}
        for p in positions:
            sector = self._sector_map.get(p.symbol, "unknown")
            sector_values[sector] = sector_values.get(sector, 0) + p.market_value

        sector_concentrations = {
            sector: value / total_value
            for sector, value in sector_values.items()
        }

        # Check limits
        breaches = []

        # Stock limits
        for symbol, conc in stock_concentrations.items():
            if conc > self.config.max_single_stock_pct:
                breaches.append({
                    "type": "single_stock",
                    "symbol": symbol,
                    "concentration": conc,
                    "limit": self.config.max_single_stock_pct
                })

        # Sector limits
        for sector, conc in sector_concentrations.items():
            if conc > self.config.max_sector_concentration:
                breaches.append({
                    "type": "sector",
                    "sector": sector,
                    "concentration": conc,
                    "limit": self.config.max_sector_concentration
                })

        return {
            "status": "breach" if breaches else "ok",
            "positions": len(positions),
            "total_value": total_value,
            "stock_concentrations": stock_concentrations,
            "sector_concentrations": sector_concentrations,
            "breaches": breaches
        }

    def get_status(self) -> Dict[str, Any]:
        """Return current configuration status."""
        return {
            "beta_threshold": self.config.beta_threshold,
            "beta_critical": self.config.beta_critical,
            "hedge_ratio_base": self.config.hedge_ratio_base,
            "max_hedge_pct": self.config.max_hedge_pct,
            "min_exposure_mult": self.config.min_exposure_mult,
            "max_exposure_mult": self.config.max_exposure_mult,
            "max_sector_concentration": self.config.max_sector_concentration,
            "max_single_stock_pct": self.config.max_single_stock_pct,
            "cached_betas": len(self._beta_cache),
        }


# Singleton instance
_portfolio_governor: Optional[PortfolioGovernor] = None


def get_portfolio_governor(config: Optional[PortfolioGovernorConfig] = None) -> PortfolioGovernor:
    """Get or create the singleton portfolio governor."""
    global _portfolio_governor
    if _portfolio_governor is None:
        _portfolio_governor = PortfolioGovernor(config)
    return _portfolio_governor


# Example usage
if __name__ == "__main__":
    # Demo with sample positions
    positions = [
        Position("AAPL", 50, 180.0, 175.0, beta=1.2, sector="technology"),
        Position("MSFT", 30, 370.0, 365.0, beta=1.1, sector="technology"),
        Position("JPM", 20, 150.0, 145.0, beta=1.3, sector="financial"),
        Position("JNJ", 40, 160.0, 158.0, beta=0.7, sector="healthcare"),
    ]

    governor = get_portfolio_governor()

    # Calculate portfolio beta
    beta = governor.calculate_portfolio_beta(positions)
    print(f"Portfolio Beta: {beta:.3f}")

    # Get hedge recommendation
    hedge = governor.get_hedge_action(positions)
    print(f"Hedge Action: {hedge.action.value}")
    print(f"Reason: {hedge.reason}")
    if hedge.action != HedgeAction.NONE:
        print(f"Hedge Size: {hedge.size_pct:.1%} of portfolio")
        print(f"Strike: {hedge.strike_pct:.0%} of spot")

    # Get exposure multiplier
    mult = governor.get_exposure_multiplier(confidence=0.65, regime="NEUTRAL")
    print(f"Exposure Multiplier: {mult:.2f}x")

    # Check concentrations
    conc = governor.check_concentration_limits(positions)
    print(f"Concentration Status: {conc['status']}")
    print(f"Sector Breakdown: {conc['sector_concentrations']}")
