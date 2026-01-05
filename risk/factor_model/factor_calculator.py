"""
Factor Calculator - Calculate Portfolio Factor Exposures

Calculates exposure to systematic risk factors:
- Market (beta)
- Size (small vs large cap)
- Value (cheap vs expensive)
- Momentum (trending vs reverting)
- Volatility (low vs high vol)
- Quality (profitable vs unprofitable)

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import statistics

import numpy as np
import pandas as pd

from core.structured_log import get_logger

logger = get_logger(__name__)


@dataclass
class FactorExposures:
    """All factor exposures for current portfolio."""
    # Market
    market_beta: float              # SPY beta

    # Style Factors
    size_exposure: float            # Small vs large cap tilt (-1 to +1)
    value_exposure: float           # Value vs growth tilt
    momentum_exposure: float        # Momentum factor loading
    volatility_exposure: float      # Low-vol vs high-vol tilt
    quality_exposure: float         # Quality factor loading

    # Sector Exposure
    sector_weights: Dict[str, float]    # {sector: weight}
    max_sector_concentration: float     # Largest sector weight

    # Risk Metrics
    tracking_error: float           # Deviation from benchmark (estimated)
    active_share: float             # How different from SPY (0-100%)

    # Concentration
    top_5_weight: float             # Weight in top 5 positions
    effective_n: float              # Effective number of positions (1/sum(w^2))

    # Timestamp
    as_of: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_beta": self.market_beta,
            "size_exposure": self.size_exposure,
            "value_exposure": self.value_exposure,
            "momentum_exposure": self.momentum_exposure,
            "volatility_exposure": self.volatility_exposure,
            "quality_exposure": self.quality_exposure,
            "sector_weights": self.sector_weights,
            "max_sector_concentration": self.max_sector_concentration,
            "tracking_error": self.tracking_error,
            "active_share": self.active_share,
            "top_5_weight": self.top_5_weight,
            "effective_n": self.effective_n,
            "as_of": self.as_of.isoformat(),
        }

    def get_risk_flags(self) -> List[str]:
        """Get list of risk flags based on exposures."""
        flags = []

        if self.market_beta > 1.3:
            flags.append(f"HIGH_BETA ({self.market_beta:.2f})")
        elif self.market_beta < 0.5:
            flags.append(f"LOW_BETA ({self.market_beta:.2f})")

        if self.max_sector_concentration > 0.40:
            flags.append(f"SECTOR_CONCENTRATION ({self.max_sector_concentration:.0%})")

        if self.top_5_weight > 0.60:
            flags.append(f"POSITION_CONCENTRATION ({self.top_5_weight:.0%})")

        if abs(self.momentum_exposure) > 0.5:
            direction = "long" if self.momentum_exposure > 0 else "short"
            flags.append(f"MOMENTUM_{direction.upper()} ({abs(self.momentum_exposure):.2f})")

        if self.effective_n < 5:
            flags.append(f"LOW_DIVERSIFICATION (EffN={self.effective_n:.1f})")

        return flags

    def to_summary(self) -> str:
        """Generate plain English summary."""
        flags = self.get_risk_flags()

        lines = [
            "**Portfolio Factor Exposures**",
            "",
            f"Market Beta: {self.market_beta:.2f}",
            f"Size (Small+/Large-): {self.size_exposure:+.2f}",
            f"Value (Value+/Growth-): {self.value_exposure:+.2f}",
            f"Momentum: {self.momentum_exposure:+.2f}",
            f"Volatility (LowVol+/HighVol-): {self.volatility_exposure:+.2f}",
            f"Quality: {self.quality_exposure:+.2f}",
            "",
            f"**Concentration:**",
            f"  Top 5 positions: {self.top_5_weight:.0%}",
            f"  Effective N: {self.effective_n:.1f}",
            f"  Max sector: {self.max_sector_concentration:.0%}",
            "",
        ]

        if flags:
            lines.append("**Risk Flags:**")
            for flag in flags:
                lines.append(f"  - {flag}")
        else:
            lines.append("**Risk Flags:** None")

        return "\n".join(lines)


class FactorCalculator:
    """
    Calculate factor exposures for a portfolio.

    Simplified Barra-style factor model for solo traders.
    Uses fundamental characteristics and ETF proxies.
    """

    STATE_FILE = Path("state/risk/factor_exposures.json")

    # ETF proxies for factor calculation
    FACTOR_ETFS = {
        "market": "SPY",
        "size": "IWM",       # Small cap
        "value": "IWD",       # Value
        "momentum": "MTUM",   # Momentum
        "low_vol": "SPLV",    # Low volatility
        "quality": "QUAL",    # Quality
    }

    # Sector mapping
    SECTOR_MAP = {
        "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
        "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
        "META": "Technology", "NVDA": "Technology", "AMD": "Technology",
        "JPM": "Financials", "BAC": "Financials", "GS": "Financials", "WFC": "Financials",
        "XOM": "Energy", "CVX": "Energy", "OXY": "Energy", "COP": "Energy",
        "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare",
        "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
        "CAT": "Industrials", "GE": "Industrials", "BA": "Industrials",
        "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
        "PLD": "Real Estate", "AMT": "Real Estate", "SPG": "Real Estate",
        "T": "Communication Services", "VZ": "Communication Services",
    }

    # Size classification (market cap in billions)
    SIZE_THRESHOLDS = {
        "large": 100,    # > $100B
        "mid": 10,       # $10B - $100B
        "small": 0,      # < $10B
    }

    def __init__(self):
        """Initialize factor calculator."""
        self._exposure_history: List[Dict] = []
        self._cached_exposures: Optional[FactorExposures] = None

        # Ensure directory exists
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        self._load_state()

    def _load_state(self) -> None:
        """Load exposure history."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    self._exposure_history = data.get("history", [])
            except Exception as e:
                logger.warning(f"Failed to load factor state: {e}")

    def _save_state(self) -> None:
        """Save exposure history."""
        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump({
                    "history": self._exposure_history[-365:],
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save factor state: {e}")

    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self.SECTOR_MAP.get(symbol, "Other")

    def _calculate_beta(
        self,
        portfolio_returns: Optional[pd.Series] = None,
        market_returns: Optional[pd.Series] = None,
    ) -> float:
        """Calculate portfolio beta vs SPY."""
        if portfolio_returns is None or market_returns is None:
            return 1.0  # Default assumption

        try:
            aligned = pd.concat([portfolio_returns, market_returns], axis=1).dropna()

            if len(aligned) < 20:
                return 1.0

            port = aligned.iloc[:, 0]
            mkt = aligned.iloc[:, 1]

            covariance = port.cov(mkt)
            variance = mkt.var()

            if variance == 0:
                return 1.0

            return float(covariance / variance)

        except Exception as e:
            logger.warning(f"Failed to calculate beta: {e}")
            return 1.0

    def _calculate_effective_n(self, weights: List[float]) -> float:
        """Calculate effective number of positions (1/HHI)."""
        if not weights:
            return 0.0

        hhi = sum(w ** 2 for w in weights)
        return 1.0 / hhi if hhi > 0 else 0.0

    def calculate_exposures(
        self,
        positions: List[Dict],
    ) -> FactorExposures:
        """
        Calculate factor exposures for a portfolio.

        Args:
            positions: List of position dicts with:
                - symbol: str
                - shares: int
                - current_price: float
                - market_cap: str (optional: "large", "mid", "small")
                - pe_ratio: float (optional, for value)
                - momentum_score: float (optional, 0-1)
                - volatility: float (optional, annualized)
                - roe: float (optional, for quality)

        Returns:
            FactorExposures
        """
        if not positions:
            return FactorExposures(
                market_beta=0.0,
                size_exposure=0.0,
                value_exposure=0.0,
                momentum_exposure=0.0,
                volatility_exposure=0.0,
                quality_exposure=0.0,
                sector_weights={},
                max_sector_concentration=0.0,
                tracking_error=0.0,
                active_share=0.0,
                top_5_weight=0.0,
                effective_n=0.0,
                as_of=datetime.now(),
            )

        # Calculate position values and weights
        for pos in positions:
            pos["value"] = pos.get("shares", 0) * pos.get("current_price", 0)

        total_value = sum(pos["value"] for pos in positions)

        if total_value <= 0:
            total_value = 1.0

        for pos in positions:
            pos["weight"] = pos["value"] / total_value

        weights = [pos["weight"] for pos in positions]

        # Size exposure (-1 = all large, +1 = all small)
        size_score = 0.0
        for pos in positions:
            cap = pos.get("market_cap", "large")
            if cap == "small":
                size_score += pos["weight"] * 1.0
            elif cap == "mid":
                size_score += pos["weight"] * 0.0
            else:  # large
                size_score += pos["weight"] * -1.0
        size_exposure = size_score

        # Value exposure (low P/E = value = positive)
        value_score = 0.0
        pe_positions = [p for p in positions if p.get("pe_ratio")]
        if pe_positions:
            median_pe = statistics.median([p["pe_ratio"] for p in pe_positions])
            for pos in pe_positions:
                if pos["pe_ratio"] < median_pe:
                    value_score += pos["weight"] * 1.0
                else:
                    value_score += pos["weight"] * -1.0
        value_exposure = value_score

        # Momentum exposure
        momentum_score = 0.0
        for pos in positions:
            mom = pos.get("momentum_score", 0.5)  # 0.5 = neutral
            momentum_score += pos["weight"] * (mom - 0.5) * 2  # Scale to -1 to +1
        momentum_exposure = momentum_score

        # Volatility exposure (low vol = positive)
        vol_score = 0.0
        vol_positions = [p for p in positions if p.get("volatility")]
        if vol_positions:
            median_vol = statistics.median([p["volatility"] for p in vol_positions])
            for pos in vol_positions:
                if pos["volatility"] < median_vol:
                    vol_score += pos["weight"] * 1.0
                else:
                    vol_score += pos["weight"] * -1.0
        volatility_exposure = vol_score

        # Quality exposure (high ROE = quality = positive)
        quality_score = 0.0
        quality_positions = [p for p in positions if p.get("roe")]
        if quality_positions:
            median_roe = statistics.median([p["roe"] for p in quality_positions])
            for pos in quality_positions:
                if pos["roe"] > median_roe:
                    quality_score += pos["weight"] * 1.0
                else:
                    quality_score += pos["weight"] * -1.0
        quality_exposure = quality_score

        # Sector weights
        sector_weights: Dict[str, float] = {}
        for pos in positions:
            sector = self._get_sector(pos.get("symbol", ""))
            sector_weights[sector] = sector_weights.get(sector, 0) + pos["weight"]

        max_sector = max(sector_weights.values()) if sector_weights else 0.0

        # Concentration metrics
        sorted_weights = sorted(weights, reverse=True)
        top_5_weight = sum(sorted_weights[:5]) if len(sorted_weights) >= 5 else sum(sorted_weights)
        effective_n = self._calculate_effective_n(weights)

        # Beta (default to 1.0 without return data)
        market_beta = 1.0

        # Create exposures
        exposures = FactorExposures(
            market_beta=market_beta,
            size_exposure=size_exposure,
            value_exposure=value_exposure,
            momentum_exposure=momentum_exposure,
            volatility_exposure=volatility_exposure,
            quality_exposure=quality_exposure,
            sector_weights=sector_weights,
            max_sector_concentration=max_sector,
            tracking_error=0.0,  # Would need benchmark comparison
            active_share=0.0,    # Would need benchmark comparison
            top_5_weight=top_5_weight,
            effective_n=effective_n,
            as_of=datetime.now(),
        )

        # Cache and save
        self._cached_exposures = exposures
        self._exposure_history.append(exposures.to_dict())
        self._save_state()

        return exposures

    def get_current_exposures(self) -> Optional[FactorExposures]:
        """Get most recently calculated exposures."""
        return self._cached_exposures

    def get_exposure_drift(self, days: int = 30) -> Dict[str, float]:
        """
        Calculate how factor exposures have drifted over time.

        Returns:
            Dict of factor -> drift amount
        """
        if len(self._exposure_history) < 2:
            return {}

        cutoff = datetime.now() - timedelta(days=days)

        old_exposures = [
            e for e in self._exposure_history[:-10]
            if datetime.fromisoformat(e["as_of"]) < cutoff
        ]

        recent_exposures = self._exposure_history[-10:]

        if not old_exposures or not recent_exposures:
            return {}

        drift = {}
        for factor in ["market_beta", "size_exposure", "value_exposure",
                       "momentum_exposure", "volatility_exposure", "quality_exposure"]:
            old_avg = statistics.mean([e.get(factor, 0) for e in old_exposures])
            recent_avg = statistics.mean([e.get(factor, 0) for e in recent_exposures])
            drift[factor] = recent_avg - old_avg

        return drift

    def get_summary(self) -> Dict[str, Any]:
        """Get summary for dashboard."""
        if not self._cached_exposures:
            return {"has_data": False}

        exp = self._cached_exposures

        return {
            "has_data": True,
            "market_beta": exp.market_beta,
            "effective_n": exp.effective_n,
            "max_sector": exp.max_sector_concentration,
            "risk_flags": exp.get_risk_flags(),
            "dominant_factors": {
                "size": "Small Cap" if exp.size_exposure > 0.2 else "Large Cap" if exp.size_exposure < -0.2 else "Neutral",
                "value": "Value" if exp.value_exposure > 0.2 else "Growth" if exp.value_exposure < -0.2 else "Neutral",
                "momentum": "High Mom" if exp.momentum_exposure > 0.2 else "Low Mom" if exp.momentum_exposure < -0.2 else "Neutral",
            },
        }


# Singleton
_calculator: Optional[FactorCalculator] = None


def get_factor_calculator() -> FactorCalculator:
    """Get or create singleton calculator."""
    global _calculator
    if _calculator is None:
        _calculator = FactorCalculator()
    return _calculator


if __name__ == "__main__":
    # Demo
    calculator = FactorCalculator()

    print("=== Factor Calculator Demo ===\n")

    # Sample positions
    positions = [
        {"symbol": "AAPL", "shares": 100, "current_price": 175, "market_cap": "large", "momentum_score": 0.7},
        {"symbol": "MSFT", "shares": 50, "current_price": 380, "market_cap": "large", "momentum_score": 0.6},
        {"symbol": "TSLA", "shares": 40, "current_price": 250, "market_cap": "large", "momentum_score": 0.8},
        {"symbol": "AMD", "shares": 80, "current_price": 120, "market_cap": "large", "momentum_score": 0.75},
        {"symbol": "JPM", "shares": 30, "current_price": 190, "market_cap": "large", "pe_ratio": 12},
    ]

    exposures = calculator.calculate_exposures(positions)
    print(exposures.to_summary())
