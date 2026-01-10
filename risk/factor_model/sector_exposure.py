"""
Sector Exposure Analyzer - Sector Concentration Risk

Monitor sector concentrations to avoid hidden correlations.
During market stress, same-sector stocks move together.

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import json

from core.structured_log import get_logger

logger = get_logger(__name__)


# GICS Sectors
SECTORS = [
    "Technology",
    "Healthcare",
    "Financials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Industrials",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
    "Communication Services",
    "Other",
]

# SPY sector weights (approximate for benchmarking)
SPY_SECTOR_WEIGHTS = {
    "Technology": 0.30,
    "Healthcare": 0.13,
    "Financials": 0.12,
    "Consumer Discretionary": 0.11,
    "Communication Services": 0.09,
    "Industrials": 0.08,
    "Consumer Staples": 0.06,
    "Energy": 0.04,
    "Utilities": 0.02,
    "Real Estate": 0.02,
    "Materials": 0.02,
    "Other": 0.01,
}

# Sector ETF proxies
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
    "Communication Services": "XLC",
}


@dataclass
class SectorExposures:
    """Sector exposure analysis."""
    weights: Dict[str, float]           # Sector -> weight
    vs_benchmark: Dict[str, float]      # Sector -> over/underweight vs SPY
    max_concentration: float            # Highest sector weight
    max_sector: str                     # Name of most concentrated sector
    overweight_sectors: List[str]       # Sectors > 5% overweight
    underweight_sectors: List[str]      # Sectors > 5% underweight
    concentration_score: float          # 0-100, higher = more concentrated
    is_balanced: bool                   # True if no sector > 30%
    as_of: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "vs_benchmark": self.vs_benchmark,
            "max_concentration": self.max_concentration,
            "max_sector": self.max_sector,
            "overweight_sectors": self.overweight_sectors,
            "underweight_sectors": self.underweight_sectors,
            "concentration_score": self.concentration_score,
            "is_balanced": self.is_balanced,
            "as_of": self.as_of.isoformat(),
        }

    def get_risk_level(self) -> str:
        """Get overall sector risk level."""
        if self.max_concentration > 0.50:
            return "CRITICAL"
        elif self.max_concentration > 0.40:
            return "HIGH"
        elif self.max_concentration > 0.30:
            return "ELEVATED"
        elif self.max_concentration > 0.20:
            return "MODERATE"
        else:
            return "LOW"

    def to_summary(self) -> str:
        """Generate plain English summary."""
        lines = [
            "**Sector Exposure Analysis**",
            f"Risk Level: {self.get_risk_level()}",
            "",
            "**Sector Weights:**",
        ]

        # Sort by weight
        sorted_sectors = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        for sector, weight in sorted_sectors[:5]:  # Top 5
            vs_spy = self.vs_benchmark.get(sector, 0)
            sign = "+" if vs_spy >= 0 else ""
            lines.append(f"  {sector}: {weight:.1%} ({sign}{vs_spy:.1%} vs SPY)")

        lines.append("")

        if self.overweight_sectors:
            lines.append(f"**Overweight:** {', '.join(self.overweight_sectors)}")

        if self.underweight_sectors:
            lines.append(f"**Underweight:** {', '.join(self.underweight_sectors)}")

        if not self.is_balanced:
            lines.append(f"\n**WARNING:** High concentration in {self.max_sector} ({self.max_concentration:.1%})")

        return "\n".join(lines)


class SectorAnalyzer:
    """
    Analyze sector exposures and concentration risk.

    Features:
    - Sector weight calculation
    - Benchmark comparison (vs SPY)
    - Concentration alerts
    - Hedging suggestions
    """

    STATE_FILE = Path("state/risk/sector_history.json")

    # Alert thresholds
    CONCENTRATION_ALERT = 0.30        # Alert if any sector > 30%
    OVERWEIGHT_THRESHOLD = 0.05       # Consider overweight if > 5% vs benchmark

    # Comprehensive sector mapping
    SECTOR_MAP = {
        # Technology
        "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
        "GOOG": "Technology", "META": "Technology", "NVDA": "Technology",
        "AMD": "Technology", "INTC": "Technology", "CRM": "Technology",
        "ADBE": "Technology", "ORCL": "Technology", "CSCO": "Technology",
        "IBM": "Technology", "TXN": "Technology", "QCOM": "Technology",
        "MU": "Technology", "AMAT": "Technology", "LRCX": "Technology",
        "NOW": "Technology", "INTU": "Technology", "PANW": "Technology",

        # Healthcare
        "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
        "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
        "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare",
        "BMY": "Healthcare", "AMGN": "Healthcare", "GILD": "Healthcare",
        "CVS": "Healthcare", "MDT": "Healthcare", "ISRG": "Healthcare",

        # Financials
        "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
        "GS": "Financials", "MS": "Financials", "C": "Financials",
        "BLK": "Financials", "SCHW": "Financials", "AXP": "Financials",
        "USB": "Financials", "PNC": "Financials", "COF": "Financials",
        "TFC": "Financials", "BK": "Financials", "STT": "Financials",

        # Consumer Discretionary
        "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
        "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
        "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
        "LOW": "Consumer Discretionary", "TGT": "Consumer Discretionary",
        "BKNG": "Consumer Discretionary", "TJX": "Consumer Discretionary",
        "ORLY": "Consumer Discretionary", "MAR": "Consumer Discretionary",

        # Consumer Staples
        "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
        "WMT": "Consumer Staples", "COST": "Consumer Staples", "PM": "Consumer Staples",
        "MO": "Consumer Staples", "MDLZ": "Consumer Staples", "CL": "Consumer Staples",
        "KMB": "Consumer Staples", "GIS": "Consumer Staples", "K": "Consumer Staples",

        # Industrials
        "CAT": "Industrials", "GE": "Industrials", "BA": "Industrials",
        "HON": "Industrials", "UNP": "Industrials", "UPS": "Industrials",
        "RTX": "Industrials", "DE": "Industrials", "LMT": "Industrials",
        "MMM": "Industrials", "FDX": "Industrials", "GD": "Industrials",

        # Energy
        "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "OXY": "Energy",
        "SLB": "Energy", "EOG": "Energy", "MPC": "Energy", "VLO": "Energy",
        "PSX": "Energy", "PXD": "Energy", "DVN": "Energy", "HAL": "Energy",

        # Utilities
        "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
        "D": "Utilities", "AEP": "Utilities", "EXC": "Utilities",
        "SRE": "Utilities", "XEL": "Utilities", "WEC": "Utilities",

        # Real Estate
        "PLD": "Real Estate", "AMT": "Real Estate", "CCI": "Real Estate",
        "EQIX": "Real Estate", "PSA": "Real Estate", "SPG": "Real Estate",
        "O": "Real Estate", "WELL": "Real Estate", "DLR": "Real Estate",

        # Communication Services
        "T": "Communication Services", "VZ": "Communication Services",
        "TMUS": "Communication Services", "CMCSA": "Communication Services",
        "DIS": "Communication Services", "NFLX": "Communication Services",
        "CHTR": "Communication Services", "EA": "Communication Services",

        # Materials
        "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
        "ECL": "Materials", "FCX": "Materials", "NEM": "Materials",
        "NUE": "Materials", "DD": "Materials", "DOW": "Materials",
    }

    def __init__(self):
        """Initialize sector analyzer."""
        self._exposure_history: List[Dict] = []

        # Ensure directory exists
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        self._load_state()

    def _load_state(self) -> None:
        """Load sector history."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    self._exposure_history = data.get("history", [])
            except Exception as e:
                logger.warning(f"Failed to load sector state: {e}")

    def _save_state(self) -> None:
        """Save sector history."""
        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump({
                    "history": self._exposure_history[-365:],
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save sector state: {e}")

    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self.SECTOR_MAP.get(symbol, "Other")

    def analyze(
        self,
        positions: List[Dict],
    ) -> SectorExposures:
        """
        Analyze sector exposures for a portfolio.

        Args:
            positions: List of position dicts with symbol, shares, current_price

        Returns:
            SectorExposures
        """
        # Initialize sector weights
        sector_weights = {s: 0.0 for s in SECTORS}

        # Calculate position values
        total_value = 0.0
        for pos in positions:
            value = pos.get("shares", 0) * pos.get("current_price", 0)
            pos["value"] = value
            total_value += value

        if total_value <= 0:
            total_value = 1.0

        # Calculate sector weights
        for pos in positions:
            sector = self._get_sector(pos.get("symbol", ""))
            weight = pos["value"] / total_value
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        # Compare to benchmark
        vs_benchmark = {}
        for sector, weight in sector_weights.items():
            spy_weight = SPY_SECTOR_WEIGHTS.get(sector, 0)
            vs_benchmark[sector] = weight - spy_weight

        # Find concentration
        max_sector = max(sector_weights, key=sector_weights.get)
        max_concentration = sector_weights[max_sector]

        # Identify over/underweight
        overweight = [s for s, diff in vs_benchmark.items() if diff > self.OVERWEIGHT_THRESHOLD]
        underweight = [s for s, diff in vs_benchmark.items() if diff < -self.OVERWEIGHT_THRESHOLD]

        # Calculate concentration score (HHI-based)
        hhi = sum(w ** 2 for w in sector_weights.values())
        concentration_score = min(100, hhi * 100 / 0.10)  # Normalize

        is_balanced = max_concentration <= self.CONCENTRATION_ALERT

        exposures = SectorExposures(
            weights=sector_weights,
            vs_benchmark=vs_benchmark,
            max_concentration=max_concentration,
            max_sector=max_sector,
            overweight_sectors=overweight,
            underweight_sectors=underweight,
            concentration_score=concentration_score,
            is_balanced=is_balanced,
            as_of=datetime.now(),
        )

        # Save to history
        self._exposure_history.append(exposures.to_dict())
        self._save_state()

        # Alert if concentrated
        if not is_balanced:
            logger.warning(
                f"High sector concentration: {max_sector} at {max_concentration:.1%}"
            )

        return exposures

    def get_hedge_suggestions(self, exposures: SectorExposures) -> List[str]:
        """
        Suggest hedges for sector overweights.

        Args:
            exposures: Current sector exposures

        Returns:
            List of hedge suggestions
        """
        suggestions = []

        for sector in exposures.overweight_sectors:
            overweight = exposures.vs_benchmark.get(sector, 0)
            etf = SECTOR_ETFS.get(sector)

            if etf and overweight > 0.10:
                suggestions.append(
                    f"Consider shorting {etf} to hedge {sector} "
                    f"({overweight:.1%} overweight)"
                )
            elif overweight > 0.05:
                suggestions.append(
                    f"Monitor {sector} exposure ({overweight:.1%} overweight)"
                )

        if exposures.max_concentration > 0.40:
            suggestions.append(
                f"CRITICAL: {exposures.max_sector} at {exposures.max_concentration:.1%}. "
                "Consider reducing positions or hedging."
            )

        return suggestions


if __name__ == "__main__":
    # Demo
    analyzer = SectorAnalyzer()

    print("=== Sector Exposure Demo ===\n")

    positions = [
        {"symbol": "AAPL", "shares": 100, "current_price": 175},
        {"symbol": "MSFT", "shares": 50, "current_price": 380},
        {"symbol": "NVDA", "shares": 30, "current_price": 480},
        {"symbol": "AMD", "shares": 80, "current_price": 120},
        {"symbol": "JPM", "shares": 40, "current_price": 190},
    ]

    exposures = analyzer.analyze(positions)
    print(exposures.to_summary())

    print("\n--- Hedge Suggestions ---")
    for suggestion in analyzer.get_hedge_suggestions(exposures):
        print(f"  - {suggestion}")
