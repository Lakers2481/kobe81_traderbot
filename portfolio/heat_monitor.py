"""
Portfolio Heat Monitor - Risk Exposure Tracking.

Monitors portfolio-level risk metrics:
- Position concentration
- Sector exposure
- Correlation risk
- Total capital at risk

Provides alerts when thresholds are exceeded.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HeatLevel(Enum):
    """Portfolio heat level classifications."""
    COLD = "COLD"        # Very low risk - room to add
    NORMAL = "NORMAL"    # Healthy risk levels
    WARM = "WARM"        # Elevated but acceptable
    HOT = "HOT"          # High risk - caution
    OVERHEATED = "OVERHEATED"  # Exceed limits - reduce exposure


@dataclass
class HeatStatus:
    """Portfolio heat status report."""
    timestamp: datetime
    heat_level: HeatLevel
    heat_score: float  # 0-100 (100 = max heat)

    # Position metrics
    total_positions: int = 0
    max_positions: int = 5
    position_utilization: float = 0.0

    # Capital metrics
    total_exposure: float = 0.0
    max_exposure: float = 0.0
    exposure_pct: float = 0.0

    # Concentration metrics
    largest_position_pct: float = 0.0
    top_3_concentration: float = 0.0
    max_single_position_pct: float = 20.0

    # Sector metrics
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    max_sector_pct: float = 0.0

    # Correlation metrics
    avg_correlation: float = 0.0
    high_correlation_pairs: List[tuple] = field(default_factory=list)

    # Risk metrics
    total_risk_dollars: float = 0.0
    risk_pct_of_equity: float = 0.0
    max_risk_pct: float = 2.0

    # Alerts
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "heat_level": self.heat_level.value,
            "heat_score": round(self.heat_score, 2),
            "position_metrics": {
                "total_positions": self.total_positions,
                "max_positions": self.max_positions,
                "utilization_pct": round(self.position_utilization * 100, 1),
            },
            "capital_metrics": {
                "total_exposure": round(self.total_exposure, 2),
                "max_exposure": round(self.max_exposure, 2),
                "exposure_pct": round(self.exposure_pct * 100, 1),
            },
            "concentration_metrics": {
                "largest_position_pct": round(self.largest_position_pct * 100, 1),
                "top_3_concentration_pct": round(self.top_3_concentration * 100, 1),
                "max_single_pct": self.max_single_position_pct,
            },
            "sector_exposure": {k: round(v * 100, 1) for k, v in self.sector_exposure.items()},
            "correlation": {
                "avg_correlation": round(self.avg_correlation, 3),
                "high_correlation_pairs": self.high_correlation_pairs[:5],
            },
            "risk_metrics": {
                "total_risk_dollars": round(self.total_risk_dollars, 2),
                "risk_pct_of_equity": round(self.risk_pct_of_equity * 100, 2),
                "max_risk_pct": self.max_risk_pct,
            },
            "alerts": self.alerts,
            "recommendations": self.recommendations,
        }


class PortfolioHeatMonitor:
    """
    Portfolio Heat Monitor.

    Tracks portfolio-level risk metrics and provides alerts
    when thresholds are exceeded.
    """

    # Default thresholds
    MAX_POSITIONS = 5
    MAX_SINGLE_POSITION_PCT = 0.20  # 20%
    MAX_SECTOR_PCT = 0.40           # 40%
    MAX_RISK_PCT = 0.02             # 2% of equity at risk
    HIGH_CORRELATION_THRESHOLD = 0.70

    def __init__(
        self,
        max_positions: int = 5,
        max_single_position_pct: float = 0.20,
        max_sector_pct: float = 0.40,
        max_risk_pct: float = 0.02
    ):
        """
        Initialize heat monitor.

        Args:
            max_positions: Maximum allowed positions
            max_single_position_pct: Max % in single position
            max_sector_pct: Max % in single sector
            max_risk_pct: Max % of equity at risk
        """
        self.max_positions = max_positions
        self.max_single_position_pct = max_single_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_risk_pct = max_risk_pct
        self.sector_map: Dict[str, str] = {}

    def calculate_heat(
        self,
        positions: List[Dict[str, Any]],
        equity: float,
        sector_map: Optional[Dict[str, str]] = None,
        price_history: Optional[Dict[str, pd.DataFrame]] = None
    ) -> HeatStatus:
        """
        Calculate portfolio heat status.

        Args:
            positions: List of position dicts with symbol, market_value, unrealized_pnl
            equity: Total account equity
            sector_map: Optional dict mapping symbols to sectors
            price_history: Optional dict of price DataFrames for correlation

        Returns:
            HeatStatus with complete heat analysis
        """
        status = HeatStatus(
            timestamp=datetime.now(),
            heat_level=HeatLevel.COLD,
            heat_score=0.0,
            max_positions=self.max_positions,
            max_single_position_pct=self.max_single_position_pct * 100,
            max_risk_pct=self.max_risk_pct * 100
        )

        if not positions or equity <= 0:
            return status

        # Update sector map if provided
        if sector_map:
            self.sector_map.update(sector_map)

        # Position metrics
        status.total_positions = len(positions)
        status.position_utilization = len(positions) / self.max_positions

        # Capital metrics
        total_exposure = sum(abs(float(p.get('market_value', 0))) for p in positions)
        status.total_exposure = total_exposure
        status.max_exposure = equity
        status.exposure_pct = total_exposure / equity if equity > 0 else 0

        # Concentration metrics
        position_values = sorted(
            [abs(float(p.get('market_value', 0))) for p in positions],
            reverse=True
        )

        if position_values:
            status.largest_position_pct = position_values[0] / equity if equity > 0 else 0
            top_3 = sum(position_values[:3])
            status.top_3_concentration = top_3 / equity if equity > 0 else 0

        # Sector exposure
        sector_values: Dict[str, float] = {}
        for pos in positions:
            symbol = pos.get('symbol', '')
            value = abs(float(pos.get('market_value', 0)))
            sector = self.sector_map.get(symbol, 'Unknown')
            sector_values[sector] = sector_values.get(sector, 0) + value

        for sector, value in sector_values.items():
            status.sector_exposure[sector] = value / equity if equity > 0 else 0

        if status.sector_exposure:
            status.max_sector_pct = max(status.sector_exposure.values())

        # Risk metrics (using stops if available)
        total_risk = 0.0
        for pos in positions:
            entry = float(pos.get('entry_price', pos.get('avg_entry_price', 0)))
            stop = float(pos.get('stop_loss', entry * 0.98))  # Default 2% stop
            qty = abs(int(pos.get('qty', 0)))
            risk = abs(entry - stop) * qty
            total_risk += risk

        status.total_risk_dollars = total_risk
        status.risk_pct_of_equity = total_risk / equity if equity > 0 else 0

        # Correlation analysis (if price history provided)
        if price_history and len(price_history) >= 2:
            status.avg_correlation, status.high_correlation_pairs = \
                self._calculate_correlations(positions, price_history)

        # Calculate heat score and level
        status.heat_score = self._calculate_heat_score(status)
        status.heat_level = self._determine_heat_level(status.heat_score)

        # Generate alerts and recommendations
        status.alerts = self._generate_alerts(status)
        status.recommendations = self._generate_recommendations(status)

        return status

    def _calculate_correlations(
        self,
        positions: List[Dict],
        price_history: Dict[str, pd.DataFrame]
    ) -> tuple:
        """Calculate position correlations."""
        symbols = [p.get('symbol', '') for p in positions]
        symbols = [s for s in symbols if s in price_history]

        if len(symbols) < 2:
            return 0.0, []

        try:
            # Build returns DataFrame
            returns_data = {}
            for symbol in symbols:
                df = price_history[symbol]
                close_col = 'close' if 'close' in df.columns else 'Close'
                if close_col in df.columns:
                    returns_data[symbol] = df[close_col].pct_change().dropna()

            if len(returns_data) < 2:
                return 0.0, []

            returns_df = pd.DataFrame(returns_data).dropna()

            if len(returns_df) < 20:
                return 0.0, []

            # Calculate correlation matrix
            corr_matrix = returns_df.corr()

            # Get average correlation (excluding diagonal)
            mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            upper_corrs = corr_matrix.values[mask]
            avg_corr = np.mean(upper_corrs) if len(upper_corrs) > 0 else 0

            # Find high correlation pairs
            high_pairs = []
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if i < j:
                        corr = corr_matrix.loc[sym1, sym2]
                        if abs(corr) > self.HIGH_CORRELATION_THRESHOLD:
                            high_pairs.append((sym1, sym2, round(corr, 3)))

            return float(avg_corr), high_pairs

        except Exception as e:
            logger.debug(f"Correlation calculation error: {e}")
            return 0.0, []

    def _calculate_heat_score(self, status: HeatStatus) -> float:
        """Calculate overall heat score (0-100)."""
        score = 0.0

        # Position utilization (0-20 points)
        score += status.position_utilization * 20

        # Exposure level (0-20 points)
        score += min(status.exposure_pct, 1.0) * 20

        # Concentration (0-20 points)
        if status.largest_position_pct > self.max_single_position_pct:
            score += 20
        else:
            score += (status.largest_position_pct / self.max_single_position_pct) * 20

        # Sector concentration (0-15 points)
        if status.max_sector_pct > self.max_sector_pct:
            score += 15
        else:
            score += (status.max_sector_pct / self.max_sector_pct) * 15

        # Risk level (0-15 points)
        if status.risk_pct_of_equity > self.max_risk_pct:
            score += 15
        else:
            score += (status.risk_pct_of_equity / self.max_risk_pct) * 15

        # Correlation risk (0-10 points)
        if status.avg_correlation > 0.5:
            score += 10
        else:
            score += (status.avg_correlation / 0.5) * 10

        return min(score, 100)

    def _determine_heat_level(self, score: float) -> HeatLevel:
        """Determine heat level from score."""
        if score >= 80:
            return HeatLevel.OVERHEATED
        elif score >= 60:
            return HeatLevel.HOT
        elif score >= 40:
            return HeatLevel.WARM
        elif score >= 20:
            return HeatLevel.NORMAL
        else:
            return HeatLevel.COLD

    def _generate_alerts(self, status: HeatStatus) -> List[str]:
        """Generate alerts for threshold violations."""
        alerts = []

        if status.total_positions >= self.max_positions:
            alerts.append(f"Position limit reached: {status.total_positions}/{self.max_positions}")

        if status.largest_position_pct > self.max_single_position_pct:
            alerts.append(f"Single position too large: {status.largest_position_pct*100:.1f}% > {self.max_single_position_pct*100:.0f}%")

        if status.max_sector_pct > self.max_sector_pct:
            alerts.append(f"Sector concentration too high: {status.max_sector_pct*100:.1f}% > {self.max_sector_pct*100:.0f}%")

        if status.risk_pct_of_equity > self.max_risk_pct:
            alerts.append(f"Risk exceeds limit: {status.risk_pct_of_equity*100:.2f}% > {self.max_risk_pct*100:.0f}%")

        if len(status.high_correlation_pairs) > 0:
            alerts.append(f"High correlation detected: {len(status.high_correlation_pairs)} pairs > {self.HIGH_CORRELATION_THRESHOLD}")

        return alerts

    def _generate_recommendations(self, status: HeatStatus) -> List[str]:
        """Generate recommendations based on heat status."""
        recs = []

        if status.heat_level == HeatLevel.OVERHEATED:
            recs.append("REDUCE EXPOSURE: Close weakest positions immediately")
            recs.append("NO NEW TRADES until heat score drops below 60")

        elif status.heat_level == HeatLevel.HOT:
            recs.append("CAUTION: Consider trimming largest positions")
            recs.append("Avoid adding correlated positions")

        elif status.heat_level == HeatLevel.WARM:
            recs.append("MONITOR: Approaching risk limits")
            recs.append("Be selective with new entries")

        elif status.heat_level == HeatLevel.NORMAL:
            recs.append("HEALTHY: Portfolio risk is well-managed")

        else:  # COLD
            recs.append("ROOM TO ADD: Low exposure, consider new opportunities")

        return recs

    def can_add_position(self, status: HeatStatus) -> bool:
        """Check if a new position can be added."""
        return (
            status.total_positions < self.max_positions and
            status.heat_level not in [HeatLevel.HOT, HeatLevel.OVERHEATED]
        )


# Singleton instance
_heat_monitor: Optional[PortfolioHeatMonitor] = None


def get_heat_monitor() -> PortfolioHeatMonitor:
    """Get or create the global heat monitor instance."""
    global _heat_monitor
    if _heat_monitor is None:
        _heat_monitor = PortfolioHeatMonitor()
    return _heat_monitor
