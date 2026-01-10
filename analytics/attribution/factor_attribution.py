"""
Factor Attribution - Decompose P&L by Systematic Factors

Understand what factors drove your returns:
- Market (beta)
- Size (small vs large cap)
- Value (cheap vs expensive)
- Momentum (trending vs reverting)
- Volatility (low vs high vol)
- Quality (profitable vs unprofitable)

This is Barra-style factor decomposition simplified for a solo trader.

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json


from core.structured_log import get_logger

logger = get_logger(__name__)


@dataclass
class FactorPnL:
    """P&L attributed to a specific factor."""
    factor_name: str
    exposure: float               # Factor loading (-1 to +1)
    factor_return: float          # Factor return (%)
    attributed_pnl: float         # P&L due to this factor
    contribution_pct: float       # % of total P&L

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "exposure": self.exposure,
            "factor_return": self.factor_return,
            "attributed_pnl": self.attributed_pnl,
            "contribution_pct": self.contribution_pct,
        }


@dataclass
class FactorDecomposition:
    """Full factor decomposition of returns."""
    date: date
    total_pnl: float
    factor_pnl: Dict[str, FactorPnL]
    alpha_pnl: float              # Unexplained P&L (true alpha)
    r_squared: float              # How much is explained by factors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "total_pnl": self.total_pnl,
            "factor_pnl": {k: v.to_dict() for k, v in self.factor_pnl.items()},
            "alpha_pnl": self.alpha_pnl,
            "r_squared": self.r_squared,
        }

    def to_summary(self) -> str:
        """Generate plain English summary."""
        lines = [
            f"**Factor Attribution - {self.date}**",
            f"Total P&L: ${self.total_pnl:,.2f}",
            "",
            "**Factor Contributions:**",
        ]

        # Sort by absolute contribution
        sorted_factors = sorted(
            self.factor_pnl.values(),
            key=lambda x: abs(x.attributed_pnl),
            reverse=True
        )

        for f in sorted_factors:
            emoji = "+" if f.attributed_pnl >= 0 else ""
            lines.append(
                f"  {f.factor_name}: {emoji}${f.attributed_pnl:,.2f} "
                f"({f.contribution_pct:+.1f}%)"
            )

        lines.extend([
            "",
            f"**Alpha (unexplained):** ${self.alpha_pnl:,.2f}",
            f"**R-squared:** {self.r_squared:.1%}",
        ])

        return "\n".join(lines)


class FactorAttributor:
    """
    Attribute P&L to systematic factors.

    Factors:
    - Market: SPY return exposure
    - Size: SMB (Small Minus Big) exposure
    - Value: HML (High Minus Low book/market) exposure
    - Momentum: MOM (Winners Minus Losers) exposure
    - Volatility: LVOL (Low Volatility Minus High) exposure
    - Quality: QMJ (Quality Minus Junk) exposure

    Simplified for solo trader - uses ETF proxies instead of
    full Fama-French factor construction.
    """

    FACTOR_PROXIES = {
        "market": "SPY",      # Market factor
        "size": "IWM",        # Small cap proxy (vs SPY for large)
        "value": "IWD",       # Value proxy
        "momentum": "MTUM",   # Momentum factor ETF
        "low_vol": "SPLV",    # Low volatility factor
        "quality": "QUAL",    # Quality factor
    }

    STATE_FILE = Path("state/pnl/factor_history.json")

    def __init__(self):
        """Initialize factor attributor."""
        self._factor_returns: Dict[str, List[Tuple[date, float]]] = {}
        self._decomposition_history: List[Dict] = []

        # Ensure directory exists
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        self._load_state()

    def _load_state(self) -> None:
        """Load factor history."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    self._decomposition_history = data.get("history", [])
            except Exception as e:
                logger.warning(f"Failed to load factor history: {e}")

    def _save_state(self) -> None:
        """Save factor history."""
        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump({
                    "history": self._decomposition_history[-365:],
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save factor history: {e}")

    def _fetch_factor_returns(self, for_date: date) -> Dict[str, float]:
        """
        Fetch factor returns for a date.

        In production, this would fetch from actual data sources.
        For now, returns cached or estimated values.
        """
        factor_returns = {}

        try:
            from data.providers.polygon_eod import PolygonEODProvider

            provider = PolygonEODProvider()

            for factor_name, proxy in self.FACTOR_PROXIES.items():
                try:
                    df = provider.get(proxy, limit=5)
                    if df is not None and len(df) >= 2:
                        returns = df["close"].pct_change().dropna()
                        factor_returns[factor_name] = float(returns.iloc[-1])
                except Exception:
                    factor_returns[factor_name] = 0.0

        except ImportError:
            # No data provider - use placeholders
            for factor_name in self.FACTOR_PROXIES:
                factor_returns[factor_name] = 0.0

        return factor_returns

    def _estimate_exposures(
        self,
        positions: List[Dict],
    ) -> Dict[str, float]:
        """
        Estimate factor exposures for current portfolio.

        Simplified estimation based on position characteristics.
        In production, would use proper factor loadings from a risk model.
        """
        if not positions:
            return {f: 0.0 for f in self.FACTOR_PROXIES}

        exposures = {f: 0.0 for f in self.FACTOR_PROXIES}
        total_value = sum(abs(p.get("value", 0)) for p in positions)

        if total_value == 0:
            return exposures

        # Market exposure (beta)
        exposures["market"] = 1.0  # Assume beta = 1 for simplicity

        # Size exposure (based on market cap hints)
        small_cap_value = sum(
            p.get("value", 0) for p in positions
            if p.get("market_cap", "large") == "small"
        )
        exposures["size"] = small_cap_value / total_value if total_value else 0

        # Momentum exposure (based on recent returns)
        momentum_value = sum(
            p.get("value", 0) for p in positions
            if p.get("momentum_score", 0) > 0.5
        )
        exposures["momentum"] = momentum_value / total_value if total_value else 0

        # Value exposure (based on valuation hints)
        value_value = sum(
            p.get("value", 0) for p in positions
            if p.get("value_score", 0) > 0.5
        )
        exposures["value"] = value_value / total_value if total_value else 0

        # Low vol and quality - use defaults
        exposures["low_vol"] = 0.0
        exposures["quality"] = 0.0

        return exposures

    def decompose(
        self,
        total_pnl: float,
        positions: List[Dict],
        for_date: Optional[date] = None,
    ) -> FactorDecomposition:
        """
        Decompose P&L by factors.

        Args:
            total_pnl: Total P&L to decompose
            positions: List of position dicts with value, etc.
            for_date: Date for factor returns

        Returns:
            FactorDecomposition
        """
        if for_date is None:
            for_date = date.today()

        # Get factor returns
        factor_returns = self._fetch_factor_returns(for_date)

        # Estimate exposures
        exposures = self._estimate_exposures(positions)

        # Calculate factor P&L
        factor_pnl_dict: Dict[str, FactorPnL] = {}
        total_factor_pnl = 0.0
        portfolio_value = sum(abs(p.get("value", 0)) for p in positions) or 1.0

        for factor_name in self.FACTOR_PROXIES:
            exposure = exposures.get(factor_name, 0)
            factor_return = factor_returns.get(factor_name, 0)

            # P&L = exposure * factor_return * portfolio_value
            attributed = exposure * factor_return * portfolio_value

            factor_pnl_dict[factor_name] = FactorPnL(
                factor_name=factor_name,
                exposure=exposure,
                factor_return=factor_return,
                attributed_pnl=attributed,
                contribution_pct=attributed / total_pnl * 100 if total_pnl != 0 else 0,
            )

            total_factor_pnl += attributed

        # Alpha is unexplained P&L
        alpha_pnl = total_pnl - total_factor_pnl

        # R-squared = 1 - (alpha^2 / total^2)
        r_squared = 1.0 - (alpha_pnl ** 2 / (total_pnl ** 2 + 0.0001))
        r_squared = max(0, min(1, r_squared))

        decomposition = FactorDecomposition(
            date=for_date,
            total_pnl=total_pnl,
            factor_pnl=factor_pnl_dict,
            alpha_pnl=alpha_pnl,
            r_squared=r_squared,
        )

        # Save to history
        self._decomposition_history.append(decomposition.to_dict())
        self._save_state()

        return decomposition

    def get_factor_contribution_history(
        self,
        factor: str,
        days: int = 30,
    ) -> List[Tuple[date, float]]:
        """Get historical contribution of a factor."""
        cutoff = date.today() - timedelta(days=days)
        result = []

        for d in self._decomposition_history:
            d_date = date.fromisoformat(d["date"])
            if d_date >= cutoff:
                factor_data = d.get("factor_pnl", {}).get(factor, {})
                result.append((d_date, factor_data.get("attributed_pnl", 0)))

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary for dashboard."""
        if not self._decomposition_history:
            return {"has_data": False}

        recent = self._decomposition_history[-20:]

        # Aggregate factor contributions
        factor_totals = {f: 0.0 for f in self.FACTOR_PROXIES}
        total_alpha = 0.0

        for d in recent:
            for factor, data in d.get("factor_pnl", {}).items():
                factor_totals[factor] = factor_totals.get(factor, 0) + data.get("attributed_pnl", 0)
            total_alpha += d.get("alpha_pnl", 0)

        return {
            "has_data": True,
            "factor_contributions": factor_totals,
            "alpha_total": total_alpha,
            "days_analyzed": len(recent),
        }


if __name__ == "__main__":
    # Demo
    attributor = FactorAttributor()

    print("=== Factor Attribution Demo ===\n")

    # Simulate positions
    positions = [
        {"symbol": "AAPL", "value": 10000, "market_cap": "large", "momentum_score": 0.7},
        {"symbol": "MSFT", "value": 8000, "market_cap": "large", "value_score": 0.6},
        {"symbol": "TSLA", "value": 5000, "market_cap": "large", "momentum_score": 0.8},
    ]

    # Decompose P&L
    decomp = attributor.decompose(
        total_pnl=500.0,
        positions=positions,
    )

    print(decomp.to_summary())
