"""
Portfolio Rebalancer - Smart Rebalancing Recommendations

Determines when and how to rebalance the portfolio to maintain
target allocations while minimizing transaction costs.

Key Features:
- Drift detection
- Tax-aware rebalancing
- Transaction cost optimization
- Band-based triggers

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum
import json

from core.structured_log import get_logger

logger = get_logger(__name__)


class RebalanceUrgency(Enum):
    """Urgency level for rebalancing."""
    NONE = "none"           # No rebalancing needed
    LOW = "low"             # Could wait, minor drift
    MEDIUM = "medium"       # Should rebalance soon
    HIGH = "high"           # Rebalance today
    CRITICAL = "critical"   # Immediate rebalancing


class TradeDirection(Enum):
    """Trade direction for rebalancing."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class RebalanceTrade:
    """Single rebalancing trade."""
    symbol: str
    direction: TradeDirection
    current_weight: float
    target_weight: float
    drift: float                    # Current - Target
    shares_to_trade: int
    dollar_amount: float
    is_tax_loss: bool = False       # Tax-loss harvesting opportunity
    days_held: Optional[int] = None # For tax considerations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "current_weight": self.current_weight,
            "target_weight": self.target_weight,
            "drift": self.drift,
            "shares_to_trade": self.shares_to_trade,
            "dollar_amount": self.dollar_amount,
            "is_tax_loss": self.is_tax_loss,
            "days_held": self.days_held,
        }


@dataclass
class RebalanceRecommendation:
    """Complete rebalancing recommendation."""
    urgency: RebalanceUrgency
    trades: List[RebalanceTrade]
    total_turnover: float           # % of portfolio to trade
    estimated_cost: float           # Transaction costs
    max_drift: float                # Largest position drift
    drift_score: float              # Overall portfolio drift
    days_since_last: int
    reason: str
    as_of: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "urgency": self.urgency.value,
            "trades": [t.to_dict() for t in self.trades],
            "total_turnover": self.total_turnover,
            "estimated_cost": self.estimated_cost,
            "max_drift": self.max_drift,
            "drift_score": self.drift_score,
            "days_since_last": self.days_since_last,
            "reason": self.reason,
            "as_of": self.as_of.isoformat(),
        }

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"**Rebalancing Recommendation**",
            f"Urgency: {self.urgency.name}",
            "",
            f"Max Drift: {self.max_drift:.1%}",
            f"Drift Score: {self.drift_score:.2f}",
            f"Total Turnover: {self.total_turnover:.1%}",
            f"Est. Costs: ${self.estimated_cost:.2f}",
            f"Days Since Last: {self.days_since_last}",
            "",
            f"Reason: {self.reason}",
            "",
        ]

        if self.trades:
            lines.append("**Trades:**")
            for trade in sorted(self.trades, key=lambda t: abs(t.drift), reverse=True):
                if trade.direction != TradeDirection.HOLD:
                    action = trade.direction.value.upper()
                    lines.append(
                        f"  {action} {trade.symbol}: "
                        f"{trade.current_weight:.1%} -> {trade.target_weight:.1%} "
                        f"(${trade.dollar_amount:,.0f})"
                    )

        return "\n".join(lines)


class PortfolioRebalancer:
    """
    Smart portfolio rebalancing.

    Features:
    - Band-based trigger (e.g., rebalance when drift > 5%)
    - Calendar-based trigger (e.g., monthly)
    - Tax-aware (avoid short-term gains)
    - Transaction cost optimization
    """

    STATE_FILE = Path("state/portfolio/rebalancer.json")

    # Default thresholds
    DRIFT_BAND_LOW = 0.03       # 3% drift = low urgency
    DRIFT_BAND_MEDIUM = 0.05   # 5% drift = medium urgency
    DRIFT_BAND_HIGH = 0.10     # 10% drift = high urgency
    DRIFT_BAND_CRITICAL = 0.15 # 15% drift = critical

    MAX_DAYS_BETWEEN = 30      # Force rebalance after 30 days

    TRANSACTION_COST_BPS = 10  # 10 bps per trade

    SHORT_TERM_DAYS = 365      # Avoid selling < 1 year (tax)

    def __init__(self):
        """Initialize rebalancer."""
        self._last_rebalance: Optional[date] = None
        self._history: List[Dict] = []

        # Ensure directory
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        self._load_state()

    def _load_state(self) -> None:
        """Load rebalancer state."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    if data.get("last_rebalance"):
                        self._last_rebalance = date.fromisoformat(data["last_rebalance"])
                    self._history = data.get("history", [])
            except Exception as e:
                logger.warning(f"Failed to load rebalancer state: {e}")

    def _save_state(self) -> None:
        """Save rebalancer state."""
        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump({
                    "last_rebalance": self._last_rebalance.isoformat() if self._last_rebalance else None,
                    "history": self._history[-100:],
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save rebalancer state: {e}")

    def _calculate_drift_score(self, drifts: List[float]) -> float:
        """Calculate overall portfolio drift score."""
        if not drifts:
            return 0.0
        # Root mean square of drifts
        return (sum(d ** 2 for d in drifts) / len(drifts)) ** 0.5

    def _determine_urgency(
        self,
        max_drift: float,
        drift_score: float,
        days_since_last: int,
    ) -> tuple[RebalanceUrgency, str]:
        """Determine rebalancing urgency and reason."""
        reasons = []

        # Time-based trigger
        if days_since_last >= self.MAX_DAYS_BETWEEN:
            reasons.append(f"Calendar trigger ({days_since_last} days)")
            return RebalanceUrgency.MEDIUM, "; ".join(reasons)

        # Drift-based triggers
        if max_drift >= self.DRIFT_BAND_CRITICAL:
            reasons.append(f"Critical drift ({max_drift:.1%})")
            return RebalanceUrgency.CRITICAL, "; ".join(reasons)

        if max_drift >= self.DRIFT_BAND_HIGH:
            reasons.append(f"High drift ({max_drift:.1%})")
            return RebalanceUrgency.HIGH, "; ".join(reasons)

        if max_drift >= self.DRIFT_BAND_MEDIUM:
            reasons.append(f"Medium drift ({max_drift:.1%})")
            return RebalanceUrgency.MEDIUM, "; ".join(reasons)

        if max_drift >= self.DRIFT_BAND_LOW:
            reasons.append(f"Low drift ({max_drift:.1%})")
            return RebalanceUrgency.LOW, "; ".join(reasons)

        return RebalanceUrgency.NONE, "Portfolio within bands"

    def recommend(
        self,
        current_positions: Dict[str, Dict],  # symbol -> {shares, price, cost_basis, purchase_date}
        target_weights: Dict[str, float],
        total_value: float,
    ) -> RebalanceRecommendation:
        """
        Generate rebalancing recommendation.

        Args:
            current_positions: Current portfolio positions
            target_weights: Target allocation weights
            total_value: Total portfolio value

        Returns:
            RebalanceRecommendation
        """
        # Calculate current weights
        current_weights = {}
        for symbol, pos in current_positions.items():
            value = pos.get("shares", 0) * pos.get("price", 0)
            current_weights[symbol] = value / total_value if total_value > 0 else 0

        # Add any target symbols not in current portfolio
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        # Calculate drifts and trades
        trades = []
        drifts = []

        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            drift = current - target
            drifts.append(abs(drift))

            # Determine direction
            if drift > 0.005:  # Overweight by 0.5%+
                direction = TradeDirection.SELL
            elif drift < -0.005:  # Underweight by 0.5%+
                direction = TradeDirection.BUY
            else:
                direction = TradeDirection.HOLD

            # Calculate trade details
            dollar_change = abs(drift) * total_value
            price = current_positions.get(symbol, {}).get("price", 0)

            if price > 0:
                shares = int(dollar_change / price)
            else:
                shares = 0

            # Check for tax-loss harvesting
            cost_basis = current_positions.get(symbol, {}).get("cost_basis", 0)
            is_tax_loss = (
                direction == TradeDirection.SELL and
                price < cost_basis
            )

            # Days held
            purchase_date = current_positions.get(symbol, {}).get("purchase_date")
            days_held = None
            if purchase_date:
                if isinstance(purchase_date, str):
                    purchase_date = date.fromisoformat(purchase_date)
                days_held = (date.today() - purchase_date).days

            trade = RebalanceTrade(
                symbol=symbol,
                direction=direction,
                current_weight=current,
                target_weight=target,
                drift=drift,
                shares_to_trade=shares,
                dollar_amount=dollar_change,
                is_tax_loss=is_tax_loss,
                days_held=days_held,
            )
            trades.append(trade)

        # Calculate metrics
        max_drift = max(drifts) if drifts else 0
        drift_score = self._calculate_drift_score(drifts)

        # Total turnover (one-way)
        total_turnover = sum(abs(t.drift) for t in trades) / 2

        # Transaction costs
        estimated_cost = total_turnover * total_value * self.TRANSACTION_COST_BPS / 10000

        # Days since last rebalance
        if self._last_rebalance:
            days_since = (date.today() - self._last_rebalance).days
        else:
            days_since = 999

        # Determine urgency
        urgency, reason = self._determine_urgency(max_drift, drift_score, days_since)

        recommendation = RebalanceRecommendation(
            urgency=urgency,
            trades=trades,
            total_turnover=total_turnover,
            estimated_cost=estimated_cost,
            max_drift=max_drift,
            drift_score=drift_score,
            days_since_last=days_since,
            reason=reason,
        )

        return recommendation

    def execute_rebalance(self) -> None:
        """Mark that rebalancing was executed."""
        self._last_rebalance = date.today()
        self._history.append({
            "date": self._last_rebalance.isoformat(),
            "timestamp": datetime.now().isoformat(),
        })
        self._save_state()

    def get_last_rebalance(self) -> Optional[date]:
        """Get date of last rebalance."""
        return self._last_rebalance


# Singleton
_rebalancer: Optional[PortfolioRebalancer] = None


def get_rebalancer() -> PortfolioRebalancer:
    """Get or create singleton rebalancer."""
    global _rebalancer
    if _rebalancer is None:
        _rebalancer = PortfolioRebalancer()
    return _rebalancer


if __name__ == "__main__":
    # Demo
    rebalancer = PortfolioRebalancer()

    print("=== Portfolio Rebalancer Demo ===\n")

    # Current positions
    positions = {
        "AAPL": {"shares": 100, "price": 175, "cost_basis": 150, "purchase_date": "2023-06-01"},
        "MSFT": {"shares": 50, "price": 380, "cost_basis": 350, "purchase_date": "2023-08-15"},
        "GOOGL": {"shares": 30, "price": 140, "cost_basis": 130, "purchase_date": "2024-01-10"},
        "NVDA": {"shares": 40, "price": 480, "cost_basis": 300, "purchase_date": "2023-03-20"},
    }

    total_value = sum(p["shares"] * p["price"] for p in positions.values())

    # Target weights (equal weight)
    target = {
        "AAPL": 0.25,
        "MSFT": 0.25,
        "GOOGL": 0.25,
        "NVDA": 0.25,
    }

    recommendation = rebalancer.recommend(
        current_positions=positions,
        target_weights=target,
        total_value=total_value,
    )

    print(recommendation.to_summary())

    # Show current vs target
    print("\n**Current vs Target:**")
    for symbol, pos in positions.items():
        value = pos["shares"] * pos["price"]
        current_pct = value / total_value
        target_pct = target[symbol]
        drift = current_pct - target_pct
        print(f"  {symbol}: {current_pct:.1%} vs {target_pct:.1%} ({drift:+.1%})")
