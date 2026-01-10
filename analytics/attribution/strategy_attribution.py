"""
Strategy Attribution - P&L by Strategy

Track which strategies are making money and which are losing.
This tells you where your REAL edge is coming from.

Answers questions like:
- Is IBS_RSI or Turtle Soup more profitable?
- Which strategy has better risk-adjusted returns?
- Is a strategy's performance drifting?

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import statistics

from core.structured_log import get_logger

logger = get_logger(__name__)


@dataclass
class StrategyPnL:
    """P&L metrics for a single strategy."""
    strategy_name: str
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    trades: int
    winners: int
    losers: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float           # Simplified daily Sharpe
    max_drawdown: float
    contribution_pct: float       # % of total portfolio P&L
    largest_winner: float
    largest_loser: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "total_pnl": self.total_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "trades": self.trades,
            "winners": self.winners,
            "losers": self.losers,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "contribution_pct": self.contribution_pct,
            "largest_winner": self.largest_winner,
            "largest_loser": self.largest_loser,
        }

    def to_summary(self) -> str:
        """Generate plain English summary."""
        emoji = "+" if self.total_pnl >= 0 else ""

        return (
            f"**{self.strategy_name}**: {emoji}${self.total_pnl:,.2f}\n"
            f"  Trades: {self.trades} | Win Rate: {self.win_rate:.0%}\n"
            f"  Profit Factor: {self.profit_factor:.2f} | Sharpe: {self.sharpe_ratio:.2f}\n"
            f"  Contribution: {self.contribution_pct:.1f}% of total P&L"
        )


@dataclass
class StrategyComparison:
    """Compare strategies side-by-side."""
    date_range: str
    strategies: Dict[str, StrategyPnL]
    best_strategy: str
    worst_strategy: str
    total_pnl: float
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date_range": self.date_range,
            "strategies": {k: v.to_dict() for k, v in self.strategies.items()},
            "best_strategy": self.best_strategy,
            "worst_strategy": self.worst_strategy,
            "total_pnl": self.total_pnl,
            "recommendation": self.recommendation,
        }


class StrategyAttributor:
    """
    Track P&L by strategy.

    Features:
    - Real-time strategy-level P&L
    - Strategy comparison
    - Drift detection (is strategy getting worse?)
    - Recommendations for rebalancing
    """

    HISTORY_FILE = Path("state/pnl/strategy_history.json")

    def __init__(self):
        """Initialize strategy attributor."""
        self._trades_by_strategy: Dict[str, List[Dict]] = {}
        self._daily_pnl: Dict[str, List[Tuple[date, float]]] = {}

        # Ensure directory exists
        self.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

        self._load_state()

    def _load_state(self) -> None:
        """Load strategy history."""
        if self.HISTORY_FILE.exists():
            try:
                with open(self.HISTORY_FILE, "r") as f:
                    data = json.load(f)
                    self._trades_by_strategy = data.get("trades", {})
                    # Convert daily_pnl back from serialized format
                    raw_daily = data.get("daily_pnl", {})
                    for strat, entries in raw_daily.items():
                        self._daily_pnl[strat] = [
                            (date.fromisoformat(e[0]), e[1]) for e in entries
                        ]
            except Exception as e:
                logger.warning(f"Failed to load strategy history: {e}")

    def _save_state(self) -> None:
        """Save strategy history."""
        try:
            # Serialize daily_pnl
            serialized_daily = {}
            for strat, entries in self._daily_pnl.items():
                serialized_daily[strat] = [
                    [e[0].isoformat(), e[1]] for e in entries[-365:]
                ]

            with open(self.HISTORY_FILE, "w") as f:
                json.dump({
                    "trades": {k: v[-1000:] for k, v in self._trades_by_strategy.items()},
                    "daily_pnl": serialized_daily,
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save strategy history: {e}")

    def record_trade(
        self,
        strategy: str,
        symbol: str,
        pnl: float,
        is_winner: bool,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record a completed trade for a strategy.

        Args:
            strategy: Strategy name
            symbol: Symbol traded
            pnl: Net P&L
            is_winner: Whether trade was profitable
            timestamp: Trade timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()

        if strategy not in self._trades_by_strategy:
            self._trades_by_strategy[strategy] = []

        self._trades_by_strategy[strategy].append({
            "symbol": symbol,
            "pnl": pnl,
            "is_winner": is_winner,
            "timestamp": timestamp.isoformat(),
        })

        # Update daily P&L
        trade_date = timestamp.date()
        if strategy not in self._daily_pnl:
            self._daily_pnl[strategy] = []

        # Add or update today's entry
        if self._daily_pnl[strategy] and self._daily_pnl[strategy][-1][0] == trade_date:
            # Update existing entry
            old_pnl = self._daily_pnl[strategy][-1][1]
            self._daily_pnl[strategy][-1] = (trade_date, old_pnl + pnl)
        else:
            self._daily_pnl[strategy].append((trade_date, pnl))

        self._save_state()

    def get_strategy_pnl(
        self,
        strategy: str,
        days: int = 30,
    ) -> Optional[StrategyPnL]:
        """
        Get P&L metrics for a strategy.

        Args:
            strategy: Strategy name
            days: Lookback period in days

        Returns:
            StrategyPnL or None
        """
        if strategy not in self._trades_by_strategy:
            return None

        cutoff = datetime.now() - timedelta(days=days)
        trades = [
            t for t in self._trades_by_strategy[strategy]
            if datetime.fromisoformat(t["timestamp"]) >= cutoff
        ]

        if not trades:
            return None

        pnls = [t["pnl"] for t in trades]
        winners = [t for t in trades if t["is_winner"]]
        losers = [t for t in trades if not t["is_winner"]]

        total_pnl = sum(pnls)
        win_pnls = [t["pnl"] for t in winners]
        loss_pnls = [abs(t["pnl"]) for t in losers]

        avg_win = statistics.mean(win_pnls) if win_pnls else 0
        avg_loss = statistics.mean(loss_pnls) if loss_pnls else 0

        gross_profit = sum(win_pnls)
        gross_loss = sum(loss_pnls)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calculate Sharpe (simplified)
        if len(pnls) > 1:
            daily_returns = pnls  # Simplified - treating each trade as a day
            sharpe = (
                statistics.mean(daily_returns) /
                (statistics.stdev(daily_returns) + 0.0001) *
                (252 ** 0.5)  # Annualize
            )
        else:
            sharpe = 0.0

        # Calculate max drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnls:
            cumulative += pnl
            peak = max(peak, cumulative)
            dd = (peak - cumulative) / (peak + 0.0001)
            max_dd = max(max_dd, dd)

        return StrategyPnL(
            strategy_name=strategy,
            total_pnl=total_pnl,
            realized_pnl=total_pnl,
            unrealized_pnl=0.0,
            trades=len(trades),
            winners=len(winners),
            losers=len(losers),
            win_rate=len(winners) / len(trades) if trades else 0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=min(profit_factor, 99.99),  # Cap for display
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            contribution_pct=100.0,  # Will be updated in comparison
            largest_winner=max(win_pnls) if win_pnls else 0,
            largest_loser=max(loss_pnls) if loss_pnls else 0,
        )

    def compare_strategies(
        self,
        strategies: Optional[List[str]] = None,
        days: int = 30,
    ) -> StrategyComparison:
        """
        Compare multiple strategies.

        Args:
            strategies: List of strategy names (all if None)
            days: Lookback period

        Returns:
            StrategyComparison
        """
        if strategies is None:
            strategies = list(self._trades_by_strategy.keys())

        strategy_pnls: Dict[str, StrategyPnL] = {}
        total_pnl = 0.0

        for strat in strategies:
            pnl = self.get_strategy_pnl(strat, days)
            if pnl:
                strategy_pnls[strat] = pnl
                total_pnl += pnl.total_pnl

        # Update contribution percentages
        for strat, pnl in strategy_pnls.items():
            pnl.contribution_pct = (
                pnl.total_pnl / total_pnl * 100 if total_pnl != 0 else 0
            )

        # Find best/worst
        if strategy_pnls:
            best = max(strategy_pnls.items(), key=lambda x: x[1].total_pnl)
            worst = min(strategy_pnls.items(), key=lambda x: x[1].total_pnl)
            best_name = best[0]
            worst_name = worst[0]

            # Generate recommendation
            if worst[1].total_pnl < 0 and len(strategy_pnls) > 1:
                recommendation = (
                    f"Consider reducing allocation to {worst_name} "
                    f"(P&L: ${worst[1].total_pnl:,.2f}, PF: {worst[1].profit_factor:.2f})"
                )
            elif best[1].sharpe_ratio > 1.5:
                recommendation = (
                    f"{best_name} showing strong risk-adjusted returns "
                    f"(Sharpe: {best[1].sharpe_ratio:.2f}). Consider increasing allocation."
                )
            else:
                recommendation = "All strategies performing within normal parameters."
        else:
            best_name = "N/A"
            worst_name = "N/A"
            recommendation = "Insufficient data for comparison."

        return StrategyComparison(
            date_range=f"Last {days} days",
            strategies=strategy_pnls,
            best_strategy=best_name,
            worst_strategy=worst_name,
            total_pnl=total_pnl,
            recommendation=recommendation,
        )

    def detect_drift(
        self,
        strategy: str,
        lookback_days: int = 60,
        recent_days: int = 14,
    ) -> Dict[str, Any]:
        """
        Detect if a strategy's performance is drifting.

        Compares recent performance to longer-term average.

        Args:
            strategy: Strategy name
            lookback_days: Full lookback period
            recent_days: Recent period to compare

        Returns:
            Dict with drift analysis
        """
        full_pnl = self.get_strategy_pnl(strategy, lookback_days)
        recent_pnl = self.get_strategy_pnl(strategy, recent_days)

        if not full_pnl or not recent_pnl:
            return {"has_data": False}

        # Calculate drift
        wr_drift = recent_pnl.win_rate - full_pnl.win_rate
        pf_drift = recent_pnl.profit_factor - full_pnl.profit_factor

        is_degrading = wr_drift < -0.10 or pf_drift < -0.3

        return {
            "has_data": True,
            "strategy": strategy,
            "full_period_wr": full_pnl.win_rate,
            "recent_wr": recent_pnl.win_rate,
            "wr_drift": wr_drift,
            "full_period_pf": full_pnl.profit_factor,
            "recent_pf": recent_pnl.profit_factor,
            "pf_drift": pf_drift,
            "is_degrading": is_degrading,
            "recommendation": (
                f"ALERT: {strategy} showing degradation" if is_degrading
                else f"{strategy} performing normally"
            ),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary for dashboard."""
        comparison = self.compare_strategies(days=30)

        return {
            "total_pnl": comparison.total_pnl,
            "best_strategy": comparison.best_strategy,
            "worst_strategy": comparison.worst_strategy,
            "strategies_count": len(comparison.strategies),
            "recommendation": comparison.recommendation,
        }


if __name__ == "__main__":
    # Demo
    attributor = StrategyAttributor()

    print("=== Strategy Attribution Demo ===\n")

    # Simulate trades
    trades = [
        ("IBS_RSI", "AAPL", 150.0, True),
        ("IBS_RSI", "MSFT", 75.0, True),
        ("IBS_RSI", "GOOGL", -50.0, False),
        ("TurtleSoup", "TSLA", 200.0, True),
        ("TurtleSoup", "AMD", -80.0, False),
        ("TurtleSoup", "NVDA", 120.0, True),
    ]

    for strat, symbol, pnl, is_win in trades:
        attributor.record_trade(strat, symbol, pnl, is_win)

    # Compare strategies
    comparison = attributor.compare_strategies()
    print(f"Best Strategy: {comparison.best_strategy}")
    print(f"Worst Strategy: {comparison.worst_strategy}")
    print(f"Total P&L: ${comparison.total_pnl:,.2f}")
    print(f"Recommendation: {comparison.recommendation}")

    print("\n--- Strategy Details ---")
    for strat, pnl in comparison.strategies.items():
        print(pnl.to_summary())
        print()

    # Drift detection
    print("--- Drift Detection ---")
    for strat in ["IBS_RSI", "TurtleSoup"]:
        drift = attributor.detect_drift(strat)
        print(f"{strat}: {drift.get('recommendation', 'No data')}")
