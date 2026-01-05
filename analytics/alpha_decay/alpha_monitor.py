"""
Alpha Decay Monitor - Strategy Health Tracking

Monitor the health of your trading edge:
- Information Coefficient (IC) - Signal quality over time
- Win rate trends - Is performance degrading?
- Profit factor trends - Is edge shrinking?
- Crowding detection - Is everyone trading this now?
- Regime matching - Does current market match backtest?

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import json
import statistics

import numpy as np

from core.structured_log import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Strategy health status."""
    HEALTHY = "healthy"           # All good
    DEGRADING = "degrading"       # Starting to decline
    DYING = "dying"               # Significant decline
    DEAD = "dead"                 # Should be retired


@dataclass
class AlphaHealth:
    """Health metrics for a signal/strategy."""
    # Signal Quality
    information_coefficient: float   # Correlation of signal to returns (-1 to +1)
    ic_rolling_30d: float           # Recent IC trend
    ic_decay_rate: float            # How fast IC is dropping (% per month)

    # Strategy Health
    win_rate_rolling: float         # Recent win rate
    win_rate_trend: float           # Change vs historical
    profit_factor_rolling: float    # Recent profit factor
    profit_factor_trend: float      # Change vs historical
    sharpe_rolling: float           # Recent Sharpe ratio

    # Crowding
    crowding_score: float           # 0-100, higher = more crowded

    # Regime
    regime_match: float             # 0-1, does current regime match backtest?

    # Overall
    overall_health: float           # 0-100 composite score
    status: HealthStatus

    def to_dict(self) -> Dict[str, Any]:
        return {
            "information_coefficient": self.information_coefficient,
            "ic_rolling_30d": self.ic_rolling_30d,
            "ic_decay_rate": self.ic_decay_rate,
            "win_rate_rolling": self.win_rate_rolling,
            "win_rate_trend": self.win_rate_trend,
            "profit_factor_rolling": self.profit_factor_rolling,
            "profit_factor_trend": self.profit_factor_trend,
            "sharpe_rolling": self.sharpe_rolling,
            "crowding_score": self.crowding_score,
            "regime_match": self.regime_match,
            "overall_health": self.overall_health,
            "status": self.status.value,
        }

    def get_color(self) -> str:
        """Get dashboard color for health status."""
        if self.status == HealthStatus.HEALTHY:
            return "green"
        elif self.status == HealthStatus.DEGRADING:
            return "yellow"
        elif self.status == HealthStatus.DYING:
            return "orange"
        else:
            return "red"


@dataclass
class StrategyHealth:
    """Complete health report for a strategy."""
    strategy_name: str
    alpha_health: AlphaHealth
    days_analyzed: int
    trades_analyzed: int
    recommendation: str
    action: str                   # "CONTINUE", "REDUCE", "PAUSE", "RETIRE"
    reasons: List[str]
    as_of: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "alpha_health": self.alpha_health.to_dict(),
            "days_analyzed": self.days_analyzed,
            "trades_analyzed": self.trades_analyzed,
            "recommendation": self.recommendation,
            "action": self.action,
            "reasons": self.reasons,
            "as_of": self.as_of.isoformat(),
        }

    def to_summary(self) -> str:
        """Generate plain English summary."""
        color = self.alpha_health.get_color()
        emoji = {"green": "OK", "yellow": "!", "orange": "!!", "red": "!!!"}[color]

        lines = [
            f"**{self.strategy_name}** [{emoji}]",
            f"Status: {self.alpha_health.status.value.upper()}",
            f"Health Score: {self.alpha_health.overall_health:.0f}/100",
            "",
            f"Win Rate: {self.alpha_health.win_rate_rolling:.1%} "
            f"({self.alpha_health.win_rate_trend:+.1%} trend)",
            f"Profit Factor: {self.alpha_health.profit_factor_rolling:.2f} "
            f"({self.alpha_health.profit_factor_trend:+.2f} trend)",
            f"Crowding: {self.alpha_health.crowding_score:.0f}/100",
            "",
            f"**Action:** {self.action}",
            f"_{self.recommendation}_",
        ]

        if self.reasons:
            lines.append("\nReasons:")
            for r in self.reasons[:3]:
                lines.append(f"  - {r}")

        return "\n".join(lines)


class AlphaDecayMonitor:
    """
    Monitor alpha decay and strategy health.

    Features:
    - Track rolling performance metrics
    - Detect degradation early
    - Crowding detection via return correlation
    - Regime change detection
    - Automatic health scoring
    """

    STATE_FILE = Path("state/alpha_decay/health_history.json")

    # Health thresholds
    HEALTHY_WIN_RATE = 0.55
    HEALTHY_PROFIT_FACTOR = 1.2
    HEALTHY_SHARPE = 0.5

    DEGRADING_WIN_RATE = 0.50
    DEGRADING_PROFIT_FACTOR = 1.0
    DEGRADING_SHARPE = 0.0

    # Trend thresholds
    SIGNIFICANT_DECLINE = -0.10  # 10% decline is significant

    def __init__(self):
        """Initialize alpha decay monitor."""
        self._strategy_trades: Dict[str, List[Dict]] = {}
        self._health_history: Dict[str, List[Dict]] = {}

        # Ensure directory exists
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        self._load_state()

    def _load_state(self) -> None:
        """Load health history."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    self._strategy_trades = data.get("trades", {})
                    self._health_history = data.get("health", {})
            except Exception as e:
                logger.warning(f"Failed to load alpha decay state: {e}")

    def _save_state(self) -> None:
        """Save health history."""
        try:
            # Trim old data
            for strat in self._strategy_trades:
                self._strategy_trades[strat] = self._strategy_trades[strat][-1000:]
            for strat in self._health_history:
                self._health_history[strat] = self._health_history[strat][-365:]

            with open(self.STATE_FILE, "w") as f:
                json.dump({
                    "trades": self._strategy_trades,
                    "health": self._health_history,
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alpha decay state: {e}")

    def record_trade(
        self,
        strategy: str,
        pnl: float,
        signal_strength: float = 0.5,
        return_after: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record a completed trade for health tracking.

        Args:
            strategy: Strategy name
            pnl: Net P&L
            signal_strength: Signal strength when generated (0-1)
            return_after: Actual return achieved
            timestamp: Trade timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()

        if strategy not in self._strategy_trades:
            self._strategy_trades[strategy] = []

        self._strategy_trades[strategy].append({
            "timestamp": timestamp.isoformat(),
            "pnl": pnl,
            "is_winner": pnl > 0,
            "signal_strength": signal_strength,
            "return_after": return_after,
        })

        self._save_state()

    def _calculate_ic(self, trades: List[Dict]) -> float:
        """Calculate Information Coefficient (signal-return correlation)."""
        if len(trades) < 10:
            return 0.0

        signals = [t.get("signal_strength", 0.5) for t in trades]
        returns = [t.get("return_after", t.get("pnl", 0)) for t in trades]

        try:
            if len(set(signals)) < 2 or len(set(returns)) < 2:
                return 0.0

            # Pearson correlation
            n = len(signals)
            sum_x = sum(signals)
            sum_y = sum(returns)
            sum_xy = sum(s * r for s, r in zip(signals, returns))
            sum_x2 = sum(s ** 2 for s in signals)
            sum_y2 = sum(r ** 2 for r in returns)

            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5

            if denominator == 0:
                return 0.0

            return numerator / denominator

        except Exception:
            return 0.0

    def _calculate_crowding(self, strategy: str) -> float:
        """
        Estimate crowding score for a strategy.

        Higher score = more crowded = less edge.
        Based on return correlation with popular factors.
        """
        # Simplified crowding estimation
        # In production, would compare returns to factor returns
        trades = self._strategy_trades.get(strategy, [])

        if len(trades) < 20:
            return 20.0  # Low confidence, assume moderate

        # Proxy: high win rates with low profit factors suggest crowding
        recent = trades[-50:]
        wins = sum(1 for t in recent if t.get("is_winner", False))
        win_rate = wins / len(recent)

        pnls = [t.get("pnl", 0) for t in recent]
        avg_pnl = statistics.mean(pnls) if pnls else 0

        # If win rate is high but average P&L is low, likely crowded
        if win_rate > 0.6 and avg_pnl < 50:
            return 70.0
        elif win_rate > 0.5 and avg_pnl < 100:
            return 50.0
        else:
            return 30.0

    def _calculate_regime_match(self, strategy: str) -> float:
        """
        Estimate how well current regime matches strategy's sweet spot.

        1.0 = perfect match, 0.0 = complete mismatch
        """
        # Simplified: compare recent performance to historical
        trades = self._strategy_trades.get(strategy, [])

        if len(trades) < 100:
            return 0.7  # Insufficient data, assume moderate match

        # Compare last 30 days to full history
        recent = trades[-30:]
        historical = trades[:-30]

        if not recent or not historical:
            return 0.7

        recent_wr = sum(1 for t in recent if t.get("is_winner", False)) / len(recent)
        hist_wr = sum(1 for t in historical if t.get("is_winner", False)) / len(historical)

        # Regime match = how close recent is to historical
        wr_diff = abs(recent_wr - hist_wr)

        if wr_diff < 0.05:
            return 0.95
        elif wr_diff < 0.10:
            return 0.80
        elif wr_diff < 0.15:
            return 0.60
        elif wr_diff < 0.20:
            return 0.40
        else:
            return 0.20

    def _determine_status(self, health_score: float) -> HealthStatus:
        """Determine health status from score."""
        if health_score >= 70:
            return HealthStatus.HEALTHY
        elif health_score >= 50:
            return HealthStatus.DEGRADING
        elif health_score >= 30:
            return HealthStatus.DYING
        else:
            return HealthStatus.DEAD

    def _determine_action(self, status: HealthStatus, health: AlphaHealth) -> Tuple[str, str]:
        """Determine recommended action based on health."""
        if status == HealthStatus.HEALTHY:
            return ("CONTINUE", "Strategy performing well. Maintain current allocation.")

        elif status == HealthStatus.DEGRADING:
            return ("REDUCE", f"Performance declining. Consider reducing allocation by 25-50%.")

        elif status == HealthStatus.DYING:
            return ("PAUSE", "Significant degradation detected. Pause new entries and review.")

        else:  # DEAD
            return ("RETIRE", "Strategy no longer viable. Consider full retirement.")

    def check_health(
        self,
        strategy: str,
        lookback_days: int = 90,
    ) -> StrategyHealth:
        """
        Check health of a strategy.

        Args:
            strategy: Strategy name
            lookback_days: Days to analyze

        Returns:
            StrategyHealth report
        """
        trades = self._strategy_trades.get(strategy, [])

        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_trades = [
            t for t in trades
            if datetime.fromisoformat(t["timestamp"]) >= cutoff
        ]

        if len(recent_trades) < 10:
            # Insufficient data
            return StrategyHealth(
                strategy_name=strategy,
                alpha_health=AlphaHealth(
                    information_coefficient=0,
                    ic_rolling_30d=0,
                    ic_decay_rate=0,
                    win_rate_rolling=0.5,
                    win_rate_trend=0,
                    profit_factor_rolling=1.0,
                    profit_factor_trend=0,
                    sharpe_rolling=0,
                    crowding_score=50,
                    regime_match=0.5,
                    overall_health=50,
                    status=HealthStatus.HEALTHY,
                ),
                days_analyzed=lookback_days,
                trades_analyzed=len(recent_trades),
                recommendation="Insufficient data for health assessment",
                action="CONTINUE",
                reasons=["Need at least 10 trades for assessment"],
                as_of=datetime.now(),
            )

        # Calculate metrics
        # Win rate
        wins = sum(1 for t in recent_trades if t.get("is_winner", False))
        win_rate = wins / len(recent_trades)

        # Win rate trend (compare to older data)
        older = [t for t in trades if datetime.fromisoformat(t["timestamp"]) < cutoff]
        if older:
            old_wr = sum(1 for t in older if t.get("is_winner", False)) / len(older)
            wr_trend = win_rate - old_wr
        else:
            wr_trend = 0

        # Profit factor
        pnls = [t.get("pnl", 0) for t in recent_trades]
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = sum(abs(p) for p in pnls if p < 0)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 2.0

        # PF trend
        if older:
            old_pnls = [t.get("pnl", 0) for t in older]
            old_gp = sum(p for p in old_pnls if p > 0)
            old_gl = sum(abs(p) for p in old_pnls if p < 0)
            old_pf = old_gp / old_gl if old_gl > 0 else 2.0
            pf_trend = profit_factor - old_pf
        else:
            pf_trend = 0

        # Sharpe
        if len(pnls) > 1:
            sharpe = statistics.mean(pnls) / (statistics.stdev(pnls) + 0.001) * (252 ** 0.5)
        else:
            sharpe = 0

        # IC
        ic_full = self._calculate_ic(trades)
        ic_recent = self._calculate_ic(recent_trades)
        ic_decay = (ic_full - ic_recent) / max(abs(ic_full), 0.1) * 100 if ic_full else 0

        # Crowding and regime
        crowding = self._calculate_crowding(strategy)
        regime_match = self._calculate_regime_match(strategy)

        # Calculate overall health score
        scores = []

        # Win rate component (0-25 points)
        if win_rate >= self.HEALTHY_WIN_RATE:
            scores.append(25)
        elif win_rate >= self.DEGRADING_WIN_RATE:
            scores.append(15)
        else:
            scores.append(5)

        # Profit factor component (0-25 points)
        if profit_factor >= self.HEALTHY_PROFIT_FACTOR:
            scores.append(25)
        elif profit_factor >= self.DEGRADING_PROFIT_FACTOR:
            scores.append(15)
        else:
            scores.append(5)

        # Trend component (0-25 points)
        if wr_trend >= 0 and pf_trend >= 0:
            scores.append(25)
        elif wr_trend > self.SIGNIFICANT_DECLINE and pf_trend > -0.3:
            scores.append(15)
        else:
            scores.append(5)

        # Regime match component (0-25 points)
        scores.append(int(regime_match * 25))

        overall_health = sum(scores)

        # Determine status
        status = self._determine_status(overall_health)
        action, recommendation = self._determine_action(status, None)

        # Build reasons
        reasons = []
        if win_rate < self.HEALTHY_WIN_RATE:
            reasons.append(f"Win rate {win_rate:.1%} below target {self.HEALTHY_WIN_RATE:.1%}")
        if profit_factor < self.HEALTHY_PROFIT_FACTOR:
            reasons.append(f"Profit factor {profit_factor:.2f} below target {self.HEALTHY_PROFIT_FACTOR:.2f}")
        if wr_trend < self.SIGNIFICANT_DECLINE:
            reasons.append(f"Win rate declining ({wr_trend:+.1%} trend)")
        if crowding > 60:
            reasons.append(f"High crowding score ({crowding:.0f}/100)")
        if regime_match < 0.5:
            reasons.append(f"Current regime may not match strategy ({regime_match:.1%} match)")

        if not reasons:
            reasons.append("All health metrics within normal range")

        alpha_health = AlphaHealth(
            information_coefficient=ic_full,
            ic_rolling_30d=ic_recent,
            ic_decay_rate=ic_decay,
            win_rate_rolling=win_rate,
            win_rate_trend=wr_trend,
            profit_factor_rolling=profit_factor,
            profit_factor_trend=pf_trend,
            sharpe_rolling=sharpe,
            crowding_score=crowding,
            regime_match=regime_match,
            overall_health=overall_health,
            status=status,
        )

        health = StrategyHealth(
            strategy_name=strategy,
            alpha_health=alpha_health,
            days_analyzed=lookback_days,
            trades_analyzed=len(recent_trades),
            recommendation=recommendation,
            action=action,
            reasons=reasons,
            as_of=datetime.now(),
        )

        # Save to history
        if strategy not in self._health_history:
            self._health_history[strategy] = []
        self._health_history[strategy].append(health.to_dict())
        self._save_state()

        return health

    def get_all_health(self) -> Dict[str, StrategyHealth]:
        """Get health for all tracked strategies."""
        result = {}
        for strategy in self._strategy_trades:
            result[strategy] = self.check_health(strategy)
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary for dashboard."""
        all_health = self.get_all_health()

        if not all_health:
            return {"has_data": False}

        return {
            "has_data": True,
            "strategies_tracked": len(all_health),
            "healthy": sum(1 for h in all_health.values() if h.alpha_health.status == HealthStatus.HEALTHY),
            "degrading": sum(1 for h in all_health.values() if h.alpha_health.status == HealthStatus.DEGRADING),
            "dying": sum(1 for h in all_health.values() if h.alpha_health.status == HealthStatus.DYING),
            "dead": sum(1 for h in all_health.values() if h.alpha_health.status == HealthStatus.DEAD),
            "strategies": {name: h.alpha_health.status.value for name, h in all_health.items()},
        }


# Singleton
_monitor: Optional[AlphaDecayMonitor] = None


def get_alpha_monitor() -> AlphaDecayMonitor:
    """Get or create singleton monitor."""
    global _monitor
    if _monitor is None:
        _monitor = AlphaDecayMonitor()
    return _monitor


if __name__ == "__main__":
    # Demo
    monitor = AlphaDecayMonitor()

    print("=== Alpha Decay Monitor Demo ===\n")

    # Simulate trades for two strategies
    import random

    # IBS_RSI - healthy strategy
    for i in range(50):
        pnl = random.gauss(100, 150)  # Positive expectancy
        monitor.record_trade("IBS_RSI", pnl, signal_strength=random.uniform(0.5, 0.9))

    # TurtleSoup - degrading strategy
    for i in range(50):
        pnl = random.gauss(20, 200)  # Lower expectancy
        monitor.record_trade("TurtleSoup", pnl, signal_strength=random.uniform(0.3, 0.7))

    # Check health
    for strategy in ["IBS_RSI", "TurtleSoup"]:
        health = monitor.check_health(strategy)
        print(health.to_summary())
        print()

    print("--- Summary ---")
    print(monitor.get_summary())
