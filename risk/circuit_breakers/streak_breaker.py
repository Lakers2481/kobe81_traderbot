"""
Streak Circuit Breaker - Consecutive Loss Protection

Protects capital by detecting losing streaks and pausing trading
before a bad run destroys your account.

Psychology: After consecutive losses, most traders make worse decisions.
This breaker forces a cooling-off period to prevent tilt trading.

Thresholds:
- 8+ consecutive losses: HALT_ALL (something is very wrong)
- 5+ consecutive losses: PAUSE_NEW (cool off period)
- 3+ consecutive losses: REDUCE_SIZE (be cautious)

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from core.structured_log import get_logger
from .breaker_manager import BreakerAction, BreakerStatus

logger = get_logger(__name__)


@dataclass
class StreakThresholds:
    """Consecutive loss thresholds for circuit breaker."""
    # Loss streak thresholds
    halt_streak: int = 8           # 8 losses in a row → HALT
    pause_streak: int = 5          # 5 losses in a row → PAUSE
    reduce_streak: int = 3         # 3 losses in a row → REDUCE

    # Win streak thresholds (for overconfidence detection)
    overconfidence_streak: int = 7  # 7 wins might lead to overconfidence

    # Time-based streaks
    daily_loss_limit: int = 4      # Max losses in one day before pause
    hourly_loss_limit: int = 2     # Max losses in one hour before pause


class StreakBreaker:
    """
    Circuit breaker that monitors trading streaks.

    Solo Trader Features:
    - Tracks consecutive wins/losses
    - Detects losing streaks before they spiral
    - Warns of overconfidence after big wins
    - Time-based streak limits (daily, hourly)
    """

    STATE_FILE = Path("state/circuit_breakers/streak_history.json")

    def __init__(self, thresholds: Optional[StreakThresholds] = None):
        """
        Initialize streak breaker.

        Args:
            thresholds: Custom thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or StreakThresholds()
        self._trade_history: List[Dict] = []
        self._current_streak: int = 0  # Positive = wins, negative = losses
        self._last_check: Optional[datetime] = None

        # Ensure state directory exists
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load history
        self._load_history()

    def _load_history(self) -> None:
        """Load trade history from state file."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    self._trade_history = data.get("trades", [])[-100:]  # Keep last 100
                    self._current_streak = data.get("current_streak", 0)
            except Exception as e:
                logger.warning(f"Failed to load streak history: {e}")

    def _save_history(self) -> None:
        """Save trade history to state file."""
        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump({
                    "trades": self._trade_history[-100:],
                    "current_streak": self._current_streak,
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save streak history: {e}")

    def record_trade(self, is_winner: bool, pnl: float, symbol: str = "") -> None:
        """
        Record a completed trade for streak tracking.

        Args:
            is_winner: True if trade was profitable
            pnl: Profit/loss amount
            symbol: Symbol traded
        """
        # Update streak
        if is_winner:
            if self._current_streak >= 0:
                self._current_streak += 1
            else:
                self._current_streak = 1
        else:
            if self._current_streak <= 0:
                self._current_streak -= 1
            else:
                self._current_streak = -1

        # Record trade
        self._trade_history.append({
            "timestamp": datetime.now().isoformat(),
            "is_winner": is_winner,
            "pnl": pnl,
            "symbol": symbol,
            "streak_after": self._current_streak,
        })

        self._save_history()

        # Log significant streaks
        if self._current_streak <= -self.thresholds.reduce_streak:
            logger.warning(f"Losing streak: {abs(self._current_streak)} consecutive losses")
        elif self._current_streak >= self.thresholds.overconfidence_streak:
            logger.info(f"Winning streak: {self._current_streak} consecutive wins (watch for overconfidence)")

    def _get_recent_losses(self, hours: float = 1.0) -> int:
        """Count losses in the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        count = 0

        for trade in reversed(self._trade_history):
            trade_time = datetime.fromisoformat(trade["timestamp"])
            if trade_time < cutoff:
                break
            if not trade["is_winner"]:
                count += 1

        return count

    def _get_daily_losses(self) -> int:
        """Count losses today."""
        today = datetime.now().date()
        count = 0

        for trade in reversed(self._trade_history):
            trade_time = datetime.fromisoformat(trade["timestamp"])
            if trade_time.date() < today:
                break
            if not trade["is_winner"]:
                count += 1

        return count

    def _calculate_streak_from_trades(self, trades: List[Dict]) -> int:
        """Calculate current streak from trade list."""
        if not trades:
            return 0

        streak = 0
        current_direction = None

        for trade in reversed(trades):
            is_winner = trade.get("is_winner") or trade.get("pnl", 0) > 0

            if current_direction is None:
                current_direction = is_winner
                streak = 1 if is_winner else -1
            elif is_winner == current_direction:
                streak += 1 if is_winner else -1
            else:
                break

        return streak

    def check(
        self,
        recent_trades: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Check streak levels against thresholds.

        Args:
            recent_trades: List of recent trades with {is_winner, pnl, timestamp}
                          If provided, updates internal history
            **kwargs: Ignored (for compatibility with BreakerManager)

        Returns:
            Dict with status, action, message, and details
        """
        # Update from recent trades if provided
        if recent_trades:
            for trade in recent_trades:
                # Check if this trade is already recorded
                trade_time = trade.get("timestamp", "")
                already_recorded = any(
                    t.get("timestamp") == trade_time
                    for t in self._trade_history
                )

                if not already_recorded:
                    self.record_trade(
                        is_winner=trade.get("is_winner", trade.get("pnl", 0) > 0),
                        pnl=trade.get("pnl", 0),
                        symbol=trade.get("symbol", ""),
                    )

        # Get current streak
        streak = self._current_streak
        abs_streak = abs(streak)
        is_losing = streak < 0

        # Get time-based losses
        hourly_losses = self._get_recent_losses(hours=1.0)
        daily_losses = self._get_daily_losses()

        # Determine status and action
        status = BreakerStatus.GREEN
        action = BreakerAction.CONTINUE
        triggered_by = None
        threshold_hit = 0

        # Check HALT condition (8+ consecutive losses)
        if is_losing and abs_streak >= self.thresholds.halt_streak:
            status = BreakerStatus.RED
            action = BreakerAction.HALT_ALL
            triggered_by = "streak_halt"
            threshold_hit = self.thresholds.halt_streak
            logger.warning(f"HALT: {abs_streak} consecutive losses >= {self.thresholds.halt_streak}")

        # Check PAUSE condition (5+ consecutive losses)
        elif is_losing and abs_streak >= self.thresholds.pause_streak:
            status = BreakerStatus.RED
            action = BreakerAction.PAUSE_NEW
            triggered_by = "streak_pause"
            threshold_hit = self.thresholds.pause_streak
            logger.warning(f"PAUSE: {abs_streak} consecutive losses >= {self.thresholds.pause_streak}")

        # Check REDUCE condition (3+ consecutive losses)
        elif is_losing and abs_streak >= self.thresholds.reduce_streak:
            status = BreakerStatus.YELLOW
            action = BreakerAction.REDUCE_SIZE
            triggered_by = "streak_reduce"
            threshold_hit = self.thresholds.reduce_streak
            logger.info(f"REDUCE: {abs_streak} consecutive losses >= {self.thresholds.reduce_streak}")

        # Check daily loss limit
        elif daily_losses >= self.thresholds.daily_loss_limit:
            status = BreakerStatus.YELLOW
            action = BreakerAction.PAUSE_NEW
            triggered_by = "daily_losses"
            threshold_hit = self.thresholds.daily_loss_limit
            logger.warning(f"PAUSE: {daily_losses} losses today >= {self.thresholds.daily_loss_limit}")

        # Check hourly loss limit
        elif hourly_losses >= self.thresholds.hourly_loss_limit:
            status = BreakerStatus.YELLOW
            action = BreakerAction.REDUCE_SIZE
            triggered_by = "hourly_losses"
            threshold_hit = self.thresholds.hourly_loss_limit
            logger.info(f"REDUCE: {hourly_losses} losses this hour >= {self.thresholds.hourly_loss_limit}")

        # Check overconfidence warning
        elif streak >= self.thresholds.overconfidence_streak:
            status = BreakerStatus.YELLOW
            action = BreakerAction.ALERT_ONLY
            triggered_by = "overconfidence"
            threshold_hit = self.thresholds.overconfidence_streak
            logger.info(f"ALERT: {streak} consecutive wins - watch for overconfidence")

        # Build message
        if triggered_by:
            if triggered_by == "overconfidence":
                message = f"Winning streak of {streak} - beware of overconfidence"
            elif "losses" in triggered_by:
                count = daily_losses if "daily" in triggered_by else hourly_losses
                message = f"{count} losses {'today' if 'daily' in triggered_by else 'this hour'}"
            else:
                message = f"Losing streak of {abs_streak} consecutive losses"
        else:
            if streak > 0:
                message = f"Winning streak: {streak} consecutive wins"
            elif streak < 0:
                message = f"Losing streak: {abs_streak} consecutive losses (within limits)"
            else:
                message = "No active streak"

        self._last_check = datetime.now()

        return {
            "status": status,
            "action": action,
            "message": message,
            "triggered_by": triggered_by,
            "threshold": threshold_hit,
            "current_value": abs_streak if is_losing else streak,
            "details": {
                "current_streak": streak,
                "is_losing_streak": is_losing,
                "daily_losses": daily_losses,
                "hourly_losses": hourly_losses,
                "total_trades_tracked": len(self._trade_history),
                "thresholds": {
                    "halt_streak": self.thresholds.halt_streak,
                    "pause_streak": self.thresholds.pause_streak,
                    "reduce_streak": self.thresholds.reduce_streak,
                    "daily_loss_limit": self.thresholds.daily_loss_limit,
                    "hourly_loss_limit": self.thresholds.hourly_loss_limit,
                },
            },
        }

    def get_streak_summary(self) -> Dict[str, Any]:
        """Get summary of current streak status."""
        return {
            "current_streak": self._current_streak,
            "is_losing": self._current_streak < 0,
            "is_winning": self._current_streak > 0,
            "daily_losses": self._get_daily_losses(),
            "hourly_losses": self._get_recent_losses(hours=1.0),
            "trades_today": len([
                t for t in self._trade_history
                if datetime.fromisoformat(t["timestamp"]).date() == datetime.now().date()
            ]),
        }

    def reset_streak(self) -> None:
        """Manually reset streak (e.g., after market close)."""
        self._current_streak = 0
        self._save_history()
        logger.info("Streak counter reset")


if __name__ == "__main__":
    # Demo
    breaker = StreakBreaker()

    print("=== Streak Breaker Demo ===\n")

    # Simulate a losing streak
    print("Simulating trades...")
    trades = [
        {"is_winner": True, "pnl": 100, "timestamp": datetime.now().isoformat()},
        {"is_winner": False, "pnl": -50, "timestamp": datetime.now().isoformat()},
        {"is_winner": False, "pnl": -75, "timestamp": datetime.now().isoformat()},
        {"is_winner": False, "pnl": -60, "timestamp": datetime.now().isoformat()},
    ]

    result = breaker.check(recent_trades=trades)
    print("\nAfter 3 consecutive losses:")
    print(f"  Status: {result['status'].value}")
    print(f"  Action: {result['action'].value}")
    print(f"  Message: {result['message']}")
    print(f"  Current streak: {result['details']['current_streak']}")

    # Add more losses
    more_losses = [
        {"is_winner": False, "pnl": -80, "timestamp": datetime.now().isoformat()},
        {"is_winner": False, "pnl": -90, "timestamp": datetime.now().isoformat()},
    ]

    result = breaker.check(recent_trades=more_losses)
    print("\nAfter 5 consecutive losses:")
    print(f"  Status: {result['status'].value}")
    print(f"  Action: {result['action'].value}")
    print(f"  Message: {result['message']}")

    # Summary
    print(f"\nStreak Summary: {breaker.get_streak_summary()}")
