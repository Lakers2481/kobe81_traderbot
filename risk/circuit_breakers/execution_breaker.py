"""
Execution Circuit Breaker - Slippage Anomaly Detection

Protects capital by detecting when execution quality degrades:
- High slippage eating into profits
- Fill rate dropping (orders not executing)
- Market impact increasing (our orders moving price)

This catches broker issues, liquidity problems, or market
microstructure changes that hurt execution.

Thresholds:
- Slippage > 100bps: HALT_ALL (something very wrong)
- Slippage > 50bps: PAUSE_NEW (execution degraded)
- Slippage > 25bps: REDUCE_SIZE (trade smaller)
- Fill rate < 80%: PAUSE_NEW (orders not executing)

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

import numpy as np

from core.structured_log import get_logger
from .breaker_manager import BreakerAction, BreakerStatus

logger = get_logger(__name__)


@dataclass
class ExecutionThresholds:
    """Execution quality thresholds for circuit breaker."""
    # Slippage thresholds (in basis points)
    slippage_halt: float = 100.0       # 1.0% slippage → HALT
    slippage_pause: float = 50.0       # 0.5% slippage → PAUSE
    slippage_reduce: float = 25.0      # 0.25% slippage → REDUCE
    slippage_alert: float = 15.0       # 0.15% slippage → ALERT

    # Fill rate thresholds (percentage of orders filled)
    fill_rate_pause: float = 0.80      # 80% fill rate → PAUSE
    fill_rate_alert: float = 0.90      # 90% fill rate → ALERT

    # Market impact thresholds (basis points)
    impact_pause: float = 30.0         # 30bps market impact → PAUSE
    impact_alert: float = 15.0         # 15bps market impact → ALERT

    # Minimum trades for analysis
    min_trades_for_analysis: int = 5


class ExecutionBreaker:
    """
    Circuit breaker that monitors execution quality.

    Solo Trader Features:
    - Tracks slippage per trade and rolling average
    - Monitors fill rates
    - Detects market impact
    - Auto-reduces size when execution degrades
    """

    STATE_FILE = Path("state/circuit_breakers/execution_history.json")
    ROLLING_WINDOW = 20  # Number of recent trades for rolling metrics

    def __init__(self, thresholds: Optional[ExecutionThresholds] = None):
        """
        Initialize execution breaker.

        Args:
            thresholds: Custom thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or ExecutionThresholds()
        self._execution_history: List[Dict] = []
        self._last_check: Optional[datetime] = None

        # Ensure state directory exists
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load history
        self._load_history()

    def _load_history(self) -> None:
        """Load execution history from state file."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    self._execution_history = data.get("executions", [])[-200:]
            except Exception as e:
                logger.warning(f"Failed to load execution history: {e}")

    def _save_history(self) -> None:
        """Save execution history to state file."""
        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump({
                    "executions": self._execution_history[-200:],
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save execution history: {e}")

    def record_execution(
        self,
        symbol: str,
        expected_price: float,
        actual_price: float,
        shares: int,
        was_filled: bool,
        pre_trade_price: Optional[float] = None,
        post_trade_price: Optional[float] = None,
    ) -> None:
        """
        Record an execution for quality tracking.

        Args:
            symbol: Symbol traded
            expected_price: Price when signal generated
            actual_price: Actual fill price
            shares: Number of shares
            was_filled: Whether order was filled
            pre_trade_price: Price before our order (for impact)
            post_trade_price: Price after our order (for impact)
        """
        # Calculate slippage in basis points
        if expected_price > 0:
            slippage_bps = abs(actual_price - expected_price) / expected_price * 10000
        else:
            slippage_bps = 0.0

        # Calculate market impact in basis points
        impact_bps = 0.0
        if pre_trade_price and post_trade_price and pre_trade_price > 0:
            impact_bps = abs(post_trade_price - pre_trade_price) / pre_trade_price * 10000

        # Record execution
        self._execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "expected_price": expected_price,
            "actual_price": actual_price,
            "shares": shares,
            "was_filled": was_filled,
            "slippage_bps": slippage_bps,
            "impact_bps": impact_bps,
        })

        self._save_history()

        # Log significant slippage
        if slippage_bps > self.thresholds.slippage_alert:
            logger.warning(
                f"High slippage on {symbol}: {slippage_bps:.1f}bps "
                f"(expected {expected_price:.2f}, got {actual_price:.2f})"
            )

    def _get_rolling_metrics(self) -> Dict[str, Any]:
        """Calculate rolling execution metrics."""
        recent = self._execution_history[-self.ROLLING_WINDOW:]

        if len(recent) < self.thresholds.min_trades_for_analysis:
            return {
                "has_data": False,
                "trades_analyzed": len(recent),
            }

        # Calculate metrics
        slippages = [e["slippage_bps"] for e in recent]
        impacts = [e["impact_bps"] for e in recent if e.get("impact_bps")]
        fills = [e["was_filled"] for e in recent]

        return {
            "has_data": True,
            "trades_analyzed": len(recent),
            "avg_slippage_bps": np.mean(slippages),
            "max_slippage_bps": np.max(slippages),
            "median_slippage_bps": np.median(slippages),
            "avg_impact_bps": np.mean(impacts) if impacts else 0,
            "fill_rate": np.mean(fills),
            "total_slippage_cost": sum(e.get("shares", 0) * e.get("actual_price", 0) * e.get("slippage_bps", 0) / 10000 for e in recent),
        }

    def _get_daily_metrics(self) -> Dict[str, Any]:
        """Calculate today's execution metrics."""
        today = datetime.now().date()

        today_executions = [
            e for e in self._execution_history
            if datetime.fromisoformat(e["timestamp"]).date() == today
        ]

        if not today_executions:
            return {"has_data": False}

        slippages = [e["slippage_bps"] for e in today_executions]

        return {
            "has_data": True,
            "trades_today": len(today_executions),
            "avg_slippage_bps": np.mean(slippages),
            "max_slippage_bps": np.max(slippages),
            "fill_rate": np.mean([e["was_filled"] for e in today_executions]),
        }

    def check(
        self,
        avg_slippage_bps: Optional[float] = None,
        fill_rate: Optional[float] = None,
        market_impact_bps: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Check execution quality against thresholds.

        Args:
            avg_slippage_bps: Average slippage in basis points (calculated if not provided)
            fill_rate: Fill rate (0-1, calculated if not provided)
            market_impact_bps: Average market impact in bps (calculated if not provided)
            **kwargs: Ignored (for compatibility with BreakerManager)

        Returns:
            Dict with status, action, message, and details
        """
        # Get rolling metrics if not provided
        metrics = self._get_rolling_metrics()

        if not metrics.get("has_data"):
            return {
                "status": BreakerStatus.GREEN,
                "action": BreakerAction.CONTINUE,
                "message": f"Insufficient execution data ({metrics.get('trades_analyzed', 0)} trades)",
                "threshold": 0,
                "current_value": 0,
                "details": {"data_available": False},
            }

        # Use provided values or calculated
        slippage = avg_slippage_bps if avg_slippage_bps is not None else metrics["avg_slippage_bps"]
        fill_rt = fill_rate if fill_rate is not None else metrics["fill_rate"]
        impact = market_impact_bps if market_impact_bps is not None else metrics.get("avg_impact_bps", 0)

        # Determine status and action
        status = BreakerStatus.GREEN
        action = BreakerAction.CONTINUE
        triggered_by = None
        threshold_hit = 0

        # Check slippage thresholds (highest priority)
        if slippage >= self.thresholds.slippage_halt:
            status = BreakerStatus.RED
            action = BreakerAction.HALT_ALL
            triggered_by = "slippage_halt"
            threshold_hit = self.thresholds.slippage_halt
            logger.warning(f"HALT: Avg slippage {slippage:.1f}bps >= {self.thresholds.slippage_halt}")

        elif slippage >= self.thresholds.slippage_pause:
            status = BreakerStatus.RED
            action = BreakerAction.PAUSE_NEW
            triggered_by = "slippage_pause"
            threshold_hit = self.thresholds.slippage_pause
            logger.warning(f"PAUSE: Avg slippage {slippage:.1f}bps >= {self.thresholds.slippage_pause}")

        elif slippage >= self.thresholds.slippage_reduce:
            status = BreakerStatus.YELLOW
            action = BreakerAction.REDUCE_SIZE
            triggered_by = "slippage_reduce"
            threshold_hit = self.thresholds.slippage_reduce
            logger.info(f"REDUCE: Avg slippage {slippage:.1f}bps >= {self.thresholds.slippage_reduce}")

        elif slippage >= self.thresholds.slippage_alert:
            status = BreakerStatus.YELLOW
            action = BreakerAction.ALERT_ONLY
            triggered_by = "slippage_alert"
            threshold_hit = self.thresholds.slippage_alert
            logger.info(f"ALERT: Avg slippage {slippage:.1f}bps >= {self.thresholds.slippage_alert}")

        # Check fill rate thresholds
        if fill_rt <= self.thresholds.fill_rate_pause:
            if action.value in ["continue", "alert_only"]:
                status = BreakerStatus.RED
                action = BreakerAction.PAUSE_NEW
                triggered_by = "fill_rate_low"
                threshold_hit = self.thresholds.fill_rate_pause
            logger.warning(f"Fill rate {fill_rt:.1%} <= {self.thresholds.fill_rate_pause:.1%}")

        elif fill_rt <= self.thresholds.fill_rate_alert:
            if action == BreakerAction.CONTINUE:
                status = BreakerStatus.YELLOW
                action = BreakerAction.ALERT_ONLY
                triggered_by = "fill_rate_watch"
                threshold_hit = self.thresholds.fill_rate_alert
            logger.info(f"Fill rate {fill_rt:.1%} <= {self.thresholds.fill_rate_alert:.1%}")

        # Check market impact
        if impact >= self.thresholds.impact_pause:
            if action.value in ["continue", "alert_only"]:
                status = BreakerStatus.YELLOW
                action = BreakerAction.REDUCE_SIZE
                triggered_by = "high_impact"
                threshold_hit = self.thresholds.impact_pause
            logger.warning(f"Market impact {impact:.1f}bps >= {self.thresholds.impact_pause}")

        # Build message
        if triggered_by:
            if "slippage" in triggered_by:
                message = f"High slippage: {slippage:.1f}bps average (threshold: {threshold_hit}bps)"
            elif "fill_rate" in triggered_by:
                message = f"Low fill rate: {fill_rt:.1%} (threshold: {threshold_hit:.1%})"
            elif "impact" in triggered_by:
                message = f"High market impact: {impact:.1f}bps (threshold: {threshold_hit}bps)"
            else:
                message = "Execution quality degraded"
        else:
            message = f"Execution quality good: {slippage:.1f}bps slippage, {fill_rt:.1%} fills"

        self._last_check = datetime.now()

        return {
            "status": status,
            "action": action,
            "message": message,
            "triggered_by": triggered_by,
            "threshold": threshold_hit,
            "current_value": slippage,
            "details": {
                "avg_slippage_bps": slippage,
                "fill_rate": fill_rt,
                "avg_impact_bps": impact,
                "trades_analyzed": metrics["trades_analyzed"],
                "max_slippage_bps": metrics.get("max_slippage_bps"),
                "total_slippage_cost": metrics.get("total_slippage_cost"),
                "thresholds": {
                    "slippage_halt": self.thresholds.slippage_halt,
                    "slippage_pause": self.thresholds.slippage_pause,
                    "slippage_reduce": self.thresholds.slippage_reduce,
                    "fill_rate_pause": self.thresholds.fill_rate_pause,
                    "impact_pause": self.thresholds.impact_pause,
                },
            },
        }

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution quality."""
        rolling = self._get_rolling_metrics()
        daily = self._get_daily_metrics()

        return {
            "rolling": rolling,
            "daily": daily,
            "total_executions": len(self._execution_history),
        }


if __name__ == "__main__":
    # Demo
    breaker = ExecutionBreaker()

    print("=== Execution Breaker Demo ===\n")

    # Simulate some executions
    print("Recording executions...")
    executions = [
        {"symbol": "AAPL", "expected": 175.00, "actual": 175.10, "shares": 100, "filled": True},
        {"symbol": "MSFT", "expected": 380.00, "actual": 380.25, "shares": 50, "filled": True},
        {"symbol": "GOOGL", "expected": 140.00, "actual": 140.05, "shares": 75, "filled": True},
        {"symbol": "TSLA", "expected": 250.00, "actual": 250.50, "shares": 40, "filled": True},
        {"symbol": "NVDA", "expected": 480.00, "actual": 480.75, "shares": 25, "filled": True},
    ]

    for e in executions:
        breaker.record_execution(
            symbol=e["symbol"],
            expected_price=e["expected"],
            actual_price=e["actual"],
            shares=e["shares"],
            was_filled=e["filled"],
        )

    # Check after normal executions
    result = breaker.check()
    print("\nAfter normal executions:")
    print(f"  Status: {result['status'].value}")
    print(f"  Action: {result['action'].value}")
    print(f"  Message: {result['message']}")
    print(f"  Avg slippage: {result['details']['avg_slippage_bps']:.1f}bps")

    # Simulate high slippage
    print("\nSimulating high slippage trades...")
    bad_executions = [
        {"symbol": "XYZ", "expected": 100.00, "actual": 100.60, "shares": 200, "filled": True},  # 60bps
        {"symbol": "ABC", "expected": 50.00, "actual": 50.35, "shares": 300, "filled": True},   # 70bps
    ]

    for e in bad_executions:
        breaker.record_execution(
            symbol=e["symbol"],
            expected_price=e["expected"],
            actual_price=e["actual"],
            shares=e["shares"],
            was_filled=e["filled"],
        )

    result = breaker.check()
    print("\nAfter high slippage executions:")
    print(f"  Status: {result['status'].value}")
    print(f"  Action: {result['action'].value}")
    print(f"  Message: {result['message']}")

    print(f"\nExecution Summary: {breaker.get_execution_summary()}")
