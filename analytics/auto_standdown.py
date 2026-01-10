"""
Auto Stand-down for Edge Degradation Response.

Automatically reduces position sizing or pauses trading when
edge degradation is detected.

Stand-down Triggers:
- Win rate drops below threshold
- Profit factor degradation
- Consecutive losses exceed limit
- Drawdown exceeds threshold
- Rolling Sharpe turns negative
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StanddownSeverity(Enum):
    """Severity levels for stand-down recommendations."""
    NONE = auto()           # No action needed
    CAUTION = auto()        # Monitor closely
    REDUCE_SIZE = auto()    # Reduce position sizes
    REDUCE_FREQUENCY = auto()  # Reduce trade frequency
    PAUSE = auto()          # Pause new trades
    FULL_STOP = auto()      # Stop all trading


class StanddownTrigger(Enum):
    """Types of stand-down triggers."""
    WIN_RATE_DROP = auto()
    PROFIT_FACTOR_DROP = auto()
    CONSECUTIVE_LOSSES = auto()
    DRAWDOWN = auto()
    NEGATIVE_SHARPE = auto()
    VOLATILITY_SPIKE = auto()
    EDGE_SCORE_DROP = auto()
    MANUAL = auto()


@dataclass
class StanddownRecommendation:
    """Stand-down recommendation with details."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: StanddownSeverity = StanddownSeverity.NONE
    triggers: List[StanddownTrigger] = field(default_factory=list)
    trigger_details: Dict[str, Any] = field(default_factory=dict)

    position_scale: float = 1.0     # Multiplier for position sizes (0.0 to 1.0)
    frequency_scale: float = 1.0    # Multiplier for trade frequency
    max_positions: Optional[int] = None  # Override max positions

    recommendation_text: str = ""
    recovery_conditions: List[str] = field(default_factory=list)
    estimated_recovery_days: Optional[int] = None

    # Metrics at time of recommendation
    current_win_rate: Optional[float] = None
    current_profit_factor: Optional[float] = None
    current_drawdown_pct: Optional[float] = None
    consecutive_losses: int = 0
    rolling_sharpe: Optional[float] = None

    @property
    def is_active(self) -> bool:
        """Check if any stand-down is active."""
        return self.severity not in (StanddownSeverity.NONE, StanddownSeverity.CAUTION)

    @property
    def should_pause(self) -> bool:
        """Check if trading should be paused."""
        return self.severity in (StanddownSeverity.PAUSE, StanddownSeverity.FULL_STOP)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.name,
            "triggers": [t.name for t in self.triggers],
            "trigger_details": self.trigger_details,
            "position_scale": self.position_scale,
            "frequency_scale": self.frequency_scale,
            "max_positions": self.max_positions,
            "recommendation_text": self.recommendation_text,
            "recovery_conditions": self.recovery_conditions,
            "is_active": self.is_active,
            "should_pause": self.should_pause,
            "metrics": {
                "win_rate": self.current_win_rate,
                "profit_factor": self.current_profit_factor,
                "drawdown_pct": self.current_drawdown_pct,
                "consecutive_losses": self.consecutive_losses,
                "rolling_sharpe": self.rolling_sharpe,
            },
        }


@dataclass
class StanddownConfig:
    """Configuration for stand-down thresholds."""
    # Win rate thresholds
    min_win_rate_caution: float = 0.50  # Below this = caution
    min_win_rate_reduce: float = 0.45   # Below this = reduce size
    min_win_rate_pause: float = 0.35    # Below this = pause

    # Profit factor thresholds
    min_pf_caution: float = 1.2
    min_pf_reduce: float = 1.0
    min_pf_pause: float = 0.8

    # Consecutive losses
    max_consecutive_losses_caution: int = 3
    max_consecutive_losses_reduce: int = 5
    max_consecutive_losses_pause: int = 7

    # Drawdown thresholds
    max_drawdown_caution_pct: float = 5.0
    max_drawdown_reduce_pct: float = 10.0
    max_drawdown_pause_pct: float = 15.0

    # Rolling Sharpe (annualized)
    min_sharpe_caution: float = 0.5
    min_sharpe_reduce: float = 0.0
    min_sharpe_pause: float = -0.5

    # Position scaling factors
    caution_scale: float = 0.75
    reduce_scale: float = 0.50
    minimal_scale: float = 0.25

    # Recovery requirements
    recovery_win_rate: float = 0.55
    recovery_trades_required: int = 10
    recovery_profit_factor: float = 1.3

    # Lookback periods
    rolling_window_days: int = 30
    sharpe_window_days: int = 60


class AutoStanddown:
    """
    Automatic stand-down system for edge degradation.

    Monitors trading performance and recommends position reductions
    or trading pauses when edge degrades.
    """

    def __init__(self, config: Optional[StanddownConfig] = None):
        """
        Initialize auto stand-down system.

        Args:
            config: Stand-down configuration (uses defaults if None)
        """
        self.config = config or StanddownConfig()
        self._current_recommendation: Optional[StanddownRecommendation] = None
        self._recommendation_history: List[StanddownRecommendation] = []
        self._last_check: Optional[datetime] = None

    def check(
        self,
        trades_df: pd.DataFrame,
        equity_curve: Optional[pd.DataFrame] = None,
        current_pnl: float = 0.0,
        peak_equity: float = 100000.0,
    ) -> StanddownRecommendation:
        """
        Check current trading performance and generate recommendation.

        Args:
            trades_df: DataFrame of recent trades with 'pnl' column
            equity_curve: Optional equity curve DataFrame
            current_pnl: Current unrealized P&L
            peak_equity: Peak equity value

        Returns:
            StanddownRecommendation
        """
        self._last_check = datetime.utcnow()

        # Initialize recommendation
        rec = StanddownRecommendation()

        if trades_df.empty:
            return rec

        # Calculate metrics
        metrics = self._calculate_metrics(trades_df, equity_curve, current_pnl, peak_equity)

        # Store metrics in recommendation
        rec.current_win_rate = metrics.get("win_rate")
        rec.current_profit_factor = metrics.get("profit_factor")
        rec.current_drawdown_pct = metrics.get("drawdown_pct")
        rec.consecutive_losses = metrics.get("consecutive_losses", 0)
        rec.rolling_sharpe = metrics.get("sharpe")

        # Check each trigger type
        triggers = []
        trigger_details = {}

        # 1. Win rate check
        wr = metrics.get("win_rate", 0.55)
        if wr < self.config.min_win_rate_pause:
            triggers.append((StanddownTrigger.WIN_RATE_DROP, StanddownSeverity.PAUSE))
            trigger_details["win_rate"] = f"{wr:.1%} < {self.config.min_win_rate_pause:.1%}"
        elif wr < self.config.min_win_rate_reduce:
            triggers.append((StanddownTrigger.WIN_RATE_DROP, StanddownSeverity.REDUCE_SIZE))
            trigger_details["win_rate"] = f"{wr:.1%} < {self.config.min_win_rate_reduce:.1%}"
        elif wr < self.config.min_win_rate_caution:
            triggers.append((StanddownTrigger.WIN_RATE_DROP, StanddownSeverity.CAUTION))
            trigger_details["win_rate"] = f"{wr:.1%} < {self.config.min_win_rate_caution:.1%}"

        # 2. Profit factor check
        pf = metrics.get("profit_factor", 1.5)
        if pf < self.config.min_pf_pause:
            triggers.append((StanddownTrigger.PROFIT_FACTOR_DROP, StanddownSeverity.PAUSE))
            trigger_details["profit_factor"] = f"{pf:.2f} < {self.config.min_pf_pause:.2f}"
        elif pf < self.config.min_pf_reduce:
            triggers.append((StanddownTrigger.PROFIT_FACTOR_DROP, StanddownSeverity.REDUCE_SIZE))
            trigger_details["profit_factor"] = f"{pf:.2f} < {self.config.min_pf_reduce:.2f}"
        elif pf < self.config.min_pf_caution:
            triggers.append((StanddownTrigger.PROFIT_FACTOR_DROP, StanddownSeverity.CAUTION))
            trigger_details["profit_factor"] = f"{pf:.2f} < {self.config.min_pf_caution:.2f}"

        # 3. Consecutive losses check
        cons_losses = metrics.get("consecutive_losses", 0)
        if cons_losses >= self.config.max_consecutive_losses_pause:
            triggers.append((StanddownTrigger.CONSECUTIVE_LOSSES, StanddownSeverity.PAUSE))
            trigger_details["consecutive_losses"] = f"{cons_losses} >= {self.config.max_consecutive_losses_pause}"
        elif cons_losses >= self.config.max_consecutive_losses_reduce:
            triggers.append((StanddownTrigger.CONSECUTIVE_LOSSES, StanddownSeverity.REDUCE_SIZE))
            trigger_details["consecutive_losses"] = f"{cons_losses} >= {self.config.max_consecutive_losses_reduce}"
        elif cons_losses >= self.config.max_consecutive_losses_caution:
            triggers.append((StanddownTrigger.CONSECUTIVE_LOSSES, StanddownSeverity.CAUTION))
            trigger_details["consecutive_losses"] = f"{cons_losses} >= {self.config.max_consecutive_losses_caution}"

        # 4. Drawdown check
        dd_pct = metrics.get("drawdown_pct", 0)
        if dd_pct >= self.config.max_drawdown_pause_pct:
            triggers.append((StanddownTrigger.DRAWDOWN, StanddownSeverity.PAUSE))
            trigger_details["drawdown"] = f"{dd_pct:.1f}% >= {self.config.max_drawdown_pause_pct:.1f}%"
        elif dd_pct >= self.config.max_drawdown_reduce_pct:
            triggers.append((StanddownTrigger.DRAWDOWN, StanddownSeverity.REDUCE_SIZE))
            trigger_details["drawdown"] = f"{dd_pct:.1f}% >= {self.config.max_drawdown_reduce_pct:.1f}%"
        elif dd_pct >= self.config.max_drawdown_caution_pct:
            triggers.append((StanddownTrigger.DRAWDOWN, StanddownSeverity.CAUTION))
            trigger_details["drawdown"] = f"{dd_pct:.1f}% >= {self.config.max_drawdown_caution_pct:.1f}%"

        # 5. Sharpe check
        sharpe = metrics.get("sharpe")
        if sharpe is not None:
            if sharpe < self.config.min_sharpe_pause:
                triggers.append((StanddownTrigger.NEGATIVE_SHARPE, StanddownSeverity.PAUSE))
                trigger_details["sharpe"] = f"{sharpe:.2f} < {self.config.min_sharpe_pause:.2f}"
            elif sharpe < self.config.min_sharpe_reduce:
                triggers.append((StanddownTrigger.NEGATIVE_SHARPE, StanddownSeverity.REDUCE_SIZE))
                trigger_details["sharpe"] = f"{sharpe:.2f} < {self.config.min_sharpe_reduce:.2f}"
            elif sharpe < self.config.min_sharpe_caution:
                triggers.append((StanddownTrigger.NEGATIVE_SHARPE, StanddownSeverity.CAUTION))
                trigger_details["sharpe"] = f"{sharpe:.2f} < {self.config.min_sharpe_caution:.2f}"

        # Determine overall severity (take the most severe)
        if triggers:
            rec.triggers = [t[0] for t in triggers]
            rec.trigger_details = trigger_details

            severities = [t[1] for t in triggers]
            severity_order = [
                StanddownSeverity.FULL_STOP,
                StanddownSeverity.PAUSE,
                StanddownSeverity.REDUCE_FREQUENCY,
                StanddownSeverity.REDUCE_SIZE,
                StanddownSeverity.CAUTION,
                StanddownSeverity.NONE,
            ]

            for severity in severity_order:
                if severity in severities:
                    rec.severity = severity
                    break

            # Set position and frequency scales
            if rec.severity == StanddownSeverity.PAUSE:
                rec.position_scale = 0.0
                rec.frequency_scale = 0.0
                rec.max_positions = 0
                rec.recommendation_text = "PAUSE ALL TRADING: Multiple degradation signals detected"
            elif rec.severity == StanddownSeverity.REDUCE_SIZE:
                rec.position_scale = self.config.reduce_scale
                rec.frequency_scale = 0.75
                rec.recommendation_text = f"REDUCE POSITION SIZE to {rec.position_scale:.0%} of normal"
            elif rec.severity == StanddownSeverity.CAUTION:
                rec.position_scale = self.config.caution_scale
                rec.frequency_scale = 0.90
                rec.recommendation_text = f"CAUTION: Consider reducing to {rec.position_scale:.0%} of normal"
            else:
                rec.recommendation_text = "System operating normally"

            # Set recovery conditions
            rec.recovery_conditions = self._get_recovery_conditions()

        # Store recommendation
        self._current_recommendation = rec
        self._recommendation_history.append(rec)

        # Limit history
        if len(self._recommendation_history) > 1000:
            self._recommendation_history = self._recommendation_history[-500:]

        return rec

    def _calculate_metrics(
        self,
        trades_df: pd.DataFrame,
        equity_curve: Optional[pd.DataFrame],
        current_pnl: float,
        peak_equity: float,
    ) -> Dict[str, Any]:
        """Calculate performance metrics for stand-down check."""
        metrics = {}

        if len(trades_df) == 0:
            return metrics

        # Win rate
        wins = (trades_df["pnl"] > 0).sum()
        total = len(trades_df)
        metrics["win_rate"] = wins / total if total > 0 else 0

        # Profit factor
        gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades_df[trades_df["pnl"] <= 0]["pnl"].sum())
        metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Consecutive losses
        if "pnl" in trades_df.columns:
            pnl_signs = (trades_df["pnl"] > 0).values
            consecutive = 0
            max_consecutive = 0
            for win in reversed(pnl_signs):
                if not win:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    break
            metrics["consecutive_losses"] = consecutive

        # Drawdown
        if peak_equity > 0:
            current_equity = peak_equity + current_pnl + trades_df["pnl"].sum()
            metrics["drawdown_pct"] = (peak_equity - current_equity) / peak_equity * 100

        # Rolling Sharpe (if we have dates)
        if "exit_date" in trades_df.columns or "exit_time" in trades_df.columns:
            date_col = "exit_date" if "exit_date" in trades_df.columns else "exit_time"
            try:
                trades_df = trades_df.copy()
                trades_df["date"] = pd.to_datetime(trades_df[date_col]).dt.date
                daily_pnl = trades_df.groupby("date")["pnl"].sum()

                if len(daily_pnl) > 10:
                    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0
                    metrics["sharpe"] = sharpe
            except Exception:
                pass

        return metrics

    def _get_recovery_conditions(self) -> List[str]:
        """Get conditions required for recovery."""
        return [
            f"Win rate above {self.config.recovery_win_rate:.0%} for {self.config.recovery_trades_required} trades",
            f"Profit factor above {self.config.recovery_profit_factor:.2f}",
            "No consecutive losses exceeding 3",
            "Drawdown recovered to within 5% of peak",
        ]

    def get_current_scale(self) -> Tuple[float, float]:
        """
        Get current position and frequency scales.

        Returns:
            Tuple of (position_scale, frequency_scale)
        """
        if self._current_recommendation is None:
            return 1.0, 1.0
        return (
            self._current_recommendation.position_scale,
            self._current_recommendation.frequency_scale,
        )

    def should_trade(self) -> bool:
        """Check if trading is allowed."""
        if self._current_recommendation is None:
            return True
        return not self._current_recommendation.should_pause

    def get_position_multiplier(self) -> float:
        """Get position size multiplier (0.0 to 1.0)."""
        if self._current_recommendation is None:
            return 1.0
        return self._current_recommendation.position_scale

    def get_status(self) -> Dict[str, Any]:
        """Get current stand-down status."""
        if self._current_recommendation is None:
            return {
                "status": "NO_DATA",
                "severity": "NONE",
                "can_trade": True,
                "position_scale": 1.0,
                "last_check": None,
            }

        return {
            "status": "ACTIVE" if self._current_recommendation.is_active else "NORMAL",
            "severity": self._current_recommendation.severity.name,
            "can_trade": not self._current_recommendation.should_pause,
            "position_scale": self._current_recommendation.position_scale,
            "triggers": [t.name for t in self._current_recommendation.triggers],
            "recommendation": self._current_recommendation.recommendation_text,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "metrics": {
                "win_rate": self._current_recommendation.current_win_rate,
                "profit_factor": self._current_recommendation.current_profit_factor,
                "drawdown_pct": self._current_recommendation.current_drawdown_pct,
                "consecutive_losses": self._current_recommendation.consecutive_losses,
            },
        }

    def reset(self) -> None:
        """Reset stand-down state."""
        self._current_recommendation = None
        logger.info("Auto stand-down state reset")


# Default instance
DEFAULT_AUTO_STANDDOWN = AutoStanddown()
