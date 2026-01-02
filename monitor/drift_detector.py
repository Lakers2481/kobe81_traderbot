"""
Production Drift Detection for Model and Performance Monitoring.

Detects performance degradation and triggers automatic position scaling.
Wired into production pipelines for real-time edge monitoring.

Features:
- Rolling window performance tracking
- Multiple drift detection methods
- Position scaling recommendations
- Alert generation on significant drift
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DriftSeverity(Enum):
    """Severity levels for detected drift."""
    NONE = auto()        # No significant drift
    MINOR = auto()       # Small degradation, monitor closely
    MODERATE = auto()    # Notable degradation, reduce size
    SEVERE = auto()      # Major degradation, minimal exposure
    CRITICAL = auto()    # System failure, halt trading


@dataclass
class DriftThresholds:
    """Thresholds for drift detection."""
    # Relative degradation thresholds
    min_delta_accuracy: float = -0.02   # Accuracy drop > 2%
    min_delta_wr: float = -0.02         # Win-rate drop > 2%
    min_delta_pf: float = -0.10         # Profit factor drop > 0.10
    min_delta_sharpe: float = -0.10     # Sharpe drop > 0.10

    # Absolute thresholds
    min_absolute_wr: float = 0.50       # WR below 50% = problem
    min_absolute_pf: float = 1.0        # PF below 1.0 = losing money
    min_absolute_sharpe: float = 0.0    # Sharpe below 0 = negative risk-adjusted

    # Severity thresholds (cumulative degradation)
    minor_drift_threshold: float = 0.02
    moderate_drift_threshold: float = 0.05
    severe_drift_threshold: float = 0.10
    critical_drift_threshold: float = 0.20

    # Position scaling factors by severity
    scale_minor: float = 0.80           # 80% position size
    scale_moderate: float = 0.50        # 50% position size
    scale_severe: float = 0.25          # 25% position size
    scale_critical: float = 0.0         # No new positions


@dataclass
class DriftResult:
    """Result of drift detection analysis."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: DriftSeverity = DriftSeverity.NONE
    position_scale: float = 1.0         # 0.0 to 1.0
    drifted_metrics: List[str] = field(default_factory=list)
    degradation_scores: Dict[str, float] = field(default_factory=dict)
    message: str = ""

    @property
    def has_drift(self) -> bool:
        return self.severity != DriftSeverity.NONE

    @property
    def should_halt(self) -> bool:
        return self.severity == DriftSeverity.CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.name,
            "position_scale": round(self.position_scale, 2),
            "drifted_metrics": self.drifted_metrics,
            "degradation_scores": {k: round(v, 4) for k, v in self.degradation_scores.items()},
            "message": self.message,
            "has_drift": self.has_drift,
            "should_halt": self.should_halt,
        }


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time."""
    timestamp: datetime
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    accuracy: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "accuracy": round(self.accuracy, 4),
            "total_trades": self.total_trades,
            "total_pnl": round(self.total_pnl, 2),
            "max_drawdown": round(self.max_drawdown, 4),
        }


class DriftDetector:
    """
    Production-grade drift detector with position scaling.

    Monitors performance metrics over rolling windows and
    recommends position size adjustments on degradation.
    """

    def __init__(
        self,
        thresholds: Optional[DriftThresholds] = None,
        baseline_window_days: int = 90,
        recent_window_days: int = 14,
        state_path: Optional[str] = None,
    ):
        """
        Initialize drift detector.

        Args:
            thresholds: Detection thresholds
            baseline_window_days: Days for baseline performance
            recent_window_days: Days for recent performance
            state_path: Path to persist state
        """
        self.thresholds = thresholds or DriftThresholds()
        self.baseline_window_days = baseline_window_days
        self.recent_window_days = recent_window_days
        self.state_path = Path(state_path) if state_path else Path("state/drift_state.json")

        # Performance history
        self._snapshots: List[PerformanceSnapshot] = []
        self._last_result: Optional[DriftResult] = None
        self._current_scale: float = 1.0

        # Load persisted state
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted drift state."""
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                self._current_scale = data.get("current_scale", 1.0)
                # Restore snapshots
                for snap_data in data.get("snapshots", []):
                    self._snapshots.append(PerformanceSnapshot(
                        timestamp=datetime.fromisoformat(snap_data["timestamp"]),
                        win_rate=snap_data["win_rate"],
                        profit_factor=snap_data["profit_factor"],
                        sharpe_ratio=snap_data["sharpe_ratio"],
                        accuracy=snap_data.get("accuracy", 0.0),
                        total_trades=snap_data.get("total_trades", 0),
                        total_pnl=snap_data.get("total_pnl", 0.0),
                        max_drawdown=snap_data.get("max_drawdown", 0.0),
                    ))
                logger.info(f"Loaded drift state: scale={self._current_scale}, snapshots={len(self._snapshots)}")
            except Exception as e:
                logger.warning(f"Failed to load drift state: {e}")

    def _save_state(self) -> None:
        """Persist drift state."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "current_scale": self._current_scale,
                "last_updated": datetime.utcnow().isoformat(),
                "snapshots": [s.to_dict() for s in self._snapshots[-100:]],  # Keep last 100
            }
            with open(self.state_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save drift state: {e}")

    def record_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """
        Record a new performance snapshot.

        Args:
            snapshot: Performance metrics snapshot
        """
        self._snapshots.append(snapshot)

        # Prune old snapshots (keep ~2x baseline window)
        cutoff = datetime.utcnow() - timedelta(days=self.baseline_window_days * 2)
        self._snapshots = [s for s in self._snapshots if s.timestamp > cutoff]

        self._save_state()
        logger.debug(f"Recorded snapshot: WR={snapshot.win_rate:.2%}, PF={snapshot.profit_factor:.2f}")

    def check(
        self,
        current_metrics: Optional[Dict[str, float]] = None,
        trades_df=None,  # Optional pandas DataFrame
    ) -> DriftResult:
        """
        Check for performance drift.

        Args:
            current_metrics: Dict with 'wr', 'pf', 'sharpe', 'accuracy' keys
            trades_df: DataFrame of recent trades (alternative to current_metrics)

        Returns:
            DriftResult with severity and recommended position scale
        """
        # Calculate current metrics from trades if provided
        if trades_df is not None and len(trades_df) > 0:
            current_metrics = self._calculate_metrics_from_trades(trades_df)

        if not current_metrics:
            return DriftResult(message="No metrics provided")

        # Get baseline metrics
        baseline = self._get_baseline_metrics()
        if not baseline:
            logger.info("No baseline metrics yet, using current as baseline")
            self.record_snapshot(PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                win_rate=current_metrics.get("wr", 0.5),
                profit_factor=current_metrics.get("pf", 1.0),
                sharpe_ratio=current_metrics.get("sharpe", 0.0),
                accuracy=current_metrics.get("accuracy", 0.0),
            ))
            return DriftResult(message="Baseline established")

        # Calculate degradation
        degradation = self._calculate_degradation(baseline, current_metrics)

        # Determine severity
        severity, position_scale = self._determine_severity(current_metrics, degradation)

        # Build result
        drifted = [k for k, v in degradation.items() if v < 0 and abs(v) > 0.01]

        result = DriftResult(
            severity=severity,
            position_scale=position_scale,
            drifted_metrics=drifted,
            degradation_scores=degradation,
            message=self._generate_message(severity, drifted, degradation),
        )

        # Update internal state
        self._last_result = result
        self._current_scale = position_scale
        self._save_state()

        if result.has_drift:
            logger.warning(f"Drift detected: {result.message}")

        return result

    def _calculate_metrics_from_trades(self, trades_df) -> Dict[str, float]:
        """Calculate metrics from trades DataFrame."""
        try:
            wins = (trades_df["pnl"] > 0).sum()
            total = len(trades_df)
            wr = wins / total if total > 0 else 0.5

            gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
            gross_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
            pf = gross_profit / gross_loss if gross_loss > 0 else 1.0

            # Simplified Sharpe (daily returns approximation)
            if "return" in trades_df.columns:
                mean_ret = trades_df["return"].mean()
                std_ret = trades_df["return"].std()
                sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
            else:
                sharpe = 0.0

            return {
                "wr": wr,
                "pf": pf,
                "sharpe": sharpe,
                "accuracy": wr,  # Use WR as accuracy proxy
            }
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return {}

    def _get_baseline_metrics(self) -> Optional[Dict[str, float]]:
        """Get baseline performance metrics from history."""
        if not self._snapshots:
            return None

        # Filter to baseline window
        cutoff = datetime.utcnow() - timedelta(days=self.baseline_window_days)
        baseline_snapshots = [s for s in self._snapshots if s.timestamp > cutoff]

        if not baseline_snapshots:
            return None

        # Average metrics across baseline period
        return {
            "wr": np.mean([s.win_rate for s in baseline_snapshots]),
            "pf": np.mean([s.profit_factor for s in baseline_snapshots]),
            "sharpe": np.mean([s.sharpe_ratio for s in baseline_snapshots]),
            "accuracy": np.mean([s.accuracy for s in baseline_snapshots]),
        }

    def _calculate_degradation(
        self,
        baseline: Dict[str, float],
        current: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate relative degradation from baseline."""
        degradation = {}

        for key in ["wr", "pf", "sharpe", "accuracy"]:
            base_val = baseline.get(key, 0)
            cur_val = current.get(key, 0)

            if base_val != 0:
                degradation[key] = (cur_val - base_val) / abs(base_val)
            else:
                degradation[key] = cur_val

        return degradation

    def _determine_severity(
        self,
        current: Dict[str, float],
        degradation: Dict[str, float],
    ) -> Tuple[DriftSeverity, float]:
        """Determine drift severity and position scale."""
        thr = self.thresholds

        # Check absolute thresholds first
        wr = current.get("wr", 0.5)
        pf = current.get("pf", 1.0)
        sharpe = current.get("sharpe", 0.0)

        if wr < 0.40 or pf < 0.7:  # Severe absolute degradation
            return DriftSeverity.CRITICAL, thr.scale_critical

        if wr < thr.min_absolute_wr or pf < thr.min_absolute_pf:
            return DriftSeverity.SEVERE, thr.scale_severe

        if sharpe < thr.min_absolute_sharpe:
            # Negative Sharpe is concerning
            return DriftSeverity.MODERATE, thr.scale_moderate

        # Check relative degradation
        avg_degradation = np.mean([
            abs(degradation.get("wr", 0)),
            abs(degradation.get("pf", 0)),
            abs(degradation.get("sharpe", 0)),
        ])

        if avg_degradation >= thr.critical_drift_threshold:
            return DriftSeverity.CRITICAL, thr.scale_critical
        elif avg_degradation >= thr.severe_drift_threshold:
            return DriftSeverity.SEVERE, thr.scale_severe
        elif avg_degradation >= thr.moderate_drift_threshold:
            return DriftSeverity.MODERATE, thr.scale_moderate
        elif avg_degradation >= thr.minor_drift_threshold:
            return DriftSeverity.MINOR, thr.scale_minor
        else:
            return DriftSeverity.NONE, 1.0

    def _generate_message(
        self,
        severity: DriftSeverity,
        drifted: List[str],
        degradation: Dict[str, float],
    ) -> str:
        """Generate human-readable drift message."""
        if severity == DriftSeverity.NONE:
            return "Performance within normal bounds"

        metrics_str = ", ".join([
            f"{k}: {degradation[k]:+.1%}" for k in drifted
        ])

        severity_msg = {
            DriftSeverity.MINOR: "Minor drift detected",
            DriftSeverity.MODERATE: "Moderate degradation detected",
            DriftSeverity.SEVERE: "Severe performance degradation",
            DriftSeverity.CRITICAL: "CRITICAL: System performance failure",
        }

        return f"{severity_msg.get(severity, 'Drift detected')}: {metrics_str}"

    def get_position_scale(self) -> float:
        """
        Get current position scaling factor.

        Returns:
            Float between 0.0 and 1.0 to multiply position sizes by
        """
        return self._current_scale

    def reset(self) -> None:
        """Reset drift detector to clean state."""
        self._snapshots = []
        self._last_result = None
        self._current_scale = 1.0
        if self.state_path.exists():
            self.state_path.unlink()
        logger.info("Drift detector reset")


# Legacy function for backward compatibility
def compare_metrics(
    prev: Dict[str, float],
    cur: Dict[str, float],
    thr: Optional[DriftThresholds] = None,
) -> Dict[str, bool]:
    """
    Legacy function: Compare two metric snapshots for drift.

    Args:
        prev: Previous metrics dict
        cur: Current metrics dict
        thr: Optional thresholds

    Returns:
        Dict with drift flags per metric
    """
    thr = thr or DriftThresholds()

    def _delta(k: str) -> float:
        return float(cur.get(k, 0.0)) - float(prev.get(k, 0.0))

    out = {
        "accuracy_drift": _delta("accuracy") <= thr.min_delta_accuracy,
        "wr_drift": _delta("wr") <= thr.min_delta_wr,
        "pf_drift": _delta("pf") <= thr.min_delta_pf,
        "sharpe_drift": _delta("sharpe") <= thr.min_delta_sharpe,
    }
    out["any_drift"] = any(out.values())
    return out


# Global detector instance
_global_detector: Optional[DriftDetector] = None


def get_drift_detector() -> DriftDetector:
    """Get or create global drift detector instance."""
    global _global_detector
    if _global_detector is None:
        _global_detector = DriftDetector()
    return _global_detector


def check_drift(current_metrics: Dict[str, float]) -> DriftResult:
    """Convenience function to check drift using global detector."""
    return get_drift_detector().check(current_metrics=current_metrics)


def get_position_scale() -> float:
    """Get current position scale from global detector."""
    return get_drift_detector().get_position_scale()
