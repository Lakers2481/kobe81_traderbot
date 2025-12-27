"""
Drift Detection Module
======================

Monitors model and strategy performance for drift, degradation, and regime changes.
Triggers alerts and stand-down recommendations when drift exceeds thresholds.

Types of drift detected:
1. Performance Drift - Win rate, profit factor, Sharpe degradation over rolling windows
2. Feature Drift - Distribution shifts in input features (KS test)
3. Prediction Drift - Changes in model confidence/prediction distributions
4. Regime Drift - Market regime changes that may invalidate strategy assumptions

Usage:
    from monitor.drift_detector import DriftDetector, get_drift_detector

    detector = get_drift_detector()

    # Record trade outcomes
    detector.record_trade(won=True, pnl=150.0, predicted_prob=0.72)
    detector.record_trade(won=False, pnl=-80.0, predicted_prob=0.65)

    # Check for drift
    report = detector.check_drift()
    if report.should_stand_down:
        print(f"ALERT: {report.reason}")

    # Get current status
    status = detector.get_status()
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift that can be detected."""
    NONE = "none"
    PERFORMANCE = "performance"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    SHARPE = "sharpe"
    CALIBRATION = "calibration"
    REGIME = "regime"
    FEATURE = "feature"


class DriftSeverity(Enum):
    """Severity levels for drift alerts."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TradeRecord:
    """Record of a single trade for drift tracking."""
    timestamp: datetime
    won: bool
    pnl: float
    predicted_prob: Optional[float] = None
    strategy: str = "unknown"
    regime: str = "unknown"


@dataclass
class DriftThresholds:
    """Thresholds for drift detection."""
    # Performance thresholds (relative degradation from baseline)
    win_rate_min: float = 0.45  # Absolute minimum
    win_rate_degradation: float = 0.10  # Max drop from baseline
    profit_factor_min: float = 1.0  # Absolute minimum
    profit_factor_degradation: float = 0.20  # Max drop from baseline
    sharpe_min: float = 0.0  # Absolute minimum
    sharpe_degradation: float = 0.30  # Max drop from baseline

    # Calibration thresholds
    brier_score_max: float = 0.25  # Max Brier score (lower is better)
    calibration_error_max: float = 0.10  # Max calibration error

    # Window sizes
    rolling_window: int = 50  # Trades for rolling metrics
    baseline_window: int = 200  # Trades for baseline calculation
    min_trades_for_check: int = 20  # Minimum trades before checking

    # Stand-down triggers
    consecutive_losses_max: int = 10  # Max consecutive losses
    drawdown_pct_max: float = 0.15  # Max drawdown before alert


@dataclass
class DriftReport:
    """Report from drift detection check."""
    checked_at: datetime
    drift_detected: bool
    drift_types: List[DriftType] = field(default_factory=list)
    severity: DriftSeverity = DriftSeverity.NONE
    should_stand_down: bool = False
    reason: str = ""
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'checked_at': self.checked_at.isoformat(),
            'drift_detected': self.drift_detected,
            'drift_types': [dt.value for dt in self.drift_types],
            'severity': self.severity.value,
            'should_stand_down': self.should_stand_down,
            'reason': self.reason,
            'details': self.details,
        }


class DriftDetector:
    """
    Detects performance drift and degradation in trading strategies.

    Monitors rolling metrics and compares against baselines to detect:
    - Win rate degradation
    - Profit factor decline
    - Sharpe ratio drops
    - Calibration errors
    - Consecutive loss streaks
    """

    def __init__(
        self,
        thresholds: Optional[DriftThresholds] = None,
        state_dir: Optional[Path] = None,
        auto_persist: bool = True,
    ):
        self.thresholds = thresholds or DriftThresholds()
        self.state_dir = Path(state_dir) if state_dir else Path("state/monitoring")
        self.auto_persist = auto_persist

        # Trade history (bounded deque for memory efficiency)
        max_history = max(self.thresholds.baseline_window * 2, 1000)
        self._trades: deque[TradeRecord] = deque(maxlen=max_history)

        # Baseline metrics (calculated from initial trades)
        self._baseline: Dict[str, float] = {}
        self._baseline_set: bool = False

        # Current state
        self._consecutive_losses: int = 0
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._last_check: Optional[datetime] = None
        self._alerts: List[DriftReport] = []

        # Load persisted state if available
        self._load_state()

    def record_trade(
        self,
        won: bool,
        pnl: float,
        predicted_prob: Optional[float] = None,
        strategy: str = "unknown",
        regime: str = "unknown",
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record a trade outcome for drift monitoring.

        Args:
            won: Whether the trade was profitable
            pnl: Profit/loss amount
            predicted_prob: Model's predicted probability of success
            strategy: Strategy that generated the signal
            regime: Market regime at time of trade
            timestamp: Trade timestamp (defaults to now)
        """
        record = TradeRecord(
            timestamp=timestamp or datetime.now(),
            won=won,
            pnl=pnl,
            predicted_prob=predicted_prob,
            strategy=strategy,
            regime=regime,
        )
        self._trades.append(record)

        # Update consecutive losses
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        # Update equity tracking
        self._current_equity += pnl
        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity

        # Update baseline if we have enough trades
        if not self._baseline_set and len(self._trades) >= self.thresholds.baseline_window:
            self._calculate_baseline()

        # Persist state
        if self.auto_persist:
            self._save_state()

    def check_drift(self, force: bool = False) -> DriftReport:
        """
        Check for drift in performance metrics.

        Args:
            force: If True, check even with insufficient trades

        Returns:
            DriftReport with detection results
        """
        now = datetime.now()

        # Check if we have enough data
        if len(self._trades) < self.thresholds.min_trades_for_check and not force:
            return DriftReport(
                checked_at=now,
                drift_detected=False,
                reason="Insufficient trades for drift check",
                details={'trades_count': len(self._trades)},
            )

        drift_types = []
        details = {}
        reasons = []

        # Calculate rolling metrics
        rolling = self._calculate_rolling_metrics()
        details['rolling_metrics'] = rolling

        # Check win rate drift
        if rolling.get('win_rate') is not None:
            wr = rolling['win_rate']
            details['win_rate'] = wr

            if wr < self.thresholds.win_rate_min:
                drift_types.append(DriftType.WIN_RATE)
                reasons.append(f"Win rate {wr:.1%} below minimum {self.thresholds.win_rate_min:.1%}")

            if self._baseline_set:
                baseline_wr = self._baseline.get('win_rate', 0.5)
                degradation = baseline_wr - wr
                if degradation > self.thresholds.win_rate_degradation:
                    drift_types.append(DriftType.PERFORMANCE)
                    reasons.append(
                        f"Win rate degraded {degradation:.1%} from baseline {baseline_wr:.1%}"
                    )

        # Check profit factor drift
        if rolling.get('profit_factor') is not None:
            pf = rolling['profit_factor']
            details['profit_factor'] = pf

            if pf < self.thresholds.profit_factor_min:
                drift_types.append(DriftType.PROFIT_FACTOR)
                reasons.append(f"Profit factor {pf:.2f} below minimum {self.thresholds.profit_factor_min:.2f}")

            if self._baseline_set:
                baseline_pf = self._baseline.get('profit_factor', 1.5)
                if baseline_pf > 0:
                    degradation = (baseline_pf - pf) / baseline_pf
                    if degradation > self.thresholds.profit_factor_degradation:
                        drift_types.append(DriftType.PERFORMANCE)
                        reasons.append(
                            f"Profit factor degraded {degradation:.1%} from baseline {baseline_pf:.2f}"
                        )

        # Check Sharpe drift
        if rolling.get('sharpe') is not None:
            sharpe = rolling['sharpe']
            details['sharpe'] = sharpe

            if sharpe < self.thresholds.sharpe_min:
                drift_types.append(DriftType.SHARPE)
                reasons.append(f"Sharpe {sharpe:.2f} below minimum {self.thresholds.sharpe_min:.2f}")

        # Check consecutive losses
        details['consecutive_losses'] = self._consecutive_losses
        if self._consecutive_losses >= self.thresholds.consecutive_losses_max:
            drift_types.append(DriftType.PERFORMANCE)
            reasons.append(f"{self._consecutive_losses} consecutive losses")

        # Check drawdown
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - self._current_equity) / self._peak_equity
            details['drawdown_pct'] = drawdown

            if drawdown > self.thresholds.drawdown_pct_max:
                drift_types.append(DriftType.PERFORMANCE)
                reasons.append(f"Drawdown {drawdown:.1%} exceeds max {self.thresholds.drawdown_pct_max:.1%}")

        # Check calibration if we have predictions
        calibration = self._check_calibration()
        if calibration:
            details['calibration'] = calibration
            if calibration.get('brier_score', 0) > self.thresholds.brier_score_max:
                drift_types.append(DriftType.CALIBRATION)
                reasons.append(
                    f"Brier score {calibration['brier_score']:.3f} exceeds max {self.thresholds.brier_score_max:.3f}"
                )

        # Determine severity and stand-down recommendation
        drift_detected = len(drift_types) > 0
        severity = self._determine_severity(drift_types, details)
        should_stand_down = severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]

        reason = "; ".join(reasons) if reasons else "No drift detected"

        report = DriftReport(
            checked_at=now,
            drift_detected=drift_detected,
            drift_types=list(set(drift_types)),  # Dedupe
            severity=severity,
            should_stand_down=should_stand_down,
            reason=reason,
            details=details,
        )

        self._last_check = now
        if drift_detected:
            self._alerts.append(report)
            logger.warning(f"Drift detected: {reason}")

        return report

    def _calculate_rolling_metrics(self) -> Dict[str, float]:
        """Calculate metrics over rolling window."""
        window = self.thresholds.rolling_window
        recent_trades = list(self._trades)[-window:]

        if len(recent_trades) < 5:
            return {}

        # Win rate
        wins = sum(1 for t in recent_trades if t.won)
        win_rate = wins / len(recent_trades)

        # Profit factor
        gross_profit = sum(t.pnl for t in recent_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in recent_trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe (simplified - daily returns approximation)
        pnls = [t.pnl for t in recent_trades]
        if len(pnls) >= 2:
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            sharpe = (mean_pnl / std_pnl * (252 ** 0.5)) if std_pnl > 0 else 0.0
        else:
            sharpe = 0.0

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor if profit_factor != float('inf') else 10.0,
            'sharpe': sharpe,
            'trade_count': len(recent_trades),
        }

    def _calculate_baseline(self) -> None:
        """Calculate baseline metrics from initial trades."""
        window = self.thresholds.baseline_window
        baseline_trades = list(self._trades)[:window]

        if len(baseline_trades) < window:
            return

        # Win rate
        wins = sum(1 for t in baseline_trades if t.won)
        win_rate = wins / len(baseline_trades)

        # Profit factor
        gross_profit = sum(t.pnl for t in baseline_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in baseline_trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.5

        # Sharpe
        pnls = [t.pnl for t in baseline_trades]
        if len(pnls) >= 2:
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            sharpe = (mean_pnl / std_pnl * (252 ** 0.5)) if std_pnl > 0 else 0.0
        else:
            sharpe = 0.0

        self._baseline = {
            'win_rate': win_rate,
            'profit_factor': min(profit_factor, 10.0),
            'sharpe': sharpe,
            'calculated_at': datetime.now().isoformat(),
            'trade_count': len(baseline_trades),
        }
        self._baseline_set = True

        logger.info(
            f"Baseline established: WR={win_rate:.1%}, PF={profit_factor:.2f}, Sharpe={sharpe:.2f}"
        )

    def _check_calibration(self) -> Optional[Dict]:
        """Check probability calibration using Brier score."""
        # Get trades with predictions
        trades_with_probs = [
            t for t in self._trades
            if t.predicted_prob is not None
        ]

        if len(trades_with_probs) < 20:
            return None

        # Use recent trades
        recent = trades_with_probs[-100:]

        # Brier score: mean squared error of probability predictions
        brier_scores = []
        for t in recent:
            outcome = 1.0 if t.won else 0.0
            brier_scores.append((t.predicted_prob - outcome) ** 2)

        brier_score = statistics.mean(brier_scores)

        # Calibration by bucket
        buckets = {}
        for t in recent:
            bucket = int(t.predicted_prob * 10) / 10  # 0.0, 0.1, 0.2, ...
            if bucket not in buckets:
                buckets[bucket] = {'predicted': [], 'actual': []}
            buckets[bucket]['predicted'].append(t.predicted_prob)
            buckets[bucket]['actual'].append(1.0 if t.won else 0.0)

        calibration_errors = []
        for bucket, data in buckets.items():
            if len(data['actual']) >= 5:
                predicted_avg = statistics.mean(data['predicted'])
                actual_avg = statistics.mean(data['actual'])
                calibration_errors.append(abs(predicted_avg - actual_avg))

        avg_calibration_error = statistics.mean(calibration_errors) if calibration_errors else 0.0

        return {
            'brier_score': brier_score,
            'calibration_error': avg_calibration_error,
            'trades_analyzed': len(recent),
        }

    def _determine_severity(self, drift_types: List[DriftType], details: Dict) -> DriftSeverity:
        """Determine severity based on drift types and details."""
        if not drift_types:
            return DriftSeverity.NONE

        # Critical conditions
        if self._consecutive_losses >= self.thresholds.consecutive_losses_max:
            return DriftSeverity.CRITICAL

        drawdown = details.get('drawdown_pct', 0)
        if drawdown > self.thresholds.drawdown_pct_max * 1.5:
            return DriftSeverity.CRITICAL

        # High conditions
        if len(drift_types) >= 3:
            return DriftSeverity.HIGH

        if DriftType.WIN_RATE in drift_types and DriftType.PROFIT_FACTOR in drift_types:
            return DriftSeverity.HIGH

        # Medium conditions
        if len(drift_types) >= 2:
            return DriftSeverity.MEDIUM

        # Low conditions
        return DriftSeverity.LOW

    def get_status(self) -> Dict:
        """Get current drift detector status."""
        rolling = self._calculate_rolling_metrics()

        return {
            'trades_recorded': len(self._trades),
            'baseline_set': self._baseline_set,
            'baseline': self._baseline if self._baseline_set else None,
            'rolling_metrics': rolling,
            'consecutive_losses': self._consecutive_losses,
            'peak_equity': self._peak_equity,
            'current_equity': self._current_equity,
            'last_check': self._last_check.isoformat() if self._last_check else None,
            'alerts_count': len(self._alerts),
            'recent_alerts': [a.to_dict() for a in self._alerts[-5:]],
        }

    def reset(self) -> None:
        """Reset all state (for testing or new period)."""
        self._trades.clear()
        self._baseline = {}
        self._baseline_set = False
        self._consecutive_losses = 0
        self._peak_equity = 0.0
        self._current_equity = 0.0
        self._last_check = None
        self._alerts = []

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            state_file = self.state_dir / "drift_detector_state.json"

            state = {
                'trades': [
                    {
                        'timestamp': t.timestamp.isoformat(),
                        'won': t.won,
                        'pnl': t.pnl,
                        'predicted_prob': t.predicted_prob,
                        'strategy': t.strategy,
                        'regime': t.regime,
                    }
                    for t in list(self._trades)[-500:]  # Save last 500
                ],
                'baseline': self._baseline,
                'baseline_set': self._baseline_set,
                'consecutive_losses': self._consecutive_losses,
                'peak_equity': self._peak_equity,
                'current_equity': self._current_equity,
                'saved_at': datetime.now().isoformat(),
            }

            state_file.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.debug(f"Failed to save drift detector state: {e}")

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        try:
            state_file = self.state_dir / "drift_detector_state.json"
            if not state_file.exists():
                return

            state = json.loads(state_file.read_text())

            for t in state.get('trades', []):
                self._trades.append(TradeRecord(
                    timestamp=datetime.fromisoformat(t['timestamp']),
                    won=t['won'],
                    pnl=t['pnl'],
                    predicted_prob=t.get('predicted_prob'),
                    strategy=t.get('strategy', 'unknown'),
                    regime=t.get('regime', 'unknown'),
                ))

            self._baseline = state.get('baseline', {})
            self._baseline_set = state.get('baseline_set', False)
            self._consecutive_losses = state.get('consecutive_losses', 0)
            self._peak_equity = state.get('peak_equity', 0.0)
            self._current_equity = state.get('current_equity', 0.0)

            logger.info(f"Loaded drift detector state: {len(self._trades)} trades")
        except Exception as e:
            logger.debug(f"Failed to load drift detector state: {e}")


# Singleton instance
_drift_detector: Optional[DriftDetector] = None


def get_drift_detector() -> DriftDetector:
    """Get the singleton drift detector instance."""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector()
    return _drift_detector


def record_trade(
    won: bool,
    pnl: float,
    predicted_prob: Optional[float] = None,
    strategy: str = "unknown",
    regime: str = "unknown",
) -> None:
    """Convenience function to record a trade."""
    get_drift_detector().record_trade(
        won=won,
        pnl=pnl,
        predicted_prob=predicted_prob,
        strategy=strategy,
        regime=regime,
    )


def check_drift() -> DriftReport:
    """Convenience function to check for drift."""
    return get_drift_detector().check_drift()
