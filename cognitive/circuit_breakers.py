"""
Cognitive Safety Circuit Breakers
==================================

Detects and halts cognitive failure modes before they can cause harm.

Top 3 Failure Modes (from Gemini AI Safety Review):
1. Overfitting to Anomalies - Learning from black swan events
2. Self-Model Corruption - Calibration drift to always-accept or always-reject
3. Feedback Loop on Bad Trades - Reinforcing wrong patterns

These circuit breakers monitor the cognitive system in real-time
and can halt trading if unsafe conditions are detected.

Usage:
    from cognitive.circuit_breakers import CognitiveSafetyMonitor, get_safety_monitor

    monitor = get_safety_monitor()
    is_safe, violations = monitor.check_all()
    if not is_safe:
        # Trading should be halted
        for v in violations:
            print(f"VIOLATION: {v}")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import json

logger = logging.getLogger(__name__)

# Global singleton
_safety_monitor: Optional['CognitiveSafetyMonitor'] = None


class BreakerId(Enum):
    """Identifiers for circuit breakers."""
    CALIBRATION_DRIFT = "calibration_drift"
    EPISODE_CONCENTRATION = "episode_concentration"
    CONFIDENCE_VARIANCE = "confidence_variance"
    REGIME_STUCK = "regime_stuck"
    LOSS_STREAK = "loss_streak"
    MODEL_COLLAPSE = "model_collapse"


@dataclass
class CircuitBreakerViolation:
    """Record of a circuit breaker violation."""
    breaker_id: BreakerId
    message: str
    severity: str  # "warning", "critical"
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'breaker_id': self.breaker_id.value,
            'message': self.message,
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details
        }


class CognitiveSafetyMonitor:
    """
    Real-time monitoring of cognitive system safety.

    Implements circuit breakers that can halt trading when
    dangerous conditions are detected.
    """

    def __init__(
        self,
        # Calibration drift thresholds
        calibration_extreme_low: float = 0.05,
        calibration_extreme_high: float = 0.95,
        # Episode concentration thresholds
        max_loss_concentration: float = 0.80,  # >80% losses is concerning
        min_episodes_for_check: int = 10,
        # Confidence variance thresholds
        min_confidence_variance: float = 0.01,  # If variance <1%, model may have collapsed
        confidence_window_size: int = 50,
        # Regime detection thresholds
        max_days_in_same_regime: int = 60,  # Stuck >60 days is suspicious
        # Loss streak thresholds
        max_consecutive_losses: int = 10,
        # State directory
        state_dir: str = "state/cognitive",
    ):
        self.calibration_extreme_low = calibration_extreme_low
        self.calibration_extreme_high = calibration_extreme_high
        self.max_loss_concentration = max_loss_concentration
        self.min_episodes_for_check = min_episodes_for_check
        self.min_confidence_variance = min_confidence_variance
        self.confidence_window_size = confidence_window_size
        self.max_days_in_same_regime = max_days_in_same_regime
        self.max_consecutive_losses = max_consecutive_losses
        self.state_dir = Path(state_dir)

        # Recent confidence history for variance check
        self._recent_confidences: List[float] = []

        # Violation history
        self._violations: List[CircuitBreakerViolation] = []

        logger.info("CognitiveSafetyMonitor initialized")

    def check_all(self) -> Tuple[bool, List[CircuitBreakerViolation]]:
        """
        Run all circuit breaker checks.

        Returns:
            (is_safe, violations): is_safe=False means trading should halt
        """
        violations = []

        # Check each breaker
        breakers = [
            self._check_calibration_drift,
            self._check_episode_concentration,
            self._check_confidence_variance,
            self._check_regime_stuck,
            self._check_loss_streak,
        ]

        for check_func in breakers:
            try:
                result = check_func()
                if result is not None:
                    violations.append(result)
                    self._violations.append(result)
            except Exception as e:
                logger.warning(f"Breaker check failed: {check_func.__name__}: {e}")

        # Log violations
        for v in violations:
            if v.severity == "critical":
                logger.error(f"CIRCUIT BREAKER TRIPPED: {v.breaker_id.value} - {v.message}")
            else:
                logger.warning(f"CIRCUIT BREAKER WARNING: {v.breaker_id.value} - {v.message}")

            # Log to structured log
            try:
                from core.structured_log import jlog
                jlog('circuit_breaker_violation', **v.to_dict())
            except ImportError:
                pass

        is_safe = not any(v.severity == "critical" for v in violations)
        return is_safe, violations

    def record_confidence(self, confidence: float) -> None:
        """
        Record a confidence score for variance monitoring.

        Call this after each prediction to track confidence distribution.
        """
        self._recent_confidences.append(confidence)

        # Keep only recent window
        if len(self._recent_confidences) > self.confidence_window_size:
            self._recent_confidences = self._recent_confidences[-self.confidence_window_size:]

    def _check_calibration_drift(self) -> Optional[CircuitBreakerViolation]:
        """
        Check if calibration has drifted to extremes.

        FAILURE MODE: Self-model always accepts or always rejects trades.
        """
        try:
            from cognitive.self_model import get_self_model

            model = get_self_model()

            # Get recent calibration values
            if not hasattr(model, '_calibration_history') or len(model._calibration_history) < 10:
                return None  # Not enough data

            recent = list(model._calibration_history)[-50:]
            avg = sum(recent) / len(recent)

            if avg < self.calibration_extreme_low:
                return CircuitBreakerViolation(
                    breaker_id=BreakerId.CALIBRATION_DRIFT,
                    message=f"Calibration stuck LOW at {avg:.2f} - rejecting everything",
                    severity="critical",
                    details={'avg_calibration': avg, 'threshold': self.calibration_extreme_low}
                )

            if avg > self.calibration_extreme_high:
                return CircuitBreakerViolation(
                    breaker_id=BreakerId.CALIBRATION_DRIFT,
                    message=f"Calibration stuck HIGH at {avg:.2f} - accepting everything",
                    severity="critical",
                    details={'avg_calibration': avg, 'threshold': self.calibration_extreme_high}
                )

            return None

        except ImportError:
            return None

    def _check_episode_concentration(self) -> Optional[CircuitBreakerViolation]:
        """
        Check if recent episodes are concentrated in losses.

        FAILURE MODE: System is learning from a losing streak and doubling down on bad patterns.
        """
        try:
            from cognitive.episodic_memory import get_episodic_memory, EpisodeOutcome

            memory = get_episodic_memory()

            # Get recent completed episodes
            episodes = list(memory._episodes.values()) if hasattr(memory, '_episodes') else []
            completed = [e for e in episodes
                        if hasattr(e, 'outcome') and e.outcome is not None]

            if len(completed) < self.min_episodes_for_check:
                return None  # Not enough data

            # Check recent episodes (last 30)
            recent = completed[-30:]
            losses = sum(1 for e in recent if e.outcome == EpisodeOutcome.LOSS)
            loss_pct = losses / len(recent)

            if loss_pct > self.max_loss_concentration:
                return CircuitBreakerViolation(
                    breaker_id=BreakerId.EPISODE_CONCENTRATION,
                    message=f"{loss_pct*100:.0f}% of recent episodes are losses - stop and review",
                    severity="critical",
                    details={
                        'recent_episodes': len(recent),
                        'losses': losses,
                        'loss_pct': loss_pct,
                        'threshold': self.max_loss_concentration
                    }
                )

            return None

        except ImportError:
            return None

    def _check_confidence_variance(self) -> Optional[CircuitBreakerViolation]:
        """
        Check if confidence predictions have variance.

        FAILURE MODE: Model collapsed to always returning same value (e.g., 0.5).
        """
        if len(self._recent_confidences) < self.confidence_window_size:
            return None  # Not enough data

        import numpy as np
        variance = np.var(self._recent_confidences)

        if variance < self.min_confidence_variance:
            mean_conf = np.mean(self._recent_confidences)
            return CircuitBreakerViolation(
                breaker_id=BreakerId.CONFIDENCE_VARIANCE,
                message=f"Confidence variance too low ({variance:.4f}) - model may have collapsed to {mean_conf:.2f}",
                severity="critical" if variance < 0.001 else "warning",
                details={
                    'variance': float(variance),
                    'mean_confidence': float(mean_conf),
                    'n_samples': len(self._recent_confidences),
                    'threshold': self.min_confidence_variance
                }
            )

        return None

    def _check_regime_stuck(self) -> Optional[CircuitBreakerViolation]:
        """
        Check if regime detector is stuck in one state.

        FAILURE MODE: HMM regime detector is not updating, always returns same regime.
        """
        try:
            # Check regime history file
            regime_history_path = self.state_dir / "regime_history.json"
            if not regime_history_path.exists():
                return None

            with open(regime_history_path, 'r') as f:
                history = json.load(f)

            if not history or len(history) < 10:
                return None

            # Check recent regimes
            recent = history[-60:]  # Last 60 entries
            regimes = [h.get('regime', 'unknown') for h in recent]

            # Check if all same
            unique_regimes = set(regimes)
            if len(unique_regimes) <= 1 and len(recent) >= self.max_days_in_same_regime:
                return CircuitBreakerViolation(
                    breaker_id=BreakerId.REGIME_STUCK,
                    message=f"Regime stuck at '{regimes[0]}' for {len(recent)}+ days",
                    severity="warning",
                    details={
                        'regime': regimes[0] if regimes else 'unknown',
                        'days_stuck': len(recent),
                        'threshold': self.max_days_in_same_regime
                    }
                )

            return None

        except Exception:
            return None

    def _check_loss_streak(self) -> Optional[CircuitBreakerViolation]:
        """
        Check for extended loss streak.

        FAILURE MODE: System keeps trading through consecutive losses.
        """
        try:
            from cognitive.episodic_memory import get_episodic_memory, EpisodeOutcome

            memory = get_episodic_memory()

            # Get recent completed episodes
            episodes = list(memory._episodes.values()) if hasattr(memory, '_episodes') else []
            completed = [e for e in episodes
                        if hasattr(e, 'outcome') and e.outcome is not None]

            if len(completed) < self.max_consecutive_losses:
                return None

            # Count consecutive losses from end
            consecutive_losses = 0
            for ep in reversed(completed):
                if ep.outcome == EpisodeOutcome.LOSS:
                    consecutive_losses += 1
                else:
                    break

            if consecutive_losses >= self.max_consecutive_losses:
                return CircuitBreakerViolation(
                    breaker_id=BreakerId.LOSS_STREAK,
                    message=f"{consecutive_losses} consecutive losses - halt trading",
                    severity="critical",
                    details={
                        'consecutive_losses': consecutive_losses,
                        'threshold': self.max_consecutive_losses
                    }
                )

            return None

        except ImportError:
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get current safety monitor status."""
        return {
            'recent_confidences': len(self._recent_confidences),
            'total_violations': len(self._violations),
            'critical_violations': sum(1 for v in self._violations if v.severity == "critical"),
            'thresholds': {
                'calibration_range': [self.calibration_extreme_low, self.calibration_extreme_high],
                'max_loss_concentration': self.max_loss_concentration,
                'min_confidence_variance': self.min_confidence_variance,
                'max_consecutive_losses': self.max_consecutive_losses,
            }
        }

    def clear_violation_history(self) -> None:
        """Clear violation history (e.g., after manual review)."""
        self._violations.clear()
        logger.info("Circuit breaker violation history cleared")


def get_safety_monitor() -> CognitiveSafetyMonitor:
    """Get or create the global safety monitor singleton."""
    global _safety_monitor
    if _safety_monitor is None:
        _safety_monitor = CognitiveSafetyMonitor()
    return _safety_monitor


def check_cognitive_safety() -> Tuple[bool, List[CircuitBreakerViolation]]:
    """
    Quick check of cognitive system safety.

    Returns:
        (is_safe, violations): is_safe=False means trading should halt
    """
    monitor = get_safety_monitor()
    return monitor.check_all()


if __name__ == "__main__":
    # Quick test
    import sys

    print("=" * 60)
    print("COGNITIVE SAFETY CHECK")
    print("=" * 60)

    is_safe, violations = check_cognitive_safety()

    if is_safe:
        print("\nStatus: SAFE - No critical violations")
    else:
        print("\nStatus: UNSAFE - Trading should halt")

    if violations:
        print("\nViolations:")
        for v in violations:
            print(f"  [{v.severity.upper()}] {v.breaker_id.value}: {v.message}")
    else:
        print("\nNo violations detected.")

    sys.exit(0 if is_safe else 1)
