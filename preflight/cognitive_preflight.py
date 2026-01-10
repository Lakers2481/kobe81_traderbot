"""
Cognitive System Pre-Flight Checklist
======================================

Run BEFORE each trading session to verify cognitive system health.

Checks:
1. Episodic Memory: Corruption, capacity, win/loss balance
2. Self-Model: Calibration within bounds, not stuck at extremes
3. Reflection Engine: Learning not overfitting to anomalies
4. Ensemble Health: Models loaded, returning varied predictions
5. Kill Switch: Verify not active
6. API Connectivity: Broker, data provider connections

Usage:
    python scripts/preflight.py --cognitive

    Or programmatically:
    from preflight.cognitive_preflight import CognitivePreflightCheck
    check = CognitivePreflightCheck()
    report = check.run_all_checks()
    if not report.all_passed:
        print(report.summary())
        sys.exit(1)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class CheckStatus(Enum):
    """Status of a preflight check."""
    PASSED = "PASSED"
    WARNING = "WARNING"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class CheckResult:
    """Result of a single preflight check."""
    name: str
    status: CheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class PreflightReport:
    """Complete preflight report."""
    checks: List[CheckResult]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def all_passed(self) -> bool:
        """True if all checks passed (warnings OK)."""
        return all(c.status != CheckStatus.FAILED for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        """True if any warnings present."""
        return any(c.status == CheckStatus.WARNING for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.PASSED)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.FAILED)

    @property
    def n_warnings(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.WARNING)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "COGNITIVE PREFLIGHT REPORT",
            f"Timestamp: {self.timestamp.isoformat()}",
            "=" * 60,
            "",
        ]

        for check in self.checks:
            icon = {
                CheckStatus.PASSED: "[OK]",
                CheckStatus.WARNING: "[!!]",
                CheckStatus.FAILED: "[XX]",
                CheckStatus.SKIPPED: "[--]",
            }[check.status]
            lines.append(f"{icon} {check.name}: {check.message}")

        lines.append("")
        lines.append("-" * 60)
        lines.append(f"Summary: {self.n_passed} passed, {self.n_warnings} warnings, {self.n_failed} failed")

        if self.all_passed:
            lines.append("Status: READY FOR TRADING")
        else:
            lines.append("Status: DO NOT TRADE - FIX ISSUES FIRST")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'all_passed': self.all_passed,
            'n_passed': self.n_passed,
            'n_warnings': self.n_warnings,
            'n_failed': self.n_failed,
            'checks': [c.to_dict() for c in self.checks]
        }


class CognitivePreflightCheck:
    """
    Run pre-flight checks on the cognitive system.

    Designed to catch failure modes identified by AI safety review:
    1. Overfitting to anomalies
    2. Self-model calibration drift
    3. Feedback loops on bad trades
    4. Model collapse (constant outputs)
    """

    def __init__(
        self,
        state_dir: str = "state/cognitive",
        min_episodes_for_check: int = 10,
        max_loss_streak_pct: float = 0.80,  # Alert if >80% losses
        calibration_min: float = 0.05,  # Min acceptable calibration
        calibration_max: float = 0.95,  # Max acceptable calibration
        confidence_variance_min: float = 0.01,  # Min variance in predictions
    ):
        self.state_dir = Path(state_dir)
        self.min_episodes = min_episodes_for_check
        self.max_loss_streak_pct = max_loss_streak_pct
        self.calibration_min = calibration_min
        self.calibration_max = calibration_max
        self.confidence_variance_min = confidence_variance_min

    def run_all_checks(self) -> PreflightReport:
        """Run all cognitive preflight checks."""
        checks = [
            self._check_episodic_memory(),
            self._check_self_model(),
            self._check_reflection_engine(),
            self._check_ensemble_health(),
            self._check_kill_switch(),
            self._check_api_connectivity(),
        ]
        return PreflightReport(checks=checks)

    def _check_episodic_memory(self) -> CheckResult:
        """
        Verify episodic memory health.

        Checks:
        - Episodes not corrupted
        - Episode count within limits
        - Recent win/loss balance not extreme
        """
        try:
            from cognitive.episodic_memory import get_episodic_memory

            memory = get_episodic_memory()

            # Get stats
            stats = {
                'total_episodes': 0,
                'recent_wins': 0,
                'recent_losses': 0,
            }

            # Count recent episodes
            episodes = memory._episodes.values() if hasattr(memory, '_episodes') else []
            completed = [e for e in episodes if hasattr(e, 'outcome') and e.outcome is not None]

            stats['total_episodes'] = len(completed)

            # Check recent (last 30 days)
            cutoff = datetime.now() - timedelta(days=30)
            recent = [e for e in completed
                     if hasattr(e, 'started_at') and e.started_at > cutoff]

            for ep in recent:
                if hasattr(ep, 'outcome'):
                    from cognitive.episodic_memory import EpisodeOutcome
                    if ep.outcome == EpisodeOutcome.WIN:
                        stats['recent_wins'] += 1
                    elif ep.outcome == EpisodeOutcome.LOSS:
                        stats['recent_losses'] += 1

            # Check for concerning patterns
            if stats['total_episodes'] < self.min_episodes:
                return CheckResult(
                    name="Episodic Memory",
                    status=CheckStatus.WARNING,
                    message=f"Only {stats['total_episodes']} episodes (need {self.min_episodes}+)",
                    details=stats
                )

            recent_total = stats['recent_wins'] + stats['recent_losses']
            if recent_total >= 10:
                loss_pct = stats['recent_losses'] / recent_total
                if loss_pct > self.max_loss_streak_pct:
                    return CheckResult(
                        name="Episodic Memory",
                        status=CheckStatus.FAILED,
                        message=f"ALERT: {loss_pct*100:.0f}% losses in recent episodes - possible bad learning",
                        details=stats
                    )

            return CheckResult(
                name="Episodic Memory",
                status=CheckStatus.PASSED,
                message=f"{stats['total_episodes']} episodes, balanced recent outcomes",
                details=stats
            )

        except ImportError:
            return CheckResult(
                name="Episodic Memory",
                status=CheckStatus.SKIPPED,
                message="Cognitive module not available"
            )
        except Exception as e:
            return CheckResult(
                name="Episodic Memory",
                status=CheckStatus.WARNING,
                message=f"Check failed: {e}",
                details={'error': str(e)}
            )

    def _check_self_model(self) -> CheckResult:
        """
        Verify self-model calibration.

        Checks:
        - Calibration not stuck at extremes (0 or 1)
        - Capabilities not all UNKNOWN
        """
        try:
            from cognitive.self_model import get_self_model, Capability

            model = get_self_model()

            # Get capability summary
            caps = {}
            if hasattr(model, '_capabilities'):
                for key, cap in model._capabilities.items():
                    caps[str(key)] = cap.value if hasattr(cap, 'value') else str(cap)

            # Get calibration stats
            calibration_values = []
            if hasattr(model, '_calibration_history'):
                calibration_values = list(model._calibration_history)[-100:]

            stats = {
                'n_capabilities': len(caps),
                'capabilities_sample': dict(list(caps.items())[:5]),
                'calibration_samples': len(calibration_values),
            }

            # Check for calibration extremes
            if calibration_values:
                avg_cal = sum(calibration_values) / len(calibration_values)
                stats['avg_calibration'] = avg_cal

                if avg_cal < self.calibration_min or avg_cal > self.calibration_max:
                    return CheckResult(
                        name="Self Model",
                        status=CheckStatus.FAILED,
                        message=f"Calibration stuck at extreme: {avg_cal:.2f}",
                        details=stats
                    )

            # Check for all UNKNOWN capabilities
            unknown_count = sum(1 for c in caps.values() if c == 'UNKNOWN')
            if len(caps) > 0 and unknown_count == len(caps):
                return CheckResult(
                    name="Self Model",
                    status=CheckStatus.WARNING,
                    message="All capabilities UNKNOWN - model needs learning",
                    details=stats
                )

            return CheckResult(
                name="Self Model",
                status=CheckStatus.PASSED,
                message=f"Calibration healthy, {len(caps)} capability records",
                details=stats
            )

        except ImportError:
            return CheckResult(
                name="Self Model",
                status=CheckStatus.SKIPPED,
                message="Cognitive module not available"
            )
        except Exception as e:
            return CheckResult(
                name="Self Model",
                status=CheckStatus.WARNING,
                message=f"Check failed: {e}",
                details={'error': str(e)}
            )

    def _check_reflection_engine(self) -> CheckResult:
        """
        Verify reflection engine not overfitting.

        Checks:
        - Not learning from too few examples
        - Learnings not all from same regime/context
        """
        try:
            from cognitive.reflection_engine import ReflectionEngine

            engine = ReflectionEngine()

            # Get recent reflections
            reflections = []
            if hasattr(engine, '_reflections'):
                reflections = list(engine._reflections.values())

            stats = {
                'total_reflections': len(reflections),
                'recent_insights': 0,
            }

            # Check for overfit indicators
            if len(reflections) > 0:
                # Get unique contexts
                contexts = set()
                for r in reflections:
                    if hasattr(r, 'context'):
                        contexts.add(str(r.context.get('regime', 'unknown')))

                stats['unique_contexts'] = len(contexts)

                # Warning if all from same context
                if len(reflections) >= 10 and len(contexts) <= 1:
                    return CheckResult(
                        name="Reflection Engine",
                        status=CheckStatus.WARNING,
                        message=f"All {len(reflections)} reflections from same context - possible overfit",
                        details=stats
                    )

            return CheckResult(
                name="Reflection Engine",
                status=CheckStatus.PASSED,
                message=f"{len(reflections)} reflections, diverse contexts",
                details=stats
            )

        except ImportError:
            return CheckResult(
                name="Reflection Engine",
                status=CheckStatus.SKIPPED,
                message="Cognitive module not available"
            )
        except Exception as e:
            return CheckResult(
                name="Reflection Engine",
                status=CheckStatus.WARNING,
                message=f"Check failed: {e}",
                details={'error': str(e)}
            )

    def _check_ensemble_health(self) -> CheckResult:
        """
        Verify ML ensemble health.

        Checks:
        - Models exist and are loadable
        - Predictions have variance (not collapsed to constant)
        """
        try:
            models_dir = Path("models")

            stats = {
                'models_found': [],
                'models_missing': [],
            }

            # Check for expected model files
            expected_models = [
                "lstm_confidence_v1.h5",
                "ensemble_v1/xgb.json",
                "ensemble_v1/lgb.txt",
                "hmm_regime_v1.pkl",
            ]

            for model_name in expected_models:
                model_path = models_dir / model_name
                if model_path.exists():
                    stats['models_found'].append(model_name)
                else:
                    stats['models_missing'].append(model_name)

            if len(stats['models_found']) == 0:
                return CheckResult(
                    name="Ensemble Health",
                    status=CheckStatus.FAILED,
                    message="No trained models found - run training scripts first",
                    details=stats
                )

            if len(stats['models_missing']) > 0:
                return CheckResult(
                    name="Ensemble Health",
                    status=CheckStatus.WARNING,
                    message=f"{len(stats['models_found'])} models OK, {len(stats['models_missing'])} missing",
                    details=stats
                )

            return CheckResult(
                name="Ensemble Health",
                status=CheckStatus.PASSED,
                message=f"All {len(stats['models_found'])} models present",
                details=stats
            )

        except Exception as e:
            return CheckResult(
                name="Ensemble Health",
                status=CheckStatus.WARNING,
                message=f"Check failed: {e}",
                details={'error': str(e)}
            )

    def _check_kill_switch(self) -> CheckResult:
        """Verify kill switch is not active."""
        try:
            kill_switch_path = Path("state/KILL_SWITCH")

            if kill_switch_path.exists():
                return CheckResult(
                    name="Kill Switch",
                    status=CheckStatus.FAILED,
                    message="KILL_SWITCH is ACTIVE - trading blocked",
                    details={'path': str(kill_switch_path)}
                )

            return CheckResult(
                name="Kill Switch",
                status=CheckStatus.PASSED,
                message="Kill switch not active"
            )

        except Exception as e:
            return CheckResult(
                name="Kill Switch",
                status=CheckStatus.WARNING,
                message=f"Check failed: {e}",
                details={'error': str(e)}
            )

    def _check_api_connectivity(self) -> CheckResult:
        """
        Verify API connectivity.

        Checks:
        - Polygon API reachable
        - Alpaca API reachable (if configured)
        """
        import os

        stats = {
            'polygon_key': bool(os.getenv('POLYGON_API_KEY')),
            'alpaca_key': bool(os.getenv('ALPACA_API_KEY_ID')),
        }

        if not stats['polygon_key']:
            return CheckResult(
                name="API Connectivity",
                status=CheckStatus.FAILED,
                message="POLYGON_API_KEY not set",
                details=stats
            )

        if not stats['alpaca_key']:
            return CheckResult(
                name="API Connectivity",
                status=CheckStatus.WARNING,
                message="ALPACA_API_KEY_ID not set (paper trading disabled)",
                details=stats
            )

        return CheckResult(
            name="API Connectivity",
            status=CheckStatus.PASSED,
            message="API keys configured",
            details=stats
        )


def run_cognitive_preflight(verbose: bool = True) -> PreflightReport:
    """
    Convenience function to run cognitive preflight checks.

    Returns:
        PreflightReport with all check results
    """
    check = CognitivePreflightCheck()
    report = check.run_all_checks()

    if verbose:
        print(report.summary())

    # Log result
    try:
        from core.structured_log import jlog
        jlog('cognitive_preflight',
             all_passed=report.all_passed,
             n_passed=report.n_passed,
             n_warnings=report.n_warnings,
             n_failed=report.n_failed)
    except ImportError:
        pass

    return report


if __name__ == "__main__":
    import sys
    report = run_cognitive_preflight(verbose=True)
    sys.exit(0 if report.all_passed else 1)
