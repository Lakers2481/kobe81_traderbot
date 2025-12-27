"""
Self Model - Robot's Self-Awareness
=====================================

Persistent model of the robot's own capabilities, limits, and recent errors.

Based on AI self-awareness research (arXiv):
- Self-knowledge (traits, capabilities, limits)
- Knowledge-boundary awareness (what it doesn't know)
- Introspection and self-reflection
- Objective stance on itself as an agent

Features:
- Tracks which strategies/regimes the robot excels at
- Maintains confidence calibration history
- Records recent errors and drift signals
- Updates based on performance feedback
- Persists across sessions

Usage:
    from cognitive.self_model import get_self_model

    self_model = get_self_model()

    # Record capability
    self_model.record_capability('donchian', 'BULL', win_rate=0.68)

    # Record error/limitation
    self_model.record_limitation('high_volatility', 'poor_exits')

    # Query self-knowledge
    my_strength = self_model.best_regime()
    my_weakness = self_model.known_limitations()
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class Capability(Enum):
    """Capability categories."""
    EXCELLENT = "excellent"      # > 65% win rate
    GOOD = "good"                # 55-65% win rate
    ADEQUATE = "adequate"        # 45-55% win rate
    WEAK = "weak"                # 35-45% win rate
    POOR = "poor"                # < 35% win rate
    UNKNOWN = "unknown"          # Insufficient data


@dataclass
class StrategyPerformance:
    """Track performance for a strategy-regime combination."""
    strategy: str
    regime: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_r_multiple: float = 0.0
    max_drawdown: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    notes: List[str] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def capability(self) -> Capability:
        if self.total_trades < 10:
            return Capability.UNKNOWN
        wr = self.win_rate
        if wr >= 0.65:
            return Capability.EXCELLENT
        elif wr >= 0.55:
            return Capability.GOOD
        elif wr >= 0.45:
            return Capability.ADEQUATE
        elif wr >= 0.35:
            return Capability.WEAK
        else:
            return Capability.POOR

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['win_rate'] = self.win_rate
        d['capability'] = self.capability.value
        d['last_updated'] = self.last_updated.isoformat()
        return d


@dataclass
class Limitation:
    """A known limitation or weakness."""
    context: str          # When does this limitation apply
    description: str      # What the limitation is
    severity: str         # 'minor', 'moderate', 'severe'
    occurrences: int = 1
    first_observed: datetime = field(default_factory=datetime.now)
    last_observed: datetime = field(default_factory=datetime.now)
    mitigation: Optional[str] = None  # How to handle this limitation

    def to_dict(self) -> Dict:
        return {
            'context': self.context,
            'description': self.description,
            'severity': self.severity,
            'occurrences': self.occurrences,
            'first_observed': self.first_observed.isoformat(),
            'last_observed': self.last_observed.isoformat(),
            'mitigation': self.mitigation,
        }


@dataclass
class ConfidenceCalibration:
    """Track how well-calibrated predictions are."""
    confidence_bucket: str  # e.g., "70-80%"
    predictions: int = 0
    correct: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def actual_accuracy(self) -> float:
        if self.predictions == 0:
            return 0.0
        return self.correct / self.predictions

    @property
    def calibration_error(self) -> float:
        """Difference between stated confidence and actual accuracy."""
        # Extract middle of bucket (e.g., "70-80%" -> 0.75)
        parts = self.confidence_bucket.replace('%', '').split('-')
        expected = (float(parts[0]) + float(parts[1])) / 200
        return abs(expected - self.actual_accuracy)


class SelfModel:
    """
    Robot's model of its own capabilities, limits, and calibration.

    This is the "Model of Self" component from the cognitive architecture.
    """

    def __init__(
        self,
        state_dir: str = "state/cognitive",
        auto_persist: bool = True,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.auto_persist = auto_persist

        # Performance by strategy-regime
        self._performance: Dict[str, StrategyPerformance] = {}

        # Known limitations
        self._limitations: List[Limitation] = []

        # Confidence calibration buckets
        self._calibration: Dict[str, ConfidenceCalibration] = {}

        # Recent errors (for pattern detection)
        self._recent_errors: List[Dict] = []
        self._error_limit = 100

        # Self-description (natural language summary)
        self._self_description: str = ""

        # Load persisted state
        self._load_state()

        logger.info("SelfModel initialized")

    def _make_key(self, strategy: str, regime: str) -> str:
        """Create key for strategy-regime combination."""
        return f"{strategy}|{regime}"

    def record_trade_outcome(
        self,
        strategy: str,
        regime: str,
        won: bool,
        pnl: float,
        r_multiple: float = 0.0,
        notes: Optional[str] = None,
    ) -> None:
        """
        Record a trade outcome to update self-knowledge.

        Args:
            strategy: Strategy used
            regime: Market regime
            won: Whether trade was profitable
            pnl: P&L of the trade
            r_multiple: R-multiple achieved
            notes: Optional notes
        """
        key = self._make_key(strategy, regime)

        if key not in self._performance:
            self._performance[key] = StrategyPerformance(
                strategy=strategy,
                regime=regime,
            )

        perf = self._performance[key]
        perf.total_trades += 1
        if won:
            perf.winning_trades += 1
        else:
            perf.losing_trades += 1
        perf.total_pnl += pnl
        perf.avg_r_multiple = (
            (perf.avg_r_multiple * (perf.total_trades - 1) + r_multiple) /
            perf.total_trades
        )
        perf.last_updated = datetime.now()
        if notes:
            perf.notes.append(notes)

        # Update self-description
        self._update_self_description()

        if self.auto_persist:
            self._save_state()

        logger.info(
            f"Self-model updated: {strategy} in {regime} - "
            f"Win rate: {perf.win_rate:.1%} ({perf.total_trades} trades)"
        )

    def record_limitation(
        self,
        context: str,
        description: str,
        severity: str = "moderate",
        mitigation: Optional[str] = None,
    ) -> None:
        """
        Record a known limitation.

        Args:
            context: When this limitation applies
            description: What the limitation is
            severity: 'minor', 'moderate', or 'severe'
            mitigation: How to handle this
        """
        # Check if limitation already exists
        for lim in self._limitations:
            if lim.context == context and lim.description == description:
                lim.occurrences += 1
                lim.last_observed = datetime.now()
                if mitigation:
                    lim.mitigation = mitigation
                if self.auto_persist:
                    self._save_state()
                return

        # New limitation
        self._limitations.append(Limitation(
            context=context,
            description=description,
            severity=severity,
            mitigation=mitigation,
        ))

        logger.info(f"New limitation recorded: {context} - {description}")

        if self.auto_persist:
            self._save_state()

    def record_error(
        self,
        error_type: str,
        context: Dict[str, Any],
        description: str,
    ) -> None:
        """Record an error for pattern detection."""
        self._recent_errors.append({
            'error_type': error_type,
            'context': context,
            'description': description,
            'timestamp': datetime.now().isoformat(),
        })

        # Limit size
        if len(self._recent_errors) > self._error_limit:
            self._recent_errors = self._recent_errors[-self._error_limit:]

        # Check for patterns
        self._detect_error_patterns()

        if self.auto_persist:
            self._save_state()

    def record_prediction(
        self,
        confidence: float,
        correct: bool,
    ) -> None:
        """
        Record a prediction for calibration tracking.

        Args:
            confidence: Stated confidence (0-1)
            correct: Whether prediction was correct
        """
        # Bucket confidence
        bucket_ranges = [
            (0.0, 0.5), (0.5, 0.6), (0.6, 0.7),
            (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)
        ]
        for low, high in bucket_ranges:
            if low <= confidence < high:
                bucket = f"{int(low*100)}-{int(high*100)}%"
                break
        else:
            bucket = "90-100%"

        if bucket not in self._calibration:
            self._calibration[bucket] = ConfidenceCalibration(
                confidence_bucket=bucket
            )

        cal = self._calibration[bucket]
        cal.predictions += 1
        if correct:
            cal.correct += 1
        cal.last_updated = datetime.now()

        if self.auto_persist:
            self._save_state()

    def get_capability(self, strategy: str, regime: str) -> Capability:
        """Get capability rating for strategy-regime combination."""
        key = self._make_key(strategy, regime)
        perf = self._performance.get(key)
        if perf:
            return perf.capability
        return Capability.UNKNOWN

    def get_performance(self, strategy: str, regime: str) -> Optional[StrategyPerformance]:
        """Get detailed performance for strategy-regime."""
        key = self._make_key(strategy, regime)
        return self._performance.get(key)

    def best_strategy_for_regime(self, regime: str) -> Optional[Tuple[str, float]]:
        """Find best performing strategy for a regime."""
        best = None
        best_wr = 0.0

        for key, perf in self._performance.items():
            if perf.regime == regime and perf.total_trades >= 10:
                if perf.win_rate > best_wr:
                    best = perf.strategy
                    best_wr = perf.win_rate

        return (best, best_wr) if best else None

    def best_regime_for_strategy(self, strategy: str) -> Optional[Tuple[str, float]]:
        """Find best regime for a strategy."""
        best = None
        best_wr = 0.0

        for key, perf in self._performance.items():
            if perf.strategy == strategy and perf.total_trades >= 10:
                if perf.win_rate > best_wr:
                    best = perf.regime
                    best_wr = perf.win_rate

        return (best, best_wr) if best else None

    def known_limitations(self, context: Optional[str] = None) -> List[Limitation]:
        """Get known limitations, optionally filtered by context."""
        if context:
            return [l for l in self._limitations if context.lower() in l.context.lower()]
        return self._limitations.copy()

    def get_calibration_error(self) -> float:
        """Get overall calibration error (lower is better)."""
        if not self._calibration:
            return 0.0
        errors = [cal.calibration_error for cal in self._calibration.values()
                  if cal.predictions >= 10]
        if not errors:
            return 0.0
        return statistics.mean(errors)

    def is_well_calibrated(self) -> bool:
        """Check if predictions are well-calibrated (error < 10%)."""
        return self.get_calibration_error() < 0.10

    def get_self_description(self) -> str:
        """Get natural language description of self."""
        return self._self_description or self._generate_self_description()

    def get_strengths(self) -> List[str]:
        """Get list of strengths."""
        strengths = []
        for key, perf in self._performance.items():
            if perf.capability in [Capability.EXCELLENT, Capability.GOOD]:
                strengths.append(
                    f"{perf.strategy} in {perf.regime} regime "
                    f"({perf.win_rate:.1%} win rate)"
                )
        return strengths

    def get_weaknesses(self) -> List[str]:
        """Get list of weaknesses."""
        weaknesses = []

        # From performance
        for key, perf in self._performance.items():
            if perf.capability in [Capability.WEAK, Capability.POOR]:
                weaknesses.append(
                    f"{perf.strategy} in {perf.regime} regime "
                    f"({perf.win_rate:.1%} win rate)"
                )

        # From limitations
        for lim in self._limitations:
            if lim.severity in ['moderate', 'severe']:
                weaknesses.append(f"{lim.context}: {lim.description}")

        return weaknesses

    def should_stand_down(self, strategy: str, regime: str) -> Tuple[bool, str]:
        """
        Check if robot should stand down for this context.

        Returns:
            Tuple of (should_stand_down, reason)
        """
        # Check capability
        cap = self.get_capability(strategy, regime)
        if cap == Capability.POOR:
            return True, f"Poor performance in {strategy}/{regime}"

        # Check for severe limitations
        for lim in self._limitations:
            if lim.severity == 'severe':
                if regime.lower() in lim.context.lower():
                    return True, f"Known limitation: {lim.description}"

        # Check calibration
        if not self.is_well_calibrated():
            cal_error = self.get_calibration_error()
            if cal_error > 0.20:
                return True, f"Poor calibration (error: {cal_error:.1%})"

        return False, ""

    def _detect_error_patterns(self) -> None:
        """Detect patterns in recent errors and record as limitations."""
        if len(self._recent_errors) < 5:
            return

        # Count error types
        error_counts: Dict[str, int] = {}
        for err in self._recent_errors[-20:]:
            et = err['error_type']
            error_counts[et] = error_counts.get(et, 0) + 1

        # If any error type appears 3+ times, record as limitation
        for error_type, count in error_counts.items():
            if count >= 3:
                self.record_limitation(
                    context=f"Repeated {error_type}",
                    description=f"Error occurred {count} times in recent trades",
                    severity="moderate",
                )

    def _update_self_description(self) -> None:
        """Update natural language self-description."""
        self._self_description = self._generate_self_description()

    def _generate_self_description(self) -> str:
        """Generate natural language self-description."""
        lines = ["I am Kobe, a trading robot with the following self-knowledge:"]

        # Strengths
        strengths = self.get_strengths()
        if strengths:
            lines.append("\nStrengths:")
            for s in strengths[:3]:
                lines.append(f"  - {s}")

        # Weaknesses
        weaknesses = self.get_weaknesses()
        if weaknesses:
            lines.append("\nWeaknesses:")
            for w in weaknesses[:3]:
                lines.append(f"  - {w}")

        # Calibration
        cal_error = self.get_calibration_error()
        if cal_error > 0:
            lines.append(f"\nCalibration error: {cal_error:.1%}")
            if self.is_well_calibrated():
                lines.append("  (Well-calibrated)")
            else:
                lines.append("  (Need recalibration)")

        # Trade count
        total_trades = sum(p.total_trades for p in self._performance.values())
        lines.append(f"\nTotal trades analyzed: {total_trades}")

        return "\n".join(lines)

    def get_status(self) -> Dict[str, Any]:
        """Get current self-model status."""
        return {
            'strategies_tracked': len(set(p.strategy for p in self._performance.values())),
            'regimes_tracked': len(set(p.regime for p in self._performance.values())),
            'total_trades': sum(p.total_trades for p in self._performance.values()),
            'known_limitations': len(self._limitations),
            'calibration_buckets': len(self._calibration),
            'calibration_error': self.get_calibration_error(),
            'is_well_calibrated': self.is_well_calibrated(),
            'recent_errors': len(self._recent_errors),
        }

    def _save_state(self) -> None:
        """Persist state to disk."""
        state_file = self.state_dir / "self_model.json"
        state = {
            'performance': {k: v.to_dict() for k, v in self._performance.items()},
            'limitations': [l.to_dict() for l in self._limitations],
            'calibration': {k: asdict(v) for k, v in self._calibration.items()},
            'recent_errors': self._recent_errors,
            'self_description': self._self_description,
            'saved_at': datetime.now().isoformat(),
        }
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "self_model.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Load performance
            for key, data in state.get('performance', {}).items():
                self._performance[key] = StrategyPerformance(
                    strategy=data['strategy'],
                    regime=data['regime'],
                    total_trades=data['total_trades'],
                    winning_trades=data['winning_trades'],
                    losing_trades=data['losing_trades'],
                    total_pnl=data['total_pnl'],
                    avg_r_multiple=data.get('avg_r_multiple', 0),
                    notes=data.get('notes', []),
                )

            # Load limitations
            for data in state.get('limitations', []):
                self._limitations.append(Limitation(
                    context=data['context'],
                    description=data['description'],
                    severity=data['severity'],
                    occurrences=data.get('occurrences', 1),
                    mitigation=data.get('mitigation'),
                ))

            # Load calibration
            for key, data in state.get('calibration', {}).items():
                self._calibration[key] = ConfidenceCalibration(
                    confidence_bucket=data['confidence_bucket'],
                    predictions=data['predictions'],
                    correct=data['correct'],
                )

            # Load errors
            self._recent_errors = state.get('recent_errors', [])
            self._self_description = state.get('self_description', '')

            logger.info("Self-model loaded from disk")

        except Exception as e:
            logger.warning(f"Failed to load self-model state: {e}")


# Singleton
_self_model: Optional[SelfModel] = None


def get_self_model() -> SelfModel:
    """Get or create the self-model singleton."""
    global _self_model
    if _self_model is None:
        _self_model = SelfModel()
    return _self_model
