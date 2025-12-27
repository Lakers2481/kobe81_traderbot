"""
Metacognitive Governor - Executive Function
=============================================

The "brain of the brain" that orchestrates cognitive resources.

Based on SOFAI architecture (Nature paper):
- Decides when to trust fast (System 1) answers
- Decides when to spend compute on deeper reasoning (System 2)
- Decides when to stand down entirely
- Monitors decision quality and resource usage

Features:
- Fast/Slow decision routing
- Confidence thresholds for escalation
- Resource budget management
- Self-monitoring of decision quality
- Stand-down policies

Usage:
    from cognitive.metacognitive_governor import MetacognitiveGovernor

    governor = MetacognitiveGovernor()

    # Route a decision
    routing = governor.route_decision(signal, context)

    if routing.use_fast_path:
        decision = fast_engine.decide(signal)
    else:
        decision = slow_deliberator.deliberate(signal, context)

    # Record outcome for learning
    governor.record_outcome(routing.decision_id, outcome)
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing mode for decisions."""
    FAST = "fast"           # System 1: Quick, intuitive
    SLOW = "slow"           # System 2: Deliberate, analytical
    HYBRID = "hybrid"       # Both: Fast first, then validate with slow
    STAND_DOWN = "stand_down"  # Don't act


class EscalationReason(Enum):
    """Reasons for escalating to slow processing."""
    LOW_CONFIDENCE = "low_confidence"
    HIGH_UNCERTAINTY = "high_uncertainty"
    NOVEL_SITUATION = "novel_situation"
    HIGH_STAKES = "high_stakes"
    CONFLICTING_SIGNALS = "conflicting_signals"
    SELF_MODEL_CONCERN = "self_model_concern"
    RESOURCE_AVAILABLE = "resource_available"


class StandDownReason(Enum):
    """Reasons for standing down."""
    POOR_CALIBRATION = "poor_calibration"
    KNOWN_LIMITATION = "known_limitation"
    BUDGET_EXHAUSTED = "budget_exhausted"
    HIGH_UNCERTAINTY = "high_uncertainty"
    CONFLICTING_ADVICE = "conflicting_advice"
    SELF_DOUBT = "self_doubt"


@dataclass
class RoutingDecision:
    """Result of metacognitive routing."""
    decision_id: str
    mode: ProcessingMode
    use_fast_path: bool
    use_slow_path: bool
    should_stand_down: bool
    confidence_in_routing: float
    escalation_reasons: List[EscalationReason]
    stand_down_reason: Optional[StandDownReason]
    max_compute_ms: int  # Budget for this decision
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'decision_id': self.decision_id,
            'mode': self.mode.value,
            'use_fast_path': self.use_fast_path,
            'use_slow_path': self.use_slow_path,
            'should_stand_down': self.should_stand_down,
            'confidence_in_routing': round(self.confidence_in_routing, 3),
            'escalation_reasons': [r.value for r in self.escalation_reasons],
            'stand_down_reason': self.stand_down_reason.value if self.stand_down_reason else None,
            'max_compute_ms': self.max_compute_ms,
        }


@dataclass
class DecisionRecord:
    """Record of a decision for learning."""
    decision_id: str
    routing: RoutingDecision
    started_at: datetime
    completed_at: Optional[datetime] = None
    outcome: Optional[str] = None  # 'success', 'failure', 'unknown'
    actual_compute_ms: int = 0
    was_correct: Optional[bool] = None

    @property
    def was_efficient(self) -> bool:
        """Check if compute was used efficiently."""
        return self.actual_compute_ms <= self.routing.max_compute_ms


class MetacognitiveGovernor:
    """
    Executive function that orchestrates cognitive resources.

    Implements the SOFAI metacognitive controller pattern:
    - Routes decisions between fast and slow processing
    - Monitors decision quality
    - Learns when to escalate vs. trust fast answers
    """

    def __init__(
        self,
        # Confidence thresholds
        fast_confidence_threshold: float = 0.75,
        slow_confidence_threshold: float = 0.50,
        stand_down_threshold: float = 0.30,

        # Resource budgets
        default_fast_budget_ms: int = 100,
        default_slow_budget_ms: int = 5000,

        # Calibration
        min_calibration_samples: int = 20,
        acceptable_calibration_error: float = 0.15,

        # Stakes
        high_stakes_position_pct: float = 0.03,  # > 3% is high stakes
    ):
        self.fast_confidence_threshold = fast_confidence_threshold
        self.slow_confidence_threshold = slow_confidence_threshold
        self.stand_down_threshold = stand_down_threshold
        self.default_fast_budget_ms = default_fast_budget_ms
        self.default_slow_budget_ms = default_slow_budget_ms
        self.min_calibration_samples = min_calibration_samples
        self.acceptable_calibration_error = acceptable_calibration_error
        self.high_stakes_position_pct = high_stakes_position_pct

        # Decision history
        self._decision_history: List[DecisionRecord] = []
        self._history_limit = 500

        # Statistics
        self._fast_decisions = 0
        self._slow_decisions = 0
        self._stand_downs = 0
        self._correct_routings = 0
        self._total_routings = 0

        # Self-model reference
        self._self_model = None

        logger.info("MetacognitiveGovernor initialized")

    @property
    def self_model(self):
        """Lazy load self-model."""
        if self._self_model is None:
            try:
                from cognitive.self_model import get_self_model
                self._self_model = get_self_model()
            except ImportError:
                pass
        return self._self_model

    def route_decision(
        self,
        signal: Dict[str, Any],
        context: Dict[str, Any],
        fast_confidence: Optional[float] = None,
        ml_uncertainty: Optional[float] = None,
    ) -> RoutingDecision:
        """
        Route a decision to the appropriate processing mode.

        Args:
            signal: The trade signal being evaluated
            context: Current market/portfolio context
            fast_confidence: Confidence from fast engine (if already computed)
            ml_uncertainty: Uncertainty estimate from ML models

        Returns:
            RoutingDecision with routing and resource budget
        """
        decision_id = str(uuid.uuid4())[:8]
        escalation_reasons = []
        stand_down_reason = None

        # === Check 1: Self-model concerns ===
        if self.self_model:
            strategy = signal.get('strategy', 'unknown')
            regime = context.get('regime', 'unknown')

            should_stand, reason = self.self_model.should_stand_down(strategy, regime)
            if should_stand:
                return self._create_stand_down(
                    decision_id,
                    StandDownReason.KNOWN_LIMITATION,
                    reason
                )

            # Check calibration
            if not self.self_model.is_well_calibrated():
                escalation_reasons.append(EscalationReason.SELF_MODEL_CONCERN)

        # === Check 2: Confidence level ===
        confidence = fast_confidence or context.get('confidence', 0.5)

        if confidence < self.stand_down_threshold:
            return self._create_stand_down(
                decision_id,
                StandDownReason.HIGH_UNCERTAINTY,
                f"Confidence {confidence:.2f} below stand-down threshold"
            )

        if confidence < self.slow_confidence_threshold:
            escalation_reasons.append(EscalationReason.LOW_CONFIDENCE)

        # === Check 3: Uncertainty ===
        if ml_uncertainty is not None and ml_uncertainty > 0.5:
            escalation_reasons.append(EscalationReason.HIGH_UNCERTAINTY)

        # === Check 4: Stakes ===
        position_pct = signal.get('position_pct', 0)
        if position_pct > self.high_stakes_position_pct:
            escalation_reasons.append(EscalationReason.HIGH_STAKES)

        # === Check 5: Novel situation ===
        if self._is_novel_situation(signal, context):
            escalation_reasons.append(EscalationReason.NOVEL_SITUATION)

        # === Check 6: Conflicting signals ===
        if self._has_conflicting_signals(context):
            escalation_reasons.append(EscalationReason.CONFLICTING_SIGNALS)

        # === Determine mode ===
        if len(escalation_reasons) >= 2:
            # Multiple concerns -> slow deliberation
            mode = ProcessingMode.SLOW
            max_compute = self.default_slow_budget_ms
            self._slow_decisions += 1
        elif len(escalation_reasons) == 1:
            # One concern -> hybrid (fast first, then validate)
            mode = ProcessingMode.HYBRID
            max_compute = self.default_fast_budget_ms + self.default_slow_budget_ms // 2
            self._slow_decisions += 1  # Count as slow since we'll validate
        elif confidence >= self.fast_confidence_threshold:
            # High confidence -> fast path
            mode = ProcessingMode.FAST
            max_compute = self.default_fast_budget_ms
            self._fast_decisions += 1
        else:
            # Medium confidence -> hybrid
            mode = ProcessingMode.HYBRID
            max_compute = self.default_fast_budget_ms + self.default_slow_budget_ms // 2
            self._slow_decisions += 1

        # Create routing decision
        routing = RoutingDecision(
            decision_id=decision_id,
            mode=mode,
            use_fast_path=mode in [ProcessingMode.FAST, ProcessingMode.HYBRID],
            use_slow_path=mode in [ProcessingMode.SLOW, ProcessingMode.HYBRID],
            should_stand_down=False,
            confidence_in_routing=self._calculate_routing_confidence(
                confidence, escalation_reasons
            ),
            escalation_reasons=escalation_reasons,
            stand_down_reason=None,
            max_compute_ms=max_compute,
            metadata={
                'input_confidence': confidence,
                'input_uncertainty': ml_uncertainty,
            }
        )

        # Record decision
        self._record_decision(routing)
        self._total_routings += 1

        logger.info(
            f"Decision {decision_id}: {mode.value} "
            f"(confidence={confidence:.2f}, reasons={len(escalation_reasons)})"
        )

        return routing

    def _create_stand_down(
        self,
        decision_id: str,
        reason: StandDownReason,
        details: str,
    ) -> RoutingDecision:
        """Create a stand-down routing decision."""
        self._stand_downs += 1

        routing = RoutingDecision(
            decision_id=decision_id,
            mode=ProcessingMode.STAND_DOWN,
            use_fast_path=False,
            use_slow_path=False,
            should_stand_down=True,
            confidence_in_routing=0.9,  # High confidence in standing down
            escalation_reasons=[],
            stand_down_reason=reason,
            max_compute_ms=0,
            metadata={'stand_down_details': details}
        )

        self._record_decision(routing)
        logger.info(f"Decision {decision_id}: STAND DOWN - {details}")

        return routing

    def _is_novel_situation(self, signal: Dict, context: Dict) -> bool:
        """Check if this is a novel situation the robot hasn't seen before."""
        if not self.self_model:
            return False

        strategy = signal.get('strategy', 'unknown')
        regime = context.get('regime', 'unknown')
        perf = self.self_model.get_performance(strategy, regime)

        # Novel if we have very few samples
        if perf is None or perf.total_trades < 5:
            return True

        return False

    def _has_conflicting_signals(self, context: Dict) -> bool:
        """Check if there are conflicting signals in context."""
        # Check for model disagreement
        model_predictions = context.get('model_predictions', {})
        if model_predictions:
            values = list(model_predictions.values())
            if len(values) >= 2:
                # High variance = disagreement
                mean_pred = sum(values) / len(values)
                variance = sum((v - mean_pred) ** 2 for v in values) / len(values)
                if variance > 0.1:  # > 10% disagreement
                    return True

        # Check for contradictory regime signals
        regime_confidence = context.get('regime_confidence', 1.0)
        if regime_confidence < 0.5:
            return True

        return False

    def _calculate_routing_confidence(
        self,
        input_confidence: float,
        escalation_reasons: List[EscalationReason],
    ) -> float:
        """Calculate confidence in the routing decision itself."""
        base_confidence = 0.8

        # Reduce confidence for each escalation reason
        confidence = base_confidence - (len(escalation_reasons) * 0.1)

        # Input confidence affects routing confidence
        if input_confidence < 0.5:
            confidence -= 0.1

        return max(0.3, min(1.0, confidence))

    def _record_decision(self, routing: RoutingDecision) -> None:
        """Record decision for learning."""
        record = DecisionRecord(
            decision_id=routing.decision_id,
            routing=routing,
            started_at=datetime.now(),
        )
        self._decision_history.append(record)

        # Limit history
        if len(self._decision_history) > self._history_limit:
            self._decision_history = self._decision_history[-self._history_limit:]

    def record_outcome(
        self,
        decision_id: str,
        outcome: str,
        was_correct: Optional[bool] = None,
        actual_compute_ms: int = 0,
    ) -> None:
        """
        Record the outcome of a decision for learning.

        Args:
            decision_id: ID from RoutingDecision
            outcome: 'success', 'failure', 'unknown'
            was_correct: Whether the decision was correct
            actual_compute_ms: Actual compute time used
        """
        for record in reversed(self._decision_history):
            if record.decision_id == decision_id:
                record.completed_at = datetime.now()
                record.outcome = outcome
                record.was_correct = was_correct
                record.actual_compute_ms = actual_compute_ms

                if was_correct is not None:
                    self._total_routings += 1
                    if was_correct:
                        self._correct_routings += 1

                logger.debug(f"Recorded outcome for {decision_id}: {outcome}")
                return

        logger.warning(f"Decision {decision_id} not found in history")

    def should_escalate(
        self,
        current_confidence: float,
        elapsed_ms: int,
        budget_ms: int,
    ) -> bool:
        """
        Check if fast processing should escalate to slow.

        Use during hybrid processing to decide if slow deliberation is needed.
        """
        # If we have budget and confidence is not high, escalate
        if elapsed_ms < budget_ms and current_confidence < self.fast_confidence_threshold:
            return True

        # If very low confidence, always escalate
        if current_confidence < self.slow_confidence_threshold:
            return True

        return False

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about routing decisions."""
        total = self._fast_decisions + self._slow_decisions + self._stand_downs

        return {
            'total_decisions': total,
            'fast_decisions': self._fast_decisions,
            'slow_decisions': self._slow_decisions,
            'stand_downs': self._stand_downs,
            'fast_pct': self._fast_decisions / total if total > 0 else 0,
            'stand_down_pct': self._stand_downs / total if total > 0 else 0,
            'routing_accuracy': (
                self._correct_routings / self._total_routings
                if self._total_routings > 0 else 0
            ),
            'history_size': len(self._decision_history),
        }

    def get_efficiency_report(self) -> Dict[str, Any]:
        """Get report on compute efficiency."""
        if not self._decision_history:
            return {'message': 'No history yet'}

        completed = [d for d in self._decision_history if d.completed_at]
        if not completed:
            return {'message': 'No completed decisions'}

        efficient = sum(1 for d in completed if d.was_efficient)
        over_budget = sum(1 for d in completed if not d.was_efficient)

        return {
            'completed_decisions': len(completed),
            'efficient_decisions': efficient,
            'over_budget_decisions': over_budget,
            'efficiency_rate': efficient / len(completed),
        }

    def introspect(self) -> str:
        """
        Generate introspective report on metacognitive performance.

        This is the "thinking about thinking" capability.
        """
        stats = self.get_routing_stats()
        efficiency = self.get_efficiency_report()

        lines = [
            "=== Metacognitive Introspection Report ===",
            "",
            f"Total decisions routed: {stats['total_decisions']}",
            f"  - Fast path: {stats['fast_decisions']} ({stats['fast_pct']:.1%})",
            f"  - Slow path: {stats['slow_decisions']}",
            f"  - Stand-downs: {stats['stand_downs']} ({stats['stand_down_pct']:.1%})",
            "",
            f"Routing accuracy: {stats['routing_accuracy']:.1%}",
        ]

        if 'efficiency_rate' in efficiency:
            lines.append(f"Compute efficiency: {efficiency['efficiency_rate']:.1%}")

        # Self-critique
        if stats['stand_down_pct'] > 0.3:
            lines.append("\nConcern: High stand-down rate. May be too cautious.")
        elif stats['stand_down_pct'] < 0.05:
            lines.append("\nConcern: Low stand-down rate. May be overconfident.")

        if stats['fast_pct'] > 0.8:
            lines.append("\nConcern: High fast-path rate. May be missing nuances.")
        elif stats['fast_pct'] < 0.2:
            lines.append("\nConcern: Low fast-path rate. May be overthinking.")

        return "\n".join(lines)
