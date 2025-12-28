"""
Metacognitive Governor - The AI's Executive Function
======================================================

This module acts as the "brain of the brain," providing the executive function
that orchestrates the AI's cognitive resources. It decides *how* to think about
a given problem, implementing a dual-process model inspired by human cognition
(System 1 vs. System 2, or "fast" vs. "slow" thinking).

Core Responsibilities:
- **Decision Routing:** Determines whether a new signal can be handled quickly
  and intuitively (fast path) or if it requires deep, analytical, and
  resource-intensive deliberation (slow path).
- **Resource Management:** Allocates a "compute budget" (in milliseconds) for
  each decision to ensure timely responses.
- **Stand-Down Policies:** Identifies situations where the AI should not act at
  all due to high uncertainty, known limitations, or other critical factors.
- **Self-Monitoring:** It monitors the quality and efficiency of its own routing
  decisions, enabling it to learn and improve its metacognitive strategy over time.

This component is crucial for creating an AI that is not only intelligent but
also efficient and self-aware, knowing when to "trust its gut" and when to
"stop and think."

Based on concepts from SOFAI (Self-Organizing Fuzzy AI) and Daniel Kahneman's
"Thinking, Fast and Slow".

Usage:
    from cognitive.metacognitive_governor import MetacognitiveGovernor

    governor = MetacognitiveGovernor()

    # For each new signal, the governor decides how to proceed.
    routing_decision = governor.route_decision(signal, context)

    if routing_decision.should_stand_down:
        return  // Abort
    elif routing_decision.use_fast_path and not routing_decision.use_slow_path:
        decision = fast_engine.decide(signal)
    else:
        decision = slow_deliberator.deliberate(signal, context)

    # The outcome is fed back to the governor for its own learning process.
    governor.record_outcome(routing_decision.decision_id, outcome)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Enumerates the cognitive processing mode for a decision."""
    FAST = "fast"           # System 1: Quick, heuristic-based, low resource.
    SLOW = "slow"           # System 2: Deliberate, analytical, high resource.
    HYBRID = "hybrid"       # Use fast path first, then escalate to slow if needed.
    STAND_DOWN = "stand_down"  # Explicit decision not to act.


class EscalationReason(Enum):
    """Reasons why a decision might be escalated from the fast path to the slow path."""
    LOW_CONFIDENCE = "low_confidence"
    HIGH_UNCERTAINTY = "high_uncertainty"
    NOVEL_SITUATION = "novel_situation"
    HIGH_STAKES = "high_stakes"
    CONFLICTING_SIGNALS = "conflicting_signals"
    SELF_MODEL_CONCERN = "self_model_concern" # AI's self-model indicates it's not good at this.
    RESOURCE_AVAILABLE = "resource_available" # Has spare compute, so can afford to think deeper.


class StandDownReason(Enum):
    """Reasons why the governor might decide to abort a decision process entirely."""
    POOR_CALIBRATION = "poor_calibration"  # The AI knows its confidence estimates are unreliable.
    KNOWN_LIMITATION = "known_limitation"  # The AI's self-model identifies this as a weak area.
    BUDGET_EXHAUSTED = "budget_exhausted" # Not enough cognitive resources available.
    HIGH_UNCERTAINTY = "high_uncertainty"
    CONFLICTING_ADVICE = "conflicting_advice"
    SELF_DOUBT = "self_doubt"


@dataclass
class RoutingDecision:
    """
    A structured output from the governor, specifying how to handle a decision.
    """
    decision_id: str
    mode: ProcessingMode
    use_fast_path: bool
    use_slow_path: bool
    should_stand_down: bool
    confidence_in_routing: float  # The governor's confidence in its own routing decision.
    escalation_reasons: List[EscalationReason]
    stand_down_reason: Optional[StandDownReason]
    max_compute_ms: int  # The time budget allocated for this decision.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serializes the decision to a dictionary."""
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
    """A record used by the governor to learn from its past routing decisions."""
    decision_id: str
    routing: RoutingDecision
    started_at: datetime
    completed_at: Optional[datetime] = None
    outcome: Optional[str] = None # 'success' or 'failure'
    actual_compute_ms: int = 0
    was_correct: Optional[bool] = None # Was the routing decision itself correct?

    @property
    def was_efficient(self) -> bool:
        """Did the decision complete within its allocated time budget?"""
        return self.actual_compute_ms <= self.routing.max_compute_ms


class MetacognitiveGovernor:
    """
    The executive controller of the cognitive architecture. It routes decisions
    between fast and slow processing pathways, manages cognitive resources, and
    monitors its own performance.
    """

    def __init__(
        self,
        fast_confidence_threshold: Optional[float] = None,
        slow_confidence_threshold: Optional[float] = None,
        stand_down_threshold: Optional[float] = None,
        default_fast_budget_ms: Optional[int] = None,
        default_slow_budget_ms: Optional[int] = None,
        high_stakes_position_pct: Optional[float] = None,
    ):
        # Load from centralized config if not explicitly provided
        try:
            from config.settings_loader import get_cognitive_governor_config
            config = get_cognitive_governor_config()
        except ImportError:
            config = {}

        # Use explicit params if provided, else config, else hardcoded defaults
        self.fast_confidence_threshold = fast_confidence_threshold if fast_confidence_threshold is not None else config.get("fast_confidence_threshold", 0.75)
        self.slow_confidence_threshold = slow_confidence_threshold if slow_confidence_threshold is not None else config.get("slow_confidence_threshold", 0.50)
        self.stand_down_threshold = stand_down_threshold if stand_down_threshold is not None else config.get("stand_down_threshold", 0.30)
        self.default_fast_budget_ms = default_fast_budget_ms if default_fast_budget_ms is not None else config.get("default_fast_budget_ms", 100)
        self.default_slow_budget_ms = default_slow_budget_ms if default_slow_budget_ms is not None else config.get("default_slow_budget_ms", 5000)
        self.high_stakes_position_pct = high_stakes_position_pct if high_stakes_position_pct is not None else config.get("high_stakes_position_pct", 0.03)

        # Internal state for learning and statistics.
        self._decision_history: List[DecisionRecord] = []
        self._history_limit = 500
        self._fast_decisions = 0
        self._slow_decisions = 0
        self._stand_downs = 0
        self._correct_routings = 0
        self._total_routings = 0

        self._self_model = None # Lazy-loaded dependency.
        logger.info("MetacognitiveGovernor initialized with config-driven settings.")

    @property
    def self_model(self):
        """Lazy-loads the SelfModel to avoid circular dependencies."""
        if self._self_model is None:
            from cognitive.self_model import get_self_model
            self._self_model = get_self_model()
        return self._self_model

    def route_decision(
        self,
        signal: Dict[str, Any],
        context: Dict[str, Any],
        fast_confidence: Optional[float] = None,
    ) -> RoutingDecision:
        """
        The main entry point. Examines a situation and decides on the appropriate
        cognitive pathway and resource budget.

        Args:
            signal: The trading signal to be evaluated.
            context: The current market and portfolio context.
            fast_confidence: An optional pre-computed confidence score from a
                             fast, upstream process.

        Returns:
            A RoutingDecision object detailing the chosen cognitive path.
        """
        decision_id = str(uuid.uuid4())[:8]
        escalation_reasons = []

        # === CHECK 1: CONSULT SELF-MODEL ===
        # First, ask the self-model: "Am I any good at this?"
        if self.self_model:
            strategy = signal.get('strategy', 'unknown')
            regime = context.get('regime', 'unknown')
            if self.self_model.should_stand_down(strategy, regime)[0]:
                return self._create_stand_down(decision_id, StandDownReason.KNOWN_LIMITATION,
                                               f"Self-model indicates poor performance in {strategy}/{regime}.")
            if not self.self_model.is_well_calibrated():
                escalation_reasons.append(EscalationReason.SELF_MODEL_CONCERN)

        # === CHECK 2: INITIAL CONFIDENCE ===
        # Use the fast confidence score as a primary indicator.
        confidence = fast_confidence or context.get('confidence', 0.5)
        if confidence < self.stand_down_threshold:
            return self._create_stand_down(decision_id, StandDownReason.HIGH_UNCERTAINTY,
                                           f"Initial confidence {confidence:.2f} is below stand-down threshold.")
        if confidence < self.slow_confidence_threshold:
            escalation_reasons.append(EscalationReason.LOW_CONFIDENCE)

        # === CHECK 3: HIGH STAKES ===
        # Is this a particularly large or risky trade? If so, think harder.
        if signal.get('position_pct', 0) > self.high_stakes_position_pct:
            escalation_reasons.append(EscalationReason.HIGH_STAKES)

        # === CHECK 4: NOVELTY ===
        # Is this a situation the AI hasn't seen many times before?
        if self._is_novel_situation(signal, context):
            escalation_reasons.append(EscalationReason.NOVEL_SITUATION)

        # === CHECK 5: CONFLICTING SIGNALS ===
        # Are there disagreements among internal models or data sources?
        if context.get('conflicting_signals', False):
            escalation_reasons.append(EscalationReason.CONFLICTING_SIGNALS)


        # === MAKE ROUTING DECISION ===
        # The number of red flags determines the cognitive path.
        if len(escalation_reasons) >= 2:
            # Multiple concerns warrant a full, slow deliberation.
            mode = ProcessingMode.SLOW
            max_compute = self.default_slow_budget_ms
            self._slow_decisions += 1
        elif len(escalation_reasons) == 1:
            # A single concern triggers a hybrid approach: check the fast path,
            # but then validate with the slow path.
            mode = ProcessingMode.HYBRID
            max_compute = self.default_fast_budget_ms + self.default_slow_budget_ms // 2
            self._slow_decisions += 1 # Counted as slow because it involves deliberation.
        elif confidence >= self.fast_confidence_threshold:
            # No red flags and high confidence -> trust the fast path.
            mode = ProcessingMode.FAST
            max_compute = self.default_fast_budget_ms
            self._fast_decisions += 1
        else:
            # Medium confidence, no other red flags -> default to a cautious hybrid path.
            mode = ProcessingMode.HYBRID
            max_compute = self.default_fast_budget_ms + self.default_slow_budget_ms // 2
            self._slow_decisions += 1

        routing = RoutingDecision(
            decision_id=decision_id,
            mode=mode,
            use_fast_path=(mode in [ProcessingMode.FAST, ProcessingMode.HYBRID]),
            use_slow_path=(mode in [ProcessingMode.SLOW, ProcessingMode.HYBRID]),
            should_stand_down=False,
            confidence_in_routing=1.0 - (len(escalation_reasons) * 0.1),
            escalation_reasons=escalation_reasons,
            stand_down_reason=None,
            max_compute_ms=max_compute,
            metadata={'input_confidence': confidence}
        )

        self._record_decision(routing)
        logger.info(f"Decision {decision_id} routed to {mode.value} path due to {len(escalation_reasons)} escalation reasons.")
        return routing

    def _create_stand_down(self, decision_id: str, reason: StandDownReason, details: str) -> RoutingDecision:
        """Helper to create and log a STAND_DOWN decision."""
        self._stand_downs += 1
        routing = RoutingDecision(
            decision_id=decision_id,
            mode=ProcessingMode.STAND_DOWN,
            use_fast_path=False, use_slow_path=False,
            should_stand_down=True,
            confidence_in_routing=0.95, # Usually confident in a decision to be cautious.
            escalation_reasons=[],
            stand_down_reason=reason,
            max_compute_ms=0,
            metadata={'stand_down_details': details}
        )
        self._record_decision(routing)
        logger.info(f"Decision {decision_id}: STAND DOWN advised. Reason: {reason.value} - {details}")
        return routing

    def _is_novel_situation(self, signal: Dict, context: Dict) -> bool:
        """Checks if the current situation is novel based on past experience."""
        if not self.self_model: return False
        strategy = signal.get('strategy', 'unknown')
        regime = context.get('regime', 'unknown')
        perf = self.self_model.get_performance(strategy, regime)
        return perf is None or perf.total_trades < 5

    def _record_decision(self, routing: RoutingDecision) -> None:
        """Records the routing decision for future self-evaluation."""
        record = DecisionRecord(decision_id=routing.decision_id, routing=routing, started_at=datetime.now())
        self._decision_history.append(record)
        if len(self._decision_history) > self._history_limit:
            self._decision_history.pop(0)

    def record_outcome(
        self,
        decision_id: str,
        outcome: str,
        was_correct: Optional[bool] = None,
        actual_compute_ms: int = 0,
    ) -> None:
        """
        Records the final outcome of a decision, closing the learning loop for
        the governor.

        Args:
            decision_id: The ID from the original `RoutingDecision`.
            outcome: 'success' or 'failure'.
            was_correct: Whether the final decision was deemed correct. This helps
                         the governor learn if its routing was appropriate.
            actual_compute_ms: The actual time taken for the deliberation.
        """
        for record in reversed(self._decision_history):
            if record.decision_id == decision_id:
                record.completed_at = datetime.now()
                record.outcome = outcome
                record.was_correct = was_correct
                record.actual_compute_ms = actual_compute_ms

                # Metacognitive learning: if the routing decision was correct,
                # increment the accuracy score.
                if was_correct is not None:
                    self._total_routings += 1
                    if was_correct:
                        self._correct_routings += 1
                return
        logger.warning(f"Could not find decision_id {decision_id} in history to record outcome.")

    def get_routing_stats(self) -> Dict[str, Any]:
        """Returns statistics on how decisions have been routed."""
        total = self._fast_decisions + self._slow_decisions + self._stand_downs
        if total == 0: return {"total_decisions": 0}
        return {
            'total_decisions': total,
            'fast_decisions': self._fast_decisions,
            'slow_decisions': self._slow_decisions,
            'stand_downs': self._stand_downs,
            'fast_path_pct': self._fast_decisions / total,
            'stand_down_pct': self._stand_downs / total,
            'routing_accuracy': self._correct_routings / self._total_routings if self._total_routings > 0 else 0,
        }

    def introspect(self) -> str:
        """
        Generates a human-readable report on the governor's own performance,
        enabling "thinking about thinking."
        """
        stats = self.get_routing_stats()
        lines = [
            "--- Metacognitive Introspection ---",
            f"I have routed {stats['total_decisions']} total decisions.",
            f"  - Fast Path: {stats.get('fast_path_pct', 0):.1%}",
            f"  - Slow Path: {stats.get('slow_decisions', 0)}",
            f"  - Stood Down: {stats.get('stand_down_pct', 0):.1%}",
            f"My estimated routing accuracy is {stats.get('routing_accuracy', 0):.1%}.",
            "\n--- Self-Critique ---",
        ]

        if stats.get('stand_down_pct', 0) > 0.4:
            lines.append("Concern: My stand-down rate is high. I may be acting overly cautious.")
        if stats.get('fast_path_pct', 0) > 0.8:
            lines.append("Concern: My fast-path rate is high. I may be under-thinking and missing nuances.")
        elif stats.get('fast_path_pct', 0) < 0.2 and stats.get('total_decisions',0) > 20:
            lines.append("Concern: My fast-path rate is low. I may be over-thinking simple decisions.")

        if not lines[-1].startswith("Concern"):
            lines.append("Current performance seems balanced.")
            
        return "\n".join(lines)
