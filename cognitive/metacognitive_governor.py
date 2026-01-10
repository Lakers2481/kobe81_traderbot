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
from typing import Any, Dict, List, Optional
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
    EXTREME_MARKET_MOOD = "extreme_market_mood"  # Market in extreme fear/greed state.
    MODEL_DISAGREEMENT = "model_disagreement"  # Multiple models strongly disagree.
    TOT_REQUIRED = "tot_required"  # Tree-of-Thoughts multi-path reasoning required.


class StandDownReason(Enum):
    """Reasons why the governor might decide to abort a decision process entirely."""
    POOR_CALIBRATION = "poor_calibration"  # The AI knows its confidence estimates are unreliable.
    KNOWN_LIMITATION = "known_limitation"  # The AI's self-model identifies this as a weak area.
    BUDGET_EXHAUSTED = "budget_exhausted" # Not enough cognitive resources available.
    HIGH_UNCERTAINTY = "high_uncertainty"
    CONFLICTING_ADVICE = "conflicting_advice"
    SELF_DOUBT = "self_doubt"
    EXTREME_MARKET_MOOD = "extreme_market_mood"  # Market mood is dangerously extreme.


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
    use_tree_of_thoughts: bool = False  # Whether to use ToT multi-path reasoning.
    use_self_consistency: bool = False  # Whether to use self-consistency decoding.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serializes the decision to a dictionary."""
        return {
            'decision_id': self.decision_id,
            'mode': self.mode.value,
            'use_fast_path': self.use_fast_path,
            'use_slow_path': self.use_slow_path,
            'use_tree_of_thoughts': self.use_tree_of_thoughts,
            'use_self_consistency': self.use_self_consistency,
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

        # --- Meta-metacognitive state (Task B1) ---
        # Context-specific threshold adjustments: {context_key: {param: value}}
        self._adaptive_thresholds: Dict[str, Dict[str, float]] = {}
        self._adjustment_audit_log: List[Dict[str, Any]] = []
        self._pending_adjustments: List[Dict[str, Any]] = []

        self._self_model = None # Lazy-loaded dependency.
        self._policy_generator = None  # Lazy-loaded policy generator.
        logger.info("MetacognitiveGovernor initialized with config-driven settings.")

    @property
    def self_model(self):
        """Lazy-loads the SelfModel to avoid circular dependencies."""
        if self._self_model is None:
            from cognitive.self_model import get_self_model
            self._self_model = get_self_model()
        return self._self_model

    @property
    def policy_generator(self):
        """Lazy-loads the DynamicPolicyGenerator for policy-based routing."""
        if self._policy_generator is None:
            from cognitive.dynamic_policy_generator import get_policy_generator
            self._policy_generator = get_policy_generator()
        return self._policy_generator

    # ==========================================================================
    # ADAPTIVE THRESHOLDS (Task B1: Meta-Metacognitive Self-Configuration)
    # ==========================================================================

    def get_adaptive_threshold(
        self,
        param_name: str,
        strategy: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> float:
        """
        Gets an adaptive threshold for a parameter, potentially adjusted for context.

        This enables context-specific tuning of cognitive parameters. For example,
        the fast_confidence_threshold might be higher in volatile markets.

        Args:
            param_name: The parameter name (e.g., 'fast_confidence_threshold')
            strategy: Optional strategy context
            regime: Optional regime context

        Returns:
            The threshold value, adjusted if a context-specific override exists.
        """
        # First, check for context-specific override
        if strategy and regime:
            context_key = f"{strategy.lower()}|{regime.lower()}"
            if context_key in self._adaptive_thresholds:
                if param_name in self._adaptive_thresholds[context_key]:
                    return self._adaptive_thresholds[context_key][param_name]

        # Fall back to global adaptive adjustment (if any)
        if 'global' in self._adaptive_thresholds:
            if param_name in self._adaptive_thresholds['global']:
                return self._adaptive_thresholds['global'][param_name]

        # Fall back to default threshold
        return getattr(self, param_name, 0.5)

    def apply_proposed_adjustments(
        self,
        adjustments: List[Dict[str, Any]],
        require_confirmation: bool = True,
    ) -> Dict[str, Any]:
        """
        Applies proposed cognitive parameter adjustments from the SelfModel.

        This is how the AI tunes its own cognitive parameters based on learned
        efficiency data.

        Args:
            adjustments: List of adjustment dicts with param_name, proposed_value, context
            require_confirmation: If True, adds to pending; if False, applies immediately

        Returns:
            Dict with applied count, pending count, and details
        """
        from datetime import datetime

        applied = []
        pending = []

        for adj in adjustments:
            param_name = adj.get('param_name')
            proposed_value = adj.get('proposed_value')
            context = adj.get('context', 'global')
            adjustment_id = adj.get('adjustment_id', 'unknown')

            if param_name is None or proposed_value is None:
                continue

            if require_confirmation:
                # Add to pending for human review
                pending_adj = {
                    'adjustment_id': adjustment_id,
                    'param_name': param_name,
                    'proposed_value': proposed_value,
                    'context': context,
                    'rationale': adj.get('rationale', ''),
                    'requested_at': datetime.now().isoformat(),
                }
                self._pending_adjustments.append(pending_adj)
                pending.append(pending_adj)
            else:
                # Apply immediately
                if context not in self._adaptive_thresholds:
                    self._adaptive_thresholds[context] = {}

                old_value = self._adaptive_thresholds.get(context, {}).get(
                    param_name, getattr(self, param_name, None)
                )
                self._adaptive_thresholds[context][param_name] = proposed_value

                applied_adj = {
                    'adjustment_id': adjustment_id,
                    'param_name': param_name,
                    'old_value': old_value,
                    'new_value': proposed_value,
                    'context': context,
                    'applied_at': datetime.now().isoformat(),
                }
                self._adjustment_audit_log.append(applied_adj)
                applied.append(applied_adj)

                # Mark as applied in SelfModel
                if self.self_model:
                    self.self_model.mark_adjustment_applied(adjustment_id)

                logger.info(
                    f"Applied cognitive adjustment: {param_name} = {proposed_value} "
                    f"(was {old_value}) in context '{context}'"
                )

        return {
            'applied': applied,
            'pending': pending,
            'applied_count': len(applied),
            'pending_count': len(pending),
        }

    def confirm_pending_adjustment(self, adjustment_id: str) -> bool:
        """Confirms and applies a pending adjustment by its ID."""
        for adj in self._pending_adjustments[:]:
            if adj.get('adjustment_id') == adjustment_id:
                self._pending_adjustments.remove(adj)
                self.apply_proposed_adjustments([adj], require_confirmation=False)
                return True
        return False

    def reject_pending_adjustment(self, adjustment_id: str) -> bool:
        """Rejects and removes a pending adjustment by its ID."""
        for adj in self._pending_adjustments[:]:
            if adj.get('adjustment_id') == adjustment_id:
                self._pending_adjustments.remove(adj)
                logger.info(f"Rejected cognitive adjustment: {adjustment_id}")
                return True
        return False

    def get_pending_adjustments(self) -> List[Dict[str, Any]]:
        """Returns all pending adjustments awaiting confirmation."""
        return self._pending_adjustments.copy()

    def get_adjustment_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns recent adjustment audit log entries."""
        return self._adjustment_audit_log[-limit:]

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

        # Extract context for adaptive thresholds (Task B1)
        strategy = signal.get('strategy', 'unknown')
        regime = context.get('regime', 'unknown')

        # === CHECK 1: CONSULT SELF-MODEL ===
        # First, ask the self-model: "Am I any good at this?"
        if self.self_model:
            if self.self_model.should_stand_down(strategy, regime)[0]:
                return self._create_stand_down(decision_id, StandDownReason.KNOWN_LIMITATION,
                                               f"Self-model indicates poor performance in {strategy}/{regime}.")
            if not self.self_model.is_well_calibrated():
                escalation_reasons.append(EscalationReason.SELF_MODEL_CONCERN)

        # === CHECK 2: INITIAL CONFIDENCE ===
        # Use adaptive thresholds based on context (Task B1)
        confidence = fast_confidence or context.get('confidence', 0.5)
        stand_down_thresh = self.get_adaptive_threshold('stand_down_threshold', strategy, regime)
        slow_conf_thresh = self.get_adaptive_threshold('slow_confidence_threshold', strategy, regime)

        if confidence < stand_down_thresh:
            return self._create_stand_down(decision_id, StandDownReason.HIGH_UNCERTAINTY,
                                           f"Initial confidence {confidence:.2f} is below stand-down threshold ({stand_down_thresh:.2f}).")
        if confidence < slow_conf_thresh:
            escalation_reasons.append(EscalationReason.LOW_CONFIDENCE)

        # === CHECK 3: HIGH STAKES ===
        # Is this a particularly large or risky trade? If so, think harder.
        high_stakes_thresh = self.get_adaptive_threshold('high_stakes_position_pct', strategy, regime)
        if signal.get('position_pct', 0) > high_stakes_thresh:
            escalation_reasons.append(EscalationReason.HIGH_STAKES)

        # === CHECK 4: NOVELTY ===
        # Is this a situation the AI hasn't seen many times before?
        if self._is_novel_situation(signal, context):
            escalation_reasons.append(EscalationReason.NOVEL_SITUATION)

        # === CHECK 5: CONFLICTING SIGNALS ===
        # Are there disagreements among internal models or data sources?
        if context.get('conflicting_signals', False):
            escalation_reasons.append(EscalationReason.CONFLICTING_SIGNALS)

        # === CHECK 6: EXTREME MARKET MOOD ===
        # Is the market in an extreme emotional state (fear/greed)?
        is_extreme_mood = context.get('is_extreme_mood', False)
        market_mood_score = context.get('market_mood_score', 0.0)

        if is_extreme_mood:
            # For very extreme mood (|score| >= 0.9), consider standing down
            if abs(market_mood_score) >= 0.9:
                return self._create_stand_down(
                    decision_id,
                    StandDownReason.EXTREME_MARKET_MOOD,
                    f"Market mood is dangerously extreme (score={market_mood_score:.2f}). "
                    "Recommend waiting for calmer conditions."
                )
            # For extreme but not critical, add as escalation reason
            escalation_reasons.append(EscalationReason.EXTREME_MARKET_MOOD)

        # === CHECK 7: ACTIVE POLICY MODIFICATIONS (Task B3) ===
        # Evaluate and apply any active trading policy's cognitive modifications
        active_policy = None
        policy_force_slow_path = False

        try:
            active_policy = self.policy_generator.evaluate_policy_activation(
                market_context=context,
                mood_score=market_mood_score,
                regime=regime,
            )

            if active_policy:
                cog_mods = active_policy.cognitive_modifications
                # Check if policy requires slow path
                if cog_mods.get('require_slow_path', False):
                    policy_force_slow_path = True
                    logger.debug(f"Policy {active_policy.policy_id} requires slow path")

                # Check if policy requires stand-down on any uncertainty
                if cog_mods.get('stand_down_on_uncertainty', False) and escalation_reasons:
                    return self._create_stand_down(
                        decision_id,
                        StandDownReason.HIGH_UNCERTAINTY,
                        f"Policy '{active_policy.policy_id}' requires stand-down on uncertainty. "
                        f"Escalation reasons: {[r.value for r in escalation_reasons]}"
                    )

                # Apply policy-specific thresholds if available
                if 'fast_path_threshold' in cog_mods:
                    # Use policy threshold temporarily for this decision
                    policy_fast_thresh = cog_mods['fast_path_threshold']
                    if policy_fast_thresh > self.fast_confidence_threshold:
                        # Higher threshold means we're more likely to need slow path
                        if confidence < policy_fast_thresh:
                            escalation_reasons.append(EscalationReason.LOW_CONFIDENCE)

        except Exception as e:
            logger.warning(f"Policy evaluation error (continuing without): {e}")

        # === MAKE ROUTING DECISION ===
        # The number of red flags determines the cognitive path.
        # Use adaptive thresholds (Task B1)
        fast_conf_thresh = self.get_adaptive_threshold('fast_confidence_threshold', strategy, regime)

        # Check if active policy forces slow path (Task B3)
        if policy_force_slow_path:
            mode = ProcessingMode.SLOW
            max_compute = self.default_slow_budget_ms
            self._slow_decisions += 1
            logger.debug(f"Policy override: forcing slow path for decision {decision_id}")
        elif len(escalation_reasons) >= 2:
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
        elif confidence >= fast_conf_thresh:
            # No red flags and high confidence -> trust the fast path.
            mode = ProcessingMode.FAST
            max_compute = self.default_fast_budget_ms
            self._fast_decisions += 1
        else:
            # Medium confidence, no other red flags -> default to a cautious hybrid path.
            mode = ProcessingMode.HYBRID
            max_compute = self.default_fast_budget_ms + self.default_slow_budget_ms // 2
            self._slow_decisions += 1

        # === CHECK 8: MODEL DISAGREEMENT (ToT trigger) ===
        model_disagreement = context.get('model_disagreement', 0.0)
        if model_disagreement > 0.4:
            escalation_reasons.append(EscalationReason.MODEL_DISAGREEMENT)

        # === DETERMINE ADVANCED REASONING MODES ===
        # Tree-of-Thoughts for complex multi-path reasoning
        use_tot = self._should_use_tree_of_thoughts(signal, context, escalation_reasons)
        if use_tot:
            escalation_reasons.append(EscalationReason.TOT_REQUIRED)
            # ToT requires more compute
            max_compute = max(max_compute, self.default_slow_budget_ms * 2)

        # Self-consistency for robust reasoning
        use_sc = self._should_use_self_consistency(signal, context, escalation_reasons)

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
            use_tree_of_thoughts=use_tot,
            use_self_consistency=use_sc,
            metadata={
                'input_confidence': confidence,
                'active_policy': active_policy.policy_id if active_policy else None,
                'policy_force_slow_path': policy_force_slow_path,
                'model_disagreement': model_disagreement,
            }
        )

        self._record_decision(routing)
        tot_status = " [ToT]" if use_tot else ""
        sc_status = " [SC]" if use_sc else ""
        logger.info(f"Decision {decision_id} routed to {mode.value} path{tot_status}{sc_status} due to {len(escalation_reasons)} escalation reasons.")
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

    def _should_use_tree_of_thoughts(
        self,
        signal: Dict,
        context: Dict,
        escalation_reasons: List[EscalationReason],
    ) -> bool:
        """
        Determines whether Tree-of-Thoughts multi-path reasoning should be used.

        ToT is computationally expensive but dramatically improves reasoning on
        complex decisions (74% vs 4% accuracy on puzzles per research).

        Triggers:
        - Model disagreement > 40% (significant divergence)
        - 3+ conflicting factors (complex trade-off)
        - High uncertainty score > 60%
        - High stakes + conflicting signals
        - Novel situation with conflicting signals

        Args:
            signal: The trading signal being evaluated
            context: Market and portfolio context
            escalation_reasons: Already-identified escalation reasons

        Returns:
            True if ToT should be used for this decision
        """
        # Trigger 1: Model disagreement (e.g., HMM vs technicals vs sentiment)
        model_disagreement = context.get('model_disagreement', 0.0)
        if model_disagreement > 0.4:
            logger.debug(f"ToT triggered: model_disagreement={model_disagreement:.2f} > 0.4")
            return True

        # Trigger 2: Multiple conflicting factors
        conflicting_factors = context.get('conflicting_factors', 0)
        if conflicting_factors >= 3:
            logger.debug(f"ToT triggered: conflicting_factors={conflicting_factors} >= 3")
            return True

        # Trigger 3: High uncertainty score
        uncertainty_score = context.get('uncertainty_score', 0.0)
        if uncertainty_score > 0.6:
            logger.debug(f"ToT triggered: uncertainty_score={uncertainty_score:.2f} > 0.6")
            return True

        # Trigger 4: High stakes with conflicting signals
        has_high_stakes = EscalationReason.HIGH_STAKES in escalation_reasons
        has_conflicting = EscalationReason.CONFLICTING_SIGNALS in escalation_reasons
        if has_high_stakes and has_conflicting:
            logger.debug("ToT triggered: high stakes + conflicting signals")
            return True

        # Trigger 5: Novel situation with conflicting signals
        has_novel = EscalationReason.NOVEL_SITUATION in escalation_reasons
        if has_novel and has_conflicting:
            logger.debug("ToT triggered: novel situation + conflicting signals")
            return True

        return False

    def _should_use_self_consistency(
        self,
        signal: Dict,
        context: Dict,
        escalation_reasons: List[EscalationReason],
    ) -> bool:
        """
        Determines whether Self-Consistency decoding should be used.

        Self-consistency samples multiple reasoning chains and picks the
        majority answer, reducing variance in LLM outputs.

        Triggers:
        - Low confidence in routing (< 0.6)
        - Model disagreement between 20-40% (moderate divergence)
        - Self-model concern (AI knows it struggles here)

        Args:
            signal: The trading signal being evaluated
            context: Market and portfolio context
            escalation_reasons: Already-identified escalation reasons

        Returns:
            True if self-consistency should be used
        """
        # Trigger 1: Moderate model disagreement
        model_disagreement = context.get('model_disagreement', 0.0)
        if 0.2 <= model_disagreement <= 0.4:
            logger.debug(f"Self-consistency triggered: moderate disagreement={model_disagreement:.2f}")
            return True

        # Trigger 2: Self-model concern
        if EscalationReason.SELF_MODEL_CONCERN in escalation_reasons:
            logger.debug("Self-consistency triggered: self-model concern")
            return True

        # Trigger 3: Low confidence with any escalation
        confidence = context.get('confidence', 0.5)
        if confidence < 0.6 and len(escalation_reasons) >= 1:
            logger.debug(f"Self-consistency triggered: low conf={confidence:.2f} + escalation")
            return True

        return False

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
