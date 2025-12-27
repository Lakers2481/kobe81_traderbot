"""
Cognitive Brain - Main Integration Layer
==========================================

The unified cognitive system that orchestrates all brain components.

Implements:
- Dual-process thinking (Fast/Slow)
- Metacognitive control
- Memory systems (episodic, semantic, working)
- Self-monitoring and reflection
- Continuous learning

This is the "brain" that ties together all cognitive components
into a coherent decision-making system.

Usage:
    from cognitive import get_cognitive_brain

    brain = get_cognitive_brain()

    # Make a deliberated decision
    decision = brain.deliberate(signal, context)

    if decision.should_act:
        result = execute(decision.action)
        brain.learn_from_outcome(decision.episode_id, result)

    # Periodic maintenance
    brain.daily_consolidation()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Type of decision made."""
    ACT = "act"           # Take action
    STAND_DOWN = "stand_down"  # Don't act
    DEFER = "defer"        # Need more information


@dataclass
class CognitiveDecision:
    """Result of cognitive deliberation."""
    decision_type: DecisionType
    should_act: bool
    action: Optional[Dict[str, Any]]
    confidence: float
    reasoning_trace: List[str]
    concerns: List[str]
    knowledge_gaps: List[str]
    invalidators: List[str]
    episode_id: str
    decision_mode: str  # fast, slow, hybrid
    processing_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'decision_type': self.decision_type.value,
            'should_act': self.should_act,
            'action': self.action,
            'confidence': round(self.confidence, 3),
            'reasoning_trace': self.reasoning_trace,
            'concerns': self.concerns,
            'knowledge_gaps': self.knowledge_gaps,
            'invalidators': self.invalidators,
            'episode_id': self.episode_id,
            'decision_mode': self.decision_mode,
            'processing_time_ms': self.processing_time_ms,
        }


class CognitiveBrain:
    """
    Unified cognitive system for intelligent trading decisions.

    Components:
    - GlobalWorkspace: Shared information bus
    - MetacognitiveGovernor: Executive function (fast/slow routing)
    - SelfModel: Self-awareness of capabilities
    - EpisodicMemory: Trade episode storage
    - SemanticMemory: Generalized rules
    - ReflectionEngine: Self-critique
    - KnowledgeBoundary: Uncertainty detection
    - CuriosityEngine: Pattern discovery
    """

    def __init__(
        self,
        min_confidence_to_act: float = 0.5,
        max_processing_time_ms: int = 5000,
    ):
        self.min_confidence_to_act = min_confidence_to_act
        self.max_processing_time_ms = max_processing_time_ms

        # Lazy-loaded components
        self._workspace = None
        self._governor = None
        self._self_model = None
        self._episodic_memory = None
        self._semantic_memory = None
        self._reflection_engine = None
        self._knowledge_boundary = None
        self._curiosity_engine = None

        # State
        self._initialized = False
        self._decision_count = 0
        self._last_maintenance = None

        logger.info("CognitiveBrain initialized")

    def _ensure_initialized(self) -> None:
        """Ensure all components are initialized."""
        if self._initialized:
            return

        # Initialize components (lazy loading)
        _ = self.workspace
        _ = self.governor
        _ = self.self_model
        _ = self.episodic_memory
        _ = self.semantic_memory
        _ = self.knowledge_boundary

        self._initialized = True
        logger.info("CognitiveBrain fully initialized")

    @property
    def workspace(self):
        if self._workspace is None:
            from cognitive.global_workspace import get_workspace
            self._workspace = get_workspace()
        return self._workspace

    @property
    def governor(self):
        if self._governor is None:
            from cognitive.metacognitive_governor import MetacognitiveGovernor
            self._governor = MetacognitiveGovernor()
        return self._governor

    @property
    def self_model(self):
        if self._self_model is None:
            from cognitive.self_model import get_self_model
            self._self_model = get_self_model()
        return self._self_model

    @property
    def episodic_memory(self):
        if self._episodic_memory is None:
            from cognitive.episodic_memory import get_episodic_memory
            self._episodic_memory = get_episodic_memory()
        return self._episodic_memory

    @property
    def semantic_memory(self):
        if self._semantic_memory is None:
            from cognitive.semantic_memory import get_semantic_memory
            self._semantic_memory = get_semantic_memory()
        return self._semantic_memory

    @property
    def reflection_engine(self):
        if self._reflection_engine is None:
            from cognitive.reflection_engine import ReflectionEngine
            self._reflection_engine = ReflectionEngine()
        return self._reflection_engine

    @property
    def knowledge_boundary(self):
        if self._knowledge_boundary is None:
            from cognitive.knowledge_boundary import KnowledgeBoundary
            self._knowledge_boundary = KnowledgeBoundary()
        return self._knowledge_boundary

    @property
    def curiosity_engine(self):
        if self._curiosity_engine is None:
            from cognitive.curiosity_engine import CuriosityEngine
            self._curiosity_engine = CuriosityEngine()
        return self._curiosity_engine

    def deliberate(
        self,
        signal: Dict[str, Any],
        context: Dict[str, Any],
        fast_confidence: Optional[float] = None,
    ) -> CognitiveDecision:
        """
        Main deliberation method - the "thinking" process.

        This implements the dual-process (Fast/Slow) architecture:
        1. Check if we can use fast path (System 1)
        2. If not, engage slow deliberation (System 2)
        3. Apply metacognitive checks
        4. Generate final decision

        Args:
            signal: Trade signal to evaluate
            context: Current market/portfolio context
            fast_confidence: Optional pre-computed confidence from fast engine

        Returns:
            CognitiveDecision with action and reasoning
        """
        import time
        start_time = time.time()

        self._ensure_initialized()

        reasoning_trace = []
        concerns = []

        # === Step 1: Start Episode ===
        episode_id = self.episodic_memory.start_episode(
            market_context=context,
            signal_context=signal,
        )

        reasoning_trace.append("Starting deliberation process")

        # === Step 2: Route Decision (Fast vs Slow) ===
        routing = self.governor.route_decision(
            signal=signal,
            context=context,
            fast_confidence=fast_confidence,
        )

        decision_mode = routing.mode.value
        reasoning_trace.append(f"Metacognitive routing: {decision_mode}")

        # Check for stand-down
        if routing.should_stand_down:
            elapsed_ms = int((time.time() - start_time) * 1000)
            self._decision_count += 1  # Count stand-downs too
            return CognitiveDecision(
                decision_type=DecisionType.STAND_DOWN,
                should_act=False,
                action=None,
                confidence=0.0,
                reasoning_trace=reasoning_trace + [
                    f"Stand-down: {routing.stand_down_reason.value if routing.stand_down_reason else 'unknown'}"
                ],
                concerns=[],
                knowledge_gaps=[],
                invalidators=[],
                episode_id=episode_id,
                decision_mode=decision_mode,
                processing_time_ms=elapsed_ms,
            )

        # === Step 3: Assess Knowledge State ===
        knowledge_assessment = self.knowledge_boundary.assess_knowledge_state(signal, context)

        if knowledge_assessment.should_stand_down:
            reasoning_trace.append("Knowledge boundary triggered stand-down")
            elapsed_ms = int((time.time() - start_time) * 1000)
            self._decision_count += 1  # Count stand-downs too
            return CognitiveDecision(
                decision_type=DecisionType.STAND_DOWN,
                should_act=False,
                action=None,
                confidence=0.0,
                reasoning_trace=reasoning_trace,
                concerns=concerns,
                knowledge_gaps=knowledge_assessment.missing_information,
                invalidators=[],
                episode_id=episode_id,
                decision_mode=decision_mode,
                processing_time_ms=elapsed_ms,
            )

        # Record knowledge gaps
        knowledge_gaps = knowledge_assessment.missing_information
        invalidators = [inv.description for inv in knowledge_assessment.invalidators]

        # === Step 4: Consult Memory Systems ===
        # Check episodic memory for similar situations
        similar_episodes = self.episodic_memory.find_similar(
            {'regime': context.get('regime'), 'strategy': signal.get('strategy')},
            limit=5
        )

        if similar_episodes:
            win_rate, sample_size = self.episodic_memory.get_win_rate_for_context(
                {'regime': context.get('regime'), 'strategy': signal.get('strategy')}
            )
            reasoning_trace.append(
                f"Historical: {win_rate:.1%} win rate from {sample_size} similar trades"
            )

        # Check semantic memory for applicable rules
        applicable_rules = self.semantic_memory.get_applicable_rules(context)

        for rule in applicable_rules[:3]:  # Top 3 rules
            reasoning_trace.append(f"Rule: {rule.condition} -> {rule.action}")
            if rule.action == "reduce_confidence":
                concerns.append(f"Caution rule: {rule.condition}")
            elif rule.action == "increase_confidence":
                reasoning_trace.append(f"Positive signal from: {rule.condition}")

        # === Step 5: Calculate Final Confidence ===
        base_confidence = fast_confidence or 0.6

        # Apply adjustments
        confidence = base_confidence

        # Adjust for knowledge assessment
        confidence += knowledge_assessment.confidence_adjustment

        # Adjust for historical performance
        if similar_episodes:
            if win_rate > 0.6:
                confidence += 0.1
            elif win_rate < 0.4:
                confidence -= 0.1

        # Adjust for semantic rules
        for rule in applicable_rules:
            if rule.action == "reduce_confidence":
                confidence -= 0.1
            elif rule.action == "increase_confidence":
                confidence += 0.05

        # Cap confidence
        confidence_ceiling = self.knowledge_boundary.get_confidence_ceiling(signal, context)
        confidence = min(confidence, confidence_ceiling)
        confidence = max(0, min(1, confidence))

        reasoning_trace.append(f"Final confidence: {confidence:.2f}")

        # === Step 6: Make Decision ===
        if confidence < self.min_confidence_to_act:
            decision_type = DecisionType.STAND_DOWN
            should_act = False
            action = None
            reasoning_trace.append(
                f"Confidence {confidence:.2f} below threshold {self.min_confidence_to_act}"
            )
        else:
            decision_type = DecisionType.ACT
            should_act = True
            action = {
                'type': 'trade',
                'signal': signal,
                'confidence': confidence,
                'size_multiplier': self._calculate_size_multiplier(confidence),
            }
            reasoning_trace.append("Proceeding with trade")

        # === Step 7: Record Reasoning ===
        for trace in reasoning_trace:
            self.episodic_memory.add_reasoning(episode_id, trace)

        for concern in concerns:
            self.episodic_memory.add_concern(episode_id, concern)

        # Record action
        self.episodic_memory.add_action(episode_id, action or {'type': 'stand_down'}, decision_mode)

        # === Step 8: Publish to Workspace ===
        elapsed_ms = int((time.time() - start_time) * 1000)

        decision = CognitiveDecision(
            decision_type=decision_type,
            should_act=should_act,
            action=action,
            confidence=confidence,
            reasoning_trace=reasoning_trace,
            concerns=concerns,
            knowledge_gaps=knowledge_gaps,
            invalidators=invalidators,
            episode_id=episode_id,
            decision_mode=decision_mode,
            processing_time_ms=elapsed_ms,
        )

        self.workspace.publish(
            topic='decision',
            data=decision.to_dict(),
            source='cognitive_brain',
        )

        self._decision_count += 1

        logger.info(
            f"Decision {episode_id}: {decision_type.value} "
            f"(confidence={confidence:.2f}, mode={decision_mode}, time={elapsed_ms}ms)"
        )

        return decision

    def _calculate_size_multiplier(self, confidence: float) -> float:
        """Calculate position size multiplier based on confidence."""
        # Linear scaling: 50% confidence = 50% size, 100% = 100%
        return max(0.25, min(1.0, confidence))

    def learn_from_outcome(
        self,
        episode_id: str,
        outcome: Dict[str, Any],
    ) -> None:
        """
        Learn from trade outcome.

        This completes the learning cycle:
        Decision -> Outcome -> Reflection -> Memory Update

        Args:
            episode_id: Episode ID from deliberation
            outcome: Dict with 'won', 'pnl', 'r_multiple', etc.
        """
        # Complete the episode
        self.episodic_memory.complete_episode(episode_id, outcome)

        # Get the episode for reflection
        episode = self.episodic_memory.get_episode(episode_id)

        if episode:
            # Reflect on the episode
            reflection = self.reflection_engine.reflect_on_episode(episode)

            logger.info(f"Learned from episode {episode_id}: {reflection.summary}")

    def think_deeper(
        self,
        signal: Dict[str, Any],
        context: Dict[str, Any],
        current_confidence: float,
    ) -> Tuple[float, List[str]]:
        """
        Engage System 2 thinking for deeper analysis.

        Called when fast path confidence is insufficient.

        Returns:
            Tuple of (adjusted_confidence, additional_reasoning)
        """
        additional_reasoning = []
        confidence_adjustment = 0.0

        # === Deep Analysis 1: What Would Change My Mind? ===
        invalidators = self.knowledge_boundary.what_would_change_mind(signal, context)
        for inv in invalidators[:3]:
            additional_reasoning.append(f"Watch for: {inv}")

        # === Deep Analysis 2: Historical Pattern Matching ===
        lessons = self.episodic_memory.get_lessons_for_context(
            {'regime': context.get('regime'), 'strategy': signal.get('strategy')}
        )
        for lesson in lessons[:3]:
            additional_reasoning.append(f"Lesson: {lesson}")
            if lesson.startswith("AVOID"):
                confidence_adjustment -= 0.1
            elif lesson.startswith("REPEAT"):
                confidence_adjustment += 0.05

        # === Deep Analysis 3: Self-Model Check ===
        strategy = signal.get('strategy', 'unknown')
        regime = context.get('regime', 'unknown')

        should_stand, reason = self.self_model.should_stand_down(strategy, regime)
        if should_stand:
            additional_reasoning.append(f"Self-model caution: {reason}")
            confidence_adjustment -= 0.2

        # === Deep Analysis 4: Rule Application ===
        rules = self.semantic_memory.get_applicable_rules(context)
        for rule in rules[:3]:
            additional_reasoning.append(f"Applying rule: {rule.condition}")

        new_confidence = current_confidence + confidence_adjustment
        new_confidence = max(0, min(1, new_confidence))

        return new_confidence, additional_reasoning

    def daily_consolidation(self) -> Dict[str, Any]:
        """
        Perform daily maintenance and consolidation.

        Should be called at end of trading day or on schedule.
        """
        logger.info("Starting daily consolidation")

        results = {}

        # Periodic reflection
        reflection = self.reflection_engine.periodic_reflection(lookback_hours=24)
        results['reflection'] = reflection.to_dict()

        # Curiosity: Generate and test hypotheses
        hypotheses = self.curiosity_engine.generate_hypotheses()
        test_results = self.curiosity_engine.test_all_pending()
        results['curiosity'] = {
            'new_hypotheses': len(hypotheses),
            'test_results': test_results,
        }

        # Get edges discovered
        edges = self.curiosity_engine.get_validated_edges()
        results['edges_count'] = len(edges)

        # Update self-model summary
        self_desc = self.self_model.get_self_description()
        results['self_description'] = self_desc

        self._last_maintenance = datetime.now()

        logger.info(f"Daily consolidation complete: {len(edges)} edges, {len(hypotheses)} new hypotheses")

        return results

    def weekly_consolidation(self) -> Dict[str, Any]:
        """
        Perform weekly deep consolidation.

        Includes rule extraction and pruning.
        """
        logger.info("Starting weekly consolidation")

        results = self.reflection_engine.consolidate_learnings()

        # Prune low-confidence rules
        pruned = self.semantic_memory.prune_low_confidence()
        results['rules_pruned'] = pruned

        logger.info(f"Weekly consolidation complete: {results}")

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get cognitive system status."""
        self._ensure_initialized()

        return {
            'initialized': self._initialized,
            'decision_count': self._decision_count,
            'last_maintenance': self._last_maintenance.isoformat() if self._last_maintenance else None,
            'components': {
                'workspace': self.workspace.get_stats(),
                'self_model': self.self_model.get_status(),
                'episodic_memory': self.episodic_memory.get_stats(),
                'semantic_memory': self.semantic_memory.get_stats(),
                'curiosity_engine': self.curiosity_engine.get_stats(),
                'governor': self.governor.get_routing_stats(),
            }
        }

    def introspect(self) -> str:
        """
        Generate comprehensive introspective report.

        This is the robot thinking about itself.
        """
        self._ensure_initialized()

        lines = [
            "=" * 60,
            "        COGNITIVE BRAIN INTROSPECTION REPORT",
            "=" * 60,
            "",
            self.self_model.get_self_description(),
            "",
            "-" * 40,
            "",
            self.governor.introspect(),
            "",
            "-" * 40,
            "",
            self.curiosity_engine.introspect(),
            "",
            "-" * 40,
            "",
            self.knowledge_boundary.introspect(),
            "",
            "=" * 60,
        ]

        return "\n".join(lines)


# Singleton
_cognitive_brain: Optional[CognitiveBrain] = None


def get_cognitive_brain() -> CognitiveBrain:
    """Get or create cognitive brain singleton."""
    global _cognitive_brain
    if _cognitive_brain is None:
        _cognitive_brain = CognitiveBrain()
    return _cognitive_brain
