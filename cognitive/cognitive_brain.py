"""
Cognitive Brain - Main Integration Layer
==========================================

The unified cognitive system that orchestrates all brain components.

This class acts as the central "nervous system" for the trading bot's advanced
AI capabilities. It integrates various cognitive components into a coherent
decision-making and learning architecture.

Implements:
- Dual-process thinking (Fast/Slow): Simulates intuitive, fast reactions and
  deeper, analytical reasoning.
- Metacognitive control: The brain's ability to monitor and regulate its own
  thinking processes, deciding when to "think harder".
- Multiple Memory Systems: Utilizes episodic (short-term memory of trades),
  semantic (long-term learned rules), and a global workspace for current context.
- Self-monitoring and Reflection: The agent can analyze its own performance
  to learn and adapt.
- Continuous Learning: The brain updates its internal models based on the
  outcomes of its decisions.

This is the "brain" that ties together all cognitive components
into a coherent decision-making system.

Usage:
    from cognitive import get_cognitive_brain

    # The brain is a singleton, ensuring one "mind" for the application.
    brain = get_cognitive_brain()

    # Make a deliberated decision on a new trading signal
    decision = brain.deliberate(signal, context)

    # If the brain decides to act, execute the trade and report the outcome
    if decision.should_act:
        result = execute(decision.action)
        brain.learn_from_outcome(decision.episode_id, result)

    # Perform periodic maintenance and learning consolidation
    brain.daily_consolidation()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Enumerates the high-level types of decisions the brain can make."""
    ACT = "act"           # Take a specific trading action.
    STAND_DOWN = "stand_down"  # Explicitly decide not to act.
    DEFER = "defer"        # Postpone the decision, pending more information.


@dataclass
class CognitiveDecision:
    """
    A structured object representing the final output of the deliberation process.
    This is a rich data packet that encapsulates not just the decision itself,
    but also the reasoning, confidence, and context behind it.
    """
    decision_type: DecisionType
    should_act: bool  # A simple boolean flag for quick checks.
    action: Optional[Dict[str, Any]]  # The specific action to take (e.g., trade details).
    confidence: float  # The brain's calculated confidence (0.0 to 1.0) in its decision.
    reasoning_trace: List[str]  # A step-by-step log of the "thought process".
    concerns: List[str]  # Factors that negatively impacted confidence.
    knowledge_gaps: List[str]  # Information the brain identified as missing.
    invalidators: List[str]  # Conditions that would invalidate this decision.
    episode_id: str  # A unique ID linking this decision to a memory episode.
    decision_mode: str  # The cognitive mode used: 'fast', 'slow', or 'hybrid'.
    processing_time_ms: int = 0  # How long the deliberation took.
    metadata: Dict[str, Any] = field(default_factory=dict)  # For any extra data.

    def to_dict(self) -> Dict:
        """Serializes the decision object to a dictionary for logging or API use."""
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
    The unified cognitive system for intelligent trading decisions.

    This class orchestrates a collection of specialized components to simulate
    an advanced reasoning process. It uses a lazy-loading pattern for its
    components to optimize startup time and resource usage.

    Core Components:
    - GlobalWorkspace: A shared information bus for inter-component communication.
    - MetacognitiveGovernor: The executive function controller. It decides whether
      to use a fast, intuitive "System 1" path or a slow, deliberate "System 2"
      path for a given decision.
    - SelfModel: Represents the agent's awareness of its own capabilities,
      performance, and limitations (e.g., "I am not good at trading in this regime").
    - EpisodicMemory: Stores the history of specific trade episodes, including the
      context, decision, and outcome. It's the agent's experiential memory.
    - SemanticMemory: Stores generalized rules, patterns, and knowledge extracted
      from experience. This is the agent's long-term, abstract knowledge base.
    - ReflectionEngine: Responsible for learning. It analyzes past episodes to
      update the SemanticMemory and SelfModel.
    - KnowledgeBoundary: Helps the agent detect uncertainty and understand the
      limits of its own knowledge, preventing overconfidence.
    - CuriosityEngine: Drives the agent to explore and generate new hypotheses
      about market behavior, enabling proactive learning.
    """

    def __init__(
        self,
        min_confidence_to_act: float = 0.5,
        max_processing_time_ms: int = 5000,
    ):
        """
        Initializes the CognitiveBrain.

        Args:
            min_confidence_to_act: The confidence threshold required to take action.
            max_processing_time_ms: A timeout for the deliberation process to prevent
                                    the system from getting stuck.
        """
        self.min_confidence_to_act = min_confidence_to_act
        self.max_processing_time_ms = max_processing_time_ms

        # Components are lazy-loaded to improve startup performance. They will be
        # instantiated only when first accessed.
        self._workspace = None
        self._governor = None
        self._self_model = None
        self._episodic_memory = None
        self._semantic_memory = None
        self._reflection_engine = None
        self._knowledge_boundary = None
        self._curiosity_engine = None

        # Internal state tracking
        self._initialized = False
        self._decision_count = 0
        self._last_maintenance = None

        logger.info("CognitiveBrain initialized. Components will be lazy-loaded.")

    def _ensure_initialized(self) -> None:
        """
        Ensures all cognitive components are loaded and ready.
        This is called at the start of the deliberation process.
        """
        if self._initialized:
            return

        # The act of accessing each property triggers its lazy-loading.
        logger.info("First deliberation: initializing all cognitive components...")
        _ = self.workspace
        _ = self.governor
        _ = self.self_model
        _ = self.episodic_memory
        _ = self.semantic_memory
        _ = self.knowledge_boundary
        # Reflection, and Curiosity engines are loaded on-demand later.

        self._initialized = True
        logger.info("CognitiveBrain fully initialized and ready for deliberation.")

    # The following properties use a lazy-loading pattern.
    # This means the component is only imported and instantiated the first time
    # it is accessed, which can save resources and speed up application startup.

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
        The main "thinking" process of the brain.

        This method orchestrates the dual-process (Fast/Slow) architecture to make
        a reasoned decision about a given trading signal.

        The process follows these steps:
        1.  **Start Episode:** Create a new memory episode for this decision.
        2.  **Route Decision:** The MetacognitiveGovernor decides whether to use a
            fast "System 1" path or a slow "System 2" path.
        3.  **Assess Knowledge:** The KnowledgeBoundary checks if the agent is
            operating within its area of expertise.
        4.  **Consult Memory:** Query episodic and semantic memory for relevant
            past experiences and learned rules.
        5.  **Calculate Confidence:** Synthesize all information into a final
            confidence score.
        6.  **Make Decision:** Compare confidence to the threshold and decide to act
            or stand down.
        7.  **Record and Publish:** Record the full reasoning and publish the final
            decision to the global workspace.

        Args:
            signal: The trading signal to be evaluated (e.g., from a strategy).
            context: The current market and portfolio context.
            fast_confidence: An optional pre-computed confidence score from a
                             fast, upstream signal processor.

        Returns:
            A CognitiveDecision object containing the final decision and all
            associated reasoning and metadata.
        """
        import time
        start_time = time.time()

        self._ensure_initialized()

        reasoning_trace = []
        concerns = []

        # === Step 1: Start Episode ===
        # Create a new, unique record for this entire thought process.
        # This allows us to link the initial signal, the decision, and the final outcome.
        episode_id = self.episodic_memory.start_episode(
            market_context=context,
            signal_context=signal,
        )
        reasoning_trace.append(f"Starting deliberation for episode {episode_id}")

        # === Step 2: Route Decision (Fast vs Slow) ===
        # The governor makes a metacognitive choice: is this a simple, familiar
        # situation for a fast "System 1" decision, or does it require the full,
        # resource-intensive "System 2" slow path?
        routing = self.governor.route_decision(
            signal=signal,
            context=context,
            fast_confidence=fast_confidence,
        )
        decision_mode = routing.mode.value
        reasoning_trace.append(f"Metacognitive routing: chose '{decision_mode}' path.")

        # The governor can issue an immediate stand-down if conditions are unsafe
        # (e.g., high volatility, kill switch enabled).
        if routing.should_stand_down:
            elapsed_ms = int((time.time() - start_time) * 1000)
            self._decision_count += 1
            reason = routing.stand_down_reason.value if routing.stand_down_reason else 'unknown'
            reasoning_trace.append(f"Immediate stand-down triggered: {reason}")
            return CognitiveDecision(
                decision_type=DecisionType.STAND_DOWN, should_act=False, action=None,
                confidence=0.0, reasoning_trace=reasoning_trace, concerns=[],
                knowledge_gaps=[], invalidators=[], episode_id=episode_id,
                decision_mode=decision_mode, processing_time_ms=elapsed_ms,
            )

        # === Step 3: Assess Knowledge State ===
        # Before proceeding, the brain checks if it even "knows enough" to make
        # a competent decision in this context.
        knowledge_assessment = self.knowledge_boundary.assess_knowledge_state(signal, context)

        if knowledge_assessment.should_stand_down:
            reasoning_trace.append("Knowledge Boundary triggered stand-down: operating outside of known expertise.")
            elapsed_ms = int((time.time() - start_time) * 1000)
            self._decision_count += 1
            return CognitiveDecision(
                decision_type=DecisionType.STAND_DOWN, should_act=False, action=None,
                confidence=0.0, reasoning_trace=reasoning_trace, concerns=concerns,
                knowledge_gaps=knowledge_assessment.missing_information,
                invalidators=[], episode_id=episode_id,
                decision_mode=decision_mode, processing_time_ms=elapsed_ms,
            )

        # Record any identified gaps or conditions that would invalidate the decision.
        knowledge_gaps = knowledge_assessment.missing_information
        invalidators = [inv.description for inv in knowledge_assessment.invalidators]
        if knowledge_gaps:
            reasoning_trace.append(f"Identified knowledge gaps: {knowledge_gaps}")
        if invalidators:
            reasoning_trace.append(f"Identified invalidators: {invalidators}")

        # === Step 4: Consult Memory Systems ===
        # Query episodic memory for similar past trades.
        similar_episodes = self.episodic_memory.find_similar(
            {'regime': context.get('regime'), 'strategy': signal.get('strategy')},
            limit=5
        )
        if similar_episodes:
            win_rate, sample_size = self.episodic_memory.get_win_rate_for_context(
                {'regime': context.get('regime'), 'strategy': signal.get('strategy')}
            )
            reasoning_trace.append(
                f"Episodic Memory: Found {sample_size} similar trades with a {win_rate:.1%} win rate."
            )
        else:
            reasoning_trace.append("Episodic Memory: No similar past trades found.")

        # Query semantic memory for general rules applicable to the current context.
        applicable_rules = self.semantic_memory.get_applicable_rules(context)
        if applicable_rules:
            reasoning_trace.append(f"Semantic Memory: Found {len(applicable_rules)} applicable rules.")
            for rule in applicable_rules[:3]:  # Log top 3 rules for brevity
                reasoning_trace.append(f"Applying rule: IF {rule.condition} THEN {rule.action}")
                if rule.action == "reduce_confidence":
                    concerns.append(f"Caution rule triggered: {rule.condition}")
        else:
            reasoning_trace.append("Semantic Memory: No applicable rules found.")


        # === Step 5: Calculate Final Confidence ===
        # Synthesize all evidence into a single confidence score.
        base_confidence = fast_confidence or 0.6  # Default if no fast confidence provided
        reasoning_trace.append(f"Base confidence set to {base_confidence:.2f}")

        confidence = base_confidence
        # Adjust based on knowledge assessment
        confidence += knowledge_assessment.confidence_adjustment
        if knowledge_assessment.confidence_adjustment != 0:
            reasoning_trace.append(f"Knowledge boundary adjusted confidence by {knowledge_assessment.confidence_adjustment:.2f}")

        # Adjust based on historical win rate
        if similar_episodes:
            if win_rate > 0.6:
                confidence += 0.1
                reasoning_trace.append("Adjusted +0.1 for positive historical win rate.")
            elif win_rate < 0.4:
                confidence -= 0.1
                reasoning_trace.append("Adjusted -0.1 for negative historical win rate.")

        # Adjust based on semantic rules
        for rule in applicable_rules:
            if rule.action == "reduce_confidence":
                confidence -= 0.1
                reasoning_trace.append(f"Adjusted -0.1 due to rule: {rule.condition}")
            elif rule.action == "increase_confidence":
                confidence += 0.05
                reasoning_trace.append(f"Adjusted +0.05 due to rule: {rule.condition}")

        # Apply a confidence ceiling based on how well the agent knows the context.
        confidence_ceiling = self.knowledge_boundary.get_confidence_ceiling(signal, context)
        if confidence > confidence_ceiling:
            reasoning_trace.append(f"Confidence capped from {confidence:.2f} to {confidence_ceiling:.2f} by knowledge boundary.")
            confidence = confidence_ceiling

        # Clamp the final value between 0 and 1.
        confidence = max(0, min(1, confidence))
        reasoning_trace.append(f"Final calculated confidence: {confidence:.2f}")

        # === Step 6: Make Decision ===
        if confidence < self.min_confidence_to_act:
            decision_type = DecisionType.STAND_DOWN
            should_act = False
            action = None
            reasoning_trace.append(
                f"Decision: STAND DOWN (confidence {confidence:.2f} is below threshold {self.min_confidence_to_act})"
            )
        else:
            decision_type = DecisionType.ACT
            should_act = True
            size_multiplier = self._calculate_size_multiplier(confidence)
            action = {
                'type': 'trade',
                'signal': signal,
                'confidence': confidence,
                'size_multiplier': size_multiplier,
            }
            reasoning_trace.append(f"Decision: ACT (confidence {confidence:.2f} meets threshold)")
            reasoning_trace.append(f"Position size multiplier set to {size_multiplier:.2f}")

        # === Step 7: Record Reasoning ===
        # Persist the full thought process to episodic memory for later reflection.
        self.episodic_memory.add_reasoning(episode_id, reasoning_trace)
        self.episodic_memory.add_concerns(episode_id, concerns)
        self.episodic_memory.add_action(episode_id, action or {'type': 'stand_down'}, decision_mode)

        # === Step 8: Publish to Workspace and Return ===
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

        # Publish the decision to the global workspace so other, non-core system
        # components (like a UI or alerter) can react to it.
        self.workspace.publish(
            topic='decision',
            data=decision.to_dict(),
            source='cognitive_brain',
        )

        self._decision_count += 1
        logger.info(
            f"Decision for episode {episode_id}: {decision_type.value} "
            f"(confidence={confidence:.2f}, mode={decision_mode}, time={elapsed_ms}ms)"
        )
        return decision

    def _calculate_size_multiplier(self, confidence: float) -> float:
        """
        Calculates the position size multiplier based on confidence.
        A higher confidence leads to a larger position size, up to the maximum
        allowed by the core risk management system. This allows the cognitive
        layer to express its conviction in a trade.

        Args:
            confidence: The final confidence score (0.0 to 1.0).

        Returns:
            A multiplier between a defined minimum (e.g., 0.25) and 1.0.
        """
        # Simple linear scaling from a minimum size to a full size.
        # e.g., 50% confidence = 50% of allocated budget, 100% confidence = 100%.
        # A minimum is set to ensure even low-confidence trades are meaningful.
        return max(0.25, min(1.0, confidence))

    def learn_from_outcome(
        self,
        episode_id: str,
        outcome: Dict[str, Any],
    ) -> None:
        """
        This method completes the learning cycle: Decision -> Action -> Outcome -> Learning.

        It takes the result of a trade, links it back to the original decision
        episode, and triggers the reflection and learning process.

        Args:
            episode_id: The unique ID of the deliberation episode that led to this trade.
            outcome: A dictionary containing the results of the trade, such as
                     'pnl', 'r_multiple', 'won', etc.
        """
        logger.info(f"Learning from outcome of episode {episode_id}...")
        # Step 1: Record the outcome in the original memory episode.
        self.episodic_memory.complete_episode(episode_id, outcome)

        # Step 2: Retrieve the complete episode (context + decision + outcome).
        episode = self.episodic_memory.get_episode(episode_id)

        if episode:
            # Step 3: Trigger the ReflectionEngine to analyze the episode.
            # This is where new rules are generated or existing ones are updated.
            reflection = self.reflection_engine.reflect_on_episode(episode)
            logger.info(f"Finished learning from episode {episode_id}: {reflection.summary}")
        else:
            logger.warning(f"Could not find episode {episode_id} to learn from.")


    def think_deeper(
        self,
        signal: Dict[str, Any],
        context: Dict[str, Any],
        current_confidence: float,
    ) -> Tuple[float, List[str]]:
        """
        Engages "System 2" thinking for a more profound analysis.

        This is called by the governor when the initial fast-path confidence is
        ambiguous or insufficient. It engages more resource-intensive analysis
        to refine the confidence score.

        Returns:
            A tuple containing the adjusted confidence and a list of
            additional reasoning steps.
        """
        additional_reasoning = []
        confidence_adjustment = 0.0

        # === Deep Analysis 1: Counterfactual Thinking ===
        # Ask "What information, if true, would change my mind?"
        invalidators = self.knowledge_boundary.what_would_change_mind(signal, context)
        for inv in invalidators[:3]:
            additional_reasoning.append(f"Deeper thought: A key invalidator to watch for is '{inv}'.")

        # === Deep Analysis 2: Historical Pattern Matching ===
        # Search for abstract lessons from past episodes, not just raw win rates.
        lessons = self.episodic_memory.get_lessons_for_context(
            {'regime': context.get('regime'), 'strategy': signal.get('strategy')}
        )
        for lesson in lessons[:3]:
            additional_reasoning.append(f"Deeper thought: Applying historical lesson: '{lesson}'.")
            if "AVOID" in lesson:
                confidence_adjustment -= 0.1
            elif "FAVORS" in lesson:
                confidence_adjustment += 0.05

        # === Deep Analysis 3: Self-Model Consultation ===
        # Check the agent's self-assessed competence for this specific situation.
        strategy = signal.get('strategy', 'unknown')
        regime = context.get('regime', 'unknown')

        should_stand, reason = self.self_model.should_stand_down(strategy, regime)
        if should_stand:
            additional_reasoning.append(f"Deeper thought: Self-model warns: '{reason}'.")
            confidence_adjustment -= 0.2

        # === Deep Analysis 4: Complex Rule Application ===
        # This could involve more complex, multi-condition rules from semantic memory.
        rules = self.semantic_memory.get_applicable_rules(context) # Already done, but could be deeper
        for rule in rules[:3]:
            additional_reasoning.append(f"Deeper thought: Re-evaluating rule: IF {rule.condition}...")

        new_confidence = current_confidence + confidence_adjustment
        new_confidence = max(0, min(1, new_confidence))

        return new_confidence, additional_reasoning

    def daily_consolidation(self) -> Dict[str, Any]:
        """
        Performs daily maintenance, learning, and consolidation tasks.

        This should be called at the end of each trading day. It's the AI
        equivalent of "sleeping on it," where it processes the day's events to
        extract new knowledge.
        """
        logger.info("Starting daily consolidation and reflection...")

        results = {}

        # Perform reflection on all trades from the last 24 hours.
        reflection = self.reflection_engine.periodic_reflection(lookback_hours=24)
        results['reflection'] = reflection.to_dict()

        # Allow the curiosity engine to generate and test new hypotheses based on recent data.
        hypotheses = self.curiosity_engine.generate_hypotheses()
        test_results = self.curiosity_engine.test_all_pending()
        results['curiosity'] = {
            'new_hypotheses': len(hypotheses),
            'test_results': test_results,
        }

        # Identify any newly validated "edges" (profitable patterns).
        edges = self.curiosity_engine.get_validated_edges()
        results['edges_count'] = len(edges)
        if edges:
            logger.info(f"Daily consolidation discovered {len(edges)} new trading edges.")

        # Update the self-model with the latest performance data.
        self_desc = self.self_model.get_self_description()
        results['self_description'] = self_desc

        self._last_maintenance = datetime.now()
        logger.info(f"Daily consolidation complete. {len(edges)} edges found, {len(hypotheses)} new hypotheses generated.")
        return results

    def weekly_consolidation(self) -> Dict[str, Any]:
        """
        Performs deeper, more resource-intensive consolidation tasks on a weekly basis.
        This includes pruning old or ineffective rules from semantic memory.
        """
        logger.info("Starting weekly deep consolidation...")

        results = self.reflection_engine.consolidate_learnings()

        # Clean up semantic memory by removing rules that are no longer effective.
        pruned_count = self.semantic_memory.prune_low_confidence()
        results['rules_pruned'] = pruned_count
        if pruned_count > 0:
            logger.info(f"Pruned {pruned_count} low-confidence rules from semantic memory.")

        logger.info(f"Weekly consolidation complete: {results}")
        return results

    def get_status(self) -> Dict[str, Any]:
        """Retrieves a snapshot of the cognitive system's current state and stats."""
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
        Generates a comprehensive introspective report.
        This is the robot "thinking about itself" and reporting on its own state,
        capabilities, and recent learnings. Useful for debugging and monitoring.
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
            "Metacognitive Governor Status",
            "-" * 40,
            self.governor.introspect(),
            "",
            "-" * 40,
            "Curiosity Engine Status",
            "-" * 40,
            self.curiosity_engine.introspect(),
            "",
            "-" * 40,
            "Knowledge Boundary Status",
            "-" * 40,
            self.knowledge_boundary.introspect(),
            "",
            "=" * 60,
        ]

        return "\n".join(lines)


# Singleton pattern: There should only be one "brain" in the application.
# The `_cognitive_brain` variable holds the single instance.
_cognitive_brain: Optional[CognitiveBrain] = None


def get_cognitive_brain() -> CognitiveBrain:
    """
    Factory function to get the singleton instance of the CognitiveBrain.
    Ensures that the same brain instance is used throughout the application.
    """
    global _cognitive_brain
    if _cognitive_brain is None:
        _cognitive_brain = CognitiveBrain()
    return _cognitive_brain
