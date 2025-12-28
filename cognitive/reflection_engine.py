"""
Reflection Engine - The AI's Self-Critique and Learning System
==============================================================

This module implements the "Reflexion" pattern, a sophisticated method for
AI self-improvement inspired by recent research (Shinn et al., "Reflexion:
Language Agents with Verbal Reinforcement Learning"). It acts as the AI's
"inner critic" or "learning coach," reviewing the outcomes of past decisions
to generate actionable insights.

Core Workflow (The Learning Loop):
1.  **Observe Outcome:** The engine receives a completed `Episode` from
    Episodic Memory, containing the context, decision, and final result.
2.  **Analyze and Reflect:** It performs a "postmortem," analyzing what went
    well, what went wrong, and identifying potential root causes.
3.  **Meta-Reflection (LLM):** It then passes this initial analysis to the
    `LLMNarrativeAnalyzer` to get a deeper, more abstract critique from Claude.
4.  **Update Memory:** Finally, it translates the combined insights into concrete
    updates for the other cognitive components (SelfModel, SemanticMemory, etc.).

This process allows the AI to learn and adapt its behavior without retraining
a large model. It learns by thinking about its experiences, much like a human does.

Usage:
    from cognitive.reflection_engine import ReflectionEngine

    engine = ReflectionEngine()

    # After a trade is complete, the brain asks the engine to reflect.
    completed_episode = episodic_memory.get_episode(episode_id)
    reflection = engine.reflect_on_episode(completed_episode)

    # The system can also trigger periodic, higher-level reflection.
    daily_summary = engine.periodic_reflection(lookback_hours=24)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from collections import Counter

# Import the new analyzer
from cognitive.llm_narrative_analyzer import get_llm_analyzer

logger = logging.getLogger(__name__)


@dataclass
class Reflection:
    """
    A structured report containing the output of a reflection process.
    This now includes a field for critique from an external LLM and
    proposed cognitive parameter adjustments (Task B1).
    """
    scope: str  # The scope of the reflection: "episode", "daily", "weekly".
    timestamp: datetime = field(default_factory=datetime.now)
    summary: str = "" # A concise, human-readable summary of the findings.
    what_went_well: List[str] = field(default_factory=list)
    what_went_wrong: List[str] = field(default_factory=list)
    root_causes: List[str] = field(default_factory=list) # Hypothesized reasons for failures.
    lessons: List[str] = field(default_factory=list) # Actionable takeaways.
    action_items: List[str] = field(default_factory=list) # Concrete tasks for the system.
    behavior_changes: List[Dict[str, Any]] = field(default_factory=list) # Proposed rule changes.
    confidence_adjustment: float = 0.0 # A suggested global confidence adjustment.
    # --- New field for LLM integration ---
    llm_critique: Optional[str] = None # Stores the meta-analysis from the LLM.
    # --- Meta-metacognitive field (Task B1) ---
    cognitive_adjustments: Optional[Dict[str, Any]] = None  # Proposed param adjustments
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serializes the reflection to a dictionary."""
        # asdict is not used here to have more control.
        return {
            'timestamp': self.timestamp.isoformat(),
            'scope': self.scope,
            'summary': self.summary,
            'what_went_well': self.what_went_well,
            'what_went_wrong': self.what_went_wrong,
            'root_causes': self.root_causes,
            'lessons': self.lessons,
            'action_items': self.action_items,
            'behavior_changes': self.behavior_changes,
            'confidence_adjustment': self.confidence_adjustment,
            'llm_critique': self.llm_critique,
            'cognitive_adjustments': self.cognitive_adjustments,
            'metadata': self.metadata,
        }


class ReflectionEngine:
    """
    The self-critique system that reviews outcomes and generates learnings. It
    drives the AI's adaptation and improvement by reflecting on past performance.
    """

    def __init__(self):
        # Dependencies are lazy-loaded to prevent circular import issues.
        self._episodic_memory = None
        self._semantic_memory = None
        self._self_model = None
        self._workspace = None
        self._llm_analyzer = None  # LLM analyzer for meta-reflection
        self._curiosity_engine = None  # CuriosityEngine for hypothesis testing

        self._reflections: List[Reflection] = []
        self._max_reflections = 100  # Limit the history of reflections.

        logger.info("ReflectionEngine initialized.")

    # --- Lazy-loaded properties for dependencies ---
    @property
    def episodic_memory(self):
        if self._episodic_memory is None: from cognitive.episodic_memory import get_episodic_memory; self._episodic_memory = get_episodic_memory()
        return self._episodic_memory

    @property
    def semantic_memory(self):
        if self._semantic_memory is None: from cognitive.semantic_memory import get_semantic_memory; self._semantic_memory = get_semantic_memory()
        return self._semantic_memory

    @property
    def self_model(self):
        if self._self_model is None: from cognitive.self_model import get_self_model; self._self_model = get_self_model()
        return self._self_model

    @property
    def workspace(self):
        if self._workspace is None: from cognitive.global_workspace import get_workspace; self._workspace = get_workspace()
        return self._workspace

    @property
    def llm_analyzer(self):
        """Lazy-loads the LLM Narrative Analyzer."""
        if self._llm_analyzer is None:
            self._llm_analyzer = get_llm_analyzer()
        return self._llm_analyzer

    @property
    def curiosity_engine(self):
        """Lazy-loads the CuriosityEngine."""
        if self._curiosity_engine is None:
            from cognitive.curiosity_engine import get_curiosity_engine
            self._curiosity_engine = get_curiosity_engine()
        return self._curiosity_engine

    def reflect_on_episode(self, episode: Any) -> Reflection:
        """
        Performs a detailed reflection on a single completed trading episode.
        This now includes a meta-reflection step using an LLM.

        Note: Simulated episodes (Task B2) are weighted with 0.5x confidence
        to ensure real experience is valued more highly than synthetic data.
        """
        from cognitive.episodic_memory import EpisodeOutcome
        reflection = Reflection(scope="episode")

        # --- 0. Check for Simulated Episode (Task B2) ---
        # Simulated episodes get reduced weight in learning
        is_simulated = getattr(episode, 'is_simulated', False)
        simulation_weight = 0.5 if is_simulated else 1.0

        if is_simulated:
            reflection.metadata['is_simulated'] = True
            reflection.metadata['simulation_source'] = getattr(episode, 'simulation_source', 'unknown')
            reflection.metadata['simulation_weight'] = simulation_weight
            reflection.lessons.append(
                f"[SIMULATED] This learning is from synthetic data ({episode.simulation_source or 'unknown'}) "
                f"and carries {simulation_weight:.0%} weight."
            )

        # --- 1. Analyze the Outcome (Internal Analysis) ---
        if episode.outcome == EpisodeOutcome.WIN:
            reflection.what_went_well.append(f"Trade won with {episode.r_multiple:.1f}R profit.")
            self._analyze_winning_trade(episode, reflection)
        elif episode.outcome == EpisodeOutcome.LOSS:
            reflection.what_went_wrong.append(f"Trade lost with {episode.r_multiple:.1f}R loss.")
            self._analyze_losing_trade(episode, reflection)
        elif episode.outcome == EpisodeOutcome.STAND_DOWN:
            reflection.what_went_well.append("Wisely decided to stand down and avoid potential loss.")
            self._analyze_stand_down(episode, reflection)

        # --- 2. Generate Summary and Actions (Based on Internal Analysis) ---
        reflection.summary = self._generate_summary(episode, reflection)
        reflection.action_items = self._generate_action_items(reflection)

        # --- 3. Perform Meta-Reflection with LLM ---
        # Pass the initial reflection to the LLM for a deeper, more abstract critique.
        # The LLM now returns an LLMAnalysisResult with critique, hypotheses, and strategy ideas.
        if self.llm_analyzer:
            llm_result = self.llm_analyzer.analyze_reflection(reflection)
            if llm_result.critique:
                reflection.llm_critique = llm_result.critique
                logger.info("LLM meta-reflection critique added.")

            # Pass any extracted hypotheses to the CuriosityEngine for testing
            if llm_result.hypotheses:
                added = self.curiosity_engine.add_llm_generated_hypotheses(llm_result.hypotheses)
                if added:
                    logger.info(f"Added {len(added)} LLM hypotheses to CuriosityEngine.")

            # Pass any extracted strategy ideas to the CuriosityEngine
            if llm_result.strategy_ideas:
                strategy_records = self.curiosity_engine.add_llm_generated_strategy_ideas(llm_result.strategy_ideas)
                if strategy_records:
                    logger.info(f"Added {len(strategy_records)} LLM strategy ideas to CuriosityEngine.")

        # --- 3.5 Propose Cognitive Parameter Adjustments (Task B1) ---
        # Ask the self-model if any cognitive parameters should be adjusted based on efficiency data
        cognitive_adjustments = self.self_model.propose_cognitive_param_adjustments()
        if cognitive_adjustments.get('pending_count', 0) > 0:
            reflection.cognitive_adjustments = cognitive_adjustments
            logger.info(f"Proposed {cognitive_adjustments['pending_count']} cognitive parameter adjustments.")

        # --- 4. Apply the Learnings ---
        # This is the most critical step: turning reflection into concrete changes.
        self._apply_learnings(episode, reflection)

        self._reflections.append(reflection)
        if len(self._reflections) > self._max_reflections:
            self._reflections.pop(0)

        # Announce the complete reflection (including LLM critique) to the rest of the system.
        self.workspace.publish(topic='reflection', data=reflection.to_dict(), source='ReflectionEngine')
        logger.info(f"Reflection on episode {episode.episode_id} complete. Found {len(reflection.lessons)} lessons.")
        return reflection

    def _analyze_winning_trade(self, episode: Any, reflection: Reflection) -> None:
        """Analyzes a successful trade to identify repeatable patterns."""
        if episode.reasoning_trace:
            reflection.what_went_well.append("The reasoning process was clear and logical.")
        
        avg_conf = sum(episode.confidence_levels.values()) / len(episode.confidence_levels) if episode.confidence_levels else 0
        if avg_conf > 0.7:
            reflection.what_went_well.append("The high confidence in the decision was justified.")
            reflection.lessons.append("Trust high-confidence signals in similar contexts.")
        
        context_str = self._describe_context(episode)
        lesson = f"The pattern '{episode.signal_context.get('strategy', 'unknown')}' seems effective in the context of '{context_str}'."
        reflection.lessons.append(lesson)
        reflection.action_items.append(f"Strengthen rule for '{lesson}'")

    def _analyze_losing_trade(self, episode: Any, reflection: Reflection) -> None:
        """Analyzes a failed trade to find root causes and prevent recurrence."""
        if episode.concerns_noted:
            reflection.what_went_wrong.append("Proceeded despite noting valid concerns.")
            reflection.root_causes.append("Ignored internal warning signs.")
            reflection.lessons.append("Give more weight to noted concerns before acting.")

        avg_conf = sum(episode.confidence_levels.values()) / len(episode.confidence_levels) if episode.confidence_levels else 0
        if avg_conf > 0.7:
            reflection.what_went_wrong.append(f"Was overconfident ({avg_conf:.1%}) in a losing trade.")
            reflection.root_causes.append("Confidence may be miscalibrated for this context.")
            reflection.lessons.append("Re-evaluate confidence model for this context; high confidence is not infallible.")
            reflection.confidence_adjustment = -0.1 # Suggests a global downward adjustment.
        
        if episode.decision_mode == "fast":
            reflection.what_went_wrong.append("A fast-path decision may have missed crucial details.")
            reflection.root_causes.append("Insufficient deliberation for a complex situation.")
            reflection.lessons.append("Consider escalating similar situations to the slow path in the future.")

        context_str = self._describe_context(episode)
        lesson = f"The pattern '{episode.signal_context.get('strategy', 'unknown')}' may be unreliable in the context of '{context_str}'."
        reflection.lessons.append(lesson)
        reflection.action_items.append(f"Create or strengthen a caution rule for '{lesson}'")

    def _analyze_stand_down(self, episode: Any, reflection: Reflection) -> None:
        """Analyzes whether a decision to stand down was correct."""
        # To judge a stand-down, we need to know what *would* have happened.
        # This requires a market data provider that can give historical data
        # for what the trade would have done. (This is a simplification).
        # For now, we check if similar trades that *were* taken succeeded.
        from cognitive.episodic_memory import EpisodeOutcome
        similar_episodes = self.episodic_memory.find_similar(episode.signal_context, limit=10)
        
        proceeded_trades = [e for e in similar_episodes if e.outcome not in [EpisodeOutcome.STAND_DOWN, EpisodeOutcome.PENDING]]
        if not proceeded_trades:
            reflection.what_went_well.append("No comparable trades found; cautious stand-down was prudent.")
            return

        win_rate = sum(1 for e in proceeded_trades if e.outcome == EpisodeOutcome.WIN) / len(proceeded_trades)
        if win_rate < 0.5:
            reflection.what_went_well.append(f"Correctly avoided a low-probability setup (similar trades had a {win_rate:.1%} win rate).")
            reflection.lessons.append("Continue to trust stand-down decisions in this context.")
        else:
            reflection.what_went_wrong.append(f"May have been overly cautious; similar trades had a {win_rate:.1%} win rate.")
            reflection.lessons.append("Consider lowering the stand-down threshold for this context.")

    def _describe_context(self, episode: Any) -> str:
        """Creates a simple, human-readable description of the episode's context."""
        parts = [
            episode.market_context.get('regime'),
            episode.signal_context.get('strategy')
        ]
        return " / ".join(filter(None, parts)) or "unknown context"

    def _generate_summary(self, episode: Any, reflection: Reflection) -> str:
        """Generates a concise summary of the reflection."""
        summary = f"Reflection on {episode.outcome.value} episode for {self._describe_context(episode)}."
        if reflection.lessons:
            summary += f" Key lesson: {reflection.lessons[0]}"
        return summary

    def _generate_action_items(self, reflection: Reflection) -> List[str]:
        """Translates lessons into concrete system actions."""
        actions = []
        for lesson in reflection.lessons:
            if "strengthen rule" in lesson or "create... caution rule" in lesson:
                actions.append(f"Update Semantic Memory with rule based on: '{lesson}'")
        if "Re-evaluate confidence model" in reflection.lessons:
            actions.append("Trigger confidence calibration analysis in SelfModel.")
        return actions

    def _apply_learnings(self, episode: Any, reflection: Reflection) -> None:
        """
        This is the most important method. It takes the abstract lessons from a
        reflection and applies them as concrete updates to the AI's memory systems.
        """
        from cognitive.episodic_memory import EpisodeOutcome

        # 1. Update Episodic Memory: Add the reflection to the original memory trace.
        self.episodic_memory.add_postmortem(
            episode.episode_id,
            postmortem=reflection.summary,
            lessons=reflection.lessons,
            mistakes=reflection.what_went_wrong,
            what_to_repeat=reflection.what_went_well if episode.outcome == EpisodeOutcome.WIN else [],
            what_to_avoid=reflection.what_went_wrong if episode.outcome == EpisodeOutcome.LOSS else [],
        )

        # 2. Update Self-Model: Let the AI know how it performed on this specific task.
        self.self_model.record_trade_outcome(
            strategy=episode.signal_context.get('strategy', 'unknown'),
            regime=episode.market_context.get('regime', 'unknown'),
            won=(episode.outcome == EpisodeOutcome.WIN),
            pnl=episode.pnl,
            r_multiple=episode.r_multiple,
            notes=reflection.summary
        )

        # 3. Update Semantic Memory: Create or modify general rules based on the experience.
        # This is how a single experience becomes generalized knowledge.
        # Apply simulation weight (Task B2) to reduce confidence for synthetic data
        is_simulated = getattr(episode, 'is_simulated', False)
        simulation_weight = 0.5 if is_simulated else 1.0
        context_condition = f"regime = {episode.market_context.get('regime', 'unknown')} AND strategy = {episode.signal_context.get('strategy', 'unknown')}"

        # Build tags including simulation flag if applicable
        base_tags = ['reflection', 'auto-generated']
        if is_simulated:
            base_tags.append('simulated')

        if episode.outcome == EpisodeOutcome.LOSS and "Overconfident" in " ".join(reflection.what_went_wrong):
             # If we were overconfident and lost, create a rule to be more cautious.
            self.semantic_memory.add_rule(
                condition=context_condition,
                action="reduce_confidence",
                parameters={'reason': f"Past overconfident loss in this context ({reflection.summary})"},
                confidence=0.7 * simulation_weight,  # Apply simulation weight
                source=f"Reflection on episode {episode.episode_id}",
                tags=base_tags + ['caution']
            )
        elif episode.outcome == EpisodeOutcome.WIN and "justified" in " ".join(reflection.what_went_well):
            # If we were confident and won, reinforce that behavior.
            self.semantic_memory.add_rule(
                condition=context_condition,
                action="increase_confidence",
                parameters={'reason': f"Past justified high-confidence win ({reflection.summary})"},
                confidence=0.6 * simulation_weight,  # Apply simulation weight
                source=f"Reflection on episode {episode.episode_id}",
                tags=base_tags + ['positive']
            )

        # 4. Record Cognitive Efficiency Feedback (Task B1)
        # Track which decision mode (fast/slow/hybrid) worked in which context
        try:
            decision_mode = episode.metadata.get('decision_mode', 'slow') if hasattr(episode, 'metadata') and isinstance(episode.metadata, dict) else 'slow'
            decision_time_ms = episode.metadata.get('decision_time_ms', 0) if hasattr(episode, 'metadata') and isinstance(episode.metadata, dict) else 0
            # Ensure decision_time_ms is numeric
            if not isinstance(decision_time_ms, (int, float)):
                decision_time_ms = 0
        except (AttributeError, TypeError):
            decision_mode = 'slow'
            decision_time_ms = 0

        llm_summary = reflection.llm_critique[:200] if reflection.llm_critique else None

        self.self_model.record_cognitive_efficiency_feedback(
            decision_id=episode.episode_id,
            decision_mode_used=decision_mode,
            strategy=episode.signal_context.get('strategy', 'unknown'),
            regime=episode.market_context.get('regime', 'unknown'),
            actual_pnl=episode.pnl,
            llm_critique_summary=llm_summary,
            time_taken_ms=float(decision_time_ms),
        )

    def periodic_reflection(self, lookback_hours: int = 24) -> Reflection:
        """
        Performs a higher-level reflection on all episodes within a recent time
        window (e.g., daily). It looks for recurring patterns and trends.
        """
        from cognitive.episodic_memory import EpisodeOutcome
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        episodes = [e for e in self.episodic_memory.get_recent_episodes(limit=100) if e.completed_at and e.completed_at >= cutoff]

        if not episodes:
            return Reflection(scope="daily", summary="No trades to reflect on.")

        reflection = Reflection(scope="daily")
        wins = [e for e in episodes if e.outcome == EpisodeOutcome.WIN]
        losses = [e for e in episodes if e.outcome == EpisodeOutcome.LOSS]
        total_pnl = sum(e.pnl for e in episodes)
        win_rate = len(wins) / (len(wins) + len(losses)) if (wins or losses) else 0

        reflection.metadata = {'episode_count': len(episodes), 'win_rate': win_rate, 'total_pnl': total_pnl}
        reflection.summary = f"Daily reflection: {len(episodes)} trades resulted in PnL of ${total_pnl:.2f} with a {win_rate:.1%} win rate."

        # Look for the most common context for losses.
        if losses:
            loss_contexts = Counter(self._describe_context(e) for e in losses)
            if common_loss_contexts := loss_contexts.most_common(1):
                if common_loss_contexts[0][1] > 1:
                    ctx, count = common_loss_contexts[0]
                    reflection.what_went_wrong.append(f"Encountered repeated losses ({count}x) in the context: '{ctx}'.")
                    reflection.lessons.append(f"System should be more cautious or stand down in '{ctx}' context.")
                    reflection.action_items.append(f"Create/strengthen a caution rule for '{ctx}'.")
        
        # Add LLM critique to periodic reflection as well
        if self.llm_analyzer:
            llm_result = self.llm_analyzer.analyze_reflection(reflection)
            if llm_result.critique:
                reflection.llm_critique = llm_result.critique
            # Pass any extracted hypotheses to the CuriosityEngine
            if llm_result.hypotheses:
                added = self.curiosity_engine.add_llm_generated_hypotheses(llm_result.hypotheses)
                if added:
                    logger.info(f"Periodic reflection added {len(added)} LLM hypotheses.")
            # Pass any extracted strategy ideas to the CuriosityEngine
            # Periodic reflections are great times for strategy innovation
            if llm_result.strategy_ideas:
                strategy_records = self.curiosity_engine.add_llm_generated_strategy_ideas(llm_result.strategy_ideas)
                if strategy_records:
                    logger.info(f"Periodic reflection added {len(strategy_records)} LLM strategy ideas.")

        self.workspace.publish(topic='reflection', data=reflection.to_dict(), source='ReflectionEngine')
        logger.info(reflection.summary)
        return reflection

    def consolidate_learnings(self) -> Dict[str, Any]:
        """
        A deeper, weekly process to consolidate knowledge, prune old rules,
        and update long-term models.
        """
        logger.info("Performing weekly learning consolidation...")
        # 1. Have the semantic memory try to extract new, high-level rules from recent history.
        new_rules = self.semantic_memory.extract_rules_from_episodes(self.episodic_memory.get_recent_episodes(limit=200))
        # 2. Clean up by pruning old or low-performing rules.
        pruned_count = self.semantic_memory.prune_low_confidence(threshold=0.4)
        # 3. Ask the self-model to generate a fresh summary of its capabilities.
        self_description = self.self_model.get_self_description()
        
        result = {
            'new_rules_extracted': len(new_rules),
            'rules_pruned': pruned_count,
            'new_self_description': self_description,
        }
        logger.info(f"Weekly consolidation complete: {result}")
        return result
        
    def introspect(self) -> str:
        """Generates a human-readable report on the engine's activities."""
        total_lessons = sum(len(r.lessons) for r in self._reflections)
        return (
            "--- Reflection Engine Introspection ---\n"
            f"I have performed {len(self._reflections)} reflections and extracted {total_lessons} lessons.\n"
            "My purpose is to learn from every action, win or lose, to continuously improve."
        )


# --- Singleton Implementation ---
_reflection_engine: Optional[ReflectionEngine] = None

def get_reflection_engine() -> ReflectionEngine:
    """Factory function to get the singleton instance of ReflectionEngine."""
    global _reflection_engine
    if _reflection_engine is None:
        _reflection_engine = ReflectionEngine()
    return _reflection_engine