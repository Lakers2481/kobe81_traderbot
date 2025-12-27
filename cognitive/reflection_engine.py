"""
Reflection Engine - Self-Critique System
==========================================

Implements the Reflexion pattern from arXiv:
- Reviews outcomes and generates reflections
- Writes insights into memory without retraining
- Changes behavior through memory updates

This is the "judge/critic" that reviews what happened and
generates actionable feedback for improvement.

Features:
- Automated postmortem analysis
- Pattern detection across failures
- Lesson extraction and consolidation
- Behavior modification through memory

Usage:
    from cognitive.reflection_engine import ReflectionEngine

    engine = ReflectionEngine()

    # After trade completes
    reflection = engine.reflect_on_episode(episode)

    # Periodic reflection on recent performance
    insights = engine.periodic_reflection()

    # Deep reflection (weekly consolidation)
    engine.consolidate_learnings()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class Reflection:
    """Result of reflecting on an episode or period."""
    timestamp: datetime = field(default_factory=datetime.now)
    scope: str = "episode"  # episode, daily, weekly
    summary: str = ""
    what_went_well: List[str] = field(default_factory=list)
    what_went_wrong: List[str] = field(default_factory=list)
    root_causes: List[str] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    behavior_changes: List[Dict[str, Any]] = field(default_factory=list)
    confidence_adjustment: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
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
        }


class ReflectionEngine:
    """
    Self-critique system that reviews outcomes and generates learnings.

    Implements the Reflexion pattern:
    - Episode -> Outcome -> Reflection -> Memory Update

    Key difference from standard learning: changes behavior through
    memory updates, not weight updates.
    """

    def __init__(self):
        # Lazy load dependencies
        self._episodic_memory = None
        self._semantic_memory = None
        self._self_model = None
        self._workspace = None

        # Reflection history
        self._reflections: List[Reflection] = []
        self._max_reflections = 100

        logger.info("ReflectionEngine initialized")

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
    def self_model(self):
        if self._self_model is None:
            from cognitive.self_model import get_self_model
            self._self_model = get_self_model()
        return self._self_model

    @property
    def workspace(self):
        if self._workspace is None:
            from cognitive.global_workspace import get_workspace
            self._workspace = get_workspace()
        return self._workspace

    def reflect_on_episode(self, episode: Any) -> Reflection:
        """
        Reflect on a single completed episode.

        Generates:
        - What went well/wrong
        - Root cause analysis
        - Lessons learned
        - Action items

        Args:
            episode: Completed Episode object

        Returns:
            Reflection with insights
        """
        from cognitive.episodic_memory import EpisodeOutcome

        reflection = Reflection(scope="episode")

        # === Analyze Outcome ===
        if episode.outcome == EpisodeOutcome.WIN:
            reflection.what_went_well.append(
                f"Trade won with {episode.r_multiple:.1f}R profit"
            )
            self._analyze_winning_trade(episode, reflection)

        elif episode.outcome == EpisodeOutcome.LOSS:
            reflection.what_went_wrong.append(
                f"Trade lost with {episode.r_multiple:.1f}R loss"
            )
            self._analyze_losing_trade(episode, reflection)

        elif episode.outcome == EpisodeOutcome.STAND_DOWN:
            reflection.what_went_well.append("Correctly stood down when uncertain")
            self._analyze_stand_down(episode, reflection)

        # === Generate Summary ===
        reflection.summary = self._generate_summary(episode, reflection)

        # === Generate Action Items ===
        reflection.action_items = self._generate_action_items(reflection)

        # === Apply Learnings ===
        self._apply_learnings(episode, reflection)

        # Store reflection
        self._reflections.append(reflection)
        if len(self._reflections) > self._max_reflections:
            self._reflections = self._reflections[-self._max_reflections:]

        # Publish to workspace
        self.workspace.publish(
            topic='reflection',
            data=reflection.to_dict(),
            source='reflection_engine',
        )

        logger.info(f"Reflected on episode: {len(reflection.lessons)} lessons")

        return reflection

    def _analyze_winning_trade(self, episode: Any, reflection: Reflection) -> None:
        """Analyze what made a winning trade successful."""
        # Check reasoning quality
        if episode.reasoning_trace:
            reflection.what_went_well.append(
                f"Clear reasoning with {len(episode.reasoning_trace)} steps"
            )

        # Check confidence calibration
        if episode.confidence_levels:
            avg_conf = sum(episode.confidence_levels.values()) / len(episode.confidence_levels)
            if avg_conf > 0.7:
                reflection.what_went_well.append("High confidence was justified")
                reflection.lessons.append("Trust high-confidence signals in similar contexts")

        # Check decision mode
        if episode.decision_mode == "slow":
            reflection.lessons.append("Deliberate analysis paid off - continue using slow mode for similar situations")
        elif episode.decision_mode == "fast":
            reflection.lessons.append("Fast decision was accurate - this context is well-understood")

        # What to repeat
        context_str = self._describe_context(episode)
        reflection.what_went_well.append(f"Context: {context_str}")
        reflection.lessons.append(f"REPEAT: Trade {episode.signal_context.get('strategy', 'unknown')} in {context_str}")

    def _analyze_losing_trade(self, episode: Any, reflection: Reflection) -> None:
        """Analyze what went wrong in a losing trade."""
        # Check if concerns were noted but ignored
        if episode.concerns_noted:
            reflection.what_went_wrong.append(
                f"Had concerns but proceeded: {', '.join(episode.concerns_noted[:2])}"
            )
            reflection.root_causes.append("Ignored warning signs")
            reflection.lessons.append("RESPECT concerns - if noted, act on them")

        # Check if alternatives were considered
        if not episode.alternatives_considered:
            reflection.what_went_wrong.append("No alternatives considered")
            reflection.root_causes.append("Tunnel vision on single approach")
            reflection.lessons.append("Always consider at least one alternative")

        # Check confidence vs outcome
        if episode.confidence_levels:
            avg_conf = sum(episode.confidence_levels.values()) / len(episode.confidence_levels)
            if avg_conf > 0.7:
                reflection.what_went_wrong.append(f"Overconfident ({avg_conf:.0%}) despite loss")
                reflection.root_causes.append("Miscalibrated confidence")
                reflection.lessons.append("Recalibrate - high confidence doesn't guarantee success")
                reflection.confidence_adjustment = -0.1

        # Context analysis
        context_str = self._describe_context(episode)
        reflection.what_went_wrong.append(f"Context: {context_str}")
        reflection.lessons.append(f"CAUTION: Review {episode.signal_context.get('strategy', 'unknown')} in {context_str}")

        # Decision mode analysis
        if episode.decision_mode == "fast":
            reflection.what_went_wrong.append("Used fast path - may have missed details")
            reflection.root_causes.append("Insufficient deliberation")
            reflection.lessons.append("Use slow mode for this context in future")

    def _analyze_stand_down(self, episode: Any, reflection: Reflection) -> None:
        """Analyze a stand-down decision."""
        # Check what triggered stand-down
        if episode.concerns_noted:
            reflection.what_went_well.append(
                f"Correctly identified risks: {', '.join(episode.concerns_noted[:2])}"
            )

        # Find similar episodes that proceeded - were they successful?
        similar = self.episodic_memory.find_similar({
            'regime': episode.market_context.get('regime'),
            'strategy': episode.signal_context.get('strategy'),
        }, limit=10)

        from cognitive.episodic_memory import EpisodeOutcome
        proceeded = [e for e in similar if e.outcome != EpisodeOutcome.STAND_DOWN]
        if proceeded:
            wins = sum(1 for e in proceeded if e.outcome == EpisodeOutcome.WIN)
            losses = sum(1 for e in proceeded if e.outcome == EpisodeOutcome.LOSS)
            if wins + losses > 0:
                win_rate = wins / (wins + losses)
                if win_rate < 0.5:
                    reflection.what_went_well.append(
                        f"Good call - similar trades had {win_rate:.0%} win rate"
                    )
                    reflection.lessons.append("Continue standing down in similar situations")
                else:
                    reflection.what_went_wrong.append(
                        f"May have been too cautious - similar trades had {win_rate:.0%} win rate"
                    )
                    reflection.lessons.append("Consider reducing stand-down threshold for this context")

    def _describe_context(self, episode: Any) -> str:
        """Generate a description of the episode context."""
        parts = []
        if episode.market_context.get('regime'):
            parts.append(episode.market_context['regime'])
        if episode.signal_context.get('strategy'):
            parts.append(episode.signal_context['strategy'])
        return " + ".join(parts) if parts else "unknown context"

    def _generate_summary(self, episode: Any, reflection: Reflection) -> str:
        """Generate a natural language summary."""
        from cognitive.episodic_memory import EpisodeOutcome

        outcome_word = {
            EpisodeOutcome.WIN: "winning",
            EpisodeOutcome.LOSS: "losing",
            EpisodeOutcome.STAND_DOWN: "standing down from",
            EpisodeOutcome.BREAKEVEN: "breakeven",
        }.get(episode.outcome, "completing")

        summary_parts = [
            f"Reflected on {outcome_word} trade.",
        ]

        if reflection.lessons:
            summary_parts.append(f"Key lesson: {reflection.lessons[0]}")

        if reflection.what_went_well:
            summary_parts.append(f"Positive: {reflection.what_went_well[0]}")

        if reflection.what_went_wrong:
            summary_parts.append(f"To improve: {reflection.what_went_wrong[0]}")

        return " ".join(summary_parts)

    def _generate_action_items(self, reflection: Reflection) -> List[str]:
        """Generate actionable items from reflection."""
        actions = []

        # From lessons
        for lesson in reflection.lessons:
            if lesson.startswith("REPEAT:"):
                actions.append(f"Continue: {lesson[7:].strip()}")
            elif lesson.startswith("CAUTION:"):
                actions.append(f"Add caution rule for: {lesson[8:].strip()}")
            elif lesson.startswith("AVOID:"):
                actions.append(f"Add avoid rule for: {lesson[6:].strip()}")

        # From root causes
        if "Miscalibrated confidence" in reflection.root_causes:
            actions.append("Run calibration analysis on recent predictions")

        if "Ignored warning signs" in reflection.root_causes:
            actions.append("Increase weight of concerns in decision making")

        return actions

    def _apply_learnings(self, episode: Any, reflection: Reflection) -> None:
        """Apply learnings to memory systems."""
        from cognitive.episodic_memory import EpisodeOutcome

        # Update episode with postmortem
        self.episodic_memory.add_postmortem(
            episode.episode_id,
            postmortem=reflection.summary,
            lessons=reflection.lessons,
            mistakes=reflection.what_went_wrong,
            what_to_repeat=reflection.what_went_well if episode.outcome == EpisodeOutcome.WIN else [],
            what_to_avoid=reflection.what_went_wrong if episode.outcome == EpisodeOutcome.LOSS else [],
        )

        # Update self-model
        strategy = episode.signal_context.get('strategy', 'unknown')
        regime = episode.market_context.get('regime', 'unknown')

        self.self_model.record_trade_outcome(
            strategy=strategy,
            regime=regime,
            won=episode.outcome == EpisodeOutcome.WIN,
            pnl=episode.pnl,
            r_multiple=episode.r_multiple,
            notes=reflection.summary,
        )

        # Create semantic rules from strong lessons
        for lesson in reflection.lessons:
            if lesson.startswith("CAUTION:") or lesson.startswith("AVOID:"):
                condition = f"regime = {regime} AND strategy = {strategy}"
                self.semantic_memory.add_rule(
                    condition=condition,
                    action="reduce_confidence",
                    parameters={'reason': lesson},
                    confidence=0.6,
                    source="Reflection on loss",
                    tags=['reflection', 'caution'],
                )
            elif lesson.startswith("REPEAT:"):
                condition = f"regime = {regime} AND strategy = {strategy}"
                self.semantic_memory.add_rule(
                    condition=condition,
                    action="increase_confidence",
                    parameters={'reason': lesson},
                    confidence=0.6,
                    source="Reflection on win",
                    tags=['reflection', 'positive'],
                )

    def periodic_reflection(
        self,
        lookback_hours: int = 24,
    ) -> Reflection:
        """
        Perform periodic reflection on recent episodes.

        Should be called daily or at end of trading session.
        """
        from cognitive.episodic_memory import EpisodeOutcome

        cutoff = datetime.now() - timedelta(hours=lookback_hours)

        episodes = [
            e for e in self.episodic_memory.get_recent_episodes(limit=50)
            if e.completed_at and e.completed_at >= cutoff
        ]

        if not episodes:
            return Reflection(
                scope="daily",
                summary="No trades to reflect on",
            )

        reflection = Reflection(scope="daily")

        # === Aggregate Statistics ===
        wins = [e for e in episodes if e.outcome == EpisodeOutcome.WIN]
        losses = [e for e in episodes if e.outcome == EpisodeOutcome.LOSS]
        stand_downs = [e for e in episodes if e.outcome == EpisodeOutcome.STAND_DOWN]

        total_pnl = sum(e.pnl for e in episodes)
        win_rate = len(wins) / (len(wins) + len(losses)) if (wins or losses) else 0

        reflection.metadata['episode_count'] = len(episodes)
        reflection.metadata['win_rate'] = win_rate
        reflection.metadata['total_pnl'] = total_pnl

        # === What Went Well ===
        if win_rate > 0.6:
            reflection.what_went_well.append(f"Strong win rate: {win_rate:.0%}")
        if total_pnl > 0:
            reflection.what_went_well.append(f"Profitable day: ${total_pnl:.2f}")
        if stand_downs:
            reflection.what_went_well.append(f"Showed discipline: {len(stand_downs)} stand-downs")

        # === What Went Wrong ===
        if win_rate < 0.4:
            reflection.what_went_wrong.append(f"Low win rate: {win_rate:.0%}")
        if total_pnl < 0:
            reflection.what_went_wrong.append(f"Losing day: ${total_pnl:.2f}")

        # === Pattern Detection ===
        if losses:
            loss_contexts = Counter(self._describe_context(e) for e in losses)
            common_loss_context = loss_contexts.most_common(1)
            if common_loss_context and common_loss_context[0][1] >= 2:
                ctx, count = common_loss_context[0]
                reflection.what_went_wrong.append(
                    f"Repeated losses ({count}x) in: {ctx}"
                )
                reflection.lessons.append(f"AVOID or reduce size in: {ctx}")

        # === Generate Summary ===
        reflection.summary = (
            f"Daily reflection: {len(episodes)} trades, "
            f"{win_rate:.0%} win rate, ${total_pnl:.2f} P&L"
        )

        # Store
        self._reflections.append(reflection)

        # Publish
        self.workspace.publish(
            topic='reflection',
            data=reflection.to_dict(),
            source='reflection_engine',
        )

        logger.info(f"Daily reflection: {reflection.summary}")

        return reflection

    def consolidate_learnings(self) -> Dict[str, Any]:
        """
        Weekly consolidation of learnings.

        - Extract rules from episodes
        - Prune low-confidence rules
        - Update self-model summary
        """
        logger.info("Starting weekly consolidation")

        # Extract rules from recent episodes
        episodes = self.episodic_memory.get_recent_episodes(limit=100)
        new_rules = self.semantic_memory.extract_rules_from_episodes(episodes)

        # Prune low-confidence rules
        pruned = self.semantic_memory.prune_low_confidence(threshold=0.3)

        # Generate fresh self-description
        self_desc = self.self_model.get_self_description()

        result = {
            'new_rules_extracted': len(new_rules),
            'rules_pruned': pruned,
            'total_episodes_analyzed': len(episodes),
            'self_description': self_desc,
        }

        logger.info(f"Consolidation complete: {result}")

        return result

    def get_recent_reflections(self, limit: int = 10) -> List[Dict]:
        """Get recent reflections."""
        return [r.to_dict() for r in self._reflections[-limit:]]

    def introspect(self) -> str:
        """Generate introspective report on reflection quality."""
        if not self._reflections:
            return "No reflections yet."

        daily_reflections = [r for r in self._reflections if r.scope == "daily"]
        episode_reflections = [r for r in self._reflections if r.scope == "episode"]

        total_lessons = sum(len(r.lessons) for r in self._reflections)
        total_actions = sum(len(r.action_items) for r in self._reflections)

        lines = [
            "=== Reflection Engine Introspection ===",
            "",
            f"Total reflections: {len(self._reflections)}",
            f"  - Episode reflections: {len(episode_reflections)}",
            f"  - Daily reflections: {len(daily_reflections)}",
            f"Total lessons extracted: {total_lessons}",
            f"Total action items generated: {total_actions}",
        ]

        return "\n".join(lines)
