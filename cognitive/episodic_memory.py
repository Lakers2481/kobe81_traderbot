"""
Episodic Memory - Trade Episode Storage
=========================================

Stores complete trade episodes with full context for learning.

Based on Reflexion paper (arXiv):
- Context -> Reasoning -> Action -> Outcome -> Postmortem
- Each episode is a complete learning unit
- Enables reflection and pattern extraction

Features:
- Full episode capture (what we saw, thought, did, got)
- Context reconstruction for replay
- Pattern matching across episodes
- Foundation for semantic memory extraction

Usage:
    from cognitive.episodic_memory import get_episodic_memory

    memory = get_episodic_memory()

    # Record an episode
    episode_id = memory.start_episode(context)
    memory.add_reasoning(episode_id, "RSI oversold + above SMA200")
    memory.add_action(episode_id, {'type': 'buy', 'symbol': 'AAPL', 'shares': 100})
    memory.complete_episode(episode_id, outcome={'pnl': 500, 'won': True})

    # Query episodes
    similar = memory.find_similar(current_context)
    lessons = memory.get_lessons_for_context(current_context)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class EpisodeOutcome(Enum):
    """Possible episode outcomes."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    STAND_DOWN = "stand_down"
    ERROR = "error"
    PENDING = "pending"


@dataclass
class Episode:
    """A complete trading episode."""
    episode_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None

    # CONTEXT: What did we see?
    market_context: Dict[str, Any] = field(default_factory=dict)
    signal_context: Dict[str, Any] = field(default_factory=dict)
    portfolio_context: Dict[str, Any] = field(default_factory=dict)

    # REASONING: What did we think?
    reasoning_trace: List[str] = field(default_factory=list)
    confidence_levels: Dict[str, float] = field(default_factory=dict)
    alternatives_considered: List[str] = field(default_factory=list)
    concerns_noted: List[str] = field(default_factory=list)

    # ACTION: What did we do?
    action_taken: Optional[Dict[str, Any]] = None
    decision_mode: str = "unknown"  # fast, slow, hybrid, stand_down

    # OUTCOME: What happened?
    outcome: EpisodeOutcome = EpisodeOutcome.PENDING
    pnl: float = 0.0
    r_multiple: float = 0.0
    outcome_details: Dict[str, Any] = field(default_factory=dict)

    # POSTMORTEM: What did we learn?
    postmortem: str = ""
    lessons_learned: List[str] = field(default_factory=list)
    mistakes_made: List[str] = field(default_factory=list)
    what_to_repeat: List[str] = field(default_factory=list)
    what_to_avoid: List[str] = field(default_factory=list)

    # Metadata
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5  # 0-1, how important for future learning

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['started_at'] = self.started_at.isoformat()
        d['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        d['outcome'] = self.outcome.value
        return d

    @staticmethod
    def from_dict(d: Dict) -> 'Episode':
        return Episode(
            episode_id=d['episode_id'],
            started_at=datetime.fromisoformat(d['started_at']),
            completed_at=datetime.fromisoformat(d['completed_at']) if d.get('completed_at') else None,
            market_context=d.get('market_context', {}),
            signal_context=d.get('signal_context', {}),
            portfolio_context=d.get('portfolio_context', {}),
            reasoning_trace=d.get('reasoning_trace', []),
            confidence_levels=d.get('confidence_levels', {}),
            alternatives_considered=d.get('alternatives_considered', []),
            concerns_noted=d.get('concerns_noted', []),
            action_taken=d.get('action_taken'),
            decision_mode=d.get('decision_mode', 'unknown'),
            outcome=EpisodeOutcome(d.get('outcome', 'pending')),
            pnl=d.get('pnl', 0.0),
            r_multiple=d.get('r_multiple', 0.0),
            outcome_details=d.get('outcome_details', {}),
            postmortem=d.get('postmortem', ''),
            lessons_learned=d.get('lessons_learned', []),
            mistakes_made=d.get('mistakes_made', []),
            what_to_repeat=d.get('what_to_repeat', []),
            what_to_avoid=d.get('what_to_avoid', []),
            tags=d.get('tags', []),
            importance=d.get('importance', 0.5),
        )

    def context_signature(self) -> str:
        """Generate signature for context matching."""
        key_elements = [
            self.market_context.get('regime', ''),
            self.signal_context.get('strategy', ''),
            self.signal_context.get('side', ''),
        ]
        return hashlib.md5('|'.join(str(e) for e in key_elements).encode()).hexdigest()[:8]


class EpisodicMemory:
    """
    Long-term storage for trade episodes.

    Each episode captures the complete decision cycle:
    Context -> Reasoning -> Action -> Outcome -> Postmortem
    """

    def __init__(
        self,
        storage_dir: str = "state/cognitive/episodes",
        max_episodes: int = 1000,
        auto_persist: bool = True,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_episodes = max_episodes
        self.auto_persist = auto_persist

        # In-memory cache
        self._episodes: Dict[str, Episode] = {}
        self._active_episodes: Dict[str, Episode] = {}  # Not yet completed

        # Index by context signature for fast lookup
        self._context_index: Dict[str, List[str]] = {}

        # Load existing episodes
        self._load_episodes()

        logger.info(f"EpisodicMemory initialized with {len(self._episodes)} episodes")

    def start_episode(
        self,
        market_context: Dict[str, Any],
        signal_context: Dict[str, Any],
        portfolio_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start a new episode.

        Args:
            market_context: Current market state (regime, volatility, etc.)
            signal_context: The signal being evaluated
            portfolio_context: Current portfolio state

        Returns:
            episode_id for tracking
        """
        episode_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + \
                     hashlib.md5(str(signal_context).encode()).hexdigest()[:6]

        episode = Episode(
            episode_id=episode_id,
            started_at=datetime.now(),
            market_context=market_context or {},
            signal_context=signal_context or {},
            portfolio_context=portfolio_context or {},
        )

        self._active_episodes[episode_id] = episode
        logger.debug(f"Started episode {episode_id}")

        return episode_id

    def add_reasoning(
        self,
        episode_id: str,
        reasoning: str,
        confidence: Optional[float] = None,
        label: Optional[str] = None,
    ) -> None:
        """Add reasoning step to an episode."""
        episode = self._active_episodes.get(episode_id)
        if not episode:
            logger.warning(f"Episode {episode_id} not found")
            return

        episode.reasoning_trace.append(reasoning)
        if confidence is not None and label:
            episode.confidence_levels[label] = confidence

    def add_concern(self, episode_id: str, concern: str) -> None:
        """Add a concern noted during reasoning."""
        episode = self._active_episodes.get(episode_id)
        if episode:
            episode.concerns_noted.append(concern)

    def add_alternative(self, episode_id: str, alternative: str) -> None:
        """Add an alternative that was considered."""
        episode = self._active_episodes.get(episode_id)
        if episode:
            episode.alternatives_considered.append(alternative)

    def add_action(
        self,
        episode_id: str,
        action: Dict[str, Any],
        decision_mode: str = "unknown",
    ) -> None:
        """Record the action taken."""
        episode = self._active_episodes.get(episode_id)
        if episode:
            episode.action_taken = action
            episode.decision_mode = decision_mode

    def complete_episode(
        self,
        episode_id: str,
        outcome: Dict[str, Any],
        postmortem: Optional[str] = None,
        lessons: Optional[List[str]] = None,
        mistakes: Optional[List[str]] = None,
    ) -> None:
        """
        Complete an episode with its outcome.

        Args:
            episode_id: Episode to complete
            outcome: Dict with 'won', 'pnl', 'r_multiple', etc.
            postmortem: Optional reflection text
            lessons: Optional list of lessons learned
            mistakes: Optional list of mistakes made
        """
        episode = self._active_episodes.pop(episode_id, None)
        if not episode:
            logger.warning(f"Episode {episode_id} not found in active episodes")
            return

        episode.completed_at = datetime.now()

        # Parse outcome
        if outcome.get('won') is True or outcome.get('pnl', 0) > 0:
            episode.outcome = EpisodeOutcome.WIN
        elif outcome.get('won') is False or outcome.get('pnl', 0) < 0:
            episode.outcome = EpisodeOutcome.LOSS
        elif outcome.get('stand_down'):
            episode.outcome = EpisodeOutcome.STAND_DOWN
        else:
            episode.outcome = EpisodeOutcome.BREAKEVEN

        episode.pnl = outcome.get('pnl', 0.0)
        episode.r_multiple = outcome.get('r_multiple', 0.0)
        episode.outcome_details = outcome

        # Postmortem
        if postmortem:
            episode.postmortem = postmortem
        if lessons:
            episode.lessons_learned = lessons
        if mistakes:
            episode.mistakes_made = mistakes

        # Calculate importance
        episode.importance = self._calculate_importance(episode)

        # Store
        self._episodes[episode_id] = episode

        # Index
        sig = episode.context_signature()
        if sig not in self._context_index:
            self._context_index[sig] = []
        self._context_index[sig].append(episode_id)

        # Prune if needed
        self._prune_if_needed()

        # Persist
        if self.auto_persist:
            self._save_episode(episode)

        logger.info(
            f"Completed episode {episode_id}: {episode.outcome.value} "
            f"(PnL: ${episode.pnl:.2f})"
        )

    def _calculate_importance(self, episode: Episode) -> float:
        """Calculate importance score for an episode."""
        importance = 0.5  # Base

        # Significant PnL (positive or negative)
        if abs(episode.pnl) > 500:
            importance += 0.2
        if abs(episode.r_multiple) > 2:
            importance += 0.1

        # Learning content
        if episode.lessons_learned:
            importance += 0.1
        if episode.mistakes_made:
            importance += 0.1

        # Unusual outcome
        if episode.outcome == EpisodeOutcome.STAND_DOWN:
            importance += 0.05  # Stand-downs are interesting

        return min(1.0, importance)

    def find_similar(
        self,
        context: Dict[str, Any],
        limit: int = 5,
    ) -> List[Episode]:
        """
        Find episodes with similar context.

        Args:
            context: Context to match (should have regime, strategy, etc.)
            limit: Maximum episodes to return

        Returns:
            List of similar episodes, most recent first
        """
        # Create signature from context
        key_elements = [
            context.get('regime', ''),
            context.get('strategy', ''),
            context.get('side', ''),
        ]
        sig = hashlib.md5('|'.join(str(e) for e in key_elements).encode()).hexdigest()[:8]

        # Find matching episodes
        episode_ids = self._context_index.get(sig, [])

        episodes = [self._episodes[eid] for eid in episode_ids if eid in self._episodes]

        # Sort by recency and importance
        episodes.sort(key=lambda e: (e.importance, e.started_at), reverse=True)

        return episodes[:limit]

    def get_lessons_for_context(
        self,
        context: Dict[str, Any],
        outcome_filter: Optional[EpisodeOutcome] = None,
    ) -> List[str]:
        """
        Get all lessons learned from similar contexts.

        Args:
            context: Context to match
            outcome_filter: Optional filter by outcome type

        Returns:
            List of lesson strings
        """
        episodes = self.find_similar(context, limit=20)

        if outcome_filter:
            episodes = [e for e in episodes if e.outcome == outcome_filter]

        lessons = []
        for ep in episodes:
            lessons.extend(ep.lessons_learned)
            if ep.outcome == EpisodeOutcome.LOSS:
                lessons.extend([f"AVOID: {m}" for m in ep.mistakes_made])
            elif ep.outcome == EpisodeOutcome.WIN:
                lessons.extend([f"REPEAT: {w}" for w in ep.what_to_repeat])

        # Deduplicate while preserving order
        seen = set()
        unique_lessons = []
        for lesson in lessons:
            if lesson not in seen:
                seen.add(lesson)
                unique_lessons.append(lesson)

        return unique_lessons

    def get_win_rate_for_context(
        self,
        context: Dict[str, Any],
    ) -> Tuple[float, int]:
        """
        Get historical win rate for similar context.

        Returns:
            Tuple of (win_rate, sample_count)
        """
        episodes = self.find_similar(context, limit=50)

        if not episodes:
            return 0.0, 0

        wins = sum(1 for e in episodes if e.outcome == EpisodeOutcome.WIN)
        total = len([e for e in episodes if e.outcome in [EpisodeOutcome.WIN, EpisodeOutcome.LOSS]])

        if total == 0:
            return 0.0, 0

        return wins / total, total

    def get_recent_episodes(self, limit: int = 10) -> List[Episode]:
        """Get most recent episodes."""
        episodes = list(self._episodes.values())
        episodes.sort(key=lambda e: e.started_at, reverse=True)
        return episodes[:limit]

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode by ID."""
        return self._episodes.get(episode_id)

    def add_postmortem(
        self,
        episode_id: str,
        postmortem: str,
        lessons: Optional[List[str]] = None,
        mistakes: Optional[List[str]] = None,
        what_to_repeat: Optional[List[str]] = None,
        what_to_avoid: Optional[List[str]] = None,
    ) -> None:
        """Add or update postmortem for a completed episode."""
        episode = self._episodes.get(episode_id)
        if not episode:
            logger.warning(f"Episode {episode_id} not found")
            return

        episode.postmortem = postmortem
        if lessons:
            episode.lessons_learned.extend(lessons)
        if mistakes:
            episode.mistakes_made.extend(mistakes)
        if what_to_repeat:
            episode.what_to_repeat.extend(what_to_repeat)
        if what_to_avoid:
            episode.what_to_avoid.extend(what_to_avoid)

        if self.auto_persist:
            self._save_episode(episode)

    def tag_episode(self, episode_id: str, tags: List[str]) -> None:
        """Add tags to an episode."""
        episode = self._episodes.get(episode_id)
        if episode:
            episode.tags.extend(tags)
            if self.auto_persist:
                self._save_episode(episode)

    def search_by_tag(self, tag: str) -> List[Episode]:
        """Find episodes by tag."""
        return [e for e in self._episodes.values() if tag in e.tags]

    def get_stats(self) -> Dict[str, Any]:
        """Get episodic memory statistics."""
        if not self._episodes:
            return {'total_episodes': 0}

        outcomes = [e.outcome for e in self._episodes.values()]
        wins = outcomes.count(EpisodeOutcome.WIN)
        losses = outcomes.count(EpisodeOutcome.LOSS)

        return {
            'total_episodes': len(self._episodes),
            'active_episodes': len(self._active_episodes),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
            'stand_downs': outcomes.count(EpisodeOutcome.STAND_DOWN),
            'total_lessons': sum(len(e.lessons_learned) for e in self._episodes.values()),
            'total_mistakes': sum(len(e.mistakes_made) for e in self._episodes.values()),
        }

    def _prune_if_needed(self) -> None:
        """Prune old episodes if over limit."""
        if len(self._episodes) <= self.max_episodes:
            return

        # Sort by importance and age
        episodes = list(self._episodes.values())
        episodes.sort(key=lambda e: (e.importance, e.started_at))

        # Remove lowest importance, oldest episodes
        to_remove = len(self._episodes) - self.max_episodes
        for ep in episodes[:to_remove]:
            del self._episodes[ep.episode_id]
            # Clean up file
            ep_file = self.storage_dir / f"{ep.episode_id}.json"
            if ep_file.exists():
                ep_file.unlink()

        logger.info(f"Pruned {to_remove} old episodes")

    def _save_episode(self, episode: Episode) -> None:
        """Save episode to disk."""
        ep_file = self.storage_dir / f"{episode.episode_id}.json"
        with open(ep_file, 'w') as f:
            json.dump(episode.to_dict(), f, indent=2)

    def _load_episodes(self) -> None:
        """Load episodes from disk."""
        for ep_file in self.storage_dir.glob("*.json"):
            try:
                with open(ep_file, 'r') as f:
                    data = json.load(f)
                episode = Episode.from_dict(data)
                self._episodes[episode.episode_id] = episode

                # Rebuild index
                sig = episode.context_signature()
                if sig not in self._context_index:
                    self._context_index[sig] = []
                self._context_index[sig].append(episode.episode_id)

            except Exception as e:
                logger.warning(f"Failed to load episode {ep_file}: {e}")


# Singleton
_episodic_memory: Optional[EpisodicMemory] = None


def get_episodic_memory() -> EpisodicMemory:
    """Get or create episodic memory singleton."""
    global _episodic_memory
    if _episodic_memory is None:
        _episodic_memory = EpisodicMemory()
    return _episodic_memory
