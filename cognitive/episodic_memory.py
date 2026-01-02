"""
Episodic Memory - The AI's Experiential Journal
================================================

This module provides the cognitive architecture with a detailed memory of its
past experiences. It stores every trading decision as a complete "Episode,"
capturing the full context from the initial signal to the final outcome and
any subsequent reflections.

This is the foundation for all higher-level learning. By recording not just
*what* happened, but also *what the agent was thinking*, it enables sophisticated
reflection and adaptation, inspired by cognitive science models like the
Reflexion paper (Shinn et al., arXiv).

Core Principles:
- **Complete Learning Unit:** Each episode contains the full sequence:
  Context -> Reasoning -> Action -> Outcome -> Postmortem.
- **Context is Key:** Episodes are indexed by their context, allowing the agent
  to ask, "What happened the last time I was in a situation like this?"
- **Foundation for Abstraction:** The raw data in episodic memory is the source
  material from which the ReflectionEngine extracts general rules for
  SemanticMemory.

Usage:
    from cognitive.episodic_memory import get_episodic_memory

    memory = get_episodic_memory()

    # 1. Start a new memory when a decision process begins.
    episode_id = memory.start_episode(market_context, signal_context)

    # 2. Record the "thought process" during deliberation.
    memory.add_reasoning(episode_id, "RSI is oversold and market is in a bullish regime.")
    memory.add_action(episode_id, {'type': 'buy', 'symbol': 'AAPL', 'shares': 100})

    # 3. Complete the memory with the final outcome.
    trade_outcome = {'pnl': 500.0, 'won': True}
    memory.complete_episode(episode_id, outcome=trade_outcome)

    # The memory can now be queried by other cognitive components.
    similar_past_trades = memory.find_similar(current_context)
    lessons_from_history = memory.get_lessons_for_context(current_context)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import hashlib

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert non-JSON-serializable objects to JSON-safe types.
    Handles pandas Timestamps, numpy types, datetimes, etc.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


class EpisodeOutcome(Enum):
    """Enumerates the possible outcomes of a trading episode."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    STAND_DOWN = "stand_down"  # The agent decided not to act.
    ERROR = "error"          # The episode resulted in an error.
    PENDING = "pending"        # The outcome is not yet known.


@dataclass
class Episode:
    """
    A dataclass representing a single, complete trading experience.
    It captures the entire decision-making process from start to finish.
    """
    episode_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None

    # --- Section 1: CONTEXT ---
    # What was the state of the world when the decision was made?
    market_context: Dict[str, Any] = field(default_factory=dict)
    signal_context: Dict[str, Any] = field(default_factory=dict)
    portfolio_context: Dict[str, Any] = field(default_factory=dict)

    # --- Section 2: REASONING ---
    # What was the agent's "thought process"?
    reasoning_trace: List[str] = field(default_factory=list)
    confidence_levels: Dict[str, float] = field(default_factory=dict)
    alternatives_considered: List[str] = field(default_factory=list)
    concerns_noted: List[str] = field(default_factory=list)

    # --- Section 3: ACTION ---
    # What did the agent ultimately decide to do?
    action_taken: Optional[Dict[str, Any]] = None
    decision_mode: str = "unknown"  # e.g., 'fast', 'slow', 'hybrid'

    # --- Section 4: OUTCOME ---
    # What was the result of the action?
    outcome: EpisodeOutcome = EpisodeOutcome.PENDING
    pnl: float = 0.0
    r_multiple: float = 0.0
    outcome_details: Dict[str, Any] = field(default_factory=dict)

    # --- Section 5: POSTMORTEM / REFLECTION ---
    # What did the agent learn from this experience?
    postmortem: str = ""
    lessons_learned: List[str] = field(default_factory=list)
    mistakes_made: List[str] = field(default_factory=list)
    what_to_repeat: List[str] = field(default_factory=list)
    what_to_avoid: List[str] = field(default_factory=list)

    # --- Metadata ---
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5  # Subjective score (0-1) of how valuable this memory is for future learning.
    metadata: Dict[str, Any] = field(default_factory=dict)  # Generic metadata storage

    # --- Simulation Fields (Task B2) ---
    is_simulated: bool = False  # True if this episode was generated synthetically
    simulation_source: Optional[str] = None  # e.g., 'counterfactual', 'hypothesis_test', 'stress_test'
    simulation_params: Dict[str, Any] = field(default_factory=dict)  # Parameters used to generate

    def to_dict(self) -> Dict:
        """Serializes the Episode to a dictionary for storage."""
        d = asdict(self)
        # Convert datetime and enum objects to JSON-serializable formats.
        d['started_at'] = self.started_at.isoformat()
        d['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        d['outcome'] = self.outcome.value
        return d

    @staticmethod
    def from_dict(d: Dict) -> 'Episode':
        """Creates an Episode object from a dictionary."""
        # This is a bit manual to handle the nested and typed fields correctly.
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
        """
        Generates a normalized hash signature for the episode's context.
        Uses upper-case and consistent defaults for reproducible lookups.
        Includes VIX band for volatility stratification (must match normalize_context_signature).
        """
        regime = str(self.market_context.get('regime', 'unknown')).upper()
        strategy = str(self.signal_context.get('strategy', 'unknown')).upper()
        side = str(self.signal_context.get('side', 'long')).upper()

        # Add VIX band for volatility stratification (must match normalize_context_signature)
        vix = self.market_context.get('vix', self.market_context.get('vix_level', 20))
        try:
            vix = float(vix)
        except (TypeError, ValueError):
            vix = 20.0
        if vix < 15:
            vix_band = 'LOW'
        elif vix < 25:
            vix_band = 'MED'
        elif vix < 35:
            vix_band = 'HIGH'
        else:
            vix_band = 'EXTREME'

        key_elements = [regime, strategy, side, vix_band]
        return hashlib.md5('|'.join(key_elements).encode()).hexdigest()[:8]


class EpisodicMemory:
    """
    Manages the storage, retrieval, and indexing of all trading `Episode` objects.
    It acts as the AI's long-term memory, persisting episodes to disk.
    """

    def __init__(
        self,
        storage_dir: str = "state/cognitive/episodes",
        max_episodes: int = 1000,
        auto_persist: bool = True,
    ):
        """
        Initializes the EpisodicMemory.

        Args:
            storage_dir: The directory where individual episode files will be saved.
            max_episodes: The maximum number of episodes to keep in memory/disk
                          before pruning the least important ones.
            auto_persist: If True, saves each episode to disk as it's completed.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_episodes = max_episodes
        self.auto_persist = auto_persist

        # In-memory cache for fast access.
        self._episodes: Dict[str, Episode] = {}
        # Episodes that have started but not yet completed.
        self._active_episodes: Dict[str, Episode] = {}

        # An index mapping context signatures to lists of episode IDs. This is the
        # key to efficiently finding similar past experiences.
        self._context_index: Dict[str, List[str]] = {}

        self._load_episodes()
        logger.info(f"EpisodicMemory initialized, loaded {len(self._episodes)} episodes from disk.")

    def start_episode(
        self,
        market_context: Dict[str, Any],
        signal_context: Dict[str, Any],
        portfolio_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Begins a new trading episode and returns its unique ID.

        Args:
            market_context: The state of the market (e.g., regime, volatility).
            signal_context: The trading signal being evaluated.
            portfolio_context: The current state of the portfolio.

        Returns:
            A unique episode_id string for tracking this decision process.
        """
        # Sanitize contexts for JSON serialization (handles Timestamps, numpy types, etc.)
        safe_signal_ctx = _sanitize_for_json(signal_context or {})
        safe_market_ctx = _sanitize_for_json(market_context or {})
        safe_portfolio_ctx = _sanitize_for_json(portfolio_context or {})

        # Create a unique but deterministic-ish ID.
        episode_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S_") +
            hashlib.md5(json.dumps(safe_signal_ctx, sort_keys=True).encode()).hexdigest()[:6]
        )

        episode = Episode(
            episode_id=episode_id,
            started_at=datetime.now(),
            market_context=safe_market_ctx,
            signal_context=safe_signal_ctx,
            portfolio_context=safe_portfolio_ctx,
        )

        self._active_episodes[episode_id] = episode
        logger.debug(f"Started new episode {episode_id}")
        return episode_id

    def add_reasoning(self, episode_id: str, reasoning: Any) -> None:
        """Adds reasoning step(s) to the trace of an active episode.

        Args:
            episode_id: The ID of the episode.
            reasoning: A single reasoning step (str) or a list of steps.
        """
        if episode := self._active_episodes.get(episode_id):
            if isinstance(reasoning, list):
                episode.reasoning_trace.extend(reasoning)
            else:
                episode.reasoning_trace.append(reasoning)

    def add_concern(self, episode_id: str, concern: str) -> None:
        """Adds a concern (a negative factor) noted during deliberation."""
        if episode := self._active_episodes.get(episode_id):
            episode.concerns_noted.append(concern)

    def add_concerns(self, episode_id: str, concerns: List[str]) -> None:
        """Adds multiple concerns (negative factors) noted during deliberation."""
        for concern in concerns:
            self.add_concern(episode_id, concern)

    def add_action(self, episode_id: str, action: Dict[str, Any], decision_mode: str = "unknown") -> None:
        """Records the final action taken in an active episode."""
        if episode := self._active_episodes.get(episode_id):
            episode.action_taken = action
            episode.decision_mode = decision_mode

    def complete_episode(self, episode_id: str, outcome: Dict[str, Any]) -> None:
        """
        Finalizes an episode with its outcome, moving it from "active" to
        "completed" memory.

        Args:
            episode_id: The ID of the episode to complete.
            outcome: A dictionary describing the result (e.g., {'pnl': 100, 'won': True}).
        """
        episode = self._active_episodes.pop(episode_id, None)
        if not episode:
            logger.warning(f"Attempted to complete non-active or unknown episode {episode_id}")
            return

        episode.completed_at = datetime.now()

        # --- Parse and store the outcome ---
        if outcome.get('stand_down'):
            episode.outcome = EpisodeOutcome.STAND_DOWN
        elif outcome.get('won') is True or outcome.get('pnl', 0) > 0:
            episode.outcome = EpisodeOutcome.WIN
        elif outcome.get('won') is False or outcome.get('pnl', 0) < 0:
            episode.outcome = EpisodeOutcome.LOSS
        else:
            episode.outcome = EpisodeOutcome.BREAKEVEN

        episode.pnl = outcome.get('pnl', 0.0)
        episode.r_multiple = outcome.get('r_multiple', 0.0)
        episode.outcome_details = outcome

        # --- Finalize and store ---
        episode.importance = self._calculate_importance(episode)
        self._episodes[episode_id] = episode

        # Add the completed episode to the context index for future lookups.
        sig = episode.context_signature()
        self._context_index.setdefault(sig, []).append(episode_id)

        self._prune_if_needed()

        if self.auto_persist:
            self._save_episode(episode)

        logger.info(
            f"Completed and archived episode {episode_id}: {episode.outcome.value} "
            f"(PnL: ${episode.pnl:.2f}, Importance: {episode.importance:.2f})"
        )

    def _calculate_importance(self, episode: Episode) -> float:
        """Calculates a score for how "interesting" or "important" an episode is."""
        importance = 0.5  # Baseline importance

        # Large wins or losses are more significant.
        if abs(episode.r_multiple) > 2.0:
            importance += 0.3
        elif abs(episode.pnl) > 500: # Fallback if R-multiple isn't available
            importance += 0.2

        # Episodes that contain explicit lessons are more valuable.
        if episode.lessons_learned or episode.mistakes_made:
            importance += 0.2
        
        # Surprising outcomes (e.g. a loss in a high-confidence trade) could be important.
        # (This logic could be added here).

        return min(1.0, importance)

    def find_similar(self, context: Dict[str, Any], limit: int = 5) -> List[Episode]:
        """
        Finds past episodes that occurred in a similar context. This is a core
        function for learning and decision-making.

        Args:
            context: A dictionary describing the current context to match.
            limit: The maximum number of similar episodes to return.

        Returns:
            A list of matching Episode objects, sorted by importance and recency.
        """
        # 1. Generate the normalized context signature.
        sig = self.normalize_context_signature(context)

        # 2. Use the index to quickly find all episodes with the same signature.
        episode_ids = self._context_index.get(sig, [])
        if not episode_ids:
            return []

        # 3. Retrieve the full episode objects.
        episodes = [self._episodes[eid] for eid in episode_ids if eid in self._episodes]

        # 4. Sort them to return the most relevant ones first.
        episodes.sort(key=lambda e: (e.importance, e.started_at), reverse=True)

        return episodes[:limit]

    def get_lessons_for_context(
        self,
        context: Dict[str, Any],
        outcome_filter: Optional[EpisodeOutcome] = None,
    ) -> List[str]:
        """
        Aggregates all explicit "lessons learned" from past episodes in a
        similar context.

        Args:
            context: The context to match against.
            outcome_filter: Optional, to get lessons only from wins or losses.

        Returns:
            A deduplicated list of lesson strings.
        """
        similar_episodes = self.find_similar(context, limit=20)

        if outcome_filter:
            similar_episodes = [e for e in similar_episodes if e.outcome == outcome_filter]

        lessons = []
        for ep in similar_episodes:
            lessons.extend(ep.lessons_learned)
            # Frame mistakes and successes as actionable advice.
            if ep.outcome == EpisodeOutcome.LOSS:
                lessons.extend([f"AVOID: {m}" for m in ep.mistakes_made])
            elif ep.outcome == EpisodeOutcome.WIN:
                lessons.extend([f"REPEAT: {w}" for w in ep.what_to_repeat])

        # Return a unique list of lessons.
        return list(dict.fromkeys(lessons))

    @staticmethod
    def normalize_context_signature(context: Dict[str, Any]) -> str:
        """
        Creates a normalized context signature for consistent lookups.
        All fields are upper-cased and joined with '|'.
        Includes VIX band for volatility stratification.

        Args:
            context: Dict with 'regime', 'strategy', 'side', and optionally 'vix' keys.

        Returns:
            A consistent MD5 hash signature (8 chars).
        """
        regime = str(context.get('regime', 'unknown')).upper()
        strategy = str(context.get('strategy', 'unknown')).upper()
        side = str(context.get('side', 'long')).upper()

        # Add VIX band for volatility stratification
        vix = context.get('vix', context.get('vix_level', 20))
        try:
            vix = float(vix)
        except (TypeError, ValueError):
            vix = 20.0
        if vix < 15:
            vix_band = 'LOW'
        elif vix < 25:
            vix_band = 'MED'
        elif vix < 35:
            vix_band = 'HIGH'
        else:
            vix_band = 'EXTREME'

        key_elements = [regime, strategy, side, vix_band]
        return hashlib.md5('|'.join(key_elements).encode()).hexdigest()[:8]

    def get_stats_for_signature(self, sig: str) -> Dict[str, Any]:
        """
        Returns stats for a specific context signature.

        Args:
            sig: An 8-character MD5 context signature.

        Returns:
            Dict with 'n' (sample size) and 'win_rate'.
        """
        episode_ids = self._context_index.get(sig, [])
        if not episode_ids:
            return {'n': 0, 'win_rate': 0.0}

        episodes = [self._episodes[eid] for eid in episode_ids if eid in self._episodes]

        wins = sum(1 for e in episodes if e.outcome == EpisodeOutcome.WIN)
        losses = sum(1 for e in episodes if e.outcome == EpisodeOutcome.LOSS)
        total = wins + losses

        return {
            'n': total,
            'win_rate': (wins / total) if total > 0 else 0.0
        }

    def get_win_rate_for_context(self, context: Dict[str, Any]) -> Tuple[float, int]:
        """
        Calculates the historical win rate for a given context.

        Returns:
            A tuple of (win_rate, sample_size).
        """
        episodes = self.find_similar(context, limit=100) # Use a larger sample for stats
        if not episodes:
            return 0.0, 0

        wins = sum(1 for e in episodes if e.outcome == EpisodeOutcome.WIN)
        losses = sum(1 for e in episodes if e.outcome == EpisodeOutcome.LOSS)
        total = wins + losses

        return (wins / total, total) if total > 0 else (0.0, 0)

    def get_recent_episodes(self, limit: int = 10) -> List[Episode]:
        """Returns the most recently completed episodes."""
        episodes = sorted(self._episodes.values(), key=lambda e: e.started_at, reverse=True)
        return episodes[:limit]

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieves a single episode by its unique ID."""
        return self._episodes.get(episode_id)

    def add_postmortem(
        self,
        episode_id: str,
        postmortem: str,
        lessons: List[str] = [],
        mistakes: List[str] = [],
        what_to_repeat: List[str] = [],
        what_to_avoid: List[str] = [],
    ) -> None:
        """Adds reflection data to an already completed episode."""
        episode = self._episodes.get(episode_id)
        if not episode:
            logger.warning(f"Cannot add postmortem, episode {episode_id} not found.")
            return

        episode.postmortem = postmortem
        episode.lessons_learned.extend(lessons)
        episode.mistakes_made.extend(mistakes)
        episode.what_to_repeat.extend(what_to_repeat)
        episode.what_to_avoid.extend(what_to_avoid)

        # Recalculate importance since it now has more learning content.
        episode.importance = self._calculate_importance(episode)

        if self.auto_persist:
            self._save_episode(episode)
        logger.info(f"Added postmortem to episode {episode_id}.")

    def get_stats(self) -> Dict[str, Any]:
        """Returns high-level statistics about the memory's contents."""
        if not self._episodes:
            return {'total_episodes': 0, 'active_episodes': len(self._active_episodes)}

        outcomes = [e.outcome for e in self._episodes.values()]
        wins = outcomes.count(EpisodeOutcome.WIN)
        losses = outcomes.count(EpisodeOutcome.LOSS)
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        return {
            'total_episodes': len(self._episodes),
            'active_episodes': len(self._active_episodes),
            'win_rate': f"{win_rate:.2%}",
            'wins': wins,
            'losses': losses,
            'stand_downs': outcomes.count(EpisodeOutcome.STAND_DOWN),
            'total_lessons': sum(len(e.lessons_learned) for e in self._episodes.values()),
        }

    def _prune_if_needed(self) -> None:
        """If the number of episodes exceeds the max limit, remove the least important ones."""
        if len(self._episodes) <= self.max_episodes:
            return

        # Sort episodes by importance and then by age, ascending.
        episodes_to_prune = sorted(self._episodes.values(), key=lambda e: (e.importance, e.started_at))

        num_to_remove = len(self._episodes) - self.max_episodes
        for ep in episodes_to_prune[:num_to_remove]:
            # Remove from memory
            del self._episodes[ep.episode_id]

            # Remove from index
            sig = ep.context_signature()
            if sig in self._context_index and ep.episode_id in self._context_index[sig]:
                self._context_index[sig].remove(ep.episode_id)

            # Delete the file from disk.
            ep_file = self.storage_dir / f"{ep.episode_id}.json"
            if ep_file.exists():
                ep_file.unlink()

        logger.info(f"Pruned {num_to_remove} least important episodes to maintain memory size.")

    def _save_episode(self, episode: Episode) -> None:
        """Saves a single episode to a JSON file."""
        try:
            ep_file = self.storage_dir / f"{episode.episode_id}.json"
            with open(ep_file, 'w') as f:
                json.dump(episode.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save episode {episode.episode_id}: {e}")

    def _load_episodes(self) -> None:
        """Loads all episode files from the storage directory into memory on startup."""
        for ep_file in self.storage_dir.glob("*.json"):
            try:
                with open(ep_file, 'r') as f:
                    data = json.load(f)
                episode = Episode.from_dict(data)
                self._episodes[episode.episode_id] = episode

                # Rebuild the context index as we load.
                sig = episode.context_signature()
                self._context_index.setdefault(sig, []).append(episode.episode_id)

            except Exception as e:
                logger.warning(f"Failed to load episode from {ep_file}: {e}")

# Singleton pattern to ensure only one instance of EpisodicMemory exists.
_episodic_memory: Optional[EpisodicMemory] = None

def get_episodic_memory() -> EpisodicMemory:
    """Factory function to get the singleton instance of EpisodicMemory."""
    global _episodic_memory
    if _episodic_memory is None:
        _episodic_memory = EpisodicMemory()
    return _episodic_memory
