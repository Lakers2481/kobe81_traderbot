"""
Curiosity Engine - Autonomous Pattern Discovery
=================================================

This module provides the cognitive architecture with the ability to "get smarter"
over time through autonomous hypothesis generation and testing. It acts as the
research and development department for the AI, constantly seeking new,
quantifiable trading edges.

Core Workflow:
1.  **Generate Hypotheses:** Proposes new, testable ideas about market behavior
    based on recent trade episodes, market observations, or combinatorial exploration.
2.  **Test Hypotheses:** Runs statistical tests on these hypotheses against
    historical data stored in Episodic Memory.
3.  **Validate Edges:** If a hypothesis is validated with statistical significance,
    it is promoted to an "Edge".
4.  **Inform the Brain:** Validated edges are converted into new rules in
    Semantic Memory, directly influencing the CognitiveBrain's future decisions.

This engine allows the trading bot to adapt to changing market dynamics and
discover novel strategies without human intervention.

Usage:
    from cognitive.curiosity_engine import CuriosityEngine

    engine = CuriosityEngine()

    # Periodically, the system can ask the engine to generate new ideas.
    new_hypotheses = engine.generate_hypotheses()

    # The engine can then test all its pending ideas.
    test_results = engine.test_all_pending()

    # The brain can then query for proven, high-confidence strategies.
    proven_edges = engine.get_validated_edges()
"""

import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class HypothesisStatus(Enum):
    """Enumerates the lifecycle status of a hypothesis."""
    PROPOSED = "proposed"       # Newly generated, not yet tested.
    TESTING = "testing"         # Currently undergoing statistical validation.
    VALIDATED = "validated"     # Statistically significant and proven.
    REJECTED = "rejected"       # Statistically proven to be false.
    INCONCLUSIVE = "inconclusive"  # Not enough data to make a confident conclusion.


@dataclass
class Hypothesis:
    """
    Represents a testable idea about market behavior. It's a question the
    engine poses, e.g., "Does strategy X work well in market regime Y?"
    """
    hypothesis_id: str  # Unique identifier.
    description: str  # Human-readable summary of the idea.
    condition: str  # The context in which the hypothesis applies (e.g., "regime = BULL").
    prediction: str  # The expected outcome (e.g., "win_rate > 0.6").
    rationale: str  # Why the engine thought this was a good idea to test.
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    created_at: datetime = field(default_factory=datetime.now)
    tested_at: Optional[datetime] = None
    # --- Test Results ---
    sample_size: int = 0  # Number of data points used in the test.
    observed_value: float = 0.0  # The actual value measured from the data.
    expected_value: float = 0.0  # The value predicted by the hypothesis.
    p_value: Optional[float] = None  # The statistical significance of the result.
    confidence_interval: Optional[Tuple[float, float]] = None
    source: str = ""  # How the hypothesis was generated (e.g., "episode_pattern").

    def to_dict(self) -> Dict:
        """Serializes the hypothesis to a dictionary."""
        return {
            'hypothesis_id': self.hypothesis_id,
            'description': self.description,
            'condition': self.condition,
            'prediction': self.prediction,
            'rationale': self.rationale,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'tested_at': self.tested_at.isoformat() if self.tested_at else None,
            'sample_size': self.sample_size,
            'observed_value': round(self.observed_value, 4),
            'expected_value': round(self.expected_value, 4),
            'p_value': round(self.p_value, 4) if self.p_value else None,
            'source': self.source,
        }


@dataclass
class Edge:
    """
    Represents a validated, statistically significant trading pattern or advantage.
    An "Edge" is a `Hypothesis` that has been proven true. It is an actionable
    piece of knowledge that can be used to improve trading decisions.
    """
    edge_id: str
    description: str  # Human-readable summary of the edge.
    condition: str  # The specific market condition where this edge applies.
    expected_win_rate: float  # The validated win rate.
    expected_profit_factor: float  # The validated profit factor.
    sample_size: int  # The sample size that validated this edge.
    confidence: float  # Confidence score (0-1) based on statistical significance.
    first_discovered: datetime  # When this edge was first validated.
    last_validated: datetime  # The last time this edge was re-validated.
    times_validated: int = 1  # How many times this edge has been confirmed.
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serializes the edge to a dictionary."""
        return {
            'edge_id': self.edge_id,
            'description': self.description,
            'condition': self.condition,
            'expected_win_rate': round(self.expected_win_rate, 3),
            'expected_profit_factor': round(self.expected_profit_factor, 3),
            'sample_size': self.sample_size,
            'confidence': round(self.confidence, 3),
            'first_discovered': self.first_discovered.isoformat(),
            'last_validated': self.last_validated.isoformat(),
            'times_validated': self.times_validated,
        }


@dataclass
class StrategyIdeaRecord:
    """
    A record of an LLM-generated strategy idea, enriched with metadata
    for tracking and potential implementation.

    Strategy ideas are novel trading concepts proposed by the LLM that
    could be developed into full strategies. The CuriosityEngine tracks
    these and creates corresponding hypotheses to test their viability.
    """
    idea_id: str
    name: str  # Short strategy name
    concept: str  # Core trading concept
    market_context: str  # When this strategy applies
    entry_conditions: List[str]  # Entry conditions
    exit_conditions: List[str]  # Exit conditions
    risk_management: str  # Position sizing and risk rules
    rationale: str  # Why this might work
    status: str = "proposed"  # proposed, under_review, implemented, rejected
    created_at: datetime = field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None
    reviewer_notes: List[str] = field(default_factory=list)
    implementation_priority: int = 0  # 0 = low, 1 = medium, 2 = high
    related_hypothesis_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5

    def to_dict(self) -> Dict:
        """Serializes the record to a dictionary."""
        return {
            'idea_id': self.idea_id,
            'name': self.name,
            'concept': self.concept,
            'market_context': self.market_context,
            'entry_conditions': self.entry_conditions,
            'exit_conditions': self.exit_conditions,
            'risk_management': self.risk_management,
            'rationale': self.rationale,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at else None,
            'reviewer_notes': self.reviewer_notes,
            'implementation_priority': self.implementation_priority,
            'related_hypothesis_ids': self.related_hypothesis_ids,
            'confidence': self.confidence,
        }


class CuriosityEngine:
    """
    Manages the lifecycle of hypothesis generation, testing, and promotion to
    validated edges. This engine persists its state to disk, allowing it to
    build a long-term understanding of the market.
    """

    def __init__(
        self,
        storage_dir: str = "state/cognitive",
        min_sample_size: int = 30,
        significance_level: float = 0.05,
        min_edge_win_rate: float = 0.55,
    ):
        """
        Initializes the CuriosityEngine.

        Args:
            storage_dir: Directory to save and load the engine's state.
            min_sample_size: The minimum number of trades required to test a hypothesis.
            significance_level: The p-value threshold for validating a hypothesis.
            min_edge_win_rate: The minimum win rate for a pattern to be considered an edge.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        self.min_edge_win_rate = min_edge_win_rate

        # In-memory storage for hypotheses and discovered edges.
        self._hypotheses: Dict[str, Hypothesis] = {}
        self._edges: Dict[str, Edge] = {}
        self._strategy_ideas: Dict[str, StrategyIdeaRecord] = {}

        # Dependencies on other cognitive components are lazy-loaded.
        self._episodic_memory = None
        self._semantic_memory = None
        self._workspace = None

        self._load_state()
        logger.info(
            f"CuriosityEngine initialized, loaded {len(self._hypotheses)} hypotheses, "
            f"{len(self._edges)} edges, and {len(self._strategy_ideas)} strategy ideas from state."
        )

    # --- Lazy-loaded properties for dependencies ---
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
    def workspace(self):
        if self._workspace is None:
            from cognitive.global_workspace import get_workspace
            self._workspace = get_workspace()
        return self._workspace

    def generate_hypotheses(
        self,
        observations: Optional[Dict[str, Any]] = None,
    ) -> List[Hypothesis]:
        """
        The main entry point for generating new hypotheses. It combines several
        strategies to create a diverse set of testable ideas.

        Args:
            observations: Optional dictionary of current market observations
                          (e.g., VIX, volume spikes) to inspire new hypotheses.

        Returns:
            A list of newly generated Hypothesis objects.
        """
        new_hypotheses = []

        # Strategy 1: Look for interesting patterns in recent trade history.
        episodes = self.episodic_memory.get_recent_episodes(limit=50)
        new_hypotheses.extend(self._hypotheses_from_episodes(episodes))

        # Strategy 2: Generate ideas based on real-time market observations.
        if observations:
            new_hypotheses.extend(self._hypotheses_from_observations(observations))

        # Strategy 3: Create hypotheses by combining known factors in new ways.
        new_hypotheses.extend(self._combinatorial_hypotheses())

        # Add new, unique hypotheses to the internal collection.
        added_count = 0
        for hyp in new_hypotheses:
            if hyp.hypothesis_id not in self._hypotheses:
                self._hypotheses[hyp.hypothesis_id] = hyp
                logger.info(f"Generated new hypothesis: '{hyp.description}'")
                added_count += 1
        
        if added_count > 0:
            self._save_state()

        return new_hypotheses

    def _hypotheses_from_episodes(self, episodes: List[Any]) -> List[Hypothesis]:
        """Generate hypotheses by finding patterns in recent episodes."""
        from cognitive.episodic_memory import EpisodeOutcome
        hypotheses = []

        # Group episodes by their context (e.g., regime and strategy).
        by_context: Dict[str, List[Any]] = {}
        for ep in episodes:
            key = f"{ep.market_context.get('regime', 'unknown')}|{ep.signal_context.get('strategy', 'unknown')}"
            by_context.setdefault(key, []).append(ep)

        # Analyze each context group for interesting performance.
        for key, context_episodes in by_context.items():
            if len(context_episodes) < 5:  # Need a minimum number of examples.
                continue

            regime, strategy = key.split('|')
            wins = [e for e in context_episodes if e.outcome == EpisodeOutcome.WIN]
            losses = [e for e in context_episodes if e.outcome == EpisodeOutcome.LOSS]
            total_trades = len(wins) + len(losses)

            if total_trades < 5:
                continue

            win_rate = len(wins) / total_trades

            # If a high win rate is observed, create a hypothesis about it.
            if win_rate > 0.65:
                hyp_id = hashlib.md5(f"high_wr_{key}".encode()).hexdigest()[:8]
                if hyp_id not in self._hypotheses:
                    hypotheses.append(Hypothesis(
                        hypothesis_id=hyp_id,
                        description=f"Performance of {strategy} seems high in {regime} regime",
                        condition=f"regime = {regime} AND strategy = {strategy}",
                        prediction="win_rate > 0.6",
                        rationale=f"Observed {win_rate:.1%} win rate over {total_trades} trades.",
                        source="episode_pattern",
                    ))
            # If a low win rate is observed, create a hypothesis to avoid this context.
            elif win_rate < 0.35:
                hyp_id = hashlib.md5(f"low_wr_{key}".encode()).hexdigest()[:8]
                if hyp_id not in self._hypotheses:
                    hypotheses.append(Hypothesis(
                        hypothesis_id=hyp_id,
                        description=f"{strategy} seems to underperform in {regime} regime",
                        condition=f"regime = {regime} AND strategy = {strategy}",
                        prediction="win_rate < 0.4",
                        rationale=f"Observed {win_rate:.1%} win rate over {total_trades} trades.",
                        source="episode_pattern",
                    ))
        return hypotheses

    def _hypotheses_from_observations(self, observations: Dict) -> List[Hypothesis]:
        """Generate hypotheses from current market observations (e.g., high VIX)."""
        hypotheses = []
        # Example: If VIX is high, hypothesize that mean reversion is more effective.
        vix = observations.get('vix', 20)
        if vix > 30:
            hyp_id = hashlib.md5("high_vix_mr_perf".encode()).hexdigest()[:8]
            if hyp_id not in self._hypotheses:
                hypotheses.append(Hypothesis(
                    hypothesis_id=hyp_id,
                    description="Mean reversion strategies might work better when VIX > 30",
                    condition="vix > 30 AND is_mean_reversion",
                    prediction="win_rate > 0.6",
                    rationale=f"High VIX ({vix}) often signals fear and oversold conditions, ideal for bounces.",
                    source="observation",
                ))
        return hypotheses

    def _combinatorial_hypotheses(self) -> List[Hypothesis]:
        """Generate hypotheses by combining known factors in novel ways."""
        hypotheses = []
        import random
        import os

        # REPRODUCIBILITY: Use seeded RNG for deterministic hypothesis generation
        # The seed can be overridden via environment variable KOBE_RANDOM_SEED
        _curiosity_seed = int(os.getenv("KOBE_RANDOM_SEED", "42"))
        _curiosity_rng = random.Random(_curiosity_seed)

        # Define known factors to combine.
        regimes = ['BULL', 'BEAR', 'CHOPPY']
        strategies = ['ibs_rsi', 'turtle_soup']
        vol_conditions = ['vix < 20', 'vix >= 20 AND vix < 30', 'vix >= 30']

        # To avoid a combinatorial explosion, only generate a few random combinations at a time.
        combinations = [(r, s, v) for r in regimes for s in strategies for v in vol_conditions]
        sample = _curiosity_rng.sample(combinations, min(3, len(combinations)))

        for regime, strategy, vol_cond in sample:
            hyp_id = hashlib.md5(f"combo_{regime}_{strategy}_{vol_cond}".encode()).hexdigest()[:8]
            if hyp_id not in self._hypotheses:
                hypotheses.append(Hypothesis(
                    hypothesis_id=hyp_id,
                    description=f"Explore {strategy} in {regime} regime when {vol_cond}",
                    condition=f"regime = {regime} AND strategy = {strategy} AND {vol_cond}",
                    prediction="win_rate > 0.55", # A generic "let's see if this works" prediction
                    rationale="Exploring a new combination of known factors.",
                    source="combinatorial",
                ))
        return hypotheses

    def add_llm_generated_hypotheses(self, llm_hypotheses: List[Any]) -> List[Hypothesis]:
        """
        Add hypotheses generated by LLM analysis to the engine.

        This method accepts LLMHypothesis objects from the LLMNarrativeAnalyzer
        and converts them into proper Hypothesis objects for testing.

        Args:
            llm_hypotheses: List of LLMHypothesis objects from the LLM analyzer.

        Returns:
            List of newly added Hypothesis objects.
        """
        added = []

        for llm_hyp in llm_hypotheses:
            # Create a unique ID based on the description
            hyp_id = hashlib.md5(f"llm_{llm_hyp.description}".encode()).hexdigest()[:8]

            # Skip if we already have this hypothesis
            if hyp_id in self._hypotheses:
                logger.debug(f"LLM hypothesis already exists: '{llm_hyp.description}'")
                continue

            # Create a proper Hypothesis object
            hypothesis = Hypothesis(
                hypothesis_id=hyp_id,
                description=llm_hyp.description,
                condition=llm_hyp.condition,
                prediction=llm_hyp.prediction,
                rationale=f"LLM suggested: {llm_hyp.rationale}",
                source="llm_generated",
            )

            self._hypotheses[hyp_id] = hypothesis
            added.append(hypothesis)
            logger.info(f"Added LLM-generated hypothesis: '{hypothesis.description}'")

        if added:
            self._save_state()
            # Announce new hypotheses on the workspace
            self.workspace.publish(
                topic='insight',
                data={'type': 'llm_hypotheses_added', 'count': len(added)},
                source='curiosity_engine',
            )

        return added

    def add_llm_generated_strategy_ideas(self, ideas: List[Any]) -> List[StrategyIdeaRecord]:
        """
        Add strategy ideas generated by LLM analysis to the engine.

        Strategy ideas are novel trading concepts proposed by the LLM that could
        be developed into full strategies. This method converts them into
        StrategyIdeaRecord objects and creates corresponding hypotheses for testing.

        Args:
            ideas: List of StrategyIdea objects from the LLMNarrativeAnalyzer.

        Returns:
            List of newly added StrategyIdeaRecord objects.
        """
        added = []

        for idea in ideas:
            # Create a unique ID based on the strategy name and concept
            idea_id = hashlib.md5(f"strategy_{idea.name}_{idea.concept}".encode()).hexdigest()[:12]

            # Skip if we already have this strategy idea
            if idea_id in self._strategy_ideas:
                logger.debug(f"Strategy idea already exists: '{idea.name}'")
                continue

            # Create a StrategyIdeaRecord
            record = StrategyIdeaRecord(
                idea_id=idea_id,
                name=idea.name,
                concept=idea.concept,
                market_context=idea.market_context,
                entry_conditions=idea.entry_conditions,
                exit_conditions=idea.exit_conditions,
                risk_management=idea.risk_management,
                rationale=idea.rationale,
                confidence=idea.confidence,
                status="proposed",
            )

            # Create a corresponding hypothesis to test the strategy's viability
            hyp_id = self._create_hypothesis_for_strategy(record)
            if hyp_id:
                record.related_hypothesis_ids.append(hyp_id)

            self._strategy_ideas[idea_id] = record
            added.append(record)
            logger.info(f"Added LLM-generated strategy idea: '{record.name}' (concept: {record.concept[:50]}...)")

        if added:
            self._save_state()
            # Announce new strategy ideas on the workspace
            self.workspace.publish(
                topic='insight',
                data={
                    'type': 'strategy_ideas_added',
                    'count': len(added),
                    'names': [r.name for r in added],
                },
                source='curiosity_engine',
            )

        return added

    def _create_hypothesis_for_strategy(self, record: StrategyIdeaRecord) -> Optional[str]:
        """
        Creates a testable hypothesis from a strategy idea.

        The hypothesis will test whether the strategy's core concept shows
        promise under the specified market context.

        Args:
            record: The StrategyIdeaRecord to create a hypothesis for.

        Returns:
            The hypothesis ID if created, None otherwise.
        """
        # Build a condition string from the strategy's market context
        condition = self._parse_market_context_to_condition(record.market_context)

        # Build a prediction - strategies should achieve at least 55% win rate
        prediction = "win_rate > 0.55"

        # Create a unique hypothesis ID
        hyp_id = hashlib.md5(f"strategy_hyp_{record.idea_id}".encode()).hexdigest()[:8]

        # Skip if hypothesis already exists
        if hyp_id in self._hypotheses:
            return hyp_id

        hypothesis = Hypothesis(
            hypothesis_id=hyp_id,
            description=f"LLM strategy '{record.name}' may be viable: {record.concept[:60]}",
            condition=condition,
            prediction=prediction,
            rationale=f"LLM proposed strategy for {record.market_context}. {record.rationale[:100]}",
            source="llm_strategy_idea",
        )

        self._hypotheses[hyp_id] = hypothesis
        logger.info(f"Created hypothesis '{hyp_id}' for strategy idea '{record.name}'")
        return hyp_id

    def _parse_market_context_to_condition(self, market_context: str) -> str:
        """
        Converts a natural language market context into a testable condition string.

        Args:
            market_context: Natural language description (e.g., "high volatility regimes")

        Returns:
            A condition string suitable for hypothesis testing.
        """
        context_lower = market_context.lower()
        conditions = []

        # Parse volatility conditions
        if 'high volatility' in context_lower or 'high vix' in context_lower:
            conditions.append('vix >= 25')
        elif 'low volatility' in context_lower or 'low vix' in context_lower:
            conditions.append('vix < 18')
        elif 'extreme volatility' in context_lower:
            conditions.append('vix >= 35')

        # Parse regime conditions
        if 'bull' in context_lower or 'uptrend' in context_lower:
            conditions.append('regime = BULL')
        elif 'bear' in context_lower or 'downtrend' in context_lower:
            conditions.append('regime = BEAR')
        elif 'chop' in context_lower or 'sideways' in context_lower or 'range' in context_lower:
            conditions.append('regime = CHOPPY')

        # Parse fear/greed conditions
        if 'fear' in context_lower or 'panic' in context_lower:
            conditions.append('sentiment < -0.3')
        elif 'greed' in context_lower or 'euphoria' in context_lower:
            conditions.append('sentiment > 0.3')

        # Default condition if no specific patterns matched
        if not conditions:
            conditions.append('TRUE')  # Match all episodes

        return ' AND '.join(conditions)

    def get_strategy_ideas(self, status: Optional[str] = None) -> List[StrategyIdeaRecord]:
        """
        Returns strategy ideas, optionally filtered by status.

        Args:
            status: Optional filter for status ('proposed', 'under_review',
                    'implemented', 'rejected'). If None, returns all.

        Returns:
            List of StrategyIdeaRecord objects.
        """
        if status is None:
            return list(self._strategy_ideas.values())
        return [r for r in self._strategy_ideas.values() if r.status == status]

    def update_strategy_idea_status(
        self,
        idea_id: str,
        new_status: str,
        notes: Optional[str] = None,
        priority: Optional[int] = None,
    ) -> Optional[StrategyIdeaRecord]:
        """
        Updates the status and metadata of a strategy idea.

        Args:
            idea_id: The ID of the strategy idea to update.
            new_status: The new status ('proposed', 'under_review', 'implemented', 'rejected').
            notes: Optional review notes to add.
            priority: Optional new implementation priority (0=low, 1=medium, 2=high).

        Returns:
            The updated StrategyIdeaRecord, or None if not found.
        """
        if idea_id not in self._strategy_ideas:
            logger.warning(f"Strategy idea not found: {idea_id}")
            return None

        record = self._strategy_ideas[idea_id]
        record.status = new_status
        record.reviewed_at = datetime.now()

        if notes:
            record.reviewer_notes.append(f"[{datetime.now().strftime('%Y-%m-%d')}] {notes}")

        if priority is not None:
            record.implementation_priority = priority

        self._save_state()
        logger.info(f"Updated strategy idea '{record.name}' to status '{new_status}'")
        return record

    def test_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        """
        Tests a single hypothesis against historical data from episodic memory.

        Args:
            hypothesis: The Hypothesis object to be tested.

        Returns:
            The updated Hypothesis object with test results and a new status.
        """
        from cognitive.episodic_memory import EpisodeOutcome
        hypothesis.status = HypothesisStatus.TESTING
        hypothesis.tested_at = datetime.now()

        # 1. Get a large sample of historical data to test against.
        test_data = self.episodic_memory.get_recent_episodes(limit=1000)

        # 2. Filter the data to only include episodes that match the hypothesis condition.
        matching_episodes = self._filter_by_condition(test_data, hypothesis.condition)
        hypothesis.sample_size = len(matching_episodes)

        # 3. If the sample size is too small, the result is inconclusive.
        if hypothesis.sample_size < self.min_sample_size:
            hypothesis.status = HypothesisStatus.INCONCLUSIVE
            logger.info(f"Hypothesis '{hypothesis.description}' is inconclusive: sample size {hypothesis.sample_size} is below threshold {self.min_sample_size}.")
            return hypothesis

        # 4. Calculate the observed metric (e.g., win rate) from the matching episodes.
        wins = [e for e in matching_episodes if e.outcome == EpisodeOutcome.WIN]
        total_relevant = len([e for e in matching_episodes if e.outcome is not None])
        if total_relevant == 0:
            hypothesis.status = HypothesisStatus.INCONCLUSIVE
            return hypothesis
        hypothesis.observed_value = len(wins) / total_relevant

        # 5. Parse the expected value from the prediction string (e.g., "win_rate > 0.6" -> 0.6).
        hypothesis.expected_value = self._parse_expected_value(hypothesis.prediction)

        # 6. Perform a statistical test to get a p-value.
        hypothesis.p_value = self._calculate_p_value(
            hypothesis.observed_value, hypothesis.expected_value, total_relevant
        )

        # 7. Determine the final status based on the p-value.
        if hypothesis.p_value < self.significance_level:
            if hypothesis.observed_value >= hypothesis.expected_value:
                hypothesis.status = HypothesisStatus.VALIDATED
                # This is a real edge! Promote it.
                self._create_edge_from_hypothesis(hypothesis)
            else:
                hypothesis.status = HypothesisStatus.REJECTED
        else:
            hypothesis.status = HypothesisStatus.INCONCLUSIVE
        
        self._save_state()
        logger.info(
            f"Tested hypothesis '{hypothesis.description}': {hypothesis.status.value} "
            f"(p-value={hypothesis.p_value:.3f}, observed={hypothesis.observed_value:.3f}, n={hypothesis.sample_size})"
        )
        return hypothesis

    def _filter_by_condition(self, episodes: List[Any], condition: str) -> List[Any]:
        """Filters a list of episodes based on a condition string."""
        from cognitive.semantic_memory import ConditionMatcher
        matcher = ConditionMatcher()
        
        matching = []
        for ep in episodes:
            # Create a context dictionary for the matcher from the episode data.
            context = {
                'regime': ep.market_context.get('regime', '').lower(),
                'strategy': ep.signal_context.get('strategy', '').lower(),
                'vix': ep.market_context.get('vix', 20),
                'is_mean_reversion': 'rsi' in ep.signal_context.get('strategy', ''), # Example
            }
            if matcher.matches(condition, context):
                matching.append(ep)
        return matching

    def _parse_expected_value(self, prediction: str) -> float:
        """Parses the numerical value from a prediction string like 'win_rate > 0.6'."""
        for op in ['>=', '>', '<=', '<']:
            if op in prediction:
                parts = prediction.split(op)
                try:
                    return float(parts[1].strip())
                except (ValueError, IndexError):
                    pass
        return 0.5  # Default expectation if parsing fails.

    def _calculate_p_value(self, observed: float, expected: float, n: int) -> float:
        """
        Calculates a one-tailed p-value for an observed proportion using a
        normal approximation to the binomial test.
        """
        import math
        if n == 0: return 1.0

        # Clamp expected to valid range [0, 1] and guard against edge cases
        expected = max(0.0, min(1.0, expected))
        if expected == 0.0 or expected == 1.0:
            return 0.0 if observed == expected else 1.0

        # Standard error of a proportion
        variance = expected * (1 - expected) / n
        if variance <= 0: return 0.0 if observed == expected else 1.0
        se = math.sqrt(variance)
        if se == 0: return 0.0 if observed == expected else 1.0

        # Z-score: how many standard deviations the observation is from the expectation.
        z = (observed - expected) / se
        
        # One-tailed p-value from Z-score.
        p = 1.0 - (0.5 * (1 + math.erf(z / math.sqrt(2))))
        return p

    def _create_edge_from_hypothesis(self, hypothesis: Hypothesis) -> None:
        """
        Promotes a validated hypothesis into an actionable Edge and creates a
        corresponding rule in Semantic Memory.
        """
        edge_id = f"edge_{hypothesis.hypothesis_id}"

        if edge_id in self._edges:
            # If edge already exists, update its validation stats.
            edge = self._edges[edge_id]
            edge.last_validated = datetime.now()
            edge.times_validated += 1
            edge.sample_size = hypothesis.sample_size
            edge.expected_win_rate = hypothesis.observed_value
            edge.confidence = 1 - (hypothesis.p_value or 1.0)
            logger.info(f"Re-validated existing edge: '{edge.description}'")
        else:
            # Create a new Edge object.
            edge = Edge(
                edge_id=edge_id,
                description=hypothesis.description,
                condition=hypothesis.condition,
                expected_win_rate=hypothesis.observed_value,
                # Simple heuristic for profit factor.
                expected_profit_factor=1.0 + (hypothesis.observed_value - 0.5) * 2,
                sample_size=hypothesis.sample_size,
                confidence=1 - (hypothesis.p_value or 1.0),
                first_discovered=datetime.now(),
                last_validated=datetime.now(),
            )
            self._edges[edge_id] = edge
            logger.info(f"*** New trading edge discovered: '{edge.description}' ***")

            # **Crucial Step:** Convert this new knowledge into an active rule
            # for the CognitiveBrain to use in future decisions.
            self.semantic_memory.add_rule(
                condition=edge.condition,
                action="increase_confidence",
                parameters={'edge_win_rate': edge.expected_win_rate},
                confidence=edge.confidence,
                source=f"Curiosity Engine: {edge.description}",
                tags=['edge', 'validated', 'auto-generated'],
            )

            # Announce the discovery on the global workspace.
            self.workspace.publish(
                topic='insight',
                data={'type': 'edge_discovered', 'edge': edge.to_dict()},
                source='curiosity_engine',
            )

    def test_all_pending(self) -> Dict[str, int]:
        """Finds and tests all hypotheses with 'proposed' status."""
        results = {'tested': 0, 'validated': 0, 'rejected': 0, 'inconclusive': 0}
        pending = [h for h in self._hypotheses.values() if h.status == HypothesisStatus.PROPOSED]
        
        if not pending:
            logger.info("No pending hypotheses to test.")
            return results

        logger.info(f"Testing {len(pending)} pending hypotheses...")
        for hyp in pending:
            self.test_hypothesis(hyp)
            results['tested'] += 1
            results[hyp.status.value] = results.get(hyp.status.value, 0) + 1
        
        logger.info(f"Hypothesis testing round complete: {results}")
        return results

    def get_validated_edges(self) -> List[Edge]:
        """Returns all validated edges, sorted by confidence."""
        edges = list(self._edges.values())
        edges.sort(key=lambda e: e.confidence, reverse=True)
        return edges

    def get_active_hypotheses(self) -> List[Hypothesis]:
        """Returns hypotheses that are still under consideration (not rejected)."""
        return [h for h in self._hypotheses.values() if h.status != HypothesisStatus.REJECTED]

    # ==========================================================================
    # COUNTERFACTUAL SIMULATION (Task B2)
    # ==========================================================================

    def design_counterfactual_tests(self, episode_id: str) -> List[Dict[str, Any]]:
        """
        Designs counterfactual tests for a given episode.

        This generates a set of "what-if" scenarios that can be simulated to
        understand what would have happened under different conditions.

        Args:
            episode_id: The ID of the episode to analyze

        Returns:
            List of counterfactual test specifications
        """
        # Try to load the episode
        try:
            from cognitive.episodic_memory import get_episodic_memory
            episode = get_episodic_memory().get_episode(episode_id)
            if not episode:
                logger.warning(f"Episode {episode_id} not found for counterfactual design")
                return []
        except Exception as e:
            logger.error(f"Error loading episode for counterfactual: {e}")
            return []

        counterfactuals = []

        # Design 1: VIX Spike - What if volatility had spiked?
        vix = episode.market_context.get('vix', 20)
        if vix < 30:  # If VIX was calm, test spike scenario
            counterfactuals.append({
                'scenario_type': 'vix_spike',
                'description': f"What if VIX had spiked from {vix:.1f} to 35+ during the trade?",
                'deviation_type': 'vix_spike',
                'magnitude': 1.5,
                'original_context': {'vix': vix},
                'test_context': {'vix': 35},
                'episode_id': episode_id,
            })

        # Design 2: Regime Change - What if regime had shifted?
        regime = episode.market_context.get('regime', 'unknown')
        alternative_regimes = {'bull': 'bear', 'bear': 'bull', 'neutral': 'choppy', 'choppy': 'neutral'}
        if regime in alternative_regimes:
            alt_regime = alternative_regimes[regime]
            counterfactuals.append({
                'scenario_type': 'regime_change',
                'description': f"What if regime had been {alt_regime} instead of {regime}?",
                'deviation_type': 'regime_change',
                'magnitude': 1.0,
                'original_context': {'regime': regime},
                'test_context': {'regime': alt_regime},
                'episode_id': episode_id,
            })

        # Design 3: Sentiment Shift - What if sentiment was opposite?
        sentiment = episode.market_context.get('market_sentiment', {}).get('compound', 0)
        if abs(sentiment) > 0.2:  # If there was notable sentiment
            counterfactuals.append({
                'scenario_type': 'sentiment_shift',
                'description': f"What if sentiment had been opposite ({-sentiment:.2f} vs {sentiment:.2f})?",
                'deviation_type': 'sentiment_shift',
                'magnitude': -2 * sentiment,  # Flip to opposite
                'original_context': {'sentiment': sentiment},
                'test_context': {'sentiment': -sentiment},
                'episode_id': episode_id,
            })

        # Design 4: Price Shock - What if there was an unexpected move?
        counterfactuals.append({
            'scenario_type': 'price_shock',
            'description': "What if there was a sudden 3% adverse move after entry?",
            'deviation_type': 'price_shock',
            'magnitude': -0.03,  # 3% adverse
            'original_context': {},
            'test_context': {'shock_pct': -0.03},
            'episode_id': episode_id,
        })

        logger.info(f"Designed {len(counterfactuals)} counterfactual tests for episode {episode_id}")
        return counterfactuals

    def trigger_scenario_generation(
        self,
        hypothesis: 'Hypothesis',
        seed_data: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Triggers synthetic data generation to test a hypothesis.

        Args:
            hypothesis: The hypothesis to test
            seed_data: Historical price data to seed the generator

        Returns:
            Generated scenario data, or None if generation fails
        """
        try:
            from data.ml import get_generative_model, ScenarioParams

            model = get_generative_model()

            # Parse hypothesis to determine scenario parameters
            params = self._hypothesis_to_scenario_params(hypothesis)

            # Generate the scenario
            result = model.generate_future_scenario(seed_data, params)

            logger.info(
                f"Generated scenario for hypothesis {hypothesis.hypothesis_id}: "
                f"{params.regime_type}/{params.volatility_level} for {params.duration_days} days"
            )

            return result

        except ImportError as e:
            logger.warning(f"Generative model not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Scenario generation failed: {e}")
            return None

    def _hypothesis_to_scenario_params(self, hypothesis: 'Hypothesis') -> 'ScenarioParams':
        """
        Converts a hypothesis into scenario generation parameters.
        """
        from data.ml import ScenarioParams

        # Default parameters
        regime = 'neutral'
        volatility = 'normal'
        duration = 20

        # Parse hypothesis condition for context clues
        condition = hypothesis.condition.lower() if hypothesis.condition else ''

        if 'bull' in condition:
            regime = 'bull'
        elif 'bear' in condition:
            regime = 'bear'
        elif 'choppy' in condition or 'volatile' in condition:
            regime = 'choppy'

        if 'high volatility' in condition or 'vix > 30' in condition:
            volatility = 'high'
        elif 'low volatility' in condition or 'vix < 15' in condition:
            volatility = 'low'
        elif 'extreme' in condition:
            volatility = 'extreme'

        return ScenarioParams(
            regime_type=regime,
            volatility_level=volatility,
            duration_days=duration,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about the engine's knowledge base."""
        statuses = [h.status for h in self._hypotheses.values()]
        strategy_statuses = [s.status for s in self._strategy_ideas.values()]
        return {
            'total_hypotheses': len(self._hypotheses),
            'proposed': statuses.count(HypothesisStatus.PROPOSED),
            'validated': statuses.count(HypothesisStatus.VALIDATED),
            'rejected': statuses.count(HypothesisStatus.REJECTED),
            'inconclusive': statuses.count(HypothesisStatus.INCONCLUSIVE),
            'total_edges': len(self._edges),
            'total_strategy_ideas': len(self._strategy_ideas),
            'strategy_ideas_proposed': strategy_statuses.count('proposed'),
            'strategy_ideas_under_review': strategy_statuses.count('under_review'),
            'strategy_ideas_implemented': strategy_statuses.count('implemented'),
            'strategy_ideas_rejected': strategy_statuses.count('rejected'),
        }

    def _save_state(self) -> None:
        """Saves the current state of all hypotheses, edges, and strategy ideas to a JSON file."""
        try:
            state = {
                'hypotheses': {k: v.to_dict() for k, v in self._hypotheses.items()},
                'edges': {k: v.to_dict() for k, v in self._edges.items()},
                'strategy_ideas': {k: v.to_dict() for k, v in self._strategy_ideas.items()},
                'saved_at': datetime.now().isoformat(),
            }
            state_file = self.storage_dir / "curiosity_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save curiosity engine state: {e}")

    def _load_state(self) -> None:
        """Loads the engine's state from a JSON file upon initialization."""
        state_file = self.storage_dir / "curiosity_state.json"
        if not state_file.exists():
            logger.info("No curiosity state file found. Starting fresh.")
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            for hid, hdata in state.get('hypotheses', {}).items():
                # Reconstruct Hypothesis objects from the saved dictionary data.
                self._hypotheses[hid] = Hypothesis(
                    **{k: v for k, v in hdata.items() if k not in ['status', 'created_at', 'tested_at']},
                    status=HypothesisStatus(hdata.get('status', 'proposed')),
                    created_at=datetime.fromisoformat(hdata['created_at']) if hdata.get('created_at') else datetime.now(),
                    tested_at=datetime.fromisoformat(hdata['tested_at']) if hdata.get('tested_at') else None
                )

            for eid, edata in state.get('edges', {}).items():
                # Reconstruct Edge objects.
                self._edges[eid] = Edge(
                     **{k: v for k, v in edata.items() if k not in ['first_discovered', 'last_validated']},
                     first_discovered=datetime.fromisoformat(edata['first_discovered']),
                     last_validated=datetime.fromisoformat(edata['last_validated']),
                )

            # Load strategy ideas
            for sid, sdata in state.get('strategy_ideas', {}).items():
                # Reconstruct StrategyIdeaRecord objects.
                self._strategy_ideas[sid] = StrategyIdeaRecord(
                    idea_id=sdata['idea_id'],
                    name=sdata['name'],
                    concept=sdata['concept'],
                    market_context=sdata['market_context'],
                    entry_conditions=sdata['entry_conditions'],
                    exit_conditions=sdata['exit_conditions'],
                    risk_management=sdata['risk_management'],
                    rationale=sdata['rationale'],
                    status=sdata.get('status', 'proposed'),
                    created_at=datetime.fromisoformat(sdata['created_at']) if sdata.get('created_at') else datetime.now(),
                    reviewed_at=datetime.fromisoformat(sdata['reviewed_at']) if sdata.get('reviewed_at') else None,
                    reviewer_notes=sdata.get('reviewer_notes', []),
                    implementation_priority=sdata.get('implementation_priority', 0),
                    related_hypothesis_ids=sdata.get('related_hypothesis_ids', []),
                    confidence=sdata.get('confidence', 0.5),
                )
        except Exception as e:
            logger.warning(f"Failed to load curiosity state from {state_file}: {e}. Starting fresh.")
            self._hypotheses = {}
            self._edges = {}
            self._strategy_ideas = {}

    def introspect(self) -> str:
        """Generates a human-readable report of the engine's current state."""
        stats = self.get_stats()
        lines = [
            "--- Curiosity Engine Introspection ---",
            f"I have {stats['total_hypotheses']} hypotheses about the market.",
            f"  - {stats['validated']} have been validated and are now trading edges.",
            f"  - {stats['rejected']} have been proven false.",
            f"  - {stats['inconclusive']} need more data.",
            f"  - {stats['proposed']} are waiting to be tested.",
            f"I have discovered a total of {stats['total_edges']} trading edges.",
            "",
            f"I have {stats['total_strategy_ideas']} LLM-generated strategy ideas:",
            f"  - {stats['strategy_ideas_proposed']} proposed and awaiting review.",
            f"  - {stats['strategy_ideas_under_review']} currently under review.",
            f"  - {stats['strategy_ideas_implemented']} successfully implemented.",
            f"  - {stats['strategy_ideas_rejected']} rejected after evaluation.",
            "\nMy top 3 curiosities right now are:",
        ]

        pending = sorted(
            [h for h in self._hypotheses.values() if h.status == HypothesisStatus.PROPOSED],
            key=lambda h: h.created_at, reverse=True
        )

        for h in pending[:3]:
            lines.append(f"  - I wonder if '{h.description}'")

        # Add recent strategy ideas
        if self._strategy_ideas:
            lines.append("\nRecent strategy ideas from LLM:")
            recent_ideas = sorted(
                self._strategy_ideas.values(),
                key=lambda s: s.created_at, reverse=True
            )[:3]
            for idea in recent_ideas:
                lines.append(f"  - [{idea.status}] '{idea.name}': {idea.concept[:50]}...")

        return "\n".join(lines)


# --- Singleton Implementation ---
_curiosity_engine: Optional[CuriosityEngine] = None

def get_curiosity_engine() -> CuriosityEngine:
    """Factory function to get the singleton instance of CuriosityEngine."""
    global _curiosity_engine
    if _curiosity_engine is None:
        _curiosity_engine = CuriosityEngine()
    return _curiosity_engine
