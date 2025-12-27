"""
Curiosity Engine - Pattern Discovery
======================================

Autonomous hypothesis generation and testing.

This is how the robot "gets smarter" without human intervention:
- Proposes hypotheses about market behavior
- Tests them against data
- Maintains leaderboard of robust edges
- Runs as background process

Features:
- Hypothesis generation from observations
- Automated testing framework
- Edge discovery and validation
- Continuous learning loop

Usage:
    from cognitive.curiosity_engine import CuriosityEngine

    engine = CuriosityEngine()

    # Generate hypotheses from recent data
    hypotheses = engine.generate_hypotheses(market_data)

    # Test hypotheses
    results = engine.test_hypotheses(hypotheses, historical_data)

    # Get proven edges
    edges = engine.get_validated_edges()
"""

import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class HypothesisStatus(Enum):
    """Status of a hypothesis."""
    PROPOSED = "proposed"       # Just generated
    TESTING = "testing"         # Currently being tested
    VALIDATED = "validated"     # Proven with significance
    REJECTED = "rejected"       # Failed testing
    INCONCLUSIVE = "inconclusive"  # Needs more data


@dataclass
class Hypothesis:
    """A testable hypothesis about market behavior."""
    hypothesis_id: str
    description: str
    condition: str  # When to test (e.g., "regime = BULL")
    prediction: str  # What we predict (e.g., "win_rate > 0.6")
    rationale: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    created_at: datetime = field(default_factory=datetime.now)
    tested_at: Optional[datetime] = None
    sample_size: int = 0
    observed_value: float = 0.0
    expected_value: float = 0.0
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    source: str = ""  # Where this came from

    def to_dict(self) -> Dict:
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
    """A validated trading edge."""
    edge_id: str
    description: str
    condition: str
    expected_win_rate: float
    expected_profit_factor: float
    sample_size: int
    confidence: float  # 0-1
    first_discovered: datetime
    last_validated: datetime
    times_validated: int = 1
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
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


class CuriosityEngine:
    """
    Autonomous pattern discovery and hypothesis testing.

    This is the "curiosity" component that drives continuous learning.
    """

    def __init__(
        self,
        storage_dir: str = "state/cognitive",
        min_sample_size: int = 30,
        significance_level: float = 0.05,
        min_edge_win_rate: float = 0.55,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        self.min_edge_win_rate = min_edge_win_rate

        # Hypothesis storage
        self._hypotheses: Dict[str, Hypothesis] = {}
        self._edges: Dict[str, Edge] = {}

        # Lazy dependencies
        self._episodic_memory = None
        self._semantic_memory = None
        self._workspace = None

        self._load_state()
        logger.info(
            f"CuriosityEngine initialized with {len(self._hypotheses)} hypotheses, "
            f"{len(self._edges)} edges"
        )

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
        Generate hypotheses from observations or recent episodes.

        Args:
            observations: Optional dict of current observations

        Returns:
            List of new hypotheses
        """
        new_hypotheses = []

        # === Strategy 1: From Recent Episodes ===
        episodes = self.episodic_memory.get_recent_episodes(limit=50)
        new_hypotheses.extend(self._hypotheses_from_episodes(episodes))

        # === Strategy 2: From Observations ===
        if observations:
            new_hypotheses.extend(self._hypotheses_from_observations(observations))

        # === Strategy 3: Combinatorial ===
        new_hypotheses.extend(self._combinatorial_hypotheses())

        # Deduplicate and store
        for hyp in new_hypotheses:
            if hyp.hypothesis_id not in self._hypotheses:
                self._hypotheses[hyp.hypothesis_id] = hyp
                logger.info(f"New hypothesis: {hyp.description}")

        self._save_state()

        return new_hypotheses

    def _hypotheses_from_episodes(self, episodes: List[Any]) -> List[Hypothesis]:
        """Generate hypotheses from episode patterns."""
        from cognitive.episodic_memory import EpisodeOutcome

        hypotheses = []

        # Group by context
        by_context: Dict[str, List[Any]] = {}
        for ep in episodes:
            key = f"{ep.market_context.get('regime', 'unknown')}|{ep.signal_context.get('strategy', 'unknown')}"
            if key not in by_context:
                by_context[key] = []
            by_context[key].append(ep)

        # Look for interesting patterns
        for key, context_episodes in by_context.items():
            if len(context_episodes) < 5:
                continue

            regime, strategy = key.split('|')

            wins = [e for e in context_episodes if e.outcome == EpisodeOutcome.WIN]
            losses = [e for e in context_episodes if e.outcome == EpisodeOutcome.LOSS]

            if len(wins) + len(losses) < 5:
                continue

            win_rate = len(wins) / (len(wins) + len(losses))

            # High win rate hypothesis
            if win_rate > 0.65:
                hyp_id = hashlib.md5(f"high_wr_{key}".encode()).hexdigest()[:8]
                if hyp_id not in self._hypotheses:
                    hypotheses.append(Hypothesis(
                        hypothesis_id=hyp_id,
                        description=f"{strategy} has high win rate in {regime} regime",
                        condition=f"regime = {regime} AND strategy = {strategy}",
                        prediction="win_rate > 0.6",
                        rationale=f"Observed {win_rate:.1%} win rate in {len(context_episodes)} episodes",
                        source="episode_pattern",
                    ))

            # Low win rate hypothesis
            elif win_rate < 0.35:
                hyp_id = hashlib.md5(f"low_wr_{key}".encode()).hexdigest()[:8]
                if hyp_id not in self._hypotheses:
                    hypotheses.append(Hypothesis(
                        hypothesis_id=hyp_id,
                        description=f"{strategy} underperforms in {regime} regime",
                        condition=f"regime = {regime} AND strategy = {strategy}",
                        prediction="win_rate < 0.4",
                        rationale=f"Observed {win_rate:.1%} win rate in {len(context_episodes)} episodes",
                        source="episode_pattern",
                    ))

        return hypotheses

    def _hypotheses_from_observations(self, observations: Dict) -> List[Hypothesis]:
        """Generate hypotheses from current observations."""
        hypotheses = []

        # VIX-based hypothesis
        vix = observations.get('vix', 20)
        if vix > 30:
            hyp_id = hashlib.md5(f"high_vix_{vix:.0f}".encode()).hexdigest()[:8]
            hypotheses.append(Hypothesis(
                hypothesis_id=hyp_id,
                description=f"Mean reversion works better when VIX > 30",
                condition="vix > 30",
                prediction="mean_reversion_win_rate > 0.6",
                rationale=f"High VIX ({vix}) often indicates oversold conditions",
                source="observation",
            ))

        # Volume-based hypothesis
        volume_ratio = observations.get('volume_ratio', 1.0)
        if volume_ratio > 2.0:
            hyp_id = hashlib.md5(f"high_vol_{volume_ratio:.1f}".encode()).hexdigest()[:8]
            hypotheses.append(Hypothesis(
                hypothesis_id=hyp_id,
                description="High volume precedes trend continuation",
                condition="volume_ratio > 2.0",
                prediction="trend_following_success > 0.55",
                rationale=f"Volume {volume_ratio:.1f}x average suggests conviction",
                source="observation",
            ))

        return hypotheses

    def _combinatorial_hypotheses(self) -> List[Hypothesis]:
        """Generate hypotheses by combining factors."""
        hypotheses = []

        # Combine regime + strategy + volatility
        regimes = ['BULL', 'BEAR', 'CHOPPY']
        strategies = ['donchian', 'turtle_soup']
        vol_conditions = ['vix < 20', 'vix >= 20 AND vix < 30', 'vix >= 30']

        # Only generate a few at a time to avoid explosion
        import random
        combinations = [
            (r, s, v) for r in regimes for s in strategies for v in vol_conditions
        ]
        sample = random.sample(combinations, min(3, len(combinations)))

        for regime, strategy, vol_cond in sample:
            hyp_id = hashlib.md5(f"{regime}_{strategy}_{vol_cond}".encode()).hexdigest()[:8]
            if hyp_id not in self._hypotheses:
                hypotheses.append(Hypothesis(
                    hypothesis_id=hyp_id,
                    description=f"{strategy} performance in {regime} with {vol_cond}",
                    condition=f"regime = {regime} AND strategy = {strategy} AND {vol_cond}",
                    prediction="win_rate > 0.55",
                    rationale="Exploring factor combination",
                    source="combinatorial",
                ))

        return hypotheses

    def test_hypothesis(
        self,
        hypothesis: Hypothesis,
        test_data: Optional[List[Any]] = None,
    ) -> Hypothesis:
        """
        Test a hypothesis against data.

        Args:
            hypothesis: Hypothesis to test
            test_data: Optional list of episodes to test against

        Returns:
            Updated hypothesis with test results
        """
        from cognitive.episodic_memory import EpisodeOutcome

        hypothesis.status = HypothesisStatus.TESTING
        hypothesis.tested_at = datetime.now()

        # Get test data
        if test_data is None:
            test_data = self.episodic_memory.get_recent_episodes(limit=200)

        # Filter to matching condition
        matching = self._filter_by_condition(test_data, hypothesis.condition)

        hypothesis.sample_size = len(matching)

        if hypothesis.sample_size < self.min_sample_size:
            hypothesis.status = HypothesisStatus.INCONCLUSIVE
            return hypothesis

        # Calculate observed value
        wins = [e for e in matching if e.outcome == EpisodeOutcome.WIN]
        losses = [e for e in matching if e.outcome == EpisodeOutcome.LOSS]
        total = len(wins) + len(losses)

        if total == 0:
            hypothesis.status = HypothesisStatus.INCONCLUSIVE
            return hypothesis

        hypothesis.observed_value = len(wins) / total

        # Parse expected value from prediction
        hypothesis.expected_value = self._parse_expected_value(hypothesis.prediction)

        # Statistical test (simple binomial)
        hypothesis.p_value = self._calculate_p_value(
            hypothesis.observed_value,
            hypothesis.expected_value,
            total
        )

        # Determine status
        if hypothesis.p_value < self.significance_level:
            if hypothesis.observed_value >= hypothesis.expected_value:
                hypothesis.status = HypothesisStatus.VALIDATED
                # Create edge
                self._create_edge_from_hypothesis(hypothesis)
            else:
                hypothesis.status = HypothesisStatus.REJECTED
        else:
            hypothesis.status = HypothesisStatus.INCONCLUSIVE

        self._save_state()

        logger.info(
            f"Hypothesis {hypothesis.hypothesis_id}: {hypothesis.status.value} "
            f"(observed={hypothesis.observed_value:.3f}, n={hypothesis.sample_size})"
        )

        return hypothesis

    def _filter_by_condition(self, episodes: List[Any], condition: str) -> List[Any]:
        """Filter episodes matching a condition."""
        from cognitive.semantic_memory import ConditionMatcher
        matcher = ConditionMatcher()

        matching = []
        for ep in episodes:
            context = {
                'regime': ep.market_context.get('regime', '').lower(),
                'strategy': ep.signal_context.get('strategy', '').lower(),
                'vix': ep.market_context.get('vix', 20),
            }
            if matcher.matches(condition, context):
                matching.append(ep)

        return matching

    def _parse_expected_value(self, prediction: str) -> float:
        """Parse expected value from prediction string."""
        # Simple parsing for "metric > value" or "metric >= value"
        for op in ['>=', '>', '<=', '<']:
            if op in prediction:
                parts = prediction.split(op)
                try:
                    return float(parts[1].strip())
                except:
                    pass
        return 0.5  # Default

    def _calculate_p_value(
        self,
        observed: float,
        expected: float,
        n: int,
    ) -> float:
        """Calculate p-value for observed vs expected proportion."""
        import math

        if n == 0:
            return 1.0

        # Standard error
        se = math.sqrt(expected * (1 - expected) / n)
        if se == 0:
            return 0.0 if observed != expected else 1.0

        # Z-score
        z = (observed - expected) / se

        # One-tailed p-value (simplified)
        # Using normal approximation
        p = 0.5 * (1 + math.erf(-abs(z) / math.sqrt(2)))

        return p

    def _create_edge_from_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Create a validated edge from a hypothesis."""
        edge_id = f"edge_{hypothesis.hypothesis_id}"

        if edge_id in self._edges:
            # Update existing edge
            edge = self._edges[edge_id]
            edge.last_validated = datetime.now()
            edge.times_validated += 1
            edge.sample_size = hypothesis.sample_size
            edge.expected_win_rate = hypothesis.observed_value
        else:
            # Create new edge
            edge = Edge(
                edge_id=edge_id,
                description=hypothesis.description,
                condition=hypothesis.condition,
                expected_win_rate=hypothesis.observed_value,
                expected_profit_factor=1.0 + (hypothesis.observed_value - 0.5) * 2,
                sample_size=hypothesis.sample_size,
                confidence=1 - hypothesis.p_value if hypothesis.p_value else 0.5,
                first_discovered=datetime.now(),
                last_validated=datetime.now(),
            )
            self._edges[edge_id] = edge

            # Create semantic rule
            self.semantic_memory.add_rule(
                condition=edge.condition,
                action="increase_confidence",
                parameters={'edge_win_rate': edge.expected_win_rate},
                confidence=edge.confidence,
                source=f"Edge discovery: {edge.description}",
                tags=['edge', 'validated'],
            )

            # Publish discovery
            self.workspace.publish(
                topic='insight',
                data={
                    'type': 'edge_discovered',
                    'edge': edge.to_dict(),
                },
                source='curiosity_engine',
            )

            logger.info(f"New edge discovered: {edge.description}")

    def test_all_pending(self) -> Dict[str, int]:
        """Test all pending hypotheses."""
        results = {
            'tested': 0,
            'validated': 0,
            'rejected': 0,
            'inconclusive': 0,
        }

        pending = [h for h in self._hypotheses.values()
                   if h.status == HypothesisStatus.PROPOSED]

        for hyp in pending:
            self.test_hypothesis(hyp)
            results['tested'] += 1
            results[hyp.status.value] = results.get(hyp.status.value, 0) + 1

        return results

    def get_validated_edges(self) -> List[Edge]:
        """Get all validated edges, sorted by confidence."""
        edges = list(self._edges.values())
        edges.sort(key=lambda e: e.confidence, reverse=True)
        return edges

    def get_active_hypotheses(self) -> List[Hypothesis]:
        """Get hypotheses that are not yet rejected."""
        return [h for h in self._hypotheses.values()
                if h.status != HypothesisStatus.REJECTED]

    def get_stats(self) -> Dict[str, Any]:
        """Get curiosity engine statistics."""
        hypotheses = list(self._hypotheses.values())
        statuses = [h.status for h in hypotheses]

        return {
            'total_hypotheses': len(hypotheses),
            'proposed': statuses.count(HypothesisStatus.PROPOSED),
            'validated': statuses.count(HypothesisStatus.VALIDATED),
            'rejected': statuses.count(HypothesisStatus.REJECTED),
            'inconclusive': statuses.count(HypothesisStatus.INCONCLUSIVE),
            'total_edges': len(self._edges),
        }

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            'hypotheses': {k: v.to_dict() for k, v in self._hypotheses.items()},
            'edges': {k: v.to_dict() for k, v in self._edges.items()},
            'saved_at': datetime.now().isoformat(),
        }
        state_file = self.storage_dir / "curiosity_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.storage_dir / "curiosity_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Load hypotheses
            for hid, hdata in state.get('hypotheses', {}).items():
                self._hypotheses[hid] = Hypothesis(
                    hypothesis_id=hdata['hypothesis_id'],
                    description=hdata['description'],
                    condition=hdata['condition'],
                    prediction=hdata['prediction'],
                    rationale=hdata['rationale'],
                    status=HypothesisStatus(hdata.get('status', 'proposed')),
                    source=hdata.get('source', ''),
                )

            # Load edges
            for eid, edata in state.get('edges', {}).items():
                self._edges[eid] = Edge(
                    edge_id=edata['edge_id'],
                    description=edata['description'],
                    condition=edata['condition'],
                    expected_win_rate=edata['expected_win_rate'],
                    expected_profit_factor=edata['expected_profit_factor'],
                    sample_size=edata['sample_size'],
                    confidence=edata['confidence'],
                    first_discovered=datetime.fromisoformat(edata['first_discovered']),
                    last_validated=datetime.fromisoformat(edata['last_validated']),
                    times_validated=edata.get('times_validated', 1),
                )

        except Exception as e:
            logger.warning(f"Failed to load curiosity state: {e}")

    def introspect(self) -> str:
        """Generate introspective report."""
        stats = self.get_stats()

        lines = [
            "=== Curiosity Engine Introspection ===",
            "",
            f"Total hypotheses generated: {stats['total_hypotheses']}",
            f"  - Validated: {stats['validated']}",
            f"  - Rejected: {stats['rejected']}",
            f"  - Inconclusive: {stats['inconclusive']}",
            f"  - Pending: {stats['proposed']}",
            f"",
            f"Total edges discovered: {stats['total_edges']}",
            "",
            "I am curious about:",
            "- Which strategy-regime combinations work best",
            "- How volatility affects performance",
            "- What patterns repeat in winning trades",
        ]

        return "\n".join(lines)
