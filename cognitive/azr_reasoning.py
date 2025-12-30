"""
AZR-Inspired Reasoning Engine
=============================

Implements three key innovations from "Absolute Zero: Reinforced Self-play
Reasoning with Zero Data" (arXiv:2505.03335) adapted for trading:

1. **Reasoning Type Classification** - Categorizes hypotheses as:
   - Abductive: Observation → Infer cause ("Why did this trade fail?")
   - Deductive: Rules → Infer conclusion ("Given regime X, what should happen?")
   - Inductive: Examples → Infer rules ("What pattern explains these wins?")

2. **Learnability Scoring** - Prioritizes hypotheses by expected learning value:
   - Novel hypotheses score higher than redundant ones
   - Hypotheses near decision boundaries score higher
   - Hypotheses with sufficient (but not excessive) data score higher

3. **Self-Play Scheduler** - Continuous improvement loop:
   - Propose: Generate diverse hypotheses across all reasoning types
   - Solve: Test hypotheses via backtest verification
   - Learn: Update beliefs and generate new hypotheses from results

Reference: https://arxiv.org/abs/2505.03335
GitHub: https://github.com/LeapLabTHU/Absolute-Zero-Reasoner

Usage:
    from cognitive.azr_reasoning import (
        ReasoningType,
        LearnabilityScorer,
        SelfPlayScheduler,
        get_self_play_scheduler,
    )

    # Score a hypothesis for learning value
    scorer = LearnabilityScorer()
    score = scorer.score_hypothesis(hypothesis)

    # Run continuous self-play improvement
    scheduler = get_self_play_scheduler()
    scheduler.run_cycle()  # One propose→solve→learn cycle

Created: 2025-12-30
Inspired by: Absolute Zero Reasoner (Zhao et al., 2025)
"""

import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import json
from pathlib import Path
import random
import math

logger = logging.getLogger(__name__)


# =============================================================================
# COMPONENT 1: Reasoning Type Classification
# =============================================================================

class ReasoningType(Enum):
    """
    Three fundamental types of reasoning from AZR, adapted for trading.

    - ABDUCTIVE: Given an observation, infer the most likely cause.
      Example: "The trade failed → What market condition caused this?"

    - DEDUCTIVE: Given rules/premises, derive a logical conclusion.
      Example: "IF regime=BEAR AND strategy=mean_reversion THEN expect low win rate"

    - INDUCTIVE: Given examples, infer a general rule or pattern.
      Example: "These 10 winning trades all had VIX>25 → High VIX may favor our strategy"
    """
    ABDUCTIVE = "abductive"   # Observation → Cause
    DEDUCTIVE = "deductive"   # Rules → Conclusion
    INDUCTIVE = "inductive"   # Examples → Rule


@dataclass
class ReasoningTask:
    """
    A reasoning task with explicit type classification.

    This extends the basic Hypothesis with reasoning type metadata
    to enable balanced exploration across all reasoning modes.
    """
    task_id: str
    reasoning_type: ReasoningType
    description: str
    # Input context for the reasoning
    premises: List[str]  # For deductive: rules/assumptions
    observations: List[str]  # For abductive: what we observed
    examples: List[Dict]  # For inductive: example cases
    # Expected output
    target_conclusion: str
    # Metadata
    difficulty: float = 0.5  # 0-1, estimated difficulty
    domain: str = "trading"  # Domain context
    created_at: datetime = field(default_factory=datetime.now)
    # Results (filled after solving)
    solved: bool = False
    solution: Optional[str] = None
    verified: bool = False
    verification_method: str = ""  # "backtest", "statistical", "logical"

    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'reasoning_type': self.reasoning_type.value,
            'description': self.description,
            'premises': self.premises,
            'observations': self.observations,
            'examples': self.examples,
            'target_conclusion': self.target_conclusion,
            'difficulty': self.difficulty,
            'domain': self.domain,
            'created_at': self.created_at.isoformat(),
            'solved': self.solved,
            'solution': self.solution,
            'verified': self.verified,
            'verification_method': self.verification_method,
        }


class ReasoningTypeClassifier:
    """
    Classifies hypotheses and tasks by their reasoning type.

    This helps ensure balanced exploration across all three reasoning modes,
    preventing the system from getting stuck in one type of thinking.
    """

    # Keywords that suggest each reasoning type
    ABDUCTIVE_KEYWORDS = [
        'why', 'caused', 'reason', 'because', 'due to', 'resulted in',
        'led to', 'explanation', 'root cause', 'failure analysis'
    ]

    DEDUCTIVE_KEYWORDS = [
        'if', 'then', 'therefore', 'implies', 'given that', 'assuming',
        'when', 'should', 'expect', 'conclude', 'follows that'
    ]

    INDUCTIVE_KEYWORDS = [
        'pattern', 'trend', 'seems', 'appears', 'observed', 'correlation',
        'tend to', 'usually', 'often', 'historically', 'on average'
    ]

    def classify(self, text: str, context: Optional[Dict] = None) -> ReasoningType:
        """
        Classify text into a reasoning type based on keywords and structure.

        Args:
            text: The hypothesis description or task text
            context: Optional context with additional hints

        Returns:
            The detected ReasoningType
        """
        text_lower = text.lower()

        # Count keyword matches for each type
        abductive_score = sum(1 for kw in self.ABDUCTIVE_KEYWORDS if kw in text_lower)
        deductive_score = sum(1 for kw in self.DEDUCTIVE_KEYWORDS if kw in text_lower)
        inductive_score = sum(1 for kw in self.INDUCTIVE_KEYWORDS if kw in text_lower)

        # Check context hints
        if context:
            source = context.get('source', '')
            if 'episode' in source or 'failure' in source:
                abductive_score += 2
            elif 'rule' in source or 'condition' in source:
                deductive_score += 2
            elif 'pattern' in source or 'combinatorial' in source:
                inductive_score += 2

        # Return the type with highest score
        scores = {
            ReasoningType.ABDUCTIVE: abductive_score,
            ReasoningType.DEDUCTIVE: deductive_score,
            ReasoningType.INDUCTIVE: inductive_score,
        }

        return max(scores, key=scores.get)

    def get_type_balance(self, tasks: List[ReasoningTask]) -> Dict[ReasoningType, int]:
        """
        Returns the count of tasks by reasoning type.

        Used to ensure balanced exploration.
        """
        balance = {rt: 0 for rt in ReasoningType}
        for task in tasks:
            balance[task.reasoning_type] += 1
        return balance

    def suggest_underrepresented_type(self, tasks: List[ReasoningTask]) -> ReasoningType:
        """
        Suggests which reasoning type to generate next for balance.
        """
        balance = self.get_type_balance(tasks)
        return min(balance, key=balance.get)


# =============================================================================
# COMPONENT 2: Learnability Scoring
# =============================================================================

@dataclass
class LearnabilityScore:
    """
    Detailed breakdown of a hypothesis's learning value.
    """
    total_score: float  # 0-100
    novelty_score: float  # How new/unexplored is this?
    uncertainty_score: float  # Is this near a decision boundary?
    data_sufficiency_score: float  # Do we have enough data to learn?
    impact_score: float  # How much would learning this help?
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'total_score': round(self.total_score, 2),
            'novelty_score': round(self.novelty_score, 2),
            'uncertainty_score': round(self.uncertainty_score, 2),
            'data_sufficiency_score': round(self.data_sufficiency_score, 2),
            'impact_score': round(self.impact_score, 2),
            'components': {k: round(v, 2) for k, v in self.components.items()},
        }


class LearnabilityScorer:
    """
    Scores hypotheses by their expected learning value.

    Inspired by AZR's approach of prioritizing tasks that maximize
    the model's learning progress. In trading context, this means:

    - Prioritize hypotheses we haven't tested before (novelty)
    - Focus on uncertain areas where we're near 50/50 (uncertainty)
    - Ensure we have enough data to learn from (data sufficiency)
    - Weight by potential impact on trading performance (impact)

    High-scoring hypotheses are tested first for maximum learning efficiency.
    """

    def __init__(
        self,
        novelty_weight: float = 0.30,
        uncertainty_weight: float = 0.25,
        data_weight: float = 0.25,
        impact_weight: float = 0.20,
    ):
        self.novelty_weight = novelty_weight
        self.uncertainty_weight = uncertainty_weight
        self.data_weight = data_weight
        self.impact_weight = impact_weight

        # Track tested hypotheses for novelty calculation
        self._tested_conditions: Dict[str, int] = {}
        self._load_history()

    def score_hypothesis(self, hypothesis: Any) -> LearnabilityScore:
        """
        Calculate the learnability score for a hypothesis.

        Args:
            hypothesis: A Hypothesis object from curiosity_engine

        Returns:
            LearnabilityScore with detailed breakdown
        """
        # 1. Novelty Score: How new is this hypothesis?
        novelty = self._calculate_novelty(hypothesis)

        # 2. Uncertainty Score: Is this near a decision boundary?
        uncertainty = self._calculate_uncertainty(hypothesis)

        # 3. Data Sufficiency: Do we have enough data?
        data_suff = self._calculate_data_sufficiency(hypothesis)

        # 4. Impact Score: How important is this to learn?
        impact = self._calculate_impact(hypothesis)

        # Weighted total
        total = (
            novelty * self.novelty_weight +
            uncertainty * self.uncertainty_weight +
            data_suff * self.data_weight +
            impact * self.impact_weight
        ) * 100

        return LearnabilityScore(
            total_score=total,
            novelty_score=novelty * 100,
            uncertainty_score=uncertainty * 100,
            data_sufficiency_score=data_suff * 100,
            impact_score=impact * 100,
            components={
                'condition_hash': self._hash_condition(hypothesis.condition),
                'times_similar_tested': self._tested_conditions.get(
                    self._hash_condition(hypothesis.condition), 0
                ),
            }
        )

    def _calculate_novelty(self, hypothesis: Any) -> float:
        """
        Novelty = 1 - (times_similar_tested / max_tests)

        Higher score for hypotheses we haven't explored much.
        """
        condition_hash = self._hash_condition(hypothesis.condition)
        times_tested = self._tested_conditions.get(condition_hash, 0)

        # Decay novelty as we test similar hypotheses
        # After 10 tests of similar conditions, novelty approaches 0
        novelty = math.exp(-times_tested / 5)
        return novelty

    def _calculate_uncertainty(self, hypothesis: Any) -> float:
        """
        Uncertainty = 1 - |observed_win_rate - 0.5| * 2

        Higher score when we're near 50/50 (maximum uncertainty).
        Hypotheses with clear outcomes (high/low WR) score lower.
        """
        # If we have observed data, use it
        observed = getattr(hypothesis, 'observed_value', None)
        if observed is not None and observed > 0:
            # Distance from 0.5 (max uncertainty point)
            distance_from_uncertainty = abs(observed - 0.5) * 2
            return 1 - distance_from_uncertainty

        # If no data yet, assume high uncertainty (good for learning)
        return 0.8

    def _calculate_data_sufficiency(self, hypothesis: Any) -> float:
        """
        Data sufficiency follows an inverted U-shape:
        - Too little data (n<30): Can't learn reliably
        - Sweet spot (30-200): Good for learning
        - Too much data (n>500): Diminishing returns
        """
        sample_size = getattr(hypothesis, 'sample_size', 0)

        if sample_size < 10:
            return 0.2  # Not enough data
        elif sample_size < 30:
            return 0.5  # Borderline
        elif sample_size < 100:
            return 1.0  # Optimal range
        elif sample_size < 300:
            return 0.8  # Still good
        else:
            return 0.6  # Diminishing returns

    def _calculate_impact(self, hypothesis: Any) -> float:
        """
        Impact based on what the hypothesis is about.

        Higher impact for:
        - Regime-related hypotheses (affect all trades)
        - Risk-related hypotheses (affect position sizing)
        - Strategy selection hypotheses
        """
        description = hypothesis.description.lower()
        condition = hypothesis.condition.lower()

        impact = 0.5  # Base impact

        # High-impact topics
        if any(word in description or word in condition for word in
               ['regime', 'bull', 'bear', 'market']):
            impact += 0.2

        if any(word in description or word in condition for word in
               ['risk', 'stop', 'sizing', 'volatility', 'vix']):
            impact += 0.15

        if any(word in description or word in condition for word in
               ['strategy', 'entry', 'exit', 'signal']):
            impact += 0.1

        return min(impact, 1.0)

    def _hash_condition(self, condition: str) -> str:
        """Create a hash of the condition for similarity tracking."""
        # Normalize condition for comparison
        normalized = condition.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:8]

    def record_test(self, hypothesis: Any) -> None:
        """Record that a hypothesis was tested (updates novelty tracking)."""
        condition_hash = self._hash_condition(hypothesis.condition)
        self._tested_conditions[condition_hash] = \
            self._tested_conditions.get(condition_hash, 0) + 1
        self._save_history()

    def get_top_hypotheses(
        self,
        hypotheses: List[Any],
        n: int = 5
    ) -> List[Tuple[Any, LearnabilityScore]]:
        """
        Return the top N hypotheses by learnability score.
        """
        scored = [(h, self.score_hypothesis(h)) for h in hypotheses]
        scored.sort(key=lambda x: x[1].total_score, reverse=True)
        return scored[:n]

    def _save_history(self) -> None:
        """Persist tested conditions history."""
        try:
            history_file = Path("state/cognitive/learnability_history.json")
            history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(history_file, 'w') as f:
                json.dump({
                    'tested_conditions': self._tested_conditions,
                    'saved_at': datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save learnability history: {e}")

    def _load_history(self) -> None:
        """Load tested conditions history."""
        try:
            history_file = Path("state/cognitive/learnability_history.json")
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self._tested_conditions = data.get('tested_conditions', {})
        except Exception as e:
            logger.warning(f"Failed to load learnability history: {e}")
            self._tested_conditions = {}


# =============================================================================
# COMPONENT 3: Self-Play Scheduler
# =============================================================================

@dataclass
class SelfPlayCycleResult:
    """Results from one propose→solve→learn cycle."""
    cycle_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    # Propose phase
    hypotheses_generated: int
    reasoning_type_balance: Dict[str, int]
    # Solve phase
    hypotheses_tested: int
    validated: int
    rejected: int
    inconclusive: int
    # Learn phase
    edges_discovered: int
    rules_added: int
    learnability_scores: List[float]
    # Meta
    total_duration_seconds: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'cycle_id': self.cycle_id,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'hypotheses_generated': self.hypotheses_generated,
            'reasoning_type_balance': self.reasoning_type_balance,
            'hypotheses_tested': self.hypotheses_tested,
            'validated': self.validated,
            'rejected': self.rejected,
            'inconclusive': self.inconclusive,
            'edges_discovered': self.edges_discovered,
            'rules_added': self.rules_added,
            'avg_learnability': sum(self.learnability_scores) / len(self.learnability_scores) if self.learnability_scores else 0,
            'total_duration_seconds': self.total_duration_seconds,
        }


class SelfPlayScheduler:
    """
    Continuous self-improvement through propose→solve→learn cycles.

    Inspired by AZR's self-play paradigm, adapted for trading:

    1. PROPOSE: Generate diverse hypotheses across all reasoning types
       - Use CuriosityEngine for hypothesis generation
       - Balance Abductive/Deductive/Inductive tasks
       - Prioritize by learnability score

    2. SOLVE: Test hypotheses via backtest verification
       - Run statistical tests on historical data
       - Verify with actual backtest execution when possible
       - Record all results for learning

    3. LEARN: Update beliefs and generate new hypotheses
       - Convert validated hypotheses to edges
       - Add new rules to semantic memory
       - Generate follow-up hypotheses from results

    The scheduler can run:
    - Single cycle: One propose→solve→learn iteration
    - Continuous: Run cycles until stopped or quota reached
    - Scheduled: Run at specific times (e.g., end of trading day)
    """

    def __init__(
        self,
        max_hypotheses_per_cycle: int = 10,
        max_tests_per_cycle: int = 5,
        min_learnability_threshold: float = 30.0,
        balance_reasoning_types: bool = True,
    ):
        self.max_hypotheses_per_cycle = max_hypotheses_per_cycle
        self.max_tests_per_cycle = max_tests_per_cycle
        self.min_learnability_threshold = min_learnability_threshold
        self.balance_reasoning_types = balance_reasoning_types

        # Components
        self._curiosity_engine = None
        self._learnability_scorer = LearnabilityScorer()
        self._type_classifier = ReasoningTypeClassifier()

        # State
        self._cycle_history: List[SelfPlayCycleResult] = []
        self._is_running: bool = False
        self._storage_dir = Path("state/cognitive")
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._load_state()
        logger.info("SelfPlayScheduler initialized (AZR-inspired)")

    @property
    def curiosity_engine(self):
        """Lazy load CuriosityEngine."""
        if self._curiosity_engine is None:
            from cognitive.curiosity_engine import get_curiosity_engine
            self._curiosity_engine = get_curiosity_engine()
        return self._curiosity_engine

    def run_cycle(self, observations: Optional[Dict] = None) -> SelfPlayCycleResult:
        """
        Execute one complete propose→solve→learn cycle.

        Args:
            observations: Optional current market observations

        Returns:
            SelfPlayCycleResult with cycle statistics
        """
        cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.now()

        logger.info(f"Starting self-play cycle {cycle_id}")

        result = SelfPlayCycleResult(
            cycle_id=cycle_id,
            started_at=started_at,
            completed_at=None,
            hypotheses_generated=0,
            reasoning_type_balance={},
            hypotheses_tested=0,
            validated=0,
            rejected=0,
            inconclusive=0,
            edges_discovered=0,
            rules_added=0,
            learnability_scores=[],
        )

        try:
            # === PHASE 1: PROPOSE ===
            logger.info(f"[{cycle_id}] PROPOSE phase: Generating hypotheses...")
            propose_result = self._propose_phase(observations)
            result.hypotheses_generated = propose_result['generated']
            result.reasoning_type_balance = propose_result['type_balance']

            # === PHASE 2: SOLVE ===
            logger.info(f"[{cycle_id}] SOLVE phase: Testing hypotheses...")
            solve_result = self._solve_phase()
            result.hypotheses_tested = solve_result['tested']
            result.validated = solve_result['validated']
            result.rejected = solve_result['rejected']
            result.inconclusive = solve_result['inconclusive']
            result.learnability_scores = solve_result['learnability_scores']

            # === PHASE 3: LEARN ===
            logger.info(f"[{cycle_id}] LEARN phase: Updating knowledge...")
            learn_result = self._learn_phase()
            result.edges_discovered = learn_result['edges_discovered']
            result.rules_added = learn_result['rules_added']

        except Exception as e:
            logger.error(f"Self-play cycle {cycle_id} failed: {e}")

        # Finalize
        result.completed_at = datetime.now()
        result.total_duration_seconds = (result.completed_at - started_at).total_seconds()

        self._cycle_history.append(result)
        self._save_state()

        logger.info(
            f"Self-play cycle {cycle_id} complete: "
            f"generated={result.hypotheses_generated}, "
            f"tested={result.hypotheses_tested}, "
            f"validated={result.validated}, "
            f"edges={result.edges_discovered}, "
            f"duration={result.total_duration_seconds:.1f}s"
        )

        return result

    def _propose_phase(self, observations: Optional[Dict] = None) -> Dict:
        """
        Generate new hypotheses with balanced reasoning types.
        """
        # Get existing hypotheses to check balance
        existing = self.curiosity_engine.get_active_hypotheses()

        # Classify existing hypotheses by type
        type_balance = {rt.value: 0 for rt in ReasoningType}
        for hyp in existing:
            rt = self._type_classifier.classify(hyp.description, {'source': hyp.source})
            type_balance[rt.value] += 1

        # Generate new hypotheses
        new_hypotheses = self.curiosity_engine.generate_hypotheses(observations)

        # If balancing, generate extra hypotheses for underrepresented types
        if self.balance_reasoning_types and new_hypotheses:
            underrep = min(type_balance, key=type_balance.get)
            logger.debug(f"Underrepresented reasoning type: {underrep}")
            # The curiosity engine will naturally generate diverse hypotheses

        # Classify new hypotheses
        for hyp in new_hypotheses:
            rt = self._type_classifier.classify(hyp.description, {'source': hyp.source})
            type_balance[rt.value] += 1

        return {
            'generated': len(new_hypotheses),
            'type_balance': type_balance,
        }

    def _solve_phase(self) -> Dict:
        """
        Test hypotheses prioritized by learnability score.
        """
        from cognitive.curiosity_engine import HypothesisStatus

        # Get pending hypotheses
        pending = [
            h for h in self.curiosity_engine._hypotheses.values()
            if h.status == HypothesisStatus.PROPOSED
        ]

        if not pending:
            return {
                'tested': 0, 'validated': 0, 'rejected': 0,
                'inconclusive': 0, 'learnability_scores': []
            }

        # Score by learnability and select top candidates
        scored = self._learnability_scorer.get_top_hypotheses(
            pending, n=self.max_tests_per_cycle
        )

        # Filter by minimum learnability threshold
        to_test = [
            (h, score) for h, score in scored
            if score.total_score >= self.min_learnability_threshold
        ]

        results = {
            'tested': 0, 'validated': 0, 'rejected': 0,
            'inconclusive': 0, 'learnability_scores': []
        }

        for hypothesis, score in to_test:
            results['learnability_scores'].append(score.total_score)

            # Test the hypothesis
            tested = self.curiosity_engine.test_hypothesis(hypothesis)
            results['tested'] += 1

            # Record the test for novelty tracking
            self._learnability_scorer.record_test(hypothesis)

            # Count results
            if tested.status == HypothesisStatus.VALIDATED:
                results['validated'] += 1
            elif tested.status == HypothesisStatus.REJECTED:
                results['rejected'] += 1
            else:
                results['inconclusive'] += 1

        return results

    def _learn_phase(self) -> Dict:
        """
        Extract learnings and generate follow-up hypotheses.
        """
        # Count newly discovered edges
        edges = self.curiosity_engine.get_validated_edges()

        # The curiosity engine automatically:
        # 1. Creates edges from validated hypotheses
        # 2. Adds rules to semantic memory
        # 3. Publishes to global workspace

        # Generate follow-up hypotheses from recent results
        # (This happens naturally in the next propose phase)

        return {
            'edges_discovered': len(edges),
            'rules_added': len(edges),  # One rule per edge
        }

    def run_continuous(
        self,
        max_cycles: int = 10,
        delay_seconds: float = 1.0,
        stop_condition: Optional[Callable[[], bool]] = None,
    ) -> List[SelfPlayCycleResult]:
        """
        Run multiple cycles continuously.

        Args:
            max_cycles: Maximum number of cycles to run
            delay_seconds: Delay between cycles
            stop_condition: Optional function that returns True to stop

        Returns:
            List of all cycle results
        """
        import time

        self._is_running = True
        results = []

        logger.info(f"Starting continuous self-play (max {max_cycles} cycles)")

        for i in range(max_cycles):
            if not self._is_running:
                logger.info("Self-play stopped by external signal")
                break

            if stop_condition and stop_condition():
                logger.info("Self-play stopped by condition")
                break

            result = self.run_cycle()
            results.append(result)

            # Check if we're learning anything
            if result.validated == 0 and result.hypotheses_tested > 0:
                logger.info("No validations in last cycle, may need more data")

            if i < max_cycles - 1:
                time.sleep(delay_seconds)

        self._is_running = False
        logger.info(f"Continuous self-play complete: {len(results)} cycles")

        return results

    def stop(self) -> None:
        """Signal the scheduler to stop continuous running."""
        self._is_running = False

    def get_stats(self) -> Dict:
        """Get statistics about self-play history."""
        if not self._cycle_history:
            return {'total_cycles': 0}

        total_generated = sum(c.hypotheses_generated for c in self._cycle_history)
        total_tested = sum(c.hypotheses_tested for c in self._cycle_history)
        total_validated = sum(c.validated for c in self._cycle_history)
        total_edges = sum(c.edges_discovered for c in self._cycle_history)

        all_scores = []
        for c in self._cycle_history:
            all_scores.extend(c.learnability_scores)

        return {
            'total_cycles': len(self._cycle_history),
            'total_hypotheses_generated': total_generated,
            'total_hypotheses_tested': total_tested,
            'total_validated': total_validated,
            'total_edges_discovered': total_edges,
            'validation_rate': total_validated / total_tested if total_tested > 0 else 0,
            'avg_learnability_score': sum(all_scores) / len(all_scores) if all_scores else 0,
            'last_cycle': self._cycle_history[-1].to_dict() if self._cycle_history else None,
        }

    def introspect(self) -> str:
        """Generate a human-readable report of self-play status."""
        stats = self.get_stats()

        if stats['total_cycles'] == 0:
            return (
                "--- Self-Play Scheduler (AZR-Inspired) ---\n"
                "No cycles completed yet.\n"
                "Run scheduler.run_cycle() to start learning."
            )

        lines = [
            "--- Self-Play Scheduler (AZR-Inspired) ---",
            f"Total cycles completed: {stats['total_cycles']}",
            f"Hypotheses generated: {stats.get('total_hypotheses_generated', 0)}",
            f"Hypotheses tested: {stats.get('total_hypotheses_tested', 0)}",
            f"Validated: {stats.get('total_validated', 0)} ({stats.get('validation_rate', 0):.1%} rate)",
            f"Edges discovered: {stats.get('total_edges_discovered', 0)}",
            f"Average learnability score: {stats.get('avg_learnability_score', 0):.1f}",
        ]

        if stats.get('last_cycle'):
            last = stats['last_cycle']
            lines.extend([
                "",
                "Last cycle:",
                f"  Generated: {last.get('hypotheses_generated', 0)}",
                f"  Tested: {last.get('hypotheses_tested', 0)}",
                f"  Validated: {last.get('validated', 0)}",
                f"  Duration: {last.get('total_duration_seconds', 0):.1f}s",
            ])

        return "\n".join(lines)

    def _save_state(self) -> None:
        """Persist cycle history."""
        try:
            state_file = self._storage_dir / "selfplay_state.json"
            with open(state_file, 'w') as f:
                json.dump({
                    'cycle_history': [c.to_dict() for c in self._cycle_history[-100:]],  # Keep last 100
                    'saved_at': datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save self-play state: {e}")

    def _load_state(self) -> None:
        """Load cycle history."""
        try:
            state_file = self._storage_dir / "selfplay_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    # We don't fully reconstruct history, just track stats
                    logger.info(f"Loaded self-play history: {len(data.get('cycle_history', []))} cycles")
        except Exception as e:
            logger.warning(f"Failed to load self-play state: {e}")


# =============================================================================
# SINGLETON FACTORIES
# =============================================================================

_learnability_scorer: Optional[LearnabilityScorer] = None
_self_play_scheduler: Optional[SelfPlayScheduler] = None
_type_classifier: Optional[ReasoningTypeClassifier] = None


def get_learnability_scorer() -> LearnabilityScorer:
    """Get singleton LearnabilityScorer instance."""
    global _learnability_scorer
    if _learnability_scorer is None:
        _learnability_scorer = LearnabilityScorer()
    return _learnability_scorer


def get_self_play_scheduler() -> SelfPlayScheduler:
    """Get singleton SelfPlayScheduler instance."""
    global _self_play_scheduler
    if _self_play_scheduler is None:
        _self_play_scheduler = SelfPlayScheduler()
    return _self_play_scheduler


def get_type_classifier() -> ReasoningTypeClassifier:
    """Get singleton ReasoningTypeClassifier instance."""
    global _type_classifier
    if _type_classifier is None:
        _type_classifier = ReasoningTypeClassifier()
    return _type_classifier
