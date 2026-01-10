"""
RAG Evaluator for Kobe's Cognitive Brain.

Measures and improves LLM reasoning quality for trade decisions:
1. Multiple retriever comparison (episodic vs semantic memory)
2. Explanation quality scoring
3. A/B testing of reasoning approaches
4. Feedback loop for continuous improvement

Created: 2026-01-07
Based on: Firecrawl rag-arena patterns
"""

from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class RetrieverType(str, Enum):
    """Types of knowledge retrievers."""
    EPISODIC = "episodic"  # Past trade experiences
    SEMANTIC = "semantic"  # General rules and knowledge
    PATTERN = "pattern"  # Historical price patterns
    NEWS = "news"  # Market news and sentiment
    TECHNICAL = "technical"  # Technical analysis
    HYBRID = "hybrid"  # Combined retrieval


class ExplanationQuality(str, Enum):
    """Quality tiers for explanations."""
    EXCELLENT = "excellent"  # Score >= 90
    GOOD = "good"  # Score >= 75
    ACCEPTABLE = "acceptable"  # Score >= 60
    POOR = "poor"  # Score >= 40
    UNACCEPTABLE = "unacceptable"  # Score < 40


@dataclass
class RetrievalContext:
    """Context retrieved for a trade decision."""
    retriever_type: RetrieverType
    documents: List[Dict[str, Any]] = field(default_factory=list)
    relevance_scores: List[float] = field(default_factory=list)
    retrieval_time_ms: float = 0.0

    @property
    def avg_relevance(self) -> float:
        """Average relevance score."""
        return np.mean(self.relevance_scores) if self.relevance_scores else 0.0


@dataclass
class TradeExplanation:
    """Generated explanation for a trade decision."""
    explanation_id: str
    retriever_type: RetrieverType
    context: RetrievalContext
    explanation_text: str
    confidence: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.explanation_id:
            self.explanation_id = hashlib.sha256(
                f"{self.timestamp}_{self.explanation_text[:50]}".encode()
            ).hexdigest()[:12]


@dataclass
class ExplanationEvaluation:
    """Evaluation of an explanation's quality."""
    explanation_id: str

    # Automated scores (0-100)
    completeness_score: float = 0.0  # Covers all relevant factors
    coherence_score: float = 0.0  # Logical flow
    accuracy_score: float = 0.0  # Factually correct
    actionability_score: float = 0.0  # Clear recommendations
    confidence_calibration: float = 0.0  # Confidence matches reality

    # Human feedback
    human_rating: Optional[int] = None  # 1-5 stars
    human_feedback: str = ""

    # Outcome tracking
    trade_outcome: Optional[float] = None  # P&L if trade was taken
    prediction_correct: Optional[bool] = None

    @property
    def overall_score(self) -> float:
        """Weighted overall score."""
        weights = {
            "completeness": 0.25,
            "coherence": 0.20,
            "accuracy": 0.25,
            "actionability": 0.15,
            "calibration": 0.15,
        }
        return (
            weights["completeness"] * self.completeness_score +
            weights["coherence"] * self.coherence_score +
            weights["accuracy"] * self.accuracy_score +
            weights["actionability"] * self.actionability_score +
            weights["calibration"] * self.confidence_calibration
        )

    @property
    def quality_tier(self) -> ExplanationQuality:
        """Get quality tier from score."""
        score = self.overall_score
        if score >= 90:
            return ExplanationQuality.EXCELLENT
        elif score >= 75:
            return ExplanationQuality.GOOD
        elif score >= 60:
            return ExplanationQuality.ACCEPTABLE
        elif score >= 40:
            return ExplanationQuality.POOR
        else:
            return ExplanationQuality.UNACCEPTABLE


@dataclass
class ABTestResult:
    """Result of A/B testing two retrieval approaches."""
    test_id: str
    retriever_a: RetrieverType
    retriever_b: RetrieverType

    # Sample sizes
    n_a: int = 0
    n_b: int = 0

    # Scores
    avg_score_a: float = 0.0
    avg_score_b: float = 0.0

    # Human preferences
    preferred_a: int = 0
    preferred_b: int = 0
    no_preference: int = 0

    # Statistical significance
    p_value: float = 1.0
    effect_size: float = 0.0
    is_significant: bool = False

    @property
    def winner(self) -> Optional[RetrieverType]:
        """Determine winner if significant."""
        if not self.is_significant:
            return None
        return self.retriever_a if self.avg_score_a > self.avg_score_b else self.retriever_b


# =============================================================================
# RAG EVALUATOR
# =============================================================================

class RAGEvaluator:
    """
    Evaluator for RAG-based trade explanations.

    Compares different retrieval strategies and measures
    explanation quality over time.

    Usage:
        evaluator = RAGEvaluator()

        # Generate explanations from different retrievers
        explanations = evaluator.generate_explanations(trade_context)

        # Evaluate quality
        for exp in explanations:
            eval_result = evaluator.evaluate_explanation(exp)

        # Record human feedback
        evaluator.record_human_feedback(explanation_id, rating=4, feedback="Clear reasoning")

        # Run A/B test
        result = evaluator.run_ab_test(RetrieverType.EPISODIC, RetrieverType.SEMANTIC)
    """

    def __init__(self, state_dir: Optional[str] = None):
        self.state_dir = Path(state_dir or "state/rag_evaluation")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.explanations: Dict[str, TradeExplanation] = {}
        self.evaluations: Dict[str, ExplanationEvaluation] = {}
        self.ab_tests: Dict[str, ABTestResult] = {}

        self._retrievers: Dict[RetrieverType, Callable] = {}
        self._load_state()

        logger.info("RAGEvaluator initialized")

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def _load_state(self):
        """Load state from disk."""
        eval_path = self.state_dir / "evaluations.json"
        if eval_path.exists():
            try:
                with open(eval_path, "r") as f:
                    data = json.load(f)
                    # Simplified loading
                    self.evaluations = {k: self._dict_to_evaluation(v) for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Failed to load evaluations: {e}")

    def _save_state(self):
        """Save state to disk."""
        eval_path = self.state_dir / "evaluations.json"
        try:
            with open(eval_path, "w") as f:
                data = {k: self._evaluation_to_dict(v) for k, v in self.evaluations.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save evaluations: {e}")

    def _evaluation_to_dict(self, eval: ExplanationEvaluation) -> Dict:
        """Convert evaluation to dict."""
        return {
            "explanation_id": eval.explanation_id,
            "completeness_score": eval.completeness_score,
            "coherence_score": eval.coherence_score,
            "accuracy_score": eval.accuracy_score,
            "actionability_score": eval.actionability_score,
            "confidence_calibration": eval.confidence_calibration,
            "human_rating": eval.human_rating,
            "human_feedback": eval.human_feedback,
            "trade_outcome": eval.trade_outcome,
            "prediction_correct": eval.prediction_correct,
        }

    def _dict_to_evaluation(self, d: Dict) -> ExplanationEvaluation:
        """Convert dict to evaluation."""
        return ExplanationEvaluation(**d)

    # -------------------------------------------------------------------------
    # Retriever Registration
    # -------------------------------------------------------------------------

    def register_retriever(
        self,
        retriever_type: RetrieverType,
        retriever_func: Callable[[Dict], RetrievalContext],
    ):
        """Register a retrieval function."""
        self._retrievers[retriever_type] = retriever_func
        logger.info(f"Registered retriever: {retriever_type}")

    def _get_default_retrievers(self) -> Dict[RetrieverType, Callable]:
        """Get default retriever implementations."""
        retrievers = {}

        # Episodic memory retriever
        def episodic_retriever(context: Dict) -> RetrievalContext:
            try:
                from cognitive.episodic_memory import get_episodic_memory
                memory = get_episodic_memory()
                episodes = memory.recall_similar(
                    context.get("symbol", ""),
                    limit=5,
                )
                return RetrievalContext(
                    retriever_type=RetrieverType.EPISODIC,
                    documents=[{"episode": e} for e in episodes],
                    relevance_scores=[0.8] * len(episodes),
                )
            except ImportError:
                return RetrievalContext(retriever_type=RetrieverType.EPISODIC)

        retrievers[RetrieverType.EPISODIC] = episodic_retriever

        # Semantic memory retriever
        def semantic_retriever(context: Dict) -> RetrievalContext:
            try:
                from cognitive.semantic_memory import get_semantic_memory
                memory = get_semantic_memory()
                rules = memory.get_relevant_rules(
                    symbol=context.get("symbol", ""),
                    strategy=context.get("strategy", ""),
                )
                return RetrievalContext(
                    retriever_type=RetrieverType.SEMANTIC,
                    documents=[{"rule": r} for r in rules],
                    relevance_scores=[0.9] * len(rules),
                )
            except ImportError:
                return RetrievalContext(retriever_type=RetrieverType.SEMANTIC)

        retrievers[RetrieverType.SEMANTIC] = semantic_retriever

        # Pattern retriever
        def pattern_retriever(context: Dict) -> RetrievalContext:
            try:
                from analysis.historical_patterns import analyze_consecutive_days
                symbol = context.get("symbol", "SPY")
                patterns = analyze_consecutive_days(symbol)
                return RetrievalContext(
                    retriever_type=RetrieverType.PATTERN,
                    documents=[patterns] if patterns else [],
                    relevance_scores=[0.85] if patterns else [],
                )
            except ImportError:
                return RetrievalContext(retriever_type=RetrieverType.PATTERN)

        retrievers[RetrieverType.PATTERN] = pattern_retriever

        return retrievers

    # -------------------------------------------------------------------------
    # Explanation Generation
    # -------------------------------------------------------------------------

    def generate_explanations(
        self,
        trade_context: Dict[str, Any],
        retriever_types: Optional[List[RetrieverType]] = None,
    ) -> List[TradeExplanation]:
        """
        Generate explanations using different retrievers.

        Args:
            trade_context: Context for the trade decision
            retriever_types: List of retriever types to use

        Returns:
            List of TradeExplanation objects
        """
        if retriever_types is None:
            retriever_types = [RetrieverType.EPISODIC, RetrieverType.SEMANTIC, RetrieverType.PATTERN]

        # Ensure we have retrievers
        if not self._retrievers:
            self._retrievers = self._get_default_retrievers()

        explanations = []

        for rt in retriever_types:
            if rt not in self._retrievers:
                logger.warning(f"No retriever registered for {rt}")
                continue

            try:
                # Retrieve context
                context = self._retrievers[rt](trade_context)

                # Generate explanation
                explanation_text = self._generate_explanation_text(trade_context, context)

                explanation = TradeExplanation(
                    explanation_id="",  # Will be set in __post_init__
                    retriever_type=rt,
                    context=context,
                    explanation_text=explanation_text,
                    confidence=self._estimate_confidence(context),
                )

                self.explanations[explanation.explanation_id] = explanation
                explanations.append(explanation)

            except Exception as e:
                logger.error(f"Failed to generate explanation with {rt}: {e}")

        return explanations

    def _generate_explanation_text(
        self,
        trade_context: Dict,
        retrieval_context: RetrievalContext,
    ) -> str:
        """Generate explanation text from context."""
        symbol = trade_context.get("symbol", "UNKNOWN")
        side = trade_context.get("side", "long")
        score = trade_context.get("score", 0)

        # Build explanation based on retriever type
        parts = [f"Trade Recommendation: {side.upper()} {symbol}"]
        parts.append(f"Signal Score: {score}")
        parts.append("")

        if retrieval_context.retriever_type == RetrieverType.EPISODIC:
            parts.append("Based on Similar Past Trades:")
            for doc in retrieval_context.documents[:3]:
                episode = doc.get("episode", {})
                if isinstance(episode, dict):
                    parts.append(f"  - {episode.get('symbol', 'N/A')}: {episode.get('outcome', 'N/A')}")

        elif retrieval_context.retriever_type == RetrieverType.SEMANTIC:
            parts.append("Based on Trading Rules:")
            for doc in retrieval_context.documents[:3]:
                rule = doc.get("rule", {})
                if isinstance(rule, dict):
                    parts.append(f"  - {rule.get('name', 'Rule')}: {rule.get('description', 'N/A')}")

        elif retrieval_context.retriever_type == RetrieverType.PATTERN:
            parts.append("Based on Historical Patterns:")
            for doc in retrieval_context.documents[:3]:
                if isinstance(doc, dict):
                    parts.append(f"  - Pattern found with {doc.get('sample_size', 'N/A')} samples")

        parts.append("")
        parts.append(f"Confidence: {retrieval_context.avg_relevance:.1%}")

        return "\n".join(parts)

    def _estimate_confidence(self, context: RetrievalContext) -> float:
        """Estimate confidence based on retrieval quality."""
        if not context.documents:
            return 0.3

        # Base confidence on relevance and document count
        base = context.avg_relevance
        doc_factor = min(1.0, len(context.documents) / 5)

        return base * 0.7 + doc_factor * 0.3

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def evaluate_explanation(
        self,
        explanation: TradeExplanation,
        ground_truth: Optional[Dict] = None,
    ) -> ExplanationEvaluation:
        """
        Evaluate an explanation's quality.

        Args:
            explanation: The explanation to evaluate
            ground_truth: Optional ground truth for accuracy checking

        Returns:
            ExplanationEvaluation with scores
        """
        evaluation = ExplanationEvaluation(explanation_id=explanation.explanation_id)

        # Completeness: Does it cover key factors?
        evaluation.completeness_score = self._score_completeness(explanation)

        # Coherence: Is it logically structured?
        evaluation.coherence_score = self._score_coherence(explanation)

        # Accuracy: Is the information correct?
        evaluation.accuracy_score = self._score_accuracy(explanation, ground_truth)

        # Actionability: Can user act on this?
        evaluation.actionability_score = self._score_actionability(explanation)

        # Confidence calibration
        evaluation.confidence_calibration = self._score_calibration(explanation)

        # Store
        self.evaluations[explanation.explanation_id] = evaluation
        self._save_state()

        return evaluation

    def _score_completeness(self, explanation: TradeExplanation) -> float:
        """Score explanation completeness (0-100)."""
        text = explanation.explanation_text.lower()

        # Check for key elements
        elements = {
            "symbol": any(w in text for w in ["symbol", "stock", "ticker"]),
            "direction": any(w in text for w in ["long", "short", "buy", "sell"]),
            "reasoning": len(text) > 100,
            "confidence": "confidence" in text or "score" in text,
            "risk": any(w in text for w in ["risk", "stop", "loss"]),
        }

        score = sum(elements.values()) / len(elements) * 100
        return score

    def _score_coherence(self, explanation: TradeExplanation) -> float:
        """Score explanation coherence (0-100)."""
        text = explanation.explanation_text

        # Check structure
        has_structure = "\n" in text  # Has line breaks
        has_sections = any(c in text for c in [":", "-", "*"])  # Has formatting

        # Check length (not too short, not too long)
        length_score = min(1.0, len(text) / 500) * 0.5 + (1 - min(1.0, len(text) / 2000)) * 0.5

        base_score = 50
        if has_structure:
            base_score += 25
        if has_sections:
            base_score += 25

        return base_score * length_score

    def _score_accuracy(
        self,
        explanation: TradeExplanation,
        ground_truth: Optional[Dict],
    ) -> float:
        """Score explanation accuracy (0-100)."""
        if ground_truth is None:
            # Can't verify without ground truth
            return 70  # Neutral score

        # Check if mentioned facts match ground truth
        text = explanation.explanation_text.lower()
        matches = 0
        total = 0

        for key, value in ground_truth.items():
            if str(value).lower() in text:
                matches += 1
            total += 1

        if total == 0:
            return 70

        return (matches / total) * 100

    def _score_actionability(self, explanation: TradeExplanation) -> float:
        """Score explanation actionability (0-100)."""
        text = explanation.explanation_text.lower()

        # Check for actionable elements
        has_direction = any(w in text for w in ["buy", "sell", "long", "short"])
        has_entry = any(w in text for w in ["entry", "price", "at"])
        has_stop = any(w in text for w in ["stop", "loss", "exit"])
        has_target = any(w in text for w in ["target", "profit", "take"])

        score = 0
        if has_direction:
            score += 40
        if has_entry:
            score += 20
        if has_stop:
            score += 20
        if has_target:
            score += 20

        return score

    def _score_calibration(self, explanation: TradeExplanation) -> float:
        """Score confidence calibration (0-100)."""
        # Check if confidence in explanation matches retrieval quality
        retrieval_conf = explanation.context.avg_relevance
        stated_conf = explanation.confidence

        # Perfect calibration = stated matches retrieval
        diff = abs(retrieval_conf - stated_conf)
        return max(0, 100 - diff * 100)

    # -------------------------------------------------------------------------
    # Human Feedback
    # -------------------------------------------------------------------------

    def record_human_feedback(
        self,
        explanation_id: str,
        rating: int,
        feedback: str = "",
    ):
        """
        Record human feedback for an explanation.

        Args:
            explanation_id: ID of the explanation
            rating: 1-5 star rating
            feedback: Optional text feedback
        """
        if explanation_id not in self.evaluations:
            # Create new evaluation
            self.evaluations[explanation_id] = ExplanationEvaluation(
                explanation_id=explanation_id
            )

        self.evaluations[explanation_id].human_rating = max(1, min(5, rating))
        self.evaluations[explanation_id].human_feedback = feedback
        self._save_state()

        logger.info(f"Recorded feedback for {explanation_id}: {rating} stars")

    def record_trade_outcome(
        self,
        explanation_id: str,
        pnl: float,
        prediction_correct: bool,
    ):
        """Record actual trade outcome for an explanation."""
        if explanation_id in self.evaluations:
            self.evaluations[explanation_id].trade_outcome = pnl
            self.evaluations[explanation_id].prediction_correct = prediction_correct
            self._save_state()

    # -------------------------------------------------------------------------
    # A/B Testing
    # -------------------------------------------------------------------------

    def run_ab_test(
        self,
        retriever_a: RetrieverType,
        retriever_b: RetrieverType,
        min_samples: int = 30,
    ) -> ABTestResult:
        """
        Run A/B test comparing two retrieval approaches.

        Args:
            retriever_a: First retriever type
            retriever_b: Second retriever type
            min_samples: Minimum samples per group

        Returns:
            ABTestResult with statistical analysis
        """
        test_id = f"{retriever_a.value}_vs_{retriever_b.value}_{datetime.now().strftime('%Y%m%d')}"

        # Collect scores for each retriever
        scores_a = []
        scores_b = []
        prefs_a = 0
        prefs_b = 0

        for exp_id, exp in self.explanations.items():
            if exp_id not in self.evaluations:
                continue

            eval_result = self.evaluations[exp_id]

            if exp.retriever_type == retriever_a:
                scores_a.append(eval_result.overall_score)
                if eval_result.human_rating:
                    prefs_a += eval_result.human_rating

            elif exp.retriever_type == retriever_b:
                scores_b.append(eval_result.overall_score)
                if eval_result.human_rating:
                    prefs_b += eval_result.human_rating

        result = ABTestResult(
            test_id=test_id,
            retriever_a=retriever_a,
            retriever_b=retriever_b,
            n_a=len(scores_a),
            n_b=len(scores_b),
            avg_score_a=np.mean(scores_a) if scores_a else 0,
            avg_score_b=np.mean(scores_b) if scores_b else 0,
            preferred_a=prefs_a,
            preferred_b=prefs_b,
        )

        # Statistical test if enough samples
        if len(scores_a) >= min_samples and len(scores_b) >= min_samples:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
            result.p_value = p_value
            result.is_significant = p_value < 0.05

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)
            if pooled_std > 0:
                result.effect_size = (result.avg_score_a - result.avg_score_b) / pooled_std

        self.ab_tests[test_id] = result
        return result

    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------

    def get_retriever_stats(self) -> Dict[RetrieverType, Dict[str, float]]:
        """Get performance stats for each retriever type."""
        stats = {}

        for rt in RetrieverType:
            scores = []
            ratings = []

            for exp_id, exp in self.explanations.items():
                if exp.retriever_type != rt:
                    continue
                if exp_id not in self.evaluations:
                    continue

                eval_result = self.evaluations[exp_id]
                scores.append(eval_result.overall_score)
                if eval_result.human_rating:
                    ratings.append(eval_result.human_rating)

            if scores:
                stats[rt] = {
                    "n_samples": len(scores),
                    "avg_score": np.mean(scores),
                    "std_score": np.std(scores),
                    "avg_human_rating": np.mean(ratings) if ratings else 0,
                }

        return stats

    def generate_report(self) -> str:
        """Generate evaluation report."""
        lines = [
            "=" * 60,
            "RAG EVALUATOR REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            "",
            "RETRIEVER PERFORMANCE:",
            "-" * 40,
        ]

        stats = self.get_retriever_stats()
        for rt, rt_stats in stats.items():
            lines.append(f"\n{rt.value.upper()}:")
            lines.append(f"  Samples: {rt_stats['n_samples']}")
            lines.append(f"  Avg Score: {rt_stats['avg_score']:.1f}")
            lines.append(f"  Std Score: {rt_stats['std_score']:.1f}")
            if rt_stats['avg_human_rating'] > 0:
                lines.append(f"  Avg Human Rating: {rt_stats['avg_human_rating']:.1f}/5")

        if self.ab_tests:
            lines.append("")
            lines.append("A/B TEST RESULTS:")
            lines.append("-" * 40)
            for test_id, result in self.ab_tests.items():
                lines.append(f"\n{test_id}:")
                lines.append(f"  {result.retriever_a.value}: {result.avg_score_a:.1f} (n={result.n_a})")
                lines.append(f"  {result.retriever_b.value}: {result.avg_score_b:.1f} (n={result.n_b})")
                if result.is_significant:
                    winner = result.winner
                    lines.append(f"  WINNER: {winner.value if winner else 'None'} (p={result.p_value:.3f})")
                else:
                    lines.append(f"  No significant difference (p={result.p_value:.3f})")

        return "\n".join(lines)


# =============================================================================
# SINGLETON
# =============================================================================

_rag_evaluator: Optional[RAGEvaluator] = None


def get_rag_evaluator() -> RAGEvaluator:
    """Get singleton RAGEvaluator instance."""
    global _rag_evaluator
    if _rag_evaluator is None:
        _rag_evaluator = RAGEvaluator()
    return _rag_evaluator


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    # Demo
    print("RAG Evaluator Demo")
    print("=" * 60)

    evaluator = get_rag_evaluator()

    # Generate sample explanation
    trade_context = {
        "symbol": "AAPL",
        "side": "long",
        "score": 75,
        "entry_price": 185.50,
        "stop_loss": 182.00,
        "take_profit": 195.00,
    }

    print("\nGenerating explanations for trade context...")
    explanations = evaluator.generate_explanations(trade_context)

    for exp in explanations:
        print(f"\n--- {exp.retriever_type.value.upper()} ---")
        print(exp.explanation_text[:300] + "..." if len(exp.explanation_text) > 300 else exp.explanation_text)

        # Evaluate
        eval_result = evaluator.evaluate_explanation(exp)
        print(f"\nEvaluation:")
        print(f"  Overall Score: {eval_result.overall_score:.1f}")
        print(f"  Quality Tier: {eval_result.quality_tier.value}")

    print("\n" + evaluator.generate_report())
