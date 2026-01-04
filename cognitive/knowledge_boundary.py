"""
Knowledge Boundary - The AI's Self-Awareness Layer
=====================================================

This module provides the cognitive architecture with a critical component of
self-awareness: the ability to recognize the limits of its own knowledge. It
acts as an anti-overconfidence layer, preventing the AI from making high-risk
decisions in situations it doesn't fully understand.

Core Functions:
- **Detects Uncertainty:** Assesses a given trading situation and identifies
  various sources of uncertainty (e.g., is this a new market regime? Is my
  data stale? Do my internal models disagree?).
- **Quantifies Doubt:** Calculates an "uncertainty score" and recommends a
  course of action, such as standing down or reducing position size.
- **Asks "What Would Change My Mind?":** Proactively generates a list of
  "invalidators"—conditions that, if they became true, would require
  reconsidering the current decision.

This is a key part of the "System 2" (slow, deliberate thinking) process,
forcing the AI to pause and evaluate its own competence before acting.

Based on AI safety and self-awareness research.
- See: "A Survey of Self-Awareness in Autonomous Agents"

Usage:
    from cognitive.knowledge_boundary import KnowledgeBoundary

    kb = KnowledgeBoundary()

    # During deliberation, the brain assesses its knowledge state.
    assessment = kb.assess_knowledge_state(signal, context)

    if assessment.is_uncertain:
        if assessment.should_stand_down:
            # High uncertainty, abort the decision.
            return "STAND DOWN"
        else:
            # Moderate uncertainty, proceed with caution.
            confidence = confidence + assessment.confidence_adjustment

    # The brain can also ask for its blind spots.
    invalidators = kb.what_would_change_mind(signal, context)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class UncertaintyLevel(Enum):
    """Enumerates the perceived level of uncertainty for a decision."""
    LOW = "low"           # A familiar, well-understood situation.
    MODERATE = "moderate" # Some unknowns are present, but they are manageable.
    HIGH = "high"         # Significant uncertainty; proceed with extreme caution.
    EXTREME = "extreme"   # Operating blind; standing down is strongly advised.


class UncertaintySource(Enum):
    """Categorizes the specific sources of identified uncertainty."""
    NOVEL_REGIME = "novel_regime"
    NOVEL_SYMBOL = "novel_symbol"
    MISSING_DATA = "missing_data"
    MODEL_DISAGREEMENT = "model_disagreement"
    LOW_SAMPLE_SIZE = "low_sample_size"
    REGIME_TRANSITION = "regime_transition"
    CONFLICTING_SIGNALS = "conflicting_signals"
    STALE_DATA = "stale_data"
    UNUSUAL_VOLATILITY = "unusual_volatility"
    EXTREME_SENTIMENT = "extreme_sentiment"  # Very high or low news/market sentiment
    EXTREME_MARKET_MOOD = "extreme_market_mood"  # VIX + sentiment combined extreme
    EDGE_CASE = "edge_case" # e.g., an unusually wide stop-loss


@dataclass
class Invalidator:
    """
    Represents a condition that would "change the AI's mind" about a decision.
    This is a structured representation of a specific doubt.
    """
    description: str  # Human-readable description (e.g., "VIX spikes above 35").
    data_needed: str  # The piece of data required to check this invalidator.
    check_method: str # A key identifying the logic needed to perform the check.
    importance: float   # How critical this invalidator is (0.0 to 1.0).
    time_sensitive: bool = False # Is this invalidator urgent?


@dataclass
class KnowledgeAssessment:
    """
    A structured report summarizing the AI's self-assessed knowledge state
    for a given decision.
    """
    uncertainty_level: UncertaintyLevel
    uncertainty_sources: List[UncertaintySource]
    is_uncertain: bool # A simple flag indicating if uncertainty exceeds a threshold.
    should_stand_down: bool # A flag indicating if uncertainty is critically high.
    confidence_adjustment: float # A recommended penalty to apply to the final confidence score.
    missing_information: List[str] # A list of specific data points the AI knows it's missing.
    invalidators: List[Invalidator] # A list of conditions that would invalidate the decision.
    recommendations: List[str] # Actionable advice (e.g., "Reduce position size").
    metadata: Dict[str, Any] = field(default_factory=dict) # Additional data like uncertainty_score.
    
    def to_dict(self) -> Dict:
        """Serializes the assessment to a dictionary."""
        return {
            'uncertainty_level': self.uncertainty_level.value,
            'uncertainty_sources': [s.value for s in self.uncertainty_sources],
            'is_uncertain': self.is_uncertain,
            'should_stand_down': self.should_stand_down,
            'confidence_adjustment': self.confidence_adjustment,
            'missing_information': self.missing_information,
            'invalidators': [{'description': inv.description, 'data_needed': inv.data_needed} for inv in self.invalidators],
            'recommendations': self.recommendations,
        }


class KnowledgeBoundary:
    """
    This class is responsible for detecting when the AI is operating at or
    beyond the boundaries of its knowledge and expertise. It's a metacognitive
    function that promotes safer, more robust decision-making.
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.5,
        stand_down_threshold: float = 0.8,
        min_samples_for_confidence: int = 20,
        max_data_age_hours: int = 4,
    ):
        """
        Initializes the KnowledgeBoundary detector.

        Args:
            uncertainty_threshold: The score above which a situation is deemed "uncertain".
            stand_down_threshold: The score above which a stand-down is recommended.
            min_samples_for_confidence: The minimum number of past trades required to
                                        feel confident in a given context.
            max_data_age_hours: The maximum age of market data before it's considered stale.
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.stand_down_threshold = stand_down_threshold
        self.min_samples_for_confidence = min_samples_for_confidence
        self.max_data_age_hours = max_data_age_hours

        # Lazy-loaded dependencies to other cognitive components.
        self._self_model = None
        self._episodic_memory = None
        self._workspace = None

        logger.info("KnowledgeBoundary initialized.")

    @property
    def self_model(self):
        if self._self_model is None: from cognitive.self_model import get_self_model; self._self_model = get_self_model()
        return self._self_model

    @property
    def episodic_memory(self):
        if self._episodic_memory is None: from cognitive.episodic_memory import get_episodic_memory; self._episodic_memory = get_episodic_memory()
        return self._episodic_memory

    @property
    def workspace(self):
        if self._workspace is None: from cognitive.global_workspace import get_workspace; self._workspace = get_workspace()
        return self._workspace

    def assess_knowledge_state(self, signal: Dict[str, Any], context: Dict[str, Any]) -> KnowledgeAssessment:
        """
        Performs a comprehensive assessment of the AI's knowledge and certainty
        regarding a specific trading signal and context.

        Returns:
            A KnowledgeAssessment object summarizing all identified uncertainties.
        """
        uncertainty_sources = []
        missing_info = []
        recommendations = []
        uncertainty_score = 0.0

        # === CHECK 1: FAMILIARITY WITH THE CONTEXT (REGIME & STRATEGY) ===
        regime = context.get('regime', 'unknown')
        regime_confidence = context.get('regime_confidence', 0.5)
        if regime == 'unknown' or regime_confidence < 0.6:
            uncertainty_sources.append(UncertaintySource.REGIME_TRANSITION if regime_confidence < 0.6 else UncertaintySource.NOVEL_REGIME)
            missing_info.append("Stable, high-confidence market regime identification.")
            uncertainty_score += 0.3

        # === CHECK 2: SUFFICIENT EXPERIENCE (SAMPLE SIZE) ===
        # Does the AI have enough past experience in this specific situation?
        # Check both self_model (live trades) and episodic_memory (historical trades).
        strategy = signal.get('strategy', 'unknown')
        side = signal.get('side', 'long')

        # First, check self_model for live trade experience
        performance_stats = self.self_model.get_performance(strategy, regime)
        sample_size = performance_stats.total_trades if performance_stats else 0

        # If self_model has no data, fall back to episodic memory (which has seeded historical data)
        if sample_size < self.min_samples_for_confidence:
            episodic_context = {'regime': regime, 'strategy': strategy, 'side': side}
            _, episodic_sample_size = self.episodic_memory.get_win_rate_for_context(episodic_context)
            sample_size = max(sample_size, episodic_sample_size)

        if sample_size < self.min_samples_for_confidence:
            uncertainty_sources.append(UncertaintySource.LOW_SAMPLE_SIZE)
            missing_info.append(f"More trade examples for {strategy} in {regime} (currently have {sample_size}).")
            # This is a major source of uncertainty.
            uncertainty_score += 0.4 * (1 - (sample_size / self.min_samples_for_confidence))

        # === CHECK 3: CONFLICTING INFORMATION (MODEL DISAGREEMENT) ===
        # Do different internal models or signals disagree?
        if context.get('conflicting_signals'):
            uncertainty_sources.append(UncertaintySource.CONFLICTING_SIGNALS)
            missing_info.append("Resolution of conflicting internal signals.")
            uncertainty_score += 0.2
        
        # === CHECK 4: DATA QUALITY (MISSING OR STALE DATA) ===
        required_data = ['price', 'volume', 'regime']
        for key in required_data:
            if context.get(key) is None:
                uncertainty_sources.append(UncertaintySource.MISSING_DATA)
                missing_info.append(f"Missing essential data point: '{key}'.")
                uncertainty_score += 0.15

        if data_timestamp := context.get('data_timestamp'):
            # Handle both string and datetime objects
            if isinstance(data_timestamp, str):
                ts = datetime.fromisoformat(data_timestamp)
            elif isinstance(data_timestamp, datetime):
                ts = data_timestamp
            else:
                ts = None
            if ts:
                age_hours = (datetime.now() - ts).total_seconds() / 3600
            else:
                age_hours = 0
            if age_hours > self.max_data_age_hours:
                uncertainty_sources.append(UncertaintySource.STALE_DATA)
                missing_info.append(f"Fresh market data (current is {age_hours:.1f}h old).")
                uncertainty_score += 0.2

        # === CHECK 5: EXTREME MARKET CONDITIONS (VOLATILITY) ===
        vix = context.get('vix', 20)
        if vix > 40:
            uncertainty_sources.append(UncertaintySource.UNUSUAL_VOLATILITY)
            recommendations.append(f"Reduce position size due to extreme volatility (VIX={vix:.1f}).")
            uncertainty_score += 0.15

        # === CHECK 6: EXTREME SENTIMENT ===
        # Check for unusually strong positive or negative news sentiment
        market_sentiment = context.get('market_sentiment', {})
        compound_sentiment = market_sentiment.get('compound', 0.0)
        if abs(compound_sentiment) > 0.8:  # Very extreme sentiment (>0.8 or <-0.8)
            uncertainty_sources.append(UncertaintySource.EXTREME_SENTIMENT)
            missing_info.append("Neutral market sentiment for clearer decision-making.")
            if compound_sentiment > 0.8:
                recommendations.append("Extreme positive sentiment may indicate euphoria or market top risk.")
            else:
                recommendations.append("Extreme negative sentiment may indicate panic or capitulation opportunity.")
            uncertainty_score += 0.15
        elif abs(compound_sentiment) > 0.6:  # Moderately extreme sentiment
            # Only mild concern for moderately extreme sentiment
            if compound_sentiment > 0.6:
                recommendations.append("Consider that positive sentiment may be overextended.")
            elif compound_sentiment < -0.6:
                recommendations.append("Consider that negative sentiment may present opportunity.")
            uncertainty_score += 0.05

        # === CHECK 7: EXTREME MARKET MOOD (VIX + Sentiment Combined) ===
        # Check for extreme market mood from MarketMoodAnalyzer
        is_extreme_mood = context.get('is_extreme_mood', False)
        market_mood_score = context.get('market_mood_score', 0.0)
        market_mood_state = context.get('market_mood_state', '')

        if is_extreme_mood:
            uncertainty_sources.append(UncertaintySource.EXTREME_MARKET_MOOD)
            if market_mood_score <= -0.7:  # Extreme fear
                recommendations.append(
                    f"Market in EXTREME FEAR (mood={market_mood_score:.2f}). "
                    "Consider standing down or using contrarian approach with reduced size."
                )
                # Extreme fear is very high uncertainty
                uncertainty_score += 0.25
            elif market_mood_score >= 0.7:  # Extreme greed
                recommendations.append(
                    f"Market in EXTREME GREED (mood={market_mood_score:.2f}). "
                    "Be cautious of potential market top. Consider reducing long exposure."
                )
                # Extreme greed adds moderate uncertainty
                uncertainty_score += 0.20
            else:
                # Extreme but not at the very edge
                recommendations.append(
                    f"Market mood is extreme ({market_mood_state}). Exercise caution."
                )
                uncertainty_score += 0.15

        # === FINALIZE ASSESSMENT ===
        uncertainty_score = min(1.0, uncertainty_score)
        is_uncertain = uncertainty_score >= self.uncertainty_threshold
        should_stand_down = uncertainty_score >= self.stand_down_threshold

        # Map score to a qualitative level.
        if uncertainty_score < 0.3: level = UncertaintyLevel.LOW
        elif uncertainty_score < 0.6: level = UncertaintyLevel.MODERATE
        elif uncertainty_score < 0.8: level = UncertaintyLevel.HIGH
        else: level = UncertaintyLevel.EXTREME

        if should_stand_down:
            recommendations.append("STAND DOWN: Uncertainty score is above the critical threshold.")
        elif is_uncertain:
            recommendations.append("Proceed with caution and reduced size.")

        # The higher the uncertainty, the more we should penalize the final confidence.
        # NOTE: Reduced from 0.5 to 0.25 (2025-12-31) because ML ensemble models
        # now provide good predictions even without episodic memory examples.
        confidence_adjustment = -uncertainty_score * 0.25

        assessment = KnowledgeAssessment(
            uncertainty_level=level,
            uncertainty_sources=list(set(uncertainty_sources)), # Deduplicate
            is_uncertain=is_uncertain,
            should_stand_down=should_stand_down,
            confidence_adjustment=confidence_adjustment,
            missing_information=list(set(missing_info)),
            invalidators=self._generate_invalidators(signal, context),
            recommendations=list(set(recommendations)),
            metadata={'uncertainty_score': uncertainty_score},
        )

        # Publish the assessment so other modules can be aware of the AI's state.
        self.workspace.publish(topic='knowledge_state', data=assessment.to_dict(), source='KnowledgeBoundary')
        logger.debug(f"Knowledge assessment complete: {level.value} (score={uncertainty_score:.2f})")
        return assessment

    def _generate_invalidators(self, signal: Dict[str, Any], context: Dict[str, Any]) -> List[Invalidator]:
        """Generates a list of "what would change my mind" conditions."""
        invalidators = []

        # Example 1: Regime Change Invalidator
        invalidators.append(Invalidator(
            description="The market regime changes before the order is executed.",
            data_needed="Real-time regime classification.",
            check_method="regime_stability_check",
            importance=0.8,
            time_sensitive=True,
        ))

        # Example 2: Volatility Spike Invalidator
        invalidators.append(Invalidator(
            description="Volatility (VIX) spikes unexpectedly above 35.",
            data_needed="Real-time VIX data.",
            check_method="vix_spike_check",
            importance=0.7,
            time_sensitive=True,
        ))

        # Example 3: Sentiment Shift Invalidator
        invalidators.append(Invalidator(
            description="Market sentiment shifts dramatically (>0.5 change in compound score).",
            data_needed="Real-time news sentiment aggregation.",
            check_method="sentiment_shift_check",
            importance=0.6,
            time_sensitive=True,
        ))

        # Example 4: Symbol-specific Sentiment Invalidator
        if symbol := signal.get('symbol'):
            invalidators.append(Invalidator(
                description=f"Major negative news breaks for {symbol}.",
                data_needed=f"Real-time news feed for {symbol}.",
                check_method="symbol_news_check",
                importance=0.75,
                time_sensitive=True,
            ))

        return invalidators

    def what_would_change_mind(self, signal: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """
        A high-level method that returns a human-readable list of doubts and
        conditions that would invalidate the current line of reasoning.
        """
        assessment = self.assess_knowledge_state(signal, context)
        mind_changers = {f"I would reconsider if: {inv.description}" for inv in assessment.invalidators}
        mind_changers.update({f"I would be more confident if I had: {info}" for info in assessment.missing_information})
        
        if UncertaintySource.LOW_SAMPLE_SIZE in assessment.uncertainty_sources:
            mind_changers.add("I would be more confident after seeing more historical examples of this setup.")
            
        return sorted(list(mind_changers))

    def get_confidence_ceiling(self, signal: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Calculates a maximum allowable confidence score given the current level
        of uncertainty. This acts as a cap on the brain's final confidence.
        """
        assessment = self.assess_knowledge_state(signal, context)
        # Start with a max confidence of 1.0 and penalize it for uncertainty.
        # A simple approach: reduce ceiling based on the uncertainty score.
        ceiling = 1.0 - (assessment.metadata.get('uncertainty_score', 0) * 0.5)
        # Ensure the ceiling doesn't go below a minimum reasonable value.
        return max(0.4, ceiling)

    def should_accept(
        self,
        signal: Dict[str, Any],
        context: Dict[str, Any],
        ensemble_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Combined decision rule for accepting/rejecting signals.

        Combines:
        - Episodic evidence: historical sample size (n) and win rate (wr)
        - Ensemble evidence: ML model confidence score
        - Self-model evidence: performance stats from live trading

        Decision rules:
        1. If episodic n >= 100 and wr >= 0.50 and ensemble_confidence >= 0.45 → ACCEPT
        2. Else if self_model has sufficient trades with good performance → ACCEPT
        3. Else if ensemble_confidence >= 0.50 → PROVISIONAL (accept with reduced size)
        4. Otherwise → REJECT

        Args:
            signal: The trading signal dict.
            context: Market context dict.
            ensemble_confidence: ML ensemble confidence score (0-1).

        Returns:
            Dict with 'accept', 'decision', 'reason', and 'size_multiplier'.
        """
        from cognitive.episodic_memory import EpisodicMemory

        # Extract context for signature
        regime = context.get('regime', 'unknown')
        strategy = signal.get('strategy', 'unknown')
        side = signal.get('side', 'long')
        vix = context.get('vix', context.get('vix_level', 20))

        # 1. Get episodic evidence using normalized signature (now includes VIX band)
        episodic_context = {'regime': regime, 'strategy': strategy, 'side': side, 'vix': vix}
        sig = EpisodicMemory.normalize_context_signature(episodic_context)
        episodic_stats = self.episodic_memory.get_stats_for_signature(sig)
        episodic_n = episodic_stats['n']
        episodic_wr = episodic_stats['win_rate']

        # 2. Get self-model evidence
        perf = self.self_model.get_performance(strategy, regime)
        self_model_n = perf.total_trades if perf else 0
        self_model_wr = perf.win_rate if perf else 0.0

        # Build decision result with full evidence for logging
        result = {
            'accept': False,
            'decision': 'REJECT',
            'reason': '',
            'size_multiplier': 0.0,
            'metadata': {
                'signature': sig,
                'ensemble_confidence': ensemble_confidence,
                'episodic_n': episodic_n,
                'episodic_wr': episodic_wr,
                'self_model_n': self_model_n,
                'self_model_wr': self_model_wr,
                'vix': vix,
                'regime': regime,
                'strategy': strategy,
            }
        }

        # === DECISION RULE 1: Strong episodic evidence + ensemble support ===
        # Thresholds: n >= 100, wr >= 0.50 (50%), ensemble >= 0.45
        if episodic_n >= 100 and episodic_wr >= 0.50 and ensemble_confidence >= 0.45:
            result['accept'] = True
            result['decision'] = 'ACCEPT'
            result['reason'] = (
                f'Episodic: n={episodic_n}, wr={episodic_wr:.1%}; '
                f'Ensemble: {ensemble_confidence:.2f}'
            )
            result['size_multiplier'] = 1.0
            logger.info(
                f"ACCEPT signal: sig={sig}, {result['reason']}"
            )
            return result

        # === DECISION RULE 2: Good self-model performance (from live trading) ===
        if self_model_n >= 20 and self_model_wr >= 0.55:
            result['accept'] = True
            result['decision'] = 'ACCEPT'
            result['reason'] = f'Self-model: n={self_model_n}, wr={self_model_wr:.1%}'
            result['size_multiplier'] = 1.0
            logger.info(
                f"ACCEPT signal (self-model): sig={sig}, {result['reason']}"
            )
            return result

        # === DECISION RULE 3: High ensemble confidence (provisional) ===
        if ensemble_confidence >= 0.50:
            result['accept'] = True
            result['decision'] = 'PROVISIONAL'
            result['reason'] = f'High ensemble: {ensemble_confidence:.2f} (reduced size)'
            result['size_multiplier'] = 0.5  # Half position size
            logger.info(
                f"PROVISIONAL accept: sig={sig}, {result['reason']}"
            )
            return result

        # === DECISION RULE 4: Moderate episodic evidence ===
        # Accept if we have some historical support, even with lower ensemble
        if episodic_n >= 50 and episodic_wr >= 0.48 and ensemble_confidence >= 0.40:
            result['accept'] = True
            result['decision'] = 'ACCEPT_MODERATE'
            result['reason'] = (
                f'Moderate episodic: n={episodic_n}, wr={episodic_wr:.1%}; '
                f'Ensemble: {ensemble_confidence:.2f}'
            )
            result['size_multiplier'] = 0.75
            logger.info(
                f"ACCEPT_MODERATE signal: sig={sig}, {result['reason']}"
            )
            return result

        # === REJECT ===
        result['reason'] = (
            f'Insufficient evidence: episodic n={episodic_n}, wr={episodic_wr:.1%}; '
            f'ensemble={ensemble_confidence:.2f}; self_model n={self_model_n}'
        )
        logger.info(
            f"REJECT signal: sig={sig}, {result['reason']}"
        )
        return result

    def introspect(self) -> str:
        """Generates a human-readable report of the component's internal state and philosophy."""
        return (
            "--- Knowledge Boundary Introspection ---\n"
            f"My purpose is to keep myself honest and prevent overconfidence.\n"
            f"I consider a situation 'uncertain' if its uncertainty score exceeds {self.uncertainty_threshold}.\n"
            f"I will recommend standing down if the score exceeds {self.stand_down_threshold}.\n"
            f"I am skeptical of any situation where I have fewer than {self.min_samples_for_confidence} prior examples.\n"
            "Before acting, I always try to ask: 'What would change my mind?'"
        )
