"""
Knowledge Boundary Detection
==============================

Implements "what would change my mind" thinking.

Anti-overconfidence layer that:
- Detects when operating outside known territory
- Identifies what information is missing
- Triggers deeper analysis or stand-down

Based on AI self-awareness research (arXiv):
- Knowledge-boundary awareness (what it doesn't know)
- Introspection (reviewing its own decisions)
- Uncertainty quantification

Features:
- Out-of-distribution detection
- Missing data identification
- Contradiction detection
- "What would change my mind" queries
- Stand-down recommendations

Usage:
    from cognitive.knowledge_boundary import KnowledgeBoundary

    kb = KnowledgeBoundary()

    # Check if we're in known territory
    assessment = kb.assess_knowledge_state(signal, context)

    if assessment.is_uncertain:
        if assessment.should_stand_down:
            return "STAND DOWN"
        else:
            return "NEED MORE DATA"

    # Ask what would change our mind
    invalidators = kb.what_would_change_mind(signal, context)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class UncertaintyLevel(Enum):
    """Levels of uncertainty."""
    LOW = "low"           # Well-understood territory
    MODERATE = "moderate" # Some unknowns but manageable
    HIGH = "high"         # Significant uncertainty
    EXTREME = "extreme"   # Operating blind


class UncertaintySource(Enum):
    """Sources of uncertainty."""
    NOVEL_REGIME = "novel_regime"
    NOVEL_SYMBOL = "novel_symbol"
    MISSING_DATA = "missing_data"
    MODEL_DISAGREEMENT = "model_disagreement"
    LOW_SAMPLE_SIZE = "low_sample_size"
    REGIME_TRANSITION = "regime_transition"
    CONFLICTING_SIGNALS = "conflicting_signals"
    STALE_DATA = "stale_data"
    UNUSUAL_VOLATILITY = "unusual_volatility"
    EDGE_CASE = "edge_case"


@dataclass
class Invalidator:
    """Something that would change our mind about a decision."""
    description: str
    data_needed: str
    check_method: str
    importance: float  # 0-1
    time_sensitive: bool = False


@dataclass
class KnowledgeAssessment:
    """Assessment of our knowledge state for a decision."""
    uncertainty_level: UncertaintyLevel
    uncertainty_sources: List[UncertaintySource]
    is_uncertain: bool
    should_stand_down: bool
    confidence_adjustment: float  # -1 to 1
    missing_information: List[str]
    invalidators: List[Invalidator]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'uncertainty_level': self.uncertainty_level.value,
            'uncertainty_sources': [s.value for s in self.uncertainty_sources],
            'is_uncertain': self.is_uncertain,
            'should_stand_down': self.should_stand_down,
            'confidence_adjustment': self.confidence_adjustment,
            'missing_information': self.missing_information,
            'invalidators': [
                {'description': inv.description, 'data_needed': inv.data_needed}
                for inv in self.invalidators
            ],
            'recommendations': self.recommendations,
        }


class KnowledgeBoundary:
    """
    Detects when we're operating at or beyond our knowledge boundaries.

    Key questions:
    - "Have I seen this before?"
    - "What am I uncertain about?"
    - "What data am I missing?"
    - "What would change my mind?"
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.6,  # Above this = uncertain
        stand_down_threshold: float = 0.8,    # Above this = stand down
        min_samples_for_confidence: int = 10,
        max_data_age_hours: int = 4,
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.stand_down_threshold = stand_down_threshold
        self.min_samples_for_confidence = min_samples_for_confidence
        self.max_data_age_hours = max_data_age_hours

        # Lazy load dependencies
        self._self_model = None
        self._episodic_memory = None
        self._workspace = None

        logger.info("KnowledgeBoundary initialized")

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
    def workspace(self):
        if self._workspace is None:
            from cognitive.global_workspace import get_workspace
            self._workspace = get_workspace()
        return self._workspace

    def assess_knowledge_state(
        self,
        signal: Dict[str, Any],
        context: Dict[str, Any],
    ) -> KnowledgeAssessment:
        """
        Assess our knowledge state for making a decision.

        Args:
            signal: The trade signal being evaluated
            context: Current market/portfolio context

        Returns:
            KnowledgeAssessment with uncertainty analysis
        """
        uncertainty_sources = []
        missing_info = []
        invalidators = []
        recommendations = []
        uncertainty_score = 0.0

        # === Check 1: Novel Regime ===
        regime = context.get('regime', 'unknown')
        regime_confidence = context.get('regime_confidence', 0.5)

        if regime == 'unknown':
            uncertainty_sources.append(UncertaintySource.NOVEL_REGIME)
            missing_info.append("Clear regime identification")
            uncertainty_score += 0.3
        elif regime_confidence < 0.6:
            uncertainty_sources.append(UncertaintySource.REGIME_TRANSITION)
            missing_info.append("Stable regime confirmation")
            uncertainty_score += 0.2

        # === Check 2: Sample Size ===
        strategy = signal.get('strategy', 'unknown')
        perf = self.self_model.get_performance(strategy, regime)

        if perf is None or perf.total_trades < self.min_samples_for_confidence:
            uncertainty_sources.append(UncertaintySource.LOW_SAMPLE_SIZE)
            samples = perf.total_trades if perf else 0
            missing_info.append(
                f"More experience in {strategy}/{regime} "
                f"(have {samples}, need {self.min_samples_for_confidence})"
            )
            uncertainty_score += 0.25

        # === Check 3: Model Disagreement ===
        model_predictions = context.get('model_predictions', {})
        if model_predictions:
            values = list(model_predictions.values())
            if len(values) >= 2:
                import statistics
                try:
                    std = statistics.stdev(values)
                    if std > 0.2:
                        uncertainty_sources.append(UncertaintySource.MODEL_DISAGREEMENT)
                        missing_info.append("Model consensus")
                        uncertainty_score += std

                        invalidators.append(Invalidator(
                            description="Models disagree significantly",
                            data_needed="Additional model confirmation or analysis",
                            check_method="ensemble_variance_check",
                            importance=0.8,
                        ))
                except:
                    pass

        # === Check 4: Missing Data ===
        required_data = ['price', 'volume', 'regime', 'vix']
        for key in required_data:
            if context.get(key) is None:
                uncertainty_sources.append(UncertaintySource.MISSING_DATA)
                missing_info.append(f"Missing {key} data")
                uncertainty_score += 0.15

        # === Check 5: Stale Data ===
        data_timestamp = context.get('data_timestamp')
        if data_timestamp:
            if isinstance(data_timestamp, str):
                data_timestamp = datetime.fromisoformat(data_timestamp)
            age_hours = (datetime.now() - data_timestamp).total_seconds() / 3600
            if age_hours > self.max_data_age_hours:
                uncertainty_sources.append(UncertaintySource.STALE_DATA)
                missing_info.append(f"Fresh data (current is {age_hours:.1f}h old)")
                uncertainty_score += 0.2

        # === Check 6: Unusual Volatility ===
        vix = context.get('vix', 20)
        if vix > 40:
            uncertainty_sources.append(UncertaintySource.UNUSUAL_VOLATILITY)
            missing_info.append("Normal volatility conditions")
            uncertainty_score += 0.15
            recommendations.append("Reduce position size due to high volatility")

        # === Check 7: Conflicting Signals ===
        if context.get('conflicting_signals'):
            uncertainty_sources.append(UncertaintySource.CONFLICTING_SIGNALS)
            missing_info.append("Consistent signal direction")
            uncertainty_score += 0.2

            invalidators.append(Invalidator(
                description="Receiving conflicting signals",
                data_needed="Signal clarification or additional confirmation",
                check_method="signal_consistency_check",
                importance=0.7,
            ))

        # === Check 8: Edge Cases ===
        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)

        if entry_price > 0 and stop_loss > 0:
            risk_pct = abs(entry_price - stop_loss) / entry_price
            if risk_pct > 0.05 or risk_pct < 0.005:
                uncertainty_sources.append(UncertaintySource.EDGE_CASE)
                missing_info.append(f"Normal risk/reward (current: {risk_pct:.1%})")
                uncertainty_score += 0.1

        # === Generate Invalidators ===
        invalidators.extend(self._generate_invalidators(signal, context))

        # === Calculate Final Assessment ===
        uncertainty_score = min(1.0, uncertainty_score)
        is_uncertain = uncertainty_score > self.uncertainty_threshold
        should_stand_down = uncertainty_score > self.stand_down_threshold

        # Determine level
        if uncertainty_score < 0.3:
            level = UncertaintyLevel.LOW
        elif uncertainty_score < 0.6:
            level = UncertaintyLevel.MODERATE
        elif uncertainty_score < 0.8:
            level = UncertaintyLevel.HIGH
        else:
            level = UncertaintyLevel.EXTREME

        # Generate recommendations
        if is_uncertain and not should_stand_down:
            recommendations.append("Reduce position size due to uncertainty")
            recommendations.append("Set tighter stops")
        elif should_stand_down:
            recommendations.append("STAND DOWN - uncertainty too high")

        confidence_adjustment = -uncertainty_score * 0.5  # Max -50% confidence adjustment

        assessment = KnowledgeAssessment(
            uncertainty_level=level,
            uncertainty_sources=uncertainty_sources,
            is_uncertain=is_uncertain,
            should_stand_down=should_stand_down,
            confidence_adjustment=confidence_adjustment,
            missing_information=missing_info,
            invalidators=invalidators,
            recommendations=recommendations,
            metadata={
                'uncertainty_score': uncertainty_score,
                'strategy': strategy,
                'regime': regime,
            }
        )

        # Publish to workspace
        self.workspace.publish(
            topic='knowledge_state',
            data=assessment.to_dict(),
            source='knowledge_boundary',
        )

        logger.debug(
            f"Knowledge assessment: {level.value} "
            f"(score={uncertainty_score:.2f}, sources={len(uncertainty_sources)})"
        )

        return assessment

    def _generate_invalidators(
        self,
        signal: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Invalidator]:
        """Generate 'what would change my mind' invalidators."""
        invalidators = []

        # Price invalidator
        entry_price = signal.get('entry_price', 0)
        if entry_price > 0:
            invalidators.append(Invalidator(
                description=f"Price moves {10}% against before entry",
                data_needed="Real-time price update",
                check_method="price_threshold_check",
                importance=0.9,
                time_sensitive=True,
            ))

        # Regime invalidator
        invalidators.append(Invalidator(
            description="Regime changes before execution",
            data_needed="Updated regime classification",
            check_method="regime_stability_check",
            importance=0.8,
        ))

        # VIX spike invalidator
        invalidators.append(Invalidator(
            description="VIX spikes above 35",
            data_needed="Real-time VIX",
            check_method="vix_spike_check",
            importance=0.7,
            time_sensitive=True,
        ))

        # Volume invalidator
        invalidators.append(Invalidator(
            description="Abnormal volume (>3x average)",
            data_needed="Real-time volume",
            check_method="volume_anomaly_check",
            importance=0.6,
        ))

        return invalidators

    def what_would_change_mind(
        self,
        signal: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[str]:
        """
        Generate list of things that would change our mind about a decision.

        This is the core "anti-overconfidence" question.

        Returns:
            List of natural language descriptions of invalidating conditions
        """
        assessment = self.assess_knowledge_state(signal, context)

        change_mind_list = []

        # From invalidators
        for inv in assessment.invalidators:
            change_mind_list.append(
                f"I would reconsider if: {inv.description}"
            )

        # From missing information
        for missing in assessment.missing_information:
            change_mind_list.append(
                f"I need to know: {missing}"
            )

        # From uncertainty sources
        for source in assessment.uncertainty_sources:
            if source == UncertaintySource.MODEL_DISAGREEMENT:
                change_mind_list.append(
                    "I would be more confident if models agreed"
                )
            elif source == UncertaintySource.LOW_SAMPLE_SIZE:
                change_mind_list.append(
                    "I would be more confident with more historical examples"
                )
            elif source == UncertaintySource.REGIME_TRANSITION:
                change_mind_list.append(
                    "I would wait for regime to stabilize"
                )

        return change_mind_list

    def check_invalidators(
        self,
        invalidators: List[Invalidator],
        current_data: Dict[str, Any],
    ) -> List[Tuple[Invalidator, bool, str]]:
        """
        Check if any invalidators have been triggered.

        Args:
            invalidators: List of invalidators to check
            current_data: Current market data

        Returns:
            List of (invalidator, triggered, reason) tuples
        """
        results = []

        for inv in invalidators:
            triggered = False
            reason = ""

            if inv.check_method == "price_threshold_check":
                original_price = current_data.get('original_entry_price', 0)
                current_price = current_data.get('current_price', 0)
                if original_price > 0 and current_price > 0:
                    change = abs(current_price - original_price) / original_price
                    if change > 0.10:
                        triggered = True
                        reason = f"Price moved {change:.1%}"

            elif inv.check_method == "vix_spike_check":
                vix = current_data.get('vix', 20)
                if vix > 35:
                    triggered = True
                    reason = f"VIX at {vix}"

            elif inv.check_method == "volume_anomaly_check":
                volume = current_data.get('volume', 0)
                avg_volume = current_data.get('avg_volume', 1)
                if avg_volume > 0 and volume > avg_volume * 3:
                    triggered = True
                    reason = f"Volume {volume/avg_volume:.1f}x average"

            results.append((inv, triggered, reason))

        return results

    def get_confidence_ceiling(
        self,
        signal: Dict[str, Any],
        context: Dict[str, Any],
    ) -> float:
        """
        Get maximum confidence we should have given uncertainty.

        Returns a ceiling (0-1) that confidence should not exceed.
        """
        assessment = self.assess_knowledge_state(signal, context)

        # Base ceiling is 1.0
        ceiling = 1.0

        # Reduce for each uncertainty source
        ceiling -= len(assessment.uncertainty_sources) * 0.1

        # Reduce for missing data
        ceiling -= len(assessment.missing_information) * 0.05

        # Floor at 0.2
        return max(0.2, ceiling)

    def introspect(self) -> str:
        """Generate introspective report."""
        lines = [
            "=== Knowledge Boundary Introspection ===",
            "",
            f"Uncertainty threshold: {self.uncertainty_threshold}",
            f"Stand-down threshold: {self.stand_down_threshold}",
            f"Min samples for confidence: {self.min_samples_for_confidence}",
            f"Max data age: {self.max_data_age_hours}h",
            "",
            "I am aware that I:",
            "- Can be overconfident in novel situations",
            "- Should stand down when uncertainty is high",
            "- Need to verify my assumptions before acting",
            "- Should always ask 'what would change my mind?'",
        ]

        return "\n".join(lines)
