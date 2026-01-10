"""
Contradiction Resolution System
===============================

Explicit system to resolve conflicting signals from multiple models/factors.

When different analysis sources disagree (e.g., HMM says BULL but technicals
say BEAR), this module provides principled resolution through:

1. Historical resolution - Which source was right in similar contexts?
2. Rule-based resolution - Apply learned semantic rules
3. Confidence-weighted synthesis - Weight by source confidence
4. LLM arbitration - Use reasoning for complex cases

Usage:
    from cognitive.contradiction_resolver import ContradictionResolver, get_resolver

    resolver = get_resolver()
    resolution = resolver.resolve(
        signals=[
            Signal(source="hmm_regime", direction="LONG", confidence=0.75),
            Signal(source="technicals", direction="SHORT", confidence=0.65),
        ],
        context={"regime": "NEUTRAL", "vix": 22.5}
    )

    print(f"Decision: {resolution.decision}")
    print(f"Method: {resolution.method}")
    print(f"Reasoning: {resolution.reasoning}")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Trading signal directions."""
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"
    STAND_DOWN = "STAND_DOWN"


class ResolutionMethod(Enum):
    """Methods used to resolve contradictions."""
    NO_CONTRADICTION = "no_contradiction"    # Signals agree
    HISTORICAL = "historical"                 # Based on past accuracy
    RULE_BASED = "rule_based"                # Semantic memory rules
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Weighted by confidence
    LLM_ARBITRATION = "llm_arbitration"      # LLM reasoning
    STAND_DOWN = "stand_down"                 # Too much conflict, don't trade


class ContradictionSeverity(Enum):
    """Severity levels of contradictions."""
    NONE = "none"           # No contradiction
    LOW = "low"             # Minor disagreement
    MEDIUM = "medium"       # Moderate disagreement
    HIGH = "high"           # Strong disagreement
    CRITICAL = "critical"   # Opposite signals with high confidence


@dataclass
class Signal:
    """A trading signal from a specific source."""
    source: str                         # Where the signal came from
    direction: str                      # LONG, SHORT, HOLD, STAND_DOWN
    confidence: float                   # Source's confidence (0-1)
    reason: str = ""                    # Why this signal was generated
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'direction': self.direction,
            'confidence': round(self.confidence, 3),
            'reason': self.reason,
        }


@dataclass
class Contradiction:
    """A detected contradiction between two signals."""
    signal_a: Signal
    signal_b: Signal
    severity: ContradictionSeverity
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'signal_a': self.signal_a.to_dict(),
            'signal_b': self.signal_b.to_dict(),
            'severity': self.severity.value,
            'description': self.description,
        }


@dataclass
class Resolution:
    """The result of contradiction resolution."""
    decision: str                        # Final decision (LONG, SHORT, HOLD, STAND_DOWN)
    confidence: float                    # Confidence in the decision (0-1)
    reasoning: str                       # Explanation of how we arrived at this
    method: ResolutionMethod             # Which resolution method was used
    contradictions_found: List[Contradiction] = field(default_factory=list)
    winning_signal: Optional[Signal] = None
    dissenting_signals: List[Signal] = field(default_factory=list)
    audit_trail: List[str] = field(default_factory=list)  # Step-by-step log

    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision': self.decision,
            'confidence': round(self.confidence, 3),
            'reasoning': self.reasoning,
            'method': self.method.value,
            'contradictions': [c.to_dict() for c in self.contradictions_found],
            'winning_signal': self.winning_signal.to_dict() if self.winning_signal else None,
        }


class ContradictionResolver:
    """
    Resolves conflicting signals using multiple strategies.

    Resolution cascade:
    1. Check if signals actually contradict
    2. Try historical resolution (past accuracy in similar contexts)
    3. Try rule-based resolution (semantic memory rules)
    4. Try confidence-weighted synthesis
    5. Escalate to LLM arbitration for complex cases
    6. Default to STAND_DOWN if unresolvable

    Each resolution attempt adds to an audit trail for transparency.
    """

    # Minimum samples needed for historical resolution
    MIN_HISTORICAL_SAMPLES = 10

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    RESOLUTION_CONFIDENCE_THRESHOLD = 0.6

    def __init__(
        self,
        episodic_memory=None,
        semantic_memory=None,
        llm_provider=None,
    ):
        """
        Initialize the resolver.

        Args:
            episodic_memory: For historical resolution (lazy-loaded if None)
            semantic_memory: For rule-based resolution (lazy-loaded if None)
            llm_provider: For LLM arbitration (lazy-loaded if None)
        """
        self._episodic = episodic_memory
        self._semantic = semantic_memory
        self._llm = llm_provider

        logger.info("ContradictionResolver initialized")

    @property
    def episodic(self):
        """Lazy-load episodic memory."""
        if self._episodic is None:
            try:
                from cognitive.episodic_memory import get_episodic_memory
                self._episodic = get_episodic_memory()
            except ImportError:
                logger.warning("EpisodicMemory not available")
        return self._episodic

    @property
    def semantic(self):
        """Lazy-load semantic memory."""
        if self._semantic is None:
            try:
                from cognitive.semantic_memory import get_semantic_memory
                self._semantic = get_semantic_memory()
            except ImportError:
                logger.warning("SemanticMemory not available")
        return self._semantic

    @property
    def llm(self):
        """Lazy-load LLM provider."""
        if self._llm is None:
            from llm.router import get_provider
            self._llm = get_provider(task_type="reasoning")
        return self._llm

    def resolve(
        self,
        signals: List[Signal],
        context: Optional[Dict[str, Any]] = None,
    ) -> Resolution:
        """
        Resolve conflicting signals.

        Args:
            signals: List of signals from different sources
            context: Market context (regime, vix, etc.)

        Returns:
            Resolution with final decision, confidence, and reasoning
        """
        context = context or {}
        audit_trail = []

        audit_trail.append(f"Starting resolution with {len(signals)} signals")

        # Step 1: Identify contradictions
        contradictions = self._find_contradictions(signals)
        audit_trail.append(f"Found {len(contradictions)} contradictions")

        if not contradictions:
            # No contradictions - simple aggregation
            resolution = self._simple_aggregate(signals)
            resolution.method = ResolutionMethod.NO_CONTRADICTION
            resolution.audit_trail = audit_trail
            return resolution

        # Step 2: Try historical resolution
        audit_trail.append("Attempting historical resolution...")
        historical_resolution = self._resolve_historically(
            contradictions, signals, context
        )
        if historical_resolution.confidence >= self.RESOLUTION_CONFIDENCE_THRESHOLD:
            historical_resolution.contradictions_found = contradictions
            historical_resolution.audit_trail = audit_trail + [
                f"Historical resolution succeeded: {historical_resolution.reasoning}"
            ]
            return historical_resolution

        audit_trail.append(
            f"Historical resolution insufficient (confidence={historical_resolution.confidence:.2f})"
        )

        # Step 3: Try rule-based resolution
        audit_trail.append("Attempting rule-based resolution...")
        rule_resolution = self._resolve_by_rules(contradictions, signals, context)
        if rule_resolution.confidence >= self.RESOLUTION_CONFIDENCE_THRESHOLD:
            rule_resolution.contradictions_found = contradictions
            rule_resolution.audit_trail = audit_trail + [
                f"Rule-based resolution succeeded: {rule_resolution.reasoning}"
            ]
            return rule_resolution

        audit_trail.append(
            f"Rule-based resolution insufficient (confidence={rule_resolution.confidence:.2f})"
        )

        # Step 4: Try confidence-weighted synthesis
        audit_trail.append("Attempting confidence-weighted synthesis...")
        weighted_resolution = self._resolve_by_confidence_weights(signals)
        if weighted_resolution.confidence >= self.RESOLUTION_CONFIDENCE_THRESHOLD:
            weighted_resolution.contradictions_found = contradictions
            weighted_resolution.audit_trail = audit_trail + [
                f"Confidence-weighted resolution: {weighted_resolution.reasoning}"
            ]
            return weighted_resolution

        audit_trail.append(
            f"Confidence-weighted insufficient (confidence={weighted_resolution.confidence:.2f})"
        )

        # Step 5: Escalate to LLM arbitration
        audit_trail.append("Escalating to LLM arbitration...")
        llm_resolution = self._llm_arbitrate(contradictions, signals, context)
        llm_resolution.contradictions_found = contradictions
        llm_resolution.audit_trail = audit_trail + [
            f"LLM arbitration result: {llm_resolution.reasoning}"
        ]
        return llm_resolution

    def _find_contradictions(self, signals: List[Signal]) -> List[Contradiction]:
        """Identify pairs of signals that contradict each other."""
        contradictions = []

        for i, s1 in enumerate(signals):
            for s2 in signals[i+1:]:
                if self._are_contradictory(s1, s2):
                    severity = self._assess_severity(s1, s2)
                    contradictions.append(Contradiction(
                        signal_a=s1,
                        signal_b=s2,
                        severity=severity,
                        description=self._describe_contradiction(s1, s2),
                    ))

        return contradictions

    def _are_contradictory(self, s1: Signal, s2: Signal) -> bool:
        """Check if two signals contradict each other."""
        d1 = s1.direction.upper()
        d2 = s2.direction.upper()

        # Opposite directions
        if (d1 == "LONG" and d2 == "SHORT") or (d1 == "SHORT" and d2 == "LONG"):
            return True

        # One says trade, one says don't
        if d1 in ("LONG", "SHORT") and d2 in ("HOLD", "STAND_DOWN"):
            return True
        if d2 in ("LONG", "SHORT") and d1 in ("HOLD", "STAND_DOWN"):
            return True

        # Strong disagreement on confidence for same direction
        if d1 == d2 and abs(s1.confidence - s2.confidence) > 0.5:
            return True

        return False

    def _assess_severity(self, s1: Signal, s2: Signal) -> ContradictionSeverity:
        """Assess the severity of a contradiction."""
        d1 = s1.direction.upper()
        d2 = s2.direction.upper()

        # Opposite directions with high confidence = CRITICAL
        if d1 == "LONG" and d2 == "SHORT" or d1 == "SHORT" and d2 == "LONG":
            if s1.confidence >= 0.7 and s2.confidence >= 0.7:
                return ContradictionSeverity.CRITICAL
            elif s1.confidence >= 0.5 and s2.confidence >= 0.5:
                return ContradictionSeverity.HIGH
            else:
                return ContradictionSeverity.MEDIUM

        # Trade vs no-trade
        if (d1 in ("LONG", "SHORT")) != (d2 in ("LONG", "SHORT")):
            return ContradictionSeverity.MEDIUM

        # Confidence disagreement
        return ContradictionSeverity.LOW

    def _describe_contradiction(self, s1: Signal, s2: Signal) -> str:
        """Generate a human-readable description of the contradiction."""
        return (
            f"{s1.source} says {s1.direction} (conf={s1.confidence:.0%}) "
            f"vs {s2.source} says {s2.direction} (conf={s2.confidence:.0%})"
        )

    def _simple_aggregate(self, signals: List[Signal]) -> Resolution:
        """Simple aggregation when signals don't contradict."""
        if not signals:
            return Resolution(
                decision="STAND_DOWN",
                confidence=0.0,
                reasoning="No signals to aggregate",
                method=ResolutionMethod.NO_CONTRADICTION,
            )

        # Use highest confidence signal
        best = max(signals, key=lambda s: s.confidence)
        return Resolution(
            decision=best.direction,
            confidence=best.confidence,
            reasoning=f"Signals agree. Using {best.source} (highest confidence)",
            method=ResolutionMethod.NO_CONTRADICTION,
            winning_signal=best,
        )

    def _resolve_historically(
        self,
        contradictions: List[Contradiction],
        signals: List[Signal],
        context: Dict[str, Any],
    ) -> Resolution:
        """Resolve by looking at historical accuracy in similar contexts."""
        if self.episodic is None:
            return Resolution(
                decision="STAND_DOWN",
                confidence=0.3,
                reasoning="Historical resolution unavailable (no episodic memory)",
                method=ResolutionMethod.HISTORICAL,
            )

        try:
            # Query episodic memory for similar situations
            similar_episodes = self.episodic.find_similar(context, limit=50)

            if len(similar_episodes) < self.MIN_HISTORICAL_SAMPLES:
                return Resolution(
                    decision="STAND_DOWN",
                    confidence=0.3,
                    reasoning=f"Insufficient historical data ({len(similar_episodes)} samples)",
                    method=ResolutionMethod.HISTORICAL,
                )

            # Track which sources were right
            source_accuracy: Dict[str, Dict[str, int]] = defaultdict(
                lambda: {"correct": 0, "total": 0}
            )

            for episode in similar_episodes:
                # Get the episode's signals and outcome
                ep_signals = getattr(episode, 'signals', [])
                outcome = getattr(episode, 'outcome', None)

                if not outcome:
                    continue

                was_profitable = getattr(outcome, 'was_profitable', None)
                if was_profitable is None:
                    continue

                for sig in ep_signals:
                    source = getattr(sig, 'source', None)
                    direction = getattr(sig, 'direction', '').upper()

                    if source:
                        source_accuracy[source]["total"] += 1
                        # Check if signal direction matched profitable outcome
                        if was_profitable and direction == "LONG":
                            source_accuracy[source]["correct"] += 1
                        elif not was_profitable and direction in ("SHORT", "STAND_DOWN"):
                            source_accuracy[source]["correct"] += 1

            # Find the historically most accurate source
            if not source_accuracy:
                return Resolution(
                    decision="STAND_DOWN",
                    confidence=0.3,
                    reasoning="No historical accuracy data available",
                    method=ResolutionMethod.HISTORICAL,
                )

            best_source = None
            best_accuracy = 0.0

            for source, stats in source_accuracy.items():
                if stats["total"] >= self.MIN_HISTORICAL_SAMPLES:
                    accuracy = stats["correct"] / stats["total"]
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_source = source

            if best_source:
                # Find signal from best source
                for signal in signals:
                    if signal.source == best_source:
                        return Resolution(
                            decision=signal.direction,
                            confidence=best_accuracy,
                            reasoning=(
                                f"Historically, {best_source} has been "
                                f"{best_accuracy:.0%} accurate in similar contexts"
                            ),
                            method=ResolutionMethod.HISTORICAL,
                            winning_signal=signal,
                        )

        except Exception as e:
            logger.warning(f"Historical resolution failed: {e}")

        return Resolution(
            decision="STAND_DOWN",
            confidence=0.3,
            reasoning="Historical resolution inconclusive",
            method=ResolutionMethod.HISTORICAL,
        )

    def _resolve_by_rules(
        self,
        contradictions: List[Contradiction],
        signals: List[Signal],
        context: Dict[str, Any],
    ) -> Resolution:
        """Apply semantic memory rules for resolution."""
        if self.semantic is None:
            return Resolution(
                decision="STAND_DOWN",
                confidence=0.3,
                reasoning="Rule-based resolution unavailable (no semantic memory)",
                method=ResolutionMethod.RULE_BASED,
            )

        try:
            # Query applicable rules
            rules = self.semantic.get_applicable_rules(context)

            for rule in rules:
                action = getattr(rule, 'action', None)
                params = getattr(rule, 'parameters', {})
                rule_confidence = getattr(rule, 'confidence', 0.5)

                if action == "prefer_source":
                    preferred = params.get("source")
                    for signal in signals:
                        if signal.source == preferred:
                            return Resolution(
                                decision=signal.direction,
                                confidence=rule_confidence,
                                reasoning=(
                                    f"Rule: Prefer {preferred} in this context "
                                    f"(rule confidence: {rule_confidence:.0%})"
                                ),
                                method=ResolutionMethod.RULE_BASED,
                                winning_signal=signal,
                            )

                elif action == "stand_down_on_conflict":
                    severity_threshold = params.get("severity", "HIGH")
                    for c in contradictions:
                        if c.severity.value.upper() >= severity_threshold:
                            return Resolution(
                                decision="STAND_DOWN",
                                confidence=rule_confidence,
                                reasoning=(
                                    f"Rule: Stand down on {c.severity.value} "
                                    f"severity conflicts"
                                ),
                                method=ResolutionMethod.RULE_BASED,
                            )

        except Exception as e:
            logger.warning(f"Rule-based resolution failed: {e}")

        return Resolution(
            decision="STAND_DOWN",
            confidence=0.3,
            reasoning="No applicable rules found",
            method=ResolutionMethod.RULE_BASED,
        )

    def _resolve_by_confidence_weights(
        self,
        signals: List[Signal],
    ) -> Resolution:
        """Weighted resolution based on signal confidences."""
        if not signals:
            return Resolution(
                decision="STAND_DOWN",
                confidence=0.0,
                reasoning="No signals to weight",
                method=ResolutionMethod.CONFIDENCE_WEIGHTED,
            )

        # Weight votes by confidence
        direction_weights: Dict[str, float] = defaultdict(float)

        for signal in signals:
            direction_weights[signal.direction.upper()] += signal.confidence

        if not direction_weights:
            return Resolution(
                decision="STAND_DOWN",
                confidence=0.0,
                reasoning="No valid directions",
                method=ResolutionMethod.CONFIDENCE_WEIGHTED,
            )

        # Find weighted winner
        total_weight = sum(direction_weights.values())
        best_direction = max(direction_weights, key=direction_weights.get)
        best_weight = direction_weights[best_direction]

        # Calculate confidence as proportion of total weight
        confidence = best_weight / total_weight if total_weight > 0 else 0

        # Find the winning signal
        winning = next(
            (s for s in signals if s.direction.upper() == best_direction),
            None
        )

        return Resolution(
            decision=best_direction,
            confidence=confidence,
            reasoning=(
                f"Confidence-weighted: {best_direction} has "
                f"{best_weight:.2f}/{total_weight:.2f} total weight"
            ),
            method=ResolutionMethod.CONFIDENCE_WEIGHTED,
            winning_signal=winning,
        )

    def _llm_arbitrate(
        self,
        contradictions: List[Contradiction],
        signals: List[Signal],
        context: Dict[str, Any],
    ) -> Resolution:
        """Use LLM to arbitrate complex contradictions."""
        prompt = self._build_arbitration_prompt(contradictions, signals, context)

        try:
            from llm.provider_base import LLMMessage
            response = self.llm.chat(
                [LLMMessage(role="user", content=prompt)],
                temperature=0.3,  # Lower temperature for consistency
            )

            decision, confidence, reasoning = self._parse_arbitration(response.content)

            return Resolution(
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                method=ResolutionMethod.LLM_ARBITRATION,
            )

        except Exception as e:
            logger.error(f"LLM arbitration failed: {e}")
            return Resolution(
                decision="STAND_DOWN",
                confidence=0.3,
                reasoning=f"LLM arbitration failed: {e}",
                method=ResolutionMethod.STAND_DOWN,
            )

    def _build_arbitration_prompt(
        self,
        contradictions: List[Contradiction],
        signals: List[Signal],
        context: Dict[str, Any],
    ) -> str:
        """Build the prompt for LLM arbitration."""
        prompt = """You are a trading decision arbitrator. You must resolve conflicting signals.

**Market Context:**
"""
        for key, value in context.items():
            prompt += f"- {key}: {value}\n"

        prompt += "\n**Conflicting Signals:**\n"
        for c in contradictions:
            prompt += f"""
Contradiction ({c.severity.value}):
- {c.signal_a.source}: {c.signal_a.direction} (confidence: {c.signal_a.confidence:.0%})
  Reasoning: {c.signal_a.reason or 'Not provided'}
- {c.signal_b.source}: {c.signal_b.direction} (confidence: {c.signal_b.confidence:.0%})
  Reasoning: {c.signal_b.reason or 'Not provided'}
"""

        prompt += """

**Your Task:**
Analyze these contradictions and provide a final trading decision.

Consider:
1. Which signal has stronger supporting evidence?
2. What does the market context suggest?
3. When signals conflict strongly, should we stand down?
4. Is one source typically more reliable for this type of market?

**Respond in this EXACT format:**
DECISION: [LONG, SHORT, HOLD, or STAND_DOWN]
CONFIDENCE: [0-100]
REASONING: [Your detailed reasoning in 2-3 sentences]"""

        return prompt

    def _parse_arbitration(self, response: str) -> Tuple[str, float, str]:
        """Parse LLM arbitration response."""
        import re

        decision = "STAND_DOWN"
        confidence = 0.5
        reasoning = "Unable to parse LLM response"

        # Extract decision
        decision_match = re.search(
            r'DECISION:\s*(LONG|SHORT|HOLD|STAND_DOWN|STAND DOWN)',
            response,
            re.IGNORECASE
        )
        if decision_match:
            decision = decision_match.group(1).upper().replace(' ', '_')

        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', response, re.IGNORECASE)
        if conf_match:
            confidence = int(conf_match.group(1)) / 100.0
            confidence = max(0.0, min(1.0, confidence))

        # Extract reasoning
        reason_match = re.search(
            r'REASONING:\s*(.+)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if reason_match:
            reasoning = reason_match.group(1).strip()
            # Truncate if too long
            if len(reasoning) > 500:
                reasoning = reasoning[:500] + "..."

        return decision, confidence, reasoning


# Singleton instance
_resolver_instance: Optional[ContradictionResolver] = None


def get_resolver() -> ContradictionResolver:
    """Get the singleton ContradictionResolver instance."""
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = ContradictionResolver()
    return _resolver_instance


def resolve_contradictions(
    signals: List[Signal],
    context: Optional[Dict[str, Any]] = None,
) -> Resolution:
    """
    Convenience function to resolve signal contradictions.

    Args:
        signals: List of signals to reconcile
        context: Market context

    Returns:
        Resolution with final decision
    """
    resolver = get_resolver()
    return resolver.resolve(signals, context)
