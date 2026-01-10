"""
Comprehensive Unit Tests for Contradiction Resolver Module
============================================================

Tests the contradiction resolution system that resolves conflicting
signals using multiple strategies (historical, rule-based, LLM arbitration).

Run: python -m pytest tests/cognitive/test_contradiction_resolver.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock, patch


class TestSignal:
    """Tests for Signal dataclass."""

    def test_signal_creation(self):
        """Test basic Signal instantiation."""
        from cognitive.contradiction_resolver import Signal

        signal = Signal(
            source="HMM_REGIME",
            direction="LONG",
            confidence=0.75,
            reason="Bull regime detected",
        )

        assert signal.source == "HMM_REGIME"
        assert signal.direction == "LONG"
        assert signal.confidence == 0.75
        assert signal.reason == "Bull regime detected"

    def test_signal_serialization(self):
        """Test signal to_dict serialization."""
        from cognitive.contradiction_resolver import Signal

        signal = Signal(
            source="TECHNICALS",
            direction="SHORT",
            confidence=0.65,
            reason="RSI overbought",
        )

        data = signal.to_dict()

        assert data['source'] == "TECHNICALS"
        assert data['direction'] == "SHORT"
        assert data['confidence'] == 0.65


class TestContradiction:
    """Tests for Contradiction dataclass."""

    def test_contradiction_creation(self):
        """Test Contradiction instantiation."""
        from cognitive.contradiction_resolver import Signal, Contradiction, ContradictionSeverity

        signal_a = Signal("HMM", "LONG", 0.8, "Bullish")
        signal_b = Signal("TECHNICALS", "SHORT", 0.7, "Bearish")

        contradiction = Contradiction(
            signal_a=signal_a,
            signal_b=signal_b,
            severity=ContradictionSeverity.HIGH,
        )

        assert contradiction.signal_a is signal_a
        assert contradiction.signal_b is signal_b
        assert contradiction.severity == ContradictionSeverity.HIGH


class TestResolution:
    """Tests for Resolution dataclass."""

    def test_resolution_creation(self):
        """Test Resolution instantiation."""
        from cognitive.contradiction_resolver import Resolution, ResolutionMethod

        resolution = Resolution(
            decision="LONG",
            confidence=0.72,
            reasoning="Historical accuracy favors HMM in this regime",
            method=ResolutionMethod.HISTORICAL,
        )

        assert resolution.decision == "LONG"
        assert resolution.confidence == 0.72
        assert resolution.method == ResolutionMethod.HISTORICAL

    def test_resolution_serialization(self):
        """Test resolution to_dict."""
        from cognitive.contradiction_resolver import Resolution, ResolutionMethod

        resolution = Resolution(
            decision="STAND_DOWN",
            confidence=0.45,
            reasoning="Too much uncertainty",
            method=ResolutionMethod.LLM_ARBITRATION,
        )

        data = resolution.to_dict()

        assert data['decision'] == "STAND_DOWN"
        assert data['confidence'] == 0.45


class TestContradictionResolver:
    """Tests for the main ContradictionResolver class."""

    def test_initialization(self):
        """Test resolver initialization."""
        from cognitive.contradiction_resolver import ContradictionResolver

        resolver = ContradictionResolver()

        # Resolver initializes with internal state
        assert resolver is not None

    def test_find_contradictions_opposite_directions(self):
        """Test detection of contradictory signals (opposite directions)."""
        from cognitive.contradiction_resolver import ContradictionResolver, Signal

        resolver = ContradictionResolver()

        signals = [
            Signal("HMM", "LONG", 0.8, "Bullish regime"),
            Signal("TECHNICALS", "SHORT", 0.75, "Overbought"),
            Signal("SENTIMENT", "LONG", 0.6, "Positive news"),
        ]

        contradictions = resolver._find_contradictions(signals)

        # Should find contradiction between HMM/SENTIMENT (LONG) vs TECHNICALS (SHORT)
        assert len(contradictions) >= 1

    def test_find_contradictions_no_contradictions(self):
        """Test when all signals agree."""
        from cognitive.contradiction_resolver import ContradictionResolver, Signal

        resolver = ContradictionResolver()

        signals = [
            Signal("HMM", "LONG", 0.8, "Bullish"),
            Signal("TECHNICALS", "LONG", 0.7, "Bullish"),
            Signal("SENTIMENT", "LONG", 0.6, "Bullish"),
        ]

        contradictions = resolver._find_contradictions(signals)

        assert len(contradictions) == 0

    def test_are_contradictory_opposite_directions(self):
        """Test contradiction detection for opposite directions."""
        from cognitive.contradiction_resolver import ContradictionResolver, Signal

        resolver = ContradictionResolver()

        long_signal = Signal("A", "LONG", 0.8, "Buy")
        short_signal = Signal("B", "SHORT", 0.7, "Sell")

        assert resolver._are_contradictory(long_signal, short_signal)
        assert resolver._are_contradictory(short_signal, long_signal)

    def test_are_contradictory_same_direction(self):
        """Test same-direction signals are not contradictory."""
        from cognitive.contradiction_resolver import ContradictionResolver, Signal

        resolver = ContradictionResolver()

        signal1 = Signal("A", "LONG", 0.8, "Buy")
        signal2 = Signal("B", "LONG", 0.7, "Also buy")

        assert not resolver._are_contradictory(signal1, signal2)

    def test_resolve_with_contradictions(self):
        """Test resolution with contradictions defaults to cascade."""
        from cognitive.contradiction_resolver import ContradictionResolver, Signal

        resolver = ContradictionResolver()

        signals = [
            Signal("HMM", "LONG", 0.8, "Bull regime"),
            Signal("TECHNICALS", "SHORT", 0.75, "Overbought"),
        ]

        resolution = resolver.resolve(signals)

        # Should return some resolution
        assert resolution is not None
        assert resolution.decision in ["LONG", "SHORT", "STAND_DOWN", "HOLD"]
        assert resolution.confidence >= 0

    def test_simple_aggregate_all_long(self):
        """Test simple aggregation with all long signals."""
        from cognitive.contradiction_resolver import ContradictionResolver, Signal

        resolver = ContradictionResolver()

        signals = [
            Signal("A", "LONG", 0.8, ""),
            Signal("B", "LONG", 0.7, ""),
            Signal("C", "LONG", 0.9, ""),
        ]

        resolution = resolver._simple_aggregate(signals)

        assert resolution.decision == "LONG"
        assert resolution.confidence > 0.7

    def test_simple_aggregate_mixed_with_majority(self):
        """Test aggregation picks majority direction."""
        from cognitive.contradiction_resolver import ContradictionResolver, Signal

        resolver = ContradictionResolver()

        signals = [
            Signal("A", "LONG", 0.8, ""),
            Signal("B", "LONG", 0.7, ""),
            Signal("C", "SHORT", 0.6, ""),
        ]

        resolution = resolver._simple_aggregate(signals)

        # 2 LONG vs 1 SHORT -> LONG
        assert resolution.decision == "LONG"


class TestSingletonPattern:
    """Tests for singleton accessor."""

    def test_get_resolver_returns_instance(self):
        """Test singleton accessor."""
        from cognitive.contradiction_resolver import get_resolver

        resolver1 = get_resolver()
        resolver2 = get_resolver()

        assert resolver1 is resolver2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_signals_list(self):
        """Test handling of empty signals list."""
        from cognitive.contradiction_resolver import ContradictionResolver

        resolver = ContradictionResolver()

        resolution = resolver.resolve([])

        assert resolution.decision == "STAND_DOWN"
        assert resolution.confidence < 0.5

    def test_single_signal(self):
        """Test handling of single signal (no contradiction possible)."""
        from cognitive.contradiction_resolver import ContradictionResolver, Signal

        resolver = ContradictionResolver()

        signals = [Signal("HMM", "LONG", 0.8, "Only signal")]

        resolution = resolver.resolve(signals)

        assert resolution.decision == "LONG"
        assert resolution.confidence >= 0.8

    def test_all_stand_down_signals(self):
        """Test when all signals recommend stand down."""
        from cognitive.contradiction_resolver import ContradictionResolver, Signal

        resolver = ContradictionResolver()

        signals = [
            Signal("A", "STAND_DOWN", 0.7, ""),
            Signal("B", "STAND_DOWN", 0.8, ""),
        ]

        resolution = resolver.resolve(signals)

        assert resolution.decision == "STAND_DOWN"


class TestEnums:
    """Tests for enum classes."""

    def test_signal_direction_values(self):
        """Test SignalDirection enum values."""
        from cognitive.contradiction_resolver import SignalDirection

        assert SignalDirection.LONG.value == "LONG"
        assert SignalDirection.SHORT.value == "SHORT"
        assert SignalDirection.HOLD.value == "HOLD"
        assert SignalDirection.STAND_DOWN.value == "STAND_DOWN"

    def test_resolution_method_values(self):
        """Test ResolutionMethod enum values."""
        from cognitive.contradiction_resolver import ResolutionMethod

        assert ResolutionMethod.HISTORICAL.value == "historical"
        assert ResolutionMethod.CONFIDENCE_WEIGHTED.value == "confidence_weighted"
        assert ResolutionMethod.LLM_ARBITRATION.value == "llm_arbitration"

    def test_contradiction_severity_values(self):
        """Test ContradictionSeverity enum values."""
        from cognitive.contradiction_resolver import ContradictionSeverity

        assert ContradictionSeverity.NONE.value == "none"
        assert ContradictionSeverity.LOW.value == "low"
        assert ContradictionSeverity.HIGH.value == "high"
        assert ContradictionSeverity.CRITICAL.value == "critical"
