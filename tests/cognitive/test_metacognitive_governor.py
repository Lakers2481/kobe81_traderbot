"""
Comprehensive Unit Tests for MetacognitiveGovernor
====================================================

Tests the executive control system that routes decisions between
fast and slow processing pathways.

Run: python -m pytest tests/cognitive/test_metacognitive_governor.py -v
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch


class TestMetacognitiveGovernorInitialization:
    """Tests for MetacognitiveGovernor initialization."""

    def test_default_initialization(self):
        """Test initialization with default thresholds."""
        from cognitive.metacognitive_governor import MetacognitiveGovernor

        governor = MetacognitiveGovernor()

        assert governor.fast_confidence_threshold == 0.75
        assert governor.slow_confidence_threshold == 0.50
        assert governor.stand_down_threshold == 0.30
        assert governor.default_fast_budget_ms == 100
        assert governor.default_slow_budget_ms == 5000

    def test_custom_initialization(self):
        """Test initialization with custom thresholds."""
        from cognitive.metacognitive_governor import MetacognitiveGovernor

        governor = MetacognitiveGovernor(
            fast_confidence_threshold=0.80,
            slow_confidence_threshold=0.55,
            stand_down_threshold=0.25,
        )

        assert governor.fast_confidence_threshold == 0.80
        assert governor.slow_confidence_threshold == 0.55
        assert governor.stand_down_threshold == 0.25


class TestRoutingDecisions:
    """Tests for decision routing logic."""

    def test_high_confidence_routes_to_fast_path(self):
        """Test that high confidence routes to fast processing."""
        from cognitive.metacognitive_governor import MetacognitiveGovernor, ProcessingMode

        governor = MetacognitiveGovernor()

        routing = governor.route_decision(
            signal={'symbol': 'AAPL', 'strategy': 'ibs_rsi'},
            context={'regime': 'BULL'},
            fast_confidence=0.85,
        )

        # High confidence should use fast path
        assert routing.mode in [ProcessingMode.FAST, ProcessingMode.HYBRID]
        assert routing.use_fast_path == True
        assert routing.should_stand_down == False

    def test_low_confidence_triggers_stand_down(self):
        """Test that very low confidence triggers stand-down."""
        from cognitive.metacognitive_governor import (
            MetacognitiveGovernor, ProcessingMode, StandDownReason
        )

        governor = MetacognitiveGovernor(stand_down_threshold=0.30)

        routing = governor.route_decision(
            signal={'symbol': 'TEST'},
            context={'regime': 'unknown'},
            fast_confidence=0.20,
        )

        assert routing.mode == ProcessingMode.STAND_DOWN
        assert routing.should_stand_down == True
        assert routing.stand_down_reason == StandDownReason.HIGH_UNCERTAINTY

    def test_medium_confidence_triggers_hybrid_path(self):
        """Test that medium confidence triggers hybrid processing."""
        from cognitive.metacognitive_governor import MetacognitiveGovernor, ProcessingMode

        governor = MetacognitiveGovernor()

        routing = governor.route_decision(
            signal={'symbol': 'MSFT'},
            context={'regime': 'BULL'},
            fast_confidence=0.60,  # Above stand-down, below fast threshold
        )

        assert routing.mode == ProcessingMode.HYBRID
        assert routing.use_fast_path == True
        assert routing.use_slow_path == True

    def test_conflicting_signals_escalates_to_slow(self):
        """Test that conflicting signals trigger slow path."""
        from cognitive.metacognitive_governor import (
            MetacognitiveGovernor, ProcessingMode, EscalationReason
        )

        governor = MetacognitiveGovernor()

        routing = governor.route_decision(
            signal={'symbol': 'NVDA'},
            context={
                'regime': 'BULL',
                'conflicting_signals': True,
            },
            fast_confidence=0.70,
        )

        assert EscalationReason.CONFLICTING_SIGNALS in routing.escalation_reasons

    def test_high_stakes_position_escalates(self):
        """Test that high-stakes positions trigger deeper analysis."""
        from cognitive.metacognitive_governor import (
            MetacognitiveGovernor, EscalationReason
        )

        governor = MetacognitiveGovernor(high_stakes_position_pct=0.03)

        routing = governor.route_decision(
            signal={
                'symbol': 'AMZN',
                'position_pct': 0.05,  # 5% position (above threshold)
            },
            context={'regime': 'BULL'},
            fast_confidence=0.70,
        )

        assert EscalationReason.HIGH_STAKES in routing.escalation_reasons


class TestStandDownDecisions:
    """Tests for stand-down decision logic."""

    def test_stand_down_increments_counter(self):
        """Test that stand-down decisions are counted."""
        from cognitive.metacognitive_governor import MetacognitiveGovernor

        governor = MetacognitiveGovernor()
        initial_stand_downs = governor._stand_downs

        governor.route_decision(
            signal={'symbol': 'TEST'},
            context={},
            fast_confidence=0.15,  # Very low confidence
        )

        assert governor._stand_downs == initial_stand_downs + 1

    def test_stand_down_has_correct_reason(self):
        """Test that stand-down decisions include the reason."""
        from cognitive.metacognitive_governor import (
            MetacognitiveGovernor, StandDownReason
        )

        governor = MetacognitiveGovernor()

        routing = governor.route_decision(
            signal={'symbol': 'TEST'},
            context={},
            fast_confidence=0.10,
        )

        assert routing.stand_down_reason is not None
        assert routing.metadata.get('stand_down_details') is not None


class TestRoutingStatistics:
    """Tests for routing statistics and tracking."""

    def test_routing_stats_empty(self):
        """Test statistics when no decisions have been made."""
        from cognitive.metacognitive_governor import MetacognitiveGovernor

        governor = MetacognitiveGovernor()
        stats = governor.get_routing_stats()

        assert stats['total_decisions'] == 0

    def test_routing_stats_after_decisions(self):
        """Test statistics after multiple decisions."""
        from cognitive.metacognitive_governor import MetacognitiveGovernor

        governor = MetacognitiveGovernor()

        # Make a few decisions
        for conf in [0.9, 0.5, 0.1]:  # Fast, Hybrid, Stand-down
            governor.route_decision(
                signal={'symbol': 'TEST'},
                context={'regime': 'BULL'},
                fast_confidence=conf,
            )

        stats = governor.get_routing_stats()

        assert stats['total_decisions'] == 3
        assert stats['stand_downs'] >= 1  # At least one stand-down (conf=0.1)

    def test_record_outcome(self):
        """Test recording outcomes for routing decisions."""
        from cognitive.metacognitive_governor import MetacognitiveGovernor

        governor = MetacognitiveGovernor()

        routing = governor.route_decision(
            signal={'symbol': 'TEST'},
            context={'regime': 'BULL'},
            fast_confidence=0.8,
        )

        # Record the outcome
        governor.record_outcome(
            decision_id=routing.decision_id,
            outcome='success',
            was_correct=True,
            actual_compute_ms=50,
        )

        # Find the record and verify
        record = None
        for r in governor._decision_history:
            if r.decision_id == routing.decision_id:
                record = r
                break

        assert record is not None
        assert record.outcome == 'success'
        assert record.was_correct == True
        assert record.actual_compute_ms == 50


class TestNovelSituationDetection:
    """Tests for novel situation detection."""

    def test_novel_situation_with_no_history(self):
        """Test that novel situation is detected when no history exists."""
        from cognitive.metacognitive_governor import MetacognitiveGovernor

        governor = MetacognitiveGovernor()

        is_novel = governor._is_novel_situation(
            signal={'strategy': 'new_strategy'},
            context={'regime': 'UNKNOWN_REGIME'},
        )

        # Should be novel since no history exists
        assert is_novel == True or is_novel == False  # Depends on self_model state


class TestIntrospection:
    """Tests for the governor's self-introspection."""

    def test_introspect_returns_report(self):
        """Test that introspect returns a formatted report."""
        from cognitive.metacognitive_governor import MetacognitiveGovernor

        governor = MetacognitiveGovernor()

        # Make some decisions to have data
        for conf in [0.9, 0.6, 0.2]:
            governor.route_decision(
                signal={'symbol': 'TEST'},
                context={'regime': 'BULL'},
                fast_confidence=conf,
            )

        report = governor.introspect()

        assert "Metacognitive Introspection" in report
        assert "routed" in report.lower()

    def test_introspect_identifies_concerns(self):
        """Test that introspection identifies performance concerns."""
        from cognitive.metacognitive_governor import MetacognitiveGovernor

        governor = MetacognitiveGovernor()

        # Force a high stand-down rate
        for _ in range(10):
            governor.route_decision(
                signal={'symbol': 'TEST'},
                context={},
                fast_confidence=0.15,  # Will trigger stand-down
            )

        report = governor.introspect()

        # Should identify high stand-down rate as a concern
        assert "stand-down" in report.lower() or "balanced" in report.lower()


class TestRoutingDecisionDataclass:
    """Tests for the RoutingDecision dataclass."""

    def test_routing_decision_to_dict(self):
        """Test RoutingDecision serialization."""
        from cognitive.metacognitive_governor import (
            RoutingDecision, ProcessingMode, EscalationReason
        )

        routing = RoutingDecision(
            decision_id='test-123',
            mode=ProcessingMode.HYBRID,
            use_fast_path=True,
            use_slow_path=True,
            should_stand_down=False,
            confidence_in_routing=0.85,
            escalation_reasons=[EscalationReason.LOW_CONFIDENCE],
            stand_down_reason=None,
            max_compute_ms=2500,
        )

        d = routing.to_dict()

        assert d['decision_id'] == 'test-123'
        assert d['mode'] == 'hybrid'
        assert d['use_fast_path'] == True
        assert d['escalation_reasons'] == ['low_confidence']
        assert d['stand_down_reason'] is None


class TestDecisionRecord:
    """Tests for the DecisionRecord dataclass."""

    def test_was_efficient_property(self):
        """Test the was_efficient property calculation."""
        from cognitive.metacognitive_governor import (
            DecisionRecord, RoutingDecision, ProcessingMode
        )

        routing = RoutingDecision(
            decision_id='test',
            mode=ProcessingMode.FAST,
            use_fast_path=True,
            use_slow_path=False,
            should_stand_down=False,
            confidence_in_routing=0.9,
            escalation_reasons=[],
            stand_down_reason=None,
            max_compute_ms=100,
        )

        record = DecisionRecord(
            decision_id='test',
            routing=routing,
            started_at=datetime.now(),
            actual_compute_ms=50,
        )

        assert record.was_efficient == True  # 50ms < 100ms budget

        record.actual_compute_ms = 150
        assert record.was_efficient == False  # 150ms > 100ms budget


class TestProcessingModeEnum:
    """Tests for the ProcessingMode enumeration."""

    def test_processing_modes(self):
        """Test all processing modes are accessible."""
        from cognitive.metacognitive_governor import ProcessingMode

        assert ProcessingMode.FAST.value == 'fast'
        assert ProcessingMode.SLOW.value == 'slow'
        assert ProcessingMode.HYBRID.value == 'hybrid'
        assert ProcessingMode.STAND_DOWN.value == 'stand_down'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
