"""
Unit tests for cognitive/knowledge_boundary.py

Tests the AI's ability to recognize the limits of its own knowledge.
"""
import pytest
from datetime import datetime, timedelta


class TestUncertaintyLevelEnum:
    """Tests for the UncertaintyLevel enumeration."""

    def test_uncertainty_levels(self):
        from cognitive.knowledge_boundary import UncertaintyLevel

        assert UncertaintyLevel.LOW.value == "low"
        assert UncertaintyLevel.MODERATE.value == "moderate"
        assert UncertaintyLevel.HIGH.value == "high"
        assert UncertaintyLevel.EXTREME.value == "extreme"


class TestUncertaintySourceEnum:
    """Tests for the UncertaintySource enumeration."""

    def test_uncertainty_sources(self):
        from cognitive.knowledge_boundary import UncertaintySource

        assert UncertaintySource.MISSING_DATA.value == "missing_data"
        assert UncertaintySource.STALE_DATA.value == "stale_data"
        assert UncertaintySource.LOW_SAMPLE_SIZE.value == "low_sample_size"
        assert UncertaintySource.NOVEL_REGIME.value == "novel_regime"
        assert UncertaintySource.CONFLICTING_SIGNALS.value == "conflicting_signals"


class TestInvalidatorDataclass:
    """Tests for the Invalidator dataclass."""

    def test_invalidator_creation(self):
        from cognitive.knowledge_boundary import Invalidator

        invalidator = Invalidator(
            description="VIX spikes above 35",
            data_needed="Real-time VIX data",
            check_method="vix_spike_check",
            importance=0.8,
            time_sensitive=True,
        )

        assert invalidator.description == "VIX spikes above 35"
        assert invalidator.importance == 0.8
        assert invalidator.time_sensitive is True


class TestKnowledgeAssessmentDataclass:
    """Tests for the KnowledgeAssessment dataclass."""

    def test_assessment_creation(self):
        from cognitive.knowledge_boundary import KnowledgeAssessment, UncertaintyLevel

        assessment = KnowledgeAssessment(
            uncertainty_level=UncertaintyLevel.MODERATE,
            uncertainty_sources=[],
            is_uncertain=True,
            should_stand_down=False,
            confidence_adjustment=-0.2,
            missing_information=["Fresh market data"],
            invalidators=[],
            recommendations=["Proceed with caution"],
        )

        assert assessment.uncertainty_level == UncertaintyLevel.MODERATE
        assert assessment.is_uncertain is True
        assert assessment.should_stand_down is False
        assert assessment.confidence_adjustment == -0.2

    def test_assessment_to_dict(self):
        from cognitive.knowledge_boundary import KnowledgeAssessment, UncertaintyLevel

        assessment = KnowledgeAssessment(
            uncertainty_level=UncertaintyLevel.LOW,
            uncertainty_sources=[],
            is_uncertain=False,
            should_stand_down=False,
            confidence_adjustment=-0.1,
            missing_information=[],
            invalidators=[],
            recommendations=[],
        )
        d = assessment.to_dict()

        assert d['uncertainty_level'] == 'low'
        assert d['is_uncertain'] is False


class TestKnowledgeBoundaryInitialization:
    """Tests for KnowledgeBoundary initialization."""

    def test_default_initialization(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary

        boundary = KnowledgeBoundary()

        assert boundary.uncertainty_threshold == 0.5
        assert boundary.stand_down_threshold == 0.8

    def test_custom_thresholds(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary

        boundary = KnowledgeBoundary(
            uncertainty_threshold=0.4,
            stand_down_threshold=0.7,
        )

        assert boundary.uncertainty_threshold == 0.4
        assert boundary.stand_down_threshold == 0.7


class TestAssessKnowledgeState:
    """Tests for assessing knowledge state."""

    def test_low_uncertainty_context(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary, UncertaintyLevel

        boundary = KnowledgeBoundary()

        signal = {'strategy': 'ibs_rsi', 'symbol': 'AAPL'}
        context = {
            'regime': 'BULL',
            'regime_confidence': 0.85,
            'vix': 15,
            'price': 150.0,
            'volume': 1000000,
            'data_timestamp': datetime.now(),
        }

        assessment = boundary.assess_knowledge_state(signal, context)

        assert assessment.uncertainty_level in [UncertaintyLevel.LOW, UncertaintyLevel.MODERATE]
        assert assessment.should_stand_down is False

    def test_missing_data_increases_uncertainty(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary, UncertaintySource

        boundary = KnowledgeBoundary()

        signal = {'strategy': 'ibs_rsi'}
        context = {
            'regime': 'BULL',
            'regime_confidence': 0.85,
            # Missing: price, volume
        }

        assessment = boundary.assess_knowledge_state(signal, context)

        # Should have some uncertainty due to missing data
        assert len(assessment.missing_information) > 0

    def test_stale_data_increases_uncertainty(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary, UncertaintySource

        boundary = KnowledgeBoundary()

        signal = {'strategy': 'ibs_rsi'}
        old_timestamp = datetime.now() - timedelta(hours=48)
        context = {
            'regime': 'BULL',
            'regime_confidence': 0.85,
            'price': 100.0,
            'volume': 1000000,
            'data_timestamp': old_timestamp,
        }

        assessment = boundary.assess_knowledge_state(signal, context)

        # Stale data should be noted
        assert UncertaintySource.STALE_DATA in assessment.uncertainty_sources

    def test_high_vix_increases_uncertainty(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary, UncertaintySource

        boundary = KnowledgeBoundary()

        signal = {'strategy': 'ibs_rsi'}
        context = {
            'regime': 'BEAR',
            'regime_confidence': 0.7,
            'vix': 50,  # Very high volatility
            'price': 100.0,
            'volume': 1000000,
            'data_timestamp': datetime.now(),
        }

        assessment = boundary.assess_knowledge_state(signal, context)

        assert UncertaintySource.UNUSUAL_VOLATILITY in assessment.uncertainty_sources

    def test_stand_down_on_extreme_uncertainty(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary

        boundary = KnowledgeBoundary(
            uncertainty_threshold=0.3,
            stand_down_threshold=0.6,
        )

        signal = {'strategy': 'unknown_strategy'}
        context = {
            'regime': 'unknown',
            'regime_confidence': 0.3,
            'vix': 60,
            'conflicting_signals': True,
            # Missing most data
        }

        assessment = boundary.assess_knowledge_state(signal, context)

        # With multiple uncertainty sources, should have high uncertainty
        assert assessment.is_uncertain is True


class TestWhatWouldChangeMind:
    """Tests for identifying what information would reduce uncertainty."""

    def test_returns_list_of_doubts(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary

        boundary = KnowledgeBoundary()

        signal = {'strategy': 'ibs_rsi'}
        context = {
            'regime': 'BULL',
            'regime_confidence': 0.7,
        }

        mind_changers = boundary.what_would_change_mind(signal, context)

        assert isinstance(mind_changers, list)
        assert len(mind_changers) > 0


class TestGetConfidenceCeiling:
    """Tests for getting the confidence ceiling."""

    def test_high_uncertainty_lowers_ceiling(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary

        boundary = KnowledgeBoundary()

        signal = {'strategy': 'ibs_rsi'}
        context = {
            'regime': 'unknown',
            'regime_confidence': 0.3,
            'vix': 50,
            # Missing data
        }

        ceiling = boundary.get_confidence_ceiling(signal, context)

        # High uncertainty should lower the ceiling
        assert ceiling < 1.0

    def test_low_uncertainty_high_ceiling(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary

        boundary = KnowledgeBoundary()

        signal = {'strategy': 'ibs_rsi'}
        context = {
            'regime': 'BULL',
            'regime_confidence': 0.9,
            'vix': 15,
            'price': 150.0,
            'volume': 1000000,
            'data_timestamp': datetime.now(),
        }

        ceiling = boundary.get_confidence_ceiling(signal, context)

        # Low uncertainty should keep ceiling high
        assert ceiling >= 0.7


class TestIntrospection:
    """Tests for knowledge boundary introspection."""

    def test_introspect(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary

        boundary = KnowledgeBoundary()

        report = boundary.introspect()

        assert isinstance(report, str)
        assert len(report) > 0
        assert "Knowledge Boundary" in report


class TestConfidenceAdjustment:
    """Tests for confidence adjustment calculation."""

    def test_confidence_adjustment_negative_with_uncertainty(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary

        boundary = KnowledgeBoundary()

        signal = {'strategy': 'ibs_rsi'}
        context = {
            'regime': 'unknown',
            'regime_confidence': 0.4,
            'vix': 40,
        }

        assessment = boundary.assess_knowledge_state(signal, context)

        # Confidence adjustment should be negative when uncertain
        assert assessment.confidence_adjustment < 0


class TestTimestampHandling:
    """Tests for handling different timestamp formats."""

    def test_handles_string_timestamp(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary

        boundary = KnowledgeBoundary()

        signal = {'strategy': 'ibs_rsi'}
        context = {
            'regime': 'BULL',
            'regime_confidence': 0.8,
            'price': 100.0,
            'volume': 1000000,
            'data_timestamp': datetime.now().isoformat(),  # String format
        }

        # Should not raise an exception
        assessment = boundary.assess_knowledge_state(signal, context)
        assert assessment is not None

    def test_handles_datetime_timestamp(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary

        boundary = KnowledgeBoundary()

        signal = {'strategy': 'ibs_rsi'}
        context = {
            'regime': 'BULL',
            'regime_confidence': 0.8,
            'price': 100.0,
            'volume': 1000000,
            'data_timestamp': datetime.now(),  # datetime object
        }

        # Should not raise an exception
        assessment = boundary.assess_knowledge_state(signal, context)
        assert assessment is not None
