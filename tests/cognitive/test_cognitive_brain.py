"""
Comprehensive Unit Tests for CognitiveBrain
=============================================

Tests the main orchestration layer of the cognitive architecture.

Run: python -m pytest tests/cognitive/test_cognitive_brain.py -v
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock


class TestCognitiveBrainInitialization:
    """Tests for CognitiveBrain initialization and configuration."""

    def test_default_initialization(self):
        """Test that CognitiveBrain initializes with default parameters."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()
        assert brain.min_confidence_to_act == 0.5
        assert brain.max_processing_time_ms == 5000
        assert brain._initialized == False
        assert brain._decision_count == 0

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain(
            min_confidence_to_act=0.7,
            max_processing_time_ms=3000
        )
        assert brain.min_confidence_to_act == 0.7
        assert brain.max_processing_time_ms == 3000

    def test_lazy_loading_components(self):
        """Test that components are not loaded until accessed."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()
        # Components should be None before first access
        assert brain._workspace is None
        assert brain._governor is None
        assert brain._self_model is None

    def test_singleton_factory(self):
        """Test the get_cognitive_brain factory returns singleton."""
        from cognitive.cognitive_brain import get_cognitive_brain, _cognitive_brain
        import cognitive.cognitive_brain as cb_module

        # Reset singleton for test
        cb_module._cognitive_brain = None

        brain1 = get_cognitive_brain()
        brain2 = get_cognitive_brain()
        assert brain1 is brain2


class TestCognitiveBrainDeliberation:
    """Tests for the deliberation process."""

    def test_basic_deliberation_act(self):
        """Test deliberation with high confidence leads to ACT decision."""
        from cognitive.cognitive_brain import CognitiveBrain, DecisionType

        brain = CognitiveBrain(min_confidence_to_act=0.5)

        decision = brain.deliberate(
            signal={
                'symbol': 'AAPL',
                'strategy': 'ibs_rsi',
                'entry_price': 150.0,
                'stop_loss': 145.0,
            },
            context={
                'regime': 'BULL',
                'regime_confidence': 0.8,
                'vix': 15,
            },
            fast_confidence=0.75,
        )

        assert decision is not None
        assert decision.decision_type == DecisionType.ACT
        assert decision.should_act == True
        assert decision.confidence >= 0.5
        assert decision.episode_id is not None
        assert len(decision.reasoning_trace) > 0

    def test_basic_deliberation_stand_down(self):
        """Test deliberation with low confidence leads to STAND_DOWN decision."""
        from cognitive.cognitive_brain import CognitiveBrain, DecisionType

        brain = CognitiveBrain(min_confidence_to_act=0.6)

        decision = brain.deliberate(
            signal={
                'symbol': 'TSLA',
                'strategy': 'turtle_soup',
                'entry_price': 250.0,
                'stop_loss': 240.0,
            },
            context={
                'regime': 'CHOPPY',
                'vix': 35,
            },
            fast_confidence=0.35,
        )

        assert decision.decision_type == DecisionType.STAND_DOWN
        assert decision.should_act == False
        assert decision.action is None

    def test_deliberation_increments_decision_count(self):
        """Test that decision count increments with each deliberation."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()
        initial_count = brain._decision_count

        brain.deliberate(
            signal={'symbol': 'TEST'},
            context={'regime': 'BULL'},
            fast_confidence=0.7,
        )

        assert brain._decision_count == initial_count + 1

    def test_deliberation_records_processing_time(self):
        """Test that deliberation measures processing time."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()

        decision = brain.deliberate(
            signal={'symbol': 'TEST'},
            context={'regime': 'BULL'},
            fast_confidence=0.7,
        )

        assert decision.processing_time_ms >= 0

    def test_deliberation_with_missing_context(self):
        """Test deliberation handles sparse context gracefully."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()

        # Minimal signal and context
        decision = brain.deliberate(
            signal={'symbol': 'XYZ'},
            context={},
            fast_confidence=0.6,
        )

        # Should still produce a valid decision
        assert decision is not None
        assert decision.episode_id is not None


class TestCognitiveBrainLearning:
    """Tests for learning and outcome feedback."""

    def test_learn_from_winning_outcome(self):
        """Test learning from a winning trade outcome."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()

        # First, make a decision
        decision = brain.deliberate(
            signal={'symbol': 'MSFT', 'strategy': 'ibs_rsi'},
            context={'regime': 'BULL'},
            fast_confidence=0.7,
        )

        # Then report the outcome
        brain.learn_from_outcome(
            episode_id=decision.episode_id,
            outcome={
                'won': True,
                'pnl': 250.0,
                'r_multiple': 1.5,
            }
        )

        # Verify the episode was updated
        episode = brain.episodic_memory.get_episode(decision.episode_id)
        assert episode is not None
        assert episode.pnl == 250.0

    def test_learn_from_losing_outcome(self):
        """Test learning from a losing trade outcome."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()

        decision = brain.deliberate(
            signal={'symbol': 'AMZN', 'strategy': 'turtle_soup'},
            context={'regime': 'BEAR'},
            fast_confidence=0.65,
        )

        brain.learn_from_outcome(
            episode_id=decision.episode_id,
            outcome={
                'won': False,
                'pnl': -100.0,
                'r_multiple': -0.8,
            }
        )

        episode = brain.episodic_memory.get_episode(decision.episode_id)
        assert episode is not None
        assert episode.pnl == -100.0


class TestCognitiveBrainThinkDeeper:
    """Tests for the System 2 deeper thinking pathway."""

    def test_think_deeper_returns_adjusted_confidence(self):
        """Test that think_deeper adjusts confidence and adds reasoning."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()

        new_confidence, additional_reasoning = brain.think_deeper(
            signal={'symbol': 'NVDA', 'strategy': 'ibs_rsi'},
            context={'regime': 'BULL'},
            current_confidence=0.6,
        )

        assert 0 <= new_confidence <= 1
        assert isinstance(additional_reasoning, list)

    def test_think_deeper_caps_confidence(self):
        """Test that think_deeper keeps confidence between 0 and 1."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()

        # Test with very high starting confidence
        new_conf, _ = brain.think_deeper(
            signal={'strategy': 'test'},
            context={'regime': 'BULL'},
            current_confidence=0.99,
        )
        assert new_conf <= 1.0

        # Test with very low starting confidence
        new_conf, _ = brain.think_deeper(
            signal={'strategy': 'test'},
            context={'regime': 'CHOPPY'},
            current_confidence=0.05,
        )
        assert new_conf >= 0.0


class TestCognitiveBrainConsolidation:
    """Tests for daily and weekly consolidation processes."""

    def test_daily_consolidation(self):
        """Test daily consolidation runs without errors."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()

        # Make some decisions first
        for i in range(3):
            brain.deliberate(
                signal={'symbol': f'TEST{i}'},
                context={'regime': 'BULL'},
                fast_confidence=0.7,
            )

        results = brain.daily_consolidation()

        assert 'reflection' in results
        assert 'curiosity' in results
        assert 'edges_count' in results
        assert brain._last_maintenance is not None

    def test_weekly_consolidation(self):
        """Test weekly consolidation performs cleanup."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()

        results = brain.weekly_consolidation()

        assert 'rules_pruned' in results


class TestCognitiveBrainIntrospection:
    """Tests for status and introspection methods."""

    def test_get_status(self):
        """Test get_status returns valid status dict."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()
        brain.deliberate(
            signal={'symbol': 'TEST'},
            context={'regime': 'BULL'},
            fast_confidence=0.7,
        )

        status = brain.get_status()

        assert status['initialized'] == True
        assert status['decision_count'] >= 1
        assert 'components' in status
        assert 'workspace' in status['components']
        assert 'self_model' in status['components']

    def test_introspect(self):
        """Test introspect returns formatted report."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()
        report = brain.introspect()

        assert "COGNITIVE BRAIN INTROSPECTION" in report
        assert len(report) > 100


class TestCognitiveBrainSizeMultiplier:
    """Tests for position size calculation."""

    def test_size_multiplier_high_confidence(self):
        """Test that high confidence leads to full position size."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()
        multiplier = brain._calculate_size_multiplier(0.95)

        assert multiplier == 0.95

    def test_size_multiplier_low_confidence(self):
        """Test that low confidence leads to minimum position size."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()
        multiplier = brain._calculate_size_multiplier(0.1)

        # Should be capped at minimum (0.25)
        assert multiplier == 0.25

    def test_size_multiplier_boundary(self):
        """Test size multiplier at confidence boundaries."""
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()

        # Test at 100% confidence
        assert brain._calculate_size_multiplier(1.0) == 1.0

        # Test at 0% confidence
        assert brain._calculate_size_multiplier(0.0) == 0.25


class TestCognitiveDecision:
    """Tests for the CognitiveDecision dataclass."""

    def test_cognitive_decision_to_dict(self):
        """Test CognitiveDecision serialization."""
        from cognitive.cognitive_brain import CognitiveDecision, DecisionType

        decision = CognitiveDecision(
            decision_type=DecisionType.ACT,
            should_act=True,
            action={'type': 'trade'},
            confidence=0.8,
            reasoning_trace=['Step 1', 'Step 2'],
            concerns=['Concern A'],
            knowledge_gaps=['Gap 1'],
            invalidators=['If price drops below X'],
            episode_id='test-123',
            decision_mode='fast',
            processing_time_ms=50,
        )

        d = decision.to_dict()

        assert d['decision_type'] == 'act'
        assert d['should_act'] == True
        assert d['confidence'] == 0.8
        assert d['episode_id'] == 'test-123'
        assert len(d['reasoning_trace']) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
