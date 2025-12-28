"""
Comprehensive Unit Tests for ReflectionEngine
================================================

Tests the AI's self-critique and learning system that implements
the Reflexion pattern for continuous improvement.

Run: python -m pytest tests/cognitive/test_reflection_engine.py -v
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch


class TestReflectionEngineInitialization:
    """Tests for ReflectionEngine initialization."""

    def test_default_initialization(self):
        """Test that ReflectionEngine initializes properly."""
        from cognitive.reflection_engine import ReflectionEngine

        engine = ReflectionEngine()

        assert engine._reflections == []
        assert engine._max_reflections == 100
        assert engine._episodic_memory is None  # Lazy-loaded
        assert engine._semantic_memory is None
        assert engine._self_model is None

    def test_lazy_loading_dependencies(self):
        """Test that dependencies are lazy-loaded."""
        from cognitive.reflection_engine import ReflectionEngine

        engine = ReflectionEngine()

        # Access properties to trigger loading
        _ = engine.episodic_memory
        _ = engine.semantic_memory
        _ = engine.self_model

        # Should now be loaded
        assert engine._episodic_memory is not None
        assert engine._semantic_memory is not None
        assert engine._self_model is not None


class TestReflectionDataclass:
    """Tests for the Reflection dataclass."""

    def test_reflection_creation(self):
        """Test creating a Reflection object."""
        from cognitive.reflection_engine import Reflection

        reflection = Reflection(
            scope='episode',
            summary='Test reflection',
            what_went_well=['Good timing'],
            what_went_wrong=['Poor exit'],
            lessons=['Wait for confirmation'],
        )

        assert reflection.scope == 'episode'
        assert reflection.summary == 'Test reflection'
        assert len(reflection.what_went_well) == 1
        assert len(reflection.what_went_wrong) == 1

    def test_reflection_to_dict(self):
        """Test Reflection serialization."""
        from cognitive.reflection_engine import Reflection

        reflection = Reflection(
            scope='daily',
            summary='Daily review',
            what_went_well=['Followed rules'],
            lessons=['Trust the system'],
            confidence_adjustment=-0.1,
        )

        d = reflection.to_dict()

        assert d['scope'] == 'daily'
        assert d['summary'] == 'Daily review'
        assert d['confidence_adjustment'] == -0.1
        assert 'timestamp' in d

    def test_reflection_default_values(self):
        """Test Reflection default values."""
        from cognitive.reflection_engine import Reflection

        reflection = Reflection(scope='episode')

        assert reflection.summary == ''
        assert reflection.what_went_well == []
        assert reflection.what_went_wrong == []
        assert reflection.root_causes == []
        assert reflection.lessons == []
        assert reflection.action_items == []
        assert reflection.confidence_adjustment == 0.0


class TestReflectOnEpisode:
    """Tests for episode-level reflection."""

    def test_reflect_on_winning_episode(self):
        """Test reflection on a winning trade episode."""
        from cognitive.reflection_engine import ReflectionEngine
        from cognitive.episodic_memory import Episode, EpisodeOutcome

        engine = ReflectionEngine()

        # Create a mock winning episode
        episode = Mock()
        episode.episode_id = 'test-win-123'
        episode.outcome = EpisodeOutcome.WIN
        episode.pnl = 200.0
        episode.r_multiple = 1.5
        episode.reasoning_trace = ['Entry looked good', 'Trend confirmed']
        episode.concerns_noted = []
        episode.confidence_levels = {'initial': 0.8}
        episode.market_context = {'regime': 'BULL'}
        episode.signal_context = {'strategy': 'ibs_rsi'}
        episode.decision_mode = 'hybrid'

        reflection = engine.reflect_on_episode(episode)

        assert reflection.scope == 'episode'
        assert len(reflection.what_went_well) > 0
        assert 'won' in reflection.what_went_well[0].lower() or len(reflection.lessons) > 0

    def test_reflect_on_losing_episode(self):
        """Test reflection on a losing trade episode."""
        from cognitive.reflection_engine import ReflectionEngine
        from cognitive.episodic_memory import Episode, EpisodeOutcome

        engine = ReflectionEngine()

        # Create a mock losing episode
        episode = Mock()
        episode.episode_id = 'test-loss-456'
        episode.outcome = EpisodeOutcome.LOSS
        episode.pnl = -100.0
        episode.r_multiple = -0.8
        episode.reasoning_trace = ['Took the trade despite warning signs']
        episode.concerns_noted = ['Volume was low', 'VIX spiking']
        episode.confidence_levels = {'initial': 0.75}
        episode.market_context = {'regime': 'CHOPPY'}
        episode.signal_context = {'strategy': 'turtle_soup'}
        episode.decision_mode = 'fast'

        reflection = engine.reflect_on_episode(episode)

        assert reflection.scope == 'episode'
        assert len(reflection.what_went_wrong) > 0
        # Should identify proceeding despite concerns as a problem
        assert any('concern' in item.lower() for item in reflection.what_went_wrong + reflection.root_causes)

    def test_reflect_on_stand_down_episode(self):
        """Test reflection on a stand-down decision."""
        from cognitive.reflection_engine import ReflectionEngine
        from cognitive.episodic_memory import Episode, EpisodeOutcome

        engine = ReflectionEngine()

        # Create a mock stand-down episode
        episode = Mock()
        episode.episode_id = 'test-standdown-789'
        episode.outcome = EpisodeOutcome.STAND_DOWN
        episode.pnl = 0.0
        episode.r_multiple = 0.0
        episode.reasoning_trace = ['Decided not to act']
        episode.concerns_noted = ['Confidence too low']
        episode.confidence_levels = {'initial': 0.35}
        episode.market_context = {'regime': 'BEAR'}
        episode.signal_context = {'strategy': 'ibs_rsi'}
        episode.decision_mode = 'slow'

        reflection = engine.reflect_on_episode(episode)

        assert reflection.scope == 'episode'
        # Should recognize standing down as potentially wise
        assert any('stand down' in item.lower() or 'cautious' in item.lower()
                   for item in reflection.what_went_well + reflection.lessons)


class TestPeriodicReflection:
    """Tests for periodic (daily/weekly) reflection."""

    def test_periodic_reflection_no_episodes(self):
        """Test daily reflection runs and produces a valid scope."""
        from cognitive.reflection_engine import ReflectionEngine

        engine = ReflectionEngine()

        reflection = engine.periodic_reflection(lookback_hours=24)

        assert reflection.scope == 'daily'
        # Summary will vary depending on whether episodes exist in persistent storage
        assert isinstance(reflection.summary, str)

    def test_periodic_reflection_generates_summary(self):
        """Test that periodic reflection generates a summary."""
        from cognitive.reflection_engine import ReflectionEngine

        engine = ReflectionEngine()

        # Note: This test relies on episodic_memory having data
        # In a real test, we'd mock the episodic_memory
        reflection = engine.periodic_reflection(lookback_hours=24)

        assert reflection.scope == 'daily'
        assert isinstance(reflection.summary, str)


class TestConsolidateLearnings:
    """Tests for weekly learning consolidation."""

    def test_consolidate_learnings(self):
        """Test weekly consolidation runs without errors."""
        from cognitive.reflection_engine import ReflectionEngine

        engine = ReflectionEngine()

        result = engine.consolidate_learnings()

        assert 'new_rules_extracted' in result
        assert 'rules_pruned' in result
        assert 'new_self_description' in result


class TestIntrospection:
    """Tests for engine introspection."""

    def test_introspect_returns_report(self):
        """Test that introspect returns a formatted report."""
        from cognitive.reflection_engine import ReflectionEngine

        engine = ReflectionEngine()
        report = engine.introspect()

        assert "Reflection Engine" in report
        assert "reflections" in report.lower()

    def test_introspect_after_reflections(self):
        """Test introspection after performing some reflections."""
        from cognitive.reflection_engine import ReflectionEngine, Reflection

        engine = ReflectionEngine()

        # Manually add some reflections
        engine._reflections.append(Reflection(
            scope='episode',
            summary='Test',
            lessons=['Lesson 1', 'Lesson 2'],
        ))
        engine._reflections.append(Reflection(
            scope='episode',
            summary='Test 2',
            lessons=['Lesson 3'],
        ))

        report = engine.introspect()

        assert "2 reflections" in report
        assert "3 lessons" in report


class TestHelperMethods:
    """Tests for internal helper methods."""

    def test_describe_context(self):
        """Test context description generation."""
        from cognitive.reflection_engine import ReflectionEngine

        engine = ReflectionEngine()

        # Create a mock episode with context
        episode = Mock()
        episode.market_context = {'regime': 'BULL'}
        episode.signal_context = {'strategy': 'ibs_rsi'}

        description = engine._describe_context(episode)

        assert 'BULL' in description or 'ibs_rsi' in description

    def test_describe_context_empty(self):
        """Test context description with empty context."""
        from cognitive.reflection_engine import ReflectionEngine

        engine = ReflectionEngine()

        episode = Mock()
        episode.market_context = {}
        episode.signal_context = {}

        description = engine._describe_context(episode)

        assert 'unknown' in description.lower()


class TestReflectionHistory:
    """Tests for reflection history management."""

    def test_reflection_history_limit(self):
        """Test that reflection history is limited."""
        from cognitive.reflection_engine import ReflectionEngine, Reflection
        from cognitive.episodic_memory import EpisodeOutcome

        engine = ReflectionEngine()
        engine._max_reflections = 5

        # Add more than the limit
        for i in range(10):
            episode = Mock()
            episode.episode_id = f'test-{i}'
            episode.outcome = EpisodeOutcome.WIN
            episode.pnl = 100.0
            episode.r_multiple = 1.0
            episode.reasoning_trace = []
            episode.concerns_noted = []
            episode.confidence_levels = {'initial': 0.7}
            episode.market_context = {'regime': 'BULL'}
            episode.signal_context = {'strategy': 'ibs_rsi'}
            episode.decision_mode = 'fast'

            engine.reflect_on_episode(episode)

        # Should be capped at max_reflections
        assert len(engine._reflections) <= engine._max_reflections


class TestApplyLearnings:
    """Tests for the _apply_learnings method."""

    def test_apply_learnings_updates_memories(self):
        """Test that learnings are applied to memory systems."""
        from cognitive.reflection_engine import ReflectionEngine, Reflection
        from cognitive.episodic_memory import EpisodeOutcome

        engine = ReflectionEngine()

        # Create a winning episode
        episode = Mock()
        episode.episode_id = 'test-apply-123'
        episode.outcome = EpisodeOutcome.WIN
        episode.pnl = 150.0
        episode.r_multiple = 1.2
        episode.market_context = {'regime': 'BULL'}
        episode.signal_context = {'strategy': 'ibs_rsi'}

        reflection = Reflection(
            scope='episode',
            what_went_well=['High confidence justified'],
            lessons=['Trust high-confidence signals'],
        )

        # Should not raise any exceptions
        engine._apply_learnings(episode, reflection)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
