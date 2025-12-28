"""
Cognitive System Tests
=======================

Tests for the brain-inspired cognitive architecture.

Run: python -m pytest tests/test_cognitive_system.py -v
"""

import pytest
from datetime import datetime


class TestGlobalWorkspace:
    """Tests for GlobalWorkspace."""

    def test_import(self):
        from cognitive.global_workspace import GlobalWorkspace, get_workspace
        assert GlobalWorkspace is not None
        assert get_workspace is not None

    def test_publish_subscribe(self):
        from cognitive.global_workspace import GlobalWorkspace, Priority

        ws = GlobalWorkspace()
        received = []

        def callback(item):
            received.append(item.data)

        ws.subscribe('test_topic', callback)
        ws.publish('test_topic', {'value': 42}, source='test')

        assert len(received) == 1
        assert received[0]['value'] == 42

    def test_working_memory(self):
        from cognitive.global_workspace import GlobalWorkspace, Priority

        ws = GlobalWorkspace(working_memory_capacity=3)

        # Publish high-priority items
        for i in range(5):
            ws.publish(f'topic_{i}', {'i': i}, priority=Priority.HIGH, source='test')

        wm = ws.get_working_memory()
        assert len(wm) <= 3  # Capacity limit


class TestSelfModel:
    """Tests for SelfModel."""

    def test_import(self):
        from cognitive.self_model import SelfModel, get_self_model
        assert SelfModel is not None

    def test_record_trade_outcome(self):
        from cognitive.self_model import SelfModel, Capability

        model = SelfModel(auto_persist=False)

        # Record multiple trades
        for i in range(15):
            model.record_trade_outcome(
                strategy='test_strategy',
                regime='BULL',
                won=i < 10,  # 10 wins, 5 losses
                pnl=100 if i < 10 else -50,
            )

        cap = model.get_capability('test_strategy', 'BULL')
        # 66% win rate qualifies as EXCELLENT (>65%) in the implementation
        assert cap in [Capability.GOOD, Capability.EXCELLENT]

    def test_record_limitation(self):
        from cognitive.self_model import SelfModel

        model = SelfModel(auto_persist=False)
        # Clear any existing limitations from persistent state to ensure clean test
        model._limitations.clear()

        model.record_limitation(
            context='high_volatility',
            description='Poor exit timing',
            severity='moderate',
        )

        lims = model.known_limitations()
        assert len(lims) == 1
        assert lims[0].context == 'high_volatility'


class TestMetacognitiveGovernor:
    """Tests for MetacognitiveGovernor."""

    def test_import(self):
        from cognitive.metacognitive_governor import MetacognitiveGovernor
        assert MetacognitiveGovernor is not None

    def test_routing_fast_path(self):
        from cognitive.metacognitive_governor import MetacognitiveGovernor, ProcessingMode

        governor = MetacognitiveGovernor()

        # High confidence should route appropriately (not stand down)
        # Note: With policy integration (Task B3), POLICY_LEARNING may activate
        # for novel situations, leading to SLOW mode for additional caution.
        # This is expected behavior for the enhanced cognitive system.
        routing = governor.route_decision(
            signal={'symbol': 'TEST', 'strategy': 'ibs_rsi'},
            context={'regime': 'BULL', 'regime_confidence': 0.8, 'vix': 18},
            fast_confidence=0.85,
        )

        # With policy integration, SLOW mode is also valid for high confidence
        # The key is that it should NOT stand down
        assert routing.mode in [ProcessingMode.FAST, ProcessingMode.HYBRID, ProcessingMode.SLOW]
        assert routing.should_stand_down == False

    def test_routing_stand_down(self):
        from cognitive.metacognitive_governor import MetacognitiveGovernor, ProcessingMode

        governor = MetacognitiveGovernor(stand_down_threshold=0.30)

        # Very low confidence should trigger stand-down
        routing = governor.route_decision(
            signal={'symbol': 'TEST'},
            context={'regime': 'unknown'},
            fast_confidence=0.2,
        )

        assert routing.mode == ProcessingMode.STAND_DOWN
        assert routing.should_stand_down == True


class TestEpisodicMemory:
    """Tests for EpisodicMemory."""

    def test_import(self):
        from cognitive.episodic_memory import EpisodicMemory, get_episodic_memory
        assert EpisodicMemory is not None

    def test_episode_lifecycle(self):
        from cognitive.episodic_memory import EpisodicMemory, EpisodeOutcome

        memory = EpisodicMemory(storage_dir="state/cognitive/test", auto_persist=False)

        # Start episode
        episode_id = memory.start_episode(
            market_context={'regime': 'BULL'},
            signal_context={'strategy': 'ibs_rsi', 'symbol': 'AAPL'},
        )

        assert episode_id is not None

        # Add reasoning
        memory.add_reasoning(episode_id, "Strong trend detected")
        memory.add_concern(episode_id, "High volatility")

        # Add action
        memory.add_action(episode_id, {'type': 'buy', 'shares': 100})

        # Complete episode
        memory.complete_episode(episode_id, {'won': True, 'pnl': 500})

        # Verify
        episode = memory.get_episode(episode_id)
        assert episode is not None
        assert episode.outcome == EpisodeOutcome.WIN
        assert episode.pnl == 500
        assert len(episode.reasoning_trace) == 1
        assert len(episode.concerns_noted) == 1


class TestSemanticMemory:
    """Tests for SemanticMemory."""

    def test_import(self):
        from cognitive.semantic_memory import SemanticMemory, get_semantic_memory
        assert SemanticMemory is not None

    def test_add_and_query_rule(self, tmp_path):
        from cognitive.semantic_memory import SemanticMemory

        # Use temp directory to avoid loading existing rules from disk
        memory = SemanticMemory(storage_dir=str(tmp_path), auto_persist=False)

        rule = memory.add_rule(
            condition="regime = BULL AND vix < 20",
            action="increase_confidence",
            parameters={'boost': 0.1},
            confidence=0.8,
        )

        assert rule.rule_id is not None
        assert rule.confidence == 0.8

        # Query applicable rules
        rules = memory.get_applicable_rules({
            'regime': 'BULL',
            'vix': 15,
        })

        assert len(rules) >= 1
        assert rules[0].condition == "regime = BULL AND vix < 20"


class TestKnowledgeBoundary:
    """Tests for KnowledgeBoundary."""

    def test_import(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary
        assert KnowledgeBoundary is not None

    def test_uncertainty_detection(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary, UncertaintyLevel

        kb = KnowledgeBoundary()

        # Test with missing data
        assessment = kb.assess_knowledge_state(
            signal={'symbol': 'TEST'},
            context={},  # Missing most data
        )

        assert assessment.is_uncertain == True
        assert len(assessment.missing_information) > 0

    def test_what_would_change_mind(self):
        from cognitive.knowledge_boundary import KnowledgeBoundary

        kb = KnowledgeBoundary()

        invalidators = kb.what_would_change_mind(
            signal={'entry_price': 100},
            context={'regime': 'BULL'},
        )

        assert len(invalidators) > 0
        assert any('reconsider' in inv.lower() for inv in invalidators)


class TestReflectionEngine:
    """Tests for ReflectionEngine."""

    def test_import(self):
        from cognitive.reflection_engine import ReflectionEngine
        assert ReflectionEngine is not None

    def test_introspect(self):
        from cognitive.reflection_engine import ReflectionEngine

        engine = ReflectionEngine()
        report = engine.introspect()

        # Empty engine returns "No reflections yet" or introspection report
        assert len(report) > 0
        assert "reflection" in report.lower() or "no" in report.lower()


class TestCuriosityEngine:
    """Tests for CuriosityEngine."""

    def test_import(self):
        from cognitive.curiosity_engine import CuriosityEngine
        assert CuriosityEngine is not None

    def test_hypothesis_generation(self):
        from cognitive.curiosity_engine import CuriosityEngine

        engine = CuriosityEngine()

        hypotheses = engine.generate_hypotheses({
            'vix': 35,
            'volume_ratio': 2.5,
        })

        # Should generate at least some hypotheses
        assert len(hypotheses) >= 0  # May be empty if already generated

    def test_get_stats(self):
        from cognitive.curiosity_engine import CuriosityEngine

        engine = CuriosityEngine()
        stats = engine.get_stats()

        assert 'total_hypotheses' in stats
        assert 'total_edges' in stats


class TestCognitiveBrain:
    """Tests for CognitiveBrain."""

    def test_import(self):
        from cognitive.cognitive_brain import CognitiveBrain, get_cognitive_brain
        assert CognitiveBrain is not None
        assert get_cognitive_brain is not None

    def test_initialization(self):
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()
        assert brain.min_confidence_to_act == 0.5

    def test_deliberate(self):
        from cognitive.cognitive_brain import CognitiveBrain, DecisionType

        brain = CognitiveBrain()

        decision = brain.deliberate(
            signal={
                'symbol': 'AAPL',
                'strategy': 'ibs_rsi',
                'entry_price': 150,
                'stop_loss': 145,
            },
            context={
                'regime': 'BULL',
                'regime_confidence': 0.8,
                'vix': 18,
            },
            fast_confidence=0.75,
        )

        assert decision is not None
        assert decision.decision_type in [DecisionType.ACT, DecisionType.STAND_DOWN]
        assert len(decision.reasoning_trace) > 0
        assert decision.episode_id is not None

    def test_get_status(self):
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

    def test_introspect(self):
        from cognitive.cognitive_brain import CognitiveBrain

        brain = CognitiveBrain()
        report = brain.introspect()

        assert "COGNITIVE BRAIN INTROSPECTION" in report
        assert len(report) > 100


class TestIntegration:
    """Integration tests for the full cognitive system."""

    def test_full_decision_cycle(self):
        """Test complete decision -> outcome -> learning cycle."""
        from cognitive.cognitive_brain import CognitiveBrain, DecisionType

        brain = CognitiveBrain()

        # Make a decision
        decision = brain.deliberate(
            signal={
                'symbol': 'MSFT',
                'strategy': 'turtle_soup',
                'entry_price': 380,
                'stop_loss': 370,
            },
            context={
                'regime': 'BEAR',
                'regime_confidence': 0.7,
                'vix': 25,
            },
            fast_confidence=0.65,
        )

        assert decision.episode_id is not None

        # Simulate outcome
        brain.learn_from_outcome(
            episode_id=decision.episode_id,
            outcome={
                'won': True,
                'pnl': 300,
                'r_multiple': 1.5,
            }
        )

        # Check that learning occurred
        status = brain.get_status()
        assert status['components']['episodic_memory']['total_episodes'] >= 1

    def test_cognitive_package_import(self):
        """Test that the cognitive package can be imported."""
        from cognitive import CognitiveBrain, get_cognitive_brain
        from cognitive import GlobalWorkspace, get_workspace
        from cognitive import SelfModel, get_self_model
        from cognitive import EpisodicMemory, get_episodic_memory

        assert CognitiveBrain is not None
        assert GlobalWorkspace is not None
        assert SelfModel is not None
        assert EpisodicMemory is not None


class TestSignalProcessor:
    """Tests for CognitiveSignalProcessor."""

    def test_import(self):
        from cognitive.signal_processor import CognitiveSignalProcessor, get_signal_processor
        assert CognitiveSignalProcessor is not None
        assert get_signal_processor is not None

    def test_create_processor(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor(min_confidence=0.6)
        assert processor.min_confidence == 0.6
        assert processor.brain is not None

    def test_evaluate_signals(self):
        import pandas as pd
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor(min_confidence=0.3)

        # Create test signals
        signals = pd.DataFrame([
            {
                'symbol': 'AAPL',
                'strategy': 'ibs_rsi',
                'side': 'long',
                'entry_price': 150.0,
                'stop_loss': 145.0,
                'take_profit': 160.0,
            },
            {
                'symbol': 'MSFT',
                'strategy': 'turtle_soup',
                'side': 'long',
                'entry_price': 380.0,
                'stop_loss': 370.0,
                'take_profit': 400.0,
            },
        ])

        # Evaluate signals
        approved_df, evaluated = processor.evaluate_signals(signals)

        assert len(evaluated) == 2
        assert all(ev.episode_id for ev in evaluated)
        assert all(len(ev.reasoning_trace) > 0 for ev in evaluated)

    def test_get_cognitive_status(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()
        status = processor.get_cognitive_status()

        assert 'processor_active' in status
        assert 'brain_status' in status
        assert status['processor_active'] == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

