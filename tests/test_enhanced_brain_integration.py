"""
Integration Tests for Enhanced Autonomous Brain.

Tests the full integration of:
- EnhancedResearchEngine (VectorBT, Alphalens, AlphaFactory)
- EnhancedAutonomousBrain (unified discovery system)
- KobeBrainGraph (LangGraph state machine)
- RAGEvaluator (LLM quality tracking)

Created: 2026-01-07
"""

import pytest
import json
from pathlib import Path
from datetime import datetime


class TestEnhancedResearchEngine:
    """Test EnhancedResearchEngine capabilities."""

    def test_initialization(self):
        """Test enhanced research engine initializes correctly."""
        from autonomous.enhanced_research import EnhancedResearchEngine

        engine = EnhancedResearchEngine()
        assert engine is not None
        assert hasattr(engine, 'alpha_discoveries')
        assert hasattr(engine, 'run_vectorbt_alpha_sweep')

    def test_vectorbt_alpha_sweep_interface(self):
        """Test VectorBT alpha sweep interface (may not have VectorBT installed)."""
        from autonomous.enhanced_research import EnhancedResearchEngine

        engine = EnhancedResearchEngine()
        result = engine.run_vectorbt_alpha_sweep(min_sharpe=0.5)

        # Should return a result dict even if VectorBT not installed
        assert isinstance(result, dict)
        assert 'status' in result

        # If error, should have error message
        if result['status'] == 'error':
            assert 'error' in result
        # If success, should have metrics
        elif result['status'] == 'success':
            assert 'alphas_discovered' in result

    def test_alphalens_validation_interface(self):
        """Test Alphalens validation interface."""
        from autonomous.enhanced_research import EnhancedResearchEngine

        engine = EnhancedResearchEngine()

        # Try to validate a non-existent alpha (should fail gracefully)
        result = engine.validate_alpha_with_alphalens("nonexistent_alpha")

        assert isinstance(result, dict)
        assert 'status' in result
        assert result['status'] == 'error'
        assert 'error' in result

    def test_get_top_alphas(self):
        """Test getting top alphas."""
        from autonomous.enhanced_research import EnhancedResearchEngine

        engine = EnhancedResearchEngine()
        top = engine.get_top_alphas(n=5)

        assert isinstance(top, list)
        # May be empty if no alphas discovered yet
        assert len(top) <= 5

    def test_enhanced_summary(self):
        """Test enhanced research summary."""
        from autonomous.enhanced_research import EnhancedResearchEngine

        engine = EnhancedResearchEngine()
        summary = engine.get_enhanced_research_summary()

        assert isinstance(summary, dict)
        assert 'alpha_mining' in summary
        assert 'total_discoveries' in summary['alpha_mining']


class TestEnhancedAutonomousBrain:
    """Test EnhancedAutonomousBrain integration."""

    def test_initialization(self):
        """Test enhanced brain initializes correctly."""
        from autonomous.enhanced_brain import EnhancedAutonomousBrain

        brain = EnhancedAutonomousBrain(use_langgraph=False)
        assert brain is not None
        assert brain.VERSION == "2.0.0"
        assert hasattr(brain, 'research')
        assert hasattr(brain, 'rag_evaluator')

    def test_enhanced_research_engine_integration(self):
        """Test that enhanced brain uses EnhancedResearchEngine."""
        from autonomous.enhanced_brain import EnhancedAutonomousBrain
        from autonomous.enhanced_research import EnhancedResearchEngine

        brain = EnhancedAutonomousBrain()
        assert isinstance(brain.research, EnhancedResearchEngine)

    def test_langgraph_initialization(self):
        """Test LangGraph initialization."""
        from autonomous.enhanced_brain import EnhancedAutonomousBrain

        # Try with LangGraph enabled
        brain = EnhancedAutonomousBrain(use_langgraph=True)

        # Should have brain_graph attribute
        assert hasattr(brain, 'brain_graph')

        # May be None if LangGraph not installed
        if brain.brain_graph is not None:
            assert brain.use_langgraph is True

    def test_rag_evaluator_initialization(self):
        """Test RAG evaluator initialization."""
        from autonomous.enhanced_brain import EnhancedAutonomousBrain

        brain = EnhancedAutonomousBrain()

        # Should have rag_evaluator attribute
        assert hasattr(brain, 'rag_evaluator')

        # May be None if dependencies not installed
        # Just check it doesn't crash

    def test_enhanced_status(self):
        """Test enhanced status includes all components."""
        from autonomous.enhanced_brain import EnhancedAutonomousBrain

        brain = EnhancedAutonomousBrain()
        status = brain.get_status()

        # Should have base status
        assert 'awareness' in status
        assert 'scheduler' in status

        # Should have enhanced status
        assert 'enhanced_version' in status
        assert 'alpha_mining' in status
        assert 'langgraph' in status
        assert 'rag_evaluator' in status

        # Version should be 2.0.0
        assert status['enhanced_version'] == "2.0.0"

    def test_discovery_checking(self):
        """Test unified discovery checking."""
        from autonomous.enhanced_brain import EnhancedAutonomousBrain

        brain = EnhancedAutonomousBrain()

        # Should be able to check discoveries without crashing
        discoveries = brain._check_for_discoveries()
        assert isinstance(discoveries, list)

    def test_single_cycle(self):
        """Test running a single brain cycle."""
        from autonomous.enhanced_brain import EnhancedAutonomousBrain

        brain = EnhancedAutonomousBrain()

        # Run single cycle
        result = brain.run_single_cycle()

        assert isinstance(result, dict)
        assert 'timestamp' in result
        assert 'phase' in result
        assert 'work_mode' in result


class TestBrainGraphIntegration:
    """Test LangGraph brain integration."""

    def test_brain_graph_singleton(self):
        """Test brain graph singleton."""
        try:
            from cognitive.brain_graph import get_brain_graph, HAS_LANGGRAPH

            if not HAS_LANGGRAPH:
                pytest.skip("LangGraph not installed")

            brain_graph = get_brain_graph()
            assert brain_graph is not None

            # Should be singleton
            brain_graph2 = get_brain_graph()
            assert brain_graph is brain_graph2

        except ImportError:
            pytest.skip("LangGraph not available")

    def test_brain_cycle_with_langgraph(self):
        """Test running brain cycle with LangGraph."""
        try:
            from autonomous.enhanced_brain import EnhancedAutonomousBrain

            brain = EnhancedAutonomousBrain(use_langgraph=True)

            if brain.brain_graph is None:
                pytest.skip("LangGraph not available")

            # Try running a cycle
            result = brain.think_with_langgraph()

            assert isinstance(result, dict)
            # Should indicate LangGraph was used
            if 'langgraph_enabled' in result:
                assert result['langgraph_enabled'] is True

        except ImportError:
            pytest.skip("LangGraph not available")


class TestRAGEvaluatorIntegration:
    """Test RAG evaluator integration."""

    def test_rag_evaluator_singleton(self):
        """Test RAG evaluator singleton."""
        try:
            from cognitive.rag_evaluator import get_rag_evaluator

            evaluator = get_rag_evaluator()
            assert evaluator is not None

            # Should be singleton
            evaluator2 = get_rag_evaluator()
            assert evaluator is evaluator2

        except ImportError:
            pytest.skip("RAGEvaluator not available")

    def test_explanation_generation(self):
        """Test generating explanations."""
        try:
            from cognitive.rag_evaluator import get_rag_evaluator

            evaluator = get_rag_evaluator()

            trade_context = {
                "symbol": "AAPL",
                "side": "long",
                "score": 75,
            }

            explanations = evaluator.generate_explanations(trade_context)
            assert isinstance(explanations, list)

        except ImportError:
            pytest.skip("RAGEvaluator not available")


class TestAlphaResearchIntegration:
    """Test alpha research integration layer."""

    def test_integration_singleton(self):
        """Test alpha research integration singleton."""
        try:
            from research import get_alpha_research_integration

            integration = get_alpha_research_integration()
            assert integration is not None

            # Should be singleton
            integration2 = get_alpha_research_integration()
            assert integration is integration2

        except ImportError:
            pytest.skip("Alpha research integration not available")

    def test_integration_summary(self):
        """Test integration summary."""
        try:
            from research import get_research_summary

            summary = get_research_summary()
            assert isinstance(summary, dict)
            assert 'total_discoveries' in summary

        except ImportError:
            pytest.skip("Alpha research integration not available")


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_enhanced_brain_full_cycle(self):
        """Test complete enhanced brain cycle."""
        from autonomous.enhanced_brain import EnhancedAutonomousBrain

        # Initialize brain
        brain = EnhancedAutonomousBrain(use_langgraph=False)

        # Get initial status
        status_before = brain.get_status()
        assert isinstance(status_before, dict)

        # Run a single cycle
        result = brain.run_single_cycle()
        assert isinstance(result, dict)

        # Get status after
        status_after = brain.get_status()
        assert status_after['cycles_completed'] == status_before['cycles_completed'] + 1

    def test_discovery_alerting(self):
        """Test discovery alerting across all components."""
        from autonomous.enhanced_brain import EnhancedAutonomousBrain

        brain = EnhancedAutonomousBrain()

        # Check for discoveries
        discoveries = brain._check_for_discoveries()
        assert isinstance(discoveries, list)

        # Each discovery should have required fields
        for disc in discoveries:
            assert hasattr(disc, 'discovery_type')
            assert hasattr(disc, 'description')
            assert hasattr(disc, 'source')
            assert hasattr(disc, 'confidence')


def test_package_imports():
    """Test that all enhanced components can be imported."""
    # Base imports
    from autonomous import AutonomousBrain, Discovery
    assert AutonomousBrain is not None
    assert Discovery is not None

    # Enhanced imports
    from autonomous import (
        HAS_ENHANCED_RESEARCH,
        HAS_ENHANCED_BRAIN,
    )

    # Should have flags
    assert isinstance(HAS_ENHANCED_RESEARCH, bool)
    assert isinstance(HAS_ENHANCED_BRAIN, bool)

    # If enhanced components available, should be importable
    if HAS_ENHANCED_RESEARCH:
        from autonomous import EnhancedResearchEngine
        assert EnhancedResearchEngine is not None

    if HAS_ENHANCED_BRAIN:
        from autonomous import EnhancedAutonomousBrain
        assert EnhancedAutonomousBrain is not None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
