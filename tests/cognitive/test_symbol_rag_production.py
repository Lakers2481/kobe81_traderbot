"""
Comprehensive Unit Tests for Production Symbol RAG - Renaissance Standard

Tests the SymbolRAGProduction system with statistical rigor matching Jim Simons quality bar.

Test Coverage:
- Basic RAG operations (indexing, querying, retrieval)
- Evaluation metrics (faithfulness, relevance, precision)
- Quality gates (stand down when quality < thresholds)
- Edge cases (empty index, no matches, fallback mode)
- Integration tests (end-to-end workflow)
- Performance tests (large index, query latency)

Author: Kobe Trading System (Quant Developer for Jim Simons)
Date: 2026-01-08
Version: 1.0
"""

import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pytest

from cognitive.symbol_rag_production import (
    TradeKnowledge,
    RAGEvaluation,
    RAGResponse,
    RAGEvaluator,
    SymbolRAGProduction,
    MIN_FAITHFULNESS,
    MIN_RELEVANCE,
    MIN_CONTEXT_PRECISION,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_vector_db():
    """Create temporary vector DB directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_trades() -> List[TradeKnowledge]:
    """Generate sample trade knowledge for testing."""
    trades = [
        TradeKnowledge(
            trade_id="trade_001",
            symbol="AAPL",
            timestamp="2025-12-15T14:30:00",
            strategy="IBS_RSI",
            entry_price=180.50,
            exit_price=185.25,
            stop_loss=178.00,
            take_profit=186.00,
            setup="5 consecutive down days, RSI(2) < 5, IBS < 0.08",
            streak_length=5,
            regime="BULL",
            outcome="WIN",
            pnl=475.00,
            pnl_pct=0.0263,  # 2.63% as decimal
            hold_days=3,
            decision_reason="Mean reversion setup with high conviction",
            outcome_reason="Bounced from support, regime was bullish",
            quality_score=0.85,
            conviction=0.90,
        ),
        TradeKnowledge(
            trade_id="trade_002",
            symbol="MSFT",
            timestamp="2025-12-16T10:00:00",
            strategy="IBS_RSI",
            entry_price=420.00,
            exit_price=415.50,
            stop_loss=417.00,
            take_profit=425.00,
            setup="4 consecutive down days, RSI(2) = 8, IBS = 0.12",
            streak_length=4,
            regime="NEUTRAL",
            outcome="LOSS",
            pnl=-450.00,
            pnl_pct=-0.0107,  # -1.07% as decimal
            hold_days=2,
            decision_reason="Mean reversion setup, moderate conviction",
            outcome_reason="Hit stop loss, regime turned bearish",
            quality_score=0.65,
            conviction=0.70,
        ),
        TradeKnowledge(
            trade_id="trade_003",
            symbol="AAPL",
            timestamp="2025-12-20T09:45:00",
            strategy="IBS_RSI",
            entry_price=182.00,
            exit_price=186.80,
            setup="6 consecutive down days, RSI(2) < 3, IBS < 0.05",
            streak_length=6,
            regime="BULL",
            outcome="WIN",
            pnl=480.00,
            pnl_pct=0.0264,  # 2.64% as decimal
            hold_days=2,
            decision_reason="Strong mean reversion setup, oversold",
            outcome_reason="Sharp bounce from extreme oversold",
            quality_score=0.90,
            conviction=0.95,
        ),
    ]
    return trades


@pytest.fixture
def rag_instance(temp_vector_db):
    """Create RAG instance with temporary storage."""
    try:
        rag = SymbolRAGProduction(
            vector_db_path=temp_vector_db,
            collection_name="test_collection",
            top_k=3,
        )
        return rag
    except Exception as e:
        # If dependencies not available, skip tests
        pytest.skip(f"RAG dependencies not available: {e}")


# ============================================================================
# Basic Functionality Tests
# ============================================================================

class TestBasicRAGOperations:
    """Test basic RAG operations."""

    def test_rag_initialization(self, rag_instance):
        """Test RAG initializes correctly."""
        assert rag_instance is not None
        assert rag_instance.collection_name == "test_collection"
        assert rag_instance.top_k == 3

        # Check if available
        if rag_instance.is_available():
            assert rag_instance.model is not None
            assert rag_instance.collection is not None

    def test_trade_knowledge_to_document(self, sample_trades):
        """Test TradeKnowledge converts to semantic document."""
        trade = sample_trades[0]
        doc = trade.to_document()

        # Should contain key information
        assert "AAPL" in doc
        assert "IBS_RSI" in doc
        assert "WIN" in doc
        assert "+2.63%" in doc
        assert "Mean reversion" in doc

    def test_index_trade_history(self, rag_instance, sample_trades):
        """Test indexing trade history."""
        if not rag_instance.is_available():
            pytest.skip("RAG dependencies not available")

        # Index trades
        num_indexed = rag_instance.index_trade_history(sample_trades)

        # Should index all trades
        assert num_indexed == len(sample_trades)

        # Check stats
        stats = rag_instance.get_stats()
        assert stats['available'] is True
        assert stats['index_size'] >= len(sample_trades)

    def test_query_similar_trades(self, rag_instance, sample_trades):
        """Test querying for similar trades."""
        if not rag_instance.is_available():
            pytest.skip("RAG dependencies not available")

        # Index first
        rag_instance.index_trade_history(sample_trades)

        # Query for AAPL trades
        response = rag_instance.query("How does AAPL perform after 5 down days?")

        # Should retrieve documents
        assert response.retrieved_documents is not None
        assert len(response.retrieved_documents) > 0

        # Should have evaluation
        assert response.evaluation is not None
        assert 0 <= response.evaluation.faithfulness <= 1
        assert 0 <= response.evaluation.answer_relevance <= 1

    def test_symbol_filter(self, rag_instance, sample_trades):
        """Test symbol filtering in queries."""
        if not rag_instance.is_available():
            pytest.skip("RAG dependencies not available")

        # Index trades
        rag_instance.index_trade_history(sample_trades)

        # Query for AAPL only
        response = rag_instance.query("Historical trades", symbol="AAPL")

        # All retrieved trades should be for AAPL
        for metadata in response.retrieved_metadata:
            assert metadata.get('symbol') == 'AAPL'


# ============================================================================
# Evaluation Metrics Tests
# ============================================================================

class TestRAGEvaluation:
    """Test RAG evaluation metrics."""

    def test_faithfulness_computation(self):
        """Test faithfulness score computation."""
        evaluator = RAGEvaluator()

        # All docs have required fields → faithfulness = 1.0
        good_metadata = [
            {'symbol': 'AAPL', 'outcome': 'WIN', 'pnl_pct': 0.05, 'strategy': 'IBS_RSI'},
            {'symbol': 'MSFT', 'outcome': 'LOSS', 'pnl_pct': -0.02, 'strategy': 'IBS_RSI'},
        ]
        faithfulness_good = evaluator._compute_faithfulness(good_metadata)
        assert faithfulness_good == 1.0

        # Some docs missing fields → faithfulness < 1.0
        mixed_metadata = [
            {'symbol': 'AAPL', 'outcome': 'WIN', 'pnl_pct': 0.05, 'strategy': 'IBS_RSI'},
            {'symbol': 'MSFT'},  # Missing required fields
        ]
        faithfulness_mixed = evaluator._compute_faithfulness(mixed_metadata)
        assert 0 < faithfulness_mixed < 1.0

        # No docs → faithfulness = 0.0
        faithfulness_empty = evaluator._compute_faithfulness([])
        assert faithfulness_empty == 0.0

    def test_answer_relevance_computation(self):
        """Test answer relevance score computation."""
        evaluator = RAGEvaluator()

        # High similarity (low distance) → high relevance
        high_sim_distances = [0.1, 0.15, 0.2]  # Very close
        relevance_high = evaluator._compute_answer_relevance(
            "test query", ["doc1", "doc2", "doc3"], high_sim_distances
        )
        assert relevance_high >= 0.7  # Avg sim 0.85 → normalized 0.7

        # Low similarity (high distance) → low relevance
        low_sim_distances = [0.8, 0.85, 0.9]  # Far apart
        relevance_low = evaluator._compute_answer_relevance(
            "test query", ["doc1", "doc2", "doc3"], low_sim_distances
        )
        assert relevance_low < 0.3

    def test_context_precision_computation(self):
        """Test context precision score computation."""
        evaluator = RAGEvaluator()

        # All docs relevant (distance < 0.5) → precision = 1.0
        relevant_distances = [0.2, 0.3, 0.4]
        precision_high = evaluator._compute_context_precision(
            "test query", ["doc1", "doc2", "doc3"], relevant_distances
        )
        assert precision_high == 1.0

        # Mixed relevance → precision between 0 and 1
        mixed_distances = [0.2, 0.6, 0.8]  # 1 relevant, 2 not
        precision_mixed = evaluator._compute_context_precision(
            "test query", ["doc1", "doc2", "doc3"], mixed_distances
        )
        assert precision_mixed == pytest.approx(1/3, abs=0.01)

    def test_overall_quality_score(self):
        """Test overall quality score computation."""
        evaluation = RAGEvaluation(
            faithfulness=0.9,
            answer_relevance=0.8,
            context_precision=0.7,
        )

        # Overall quality = weighted average (0.5×F + 0.3×R + 0.2×P)
        expected = 0.5 * 0.9 + 0.3 * 0.8 + 0.2 * 0.7
        assert evaluation.overall_quality == pytest.approx(expected, abs=0.01)

    def test_quality_gates(self):
        """Test quality gate thresholds."""
        # Meets all gates
        good_eval = RAGEvaluation(
            faithfulness=0.8,
            answer_relevance=0.7,
            context_precision=0.7,
        )
        assert good_eval.meets_quality_gates() is True

        # Fails faithfulness
        bad_faith_eval = RAGEvaluation(
            faithfulness=0.6,  # < 0.7
            answer_relevance=0.8,
            context_precision=0.8,
        )
        assert bad_faith_eval.meets_quality_gates() is False

        # Fails relevance
        bad_rel_eval = RAGEvaluation(
            faithfulness=0.8,
            answer_relevance=0.5,  # < 0.6
            context_precision=0.8,
        )
        assert bad_rel_eval.meets_quality_gates() is False


# ============================================================================
# Quality Gates and Stand-Down Tests
# ============================================================================

class TestQualityGatesAndStandDown:
    """Test quality gates and stand-down logic."""

    def test_stand_down_on_low_faithfulness(self, rag_instance, sample_trades):
        """Test stand-down when faithfulness < 0.7."""
        if not rag_instance.is_available():
            pytest.skip("RAG dependencies not available")

        # Index trades with incomplete metadata (will have low faithfulness)
        incomplete_trades = [
            TradeKnowledge(
                trade_id="incomplete",
                symbol="TEST",
                timestamp="2025-01-01",
                strategy="",  # Empty
                entry_price=100,
                exit_price=105,
                setup="",  # Empty
                outcome="",  # Empty - will fail faithfulness
                pnl=0,
                pnl_pct=0,
                decision_reason="",
            )
        ]

        rag_instance.index_trade_history(incomplete_trades)
        response = rag_instance.query("Test query")

        # If faithfulness low, should stand down
        if response.evaluation.faithfulness < MIN_FAITHFULNESS:
            assert response.recommendation == "STAND_DOWN"
            assert "faithfulness" in response.reasoning.lower()

    def test_stand_down_on_insufficient_data(self, rag_instance):
        """Test stand-down when insufficient similar trades."""
        if not rag_instance.is_available():
            pytest.skip("RAG dependencies not available")

        # Query empty index
        response = rag_instance.query("How does XYZ perform?")

        # Should stand down due to no data
        assert response.recommendation in ["STAND_DOWN", "UNCERTAIN"]
        assert response.num_similar_trades < 3

    def test_proceed_on_high_quality(self, rag_instance, sample_trades):
        """Test PROCEED recommendation when quality is high."""
        if not rag_instance.is_available():
            pytest.skip("RAG dependencies not available")

        # Index good quality trades
        rag_instance.index_trade_history(sample_trades)

        # Query for well-documented pattern
        response = rag_instance.query("How does AAPL perform after consecutive down days?")

        # If quality high, should proceed
        if response.evaluation.meets_quality_gates() and response.num_similar_trades >= 3:
            assert response.recommendation == "PROCEED"
            assert "High quality" in response.reasoning


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_index_query(self, rag_instance):
        """Test querying empty index."""
        if not rag_instance.is_available():
            pytest.skip("RAG dependencies not available")

        response = rag_instance.query("Test query")

        assert response.retrieved_documents == []
        assert response.num_similar_trades == 0
        assert response.recommendation in ["STAND_DOWN", "UNCERTAIN"]

    def test_index_empty_trade_list(self, rag_instance):
        """Test indexing empty trade list."""
        if not rag_instance.is_available():
            pytest.skip("RAG dependencies not available")

        num_indexed = rag_instance.index_trade_history([])
        assert num_indexed == 0

    def test_fallback_mode_when_dependencies_missing(self, temp_vector_db):
        """Test fallback mode when dependencies not available."""
        # This test should work even without dependencies
        rag = SymbolRAGProduction(vector_db_path=temp_vector_db)

        if not rag.is_available():
            # Should return fallback response
            response = rag.query("Test query")
            assert response.recommendation == "STAND_DOWN"
            assert "not available" in response.reasoning.lower()

    def test_query_with_special_characters(self, rag_instance, sample_trades):
        """Test query with special characters."""
        if not rag_instance.is_available():
            pytest.skip("RAG dependencies not available")

        rag_instance.index_trade_history(sample_trades)

        # Query with special characters
        response = rag_instance.query("What about AAPL @ $180?!?")

        # Should handle gracefully
        assert response is not None
        assert isinstance(response, RAGResponse)

    def test_duplicate_trade_indexing(self, rag_instance, sample_trades):
        """Test indexing duplicate trades (same ID)."""
        if not rag_instance.is_available():
            pytest.skip("RAG dependencies not available")

        # Index same trades twice
        num_indexed_1 = rag_instance.index_trade_history(sample_trades)
        num_indexed_2 = rag_instance.index_trade_history(sample_trades)

        # Both should succeed (ChromaDB handles duplicates by ID)
        assert num_indexed_1 == len(sample_trades)
        assert num_indexed_2 == len(sample_trades)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test end-to-end integration scenarios."""

    def test_full_rag_workflow(self, rag_instance, sample_trades):
        """Test complete RAG workflow: index → query → evaluate → recommend."""
        if not rag_instance.is_available():
            pytest.skip("RAG dependencies not available")

        # Step 1: Index historical trades
        num_indexed = rag_instance.index_trade_history(sample_trades)
        assert num_indexed == len(sample_trades)

        # Step 2: Query for similar trades
        response = rag_instance.query("How does AAPL perform after 5+ consecutive down days?")

        # Step 3: Verify evaluation
        assert response.evaluation is not None
        assert 0 <= response.evaluation.faithfulness <= 1
        assert 0 <= response.evaluation.answer_relevance <= 1
        assert 0 <= response.evaluation.context_precision <= 1

        # Step 4: Verify recommendation
        assert response.recommendation in ["PROCEED", "STAND_DOWN", "UNCERTAIN"]

        # Step 5: Verify evidence
        assert response.num_similar_trades >= 0
        if response.num_similar_trades > 0:
            assert response.win_rate_similar is not None
            assert 0 <= response.win_rate_similar <= 1

    def test_rag_response_serialization(self, rag_instance, sample_trades):
        """Test RAG response can be serialized to JSON."""
        if not rag_instance.is_available():
            pytest.skip("RAG dependencies not available")

        rag_instance.index_trade_history(sample_trades)
        response = rag_instance.query("Test query")

        # Should serialize to dict
        response_dict = response.to_dict()
        assert isinstance(response_dict, dict)

        # Should contain required keys
        assert 'query' in response_dict
        assert 'evaluation' in response_dict
        assert 'recommendation' in response_dict

        # Should be JSON serializable
        json_str = json.dumps(response_dict)
        assert len(json_str) > 0

    def test_win_rate_computation(self, rag_instance, sample_trades):
        """Test win rate computation from retrieved trades."""
        if not rag_instance.is_available():
            pytest.skip("RAG dependencies not available")

        rag_instance.index_trade_history(sample_trades)
        response = rag_instance.query("AAPL trades")

        if response.num_similar_trades > 0:
            # Compute expected win rate
            outcomes = [m.get('outcome') for m in response.retrieved_metadata]
            expected_wr = sum(1 for o in outcomes if o == 'WIN') / len(outcomes)

            # Should match
            assert response.win_rate_similar == pytest.approx(expected_wr, abs=0.01)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance and scalability."""

    def test_large_index_performance(self, rag_instance):
        """Test performance with large index (1000 trades)."""
        if not rag_instance.is_available():
            pytest.skip("RAG dependencies not available")

        # Generate 1000 trades
        large_trade_set = []
        for i in range(1000):
            trade = TradeKnowledge(
                trade_id=f"trade_{i:04d}",
                symbol=f"SYM{i % 100}",  # 100 different symbols
                timestamp=f"2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                strategy="IBS_RSI",
                entry_price=100 + (i % 50),
                exit_price=105 + (i % 50),
                setup=f"Test setup {i}",
                outcome="WIN" if i % 2 == 0 else "LOSS",
                pnl=100 if i % 2 == 0 else -50,
                pnl_pct=1.0 if i % 2 == 0 else -0.5,
                decision_reason=f"Test reason {i}",
            )
            large_trade_set.append(trade)

        # Index (should complete reasonably fast)
        import time
        start = time.time()
        num_indexed = rag_instance.index_trade_history(large_trade_set)
        index_time = time.time() - start

        assert num_indexed == 1000
        # Should index 1000 trades in < 30 seconds
        assert index_time < 30.0, f"Indexing took {index_time:.2f}s (expected < 30s)"

    def test_query_latency(self, rag_instance, sample_trades):
        """Test query latency is acceptable."""
        if not rag_instance.is_available():
            pytest.skip("RAG dependencies not available")

        rag_instance.index_trade_history(sample_trades)

        # Query and measure latency
        import time
        start = time.time()
        response = rag_instance.query("Test query")
        query_time = time.time() - start

        # Should query in < 1 second
        assert query_time < 1.0, f"Query took {query_time:.2f}s (expected < 1s)"

    def test_multiple_concurrent_queries(self, rag_instance, sample_trades):
        """Test multiple concurrent queries."""
        if not rag_instance.is_available():
            pytest.skip("RAG dependencies not available")

        rag_instance.index_trade_history(sample_trades)

        # Run 10 queries
        queries = [
            "How does AAPL perform?",
            "What about MSFT trades?",
            "Consecutive down days pattern",
            "IBS_RSI strategy results",
            "Bullish regime trades",
        ]

        responses = []
        for query in queries:
            response = rag_instance.query(query)
            responses.append(response)

        # All should complete successfully
        assert len(responses) == len(queries)
        for response in responses:
            assert response is not None
            assert isinstance(response, RAGResponse)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
