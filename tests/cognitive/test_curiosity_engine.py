"""
Unit tests for cognitive/curiosity_engine.py

Tests the AI's ability to generate and track hypotheses for continuous improvement.
"""
import pytest
import tempfile
from pathlib import Path


class TestHypothesisStatusEnum:
    """Tests for the HypothesisStatus enumeration."""

    def test_status_values(self):
        from cognitive.curiosity_engine import HypothesisStatus

        assert HypothesisStatus.PROPOSED.value == "proposed"
        assert HypothesisStatus.TESTING.value == "testing"
        assert HypothesisStatus.VALIDATED.value == "validated"
        assert HypothesisStatus.REJECTED.value == "rejected"
        assert HypothesisStatus.INCONCLUSIVE.value == "inconclusive"


class TestHypothesisDataclass:
    """Tests for the Hypothesis dataclass."""

    def test_hypothesis_creation(self):
        from cognitive.curiosity_engine import Hypothesis, HypothesisStatus

        hypothesis = Hypothesis(
            hypothesis_id="hyp_001",
            description="RSI < 5 works better than RSI < 10",
            condition="regime = BULL AND strategy = ibs_rsi",
            prediction="win_rate > 0.6",
            rationale="Observed high win rate in historical data",
        )

        assert hypothesis.hypothesis_id == "hyp_001"
        assert hypothesis.status == HypothesisStatus.PROPOSED
        assert hypothesis.sample_size == 0

    def test_hypothesis_to_dict(self):
        from cognitive.curiosity_engine import Hypothesis

        hypothesis = Hypothesis(
            hypothesis_id="hyp_002",
            description="Test statement",
            condition="vix > 30",
            prediction="win_rate > 0.55",
            rationale="Testing high volatility",
        )
        d = hypothesis.to_dict()

        assert d['hypothesis_id'] == "hyp_002"
        assert 'description' in d
        assert 'status' in d


class TestEdgeDataclass:
    """Tests for the Edge dataclass."""

    def test_edge_creation(self):
        from cognitive.curiosity_engine import Edge
        from datetime import datetime

        edge = Edge(
            edge_id="edge_001",
            description="Turtle soup works in high VIX",
            condition="vix > 30",
            expected_win_rate=0.65,
            expected_profit_factor=1.5,
            sample_size=50,
            confidence=0.85,
            first_discovered=datetime.now(),
            last_validated=datetime.now(),
        )

        assert edge.edge_id == "edge_001"
        assert edge.expected_win_rate == 0.65
        assert edge.times_validated == 1

    def test_edge_to_dict(self):
        from cognitive.curiosity_engine import Edge
        from datetime import datetime

        edge = Edge(
            edge_id="edge_002",
            description="Test edge",
            condition="regime = BULL",
            expected_win_rate=0.6,
            expected_profit_factor=1.2,
            sample_size=30,
            confidence=0.75,
            first_discovered=datetime.now(),
            last_validated=datetime.now(),
        )
        d = edge.to_dict()

        assert d['edge_id'] == "edge_002"
        assert 'expected_win_rate' in d
        assert 'confidence' in d


class TestCuriosityEngineInitialization:
    """Tests for CuriosityEngine initialization."""

    def test_default_initialization(self):
        from cognitive.curiosity_engine import CuriosityEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CuriosityEngine(storage_dir=tmpdir)

            assert engine is not None
            assert engine.min_sample_size == 30

    def test_custom_parameters(self):
        from cognitive.curiosity_engine import CuriosityEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CuriosityEngine(
                storage_dir=tmpdir,
                min_sample_size=50,
                significance_level=0.01,
                min_edge_win_rate=0.60,
            )

            assert engine.min_sample_size == 50
            assert engine.significance_level == 0.01
            assert engine.min_edge_win_rate == 0.60


class TestGetValidatedEdges:
    """Tests for getting validated edges."""

    def test_get_validated_edges_empty(self):
        from cognitive.curiosity_engine import CuriosityEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CuriosityEngine(storage_dir=tmpdir)

            edges = engine.get_validated_edges()
            assert isinstance(edges, list)
            assert len(edges) == 0


class TestGetActiveHypotheses:
    """Tests for getting active hypotheses."""

    def test_get_active_hypotheses_empty(self):
        from cognitive.curiosity_engine import CuriosityEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CuriosityEngine(storage_dir=tmpdir)

            hypotheses = engine.get_active_hypotheses()
            assert isinstance(hypotheses, list)


class TestTestAllPending:
    """Tests for testing all pending hypotheses."""

    def test_test_all_pending_empty(self):
        from cognitive.curiosity_engine import CuriosityEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CuriosityEngine(storage_dir=tmpdir)

            results = engine.test_all_pending()

            assert isinstance(results, dict)
            assert 'tested' in results


class TestStatistics:
    """Tests for curiosity engine statistics."""

    def test_get_stats_empty(self):
        from cognitive.curiosity_engine import CuriosityEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CuriosityEngine(storage_dir=tmpdir)

            stats = engine.get_stats()
            assert stats['total_hypotheses'] == 0
            assert stats['total_edges'] == 0


class TestIntrospection:
    """Tests for curiosity engine introspection."""

    def test_introspect(self):
        from cognitive.curiosity_engine import CuriosityEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CuriosityEngine(storage_dir=tmpdir)

            report = engine.introspect()

            assert isinstance(report, str)
            assert len(report) > 0
            assert "Curiosity Engine" in report


class TestStatePersistence:
    """Tests for state persistence."""

    def test_state_persists_across_instances(self):
        from cognitive.curiosity_engine import CuriosityEngine, Hypothesis, HypothesisStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create engine and add a hypothesis manually
            engine1 = CuriosityEngine(storage_dir=tmpdir)

            # Manually add a hypothesis
            hyp = Hypothesis(
                hypothesis_id="test_hyp",
                description="Test hypothesis",
                condition="regime = BULL",
                prediction="win_rate > 0.6",
                rationale="Testing persistence",
            )
            engine1._hypotheses["test_hyp"] = hyp
            engine1._save_state()

            # Create new engine and verify it loads the hypothesis
            engine2 = CuriosityEngine(storage_dir=tmpdir)

            assert "test_hyp" in engine2._hypotheses
