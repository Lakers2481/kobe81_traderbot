"""
Unit tests for evolution registry and clone detector.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from evolution.registry import (
    StrategyCandidate,
    CandidateDecision,
    EvolutionRegistry,
)
from evolution.clone_detector import (
    CloneDetector,
    CloneCheckResult,
    compute_strategy_fingerprint,
)


class TestStrategyCandidate:
    """Tests for strategy candidate dataclass."""

    def test_basic_candidate(self):
        candidate = StrategyCandidate(
            strategy_name="DonchianBreakout_v2",
            params={"lookback": 20, "atr_mult": 2.0},
            parent_name="DonchianBreakout",
            dataset_hash="abc123",
            train_start="2020-01-01",
            train_end="2023-12-31",
            test_start="2024-01-01",
            test_end="2024-12-31",
            results={
                "train_win_rate": 0.55,
                "test_win_rate": 0.52,
                "train_sharpe": 1.2,
                "test_sharpe": 0.9,
            },
        )
        assert candidate.strategy_name == "DonchianBreakout_v2"
        assert candidate.decision == CandidateDecision.PENDING

    def test_candidate_to_dict(self):
        candidate = StrategyCandidate(
            strategy_name="RSI_v1",
            params={"period": 14},
            dataset_hash="xyz789",
            train_start="2020-01-01",
            train_end="2023-12-31",
            test_start="2024-01-01",
            test_end="2024-12-31",
            results={"win_rate": 0.50},
        )
        d = candidate.to_dict()
        assert d["strategy_name"] == "RSI_v1"
        assert d["decision"] == "PENDING"


class TestEvolutionRegistry:
    """Tests for evolution registry."""

    def test_register_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = EvolutionRegistry(Path(tmpdir) / "registry.jsonl")

            candidate = StrategyCandidate(
                strategy_name="Test_v1",
                params={"a": 1},
                dataset_hash="hash1",
                train_start="2020-01-01",
                train_end="2023-12-31",
                test_start="2024-01-01",
                test_end="2024-12-31",
                results={"win_rate": 0.55},
            )

            candidate_id = registry.register_candidate(candidate)
            assert candidate_id is not None
            assert len(candidate_id) > 0

    def test_get_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = EvolutionRegistry(Path(tmpdir) / "registry.jsonl")

            # Register multiple candidates
            for i in range(3):
                candidate = StrategyCandidate(
                    strategy_name=f"Test_v{i}",
                    params={"version": i},
                    dataset_hash=f"hash{i}",
                    train_start="2020-01-01",
                    train_end="2023-12-31",
                    test_start="2024-01-01",
                    test_end="2024-12-31",
                    results={"win_rate": 0.50 + i * 0.01},
                )
                registry.register_candidate(candidate)

            candidates = registry.get_candidates()
            assert len(candidates) == 3

    def test_get_candidates_by_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = EvolutionRegistry(Path(tmpdir) / "registry.jsonl")

            # Register and mark some as promoted
            for i in range(3):
                candidate = StrategyCandidate(
                    strategy_name=f"Test_v{i}",
                    params={"version": i},
                    dataset_hash=f"hash{i}",
                    train_start="2020-01-01",
                    train_end="2023-12-31",
                    test_start="2024-01-01",
                    test_end="2024-12-31",
                    results={"win_rate": 0.50},
                )
                cid = registry.register_candidate(candidate)
                if i == 0:
                    registry.mark_decision(cid, CandidateDecision.PROMOTED, "Good performance")

            pending = registry.get_candidates(status=CandidateDecision.PENDING)
            promoted = registry.get_candidates(status=CandidateDecision.PROMOTED)

            assert len(pending) == 2
            assert len(promoted) == 1

    def test_mark_decision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = EvolutionRegistry(Path(tmpdir) / "registry.jsonl")

            candidate = StrategyCandidate(
                strategy_name="Test_v1",
                params={"a": 1},
                dataset_hash="hash1",
                train_start="2020-01-01",
                train_end="2023-12-31",
                test_start="2024-01-01",
                test_end="2024-12-31",
                results={"win_rate": 0.55},
            )
            cid = registry.register_candidate(candidate)

            registry.mark_decision(cid, CandidateDecision.REJECTED, "Poor out-of-sample")

            updated = registry.get_candidate(cid)
            assert updated.decision == CandidateDecision.REJECTED
            assert "Poor out-of-sample" in updated.decision_reason

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.jsonl"

            # Create and register
            registry1 = EvolutionRegistry(registry_path)
            candidate = StrategyCandidate(
                strategy_name="Persistent_v1",
                params={"test": True},
                dataset_hash="persist_hash",
                train_start="2020-01-01",
                train_end="2023-12-31",
                test_start="2024-01-01",
                test_end="2024-12-31",
                results={"win_rate": 0.60},
            )
            registry1.register_candidate(candidate)

            # Load in new registry
            registry2 = EvolutionRegistry(registry_path)
            candidates = registry2.get_candidates()

            assert len(candidates) == 1
            assert candidates[0].strategy_name == "Persistent_v1"


class TestCloneDetector:
    """Tests for clone detection."""

    def test_compute_fingerprint(self):
        params1 = {"lookback": 20, "atr_mult": 2.0, "sma_period": 200}
        params2 = {"lookback": 20, "atr_mult": 2.0, "sma_period": 200}
        params3 = {"lookback": 25, "atr_mult": 2.5, "sma_period": 200}

        fp1 = compute_strategy_fingerprint("Strategy", params1)
        fp2 = compute_strategy_fingerprint("Strategy", params2)
        fp3 = compute_strategy_fingerprint("Strategy", params3)

        # Same params = same fingerprint
        assert fp1 == fp2
        # Different params = different fingerprint
        assert fp1 != fp3

    def test_exact_clone_detection(self):
        detector = CloneDetector(similarity_threshold=0.95)

        existing = [
            {"strategy_name": "Donchian_v1", "params": {"lookback": 20, "atr": 2.0}},
        ]

        # Exact clone
        result = detector.check_clone(
            "Donchian_v2",
            {"lookback": 20, "atr": 2.0},
            existing,
        )

        assert result.is_clone
        assert result.similarity >= 0.95
        assert "Donchian_v1" in result.matched_strategy

    def test_near_clone_detection(self):
        detector = CloneDetector(similarity_threshold=0.90)

        existing = [
            {"strategy_name": "Donchian_v1", "params": {"lookback": 20, "atr_mult": 2.0, "sma": 200}},
        ]

        # Near clone - one param slightly different
        result = detector.check_clone(
            "Donchian_v2",
            {"lookback": 21, "atr_mult": 2.0, "sma": 200},  # lookback off by 1
            existing,
        )

        # Should detect as near-clone depending on distance metric
        assert result.similarity > 0.5  # At least somewhat similar

    def test_not_a_clone(self):
        detector = CloneDetector(similarity_threshold=0.95)

        existing = [
            {"strategy_name": "Donchian_v1", "params": {"lookback": 20, "atr": 2.0}},
        ]

        # Completely different strategy
        result = detector.check_clone(
            "RSI_v1",
            {"period": 14, "overbought": 70, "oversold": 30},
            existing,
        )

        assert not result.is_clone
        assert result.similarity < 0.5

    def test_empty_existing(self):
        detector = CloneDetector()

        result = detector.check_clone(
            "NewStrategy",
            {"param": 1},
            [],  # No existing strategies
        )

        assert not result.is_clone

    def test_different_strategy_names_same_params(self):
        detector = CloneDetector(similarity_threshold=0.95)

        existing = [
            {"strategy_name": "OriginalName", "params": {"a": 1, "b": 2}},
        ]

        # Same params but different name - still a clone
        result = detector.check_clone(
            "NewName",
            {"a": 1, "b": 2},
            existing,
        )

        assert result.is_clone


class TestCloneCheckResult:
    """Tests for clone check result."""

    def test_result_properties(self):
        result = CloneCheckResult(
            is_clone=True,
            similarity=0.98,
            matched_strategy="Original_v1",
            message="Near-exact clone detected",
        )
        assert result.is_clone
        assert result.similarity == 0.98
        assert result.matched_strategy == "Original_v1"

    def test_result_to_dict(self):
        result = CloneCheckResult(
            is_clone=False,
            similarity=0.30,
            matched_strategy=None,
            message="No clone detected",
        )
        d = result.to_dict()
        assert d["is_clone"] is False
        assert d["similarity"] == 0.30
