"""
Unit tests for data lineage tracking.
"""

import tempfile
from pathlib import Path

from core.lineage import (
    compute_file_hash,
    compute_directory_hash,
    compute_dataset_hash,
    compute_model_hash,
    compute_decision_hash,
    compute_order_hash,
    LineageTracker,
    link_lineage,
)


class TestHashFunctions:
    """Tests for hash computation functions."""

    def test_file_hash(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            f.flush()
            hash1 = compute_file_hash(f.name)
            assert len(hash1) == 64  # SHA256 hex length
            # Same content should give same hash
            f.seek(0)
            hash2 = compute_file_hash(f.name)
            assert hash1 == hash2

    def test_file_hash_different_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "file1.txt"
            path2 = Path(tmpdir) / "file2.txt"
            path1.write_text("content1")
            path2.write_text("content2")
            assert compute_file_hash(path1) != compute_file_hash(path2)

    def test_directory_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.txt").write_text("content1")
            (Path(tmpdir) / "file2.txt").write_text("content2")
            hash1 = compute_directory_hash(tmpdir)
            assert len(hash1) == 64

    def test_dataset_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "data1.csv"
            path2 = Path(tmpdir) / "data2.csv"
            path1.write_text("a,b\n1,2")
            path2.write_text("c,d\n3,4")
            hash1 = compute_dataset_hash([path1, path2])
            assert len(hash1) == 64

    def test_dataset_hash_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "exists.csv"
            path2 = Path(tmpdir) / "missing.csv"
            path1.write_text("data")
            # Should include "MISSING" in hash
            hash1 = compute_dataset_hash([path1, path2])
            assert len(hash1) == 64

    def test_model_hash(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pkl") as f:
            f.write("model weights")
            f.flush()
            config = {"param1": 10, "param2": 0.5}
            hash1 = compute_model_hash(f.name, config)
            assert len(hash1) == 64
            # Different config = different hash
            hash2 = compute_model_hash(f.name, {"param1": 20})
            assert hash1 != hash2

    def test_decision_hash(self):
        packet = {"symbol": "AAPL", "side": "buy", "price": 150.0}
        hash1 = compute_decision_hash(packet)
        assert len(hash1) == 64
        # Same packet = same hash
        hash2 = compute_decision_hash(packet)
        assert hash1 == hash2

    def test_order_hash(self):
        hash1 = compute_order_hash("order1", "AAPL", "buy", 100, 150.0, "decision_abc")
        assert len(hash1) == 64
        # Different order = different hash
        hash2 = compute_order_hash("order2", "AAPL", "buy", 100, 150.0, "decision_abc")
        assert hash1 != hash2


class TestLineageTracker:
    """Tests for lineage tracker."""

    def test_record_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = LineageTracker(Path(tmpdir) / "lineage.jsonl")
            data_file = Path(tmpdir) / "data.csv"
            data_file.write_text("a,b\n1,2")

            record = tracker.record_dataset("dataset_001", [data_file])
            assert record.record_type == "dataset"
            assert record.record_id == "dataset_001"
            assert len(record.record_hash) == 64

    def test_record_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = LineageTracker(Path(tmpdir) / "lineage.jsonl")
            model_file = Path(tmpdir) / "model.pkl"
            model_file.write_text("weights")

            record = tracker.record_model(
                "model_001", model_file, dataset_hash="abc123"
            )
            assert record.record_type == "model"
            assert "abc123" in record.parent_hashes

    def test_record_decision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = LineageTracker(Path(tmpdir) / "lineage.jsonl")

            packet = {"symbol": "AAPL", "side": "buy"}
            record = tracker.record_decision(
                "decision_001", packet, model_hash="model_abc", dataset_hash="data_abc"
            )
            assert record.record_type == "decision"
            assert len(record.parent_hashes) == 2

    def test_record_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = LineageTracker(Path(tmpdir) / "lineage.jsonl")

            record = tracker.record_order(
                order_id="order_001",
                symbol="AAPL",
                side="buy",
                qty=100,
                price=150.0,
                decision_hash="decision_abc",
            )
            assert record.record_type == "order"
            assert record.metadata["symbol"] == "AAPL"

    def test_get_lineage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = LineageTracker(Path(tmpdir) / "lineage.jsonl")
            data_file = Path(tmpdir) / "data.csv"
            data_file.write_text("data")

            # Create chain: dataset -> model -> decision -> order
            dataset_record = tracker.record_dataset("ds1", [data_file])
            model_file = Path(tmpdir) / "model.pkl"
            model_file.write_text("model")
            model_record = tracker.record_model("m1", model_file, dataset_record.record_hash)
            decision_record = tracker.record_decision(
                "d1", {"symbol": "AAPL"}, model_record.record_hash, dataset_record.record_hash
            )
            order_record = tracker.record_order(
                "o1", "AAPL", "buy", 100, 150.0, decision_record.record_hash
            )

            # Get lineage from order
            lineage = tracker.get_lineage(order_record.record_hash)
            assert len(lineage) >= 1

    def test_verify_lineage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = LineageTracker(Path(tmpdir) / "lineage.jsonl")
            data_file = Path(tmpdir) / "data.csv"
            data_file.write_text("data")
            model_file = Path(tmpdir) / "model.pkl"
            model_file.write_text("model")

            # Create complete chain
            ds = tracker.record_dataset("ds1", [data_file])
            m = tracker.record_model("m1", model_file, ds.record_hash)
            d = tracker.record_decision("d1", {"symbol": "AAPL"}, m.record_hash, ds.record_hash)
            o = tracker.record_order("o1", "AAPL", "buy", 100, 150.0, d.record_hash)

            result = tracker.verify_lineage(o.record_hash)
            assert result["chain_length"] >= 1

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lineage_path = Path(tmpdir) / "lineage.jsonl"

            # Create and record
            tracker1 = LineageTracker(lineage_path)
            record = tracker1.record("test", "id1", "hash1", [], {})

            # Load in new tracker and verify via get_lineage
            tracker2 = LineageTracker(lineage_path)
            chain = tracker2.get_lineage(record.record_hash)
            assert len(chain) == 1
            assert chain[0].record_id == "id1"


class TestLinkLineage:
    """Tests for convenience link_lineage function."""

    def test_link_full_lineage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = LineageTracker(Path(tmpdir) / "lineage.jsonl")
            data_file = Path(tmpdir) / "data.csv"
            data_file.write_text("data")
            model_file = Path(tmpdir) / "model.pkl"
            model_file.write_text("model")

            decision_packet = {"symbol": "AAPL", "side": "buy", "run_id": "test_run"}

            order_hash = link_lineage(
                order_id="order_001",
                symbol="AAPL",
                side="buy",
                qty=100,
                price=150.0,
                decision_packet=decision_packet,
                model_path=model_file,
                dataset_paths=[data_file],
                tracker=tracker,
            )

            assert len(order_hash) == 64
