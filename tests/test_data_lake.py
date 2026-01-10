"""
Tests for Frozen Data Lake.

Tests manifest computation, IO operations, and data integrity.
"""
import pytest
from datetime import datetime
from pathlib import Path
import tempfile

# Import modules under test
from data.lake.manifest import (
    compute_dataset_id,
    compute_file_hash,
    DatasetManifest,
    FileRecord,
)


class TestComputeDatasetId:
    """Tests for compute_dataset_id function."""

    def test_deterministic_id(self):
        """Same inputs should produce same ID."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("symbol\nAAPL\nMSFT\n")
            universe_path = Path(f.name)

        id1 = compute_dataset_id(
            provider='stooq',
            timeframe='1d',
            start_date='2020-01-01',
            end_date='2024-12-31',
            universe_path=universe_path,
        )

        id2 = compute_dataset_id(
            provider='stooq',
            timeframe='1d',
            start_date='2020-01-01',
            end_date='2024-12-31',
            universe_path=universe_path,
        )

        assert id1 == id2

        # Cleanup
        universe_path.unlink()

    def test_different_inputs_different_ids(self):
        """Different inputs should produce different IDs."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("symbol\nAAPL\nMSFT\n")
            universe_path = Path(f.name)

        id1 = compute_dataset_id(
            provider='stooq',
            timeframe='1d',
            start_date='2020-01-01',
            end_date='2024-12-31',
            universe_path=universe_path,
        )

        id2 = compute_dataset_id(
            provider='yfinance',  # Different provider
            timeframe='1d',
            start_date='2020-01-01',
            end_date='2024-12-31',
            universe_path=universe_path,
        )

        assert id1 != id2

        # Cleanup
        universe_path.unlink()

    def test_id_format(self):
        """ID should follow expected format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("symbol\nAAPL\n")
            universe_path = Path(f.name)

        dataset_id = compute_dataset_id(
            provider='stooq',
            timeframe='1d',
            start_date='2020-01-01',
            end_date='2024-12-31',
            universe_path=universe_path,
        )

        # Should contain provider and timeframe
        assert 'stooq' in dataset_id
        assert '1d' in dataset_id
        assert '2020' in dataset_id
        assert '2024' in dataset_id

        # Cleanup
        universe_path.unlink()


class TestComputeFileHash:
    """Tests for compute_file_hash function."""

    def test_sha256_hash(self):
        """Should compute correct SHA256 hash."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            test_path = Path(f.name)

        hash_value = compute_file_hash(test_path)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256 hex length

        # Cleanup
        test_path.unlink()

    def test_same_content_same_hash(self):
        """Same content should produce same hash."""
        content = "identical content"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
            f1.write(content)
            path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
            f2.write(content)
            path2 = Path(f2.name)

        assert compute_file_hash(path1) == compute_file_hash(path2)

        # Cleanup
        path1.unlink()
        path2.unlink()


class TestDatasetManifest:
    """Tests for DatasetManifest dataclass."""

    def test_manifest_creation(self):
        """Should create valid manifest."""
        manifest = DatasetManifest(
            dataset_id="test_1d_2020_2024_abc123",
            provider="stooq",
            timeframe="1d",
            start_date="2020-01-01",
            end_date="2024-12-31",
            universe_path="data/universe/test.csv",
            universe_sha256="abc123",
            schema_version="v1.0",
            created_at=datetime.now().isoformat(),
            total_symbols=100,
            total_rows=25000,
            files=[],
        )

        assert manifest.dataset_id == "test_1d_2020_2024_abc123"
        assert manifest.total_symbols == 100
        assert manifest.total_rows == 25000

    def test_manifest_to_dict(self):
        """Should convert to dictionary."""
        manifest = DatasetManifest(
            dataset_id="test_1d_2020_2024_abc123",
            provider="stooq",
            timeframe="1d",
            start_date="2020-01-01",
            end_date="2024-12-31",
            universe_path="data/universe/test.csv",
            universe_sha256="abc123",
            schema_version="v1.0",
            created_at=datetime.now().isoformat(),
            total_symbols=100,
            total_rows=25000,
            files=[],
        )

        d = manifest.to_dict()

        assert isinstance(d, dict)
        assert d['dataset_id'] == "test_1d_2020_2024_abc123"
        assert d['total_symbols'] == 100


class TestFileRecord:
    """Tests for FileRecord dataclass."""

    def test_file_record_creation(self):
        """Should create valid file record."""
        record = FileRecord(
            path="data.parquet",
            sha256="abc123def456",
            rows=1000,
            min_timestamp="2020-01-01",
            max_timestamp="2024-12-31",
            size_bytes=50000,
        )

        assert record.path == "data.parquet"
        assert record.rows == 1000
        assert record.size_bytes == 50000


# Run with: pytest tests/test_data_lake.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
