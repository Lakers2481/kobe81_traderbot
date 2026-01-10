"""
Tests for the Evidence Pack system.

This module tests research/evidence.py which provides the evidence pack
system for reproducibility of backtests and experiments.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from research.evidence import (
    EvidencePack,
    EvidencePackBuilder,
    create_backtest_evidence_pack,
    create_walkforward_evidence_pack,
)


class TestEvidencePackBuilder:
    """Tests for EvidencePackBuilder."""

    def test_builder_creates_unique_pack_id(self):
        """Each builder creates a unique pack ID."""
        builder1 = EvidencePackBuilder("backtest")
        builder2 = EvidencePackBuilder("backtest")
        assert builder1.pack_id != builder2.pack_id

    def test_builder_captures_git_state(self):
        """Builder can capture git state."""
        builder = EvidencePackBuilder("test")
        builder.capture_git_state()

        # Should have captured something (even if git not available)
        assert builder.git_commit != ""
        assert builder.git_branch != ""

    def test_builder_captures_environment(self):
        """Builder captures Python environment."""
        builder = EvidencePackBuilder("test")
        builder.capture_environment()

        assert builder.python_version != ""
        assert "pandas" in builder.package_versions

    def test_builder_set_config(self):
        """Builder can set configuration."""
        builder = EvidencePackBuilder("test")
        config = {"max_position_size": 1000, "risk_pct": 0.02}
        builder.set_config(config)

        assert builder.config_snapshot == config

    def test_builder_set_frozen_params(self):
        """Builder can set frozen parameters."""
        builder = EvidencePackBuilder("test")
        params = {"ts_min_sweep_strength": 0.3, "ibs_threshold": 0.08}
        builder.set_frozen_params(params, path="/path/to/params.json")

        assert builder.frozen_params == params
        assert builder.frozen_params_path == "/path/to/params.json"

    def test_builder_set_dataset(self):
        """Builder can set dataset information."""
        builder = EvidencePackBuilder("test")
        builder.set_dataset(
            dataset_id="stooq_1d_2015_2025",
            universe_sha256="abc123def456",
            start="2015-01-01",
            end="2025-12-31",
            symbol_count=800
        )

        assert builder.dataset_id == "stooq_1d_2015_2025"
        assert builder.universe_sha256 == "abc123def456"
        assert builder.date_range == ("2015-01-01", "2025-12-31")
        assert builder.symbol_count == 800

    def test_builder_set_metrics(self):
        """Builder can set result metrics."""
        builder = EvidencePackBuilder("test")
        metrics = {
            "win_rate": 0.64,
            "profit_factor": 1.60,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.15,
            "total_trades": 500,
        }
        builder.set_metrics(metrics)

        assert builder.metrics == metrics
        assert builder.win_rate == 0.64
        assert builder.profit_factor == 1.60
        assert builder.total_trades == 500

    def test_builder_add_artifact(self):
        """Builder can add artifacts with hashes."""
        builder = EvidencePackBuilder("test")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("symbol,side,qty\nAAPL,BUY,100\n")
            temp_path = Path(f.name)

        try:
            builder.add_artifact("trade_list", temp_path)
            assert "trade_list" in builder.artifacts
            assert ":" in builder.artifacts["trade_list"]  # path:hash format
        finally:
            temp_path.unlink()

    def test_builder_build(self):
        """Builder creates valid EvidencePack."""
        builder = EvidencePackBuilder("backtest")
        builder.capture_git_state()
        builder.capture_environment()
        builder.set_metrics({"win_rate": 0.64, "total_trades": 100})

        pack = builder.build()

        assert isinstance(pack, EvidencePack)
        assert pack.pack_type == "backtest"
        assert pack.win_rate == 0.64
        assert pack.total_trades == 100
        assert pack.pack_hash != ""


class TestEvidencePack:
    """Tests for EvidencePack dataclass."""

    def test_pack_hash_is_deterministic(self):
        """Same inputs produce same hash."""
        pack1 = EvidencePack(
            pack_id="test_123",
            created_at=datetime(2025, 1, 1, 12, 0, 0),
            pack_type="backtest",
            git_commit="abc123",
            git_branch="main",
            git_dirty=False,
            python_version="3.11.0",
            package_versions={"pandas": "2.0.0"},
            config_snapshot={"key": "value"},
            frozen_params={"ts_min_sweep_strength": 0.3},
            metrics={"win_rate": 0.64},
        )

        pack2 = EvidencePack(
            pack_id="test_123",
            created_at=datetime(2025, 1, 1, 12, 0, 0),
            pack_type="backtest",
            git_commit="abc123",
            git_branch="main",
            git_dirty=False,
            python_version="3.11.0",
            package_versions={"pandas": "2.0.0"},
            config_snapshot={"key": "value"},
            frozen_params={"ts_min_sweep_strength": 0.3},
            metrics={"win_rate": 0.64},
        )

        assert pack1.pack_hash == pack2.pack_hash

    def test_pack_hash_changes_with_metrics(self):
        """Different metrics produce different hash."""
        base_kwargs = {
            "pack_id": "test_123",
            "created_at": datetime(2025, 1, 1),
            "pack_type": "backtest",
            "git_commit": "abc123",
            "git_branch": "main",
            "git_dirty": False,
            "python_version": "3.11.0",
            "package_versions": {},
            "config_snapshot": {},
            "frozen_params": {},
        }

        pack1 = EvidencePack(**base_kwargs, metrics={"win_rate": 0.64})
        pack2 = EvidencePack(**base_kwargs, metrics={"win_rate": 0.70})

        assert pack1.pack_hash != pack2.pack_hash

    def test_verify_hash_valid(self):
        """verify_hash returns True for unmodified pack."""
        builder = EvidencePackBuilder("test")
        builder.capture_git_state()
        pack = builder.build()

        assert pack.verify_hash() is True

    def test_save_and_load(self):
        """Pack can be saved and loaded with integrity."""
        builder = EvidencePackBuilder("backtest")
        builder.capture_git_state()
        builder.capture_environment()
        builder.set_metrics({
            "win_rate": 0.64,
            "profit_factor": 1.60,
            "total_trades": 100
        })
        original_pack = builder.build()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            filepath = original_pack.save(output_dir)

            # Load and verify
            loaded_pack = EvidencePack.load(filepath)

            assert loaded_pack.pack_id == original_pack.pack_id
            assert loaded_pack.pack_hash == original_pack.pack_hash
            assert loaded_pack.win_rate == original_pack.win_rate
            assert loaded_pack.verify_hash() is True

    def test_to_dict(self):
        """Pack can be serialized to dictionary."""
        builder = EvidencePackBuilder("test")
        builder.capture_git_state()
        pack = builder.build()

        d = pack.to_dict()

        assert d["pack_id"] == pack.pack_id
        assert d["pack_type"] == "test"
        assert "git_commit" in d
        assert "pack_hash" in d

    def test_generate_reproduce_script(self):
        """Pack generates valid reproduction script."""
        builder = EvidencePackBuilder("backtest")
        builder.capture_git_state()
        builder.set_dataset(
            dataset_id="test",
            universe_sha256="abc123",
            start="2023-01-01",
            end="2024-12-31",
            universe_path="data/universe/test.csv"
        )
        builder.set_metrics({"win_rate": 0.64})
        pack = builder.build()

        script = pack.generate_reproduce_script()

        assert "#!/bin/bash" in script
        assert pack.pack_id in script
        assert pack.git_commit in script
        assert "scripts/backtest_dual_strategy.py" in script

    def test_generate_walkforward_script(self):
        """Walk-forward pack generates correct script."""
        builder = EvidencePackBuilder("walk_forward")
        builder.capture_git_state()
        builder.set_dataset(
            dataset_id="test",
            universe_sha256="abc123",
            start="2015-01-01",
            end="2025-12-31",
            universe_path="data/universe/test.csv"
        )
        pack = builder.build()

        script = pack.generate_reproduce_script()

        assert "run_wf_polygon.py" in script


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_backtest_evidence_pack(self):
        """create_backtest_evidence_pack creates valid pack."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock universe file
            universe_path = Path(tmpdir) / "universe.csv"
            universe_path.write_text("symbol\nAAPL\nMSFT\nGOOG\n")

            pack = create_backtest_evidence_pack(
                universe_path=universe_path,
                start_date="2023-01-01",
                end_date="2024-12-31",
                metrics={"win_rate": 0.64, "profit_factor": 1.60}
            )

            assert pack.pack_type == "backtest"
            assert pack.date_range == ("2023-01-01", "2024-12-31")
            assert pack.universe_sha256 != ""
            assert pack.win_rate == 0.64

    def test_create_walkforward_evidence_pack(self):
        """create_walkforward_evidence_pack creates valid pack."""
        with tempfile.TemporaryDirectory() as tmpdir:
            universe_path = Path(tmpdir) / "universe.csv"
            universe_path.write_text("symbol\nAAPL\nMSFT\n")

            pack = create_walkforward_evidence_pack(
                universe_path=universe_path,
                start_date="2015-01-01",
                end_date="2025-12-31",
                train_days=252,
                test_days=63,
                metrics={"win_rate": 0.60, "sharpe_ratio": 1.1}
            )

            assert pack.pack_type == "walk_forward"
            assert pack.config_snapshot["train_days"] == 252
            assert pack.config_snapshot["test_days"] == 63


class TestEvidencePackIntegrity:
    """Tests for pack integrity and tamper detection."""

    def test_modified_pack_fails_verification(self):
        """Modified pack should fail hash verification."""
        builder = EvidencePackBuilder("test")
        builder.set_metrics({"win_rate": 0.64})
        pack = builder.build()

        # Tamper with the pack
        pack.metrics["win_rate"] = 0.99

        # Recompute hash should differ
        new_hash = pack._compute_pack_hash()
        assert new_hash != pack.pack_hash

    def test_saved_pack_integrity_preserved(self):
        """Saved and loaded pack maintains integrity."""
        builder = EvidencePackBuilder("backtest")
        builder.capture_git_state()
        builder.set_metrics({"total_trades": 500, "win_rate": 0.64})
        pack = builder.build()
        original_hash = pack.pack_hash

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = pack.save(Path(tmpdir))

            # Read raw JSON and verify hash is saved
            data = json.loads(filepath.read_text())
            assert data["pack_hash"] == original_hash

            # Load and verify
            loaded = EvidencePack.load(filepath)
            assert loaded.verify_hash() is True


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_metrics(self):
        """Pack handles empty metrics gracefully."""
        builder = EvidencePackBuilder("test")
        pack = builder.build()

        assert pack.metrics == {}
        assert pack.total_trades == 0
        assert pack.win_rate == 0.0

    def test_missing_git(self):
        """Builder handles missing git gracefully."""
        builder = EvidencePackBuilder("test")
        builder.capture_git_state()

        # Should still build even if git info couldn't be captured
        pack = builder.build()
        assert pack.git_commit != ""  # Should have some value

    def test_nonexistent_artifact_ignored(self):
        """Adding nonexistent artifact is a no-op."""
        builder = EvidencePackBuilder("test")
        builder.add_artifact("missing", Path("/nonexistent/file.csv"))

        assert "missing" not in builder.artifacts

    def test_load_nonexistent_file(self):
        """Loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            EvidencePack.load(Path("/nonexistent/pack.json"))
