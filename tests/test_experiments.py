"""
Tests for Experiment Registry.

Tests experiment registration, results recording, and reproducibility.
"""
import pytest
import tempfile
from pathlib import Path
import json

from experiments import (
    ExperimentRegistry,
    ExperimentConfig,
    ExperimentResults,
    Experiment,
    ExperimentStatus,
)


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_config_creation(self):
        """Should create valid config."""
        config = ExperimentConfig(
            dataset_id="test_dataset",
            strategy="momentum",
            params={"lookback": 20},
            seed=42,
        )

        assert config.dataset_id == "test_dataset"
        assert config.strategy == "momentum"
        assert config.seed == 42

    def test_config_hash_deterministic(self):
        """Same config should produce same hash."""
        config1 = ExperimentConfig(
            dataset_id="test_dataset",
            strategy="momentum",
            params={"lookback": 20},
            seed=42,
        )

        config2 = ExperimentConfig(
            dataset_id="test_dataset",
            strategy="momentum",
            params={"lookback": 20},
            seed=42,
        )

        assert config1.config_hash == config2.config_hash

    def test_different_configs_different_hash(self):
        """Different configs should produce different hashes."""
        config1 = ExperimentConfig(
            dataset_id="test_dataset",
            strategy="momentum",
            params={"lookback": 20},
            seed=42,
        )

        config2 = ExperimentConfig(
            dataset_id="test_dataset",
            strategy="momentum",
            params={"lookback": 30},  # Different
            seed=42,
        )

        assert config1.config_hash != config2.config_hash

    def test_to_backtest_args(self):
        """Should convert to backtest arguments."""
        config = ExperimentConfig(
            dataset_id="test_dataset",
            strategy="momentum",
            params={"lookback": 20},
            seed=42,
            initial_equity=50_000,
        )

        args = config.to_backtest_args()

        assert args["dataset_id"] == "test_dataset"
        assert args["strategy"] == "momentum"
        assert args["seed"] == 42
        assert args["initial_equity"] == 50_000


class TestExperimentResults:
    """Tests for ExperimentResults."""

    def test_results_creation(self):
        """Should create valid results."""
        results = ExperimentResults(
            total_trades=100,
            win_rate=0.55,
            sharpe_ratio=1.2,
            profit_factor=1.5,
            max_drawdown=0.15,
            total_pnl=10_000,
        )

        assert results.total_trades == 100
        assert results.win_rate == 0.55

    def test_results_hash(self):
        """Should compute results hash."""
        results = ExperimentResults(
            total_trades=100,
            win_rate=0.55,
            sharpe_ratio=1.2,
            profit_factor=1.5,
            max_drawdown=0.15,
            total_pnl=10_000,
        )

        hash1 = results.compute_hash()

        assert isinstance(hash1, str)
        assert len(hash1) == 12

    def test_same_results_same_hash(self):
        """Same results should produce same hash."""
        results1 = ExperimentResults(
            total_trades=100,
            win_rate=0.55,
            sharpe_ratio=1.2,
            profit_factor=1.5,
            max_drawdown=0.15,
            total_pnl=10_000,
        )

        results2 = ExperimentResults(
            total_trades=100,
            win_rate=0.55,
            sharpe_ratio=1.2,
            profit_factor=1.5,
            max_drawdown=0.15,
            total_pnl=10_000,
        )

        assert results1.compute_hash() == results2.compute_hash()


class TestExperimentRegistry:
    """Tests for ExperimentRegistry."""

    @pytest.fixture
    def temp_registry(self):
        """Create temporary registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExperimentRegistry(
                registry_dir=tmpdir,
                registry_file="test_registry.json",
            )
            yield registry

    def test_register_experiment(self, temp_registry):
        """Should register new experiment."""
        exp_id = temp_registry.register(
            name="test_experiment",
            dataset_id="test_dataset",
            strategy="momentum",
            params={"lookback": 20},
        )

        assert exp_id is not None
        assert exp_id.startswith("exp_")

    def test_get_experiment(self, temp_registry):
        """Should retrieve registered experiment."""
        exp_id = temp_registry.register(
            name="test_experiment",
            dataset_id="test_dataset",
            strategy="momentum",
            params={"lookback": 20},
        )

        exp = temp_registry.get_experiment(exp_id)

        assert exp is not None
        assert exp.name == "test_experiment"
        assert exp.status == ExperimentStatus.REGISTERED

    def test_record_results(self, temp_registry):
        """Should record experiment results."""
        exp_id = temp_registry.register(
            name="test_experiment",
            dataset_id="test_dataset",
            strategy="momentum",
            params={"lookback": 20},
        )

        temp_registry.record_results(exp_id, {
            "total_trades": 100,
            "win_rate": 0.55,
            "sharpe_ratio": 1.2,
            "profit_factor": 1.5,
            "max_drawdown": 0.15,
            "total_pnl": 10_000,
        })

        exp = temp_registry.get_experiment(exp_id)

        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.results is not None
        assert exp.results.sharpe_ratio == 1.2

    def test_list_experiments(self, temp_registry):
        """Should list experiments."""
        temp_registry.register(
            name="exp1",
            dataset_id="ds1",
            strategy="momentum",
            params={},
        )
        temp_registry.register(
            name="exp2",
            dataset_id="ds2",
            strategy="mean_reversion",
            params={},
        )

        experiments = temp_registry.list_experiments()

        assert len(experiments) == 2

    def test_list_experiments_filter_strategy(self, temp_registry):
        """Should filter by strategy."""
        temp_registry.register(
            name="exp1",
            dataset_id="ds1",
            strategy="momentum",
            params={},
        )
        temp_registry.register(
            name="exp2",
            dataset_id="ds2",
            strategy="mean_reversion",
            params={},
        )

        experiments = temp_registry.list_experiments(strategy="momentum")

        assert len(experiments) == 1
        assert experiments[0].config.strategy == "momentum"

    def test_get_config(self, temp_registry):
        """Should get config for reproduction."""
        exp_id = temp_registry.register(
            name="test_experiment",
            dataset_id="test_dataset",
            strategy="momentum",
            params={"lookback": 20},
            seed=42,
        )

        config = temp_registry.get_config(exp_id)

        assert config is not None
        assert config.dataset_id == "test_dataset"
        assert config.seed == 42

    def test_persistence(self):
        """Registry should persist across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and register
            registry1 = ExperimentRegistry(
                registry_dir=tmpdir,
                registry_file="test_registry.json",
            )
            exp_id = registry1.register(
                name="persistent_exp",
                dataset_id="test_dataset",
                strategy="momentum",
                params={},
            )

            # New instance should load existing experiments
            registry2 = ExperimentRegistry(
                registry_dir=tmpdir,
                registry_file="test_registry.json",
            )

            exp = registry2.get_experiment(exp_id)
            assert exp is not None
            assert exp.name == "persistent_exp"

    def test_verify_reproducibility(self, temp_registry):
        """Should verify result reproducibility."""
        exp_id = temp_registry.register(
            name="test_experiment",
            dataset_id="test_dataset",
            strategy="momentum",
            params={"lookback": 20},
        )

        results = {
            "total_trades": 100,
            "win_rate": 0.55,
            "sharpe_ratio": 1.2,
            "profit_factor": 1.5,
            "max_drawdown": 0.15,
            "total_pnl": 10_000,
        }

        temp_registry.record_results(exp_id, results)

        # Same results should verify
        assert temp_registry.verify_reproducibility(exp_id, results)

        # Different results should fail
        different = results.copy()
        different["sharpe_ratio"] = 0.8
        assert not temp_registry.verify_reproducibility(exp_id, different)


# Run with: pytest tests/test_experiments.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
