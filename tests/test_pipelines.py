"""
Tests for the pipeline architecture.

Verifies:
- All pipelines can be imported
- Pipeline registry is complete
- Base pipeline class works
- PipelineResult dataclass works
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPipelineImports:
    """Test all pipeline imports work."""

    def test_import_base_pipeline(self):
        """Base pipeline should be importable."""
        from pipelines.base import Pipeline, PipelineResult
        assert Pipeline is not None
        assert PipelineResult is not None

    def test_import_universe_pipeline(self):
        """Universe pipeline should be importable."""
        from pipelines.universe_pipeline import UniversePipeline
        assert UniversePipeline is not None

    def test_import_data_audit_pipeline(self):
        """Data audit pipeline should be importable."""
        from pipelines.data_audit_pipeline import DataAuditPipeline
        assert DataAuditPipeline is not None

    def test_import_discovery_pipeline(self):
        """Discovery pipeline should be importable."""
        from pipelines.discovery_pipeline import DiscoveryPipeline
        assert DiscoveryPipeline is not None

    def test_import_spec_pipeline(self):
        """Spec pipeline should be importable."""
        from pipelines.spec_pipeline import SpecPipeline
        assert SpecPipeline is not None

    def test_import_backtest_pipeline(self):
        """Backtest pipeline should be importable."""
        from pipelines.backtest_pipeline import BacktestPipeline
        assert BacktestPipeline is not None

    def test_import_gates_pipeline(self):
        """Gates pipeline should be importable."""
        from pipelines.gates_pipeline import GatesPipeline
        assert GatesPipeline is not None


class TestPipelineRegistry:
    """Test pipeline registry functionality."""

    def test_pipeline_registry_exists(self):
        """PIPELINE_REGISTRY should exist."""
        from pipelines import PIPELINE_REGISTRY
        assert isinstance(PIPELINE_REGISTRY, dict)

    def test_pipeline_registry_has_10_pipelines(self):
        """Registry should have all 10 pipelines."""
        from pipelines import PIPELINE_REGISTRY

        expected = [
            "universe",
            "data_audit",
            "snapshot",
            "discovery",
            "spec",
            "implementation",
            "backtest",
            "gates",
            "promotion",
            "reporting",
        ]

        for name in expected:
            assert name in PIPELINE_REGISTRY, f"Missing pipeline: {name}"

    def test_get_pipeline_function(self):
        """get_pipeline should return pipeline class."""
        from pipelines import get_pipeline

        cls = get_pipeline("universe")
        assert cls is not None
        assert hasattr(cls, "execute")

    def test_get_pipeline_raises_for_unknown(self):
        """get_pipeline should raise for unknown pipeline."""
        from pipelines import get_pipeline

        with pytest.raises(ValueError) as exc_info:
            get_pipeline("nonexistent")

        assert "Unknown pipeline" in str(exc_info.value)


class TestPipelineResult:
    """Test PipelineResult dataclass."""

    def test_pipeline_result_creation(self):
        """PipelineResult can be created."""
        from pipelines.base import PipelineResult
        from datetime import datetime

        result = PipelineResult(
            success=True,
            pipeline_name="test",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration_seconds=1.0,
        )

        assert result.success is True
        assert result.pipeline_name == "test"

    def test_pipeline_result_to_dict(self):
        """PipelineResult.to_dict() works."""
        from pipelines.base import PipelineResult
        from datetime import datetime

        result = PipelineResult(
            success=True,
            pipeline_name="test",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration_seconds=1.0,
        )

        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["success"] is True
        assert d["pipeline_name"] == "test"


class TestBasePipeline:
    """Test base Pipeline class."""

    def test_pipeline_has_required_methods(self):
        """Pipeline has required abstract methods."""
        from pipelines.base import Pipeline

        # These should be defined (even if abstract)
        assert hasattr(Pipeline, "name")
        assert hasattr(Pipeline, "execute")
        assert hasattr(Pipeline, "run")

    def test_pipeline_helper_methods(self):
        """Pipeline has helper methods."""
        from pipelines.base import Pipeline

        assert hasattr(Pipeline, "add_error")
        assert hasattr(Pipeline, "add_warning")
        assert hasattr(Pipeline, "add_artifact")
        assert hasattr(Pipeline, "set_metric")
        assert hasattr(Pipeline, "load_universe")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
