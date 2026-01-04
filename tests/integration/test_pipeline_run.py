"""
Integration test for pipeline runs.

Verifies pipelines can be instantiated and run.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPipelineInstantiation:
    """Test pipeline instantiation."""

    def test_universe_pipeline_instantiation(self):
        """Universe pipeline can be instantiated."""
        from pipelines.universe_pipeline import UniversePipeline

        pipeline = UniversePipeline(universe_cap=10)
        assert pipeline is not None
        assert pipeline.name == "universe"

    def test_data_audit_pipeline_instantiation(self):
        """Data audit pipeline can be instantiated."""
        from pipelines.data_audit_pipeline import DataAuditPipeline

        pipeline = DataAuditPipeline(universe_cap=10)
        assert pipeline is not None
        assert pipeline.name == "data_audit"

    def test_discovery_pipeline_instantiation(self):
        """Discovery pipeline can be instantiated."""
        from pipelines.discovery_pipeline import DiscoveryPipeline

        pipeline = DiscoveryPipeline(universe_cap=10)
        assert pipeline is not None
        assert pipeline.name == "discovery"

    def test_reporting_pipeline_instantiation(self):
        """Reporting pipeline can be instantiated."""
        from pipelines.reporting_pipeline import ReportingPipeline

        pipeline = ReportingPipeline(universe_cap=10)
        assert pipeline is not None
        assert pipeline.name == "reporting"


class TestPipelineRun:
    """Test pipeline execution."""

    def test_discovery_pipeline_runs(self):
        """Discovery pipeline should run (may find nothing)."""
        from pipelines.discovery_pipeline import DiscoveryPipeline

        pipeline = DiscoveryPipeline(universe_cap=10)
        result = pipeline.run()

        # Should complete successfully (even if no discoveries)
        assert result is not None
        assert result.pipeline_name == "discovery"

    def test_reporting_pipeline_runs(self):
        """Reporting pipeline should run."""
        from pipelines.reporting_pipeline import ReportingPipeline

        pipeline = ReportingPipeline(universe_cap=10)
        result = pipeline.run()

        assert result is not None
        assert result.pipeline_name == "reporting"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
