"""
Kobe Pipeline Architecture
==========================

Complete pipeline architecture for the autonomous trading system.
Each pipeline is self-contained and can run independently or via scheduler.

Pipelines:
    universe_pipeline    - Validate 900-stock universe
    data_audit_pipeline  - Daily data quality check
    snapshot_pipeline    - Create data snapshots
    discovery_pipeline   - Scout for strategy ideas
    spec_pipeline        - Convert ideas to strategy specs
    implementation_pipeline - Generate strategy code
    backtest_pipeline    - Run backtests
    gates_pipeline       - Run quant gates validation
    promotion_pipeline   - Track strategy promotions
    reporting_pipeline   - Generate reports

Legacy Integration:
    quant_rd_pipeline    - Original R&D pipeline (still available)

All pipelines are PAPER-ONLY.
Human approval required for promotion.
"""

# Legacy imports (keep for backward compatibility)
from .quant_rd_pipeline import (
    run_discovery_pipeline,
    run_validation_pipeline,
    run_daily_report_pipeline,
    run_weekly_review_pipeline,
    get_quant_rd_tasks,
)

# New pipeline architecture
from .base import Pipeline, PipelineResult, run_pipeline
from .universe_pipeline import UniversePipeline
from .data_audit_pipeline import DataAuditPipeline
from .snapshot_pipeline import SnapshotPipeline
from .discovery_pipeline import DiscoveryPipeline
from .spec_pipeline import SpecPipeline
from .implementation_pipeline import ImplementationPipeline
from .backtest_pipeline import BacktestPipeline
from .gates_pipeline import GatesPipeline
from .promotion_pipeline import PromotionPipeline
from .reporting_pipeline import ReportingPipeline

__all__ = [
    # Legacy
    "run_discovery_pipeline",
    "run_validation_pipeline",
    "run_daily_report_pipeline",
    "run_weekly_review_pipeline",
    "get_quant_rd_tasks",
    # New architecture
    "Pipeline",
    "PipelineResult",
    "run_pipeline",
    "UniversePipeline",
    "DataAuditPipeline",
    "SnapshotPipeline",
    "DiscoveryPipeline",
    "SpecPipeline",
    "ImplementationPipeline",
    "BacktestPipeline",
    "GatesPipeline",
    "PromotionPipeline",
    "ReportingPipeline",
]

# Pipeline registry for dynamic lookup
PIPELINE_REGISTRY = {
    "universe": UniversePipeline,
    "data_audit": DataAuditPipeline,
    "snapshot": SnapshotPipeline,
    "discovery": DiscoveryPipeline,
    "spec": SpecPipeline,
    "implementation": ImplementationPipeline,
    "backtest": BacktestPipeline,
    "gates": GatesPipeline,
    "promotion": PromotionPipeline,
    "reporting": ReportingPipeline,
}


def get_pipeline(name: str) -> type:
    """Get pipeline class by name."""
    if name not in PIPELINE_REGISTRY:
        raise ValueError(f"Unknown pipeline: {name}. Available: {list(PIPELINE_REGISTRY.keys())}")
    return PIPELINE_REGISTRY[name]


def run_full_sequence(universe_cap: int = 150) -> dict:
    """
    Run full pipeline sequence from discovery to reporting.

    Args:
        universe_cap: Number of stocks to use

    Returns:
        dict with results from each pipeline
    """
    results = {}
    sequence = [
        "universe",
        "data_audit",
        "discovery",
        "spec",
        "implementation",
        "backtest",
        "gates",
        "promotion",
        "reporting",
    ]

    for name in sequence:
        pipeline_cls = get_pipeline(name)
        pipeline = pipeline_cls(universe_cap=universe_cap)
        result = pipeline.run()
        results[name] = result

        if not result.success and name not in ["discovery"]:
            break

    return results
