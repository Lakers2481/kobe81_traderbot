"""
Pipelines - 24/7 R&D Factory Integration
========================================

Wires the quant R&D system into the 24/7 scheduler:
- Hourly: Discovery cycle (Scout agent)
- Nightly: Validation cycle (Audit, Risk, Report)
- Weekly: Deep review (Walk-forward, stress tests)

All pipelines are PAPER-ONLY.
Human approval required for promotion.
"""

from .quant_rd_pipeline import (
    run_discovery_pipeline,
    run_validation_pipeline,
    run_daily_report_pipeline,
    run_weekly_review_pipeline,
    get_quant_rd_tasks,
)

__all__ = [
    "run_discovery_pipeline",
    "run_validation_pipeline",
    "run_daily_report_pipeline",
    "run_weekly_review_pipeline",
    "get_quant_rd_tasks",
]
