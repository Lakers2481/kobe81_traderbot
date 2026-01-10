"""
Quant R&D Pipeline Integration
==============================

Integrates the Quant R&D Factory into the 24/7 scheduler.

Pipelines:
- Discovery: Hourly Scout cycles
- Validation: Nightly Audit + Risk gates
- Daily Report: End-of-day summary
- Weekly Review: Deep walk-forward + stress tests

SAFETY:
- PAPER_ONLY = True (hardcoded)
- No live trading
- Human approval required for promotion
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Safety enforcement
PAPER_ONLY = True


@dataclass
class PipelineRun:
    """Record of a pipeline run."""
    id: str
    pipeline: str
    timestamp: str
    duration_seconds: float
    success: bool
    items_processed: int
    errors: List[str]
    summary: str


def run_discovery_pipeline(
    sources: Optional[List[str]] = None,
    max_ideas: int = 10,
) -> PipelineRun:
    """
    Run discovery pipeline (hourly).

    Uses Scout agent to find new strategy ideas.

    Args:
        sources: Optional list of sources to check
        max_ideas: Maximum ideas to discover per run

    Returns:
        PipelineRun result
    """
    if not PAPER_ONLY:
        raise RuntimeError("SAFETY: Discovery pipeline must be paper-only")

    start = datetime.now()
    errors = []
    items = 0

    try:
        from agents import get_orchestrator

        orchestrator = get_orchestrator()
        ideas = orchestrator.run_discovery_cycle(sources)
        items = len(ideas)

        summary = f"Discovered {items} ideas"
        if ideas:
            summary += f": {', '.join(i.title for i in ideas[:3])}"

    except Exception as e:
        logger.error(f"Discovery pipeline failed: {e}")
        errors.append(str(e))
        summary = f"Failed: {e}"

    duration = (datetime.now() - start).total_seconds()

    return PipelineRun(
        id=f"discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        pipeline="discovery",
        timestamp=datetime.now().isoformat(),
        duration_seconds=duration,
        success=len(errors) == 0,
        items_processed=items,
        errors=errors,
        summary=summary,
    )


def run_validation_pipeline(
    strategy_ids: Optional[List[str]] = None,
) -> PipelineRun:
    """
    Run validation pipeline (nightly).

    Runs all pending items through Audit + Risk gates.

    Args:
        strategy_ids: Specific strategies to validate (or all pending)

    Returns:
        PipelineRun result
    """
    if not PAPER_ONLY:
        raise RuntimeError("SAFETY: Validation pipeline must be paper-only")

    start = datetime.now()
    errors = []
    items = 0
    passed = 0
    failed = 0

    try:
        from agents import get_orchestrator
        from quant_gates import QuantGatesPipeline

        orchestrator = get_orchestrator()
        pipeline = QuantGatesPipeline()

        # Get pending items
        status = orchestrator.get_pipeline_status()
        pending = [
            item for item in status.get("items", [])
            if item.get("stage") in ["discovery", "strategy", "audit"]
        ]

        if strategy_ids:
            pending = [i for i in pending if i.get("id") in strategy_ids]

        for item in pending:
            items += 1
            try:
                item_id = item.get("id")

                # Run audit
                if item.get("stage") in ["discovery", "strategy"]:
                    audit_ok = orchestrator.run_audit_cycle(item_id)
                    if not audit_ok:
                        failed += 1
                        continue

                # Run risk gates
                risk_ok = orchestrator.run_risk_cycle(item_id)
                if risk_ok:
                    passed += 1
                else:
                    failed += 1

            except Exception as e:
                errors.append(f"Item {item.get('id')}: {e}")
                failed += 1

        summary = f"Validated {items} items: {passed} passed, {failed} failed"

    except Exception as e:
        logger.error(f"Validation pipeline failed: {e}")
        errors.append(str(e))
        summary = f"Failed: {e}"

    duration = (datetime.now() - start).total_seconds()

    return PipelineRun(
        id=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        pipeline="validation",
        timestamp=datetime.now().isoformat(),
        duration_seconds=duration,
        success=len(errors) == 0,
        items_processed=items,
        errors=errors,
        summary=summary,
    )


def run_daily_report_pipeline() -> PipelineRun:
    """
    Run daily report pipeline.

    Generates end-of-day summary report.

    Returns:
        PipelineRun result
    """
    start = datetime.now()
    errors = []

    try:
        from agents import get_orchestrator

        orchestrator = get_orchestrator()
        report = orchestrator.generate_daily_report()

        # Save report
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        report_file = reports_dir / f"daily_{datetime.now().strftime('%Y%m%d')}.md"
        with open(report_file, "w") as f:
            f.write(report)

        summary = f"Report saved to {report_file}"

    except Exception as e:
        logger.error(f"Daily report pipeline failed: {e}")
        errors.append(str(e))
        summary = f"Failed: {e}"

    duration = (datetime.now() - start).total_seconds()

    return PipelineRun(
        id=f"daily_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        pipeline="daily_report",
        timestamp=datetime.now().isoformat(),
        duration_seconds=duration,
        success=len(errors) == 0,
        items_processed=1,
        errors=errors,
        summary=summary,
    )


def run_weekly_review_pipeline() -> PipelineRun:
    """
    Run weekly deep review pipeline.

    Includes:
    - Walk-forward rebuild
    - Stress testing
    - Regime bucket analysis

    Returns:
        PipelineRun result
    """
    if not PAPER_ONLY:
        raise RuntimeError("SAFETY: Weekly review must be paper-only")

    start = datetime.now()
    errors = []
    items = 0

    try:
        from quant_gates import Gate2Robustness

        robustness_gate = Gate2Robustness()

        # Would run full walk-forward on production strategies here
        # For now, just a placeholder

        summary = "Weekly review completed (placeholder)"
        items = 1

    except Exception as e:
        logger.error(f"Weekly review pipeline failed: {e}")
        errors.append(str(e))
        summary = f"Failed: {e}"

    duration = (datetime.now() - start).total_seconds()

    return PipelineRun(
        id=f"weekly_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        pipeline="weekly_review",
        timestamp=datetime.now().isoformat(),
        duration_seconds=duration,
        success=len(errors) == 0,
        items_processed=items,
        errors=errors,
        summary=summary,
    )


def get_quant_rd_tasks() -> List[Dict[str, Any]]:
    """
    Get quant R&D tasks for the scheduler.

    Returns:
        List of task definitions compatible with TaskQueue
    """
    return [
        {
            "id": "quant_rd_discovery",
            "name": "Quant R&D Discovery Cycle",
            "category": "discovery",
            "priority": 4,  # LOW - background
            "description": "Scout for new strategy ideas from external sources",
            "handler": "pipelines.quant_rd_pipeline.run_discovery_pipeline",
            "params": {},
            "recurring": True,
            "recurrence_minutes": 60,  # Hourly
            "cooldown_minutes": 55,
        },
        {
            "id": "quant_rd_validation",
            "name": "Quant R&D Validation Cycle",
            "category": "research",
            "priority": 3,  # NORMAL
            "description": "Run audit and risk gates on pending strategies",
            "handler": "pipelines.quant_rd_pipeline.run_validation_pipeline",
            "params": {},
            "recurring": True,
            "recurrence_minutes": 1440,  # Daily (nightly)
            "cooldown_minutes": 1380,
        },
        {
            "id": "quant_rd_daily_report",
            "name": "Quant R&D Daily Report",
            "category": "maintenance",
            "priority": 3,  # NORMAL
            "description": "Generate daily summary report",
            "handler": "pipelines.quant_rd_pipeline.run_daily_report_pipeline",
            "params": {},
            "recurring": True,
            "recurrence_minutes": 1440,  # Daily
            "cooldown_minutes": 1380,
        },
        {
            "id": "quant_rd_weekly_review",
            "name": "Quant R&D Weekly Review",
            "category": "optimization",
            "priority": 4,  # LOW
            "description": "Deep walk-forward and stress testing",
            "handler": "pipelines.quant_rd_pipeline.run_weekly_review_pipeline",
            "params": {},
            "recurring": True,
            "recurrence_minutes": 10080,  # Weekly
            "cooldown_minutes": 10000,
        },
    ]
