"""
Agent Orchestrator - Multi-Agent Coordination
==============================================

Coordinates the full R&D pipeline:
1. Scout: Discover ideas
2. Strategist: Convert to StrategySpecs
3. Engineer: Implement code
4. Auditor: Check for bias
5. Risk: Run quant gates
6. Reporter: Generate reports

CRITICAL SAFETY:
- PAPER_ONLY = True (hardcoded)
- Human approval required for promotion
- Reject by default
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base_agent import PAPER_ONLY
from agents.scout_agent import ScoutAgent, IdeaCard
from agents.auditor_agent import AuditorAgent
from agents.risk_agent import RiskAgent
from agents.reporter_agent import ReporterAgent

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Stages in the R&D pipeline."""
    DISCOVERY = "discovery"
    STRATEGY = "strategy"
    ENGINEERING = "engineering"
    AUDIT = "audit"
    RISK = "risk"
    APPROVAL = "approval"
    PROMOTION = "promotion"
    ARCHIVED = "archived"


@dataclass
class PipelineItem:
    """An item moving through the pipeline."""
    id: str
    stage: str
    created_at: str
    updated_at: str
    idea: Optional[Dict[str, Any]] = None
    strategy_spec: Optional[Dict[str, Any]] = None
    audit_result: Optional[Dict[str, Any]] = None
    risk_result: Optional[Dict[str, Any]] = None
    approved: bool = False
    archived_reason: Optional[str] = None


class AgentOrchestrator:
    """
    Coordinates multi-agent R&D pipeline.

    Pipeline flow:
    Scout → Strategist → Engineer → Auditor → Risk → [Human Approval] → Promotion

    REJECT BY DEFAULT:
    - Any stage failure = archive forever
    - Never retest archived items
    - Human approval required for promotion
    """

    # Safety enforcement
    PAPER_ONLY = PAPER_ONLY
    APPROVE_LIVE_ACTION = False

    def __init__(self, output_dir: str = "pipeline_output"):
        """
        Initialize orchestrator.

        Args:
            output_dir: Directory for pipeline output
        """
        if not self.PAPER_ONLY:
            raise RuntimeError("SAFETY VIOLATION: Orchestrator must be paper-only")

        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._pipeline_items: Dict[str, PipelineItem] = {}
        self._agents = {
            "scout": ScoutAgent(),
            "auditor": AuditorAgent(),
            "risk": RiskAgent(),
            "reporter": ReporterAgent(),
        }

    def run_discovery_cycle(self, sources: Optional[List[str]] = None) -> List[IdeaCard]:
        """
        Run a discovery cycle with the Scout agent.

        Args:
            sources: Optional list of sources to check

        Returns:
            List of discovered IdeaCards
        """
        logger.info("Starting discovery cycle")

        scout = self._agents["scout"]

        task = "Search for new strategy ideas from external sources. "
        if sources:
            task += f"Focus on: {', '.join(sources)}. "
        task += "Create IdeaCards for any promising discoveries."

        result = scout.run(task)

        if result.success:
            ideas = scout.get_discovered_ideas()
            logger.info(f"Discovery cycle complete: {len(ideas)} ideas found")

            # Save ideas to pipeline
            for idea in ideas:
                self._add_to_pipeline(idea)

            return ideas
        else:
            logger.warning(f"Discovery cycle failed: {result.error}")
            return []

    def _add_to_pipeline(self, idea: IdeaCard) -> str:
        """Add an idea to the pipeline."""
        item = PipelineItem(
            id=idea.id,
            stage=PipelineStage.DISCOVERY.value,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            idea=asdict(idea),
        )
        self._pipeline_items[item.id] = item
        self._save_pipeline_state()
        return item.id

    def run_audit_cycle(self, item_id: str) -> bool:
        """
        Run audit on a pipeline item.

        Args:
            item_id: Pipeline item ID

        Returns:
            True if passed, False if failed/archived
        """
        if item_id not in self._pipeline_items:
            logger.error(f"Item not found: {item_id}")
            return False

        item = self._pipeline_items[item_id]

        if item.stage == PipelineStage.ARCHIVED.value:
            logger.warning(f"Item already archived: {item_id}")
            return False

        logger.info(f"Running audit on: {item_id}")

        auditor = self._agents["auditor"]

        # Determine what to audit
        if item.strategy_spec:
            target = json.dumps(item.strategy_spec, indent=2)
        elif item.idea:
            target = json.dumps(item.idea, indent=2)
        else:
            logger.error(f"Nothing to audit for: {item_id}")
            return False

        task = f"Audit this strategy/idea for bias, leakage, and integrity issues:\n\n{target}"

        result = auditor.run(task)

        if result.success and not auditor.has_critical_findings():
            # Passed audit
            item.stage = PipelineStage.RISK.value
            item.audit_result = {"passed": True, "findings": len(auditor.get_findings())}
            item.updated_at = datetime.now().isoformat()
            logger.info(f"Audit PASSED for: {item_id}")
            self._save_pipeline_state()
            return True
        else:
            # Failed audit - archive forever
            item.stage = PipelineStage.ARCHIVED.value
            item.audit_result = {"passed": False, "findings": len(auditor.get_findings())}
            item.archived_reason = "AUDIT_FAILED"
            item.updated_at = datetime.now().isoformat()
            logger.warning(f"Audit FAILED for: {item_id} - ARCHIVED")
            self._save_pipeline_state()
            return False

    def run_risk_cycle(self, item_id: str) -> bool:
        """
        Run risk/quant gates on a pipeline item.

        Args:
            item_id: Pipeline item ID

        Returns:
            True if passed all gates, False if failed
        """
        if item_id not in self._pipeline_items:
            logger.error(f"Item not found: {item_id}")
            return False

        item = self._pipeline_items[item_id]

        if item.stage != PipelineStage.RISK.value:
            logger.warning(f"Item not ready for risk: {item_id} (stage: {item.stage})")
            return False

        logger.info(f"Running risk gates on: {item_id}")

        risk = self._agents["risk"]

        target = item.strategy_spec or item.idea
        task = f"Evaluate this strategy against all quant gates (0-4):\n\n{json.dumps(target, indent=2)}"

        result = risk.run(task)

        if result.success and risk.all_gates_passed():
            # Passed all gates - ready for approval
            item.stage = PipelineStage.APPROVAL.value
            item.risk_result = {"passed": True, "gates_passed": 5}
            item.updated_at = datetime.now().isoformat()
            logger.info(f"Risk gates PASSED for: {item_id} - AWAITING APPROVAL")
            self._save_pipeline_state()
            return True
        else:
            # Failed gates - archive forever
            item.stage = PipelineStage.ARCHIVED.value
            item.risk_result = {"passed": False}
            item.archived_reason = "RISK_GATE_FAILED"
            item.updated_at = datetime.now().isoformat()
            logger.warning(f"Risk gates FAILED for: {item_id} - ARCHIVED")
            self._save_pipeline_state()
            return False

    def approve_item(self, item_id: str, approver: str) -> bool:
        """
        Human approval for promotion.

        Args:
            item_id: Pipeline item ID
            approver: Name of human approver

        Returns:
            True if approved and promoted
        """
        if not self.PAPER_ONLY:
            raise RuntimeError("SAFETY: Cannot approve in live mode")

        if item_id not in self._pipeline_items:
            logger.error(f"Item not found: {item_id}")
            return False

        item = self._pipeline_items[item_id]

        if item.stage != PipelineStage.APPROVAL.value:
            logger.warning(f"Item not ready for approval: {item_id} (stage: {item.stage})")
            return False

        # Record approval
        item.approved = True
        item.stage = PipelineStage.PROMOTION.value
        item.updated_at = datetime.now().isoformat()

        logger.info(f"Item APPROVED by {approver}: {item_id}")
        self._save_pipeline_state()

        return True

    def generate_daily_report(self) -> str:
        """Generate daily summary report."""
        reporter = self._agents["reporter"]

        # Gather stats
        stats = {
            "total_items": len(self._pipeline_items),
            "by_stage": {},
        }

        for item in self._pipeline_items.values():
            stage = item.stage
            stats["by_stage"][stage] = stats["by_stage"].get(stage, 0) + 1

        task = f"""Generate a daily report with the following pipeline stats:
{json.dumps(stats, indent=2)}

Include sections for:
1. System Health
2. Pipeline Summary (items per stage)
3. Issues & Alerts
4. Tomorrow's Focus
"""

        result = reporter.run(task)
        return result.output

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "total_items": len(self._pipeline_items),
            "items_by_stage": {
                stage.value: sum(1 for i in self._pipeline_items.values() if i.stage == stage.value)
                for stage in PipelineStage
            },
            "items": [asdict(item) for item in self._pipeline_items.values()],
        }

    def _save_pipeline_state(self) -> None:
        """Save pipeline state to disk."""
        state_file = self._output_dir / "pipeline_state.json"
        with open(state_file, "w") as f:
            json.dump(
                {id: asdict(item) for id, item in self._pipeline_items.items()},
                f,
                indent=2,
            )

    def _load_pipeline_state(self) -> None:
        """Load pipeline state from disk."""
        state_file = self._output_dir / "pipeline_state.json"
        if state_file.exists():
            with open(state_file) as f:
                data = json.load(f)
            for id, item_data in data.items():
                self._pipeline_items[id] = PipelineItem(**item_data)


# =============================================================================
# Convenience Functions
# =============================================================================

_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    """Get or create the global orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


def run_hourly_cycle() -> Dict[str, Any]:
    """Run hourly light cycle (discovery)."""
    orchestrator = get_orchestrator()

    # Run discovery
    ideas = orchestrator.run_discovery_cycle()

    return {
        "cycle": "hourly",
        "timestamp": datetime.now().isoformat(),
        "ideas_discovered": len(ideas),
    }


def run_nightly_cycle() -> Dict[str, Any]:
    """Run nightly heavy cycle (audit, risk, report)."""
    orchestrator = get_orchestrator()

    status = orchestrator.get_pipeline_status()

    # Process items in pipeline
    audited = 0
    risk_checked = 0

    for item_id, item in list(orchestrator._pipeline_items.items()):
        if item.stage == PipelineStage.DISCOVERY.value:
            orchestrator.run_audit_cycle(item_id)
            audited += 1
        elif item.stage == PipelineStage.RISK.value:
            orchestrator.run_risk_cycle(item_id)
            risk_checked += 1

    # Generate report
    report = orchestrator.generate_daily_report()

    return {
        "cycle": "nightly",
        "timestamp": datetime.now().isoformat(),
        "audited": audited,
        "risk_checked": risk_checked,
        "report_generated": True,
    }
