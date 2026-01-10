"""
Promotion Pipeline - Track strategy promotions through stages.

This pipeline manages the promotion of strategies through:
- discovered -> validated -> proposed -> approved -> implemented

Schedule: On-demand (triggered by gate validation)

Author: Kobe Trading System
Version: 1.0.0
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

from pipelines.base import Pipeline


class PromotionPipeline(Pipeline):
    """Pipeline for managing strategy promotions."""

    @property
    def name(self) -> str:
        return "promotion"

    def execute(self) -> bool:
        """
        Execute promotion tracking.

        Returns:
            True if promotion tracking completed
        """
        self.logger.info("Running promotion pipeline...")

        # Process gate results and update statuses
        updated = self._process_gate_results()

        # Generate promotion summary
        summary = self._generate_summary()

        self.set_metric("strategies_promoted", updated)
        self.set_metric("pending_approval", summary.get("pending_approval", 0))
        self.set_metric("fully_approved", summary.get("approved", 0))

        self.logger.info(f"Promotion tracking complete: {updated} updated")
        return True

    def _process_gate_results(self) -> int:
        """Process gate results and update strategy statuses."""
        updated = 0

        # Load gate results
        gate_results_file = self.state_dir / "gate_results" / "results.jsonl"
        if not gate_results_file.exists():
            return 0

        # Load current promotion status
        promotion_file = self.state_dir / "promotions" / "status.json"
        promotion_file.parent.mkdir(parents=True, exist_ok=True)

        promotions = {}
        if promotion_file.exists():
            promotions = json.loads(promotion_file.read_text())

        # Process each gate result
        with open(gate_results_file) as f:
            for line in f:
                result = json.loads(line)
                strategy_name = result.get("strategy_name", "unknown")

                if strategy_name not in promotions:
                    promotions[strategy_name] = {
                        "strategy_name": strategy_name,
                        "status": "discovered",
                        "created_at": datetime.utcnow().isoformat(),
                        "history": [],
                    }

                current_status = promotions[strategy_name]["status"]
                new_status = self._determine_new_status(
                    current_status, result
                )

                if new_status != current_status:
                    promotions[strategy_name]["status"] = new_status
                    promotions[strategy_name]["history"].append({
                        "from": current_status,
                        "to": new_status,
                        "timestamp": datetime.utcnow().isoformat(),
                        "reason": self._get_promotion_reason(result),
                    })
                    updated += 1

        # Save updated promotions
        promotion_file.write_text(json.dumps(promotions, indent=2))
        self.add_artifact(str(promotion_file))

        return updated

    def _determine_new_status(
        self, current_status: str, gate_result: Dict
    ) -> str:
        """Determine new status based on gate results."""
        all_passed = gate_result.get("all_gates_passed", False)

        status_progression = [
            "discovered",
            "validated",
            "proposed",
            "approved",
            "implemented",
        ]

        current_idx = status_progression.index(current_status)

        if all_passed:
            # Move to next stage
            if current_status == "discovered":
                return "validated"
            elif current_status == "validated":
                return "proposed"
            # approved and implemented require human approval
        else:
            # Don't demote, but don't promote either
            pass

        return current_status

    def _get_promotion_reason(self, gate_result: Dict) -> str:
        """Get reason for promotion based on gate results."""
        if gate_result.get("all_gates_passed", False):
            return "All gates passed"

        failed_gates = []
        for gate_name, gate_data in gate_result.get("gates", {}).items():
            if not gate_data.get("passed", True):
                failed_gates.append(gate_name)

        if failed_gates:
            return f"Failed gates: {', '.join(failed_gates)}"

        return "Status unchanged"

    def _generate_summary(self) -> Dict:
        """Generate promotion summary."""
        summary = {
            "discovered": 0,
            "validated": 0,
            "proposed": 0,
            "pending_approval": 0,
            "approved": 0,
            "implemented": 0,
        }

        promotion_file = self.state_dir / "promotions" / "status.json"
        if not promotion_file.exists():
            return summary

        promotions = json.loads(promotion_file.read_text())
        for strategy, data in promotions.items():
            status = data.get("status", "discovered")
            if status in summary:
                summary[status] += 1

        summary["pending_approval"] = summary["proposed"]

        return summary

    def get_pending_approvals(self) -> List[Dict]:
        """Get list of strategies pending human approval."""
        pending = []

        promotion_file = self.state_dir / "promotions" / "status.json"
        if not promotion_file.exists():
            return pending

        promotions = json.loads(promotion_file.read_text())
        for strategy, data in promotions.items():
            if data.get("status") == "proposed":
                pending.append({
                    "strategy_name": strategy,
                    "proposed_at": data.get("history", [{}])[-1].get("timestamp"),
                    "gate_results": self._get_latest_gate_result(strategy),
                })

        return pending

    def _get_latest_gate_result(self, strategy_name: str) -> Optional[Dict]:
        """Get latest gate result for a strategy."""
        gate_results_file = self.state_dir / "gate_results" / "results.jsonl"
        if not gate_results_file.exists():
            return None

        latest = None
        with open(gate_results_file) as f:
            for line in f:
                result = json.loads(line)
                if result.get("strategy_name") == strategy_name:
                    latest = result

        return latest

    def approve_strategy(self, strategy_name: str, approver: str) -> bool:
        """
        Approve a strategy for implementation.

        This must be called by a human - never automated!
        """
        promotion_file = self.state_dir / "promotions" / "status.json"
        if not promotion_file.exists():
            return False

        promotions = json.loads(promotion_file.read_text())
        if strategy_name not in promotions:
            return False

        if promotions[strategy_name]["status"] != "proposed":
            return False

        promotions[strategy_name]["status"] = "approved"
        promotions[strategy_name]["approved_by"] = approver
        promotions[strategy_name]["approved_at"] = datetime.utcnow().isoformat()
        promotions[strategy_name]["history"].append({
            "from": "proposed",
            "to": "approved",
            "timestamp": datetime.utcnow().isoformat(),
            "reason": f"Approved by {approver}",
        })

        promotion_file.write_text(json.dumps(promotions, indent=2))
        return True
