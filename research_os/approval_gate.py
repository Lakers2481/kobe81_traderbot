"""
Approval Gate - Human control for engineering changes.

CRITICAL SAFETY SYSTEM:
    - APPROVE_LIVE_ACTION = False by default (NEVER change programmatically)
    - NO AUTO-MERGE. EVER.
    - NO AUTO-LIVE-TRADING. EVER.
    - All changes require explicit human approval via CLI command

This is the gate between research discoveries and production changes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from .proposal import ResearchProposal, ProposalStore, ProposalStatus

logger = logging.getLogger(__name__)


# ============================================================================
# CRITICAL SAFETY FLAG - NEVER CHANGE THIS PROGRAMMATICALLY
# ============================================================================
APPROVE_LIVE_ACTION = False
"""
This flag MUST be False by default.

To enable live action implementation, a human must:
1. Manually edit this file
2. Change APPROVE_LIVE_ACTION to True
3. Restart the system

The system will NEVER change this flag automatically.
"""

# ============================================================================
# SECONDARY APPROVAL FLAG - ADDITIONAL SAFETY LAYER
# ============================================================================
APPROVE_LIVE_ACTION_2 = False
"""
Secondary approval flag for live trading - NEVER CHANGE PROGRAMMATICALLY.

This is a SECOND independent approval required for live orders.
Both APPROVE_LIVE_ACTION and APPROVE_LIVE_ACTION_2 must be True.

This provides defense in depth - if one flag is accidentally set,
the other still blocks live trading.
"""


class SafetyError(Exception):
    """Raised when safety constraints are violated."""
    pass


@dataclass
class ApprovalRecord:
    """Record of an approval decision."""
    request_id: str
    proposal_id: str
    approver: str
    approved: bool
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalRecord":
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class ApprovalGate:
    """
    Human approval gate for all engineering changes.

    NO AUTO-MERGE. NO AUTO-LIVE. EVER.

    This class manages the approval workflow:
    1. Proposals are submitted for review
    2. Humans review pending proposals via CLI
    3. Humans explicitly approve or reject via CLI command
    4. Only approved proposals can be implemented
    5. Even after approval, APPROVE_LIVE_ACTION must be True to apply

    Usage:
        gate = ApprovalGate()

        # Submit proposal (automated)
        request_id = gate.request_approval(proposal)

        # List pending (human reviews via CLI)
        pending = gate.get_pending()

        # Approve (human action via CLI)
        gate.approve(request_id, approver="John Doe")

        # Reject (human action via CLI)
        gate.reject(request_id, reason="Overfitting suspected")
    """

    def __init__(self, state_dir: Optional[Path] = None):
        if state_dir is None:
            state_dir = Path(__file__).parent.parent / "state" / "research_os"
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.pending_path = self.state_dir / "pending_approvals.json"
        self.approved_path = self.state_dir / "approved_changes.json"
        self.rejected_path = self.state_dir / "rejected.json"
        self.audit_log_path = self.state_dir / "approval_audit.jsonl"

        self.proposal_store = ProposalStore(self.state_dir / "proposals.json")
        self._ensure_files_exist()

    def _ensure_files_exist(self) -> None:
        """Create state files if they don't exist."""
        for path in [self.pending_path, self.approved_path, self.rejected_path]:
            if not path.exists():
                path.write_text("[]")

    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSON list from file."""
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_json(self, path: Path, data: List[Dict[str, Any]]) -> None:
        """Save JSON list to file."""
        path.write_text(json.dumps(data, indent=2))

    def _log_audit(self, action: str, data: Dict[str, Any]) -> None:
        """Append to audit log."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            **data,
        }
        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def request_approval(self, proposal: ResearchProposal) -> str:
        """
        Submit a proposal for human approval.

        Returns request_id for tracking.
        """
        if proposal.status not in [ProposalStatus.DRAFT, ProposalStatus.SUBMITTED]:
            raise ValueError(f"Proposal {proposal.proposal_id} not in submittable state")

        request_id = f"req_{proposal.proposal_id}"

        # Update proposal status
        if proposal.status == ProposalStatus.DRAFT:
            proposal.submit("research_os")

        # Save to pending
        pending = self._load_json(self.pending_path)
        pending.append({
            "request_id": request_id,
            "proposal_id": proposal.proposal_id,
            "title": proposal.title,
            "change_type": proposal.change_type.value,
            "target_file": proposal.target_file,
            "submitted_at": datetime.utcnow().isoformat(),
        })
        self._save_json(self.pending_path, pending)

        # Save proposal
        self.proposal_store.save(proposal)

        # Audit log
        self._log_audit("REQUEST_APPROVAL", {
            "request_id": request_id,
            "proposal_id": proposal.proposal_id,
            "title": proposal.title,
        })

        logger.info(f"Approval requested: {request_id} for {proposal.title}")
        return request_id

    def approve(self, request_id: str, approver: str, notes: str = "") -> bool:
        """
        Human approves a proposal.

        This is a HUMAN action - must be called via CLI.
        Returns True if approval successful.
        """
        # Find pending request
        pending = self._load_json(self.pending_path)
        request = None
        for i, p in enumerate(pending):
            if p.get("request_id") == request_id:
                request = p
                pending.pop(i)
                break

        if request is None:
            logger.warning(f"Request not found: {request_id}")
            return False

        # Get and update proposal
        proposal = self.proposal_store.get(request["proposal_id"])
        if proposal is None:
            logger.error(f"Proposal not found: {request['proposal_id']}")
            return False

        proposal.approve(approver, notes)
        self.proposal_store.save(proposal)

        # Move to approved
        approved = self._load_json(self.approved_path)
        approved.append({
            **request,
            "approved_by": approver,
            "approved_at": datetime.utcnow().isoformat(),
            "notes": notes,
        })
        self._save_json(self.approved_path, approved)
        self._save_json(self.pending_path, pending)

        # Audit log
        self._log_audit("APPROVED", {
            "request_id": request_id,
            "proposal_id": request["proposal_id"],
            "approver": approver,
            "notes": notes,
        })

        logger.info(f"APPROVED by {approver}: {request_id}")
        return True

    def reject(self, request_id: str, rejector: str, reason: str) -> bool:
        """
        Human rejects a proposal.

        This is a HUMAN action - must be called via CLI.
        Returns True if rejection successful.
        """
        # Find pending request
        pending = self._load_json(self.pending_path)
        request = None
        for i, p in enumerate(pending):
            if p.get("request_id") == request_id:
                request = p
                pending.pop(i)
                break

        if request is None:
            logger.warning(f"Request not found: {request_id}")
            return False

        # Get and update proposal
        proposal = self.proposal_store.get(request["proposal_id"])
        if proposal is None:
            logger.error(f"Proposal not found: {request['proposal_id']}")
            return False

        proposal.reject(rejector, reason)
        self.proposal_store.save(proposal)

        # Move to rejected
        rejected = self._load_json(self.rejected_path)
        rejected.append({
            **request,
            "rejected_by": rejector,
            "rejected_at": datetime.utcnow().isoformat(),
            "reason": reason,
        })
        self._save_json(self.rejected_path, rejected)
        self._save_json(self.pending_path, pending)

        # Audit log
        self._log_audit("REJECTED", {
            "request_id": request_id,
            "proposal_id": request["proposal_id"],
            "rejector": rejector,
            "reason": reason,
        })

        logger.info(f"REJECTED by {rejector}: {request_id} - {reason}")
        return True

    def get_pending(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests."""
        return self._load_json(self.pending_path)

    def get_approved(self) -> List[Dict[str, Any]]:
        """Get all approved but not yet implemented changes."""
        return self._load_json(self.approved_path)

    def get_rejected(self) -> List[Dict[str, Any]]:
        """Get all rejected proposals."""
        return self._load_json(self.rejected_path)

    def can_implement(self, request_id: str) -> tuple[bool, str]:
        """
        Check if an approved change can be implemented.

        Returns (can_implement, reason).
        """
        # Check APPROVE_LIVE_ACTION flag
        if not APPROVE_LIVE_ACTION:
            return False, "APPROVE_LIVE_ACTION is False. Manual intervention required."

        # Check if request is approved
        approved = self._load_json(self.approved_path)
        for a in approved:
            if a.get("request_id") == request_id:
                return True, "Ready for implementation"

        return False, f"Request {request_id} not found in approved list"

    def implement(self, request_id: str, implementer: str, notes: str = "") -> bool:
        """
        Mark an approved change as implemented.

        CRITICAL: This still requires APPROVE_LIVE_ACTION = True.
        Even after approval, the safety flag must be manually enabled.
        """
        can_impl, reason = self.can_implement(request_id)
        if not can_impl:
            raise SafetyError(reason)

        # Find in approved
        approved = self._load_json(self.approved_path)
        request = None
        for i, a in enumerate(approved):
            if a.get("request_id") == request_id:
                request = a
                approved.pop(i)
                break

        if request is None:
            return False

        # Update proposal
        proposal = self.proposal_store.get(request["proposal_id"])
        if proposal:
            proposal.mark_implemented(implementer, notes)
            self.proposal_store.save(proposal)

        # Save to implemented (stored in approved with implemented flag)
        implemented_path = self.state_dir / "implemented_changes.json"
        implemented = self._load_json(implemented_path)
        implemented.append({
            **request,
            "implemented_by": implementer,
            "implemented_at": datetime.utcnow().isoformat(),
            "notes": notes,
        })
        self._save_json(implemented_path, implemented)
        self._save_json(self.approved_path, approved)

        # Audit log
        self._log_audit("IMPLEMENTED", {
            "request_id": request_id,
            "proposal_id": request["proposal_id"],
            "implementer": implementer,
            "notes": notes,
        })

        logger.info(f"IMPLEMENTED by {implementer}: {request_id}")
        return True

    def summary(self) -> str:
        """Generate summary of approval gate status."""
        pending = len(self.get_pending())
        approved = len(self.get_approved())
        rejected = len(self.get_rejected())

        return (
            f"=== Approval Gate Status ===\n"
            f"APPROVE_LIVE_ACTION: {APPROVE_LIVE_ACTION}\n"
            f"Pending approval: {pending}\n"
            f"Approved (awaiting implementation): {approved}\n"
            f"Rejected: {rejected}\n"
        )
