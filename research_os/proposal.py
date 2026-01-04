"""
Research Proposal - Formal workflow for engineering changes.

Requires human approval before any changes can be applied to production.
NEVER auto-merges. NEVER enables live trading without explicit approval.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any


class ChangeType(Enum):
    """Type of engineering change."""
    PARAMETER_UPDATE = "parameter_update"      # Update frozen strategy params
    NEW_RULE = "new_rule"                      # Add semantic memory rule
    STRATEGY_MODIFICATION = "strategy_mod"     # Modify strategy logic
    FILTER_ADDITION = "filter_addition"        # Add new filter/gate
    RISK_ADJUSTMENT = "risk_adjustment"        # Modify risk parameters


class ProposalStatus(Enum):
    """Lifecycle status of a proposal."""
    DRAFT = "draft"              # Being prepared
    SUBMITTED = "submitted"      # Awaiting human review
    UNDER_REVIEW = "under_review"  # Human is actively reviewing
    APPROVED = "approved"        # Human approved
    REJECTED = "rejected"        # Human rejected
    IMPLEMENTED = "implemented"  # Applied to production
    WITHDRAWN = "withdrawn"      # Proposer withdrew


@dataclass
class ResearchProposal:
    """
    Formal proposal for engineering change.

    Requires explicit human approval before implementation.
    This is the gate between Research and Engineering lanes.
    """
    # Core identity
    proposal_id: str = field(default_factory=lambda: f"rp_{uuid.uuid4().hex[:12]}")
    knowledge_card_id: str = ""
    title: str = ""

    # What to change
    change_type: ChangeType = ChangeType.PARAMETER_UPDATE
    target_file: str = ""
    target_key: str = ""  # e.g., "ts_min_sweep_strength" for param updates
    current_value: Any = None
    proposed_value: Any = None

    # Justification
    hypothesis: str = ""
    evidence_summary: str = ""
    expected_improvement: str = ""
    risk_assessment: str = ""
    rollback_plan: str = ""

    # Evidence metrics
    sample_size: int = 0
    win_rate_before: float = 0.0
    win_rate_after: float = 0.0
    profit_factor_before: float = 0.0
    profit_factor_after: float = 0.0
    improvement_percentage: float = 0.0

    # Approval workflow
    status: ProposalStatus = ProposalStatus.DRAFT
    submitted_at: Optional[datetime] = None
    submitted_by: str = ""
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    implementation_notes: Optional[str] = None

    # Audit trail
    status_history: List[Dict[str, Any]] = field(default_factory=list)
    comments: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    priority: int = 0  # 0=low, 1=medium, 2=high
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Convert enums from strings if needed."""
        if isinstance(self.change_type, str):
            self.change_type = ChangeType(self.change_type)
        if isinstance(self.status, str):
            self.status = ProposalStatus(self.status)

    def submit(self, submitter: str) -> None:
        """Submit proposal for human review."""
        if self.status != ProposalStatus.DRAFT:
            raise ValueError(f"Cannot submit proposal in status: {self.status.value}")
        self.update_status(ProposalStatus.SUBMITTED, f"Submitted by {submitter}")
        self.submitted_at = datetime.utcnow()
        self.submitted_by = submitter

    def start_review(self, reviewer: str) -> None:
        """Mark proposal as under review."""
        if self.status != ProposalStatus.SUBMITTED:
            raise ValueError(f"Cannot start review in status: {self.status.value}")
        self.update_status(ProposalStatus.UNDER_REVIEW, f"Review started by {reviewer}")
        self.reviewed_by = reviewer
        self.reviewed_at = datetime.utcnow()

    def approve(self, approver: str, notes: str = "") -> None:
        """
        Human approves the proposal.

        This is the critical gate - only humans can call this.
        """
        if self.status not in [ProposalStatus.SUBMITTED, ProposalStatus.UNDER_REVIEW]:
            raise ValueError(f"Cannot approve proposal in status: {self.status.value}")
        self.update_status(ProposalStatus.APPROVED, f"Approved by {approver}: {notes}")
        self.approved_by = approver
        self.approved_at = datetime.utcnow()
        if notes:
            self.add_comment(approver, f"Approval notes: {notes}")

    def reject(self, rejector: str, reason: str) -> None:
        """Human rejects the proposal."""
        if self.status not in [ProposalStatus.SUBMITTED, ProposalStatus.UNDER_REVIEW]:
            raise ValueError(f"Cannot reject proposal in status: {self.status.value}")
        self.update_status(ProposalStatus.REJECTED, f"Rejected by {rejector}: {reason}")
        self.rejection_reason = reason
        self.add_comment(rejector, f"Rejection reason: {reason}")

    def withdraw(self, withdrawer: str, reason: str = "") -> None:
        """Proposer withdraws the proposal."""
        if self.status in [ProposalStatus.IMPLEMENTED, ProposalStatus.REJECTED]:
            raise ValueError(f"Cannot withdraw proposal in status: {self.status.value}")
        self.update_status(ProposalStatus.WITHDRAWN, f"Withdrawn by {withdrawer}: {reason}")

    def mark_implemented(self, implementer: str, notes: str = "") -> None:
        """Mark proposal as implemented after changes applied."""
        if self.status != ProposalStatus.APPROVED:
            raise ValueError(f"Cannot implement proposal in status: {self.status.value}")
        self.update_status(ProposalStatus.IMPLEMENTED, f"Implemented by {implementer}: {notes}")
        self.implementation_notes = notes

    def update_status(self, new_status: ProposalStatus, note: str = "") -> None:
        """Update status with history tracking."""
        self.status_history.append({
            "from_status": self.status.value,
            "to_status": new_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "note": note,
        })
        self.status = new_status
        self.updated_at = datetime.utcnow()

    def add_comment(self, author: str, comment: str) -> None:
        """Add a comment to the proposal."""
        self.comments.append({
            "author": author,
            "comment": comment,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.updated_at = datetime.utcnow()

    def is_pending_approval(self) -> bool:
        """Check if proposal is awaiting human approval."""
        return self.status in [ProposalStatus.SUBMITTED, ProposalStatus.UNDER_REVIEW]

    def is_approved(self) -> bool:
        """Check if proposal has been approved."""
        return self.status == ProposalStatus.APPROVED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = asdict(self)
        d["change_type"] = self.change_type.value
        d["status"] = self.status.value
        for field_name in ["submitted_at", "reviewed_at", "approved_at", "created_at", "updated_at"]:
            if d.get(field_name) is not None:
                d[field_name] = d[field_name].isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchProposal":
        """Create from dictionary."""
        # Parse datetime strings
        for field_name in ["submitted_at", "reviewed_at", "approved_at", "created_at", "updated_at"]:
            if data.get(field_name) and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        return cls(**data)

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"[{self.proposal_id}] {self.title}\n"
            f"  Change: {self.change_type.value} in {self.target_file}\n"
            f"  Status: {self.status.value}\n"
            f"  Evidence: n={self.sample_size}, WR: {self.win_rate_before:.1%} -> {self.win_rate_after:.1%}\n"
            f"  Expected improvement: {self.expected_improvement}\n"
            f"  Risk: {self.risk_assessment[:100]}..."
        )


class ProposalStore:
    """
    Persistent storage for research proposals.

    Uses JSON file storage in state/research_os/proposals.json
    """

    def __init__(self, store_path: Optional[Path] = None):
        if store_path is None:
            store_path = Path(__file__).parent.parent / "state" / "research_os" / "proposals.json"
        self.store_path = store_path
        self._ensure_store_exists()

    def _ensure_store_exists(self) -> None:
        """Create store file if it doesn't exist."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self.store_path.write_text("[]")

    def _load_all(self) -> List[Dict[str, Any]]:
        """Load all proposals from store."""
        try:
            return json.loads(self.store_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_all(self, proposals: List[Dict[str, Any]]) -> None:
        """Save all proposals to store."""
        self.store_path.write_text(json.dumps(proposals, indent=2))

    def save(self, proposal: ResearchProposal) -> None:
        """Save or update a proposal."""
        proposals = self._load_all()
        for i, existing in enumerate(proposals):
            if existing.get("proposal_id") == proposal.proposal_id:
                proposals[i] = proposal.to_dict()
                self._save_all(proposals)
                return
        proposals.append(proposal.to_dict())
        self._save_all(proposals)

    def get(self, proposal_id: str) -> Optional[ResearchProposal]:
        """Get a proposal by ID."""
        for data in self._load_all():
            if data.get("proposal_id") == proposal_id:
                return ResearchProposal.from_dict(data)
        return None

    def list_all(self) -> List[ResearchProposal]:
        """List all proposals."""
        return [ResearchProposal.from_dict(d) for d in self._load_all()]

    def list_pending(self) -> List[ResearchProposal]:
        """List proposals pending human approval."""
        return [p for p in self.list_all() if p.is_pending_approval()]

    def list_approved(self) -> List[ResearchProposal]:
        """List approved proposals awaiting implementation."""
        return [p for p in self.list_all() if p.status == ProposalStatus.APPROVED]

    def list_by_status(self, status: ProposalStatus) -> List[ResearchProposal]:
        """List proposals by status."""
        return [p for p in self.list_all() if p.status == status]

    def delete(self, proposal_id: str) -> bool:
        """Delete a proposal."""
        proposals = self._load_all()
        original_count = len(proposals)
        proposals = [p for p in proposals if p.get("proposal_id") != proposal_id]
        if len(proposals) < original_count:
            self._save_all(proposals)
            return True
        return False

    def count_by_status(self) -> Dict[str, int]:
        """Count proposals by status."""
        counts: Dict[str, int] = {}
        for proposal in self.list_all():
            status = proposal.status.value
            counts[status] = counts.get(status, 0) + 1
        return counts
