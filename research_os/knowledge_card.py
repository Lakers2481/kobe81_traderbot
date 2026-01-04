"""
Knowledge Card - Standardized discovery format.

Aggregates findings from multiple sources (CuriosityEngine, ResearchEngine, Scrapers)
into a unified format for research and engineering workflows.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any


class DiscoveryType(Enum):
    """Type of discovery."""
    PARAMETER = "parameter"          # Strategy parameter optimization
    PATTERN = "pattern"              # Historical pattern discovery
    STRATEGY = "strategy"            # New strategy concept
    EXTERNAL = "external"            # From external sources (GitHub, Reddit, etc.)
    HYPOTHESIS = "hypothesis"        # From CuriosityEngine hypothesis testing
    EDGE = "edge"                    # Validated trading edge


class CardStatus(Enum):
    """Lifecycle status of a knowledge card."""
    DISCOVERED = "discovered"        # Initial discovery, not yet validated
    TESTING = "testing"              # Under active testing/research
    VALIDATED = "validated"          # Passed integrity checks
    PROPOSED = "proposed"            # Submitted for engineering change
    APPROVED = "approved"            # Human approved for implementation
    REJECTED = "rejected"            # Human rejected
    IMPLEMENTED = "implemented"      # Applied to production


@dataclass
class KnowledgeCard:
    """
    Standardized format for discoveries.

    Aggregates findings from CuriosityEngine, ResearchEngine, and external scrapers
    into a unified format that can flow through the DISCOVER -> RESEARCH -> ENGINEER pipeline.
    """
    # Core identity
    card_id: str = field(default_factory=lambda: f"kc_{uuid.uuid4().hex[:12]}")
    title: str = ""
    description: str = ""
    discovery_type: DiscoveryType = DiscoveryType.PARAMETER

    # Discovery source
    discovered_by: str = ""  # 'curiosity_engine', 'research_engine', 'scraper', 'human'
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    source_hypothesis_id: Optional[str] = None
    source_experiment_id: Optional[str] = None
    source_url: Optional[str] = None  # For external discoveries

    # Evidence
    evidence_summary: str = ""
    sample_size: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    p_value: float = 1.0
    confidence: float = 0.0

    # Detailed metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    # e.g., {'sharpe_ratio': 1.5, 'max_drawdown': 0.15, 'total_trades': 500}

    # Validation status
    integrity_passed: bool = False
    reproducibility_verified: bool = False
    validation_count: int = 0
    validation_notes: List[str] = field(default_factory=list)

    # Status tracking
    status: CardStatus = CardStatus.DISCOVERED
    status_history: List[Dict[str, Any]] = field(default_factory=list)

    # Engineering change proposal
    proposed_change: Optional[str] = None
    target_file: Optional[str] = None
    current_value: Optional[Any] = None
    proposed_value: Optional[Any] = None
    approval_request_id: Optional[str] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Convert enums from strings if needed."""
        if isinstance(self.discovery_type, str):
            self.discovery_type = DiscoveryType(self.discovery_type)
        if isinstance(self.status, str):
            self.status = CardStatus(self.status)

    def update_status(self, new_status: CardStatus, note: str = "") -> None:
        """Update status with history tracking."""
        self.status_history.append({
            "from_status": self.status.value,
            "to_status": new_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "note": note,
        })
        self.status = new_status
        self.updated_at = datetime.utcnow()

    def add_validation(self, passed: bool, notes: str = "") -> None:
        """Record a validation attempt."""
        self.validation_count += 1
        if passed:
            self.integrity_passed = True
        if notes:
            self.validation_notes.append(f"[{datetime.utcnow().isoformat()}] {notes}")
        self.updated_at = datetime.utcnow()

    def is_ready_for_proposal(self) -> bool:
        """Check if card is ready for engineering proposal."""
        return (
            self.status == CardStatus.VALIDATED
            and self.integrity_passed
            and self.reproducibility_verified
            and self.sample_size >= 30
            and self.confidence >= 0.6
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = asdict(self)
        d["discovery_type"] = self.discovery_type.value
        d["status"] = self.status.value
        d["discovered_at"] = self.discovered_at.isoformat()
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeCard":
        """Create from dictionary."""
        # Parse datetime strings
        for field_name in ["discovered_at", "created_at", "updated_at"]:
            if field_name in data and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        return cls(**data)

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"[{self.card_id}] {self.title}\n"
            f"  Type: {self.discovery_type.value} | Status: {self.status.value}\n"
            f"  Evidence: n={self.sample_size}, WR={self.win_rate:.1%}, PF={self.profit_factor:.2f}\n"
            f"  Confidence: {self.confidence:.1%} | p-value: {self.p_value:.4f}\n"
            f"  Validated: {self.integrity_passed} | Reproducible: {self.reproducibility_verified}"
        )


class KnowledgeCardStore:
    """
    Persistent storage for knowledge cards.

    Uses JSON file storage in state/research_os/knowledge_cards.json
    """

    def __init__(self, store_path: Optional[Path] = None):
        if store_path is None:
            store_path = Path(__file__).parent.parent / "state" / "research_os" / "knowledge_cards.json"
        self.store_path = store_path
        self._ensure_store_exists()

    def _ensure_store_exists(self) -> None:
        """Create store file if it doesn't exist."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self.store_path.write_text("[]")

    def _load_all(self) -> List[Dict[str, Any]]:
        """Load all cards from store."""
        try:
            return json.loads(self.store_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_all(self, cards: List[Dict[str, Any]]) -> None:
        """Save all cards to store."""
        self.store_path.write_text(json.dumps(cards, indent=2))

    def save(self, card: KnowledgeCard) -> None:
        """Save or update a knowledge card."""
        cards = self._load_all()
        # Update existing or append new
        for i, existing in enumerate(cards):
            if existing.get("card_id") == card.card_id:
                cards[i] = card.to_dict()
                self._save_all(cards)
                return
        cards.append(card.to_dict())
        self._save_all(cards)

    def get(self, card_id: str) -> Optional[KnowledgeCard]:
        """Get a card by ID."""
        for card_data in self._load_all():
            if card_data.get("card_id") == card_id:
                return KnowledgeCard.from_dict(card_data)
        return None

    def list_all(self) -> List[KnowledgeCard]:
        """List all cards."""
        return [KnowledgeCard.from_dict(d) for d in self._load_all()]

    def list_by_status(self, status: CardStatus) -> List[KnowledgeCard]:
        """List cards by status."""
        return [c for c in self.list_all() if c.status == status]

    def list_validated(self) -> List[KnowledgeCard]:
        """List validated cards ready for proposals."""
        return [c for c in self.list_all() if c.is_ready_for_proposal()]

    def delete(self, card_id: str) -> bool:
        """Delete a card."""
        cards = self._load_all()
        original_count = len(cards)
        cards = [c for c in cards if c.get("card_id") != card_id]
        if len(cards) < original_count:
            self._save_all(cards)
            return True
        return False

    def count(self) -> int:
        """Count total cards."""
        return len(self._load_all())

    def count_by_status(self) -> Dict[str, int]:
        """Count cards by status."""
        counts: Dict[str, int] = {}
        for card in self.list_all():
            status = card.status.value
            counts[status] = counts.get(status, 0) + 1
        return counts
