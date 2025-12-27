"""
Strategy candidate registry for evolution tracking.

Tracks all strategy candidates, their results, and promotion decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import json
import uuid


class CandidateDecision(Enum):
    """Possible decisions for strategy candidates."""
    PENDING = auto()
    PROMOTED = auto()
    REJECTED = auto()
    RETIRED = auto()


@dataclass
class StrategyCandidate:
    """A strategy candidate being evaluated for promotion."""
    candidate_id: str
    strategy_name: str
    params: Dict[str, Any]
    parent_name: Optional[str]
    parent_id: Optional[str]
    dataset_hash: str
    created_at: str
    results: Dict[str, Any] = field(default_factory=dict)
    decision: CandidateDecision = CandidateDecision.PENDING
    decision_reason: str = ""
    decision_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def params_hash(self) -> str:
        """Get hash of strategy parameters."""
        params_str = json.dumps(self.params, sort_keys=True, default=str)
        return hashlib.sha256(params_str.encode()).hexdigest()[:16]

    @property
    def fingerprint(self) -> str:
        """Get fingerprint combining strategy name and params."""
        return f"{self.strategy_name}:{self.params_hash}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.name
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StrategyCandidate:
        data = data.copy()
        data["decision"] = CandidateDecision[data["decision"]]
        return cls(**data)


@dataclass
class EvolutionRun:
    """A single evolution/optimization run."""
    run_id: str
    started_at: str
    ended_at: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    candidates_evaluated: int = 0
    candidates_promoted: int = 0
    candidates_rejected: int = 0
    best_candidate_id: Optional[str] = None
    notes: str = ""


class EvolutionRegistry:
    """
    Registry for tracking strategy evolution.

    Maintains records of:
    - All strategy candidates evaluated
    - Promotion/rejection decisions with reasons
    - Lineage (which candidates spawned from which)
    - Evolution runs
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or Path("state/evolution_registry.jsonl")
        self.runs_path = self.registry_path.parent / "evolution_runs.jsonl"
        self._candidates: Dict[str, StrategyCandidate] = {}
        self._runs: Dict[str, EvolutionRun] = {}
        self._fingerprint_index: Dict[str, str] = {}  # fingerprint -> candidate_id
        self._load()

    def _ensure_path(self) -> None:
        """Ensure registry directory exists."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        """Load existing registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            candidate = StrategyCandidate.from_dict(data)
                            self._candidates[candidate.candidate_id] = candidate
                            self._fingerprint_index[candidate.fingerprint] = candidate.candidate_id
            except Exception:
                pass

        if self.runs_path.exists():
            try:
                with open(self.runs_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            run = EvolutionRun(**data)
                            self._runs[run.run_id] = run
            except Exception:
                pass

    def _save_candidate(self, candidate: StrategyCandidate) -> None:
        """Append candidate to registry file."""
        self._ensure_path()
        with open(self.registry_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(candidate.to_dict()) + "\n")

    def _save_run(self, run: EvolutionRun) -> None:
        """Append run to runs file."""
        self._ensure_path()
        with open(self.runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(run)) + "\n")

    def register_candidate(
        self,
        strategy_name: str,
        params: Dict[str, Any],
        dataset_hash: str,
        parent_name: Optional[str] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a new strategy candidate.

        Returns candidate_id.
        """
        candidate_id = f"cand_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        candidate = StrategyCandidate(
            candidate_id=candidate_id,
            strategy_name=strategy_name,
            params=params,
            parent_name=parent_name,
            parent_id=parent_id,
            dataset_hash=dataset_hash,
            created_at=datetime.utcnow().isoformat() + "Z",
            metadata=metadata or {},
        )

        self._candidates[candidate_id] = candidate
        self._fingerprint_index[candidate.fingerprint] = candidate_id
        self._save_candidate(candidate)

        return candidate_id

    def update_results(
        self,
        candidate_id: str,
        results: Dict[str, Any],
    ) -> None:
        """Update candidate results."""
        if candidate_id not in self._candidates:
            raise KeyError(f"Candidate {candidate_id} not found")

        self._candidates[candidate_id].results = results
        # Re-save entire registry for updates (not ideal for large registries)
        self._rewrite_registry()

    def mark_decision(
        self,
        candidate_id: str,
        decision: CandidateDecision,
        reason: str,
    ) -> None:
        """Mark candidate with promotion/rejection decision."""
        if candidate_id not in self._candidates:
            raise KeyError(f"Candidate {candidate_id} not found")

        candidate = self._candidates[candidate_id]
        candidate.decision = decision
        candidate.decision_reason = reason
        candidate.decision_at = datetime.utcnow().isoformat() + "Z"
        self._rewrite_registry()

    def _rewrite_registry(self) -> None:
        """Rewrite entire registry (for updates)."""
        self._ensure_path()
        with open(self.registry_path, "w", encoding="utf-8") as f:
            for candidate in self._candidates.values():
                f.write(json.dumps(candidate.to_dict()) + "\n")

    def get_candidate(self, candidate_id: str) -> Optional[StrategyCandidate]:
        """Get candidate by ID."""
        return self._candidates.get(candidate_id)

    def get_candidates(
        self,
        status: Optional[CandidateDecision] = None,
        strategy_name: Optional[str] = None,
    ) -> List[StrategyCandidate]:
        """Get candidates with optional filters."""
        candidates = list(self._candidates.values())

        if status is not None:
            candidates = [c for c in candidates if c.decision == status]

        if strategy_name is not None:
            candidates = [c for c in candidates if c.strategy_name == strategy_name]

        return sorted(candidates, key=lambda c: c.created_at, reverse=True)

    def get_promoted(self) -> List[StrategyCandidate]:
        """Get all promoted candidates."""
        return self.get_candidates(status=CandidateDecision.PROMOTED)

    def get_rejected(self) -> List[StrategyCandidate]:
        """Get all rejected candidates."""
        return self.get_candidates(status=CandidateDecision.REJECTED)

    def get_pending(self) -> List[StrategyCandidate]:
        """Get all pending candidates."""
        return self.get_candidates(status=CandidateDecision.PENDING)

    def find_by_fingerprint(self, fingerprint: str) -> Optional[StrategyCandidate]:
        """Find candidate by fingerprint."""
        candidate_id = self._fingerprint_index.get(fingerprint)
        if candidate_id:
            return self._candidates.get(candidate_id)
        return None

    def has_fingerprint(self, fingerprint: str) -> bool:
        """Check if fingerprint already exists."""
        return fingerprint in self._fingerprint_index

    def get_lineage(self, candidate_id: str) -> List[StrategyCandidate]:
        """Get lineage chain (ancestors) for a candidate."""
        lineage = []
        current = self.get_candidate(candidate_id)

        while current and current.parent_id:
            parent = self.get_candidate(current.parent_id)
            if parent:
                lineage.append(parent)
                current = parent
            else:
                break

        return lineage

    def start_run(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new evolution run."""
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        run = EvolutionRun(
            run_id=run_id,
            started_at=datetime.utcnow().isoformat() + "Z",
            config=config or {},
        )

        self._runs[run_id] = run
        self._save_run(run)

        return run_id

    def end_run(
        self,
        run_id: str,
        candidates_evaluated: int = 0,
        candidates_promoted: int = 0,
        candidates_rejected: int = 0,
        best_candidate_id: Optional[str] = None,
        notes: str = "",
    ) -> None:
        """End an evolution run."""
        if run_id not in self._runs:
            return

        run = self._runs[run_id]
        run.ended_at = datetime.utcnow().isoformat() + "Z"
        run.candidates_evaluated = candidates_evaluated
        run.candidates_promoted = candidates_promoted
        run.candidates_rejected = candidates_rejected
        run.best_candidate_id = best_candidate_id
        run.notes = notes

        # Rewrite runs file
        with open(self.runs_path, "w", encoding="utf-8") as f:
            for r in self._runs.values():
                f.write(json.dumps(asdict(r)) + "\n")

    def get_run(self, run_id: str) -> Optional[EvolutionRun]:
        """Get run by ID."""
        return self._runs.get(run_id)

    def get_recent_runs(self, limit: int = 10) -> List[EvolutionRun]:
        """Get most recent runs."""
        runs = list(self._runs.values())
        runs.sort(key=lambda r: r.started_at, reverse=True)
        return runs[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        candidates = list(self._candidates.values())

        return {
            "total_candidates": len(candidates),
            "pending": sum(1 for c in candidates if c.decision == CandidateDecision.PENDING),
            "promoted": sum(1 for c in candidates if c.decision == CandidateDecision.PROMOTED),
            "rejected": sum(1 for c in candidates if c.decision == CandidateDecision.REJECTED),
            "retired": sum(1 for c in candidates if c.decision == CandidateDecision.RETIRED),
            "unique_strategies": len(set(c.strategy_name for c in candidates)),
            "total_runs": len(self._runs),
        }
