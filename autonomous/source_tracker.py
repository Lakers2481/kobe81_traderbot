"""
Source Credibility Tracker.

Tracks which external sources (GitHub repos, Reddit posts, papers)
provide ideas that validate successfully.

This allows the brain to LEARN which sources are trustworthy over time
and prioritize high-quality sources in future research.
"""

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json

from core.structured_log import jlog


@dataclass
class ResearchSource:
    """
    A research source with credibility tracking.

    Tracks how many ideas from this source have been validated
    successfully vs failed validation.
    """
    source_id: str           # e.g., "github:user/repo" or "reddit:algotrading"
    source_type: str         # github, reddit, youtube, arxiv
    source_url: str          # Link to source

    # Idea tracking
    ideas_extracted: int = 0     # Total ideas from this source
    ideas_validated: int = 0     # Ideas that passed backtest (WR > 55%, PF > 1.3)
    ideas_failed: int = 0        # Ideas that failed validation

    # Performance metrics
    best_win_rate: float = 0.0
    best_profit_factor: float = 0.0

    # Credibility score (0-1)
    credibility_score: float = 0.5  # Start neutral

    # Timestamps
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def validation_rate(self) -> float:
        """Percentage of ideas that validated successfully."""
        total = self.ideas_validated + self.ideas_failed
        if total == 0:
            return 0.0
        return self.ideas_validated / total

    def update_credibility(self):
        """Recalculate credibility score based on validation history."""
        total = self.ideas_validated + self.ideas_failed

        if total < 3:
            # Not enough data, stay neutral
            self.credibility_score = 0.5
        else:
            # Base score on validation rate
            base_score = self.validation_rate

            # Bonus for high-quality discoveries
            if self.best_profit_factor > 1.5:
                base_score += 0.1
            if self.best_win_rate > 0.60:
                base_score += 0.1

            # Clip to [0, 1]
            self.credibility_score = max(0.0, min(1.0, base_score))

        self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ResearchSource":
        """Create from dictionary."""
        return cls(**data)


class SourceTracker:
    """
    Tracks and scores external research sources.

    Key responsibilities:
    - Record ideas from each source
    - Track validation success/failure
    - Calculate credibility scores
    - Prioritize high-quality sources
    """

    def __init__(self, state_dir: Optional[Path] = None):
        self.state_dir = state_dir or Path("state/autonomous/source_tracking")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.state_dir / "source_credibility.json"

        # Load existing sources
        self.sources: Dict[str, ResearchSource] = self._load_sources()

    def _load_sources(self) -> Dict[str, ResearchSource]:
        """Load sources from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                return {
                    sid: ResearchSource.from_dict(s)
                    for sid, s in data.items()
                }
            except Exception as e:
                jlog("source_tracker_load_error", level="WARNING", error=str(e))
        return {}

    def _save_sources(self):
        """Save sources to disk."""
        with open(self.state_file, "w") as f:
            json.dump(
                {sid: s.to_dict() for sid, s in self.sources.items()},
                f, indent=2
            )

    def record_idea(self, source_id: str, source_type: str, source_url: str):
        """
        Record that an idea was extracted from a source.

        Args:
            source_id: Unique source identifier
            source_type: github, reddit, youtube, arxiv
            source_url: URL to the source
        """
        if source_id not in self.sources:
            self.sources[source_id] = ResearchSource(
                source_id=source_id,
                source_type=source_type,
                source_url=source_url
            )

        self.sources[source_id].ideas_extracted += 1
        self.sources[source_id].last_updated = datetime.now().isoformat()
        self._save_sources()

        jlog("source_tracker_idea_recorded", level="DEBUG",
             source_id=source_id, total_ideas=self.sources[source_id].ideas_extracted)

    def record_validation(
        self,
        source_id: str,
        success: bool,
        win_rate: Optional[float] = None,
        profit_factor: Optional[float] = None
    ):
        """
        Record validation result for a source's idea.

        Args:
            source_id: Unique source identifier
            success: Whether validation passed (WR > 55%, PF > 1.3)
            win_rate: Achieved win rate (if tested)
            profit_factor: Achieved profit factor (if tested)
        """
        if source_id not in self.sources:
            jlog("source_tracker_unknown_source", level="WARNING",
                 source_id=source_id)
            return

        source = self.sources[source_id]

        if success:
            source.ideas_validated += 1
            if win_rate and win_rate > source.best_win_rate:
                source.best_win_rate = win_rate
            if profit_factor and profit_factor > source.best_profit_factor:
                source.best_profit_factor = profit_factor

            jlog("source_tracker_validation_success", level="INFO",
                 source_id=source_id,
                 win_rate=win_rate,
                 profit_factor=profit_factor)
        else:
            source.ideas_failed += 1
            jlog("source_tracker_validation_failed", level="DEBUG",
                 source_id=source_id)

        # Update credibility
        source.update_credibility()
        self._save_sources()

    def get_source(self, source_id: str) -> Optional[ResearchSource]:
        """Get source by ID."""
        return self.sources.get(source_id)

    def get_trusted_sources(self, min_credibility: float = 0.6) -> List[ResearchSource]:
        """
        Get sources with credibility above threshold.

        Args:
            min_credibility: Minimum credibility score (0-1)

        Returns:
            List of trusted sources, sorted by credibility
        """
        trusted = [
            s for s in self.sources.values()
            if s.credibility_score >= min_credibility
        ]

        return sorted(trusted, key=lambda s: s.credibility_score, reverse=True)

    def get_source_priority(self) -> List[str]:
        """
        Get ordered list of source IDs by priority.

        Higher priority = higher credibility + more validated ideas.

        Returns:
            List of source IDs in priority order
        """
        # Score = credibility * log(1 + validated_ideas)
        import math

        scored = []
        for source in self.sources.values():
            score = source.credibility_score * math.log(1 + source.ideas_validated + 1)
            scored.append((source.source_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in scored]

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        total_ideas = sum(s.ideas_extracted for s in self.sources.values())
        validated_ideas = sum(s.ideas_validated for s in self.sources.values())
        failed_ideas = sum(s.ideas_failed for s in self.sources.values())

        by_type = {}
        for source in self.sources.values():
            t = source.source_type
            if t not in by_type:
                by_type[t] = {"sources": 0, "ideas": 0, "validated": 0}
            by_type[t]["sources"] += 1
            by_type[t]["ideas"] += source.ideas_extracted
            by_type[t]["validated"] += source.ideas_validated

        return {
            "total_sources": len(self.sources),
            "total_ideas": total_ideas,
            "validated_ideas": validated_ideas,
            "failed_ideas": failed_ideas,
            "validation_rate": validated_ideas / max(1, validated_ideas + failed_ideas),
            "by_type": by_type,
            "trusted_sources": len(self.get_trusted_sources(0.6)),
            "top_sources": self.get_source_priority()[:5]
        }

    def get_report(self) -> str:
        """Generate a human-readable report."""
        stats = self.get_statistics()

        lines = [
            "=" * 50,
            "SOURCE CREDIBILITY REPORT",
            "=" * 50,
            f"Total Sources: {stats['total_sources']}",
            f"Total Ideas: {stats['total_ideas']}",
            f"Validated: {stats['validated_ideas']} ({stats['validation_rate']:.1%})",
            f"Failed: {stats['failed_ideas']}",
            "",
            "BY SOURCE TYPE:",
        ]

        for source_type, data in stats["by_type"].items():
            rate = data["validated"] / max(1, data["ideas"])
            lines.append(
                f"  {source_type}: {data['sources']} sources, "
                f"{data['ideas']} ideas, {data['validated']} validated ({rate:.1%})"
            )

        lines.extend([
            "",
            "TOP TRUSTED SOURCES:",
        ])

        for source in self.get_trusted_sources(0.5)[:5]:
            lines.append(
                f"  {source.source_id}: {source.credibility_score:.2f} credibility, "
                f"{source.ideas_validated} validated"
            )

        lines.append("=" * 50)

        return "\n".join(lines)
