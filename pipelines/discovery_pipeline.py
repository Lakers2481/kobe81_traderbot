"""
Discovery Pipeline - Scout for new strategy ideas.

This pipeline discovers trading ideas from multiple sources:
- Internal pattern analysis
- Parameter sweep experiments
- External sources (GitHub, Reddit, arXiv)

Schedule: Hourly (with 2-hour cooldown)

Author: Kobe Trading System
Version: 1.0.0
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from pipelines.base import Pipeline


@dataclass
class IdeaCard:
    """Standardized idea format for the discovery pipeline."""
    idea_id: str
    source: str  # internal_pattern, parameter_sweep, github, reddit, arxiv
    discovery_type: str  # pattern, parameter, strategy, edge
    title: str
    description: str
    evidence: Dict
    confidence: float  # 0.0 to 1.0
    status: str = "discovered"  # discovered, validated, proposed, approved, implemented
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


class DiscoveryPipeline(Pipeline):
    """Pipeline for discovering new trading ideas."""

    @property
    def name(self) -> str:
        return "discovery"

    def execute(self) -> bool:
        """
        Execute discovery process.

        Returns:
            True (discovery never fails, just may find nothing)
        """
        self.logger.info("Running discovery pipeline...")

        ideas = []

        # Source 1: Internal pattern analysis
        pattern_ideas = self._discover_patterns()
        ideas.extend(pattern_ideas)

        # Source 2: Parameter sweep
        param_ideas = self._discover_parameters()
        ideas.extend(param_ideas)

        # Source 3: External sources
        external_ideas = self._discover_external()
        ideas.extend(external_ideas)

        # Save discovered ideas
        self._save_ideas(ideas)

        self.set_metric("total_ideas", len(ideas))
        self.set_metric("pattern_ideas", len(pattern_ideas))
        self.set_metric("parameter_ideas", len(param_ideas))
        self.set_metric("external_ideas", len(external_ideas))

        self.logger.info(f"Discovered {len(ideas)} ideas")
        return True

    def _discover_patterns(self) -> List[IdeaCard]:
        """Discover ideas from internal pattern analysis."""
        ideas = []

        # Check for existing pattern analysis
        reports_dir = self.project_root / "reports"
        pattern_files = list(reports_dir.glob("quant_pattern_analysis_*.json"))

        if pattern_files:
            latest = max(pattern_files, key=lambda p: p.stat().st_mtime)
            try:
                patterns = json.loads(latest.read_text())
                top_patterns = patterns.get("top_50_patterns", [])[:5]

                for p in top_patterns:
                    if p.get("bounce_5d", 0) >= 80:  # High-quality patterns only
                        idea = IdeaCard(
                            idea_id=f"pattern_{p['symbol']}_{datetime.utcnow().strftime('%Y%m%d')}",
                            source="internal_pattern",
                            discovery_type="pattern",
                            title=f"{p['symbol']} Consecutive Down Pattern",
                            description=f"{p['streak']}-day down streak with {p['bounce_5d']:.0f}% bounce rate",
                            evidence={
                                "symbol": p["symbol"],
                                "streak": p["streak"],
                                "bounce_rate": p["bounce_5d"],
                                "avg_bounce": p["avg_5d"],
                                "sample_size": p.get("count", 0),
                            },
                            confidence=min(p["bounce_5d"] / 100, 0.95),
                        )
                        ideas.append(idea)
            except Exception as e:
                self.add_warning(f"Failed to parse pattern file: {e}")

        return ideas

    def _discover_parameters(self) -> List[IdeaCard]:
        """Discover ideas from parameter experiments."""
        ideas = []

        # Check research state for successful experiments
        research_file = self.state_dir / "autonomous" / "research" / "research_state.json"
        if research_file.exists():
            try:
                research = json.loads(research_file.read_text())
                experiments = research.get("successful_experiments", [])

                for exp in experiments[:3]:  # Top 3 experiments
                    if exp.get("improvement", 0) >= 0.05:  # 5%+ improvement
                        idea = IdeaCard(
                            idea_id=f"param_{exp.get('id', 'unknown')}",
                            source="parameter_sweep",
                            discovery_type="parameter",
                            title=f"Parameter Improvement: {exp.get('name', 'Unknown')}",
                            description=f"+{exp.get('improvement', 0):.1%} improvement over baseline",
                            evidence=exp,
                            confidence=min(0.5 + exp.get("improvement", 0), 0.9),
                        )
                        ideas.append(idea)
            except Exception as e:
                self.add_warning(f"Failed to parse research state: {e}")

        return ideas

    def _discover_external(self) -> List[IdeaCard]:
        """Discover ideas from external sources."""
        ideas = []

        # Check external ideas queue
        external_file = self.state_dir / "autonomous" / "research" / "external_ideas.json"
        if external_file.exists():
            try:
                external = json.loads(external_file.read_text())
                for item in external.get("ideas", [])[:5]:
                    idea = IdeaCard(
                        idea_id=f"ext_{item.get('source', 'unknown')}_{datetime.utcnow().strftime('%Y%m%d%H%M')}",
                        source=item.get("source", "external"),
                        discovery_type="strategy",
                        title=item.get("title", "External Idea"),
                        description=item.get("description", ""),
                        evidence=item,
                        confidence=0.3,  # External ideas start low confidence
                    )
                    ideas.append(idea)
            except Exception as e:
                self.add_warning(f"Failed to parse external ideas: {e}")

        return ideas

    def _save_ideas(self, ideas: List[IdeaCard]):
        """Save discovered ideas to state."""
        if not ideas:
            return

        ideas_file = self.state_dir / "discovery" / "ideas.jsonl"
        ideas_file.parent.mkdir(parents=True, exist_ok=True)

        with open(ideas_file, "a") as f:
            for idea in ideas:
                f.write(json.dumps(asdict(idea)) + "\n")

        self.add_artifact(str(ideas_file))
