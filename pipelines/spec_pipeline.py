"""
Spec Pipeline - Convert ideas to strategy specifications.

This pipeline takes IdeaCards from discovery and converts them
to formal StrategySpec objects that can be implemented.

Schedule: On-demand (triggered by discovery)

Author: Kobe Trading System
Version: 1.0.0
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from pipelines.base import Pipeline


@dataclass
class StrategySpec:
    """Formal specification for a trading strategy."""
    spec_id: str
    name: str
    version: str
    description: str

    # Entry conditions
    entry_conditions: List[Dict]  # List of condition dicts

    # Exit conditions
    exit_conditions: Dict  # stop_loss, take_profit, time_stop

    # Filters
    filters: List[Dict]  # Pre-trade filters

    # Position sizing
    position_sizing: Dict  # risk_pct, max_pct, etc.

    # Metadata
    source_idea_id: str
    confidence: float
    status: str = "draft"  # draft, validated, approved, implemented
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


class SpecPipeline(Pipeline):
    """Pipeline for converting ideas to strategy specs."""

    @property
    def name(self) -> str:
        return "spec"

    def execute(self) -> bool:
        """
        Execute spec generation.

        Returns:
            True if specs generated successfully
        """
        self.logger.info("Generating strategy specifications...")

        # Load pending ideas
        ideas = self._load_pending_ideas()
        if not ideas:
            self.logger.info("No pending ideas to convert")
            self.set_metric("specs_generated", 0)
            return True

        specs = []
        for idea in ideas:
            spec = self._convert_idea_to_spec(idea)
            if spec:
                specs.append(spec)

        # Save specs
        self._save_specs(specs)

        self.set_metric("ideas_processed", len(ideas))
        self.set_metric("specs_generated", len(specs))

        self.logger.info(f"Generated {len(specs)} strategy specs")
        return True

    def _load_pending_ideas(self) -> List[Dict]:
        """Load ideas pending conversion to specs."""
        ideas_file = self.state_dir / "discovery" / "ideas.jsonl"
        if not ideas_file.exists():
            return []

        ideas = []
        seen_ids = set()

        # Load already converted ideas
        specs_file = self.state_dir / "specs" / "specs.jsonl"
        if specs_file.exists():
            with open(specs_file) as f:
                for line in f:
                    spec = json.loads(line)
                    seen_ids.add(spec.get("source_idea_id"))

        # Load new ideas
        with open(ideas_file) as f:
            for line in f:
                idea = json.loads(line)
                if idea.get("idea_id") not in seen_ids:
                    if idea.get("confidence", 0) >= 0.5:  # Only high-confidence ideas
                        ideas.append(idea)

        return ideas[:5]  # Process 5 at a time

    def _convert_idea_to_spec(self, idea: Dict) -> Optional[StrategySpec]:
        """Convert an idea to a strategy spec."""
        try:
            discovery_type = idea.get("discovery_type", "unknown")
            evidence = idea.get("evidence", {})

            if discovery_type == "pattern":
                return self._create_pattern_spec(idea, evidence)
            elif discovery_type == "parameter":
                return self._create_parameter_spec(idea, evidence)
            elif discovery_type == "strategy":
                return self._create_strategy_spec(idea, evidence)
            else:
                self.add_warning(f"Unknown discovery type: {discovery_type}")
                return None

        except Exception as e:
            self.add_warning(f"Failed to convert idea {idea.get('idea_id')}: {e}")
            return None

    def _create_pattern_spec(self, idea: Dict, evidence: Dict) -> StrategySpec:
        """Create spec from a pattern idea."""
        symbol = evidence.get("symbol", "UNKNOWN")
        streak = evidence.get("streak", 3)

        return StrategySpec(
            spec_id=f"spec_{idea['idea_id']}",
            name=f"{symbol} Consecutive Down Pattern",
            version="1.0.0",
            description=idea.get("description", ""),
            entry_conditions=[
                {"type": "consecutive_down", "streak": streak},
                {"type": "symbol_filter", "symbols": [symbol]},
            ],
            exit_conditions={
                "stop_loss": {"type": "atr_multiple", "multiplier": 2.0},
                "take_profit": {"type": "atr_multiple", "multiplier": 3.0},
                "time_stop": {"type": "bars", "max_bars": 5},
            },
            filters=[
                {"type": "sma_filter", "period": 200, "condition": "above"},
            ],
            position_sizing={
                "risk_pct": 0.02,
                "max_pct": 0.10,
            },
            source_idea_id=idea["idea_id"],
            confidence=idea.get("confidence", 0.5),
        )

    def _create_parameter_spec(self, idea: Dict, evidence: Dict) -> StrategySpec:
        """Create spec from a parameter improvement."""
        return StrategySpec(
            spec_id=f"spec_{idea['idea_id']}",
            name=f"Parameter Variant: {evidence.get('name', 'Unknown')}",
            version="1.0.0",
            description=idea.get("description", ""),
            entry_conditions=evidence.get("entry_conditions", []),
            exit_conditions=evidence.get("exit_conditions", {}),
            filters=evidence.get("filters", []),
            position_sizing={
                "risk_pct": 0.02,
                "max_pct": 0.10,
            },
            source_idea_id=idea["idea_id"],
            confidence=idea.get("confidence", 0.5),
        )

    def _create_strategy_spec(self, idea: Dict, evidence: Dict) -> StrategySpec:
        """Create spec from a strategy idea."""
        return StrategySpec(
            spec_id=f"spec_{idea['idea_id']}",
            name=idea.get("title", "External Strategy"),
            version="1.0.0",
            description=idea.get("description", ""),
            entry_conditions=[
                {"type": "external", "source": idea.get("source", "unknown")},
            ],
            exit_conditions={
                "stop_loss": {"type": "atr_multiple", "multiplier": 2.0},
                "take_profit": {"type": "atr_multiple", "multiplier": 3.0},
                "time_stop": {"type": "bars", "max_bars": 7},
            },
            filters=[],
            position_sizing={
                "risk_pct": 0.02,
                "max_pct": 0.10,
            },
            source_idea_id=idea["idea_id"],
            confidence=idea.get("confidence", 0.3),
        )

    def _save_specs(self, specs: List[StrategySpec]):
        """Save generated specs."""
        if not specs:
            return

        specs_dir = self.state_dir / "specs"
        specs_dir.mkdir(parents=True, exist_ok=True)

        specs_file = specs_dir / "specs.jsonl"
        with open(specs_file, "a") as f:
            for spec in specs:
                f.write(json.dumps(asdict(spec)) + "\n")

        self.add_artifact(str(specs_file))
