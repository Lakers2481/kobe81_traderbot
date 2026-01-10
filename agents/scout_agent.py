"""
Scout Agent - External Source Discovery
========================================

Discovers new strategy ideas from external sources:
- arXiv papers
- GitHub repositories
- Financial blogs
- YouTube tutorials
- Reddit discussions

Outputs IdeaCards for Strategist processing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Tuple

from llm import ToolDefinition
from agents.base_agent import BaseAgent, AgentConfig, ToolResult
from agents.agent_tools import get_file_tools

logger = logging.getLogger(__name__)


@dataclass
class IdeaCard:
    """A discovered strategy idea."""
    id: str
    title: str
    source: str  # arxiv, github, blog, youtube, reddit
    source_url: str
    summary: str
    key_concepts: List[str]
    potential_edge: str
    risk_factors: List[str]
    implementation_complexity: str  # low, medium, high
    data_requirements: List[str]
    discovered_at: str
    scout_notes: str
    confidence: float  # 0-1


class ScoutAgent(BaseAgent):
    """
    Discovers and extracts strategy ideas from external sources.

    The Scout monitors:
    - arXiv quant-ph and q-fin for academic papers
    - GitHub for open-source trading systems
    - Financial blogs for practitioner insights
    - YouTube for tutorial content
    - Reddit for community discussions

    Output: IdeaCards saved to ideas/ directory
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
    ):
        if config is None:
            config = AgentConfig(
                name="scout",
                description="Discovers strategy ideas from external sources",
                max_iterations=15,
            )
        super().__init__(config)
        self._discovered_ideas: List[IdeaCard] = []

    def get_system_prompt(self) -> str:
        return """You are a Scout Agent for a quantitative trading research system.

Your mission is to discover new strategy ideas from external sources and extract actionable insights.

CRITICAL CONSTRAINTS:
- You are ADVISORY ONLY - no live trading
- Focus on MEAN REVERSION and MOMENTUM strategies
- Prioritize ideas with clear, testable hypotheses
- Reject vague or unsubstantiated claims
- Be skeptical of extraordinary claims

WHAT TO LOOK FOR:
1. Academic papers with novel approaches to:
   - Mean reversion (IBS, RSI oversold)
   - Momentum (breakouts, trend-following)
   - Market microstructure (liquidity, order flow)
   - Volatility patterns (VIX, IV, regime detection)

2. Practical implementations:
   - Open-source backtesting code
   - Parameter optimization techniques
   - Risk management approaches

3. Red flags to AVOID:
   - Strategies claiming >80% win rate (unrealistic)
   - No discussion of transaction costs
   - Cherry-picked backtests
   - Curve-fitting without out-of-sample testing

OUTPUT FORMAT:
Create IdeaCards with:
- Clear hypothesis
- Required data
- Implementation complexity
- Risk factors
- Your confidence level (0-1)

You have access to file tools to read existing strategies and write drafts.
"""

    def get_tools(self) -> List[Tuple[ToolDefinition, callable]]:
        """Get Scout-specific tools plus file tools."""
        tools = get_file_tools()

        # Add Scout-specific tools
        tools.extend([
            (
                ToolDefinition(
                    name="create_idea_card",
                    description="Create an IdeaCard for a discovered strategy idea",
                    parameters={
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Brief title"},
                            "source": {"type": "string", "description": "Source type (arxiv, github, blog, youtube, reddit)"},
                            "source_url": {"type": "string", "description": "URL of source"},
                            "summary": {"type": "string", "description": "2-3 sentence summary"},
                            "key_concepts": {"type": "array", "items": {"type": "string"}, "description": "Key concepts"},
                            "potential_edge": {"type": "string", "description": "What edge does this provide?"},
                            "risk_factors": {"type": "array", "items": {"type": "string"}, "description": "Risk factors"},
                            "implementation_complexity": {"type": "string", "description": "low/medium/high"},
                            "data_requirements": {"type": "array", "items": {"type": "string"}, "description": "Required data"},
                            "scout_notes": {"type": "string", "description": "Your analysis notes"},
                            "confidence": {"type": "number", "description": "Confidence 0-1"},
                        },
                        "required": ["title", "source", "summary", "key_concepts", "potential_edge", "confidence"],
                    },
                ),
                self._create_idea_card,
            ),
            (
                ToolDefinition(
                    name="search_existing_ideas",
                    description="Search existing ideas to avoid duplicates",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                ),
                self._search_existing_ideas,
            ),
        ])

        return tools

    def _create_idea_card(
        self,
        title: str,
        source: str,
        summary: str,
        key_concepts: List[str],
        potential_edge: str,
        confidence: float,
        source_url: str = "",
        risk_factors: Optional[List[str]] = None,
        implementation_complexity: str = "medium",
        data_requirements: Optional[List[str]] = None,
        scout_notes: str = "",
    ) -> ToolResult:
        """Create and store an IdeaCard."""
        try:
            idea = IdeaCard(
                id=f"idea_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self._discovered_ideas)}",
                title=title,
                source=source,
                source_url=source_url,
                summary=summary,
                key_concepts=key_concepts,
                potential_edge=potential_edge,
                risk_factors=risk_factors or [],
                implementation_complexity=implementation_complexity,
                data_requirements=data_requirements or [],
                discovered_at=datetime.now().isoformat(),
                scout_notes=scout_notes,
                confidence=max(0.0, min(1.0, confidence)),
            )

            self._discovered_ideas.append(idea)

            return ToolResult(
                success=True,
                output=f"Created IdeaCard: {idea.id}\n{json.dumps(asdict(idea), indent=2)}",
                data=asdict(idea),
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def _search_existing_ideas(self, query: str) -> ToolResult:
        """Search for existing ideas to avoid duplicates."""
        try:
            from pathlib import Path

            ideas_dir = Path(__file__).parent.parent / "ideas"
            if not ideas_dir.exists():
                return ToolResult(
                    success=True,
                    output="No existing ideas found (ideas/ directory doesn't exist)",
                    data={"matches": []},
                )

            matches = []
            query_lower = query.lower()

            for f in ideas_dir.glob("*.json"):
                try:
                    with open(f) as fp:
                        idea = json.load(fp)
                    # Simple text search
                    text = json.dumps(idea).lower()
                    if query_lower in text:
                        matches.append({
                            "file": f.name,
                            "title": idea.get("title", ""),
                            "source": idea.get("source", ""),
                        })
                except Exception:
                    continue

            if matches:
                return ToolResult(
                    success=True,
                    output=f"Found {len(matches)} matching ideas:\n" + json.dumps(matches, indent=2),
                    data={"matches": matches},
                )
            else:
                return ToolResult(
                    success=True,
                    output="No matching ideas found",
                    data={"matches": []},
                )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def get_discovered_ideas(self) -> List[IdeaCard]:
        """Get all ideas discovered in this session."""
        return self._discovered_ideas

    def save_ideas(self, output_dir: str = "ideas") -> List[str]:
        """Save discovered ideas to files."""
        from pathlib import Path

        output_path = Path(__file__).parent.parent / output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        saved = []
        for idea in self._discovered_ideas:
            filepath = output_path / f"{idea.id}.json"
            with open(filepath, "w") as f:
                json.dump(asdict(idea), f, indent=2)
            saved.append(str(filepath))

        return saved
