"""
Reporter Agent - Report Generation
==================================

Generates:
- Daily summary reports
- Weekly deep dives
- Audit reports
- Discovery reports

Outputs in markdown format for human review.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date
from typing import List, Optional, Tuple

from llm import ToolDefinition
from agents.base_agent import BaseAgent, AgentConfig, ToolResult
from agents.agent_tools import get_file_tools

logger = logging.getLogger(__name__)


class ReporterAgent(BaseAgent):
    """
    Generates reports for human review.

    Report types:
    - Daily: Data health, discoveries, pass/fail summary
    - Weekly: Deep performance review, strategy comparison
    - Audit: Integrity findings, recommendations
    - Discovery: New ideas found, evaluated, archived
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
    ):
        if config is None:
            config = AgentConfig(
                name="reporter",
                description="Generates reports for human review",
                max_iterations=10,
                temperature=0.5,  # Some creativity for narrative
            )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return """You are a Reporter Agent for a quantitative trading system.

Your mission is to generate clear, actionable reports for human review.

REPORT PRINCIPLES:
1. FACTS FIRST - Lead with data, not opinions
2. ACTIONABLE - Every section should inform decisions
3. HONEST - Report failures prominently, not hidden
4. CONCISE - One page max for daily, three for weekly

REPORT STRUCTURE (Daily):
```markdown
# Daily Report - {date}

## System Health
- Data freshness: ✅/❌
- API status: ✅/❌
- Strategy status: ✅/❌

## Today's Activity
- Signals generated: N
- Trades executed: N
- P&L: $X.XX

## Discoveries
- New ideas: N
- Passed gates: N
- Archived: N

## Issues & Alerts
- List any problems

## Tomorrow's Focus
- Key items to watch
```

REPORT STRUCTURE (Weekly):
```markdown
# Weekly Report - {week}

## Performance Summary
- Win rate: X%
- Profit factor: X.XX
- Total P&L: $X.XX

## Strategy Comparison
| Strategy | WR | PF | Trades |
|----------|----|----|--------|

## Gate Results
- Passed: N
- Failed: N
- Pending: N

## Lessons Learned
- What worked
- What didn't

## Recommendations
- Prioritized list
```

You have access to file tools to read data and write reports.
"""

    def get_tools(self) -> List[Tuple[ToolDefinition, callable]]:
        """Get Reporter-specific tools."""
        tools = get_file_tools()

        tools.extend([
            (
                ToolDefinition(
                    name="write_report",
                    description="Write a report to the reports directory",
                    parameters={
                        "type": "object",
                        "properties": {
                            "report_type": {
                                "type": "string",
                                "enum": ["daily", "weekly", "audit", "discovery"],
                            },
                            "content": {
                                "type": "string",
                                "description": "Markdown content",
                            },
                            "date_str": {
                                "type": "string",
                                "description": "Date string for filename (YYYY-MM-DD)",
                            },
                        },
                        "required": ["report_type", "content"],
                    },
                ),
                self._write_report,
            ),
            (
                ToolDefinition(
                    name="get_system_status",
                    description="Get current system status for reporting",
                    parameters={
                        "type": "object",
                        "properties": {},
                    },
                ),
                self._get_system_status,
            ),
            (
                ToolDefinition(
                    name="get_performance_summary",
                    description="Get performance metrics for reporting",
                    parameters={
                        "type": "object",
                        "properties": {
                            "days": {
                                "type": "integer",
                                "description": "Number of days to summarize",
                            },
                        },
                    },
                ),
                self._get_performance_summary,
            ),
        ])

        return tools

    def _write_report(
        self,
        report_type: str,
        content: str,
        date_str: Optional[str] = None,
    ) -> ToolResult:
        """Write report to reports directory."""
        try:
            from pathlib import Path

            if date_str is None:
                date_str = date.today().isoformat()

            reports_dir = Path(__file__).parent.parent / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{report_type}_{date_str}.md"
            filepath = reports_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            return ToolResult(
                success=True,
                output=f"Report written to: {filepath}",
                data={"path": str(filepath), "type": report_type},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def _get_system_status(self) -> ToolResult:
        """Get current system status."""
        try:
            from pathlib import Path
            import os

            project_root = Path(__file__).parent.parent

            # Check various status indicators
            status = {
                "timestamp": datetime.now().isoformat(),
                "data_status": "unknown",
                "api_status": "unknown",
                "strategy_status": "unknown",
                "kill_switch": False,
            }

            # Check kill switch
            kill_switch_file = project_root / "state" / "KILL_SWITCH"
            status["kill_switch"] = kill_switch_file.exists()

            # Check cache freshness
            cache_dir = project_root / "cache" / "polygon"
            if cache_dir.exists():
                csv_files = list(cache_dir.glob("*.csv"))
                if csv_files:
                    newest = max(f.stat().st_mtime for f in csv_files)
                    age_hours = (datetime.now().timestamp() - newest) / 3600
                    status["data_status"] = "fresh" if age_hours < 24 else f"stale ({age_hours:.1f}h)"
                    status["cache_files"] = len(csv_files)

            # Check API keys
            has_polygon = bool(os.environ.get("POLYGON_API_KEY"))
            has_alpaca = bool(os.environ.get("ALPACA_API_KEY_ID"))
            status["api_status"] = "configured" if (has_polygon and has_alpaca) else "missing"

            return ToolResult(
                success=True,
                output=json.dumps(status, indent=2),
                data=status,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def _get_performance_summary(self, days: int = 7) -> ToolResult:
        """Get performance metrics summary."""
        try:
            from pathlib import Path

            project_root = Path(__file__).parent.parent

            # Try to read trade logs
            signals_file = project_root / "logs" / "signals.jsonl"
            trades_file = project_root / "logs" / "trades.jsonl"

            summary = {
                "period_days": days,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
            }

            if trades_file.exists():
                try:
                    trades = []
                    with open(trades_file) as f:
                        for line in f:
                            if line.strip():
                                trades.append(json.loads(line))

                    if trades:
                        summary["trades"] = len(trades)
                        # Calculate more metrics if data available
                except Exception:
                    pass

            return ToolResult(
                success=True,
                output=json.dumps(summary, indent=2),
                data=summary,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def generate_daily_report(self) -> str:
        """Generate a daily report."""
        result = self.run(
            task="Generate a daily report with system health, today's activity, discoveries, issues, and tomorrow's focus."
        )
        return result.output
