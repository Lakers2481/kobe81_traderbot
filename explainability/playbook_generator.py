"""
Playbook generator for trade explanations.

Generates human-readable playbooks from DecisionPackets.
CRITICAL: Only references fields present in the packet - never fabricates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os

from .decision_packet import DecisionPacket


@dataclass
class Playbook:
    """Generated playbook for a trade."""
    run_id: str
    symbol: str
    timestamp: str
    executive_summary: str
    full_playbook: str
    risk_section: str
    confidence_section: str
    checklist: str
    generation_method: str  # "claude" or "deterministic"


class PlaybookGenerator:
    """
    Generates trade playbooks from decision packets.

    CRITICAL RULES:
    1. ONLY cite fields present in the DecisionPacket
    2. If a field is missing, explicitly say "Unknown" or "Not available"
    3. NEVER fabricate data or make assumptions beyond the packet
    4. Use deterministic fallback if Claude API unavailable
    """

    def __init__(
        self,
        use_claude: bool = True,
        claude_api_key: Optional[str] = None,
    ):
        self.use_claude = use_claude
        self.claude_api_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY")

        # Check if Claude is available
        self._claude_available = False
        if self.use_claude and self.claude_api_key:
            try:
                import anthropic
                self._claude_available = True
            except ImportError:
                self._claude_available = False

    def generate_from_packet(self, packet: DecisionPacket) -> Playbook:
        """
        Generate complete playbook from a decision packet.

        Uses Claude API if available, otherwise deterministic fallback.
        """
        if self._claude_available and self.use_claude:
            return self._generate_with_claude(packet)
        else:
            return self._generate_deterministic(packet)

    def _generate_with_claude(self, packet: DecisionPacket) -> Playbook:
        """Generate playbook using Claude API."""
        import anthropic

        client = anthropic.Anthropic(api_key=self.claude_api_key)

        # Build prompt with packet data
        prompt = self._build_claude_prompt(packet)

        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",  # Upgraded to Sonnet 4
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = message.content[0].text

            # Parse sections from response
            sections = self._parse_claude_response(response_text)

            return Playbook(
                run_id=packet.run_id,
                symbol=packet.symbol,
                timestamp=datetime.utcnow().isoformat(),
                executive_summary=sections.get("executive_summary", ""),
                full_playbook=sections.get("full_playbook", response_text),
                risk_section=sections.get("risk_section", ""),
                confidence_section=sections.get("confidence_section", ""),
                checklist=sections.get("checklist", ""),
                generation_method="claude",
            )

        except Exception as e:
            # Fall back to deterministic on error
            playbook = self._generate_deterministic(packet)
            playbook.generation_method = f"deterministic (claude failed: {e})"
            return playbook

    def _build_claude_prompt(self, packet: DecisionPacket) -> str:
        """Build prompt for Claude with packet data."""
        prompt = f"""You are a trading playbook generator. Generate a clear, actionable playbook for this trade.

CRITICAL RULES:
1. ONLY use information provided in the Decision Packet below
2. If information is missing, explicitly state "Unknown" or "Not available"
3. NEVER make up statistics, probabilities, or claims not in the packet
4. Be concise but complete

DECISION PACKET:
```json
{packet.to_json()}
```

Generate a playbook with these sections:

## EXECUTIVE SUMMARY (5 bullet points max)
- What we're trading and why
- Key risk/reward metrics
- Confidence level

## FULL PLAYBOOK

### The Setup
Describe the trade setup based on strategy_reasons and signal_description.

### Entry Plan
Based on execution_plan, describe entry.

### Exit Plan
Describe stop loss and take profit levels.

### Position Sizing
Describe size and risk.

## RISK SECTION
What can go wrong? Reference any risk gate warnings.

## CONFIDENCE SECTION
Based on ml_outputs and historical_analogs, describe confidence.
If no ML or analogs provided, say "Unknown - no historical data available".

## CHECKLIST
Pre-trade and post-trade checklist items.

Remember: If the packet lists items in 'unknowns', acknowledge them."""

        return prompt

    def _parse_claude_response(self, response: str) -> Dict[str, str]:
        """Parse Claude response into sections."""
        sections = {}

        # Simple section parsing
        current_section = None
        current_content = []

        for line in response.split("\n"):
            if line.startswith("## EXECUTIVE SUMMARY"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "executive_summary"
                current_content = []
            elif line.startswith("## FULL PLAYBOOK"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "full_playbook"
                current_content = []
            elif line.startswith("## RISK SECTION"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "risk_section"
                current_content = []
            elif line.startswith("## CONFIDENCE SECTION"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "confidence_section"
                current_content = []
            elif line.startswith("## CHECKLIST"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "checklist"
                current_content = []
            else:
                current_content.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def _generate_deterministic(self, packet: DecisionPacket) -> Playbook:
        """Generate playbook using deterministic templates."""
        exec_summary = self.generate_executive_summary(packet)
        full_playbook = self.generate_full_playbook(packet)
        risk_section = self.generate_risk_section(packet)
        confidence_section = self.generate_confidence_section(packet)
        checklist = self.generate_checklist(packet)

        return Playbook(
            run_id=packet.run_id,
            symbol=packet.symbol,
            timestamp=datetime.utcnow().isoformat(),
            executive_summary=exec_summary,
            full_playbook=full_playbook,
            risk_section=risk_section,
            confidence_section=confidence_section,
            checklist=checklist,
            generation_method="deterministic",
        )

    def generate_executive_summary(self, packet: DecisionPacket) -> str:
        """Generate 5-bullet executive summary."""
        lines = [f"## Executive Summary - {packet.symbol} {packet.side.upper()}", ""]

        # Bullet 1: What and why
        if packet.strategy_reasons:
            lines.append(f"- **Signal**: {packet.strategy_reasons[0]}")
        else:
            lines.append("- **Signal**: Unknown - no strategy reasons provided")

        # Bullet 2: Entry/Exit
        if packet.execution_plan:
            ep = packet.execution_plan
            lines.append(f"- **Entry**: ${ep.entry_price:.2f} | Stop: ${ep.stop_loss:.2f} | Target: ${ep.take_profit:.2f}")
        else:
            lines.append("- **Entry/Exit**: Unknown - no execution plan provided")

        # Bullet 3: Risk/Reward
        if packet.execution_plan:
            ep = packet.execution_plan
            lines.append(f"- **R:R Ratio**: {ep.reward_risk_ratio:.2f}:1 | Risk: ${ep.risk_amount:.2f}")
        else:
            lines.append("- **R:R Ratio**: Unknown")

        # Bullet 4: Position size
        if packet.execution_plan:
            ep = packet.execution_plan
            lines.append(f"- **Size**: {ep.position_size} shares (${ep.notional:.2f} notional)")
        else:
            lines.append("- **Size**: Unknown")

        # Bullet 5: Confidence
        if packet.ml_outputs and "probability" in packet.ml_outputs:
            prob = packet.ml_outputs["probability"]
            lines.append(f"- **ML Confidence**: {prob:.1%}")
        else:
            lines.append("- **Confidence**: Unknown - no ML model output")

        return "\n".join(lines)

    def generate_full_playbook(self, packet: DecisionPacket) -> str:
        """Generate full playbook text."""
        lines = [f"# Trade Playbook: {packet.symbol}", ""]
        lines.append(f"**Generated**: {packet.timestamp}")
        lines.append(f"**Run ID**: {packet.run_id}")
        lines.append(f"**Strategy**: {packet.strategy_name}")
        lines.append("")

        # The Setup
        lines.append("## The Setup")
        if packet.signal_description:
            lines.append(packet.signal_description)
        elif packet.strategy_reasons:
            for reason in packet.strategy_reasons:
                lines.append(f"- {reason}")
        else:
            lines.append("*No setup description available*")
        lines.append("")

        # Key Features
        if packet.feature_values:
            lines.append("## Key Features at Signal Time")
            for feat, val in sorted(packet.feature_values.items())[:10]:
                lines.append(f"- {feat}: {val:.4f}")
            lines.append("")

        # Entry Plan
        lines.append("## Entry Plan")
        if packet.execution_plan:
            ep = packet.execution_plan
            lines.append(f"- Entry Price: ${ep.entry_price:.2f}")
            lines.append(f"- Position Size: {ep.position_size} shares")
            lines.append(f"- Notional Value: ${ep.notional:.2f}")
        else:
            lines.append("*Execution plan not available*")
        lines.append("")

        # Exit Plan
        lines.append("## Exit Plan")
        if packet.execution_plan:
            ep = packet.execution_plan
            lines.append(f"- Stop Loss: ${ep.stop_loss:.2f} (Risk: ${ep.risk_amount:.2f})")
            lines.append(f"- Take Profit: ${ep.take_profit:.2f} (Reward: ${ep.reward_amount:.2f})")
            lines.append(f"- Reward/Risk Ratio: {ep.reward_risk_ratio:.2f}:1")
        else:
            lines.append("*Exit levels not available*")
        lines.append("")

        # Historical Analogs
        if packet.historical_analogs:
            lines.append("## Historical Analogs")
            for analog in packet.historical_analogs[:3]:
                lines.append(f"- {analog.date} {analog.symbol}: {analog.pnl_pct:+.1f}% "
                           f"(held {analog.holding_days}d, similarity: {analog.similarity_score:.0%})")
        lines.append("")

        # Unknowns
        if packet.unknowns:
            lines.append("## Unknowns / Limitations")
            for unknown in packet.unknowns:
                lines.append(f"- {unknown}")

        return "\n".join(lines)

    def generate_risk_section(self, packet: DecisionPacket) -> str:
        """Generate risk analysis section."""
        lines = ["## Risk Analysis", ""]

        # Risk gate results
        if packet.risk_gate_results:
            lines.append("### Risk Gate Results")
            for gate in packet.risk_gate_results:
                status = "PASS" if gate.passed else "FAIL"
                lines.append(f"- **{gate.gate_name}**: {status}")
                if gate.message:
                    lines.append(f"  - {gate.message}")
                if gate.value is not None and gate.limit is not None:
                    lines.append(f"  - Value: {gate.value:.2f} / Limit: {gate.limit:.2f}")
            lines.append("")
        else:
            lines.append("*No risk gate results available*")
            lines.append("")

        # What can go wrong
        lines.append("### What Can Go Wrong")
        lines.append("- Price gaps through stop loss")
        lines.append("- Unexpected news/earnings")
        lines.append("- Broad market selloff")

        if packet.market_context.get("vix"):
            vix = packet.market_context["vix"]
            if vix > 25:
                lines.append(f"- **Elevated VIX ({vix:.1f})**: Higher volatility expected")

        return "\n".join(lines)

    def generate_confidence_section(self, packet: DecisionPacket) -> str:
        """Generate confidence analysis section."""
        lines = ["## Confidence Analysis", ""]

        # ML outputs
        if packet.ml_outputs:
            lines.append("### ML Model Output")
            for key, val in packet.ml_outputs.items():
                if isinstance(val, float):
                    lines.append(f"- {key}: {val:.4f}")
                else:
                    lines.append(f"- {key}: {val}")
            lines.append("")
        else:
            lines.append("*No ML model output available*")
            lines.append("")

        # Sentiment
        if packet.sentiment_score is not None:
            source = packet.sentiment_source or "unknown"
            lines.append(f"### Sentiment ({source})")
            lines.append(f"- Score: {packet.sentiment_score:.2f}")
            lines.append("")
        else:
            lines.append("*Sentiment data not available*")
            lines.append("")

        # Historical performance
        if packet.historical_analogs:
            wins = sum(1 for a in packet.historical_analogs if a.pnl_pct > 0)
            total = len(packet.historical_analogs)
            avg_pnl = sum(a.pnl_pct for a in packet.historical_analogs) / total
            lines.append("### Historical Analog Performance")
            lines.append(f"- Win Rate: {wins}/{total} ({wins/total:.0%})")
            lines.append(f"- Average P&L: {avg_pnl:+.1f}%")
        else:
            lines.append("*No historical analogs available*")

        return "\n".join(lines)

    def generate_checklist(self, packet: DecisionPacket) -> str:
        """Generate pre/post trade checklist."""
        lines = ["## Trade Checklist", ""]

        lines.append("### Pre-Trade")
        lines.append("- [ ] Confirm market is open")
        lines.append("- [ ] Verify quote is fresh (<5 seconds)")
        lines.append("- [ ] Check spread is acceptable (<0.5%)")
        lines.append("- [ ] Verify position size matches plan")
        lines.append("- [ ] Confirm risk gates all passed")

        if packet.execution_plan:
            ep = packet.execution_plan
            lines.append(f"- [ ] Set stop loss at ${ep.stop_loss:.2f}")
            lines.append(f"- [ ] Set take profit at ${ep.take_profit:.2f}")

        lines.append("")
        lines.append("### Post-Trade")
        lines.append("- [ ] Verify fill price matches expectation")
        lines.append("- [ ] Confirm stop loss order is active")
        lines.append("- [ ] Record entry in trade journal")
        lines.append("- [ ] Set reminder for trade review")

        return "\n".join(lines)

    def save_playbook(
        self,
        playbook: Playbook,
        output_dir: Path,
        format: str = "both",  # "md", "html", or "both"
    ) -> List[Path]:
        """Save playbook to file(s)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        base_name = f"{playbook.timestamp[:10]}_{playbook.symbol}"

        # Combine all sections
        full_content = "\n\n".join([
            playbook.executive_summary,
            playbook.full_playbook,
            playbook.risk_section,
            playbook.confidence_section,
            playbook.checklist,
        ])

        if format in ("md", "both"):
            md_path = output_dir / f"{base_name}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(full_content)
            saved_paths.append(md_path)

        if format in ("html", "both"):
            try:
                import markdown
                html_content = markdown.markdown(full_content, extensions=["tables"])
                html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>Trade Playbook - {playbook.symbol}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        ul {{ padding-left: 20px; }}
        .checklist {{ list-style-type: none; }}
    </style>
</head>
<body>
{html_content}
<hr>
<p><em>Generated: {playbook.timestamp} | Method: {playbook.generation_method}</em></p>
</body>
</html>"""
                html_path = output_dir / f"{base_name}.html"
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_template)
                saved_paths.append(html_path)
            except ImportError:
                # markdown not installed, skip HTML
                pass

        return saved_paths
