"""
Auditor Agent - Integrity and Bias Detection
=============================================

Validates strategies for:
- Lookahead bias (using future data)
- Survivorship bias (only successful stocks)
- Selection bias (cherry-picked periods)
- Overfitting (too many parameters)
- Data leakage (information bleeding)

CRITICAL: This agent is the last line of defense.
Any detected issue = IMMEDIATE REJECTION.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

from llm import ToolDefinition
from agents.base_agent import BaseAgent, AgentConfig, ToolResult
from agents.agent_tools import get_file_tools, read_file

logger = logging.getLogger(__name__)


class BiasType(Enum):
    """Types of bias that can be detected."""
    LOOKAHEAD = "lookahead"           # Using future information
    SURVIVORSHIP = "survivorship"     # Only surviving stocks
    SELECTION = "selection"           # Cherry-picked periods
    OVERFITTING = "overfitting"       # Too many parameters
    DATA_LEAKAGE = "data_leakage"     # Information bleeding
    COST_OMISSION = "cost_omission"   # Missing transaction costs
    EXECUTION = "execution"           # Unrealistic execution


class Severity(Enum):
    """Severity of detected issues."""
    CRITICAL = "critical"  # Automatic rejection
    HIGH = "high"          # Likely rejection
    MEDIUM = "medium"      # Needs review
    LOW = "low"            # Minor concern


@dataclass
class AuditFinding:
    """A single audit finding."""
    id: str
    bias_type: str
    severity: str
    location: str  # File:line or component
    description: str
    evidence: str
    recommendation: str
    auto_reject: bool


@dataclass
class AuditReport:
    """Complete audit report."""
    id: str
    target: str  # What was audited
    auditor: str
    timestamp: str
    findings: List[AuditFinding]
    passed: bool
    summary: str
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int


class AuditorAgent(BaseAgent):
    """
    Audits strategies for integrity issues and bias.

    This is the LAST LINE OF DEFENSE before a strategy
    is promoted to production.

    ANY critical finding = AUTOMATIC REJECTION
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
    ):
        if config is None:
            config = AgentConfig(
                name="auditor",
                description="Validates strategies for bias and integrity",
                max_iterations=20,
                temperature=0.1,  # Very low for precise analysis
            )
        super().__init__(config)
        self._findings: List[AuditFinding] = []

    def get_system_prompt(self) -> str:
        return """You are an Auditor Agent for a quantitative trading system.

Your mission is to detect ANY form of bias, leakage, or integrity issue that would invalidate a trading strategy.

YOU ARE THE LAST LINE OF DEFENSE. Be ruthlessly thorough.

BIAS TYPES TO DETECT:

1. LOOKAHEAD BIAS (CRITICAL)
   - Using future data in calculations
   - Signals generated AFTER the price move
   - Missing .shift(1) on indicators
   - Accessing next-day data for today's signal

2. SURVIVORSHIP BIAS (HIGH)
   - Using only currently-listed stocks
   - Missing delisted/bankrupt companies
   - Universe changes not tracked

3. SELECTION BIAS (HIGH)
   - Cherry-picked date ranges
   - Excluding unfavorable periods
   - Only testing on favorable stocks

4. OVERFITTING (MEDIUM-HIGH)
   - Too many parameters (>10 is suspicious)
   - Perfect fit on training data
   - No out-of-sample testing
   - Parameter sensitivity not checked

5. DATA LEAKAGE (CRITICAL)
   - Training on test data
   - Information from future leaking to past
   - Cross-validation contamination

6. COST OMISSION (HIGH)
   - Missing transaction costs
   - Unrealistic slippage assumptions
   - No bid-ask spread modeling

7. EXECUTION BIAS (MEDIUM)
   - Assuming trades at exact prices
   - Ignoring market impact
   - Unrealistic fill rates

CODE PATTERNS TO FLAG:

```python
# LOOKAHEAD - CRITICAL
df['signal'] = df['close'] > df['sma']  # BAD: No shift
df['signal'] = df['close'] > df['sma'].shift(1)  # GOOD

# Using today's close for today's signal
entry_price = row['close']  # BAD if signal based on close

# Missing next-bar execution
if signal:
    execute_now()  # BAD: Should execute next bar
```

AUDIT PROCESS:
1. Read the strategy code
2. Check indicator calculations for .shift()
3. Verify signal timing vs execution timing
4. Check for hardcoded dates or symbols
5. Verify cost assumptions
6. Look for information leakage

OUTPUT:
Create detailed AuditFindings for each issue found.
ONE critical finding = AUTOMATIC FAIL.
"""

    def get_tools(self) -> List[Tuple[ToolDefinition, callable]]:
        """Get Auditor-specific tools plus file tools."""
        tools = get_file_tools()

        tools.extend([
            (
                ToolDefinition(
                    name="record_finding",
                    description="Record an audit finding",
                    parameters={
                        "type": "object",
                        "properties": {
                            "bias_type": {
                                "type": "string",
                                "enum": ["lookahead", "survivorship", "selection", "overfitting", "data_leakage", "cost_omission", "execution"],
                                "description": "Type of bias detected",
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["critical", "high", "medium", "low"],
                                "description": "Severity level",
                            },
                            "location": {
                                "type": "string",
                                "description": "File:line or component name",
                            },
                            "description": {
                                "type": "string",
                                "description": "What the issue is",
                            },
                            "evidence": {
                                "type": "string",
                                "description": "Code snippet or data showing issue",
                            },
                            "recommendation": {
                                "type": "string",
                                "description": "How to fix it",
                            },
                        },
                        "required": ["bias_type", "severity", "location", "description", "evidence"],
                    },
                ),
                self._record_finding,
            ),
            (
                ToolDefinition(
                    name="check_shift_usage",
                    description="Check if a file properly uses .shift() for indicators",
                    parameters={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to strategy file",
                            },
                        },
                        "required": ["file_path"],
                    },
                ),
                self._check_shift_usage,
            ),
            (
                ToolDefinition(
                    name="count_parameters",
                    description="Count tunable parameters in a strategy",
                    parameters={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to strategy file",
                            },
                        },
                        "required": ["file_path"],
                    },
                ),
                self._count_parameters,
            ),
            (
                ToolDefinition(
                    name="generate_report",
                    description="Generate final audit report",
                    parameters={
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "description": "What was audited",
                            },
                            "summary": {
                                "type": "string",
                                "description": "Summary of findings",
                            },
                        },
                        "required": ["target", "summary"],
                    },
                ),
                self._generate_report,
            ),
        ])

        return tools

    def _record_finding(
        self,
        bias_type: str,
        severity: str,
        location: str,
        description: str,
        evidence: str,
        recommendation: str = "",
    ) -> ToolResult:
        """Record an audit finding."""
        try:
            finding = AuditFinding(
                id=f"finding_{len(self._findings)}",
                bias_type=bias_type,
                severity=severity,
                location=location,
                description=description,
                evidence=evidence,
                recommendation=recommendation,
                auto_reject=severity == "critical",
            )

            self._findings.append(finding)

            return ToolResult(
                success=True,
                output=f"Recorded {severity.upper()} finding: {description}",
                data=asdict(finding),
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def _check_shift_usage(self, file_path: str) -> ToolResult:
        """Check if file properly uses .shift() for lookahead prevention."""
        result = read_file(file_path, max_lines=1000)
        if not result.success:
            return result

        content = result.output
        issues = []

        # Patterns that need .shift()
        dangerous_patterns = [
            ("close", "signal"),  # close used in signal without shift
            ("high", "signal"),
            ("low", "signal"),
            ("volume", "signal"),
        ]

        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()

            # Check for signal assignment without shift
            if "signal" in line_lower and ("close" in line_lower or "high" in line_lower or "low" in line_lower):
                if ".shift(" not in line:
                    issues.append(f"Line {i}: Potential lookahead - signal uses price without shift")

            # Check for direct comparison without shift
            if "> df[" in line or "< df[" in line or ">= df[" in line or "<= df[" in line:
                if ".shift(" not in line and "signal" not in line_lower:
                    issues.append(f"Line {i}: Direct price comparison without shift")

        if issues:
            return ToolResult(
                success=True,
                output=f"Found {len(issues)} potential lookahead issues:\n" + "\n".join(issues),
                data={"issues": issues, "has_problems": True},
            )
        else:
            return ToolResult(
                success=True,
                output="No obvious lookahead issues found (still verify manually)",
                data={"issues": [], "has_problems": False},
            )

    def _count_parameters(self, file_path: str) -> ToolResult:
        """Count tunable parameters in strategy."""
        result = read_file(file_path, max_lines=1000)
        if not result.success:
            return result

        content = result.output

        # Look for parameter definitions
        param_patterns = [
            "window=", "period=", "lookback=", "threshold=",
            "rsi_", "sma_", "ema_", "atr_", "bb_",
            "entry_", "exit_", "stop_", "take_profit",
        ]

        params_found = []
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            for pattern in param_patterns:
                if pattern in line.lower():
                    params_found.append(f"Line {i}: {line.strip()[:80]}")

        # Count unique parameters
        param_count = len(set(params_found))

        severity = "low"
        if param_count > 15:
            severity = "critical"
        elif param_count > 10:
            severity = "high"
        elif param_count > 7:
            severity = "medium"

        return ToolResult(
            success=True,
            output=f"Found {param_count} potential parameters ({severity} overfitting risk):\n" + "\n".join(params_found[:20]),
            data={"count": param_count, "severity": severity, "params": params_found},
        )

    def _generate_report(self, target: str, summary: str) -> ToolResult:
        """Generate final audit report."""
        try:
            critical = sum(1 for f in self._findings if f.severity == "critical")
            high = sum(1 for f in self._findings if f.severity == "high")
            medium = sum(1 for f in self._findings if f.severity == "medium")
            low = sum(1 for f in self._findings if f.severity == "low")

            # CRITICAL = automatic fail
            passed = critical == 0

            report = AuditReport(
                id=f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                target=target,
                auditor="AuditorAgent",
                timestamp=datetime.now().isoformat(),
                findings=[asdict(f) for f in self._findings],
                passed=passed,
                summary=summary,
                critical_count=critical,
                high_count=high,
                medium_count=medium,
                low_count=low,
            )

            status = "PASSED" if passed else "FAILED"

            return ToolResult(
                success=True,
                output=f"""
AUDIT REPORT: {status}
{'='*50}
Target: {target}
Timestamp: {report['timestamp']}

Findings:
- Critical: {critical} (auto-reject)
- High: {high}
- Medium: {medium}
- Low: {low}

Summary: {summary}

{'REJECTED - Critical issues found' if not passed else 'Approved for next stage'}
""",
                data=asdict(report),
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def get_findings(self) -> List[AuditFinding]:
        """Get all findings."""
        return self._findings

    def has_critical_findings(self) -> bool:
        """Check if any critical findings exist."""
        return any(f.severity == "critical" for f in self._findings)
