"""
Risk Agent - Quant Gate Validation
==================================

Evaluates strategies against quant gates:
- Gate 0: Sanity (lookahead, leakage)
- Gate 1: Baseline (beat buy-and-hold)
- Gate 2: Robustness (walk-forward, stress)
- Gate 3: Risk Realism (drawdown, turnover)
- Gate 4: Multiple Testing (T-stat penalty)

REJECT-BY-DEFAULT: Any gate failure = archive forever.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from llm import ToolDefinition
from agents.base_agent import BaseAgent, AgentConfig, ToolResult
from agents.agent_tools import get_file_tools, get_backtest_tools

logger = logging.getLogger(__name__)


class GateStatus(Enum):
    """Gate evaluation status."""
    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"
    SKIPPED = "skipped"


@dataclass
class GateResult:
    """Result from a single gate evaluation."""
    gate_id: int
    gate_name: str
    status: str
    metrics: Dict[str, Any]
    threshold: Dict[str, Any]
    message: str
    details: str


@dataclass
class RiskReport:
    """Complete risk evaluation report."""
    id: str
    strategy_id: str
    timestamp: str
    gates: List[GateResult]
    overall_status: str
    recommendation: str
    approval_required: bool


class RiskAgent(BaseAgent):
    """
    Evaluates strategies against 5-gate quant validation pipeline.

    GATES:
    0. SANITY - No lookahead, no leakage
    1. BASELINE - Beat buy-and-hold, MA cross, naive MR
    2. ROBUSTNESS - Walk-forward, stress tests, sensitivity
    3. RISK REALISM - Max DD 25%, min trades 100, diversification
    4. MULTIPLE TESTING - T-stat with attempt penalty

    PHILOSOPHY: REJECT BY DEFAULT
    - Any gate failure = archive forever
    - Never retest failed strategies
    - Human approval required for promotion
    """

    # Gate requirements
    GATE_REQUIREMENTS = {
        0: {
            "name": "Sanity Check",
            "requirements": {
                "lookahead_detected": False,
                "leakage_detected": False,
                "perfect_entries_pct": {"max": 10.0},  # <10% at daily low
            },
        },
        1: {
            "name": "Baseline Beat",
            "requirements": {
                "beats_buy_hold": True,
                "beats_ma_cross": True,
                "beats_naive_mr": True,
            },
        },
        2: {
            "name": "Robustness",
            "requirements": {
                "wf_consistency": {"min": 0.625},  # 5/8 splits profitable
                "sensitivity_ok": True,  # ±5% params stable
                "stress_ok": True,  # 2x costs survive
            },
        },
        3: {
            "name": "Risk Realism",
            "requirements": {
                "max_drawdown_pct": {"max": 25.0},
                "min_trades": {"min": 100},
                "min_symbols": {"min": 30},
                "max_ticker_pnl_pct": {"max": 20.0},
                "max_turnover": {"max": 100.0},  # 100x annual
            },
        },
        4: {
            "name": "Multiple Testing",
            "requirements": {
                "adjusted_t_stat": {"min": 2.0},  # After penalty
                # Penalty: base 2.0 + 0.1 per 10 attempts + 0.1 per param
            },
        },
    }

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
    ):
        if config is None:
            config = AgentConfig(
                name="risk",
                description="Quant gate validation",
                max_iterations=15,
                temperature=0.1,
            )
        super().__init__(config)
        self._gate_results: List[GateResult] = []

    def get_system_prompt(self) -> str:
        return """You are a Risk Agent for a quantitative trading system.

Your mission is to evaluate strategies against a 5-gate validation pipeline.

CRITICAL: REJECT BY DEFAULT
- ONE gate failure = archive forever
- Never retest failed strategies
- Be RUTHLESSLY strict

GATES TO EVALUATE:

GATE 0: SANITY CHECK
- No lookahead bias
- No data leakage
- <10% perfect entries at daily low

GATE 1: BASELINE BEAT
- Must beat buy-and-hold on same period/universe
- Must beat 50/200 MA crossover
- Must beat naive mean reversion

GATE 2: ROBUSTNESS
- Walk-forward: 62.5%+ splits profitable
- Stress: Survives 2x costs, 2x slippage
- Sensitivity: ±5% param changes don't break it

GATE 3: RISK REALISM
- Max drawdown: 25%
- Min trades: 100
- Min symbols: 30
- Max ticker P&L concentration: 20%
- Max turnover: 100x annual

GATE 4: MULTIPLE TESTING
- Adjusted T-statistic > 2.0
- Penalty for attempts: +0.1 per 10 attempts
- Penalty for parameters: +0.1 per free parameter

EVALUATION PROCESS:
1. Read backtest results
2. Evaluate each gate in order
3. STOP at first failure
4. Generate report with recommendation

You have access to file tools and backtest data.
"""

    def get_tools(self) -> List[Tuple[ToolDefinition, callable]]:
        """Get Risk-specific tools."""
        tools = get_file_tools() + get_backtest_tools()

        tools.extend([
            (
                ToolDefinition(
                    name="evaluate_gate",
                    description="Evaluate a specific gate",
                    parameters={
                        "type": "object",
                        "properties": {
                            "gate_id": {
                                "type": "integer",
                                "description": "Gate number (0-4)",
                            },
                            "metrics": {
                                "type": "object",
                                "description": "Metrics to evaluate",
                            },
                            "details": {
                                "type": "string",
                                "description": "Detailed analysis",
                            },
                        },
                        "required": ["gate_id", "metrics"],
                    },
                ),
                self._evaluate_gate,
            ),
            (
                ToolDefinition(
                    name="check_baseline_beat",
                    description="Check if strategy beats baselines",
                    parameters={
                        "type": "object",
                        "properties": {
                            "strategy_sharpe": {"type": "number"},
                            "strategy_cagr": {"type": "number"},
                            "benchmark_sharpe": {"type": "number"},
                            "benchmark_cagr": {"type": "number"},
                        },
                        "required": ["strategy_sharpe", "strategy_cagr"],
                    },
                ),
                self._check_baseline_beat,
            ),
            (
                ToolDefinition(
                    name="calculate_adjusted_t_stat",
                    description="Calculate T-stat with multiple testing penalty",
                    parameters={
                        "type": "object",
                        "properties": {
                            "raw_t_stat": {"type": "number"},
                            "num_attempts": {"type": "integer"},
                            "num_parameters": {"type": "integer"},
                        },
                        "required": ["raw_t_stat", "num_attempts", "num_parameters"],
                    },
                ),
                self._calculate_adjusted_t_stat,
            ),
            (
                ToolDefinition(
                    name="generate_risk_report",
                    description="Generate final risk report",
                    parameters={
                        "type": "object",
                        "properties": {
                            "strategy_id": {"type": "string"},
                            "recommendation": {"type": "string"},
                        },
                        "required": ["strategy_id", "recommendation"],
                    },
                ),
                self._generate_risk_report,
            ),
        ])

        return tools

    def _evaluate_gate(
        self,
        gate_id: int,
        metrics: Dict[str, Any],
        details: str = "",
    ) -> ToolResult:
        """Evaluate a specific gate."""
        try:
            if gate_id not in self.GATE_REQUIREMENTS:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Invalid gate ID: {gate_id}",
                )

            gate_req = self.GATE_REQUIREMENTS[gate_id]
            gate_name = gate_req["name"]
            requirements = gate_req["requirements"]

            passed = True
            messages = []

            for metric, requirement in requirements.items():
                value = metrics.get(metric)

                if value is None:
                    messages.append(f"Missing metric: {metric}")
                    passed = False
                    continue

                if isinstance(requirement, bool):
                    if value != requirement:
                        messages.append(f"{metric}: expected {requirement}, got {value}")
                        passed = False
                elif isinstance(requirement, dict):
                    if "min" in requirement and value < requirement["min"]:
                        messages.append(f"{metric}: {value} < min {requirement['min']}")
                        passed = False
                    if "max" in requirement and value > requirement["max"]:
                        messages.append(f"{metric}: {value} > max {requirement['max']}")
                        passed = False

            status = GateStatus.PASSED if passed else GateStatus.FAILED
            result = GateResult(
                gate_id=gate_id,
                gate_name=gate_name,
                status=status.value,
                metrics=metrics,
                threshold=requirements,
                message="; ".join(messages) if messages else "All checks passed",
                details=details,
            )

            self._gate_results.append(result)

            status_str = "PASSED" if passed else "FAILED"
            return ToolResult(
                success=True,
                output=f"Gate {gate_id} ({gate_name}): {status_str}\n{result.message}",
                data=asdict(result),
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def _check_baseline_beat(
        self,
        strategy_sharpe: float,
        strategy_cagr: float,
        benchmark_sharpe: float = 0.5,  # Typical S&P
        benchmark_cagr: float = 0.10,   # Typical S&P
    ) -> ToolResult:
        """Check if strategy beats baselines."""
        beats_sharpe = strategy_sharpe > benchmark_sharpe
        beats_cagr = strategy_cagr > benchmark_cagr

        result = {
            "beats_sharpe": beats_sharpe,
            "beats_cagr": beats_cagr,
            "beats_both": beats_sharpe and beats_cagr,
            "strategy_sharpe": strategy_sharpe,
            "strategy_cagr": strategy_cagr,
            "benchmark_sharpe": benchmark_sharpe,
            "benchmark_cagr": benchmark_cagr,
        }

        status = "PASS" if result["beats_both"] else "FAIL"
        return ToolResult(
            success=True,
            output=f"Baseline check: {status}\nSharpe: {strategy_sharpe:.2f} vs {benchmark_sharpe:.2f}\nCAGR: {strategy_cagr:.1%} vs {benchmark_cagr:.1%}",
            data=result,
        )

    def _calculate_adjusted_t_stat(
        self,
        raw_t_stat: float,
        num_attempts: int,
        num_parameters: int,
    ) -> ToolResult:
        """Calculate adjusted T-stat with multiple testing penalty."""
        # Base threshold
        base_threshold = 2.0

        # Penalty for attempts
        attempt_penalty = 0.1 * (num_attempts // 10)

        # Penalty for parameters
        param_penalty = 0.1 * num_parameters

        # Total threshold
        adjusted_threshold = base_threshold + attempt_penalty + param_penalty

        # Check if passes
        passes = raw_t_stat >= adjusted_threshold

        result = {
            "raw_t_stat": raw_t_stat,
            "base_threshold": base_threshold,
            "attempt_penalty": attempt_penalty,
            "param_penalty": param_penalty,
            "adjusted_threshold": adjusted_threshold,
            "passes": passes,
        }

        status = "PASS" if passes else "FAIL"
        return ToolResult(
            success=True,
            output=f"""
Multiple Testing Check: {status}
Raw T-stat: {raw_t_stat:.2f}
Adjusted threshold: {adjusted_threshold:.2f}
  Base: {base_threshold:.2f}
  + Attempt penalty ({num_attempts} attempts): {attempt_penalty:.2f}
  + Param penalty ({num_parameters} params): {param_penalty:.2f}
""",
            data=result,
        )

    def _generate_risk_report(
        self,
        strategy_id: str,
        recommendation: str,
    ) -> ToolResult:
        """Generate final risk report."""
        try:
            # Check if any gate failed
            failed_gates = [r for r in self._gate_results if r.status == "failed"]
            all_passed = len(failed_gates) == 0

            overall_status = "APPROVED" if all_passed else "REJECTED"

            report = RiskReport(
                id=f"risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                strategy_id=strategy_id,
                timestamp=datetime.now().isoformat(),
                gates=[asdict(r) for r in self._gate_results],
                overall_status=overall_status,
                recommendation=recommendation,
                approval_required=all_passed,  # Only ask for approval if passed
            )

            gate_summary = "\n".join([
                f"  Gate {r.gate_id} ({r.gate_name}): {r.status.upper()}"
                for r in self._gate_results
            ])

            return ToolResult(
                success=True,
                output=f"""
RISK EVALUATION REPORT
{'='*50}
Strategy: {strategy_id}
Overall: {overall_status}

Gates:
{gate_summary}

Recommendation: {recommendation}

{'REQUIRES HUMAN APPROVAL FOR PROMOTION' if all_passed else 'ARCHIVED - Do not retest'}
""",
                data=asdict(report),
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def all_gates_passed(self) -> bool:
        """Check if all evaluated gates passed."""
        return all(r.status == "passed" for r in self._gate_results)
