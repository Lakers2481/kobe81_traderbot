"""
Quant Gates - 5-Gate Validation Pipeline
=========================================

Validates strategies through rigorous quant gates:
- Gate 0: Sanity (lookahead, leakage)
- Gate 1: Baseline (beat buy-and-hold, MA cross, naive MR)
- Gate 2: Robustness (walk-forward, stress, sensitivity)
- Gate 3: Risk Realism (drawdown, diversification, turnover)
- Gate 4: Multiple Testing (T-stat with attempt penalty)

PHILOSOPHY: REJECT BY DEFAULT
- Any gate failure = archive forever
- Never retest failed strategies
- Track attempts per strategy family
"""

from .pipeline import (
    QuantGatesPipeline,
    GateResult,
    PipelineResult,
    run_full_validation,
)
from .gate_0_sanity import Gate0Sanity
from .gate_1_baseline import Gate1Baseline
from .gate_2_robustness import Gate2Robustness
from .gate_3_risk import Gate3RiskRealism
from .gate_4_multiple_testing import Gate4MultipleTesting

__all__ = [
    # Pipeline
    "QuantGatesPipeline",
    "GateResult",
    "PipelineResult",
    "run_full_validation",
    # Individual gates
    "Gate0Sanity",
    "Gate1Baseline",
    "Gate2Robustness",
    "Gate3RiskRealism",
    "Gate4MultipleTesting",
]
