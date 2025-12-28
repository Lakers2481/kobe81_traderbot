from __future__ import annotations

"""
Simple drift detection helpers for model/performance monitoring.

Not used directly by production pipelines yet; included for future integration
with weekly promotion and dashboards.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DriftThresholds:
    min_delta_accuracy: float = -0.02   # degrade of >2% flags drift
    min_delta_wr: float = -0.02         # win-rate degrade >2%
    min_delta_pf: float = -0.10         # profit-factor drop >0.10
    min_delta_sharpe: float = -0.10     # sharpe drop >0.10


def compare_metrics(prev: Dict[str, float], cur: Dict[str, float], thr: Optional[DriftThresholds] = None) -> Dict[str, bool]:
    thr = thr or DriftThresholds()
    def _delta(k: str) -> float:
        return float(cur.get(k, 0.0)) - float(prev.get(k, 0.0))
    out = {
        "accuracy_drift": _delta("accuracy") <= thr.min_delta_accuracy,
        "wr_drift": _delta("wr") <= thr.min_delta_wr,
        "pf_drift": _delta("pf") <= thr.min_delta_pf,
        "sharpe_drift": _delta("sharpe") <= thr.min_delta_sharpe,
    }
    out["any_drift"] = any(out.values())
    return out

