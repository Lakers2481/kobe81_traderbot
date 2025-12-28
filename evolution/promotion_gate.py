from __future__ import annotations

"""
Promotion gate (optional).

Decides whether to promote a candidate model/config based on out-of-sample
metrics and guardrails.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class PromotionThresholds:
    min_oos_sharpe: float = 1.0
    min_profit_factor: float = 1.5
    min_trades: int = 30
    min_wf_splits_pass: int = 3
    min_accuracy_delta: float = 0.0  # relative improvement allowed negative if not used


def should_promote(metrics: Dict[str, float], thr: PromotionThresholds | None = None) -> Tuple[bool, str]:
    thr = thr or PromotionThresholds()
    sharpe = float(metrics.get("oos_sharpe", 0.0))
    pf = float(metrics.get("profit_factor", 0.0))
    trades = int(metrics.get("trades", 0))
    splits = int(metrics.get("wf_splits_pass", 0))
    acc_delta = float(metrics.get("accuracy_delta", 0.0))

    if sharpe < thr.min_oos_sharpe:
        return False, f"oos_sharpe:{sharpe:.2f} < {thr.min_oos_sharpe:.2f}"
    if pf < thr.min_profit_factor:
        return False, f"profit_factor:{pf:.2f} < {thr.min_profit_factor:.2f}"
    if trades < thr.min_trades:
        return False, f"trades:{trades} < {thr.min_trades}"
    if splits < thr.min_wf_splits_pass:
        return False, f"wf_splits_pass:{splits} < {thr.min_wf_splits_pass}"
    if acc_delta < thr.min_accuracy_delta:
        return False, f"accuracy_delta:{acc_delta:.4f} < {thr.min_accuracy_delta:.4f}"
    return True, "ok"

