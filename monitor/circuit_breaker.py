from __future__ import annotations

"""
Circuit breaker helpers (optional).

Provides simple gating based on operational thresholds:
- Max daily loss (absolute $)
- Max consecutive losses
- Max error count

Callers can use evaluate() before placing orders to decide whether to pause.
This is not wired into production by default.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CircuitConfig:
    max_daily_loss: float = 1000.0
    max_consecutive_losses: int = 5
    max_error_count: int = 10


def evaluate(daily_pnl: float, consecutive_losses: int, error_count: int, cfg: CircuitConfig | None = None) -> Tuple[bool, str]:
    cfg = cfg or CircuitConfig()
    # negative PnL (loss) beyond threshold is a halt
    if daily_pnl <= -abs(cfg.max_daily_loss):
        return False, f"daily_loss_exceeded:{daily_pnl:.2f} <= -{cfg.max_daily_loss:.2f}"
    if consecutive_losses >= int(cfg.max_consecutive_losses):
        return False, f"consecutive_losses_exceeded:{consecutive_losses} >= {cfg.max_consecutive_losses}"
    if error_count >= int(cfg.max_error_count):
        return False, f"error_threshold_exceeded:{error_count} >= {cfg.max_error_count}"
    return True, "ok"

