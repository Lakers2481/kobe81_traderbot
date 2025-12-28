from __future__ import annotations

"""
Compliance Rules Engine (optional helper).

Centralizes basic trading rules:
- Max position size (as fraction of equity)
- PDT rule placeholder (pattern day trader)
- No penny stocks (min price)
- Trading hours only (RTH)

This module is not wired into the production path by default; it can be used by
submitters or scanners for extra guardrails where needed.
"""

from dataclasses import dataclass
from typing import Tuple
from datetime import datetime, time as dtime


@dataclass
class RuleConfig:
    max_position_size_frac: float = 0.10  # 10% of equity
    min_price: float = 3.0                # avoid penny stocks
    pdt_enabled: bool = False
    rth_only: bool = True
    rth_start: dtime = dtime(9, 30)
    rth_end: dtime = dtime(16, 0)


def _within_rth(ts: datetime, start: dtime, end: dtime) -> bool:
    t = dtime(ts.hour, ts.minute)
    return (t >= start) and (t <= end)


def evaluate(
    price: float,
    qty: int,
    account_equity: float,
    is_day_trade: bool = False,
    ts: datetime | None = None,
    cfg: RuleConfig | None = None,
) -> Tuple[bool, str]:
    cfg = cfg or RuleConfig()
    # Price floor (no penny stocks)
    if price < cfg.min_price:
        return False, f"min_price:{price:.2f} < {cfg.min_price:.2f}"
    # Max position size fraction
    notional = float(price) * int(qty)
    max_notional = float(cfg.max_position_size_frac) * float(account_equity)
    if notional > max_notional:
        return False, f"max_position_size_exceeded:{notional:.2f} > {max_notional:.2f}"
    # PDT placeholder
    if cfg.pdt_enabled and is_day_trade:
        # Enforcement of PDT capital rules would query broker status and day-trade counts
        # Here we flag as a warning condition instead of a hard veto.
        return False, "pdt_day_trade_restricted"
    # RTH only
    if cfg.rth_only and ts is not None and not _within_rth(ts, cfg.rth_start, cfg.rth_end):
        return False, "outside_regular_trading_hours"
    return True, "ok"

