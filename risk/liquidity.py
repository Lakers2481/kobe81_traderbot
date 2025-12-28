from __future__ import annotations

"""
Liquidity-related gates (optional helpers).

These helpers centralize simple checks used by scanners/submitters:
- Minimum 60-day ADV in USD
- Maximum bid/ask spread percentage

Production submitters already enforce spread gates; this module exists to make
the logic reusable and testable if needed.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LiquidityConfig:
    min_adv_usd: float = 5_000_000.0  # default $5M
    max_spread_pct: float = 0.02      # default 2%


def is_adv_sufficient(adv_usd60: Optional[float], cfg: LiquidityConfig | None = None) -> bool:
    cfg = cfg or LiquidityConfig()
    try:
        return float(adv_usd60 or 0.0) >= float(cfg.min_adv_usd)
    except Exception:
        return False


def is_spread_ok(bid: Optional[float], ask: Optional[float], cfg: LiquidityConfig | None = None) -> Tuple[bool, float]:
    cfg = cfg or LiquidityConfig()
    try:
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return False, 1.0
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return False, 1.0
        spread = (ask - bid) / mid
        return spread <= float(cfg.max_spread_pct), float(spread)
    except Exception:
        return False, 1.0

