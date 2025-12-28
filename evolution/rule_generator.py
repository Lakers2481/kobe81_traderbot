from __future__ import annotations

"""
Rule generator templates (optional).

Generates simple, declarative rule dicts for entry/exit to support research.
"""

from typing import Dict, Any


def generate(template: str, base: Dict[str, Any] | None = None) -> Dict[str, Any]:
    base = base or {}
    t = template.lower()
    # NOTE: Donchian breakout strategy deprecated - use ibs_rsi or turtle_soup instead
    # if t in ("donchian", "donchian_breakout"):
    #     return {...}  # Removed - strategy no longer in production
    if t in ("ibs_rsi", "ibs", "rsi2", "ibs_rsi_mean_reversion"):
        return {
            "name": "ibs_rsi",
            "entry": {"type": "ibs_rsi_combo", "ibs_max": float(base.get("ibs_max", 0.15)), "rsi_max": float(base.get("rsi_max", 10.0)), "trend_filter_sma": 200},
            "exit": {
                "stop": {"type": "atr", "n": 14, "multiple": float(base.get("atr_mult", 1.5))},
                "time": {"bars": int(base.get("time_stop_bars", 5))},
            },
            "min_price": float(base.get("min_price", 5.0)),
        }
    if t in ("ict", "turtle_soup", "ict_turtle_soup"):
        return {
            "name": "ict_turtle_soup",
            "entry": {"type": "liquidity_sweep", "lookback": int(base.get("lookback", 20)), "trend_filter_sma": 200},
            "exit": {
                "stop": {"type": "atr", "n": 14, "multiple": float(base.get("atr_mult", 2.0))},
                "time": {"bars": int(base.get("time_stop_bars", 5))},
            },
            "min_price": float(base.get("min_price", 3.0)),
        }
    return {"name": "unknown", "entry": {}, "exit": {}}
