from __future__ import annotations

"""
Narrative generation for signals/trades (optional).

Returns short technical/casual/executive narratives based on a signal dict.
"""

from typing import Dict


def generate(signal: Dict) -> Dict[str, str]:
    sym = str(signal.get("symbol", "?"))
    strat = str(signal.get("strategy", "")).upper()
    side = str(signal.get("side", "?")).upper()
    entry = signal.get("entry_price")
    stop = signal.get("stop_loss")
    take = signal.get("take_profit")

    tech = f"{strat}: {sym} {side} at {entry}, stop {stop}, target {take}. Risk‑managed with time/ATR stops."
    casual = f"Picking {sym} on a {strat} setup. Plan to enter near {entry} with a protective stop at {stop}."
    execu = f"{strat} signal: disciplined entry/exit with budget, spread, liquidity gates and earnings/regime filters."
    return {
        "technical": tech,
        "casual": casual,
        "executive": execu,
    }

