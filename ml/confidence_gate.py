from __future__ import annotations

"""
Confidence gate helper for approving Trade Of The Day (TOTD) or signals.

NOTE: The production scanner and submitters already implement --min-conf logic.
This helper is provided for future centralization and reuse.
"""

from dataclasses import dataclass


@dataclass
class GateConfig:
    min_conf: float = 0.60


def approve(conf: float, cfg: GateConfig | None = None) -> bool:
    cfg = cfg or GateConfig()
    try:
        return float(conf) >= float(cfg.min_conf)
    except Exception:
        return False

