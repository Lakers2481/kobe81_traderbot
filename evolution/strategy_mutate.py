from __future__ import annotations

"""
Strategy parameter mutation helpers (optional).
"""

from typing import Dict, Tuple
import random
import math

Params = Dict[str, float]
Bounds = Dict[str, Tuple[float, float]]


def mutate_params(params: Params, bounds: Bounds, sigma: float = 0.1, attempts: int = 10) -> Params:
    """Gaussian mutate a parameter dict within bounds.

    Ensures the mutated dict differs from the original (clone protection) up to
    a fixed number of attempts.
    """
    def _clip(x: Params) -> Params:
        out = {}
        for k, v in x.items():
            lo, hi = bounds.get(k, (-math.inf, math.inf))
            out[k] = min(max(v, lo), hi)
        return out

    for _ in range(attempts):
        out = params.copy()
        for k in out.keys():
            span = max(1e-9, bounds[k][1] - bounds[k][0])
            out[k] += random.gauss(0.0, sigma * span)
        out = _clip(out)
        if any(abs(out[k] - params[k]) > 1e-9 for k in out.keys()):
            return out
    return _clip({k: v + 1e-6 for k, v in params.items()})

