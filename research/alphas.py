from __future__ import annotations

"""
Candidate alphas for research (optional).

Implements simple, interpretable alpha signals for screening:
- Momentum 20d
- One-day reversal
- Gap close tendency
"""

from typing import Dict, Callable
import numpy as np
import pandas as pd


# Registry of available alphas for screening
ALPHA_REGISTRY: Dict[str, Callable] = {
    'alpha_mom20': lambda df: compute_alphas(df)[['timestamp', 'symbol', 'alpha_mom20']],
    'alpha_rev1': lambda df: compute_alphas(df)[['timestamp', 'symbol', 'alpha_rev1']],
    'alpha_gap_close': lambda df: compute_alphas(df)[['timestamp', 'symbol', 'alpha_gap_close']],
}


def compute_alphas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute alpha candidates per (symbol,timestamp).

    Input requires: ['timestamp','symbol','open','high','low','close','volume']
    Returns DataFrame with ['timestamp','symbol','alpha_mom20','alpha_rev1','alpha_gap_close']
    """
    if df.empty:
        return pd.DataFrame(columns=["timestamp","symbol","alpha_mom20","alpha_rev1","alpha_gap_close"])

    def _by_symbol(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp").copy()
        c = g["close"].astype(float)
        o = g["open"].astype(float)
        prev_c = c.shift(1)

        # Momentum 20d (log return)
        alpha_mom20 = (c / c.shift(20)).apply(lambda x: 0.0 if pd.isna(x) or x <= 0 else float(np.log(x)))

        # One-day reversal (negative return yesterday)
        r1 = c.pct_change()
        alpha_rev1 = (-r1).fillna(0.0)

        # Gap close: (prev close → open) sign times intraday move
        gap = (o - prev_c).fillna(0.0)
        intraday = (c - o).fillna(0.0)
        alpha_gap_close = (gap.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)) * intraday).fillna(0.0)

        out = pd.DataFrame({
            "timestamp": g["timestamp"],
            "symbol": g["symbol"],
            "alpha_mom20": alpha_mom20.astype(float).fillna(0.0),
            "alpha_rev1": alpha_rev1.astype(float).fillna(0.0),
            "alpha_gap_close": alpha_gap_close.astype(float).fillna(0.0),
        })
        return out

    parts = [_by_symbol(g) for _, g in df.groupby("symbol")]
    return pd.concat(parts, ignore_index=True)

