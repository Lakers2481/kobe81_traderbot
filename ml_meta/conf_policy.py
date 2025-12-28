from __future__ import annotations

"""
Dynamic confidence / budget policy.

Computes min_conf adjustment and a budget multiplier based on regime and volatility.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_spy(start: str, end: str) -> pd.DataFrame:
    from data.providers.polygon_eod import fetch_daily_bars_polygon
    return fetch_daily_bars_polygon('SPY', start, end, cache_dir=ROOT / 'data' / 'cache')


def compute(min_conf_base: float = 0.60) -> Tuple[float, float]:
    """Return (effective_min_conf, budget_mult)."""
    try:
        end = datetime.utcnow().date()
        start = (end - timedelta(days=300)).isoformat()
        df = _load_spy(start, end.isoformat()).sort_values('timestamp')
        if df.empty or len(df) < 220:
            return min_conf_base, 1.0
        df['sma200'] = df['close'].rolling(200, min_periods=200).mean()
        df['rv20'] = df['close'].pct_change().rolling(20, min_periods=20).std()
        bull = bool(df['close'].iloc[-1] >= df['sma200'].iloc[-1])
        vol = float(df['rv20'].iloc[-1] or 0.0)
        eff = min_conf_base
        budget = 1.0
        # Simple policy: cautious when bear or high vol
        if (not bull) or vol >= 0.02:
            eff = min(min_conf_base + 0.05, 0.80)
            budget = 0.8
        elif bull and vol <= 0.012:
            eff = max(min_conf_base - 0.05, 0.55)
            budget = 1.1
        return eff, budget
    except Exception:
        return min_conf_base, 1.0

