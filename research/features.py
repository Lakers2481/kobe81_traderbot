from __future__ import annotations

"""
Research features (optional, not part of production pipeline).

Computes a set of technical/statistical features on OHLCV data suitable for
screening and model experimentation. These are intentionally separate from
ml_meta.features (the production feature set).
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


@dataclass
class FeatureSpec:
    name: str
    description: str


FEATURE_SPECS: List[FeatureSpec] = [
    FeatureSpec("bb_width20", "Bollinger band width (20, 2σ)"),
    FeatureSpec("macd_hist", "MACD histogram (12,26,9)"),
    FeatureSpec("keltner_width20", "Keltner channel width (20)"),
    FeatureSpec("adx14", "Average Directional Index (14)"),
    FeatureSpec("obv", "On-Balance Volume"),
]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Safely divide two series, handling zero denominator."""
    out = a.astype(float).copy()
    denom = b.astype(float).replace(0.0, float('nan'))
    result = out / denom
    return result.fillna(0.0).astype(float)


def compute_research_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute research features per (symbol,timestamp).

    Input columns required: ['timestamp','symbol','open','high','low','close','volume'].
    Returns a DataFrame with ['timestamp','symbol',<features>].
    """
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", *[f.name for f in FEATURE_SPECS]])

    def _by_symbol(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp").copy()
        c, h, l, v = g["close"].astype(float), g["high"].astype(float), g["low"].astype(float), g["volume"].astype(float)

        # Bollinger width (20,2σ)
        m20 = c.rolling(20, min_periods=20).mean()
        s20 = c.rolling(20, min_periods=20).std()
        upper = m20 + 2.0 * s20
        lower = m20 - 2.0 * s20
        bb_width20 = (upper - lower).fillna(0.0)

        # MACD histogram (12,26,9)
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = (macd - signal).fillna(0.0)

        # Keltner channel width (20): EMA20 ± ATR(mult=2)
        ema20 = c.ewm(span=20, adjust=False).mean()
        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr20 = tr.rolling(20, min_periods=20).mean()
        keltner_upper = ema20 + 2.0 * atr20
        keltner_lower = ema20 - 2.0 * atr20
        keltner_width20 = (keltner_upper - keltner_lower).fillna(0.0)

        # ADX(14) (approx via DIs)
        up_move = h.diff().astype(float)
        down_move = (-l.diff()).astype(float)
        plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move
        tr14 = tr.astype(float).rolling(14, min_periods=14).sum()
        pdi14 = 100 * _safe_div(plus_dm.rolling(14, min_periods=14).sum(), tr14)
        mdi14 = 100 * _safe_div(minus_dm.rolling(14, min_periods=14).sum(), tr14)
        di_sum = (pdi14 + mdi14).replace(0.0, float('nan'))
        dx = (100 * (pdi14 - mdi14).abs() / di_sum).fillna(0.0).astype(float)
        adx14 = dx.rolling(14, min_periods=14).mean().fillna(0.0).astype(float)

        # OBV
        ret = c.diff().fillna(0.0)
        obv = (v * ret.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))).cumsum().fillna(0.0)

        out = pd.DataFrame({
            "timestamp": g["timestamp"],
            "symbol": g["symbol"],
            "bb_width20": bb_width20,
            "macd_hist": macd_hist,
            "keltner_width20": keltner_width20,
            "adx14": adx14,
            "obv": obv,
        })
        for col in ["bb_width20","macd_hist","keltner_width20","adx14","obv"]:
            out[col] = out[col].astype(float).fillna(0.0)
        return out

    parts = [_by_symbol(g) for _, g in df.groupby("symbol")]
    return pd.concat(parts, ignore_index=True)

