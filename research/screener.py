from __future__ import annotations

"""
Quick feature/alpha screener (research only).

Computes feature correlations with forward returns and writes a summary CSV for
triage. Not used by production scanner.
"""

from pathlib import Path
from typing import Tuple, List
import pandas as pd

from research.features import compute_research_features
from research.alphas import compute_alphas


def _forward_return(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    c = df["close"].astype(float)
    return (c.shift(-horizon) / c - 1.0).astype(float)


def screen_universe(ohlcv: pd.DataFrame, horizons: Tuple[int, ...] = (5, 10)) -> pd.DataFrame:
    """Return a summary DataFrame with correlations by feature/alpha and horizon.

    ohlcv columns: ['timestamp','symbol','open','high','low','close','volume']
    """
    if ohlcv.empty:
        return pd.DataFrame()

    feats = compute_research_features(ohlcv)
    alphas = compute_alphas(ohlcv)
    merged = (ohlcv[["timestamp","symbol","close"]]
              .merge(feats, on=["timestamp","symbol"], how="left")
              .merge(alphas, on=["timestamp","symbol"], how="left"))
    merged = merged.sort_values(["symbol","timestamp"]).reset_index(drop=True)

    # Build forward returns per horizon
    rows: List[dict] = []
    for h in horizons:
        # Compute forward returns per symbol and flatten to series
        fwd_ret_list = []
        for sym, g in merged.groupby("symbol"):
            fwd = _forward_return(g, h)
            fwd_ret_list.append(fwd)
        merged[f"fwd_ret_{h}"] = pd.concat(fwd_ret_list).reindex(merged.index)
        # Correlation by column
        numeric_cols = [c for c in merged.columns if c not in ("timestamp","symbol")]
        corr = merged[numeric_cols].corr(method="spearman")
        # Extract correlations to fwd return target
        tgt = f"fwd_ret_{h}"
        for col in [c for c in numeric_cols if c != tgt and not c.startswith("fwd_ret_")]:
            val = float(corr.loc[col, tgt]) if (col in corr.index and tgt in corr.columns) else 0.0
            rows.append({"feature": col, "horizon": h, "spearman": round(val, 4)})

    out = pd.DataFrame(rows).sort_values(["horizon","spearman"], ascending=[True, False]).reset_index(drop=True)
    return out


def save_screening_report(df: pd.DataFrame, outdir: Path = Path("outputs/research")) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "screening_summary.csv"
    df.to_csv(p, index=False)
    return p

