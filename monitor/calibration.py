from __future__ import annotations

"""
Calibration utilities for classification probabilities.

Functions here are optional helpers used by reports or research. The
production morning_report computes simple reliability tables inline; this
module provides reusable pieces.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple, Dict
import pandas as pd
import numpy as np


def brier_score(y_true: Iterable[int], p_pred: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(p_pred), dtype=float)
    if y.size == 0:
        return float("nan")
    p = np.clip(p, 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def reliability_table(y_true: Iterable[int], p_pred: Iterable[float], bins: int = 10) -> pd.DataFrame:
    """
    Compute a simple reliability table with equal-width bins.
    Returns columns: ['bin','count','pred_mean','obs_rate']
    """
    y = pd.Series(list(y_true), dtype=float)
    p = pd.Series(list(p_pred), dtype=float).clip(0, 1)
    if len(y) == 0:
        return pd.DataFrame(columns=["bin","count","pred_mean","obs_rate"])  # empty
    cat = pd.cut(p, bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    df = pd.DataFrame({"bin": cat, "y": y, "p": p})
    grp = df.groupby("bin").agg(count=("y","size"), pred_mean=("p","mean"), obs_rate=("y","mean"))
    return grp.reset_index()

