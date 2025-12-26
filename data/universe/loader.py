from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def load_universe(path: str | Path, cap: Optional[int] = None) -> List[str]:
    """
    Load a universe of symbols from CSV.
    - Expects a column named 'symbol'; if absent, uses the first column.
    - Uppercases symbols, drops NA/empty, preserves order, applies optional cap.
    """
    p = Path(path)
    if not p.exists():
        return []

    df = pd.read_csv(p)
    col = 'symbol'
    if col not in df.columns:
        # fallback to first column
        col = df.columns[0]
    series = df[col].astype(str).str.strip().str.upper()
    symbols = [s for s in series.tolist() if s]
    # de-dup while preserving order
    seen = set()
    uniq: List[str] = []
    for s in symbols:
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    if cap is not None:
        uniq = uniq[: max(0, int(cap))]
    return uniq

