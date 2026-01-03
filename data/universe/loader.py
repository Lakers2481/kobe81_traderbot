from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

# =============================================================================
# CANONICAL 900-STOCK UNIVERSE - SINGLE SOURCE OF TRUTH
# =============================================================================
# This is THE universe for all backtesting, scanning, analysis, and reports.
# Sorted by liquidity/volume. All stocks have options. All have historical data.
# =============================================================================

CANONICAL_UNIVERSE_FILE = Path(__file__).parent / "optionable_liquid_900.csv"
EXPECTED_COUNT = 900
MAGNIFICENT_7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']


def load_universe(path: str | Path = None, cap: Optional[int] = None) -> List[str]:
    """
    Load universe of symbols from CSV.

    If path is None, loads the CANONICAL 900-stock universe.

    Args:
        path: Path to universe CSV. If None, uses canonical 900-stock universe.
        cap: Optional limit on number of stocks to return.

    Returns:
        List of stock symbols, sorted by liquidity.
    """
    # Use canonical universe if no path specified
    if path is None:
        path = CANONICAL_UNIVERSE_FILE

    p = Path(path)
    if not p.exists():
        return []

    df = pd.read_csv(p)
    col = 'symbol'
    if col not in df.columns:
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


def load_canonical_900() -> List[str]:
    """
    Load the canonical 900-stock universe with validation.

    This is the ONLY function that should be used for production.
    Raises ValueError if universe is invalid.

    Returns:
        List of exactly 900 stock symbols, sorted by liquidity.
    """
    if not CANONICAL_UNIVERSE_FILE.exists():
        raise ValueError(f"Canonical universe not found: {CANONICAL_UNIVERSE_FILE}")

    symbols = load_universe(CANONICAL_UNIVERSE_FILE)

    if len(symbols) != EXPECTED_COUNT:
        raise ValueError(f"Universe must have {EXPECTED_COUNT} stocks, found {len(symbols)}")

    missing = [s for s in MAGNIFICENT_7 if s not in symbols]
    if missing:
        raise ValueError(f"Missing Magnificent 7: {missing}")

    return symbols


def get_canonical_path() -> Path:
    """Return path to the canonical universe file."""
    return CANONICAL_UNIVERSE_FILE

