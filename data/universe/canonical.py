"""
CANONICAL 900-STOCK UNIVERSE - SINGLE SOURCE OF TRUTH

This is the ONLY universe loader that should be used across the entire codebase.
All backtesting, scanning, analysis, and reports MUST use this.

DO NOT create alternative universe files or loaders.
"""

import pandas as pd
from pathlib import Path
from typing import List

# THE CANONICAL UNIVERSE FILE - DO NOT CHANGE
CANONICAL_UNIVERSE_FILE = Path(__file__).parent / "optionable_liquid_800.csv"
EXPECTED_COUNT = 900

# Magnificent 7 - must always be included
MAGNIFICENT_7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']


def load_canonical_universe() -> List[str]:
    """
    Load the canonical 900-stock universe.

    This is the ONLY function that should be used to load the universe.
    It validates that:
    1. Exactly 800 stocks are present
    2. All Magnificent 7 are included
    3. File exists and is readable

    Returns:
        List of 900 stock symbols, sorted by liquidity

    Raises:
        ValueError: If universe is invalid
    """
    if not CANONICAL_UNIVERSE_FILE.exists():
        raise ValueError(f"Canonical universe file not found: {CANONICAL_UNIVERSE_FILE}")

    df = pd.read_csv(CANONICAL_UNIVERSE_FILE)
    symbols = df['symbol'].tolist()

    # Validate count
    if len(symbols) != EXPECTED_COUNT:
        raise ValueError(f"Universe must have exactly {EXPECTED_COUNT} stocks, found {len(symbols)}")

    # Validate Magnificent 7
    missing_mag7 = [s for s in MAGNIFICENT_7 if s not in symbols]
    if missing_mag7:
        raise ValueError(f"Missing Magnificent 7 stocks: {missing_mag7}")

    return symbols


def get_universe_stats() -> dict:
    """Get statistics about the canonical universe."""
    symbols = load_canonical_universe()

    return {
        "total_stocks": len(symbols),
        "file": str(CANONICAL_UNIVERSE_FILE),
        "magnificent_7": {s: symbols.index(s) + 1 for s in MAGNIFICENT_7},
        "top_10": symbols[:10],
    }


def validate_universe() -> bool:
    """Validate the canonical universe. Returns True if valid."""
    try:
        symbols = load_canonical_universe()
        print(f"✓ Universe valid: {len(symbols)} stocks")
        print("✓ Magnificent 7 all present")
        print(f"✓ Top 5: {', '.join(symbols[:5])}")
        return True
    except ValueError as e:
        print(f"✗ Universe INVALID: {e}")
        return False


if __name__ == "__main__":
    validate_universe()
