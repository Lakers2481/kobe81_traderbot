#!/usr/bin/env python3
from __future__ import annotations

"""
Testing: Monte Carlo wrapper CLI.

Thin CLI around backtest.monte_carlo.run_monte_carlo_analysis for convenience.
"""

import argparse
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.monte_carlo import run_monte_carlo_analysis


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Monte Carlo analysis against a trades CSV")
    ap.add_argument("--trades", type=str, required=True, help="Path to trades CSV (must include pnl or pnl_pct)")
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--initial", type=float, default=100000.0)
    ap.add_argument("--out", type=str, default=str(ROOT / "outputs" / "monte_carlo_summary.txt"))
    args = ap.parse_args()

    df = pd.read_csv(Path(args.trades))
    res = run_monte_carlo_analysis(df, n_simulations=args.sims, initial_capital=args.initial)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(res["summary"], encoding="utf-8")
    print("Wrote:", outp)


if __name__ == "__main__":
    main()

