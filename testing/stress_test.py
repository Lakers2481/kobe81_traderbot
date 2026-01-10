#!/usr/bin/env python3
from __future__ import annotations

"""
Testing: Stress scenarios for equity paths or returns.

Provides canned scenarios:
- black_monday: single-day -22%
- covid_crash: multi-day cumulative -34%
- vix_spike: double volatility for N days
- flash_crash: intraday-like -9% shock (approximated as EOD drop)

Outputs a simple summary of final equity and drawdown impact.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def apply_shock_returns(returns: np.ndarray, shock_idx: int, shock_drop: float) -> np.ndarray:
    out = returns.copy()
    if 0 <= shock_idx < len(out):
        out[shock_idx] = (1 + out[shock_idx]) * (1 + shock_drop) - 1
    return out


def scale_volatility(returns: np.ndarray, start_idx: int, days: int, scale: float) -> np.ndarray:
    out = returns.copy()
    end_idx = min(len(out), start_idx + days)
    out[start_idx:end_idx] = out[start_idx:end_idx] * scale
    return out


def equity_from_returns(returns: np.ndarray, initial: float) -> np.ndarray:
    eq = np.empty(len(returns) + 1)
    eq[0] = initial
    for i, r in enumerate(returns):
        eq[i + 1] = eq[i] * (1 + r)
    return eq


def max_drawdown_pct(equity: np.ndarray) -> float:
    peak = equity[0]
    max_dd = 0.0
    for x in equity:
        if x > peak:
            peak = x
        dd = (peak - x) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd * 100


def run_scenarios(returns: np.ndarray, initial: float = 100000.0) -> pd.DataFrame:
    scenarios = []
    n = len(returns)
    mid = n // 2 if n > 0 else 0

    # Black Monday: -22% at midpoint
    r1 = apply_shock_returns(returns, mid, -0.22)
    e1 = equity_from_returns(r1, initial)
    scenarios.append({"scenario": "black_monday", "final_equity": float(e1[-1]), "max_dd": max_drawdown_pct(e1)})

    # COVID Crash: spread -34% over 10 days
    spread = (-0.34) / 10.0
    r2 = returns.copy()
    for i in range(10):
        idx = min(n - 1, mid + i)
        r2[idx] = (1 + r2[idx]) * (1 + spread) - 1
    e2 = equity_from_returns(r2, initial)
    scenarios.append({"scenario": "covid_crash", "final_equity": float(e2[-1]), "max_dd": max_drawdown_pct(e2)})

    # VIX Spike: double volatility for 5 days
    r3 = scale_volatility(returns, max(0, mid - 2), 5, 2.0)
    e3 = equity_from_returns(r3, initial)
    scenarios.append({"scenario": "vix_spike_2x", "final_equity": float(e3[-1]), "max_dd": max_drawdown_pct(e3)})

    # Flash Crash: -9% single-day shock
    r4 = apply_shock_returns(returns, mid, -0.09)
    e4 = equity_from_returns(r4, initial)
    scenarios.append({"scenario": "flash_crash", "final_equity": float(e4[-1]), "max_dd": max_drawdown_pct(e4)})

    return pd.DataFrame(scenarios)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stress scenarios on returns series")
    ap.add_argument("--returns", type=str, help="CSV with a 'ret' column OR trades with pnl_pct")
    ap.add_argument("--initial", type=float, default=100000.0)
    ap.add_argument("--out", type=str, default="outputs/stress_summary.csv")
    args = ap.parse_args()

    p = Path(args.returns)
    df = pd.read_csv(p)
    if 'ret' in df.columns:
        rets = df['ret'].astype(float).values
    elif 'pnl_pct' in df.columns:
        rets = df['pnl_pct'].astype(float).values
    else:
        raise SystemExit("Input CSV must have 'ret' or 'pnl_pct' column")

    summary = run_scenarios(rets, initial=args.initial)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outp, index=False)
    print("Wrote:", outp)


if __name__ == "__main__":
    main()

