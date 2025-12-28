#!/usr/bin/env python3
from __future__ import annotations

"""
Interview Quick Test for Kobe81 Traderbot

Runs a fast, reproducible walk-forward over a small universe and
emits a concise JSON/HTML summary for interview use.

Outputs:
- wf_outputs_interview_quick/wf_summary_compare.csv
- wf_outputs_interview_quick/wf_report.html
- INTERVIEW_SUMMARY.json
"""

import argparse
import json
import os
from datetime import date, timedelta
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def pick_universe() -> Path:
    """Choose the best available universe CSV in priority order."""
    candidates = [
        ROOT / "data/universe/optionable_liquid_900.csv",
        ROOT / "data/universe/optionable_liquid_900.csv",
        ROOT / "data/universe/optionable_liquid_candidates.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No universe CSV found in data/universe/")


def run_cmd(args: list[str]) -> None:
    proc = subprocess.run(args, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    ap = argparse.ArgumentParser(description="Interview quick test runner")
    ap.add_argument("--dotenv", type=str, default="./.env", help="Path to .env with API keys")
    ap.add_argument("--cap", type=int, default=50, help="Universe cap for quick test")
    ap.add_argument("--months", type=int, default=12, help="Lookback months for the window")
    ap.add_argument("--train-days", type=int, default=126, help="WF train days (~6 months)")
    ap.add_argument("--test-days", type=int, default=63, help="WF test days (~1 quarter)")
    ap.add_argument("--outdir", type=str, default="wf_outputs_interview_quick", help="Output directory")
    args = ap.parse_args()

    # Resolve inputs
    dotenv = str(Path(args.dotenv))
    universe = pick_universe()
    end = date.today()
    start = end - timedelta(days=int(args.months * 30.5))

    # 1) Run WF over small universe
    wf_cmd = [
        sys.executable,
        str(ROOT / "scripts/run_wf_polygon.py"),
        "--universe", str(universe),
        "--start", start.isoformat(),
        "--end", end.isoformat(),
        "--train-days", str(args.train_days),
        "--test-days", str(args.test_days),
        "--cap", str(args.cap),
        "--outdir", args.outdir,
        "--cache", "data/cache",
        "--dotenv", dotenv,
        "--regime-on",
    ]
    print("[1/3] Running walk-forward quick test...")
    run_cmd(wf_cmd)

    # 2) Aggregate to HTML report
    print("[2/3] Generating HTML report...")
    rep_cmd = [
        sys.executable,
        str(ROOT / "scripts/aggregate_wf_report.py"),
        "--wfdir", args.outdir,
        "--out", f"{args.outdir}/wf_report.html",
    ]
    run_cmd(rep_cmd)

    # 3) Summarize to JSON for easy sharing
    print("[3/3] Writing interview summary...")
    summary_csv = ROOT / args.outdir / "wf_summary_compare.csv"
    df = pd.read_csv(summary_csv) if summary_csv.exists() else pd.DataFrame()
    # Keep key rows if present
    keep = ["IBS_RSI", "TURTLE_SOUP"]
    if not df.empty and "strategy" in df.columns:
        df = df[df["strategy"].astype(str).isin(keep)]
    summary = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "cap": args.cap,
        "train_days": args.train_days,
        "test_days": args.test_days,
        "universe_file": str(universe.relative_to(ROOT)),
        "wf_outdir": args.outdir,
        "report_html": f"{args.outdir}/wf_report.html",
        "kpis": df.to_dict(orient="records") if not df.empty else [],
    }
    (ROOT / "INTERVIEW_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done. Artifacts:")
    print(f"- {summary['report_html']}")
    print(f"- {args.outdir}/wf_summary_compare.csv")
    print(f"- INTERVIEW_SUMMARY.json")


if __name__ == "__main__":
    main()
