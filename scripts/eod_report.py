#!/usr/bin/env python3
from __future__ import annotations

"""
Generate a simple End-of-Day (EOD) report combining Top-3, TOTD, and WF summaries.

Outputs: reports/eod_report_YYYYMMDD.html
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.clock.tz_utils import now_et, fmt_ct
import pandas as pd


def html_table(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"<h2>{title}</h2><p>No data.</p>"
    return f"<h2>{title}</h2>" + df.to_html(index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description='Generate EOD HTML report')
    ap.add_argument('--wfdir', type=str, default='wf_outputs')
    ap.add_argument('--outdir', type=str, default='reports')
    args = ap.parse_args()

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    # Internal ET, display CT 12-hour in header
    now = now_et()
    date_tag = now.strftime('%Y%m%d')
    out = outdir / f'eod_report_{date_tag}.html'

    # Load daily picks and TOTD
    picks = None
    totd = None
    p1 = ROOT / 'logs' / 'daily_picks.csv'
    p2 = ROOT / 'logs' / 'trade_of_day.csv'
    if p1.exists():
        try:
            picks = pd.read_csv(p1)
        except Exception:
            picks = None
    if p2.exists():
        try:
            totd = pd.read_csv(p2)
        except Exception:
            totd = None

    # Load WF compare summary if available
    wf = None
    wfp = ROOT / args.wfdir / 'wf_summary_compare.csv'
    if wfp.exists():
        try:
            wf = pd.read_csv(wfp)
        except Exception:
            wf = None

    html_parts = [
        '<html><head><meta charset="utf-8"><title>Kobe EOD Report</title>',
        '<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3}</style>',
        '</head><body>',
        f'<h1>Kobe EOD Report - {date_tag}</h1>',
        f'<p><em>Display: {fmt_ct(now)} | {now.strftime("%I:%M %p").lstrip("0")} ET (12-hour); operations run in ET.</em></p>',
        html_table(picks, 'Top-3 Picks'),
        html_table(totd, 'Trade of the Day'),
        html_table(wf, 'Walk-Forward Summary (IBS+RSI vs ICT)'),
        '</body></html>'
    ]
    out.write_text('\n'.join(html_parts), encoding='utf-8')
    print('EOD report written:', out)


if __name__ == '__main__':
    main()
