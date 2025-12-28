#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description='Aggregate WF outputs into a single HTML report')
    ap.add_argument('--wfdir', type=str, default='wf_outputs', help='Directory containing WF outputs')
    ap.add_argument('--out', type=str, default='wf_outputs/wf_report.html', help='Output HTML file')
    args = ap.parse_args()

    wfdir = Path(args.wfdir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load side-by-side summary
    summary_path = wfdir / 'wf_summary_compare.csv'
    if not summary_path.exists():
        raise FileNotFoundError(f'Missing {summary_path}')
    compare_df = pd.read_csv(summary_path)

    # Load per-split details if present
    don_split = wfdir / 'ibs_rsi' / 'wf_splits.csv'
    ts_split = wfdir / 'turtle_soup' / 'wf_splits.csv'
    don_df = pd.read_csv(don_split) if don_split.exists() else pd.DataFrame()
    ts_df = pd.read_csv(ts_split) if ts_split.exists() else pd.DataFrame()

    # Simple HTML assembly
    ts = datetime.utcnow().isoformat()
    parts = []
    parts.append('<html><head><meta charset="utf-8"><title>WF Report</title>')
    parts.append('<style>body{font-family:Arial, sans-serif;margin:20px} table{border-collapse:collapse;margin:10px 0} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3} h2{margin-top:30px}</style>')
    parts.append('</head><body>')
    parts.append(f'<h1>Walk-Forward Report</h1><p>Generated: {ts}</p>')

    # Dynamic heading reflecting presence of TOPN
    heading = 'Summary (IBS+RSI vs ICT Turtle Soup)'
    parts.append(f'<h2>{heading}</h2>')
    parts.append(compare_df.to_html(index=False))

    if not don_df.empty:
        parts.append('<h2>IBS+RSI Per-Split Metrics</h2>')
        parts.append(don_df.to_html(index=False))
    if not ts_df.empty:
        parts.append('<h2>Turtle Soup (ICT Liquidity Sweep) Per-Split Metrics</h2>')
        parts.append(ts_df.to_html(index=False))

    # Note about net metrics if present in compare_df
    if any(c in compare_df.columns for c in ('net_pnl','net_pnl_total','gross_pnl','gross_pnl_total','total_fees','total_fees_total')):
        parts.append('<h2>Net Metrics Note</h2>')
        parts.append('<p>gross_pnl = PnL before commissions; net_pnl = PnL after commissions; total_fees = sum of all commissions paid.</p>')

    parts.append('</body></html>')
    out.write_text('\n'.join(parts), encoding='utf-8')
    print(f'Wrote report: {out}')


if __name__ == '__main__':
    main()
