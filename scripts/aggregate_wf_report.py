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
    rsi2_split = wfdir / 'rsi2' / 'wf_splits.csv'
    ibs_split = wfdir / 'ibs' / 'wf_splits.csv'
    and_split = wfdir / 'and' / 'wf_splits.csv'
    crsi_split = wfdir / 'crsi' / 'wf_splits.csv'
    don_split = wfdir / 'donchian' / 'wf_splits.csv'
    topn_split = wfdir / 'topn' / 'wf_splits.csv'
    rsi2_df = pd.read_csv(rsi2_split) if rsi2_split.exists() else pd.DataFrame()
    ibs_df = pd.read_csv(ibs_split) if ibs_split.exists() else pd.DataFrame()
    and_df = pd.read_csv(and_split) if and_split.exists() else pd.DataFrame()
    crsi_df = pd.read_csv(crsi_split) if crsi_split.exists() else pd.DataFrame()
    don_df = pd.read_csv(don_split) if don_split.exists() else pd.DataFrame()
    topn_df = pd.read_csv(topn_split) if topn_split.exists() else pd.DataFrame()

    # Simple HTML assembly
    ts = datetime.utcnow().isoformat()
    parts = []
    parts.append('<html><head><meta charset="utf-8"><title>WF Report</title>')
    parts.append('<style>body{font-family:Arial, sans-serif;margin:20px} table{border-collapse:collapse;margin:10px 0} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3} h2{margin-top:30px}</style>')
    parts.append('</head><body>')
    parts.append(f'<h1>Walk-Forward Report</h1><p>Generated: {ts}</p>')

    # Dynamic heading reflecting presence of TOPN
    heading = 'Summary (RSI2 vs IBS vs AND)'
    if 'strategy' in compare_df.columns and any(compare_df['strategy'].astype(str).str.upper() == 'TOPN'):
        heading = 'Summary (RSI2 vs IBS vs AND vs TOPN)'
    parts.append(f'<h2>{heading}</h2>')
    parts.append(compare_df.to_html(index=False))

    if not rsi2_df.empty:
        parts.append('<h2>RSI-2 Per-Split Metrics</h2>')
        parts.append(rsi2_df.to_html(index=False))
    if not ibs_df.empty:
        parts.append('<h2>IBS Per-Split Metrics</h2>')
        parts.append(ibs_df.to_html(index=False))
    if not and_df.empty:
        parts.append('<h2>AND (RSI2+IBS) Per-Split Metrics</h2>')
        parts.append(and_df.to_html(index=False))
    if not crsi_df.empty:
        parts.append('<h2>CRSI Per-Split Metrics</h2>')
        parts.append(crsi_df.to_html(index=False))
    if not don_df.empty:
        parts.append('<h2>Donchian Breakout Per-Split Metrics</h2>')
        parts.append(don_df.to_html(index=False))
    if not topn_df.empty:
        parts.append('<h2>TOPN (Composite Scoring) Per-Split Metrics</h2>')
        parts.append(topn_df.to_html(index=False))

    # Note about net metrics if present in compare_df
    if any(c in compare_df.columns for c in ('net_pnl','net_pnl_total','gross_pnl','gross_pnl_total','total_fees','total_fees_total')):
        parts.append('<h2>Net Metrics Note</h2>')
        parts.append('<p>gross_pnl = PnL before commissions; net_pnl = PnL after commissions; total_fees = sum of all commissions paid.</p>')

    parts.append('</body></html>')
    out.write_text('\n'.join(parts), encoding='utf-8')
    print(f'Wrote report: {out}')


if __name__ == '__main__':
    main()
