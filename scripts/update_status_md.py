#!/usr/bin/env python3
from __future__ import annotations

"""
Generate a living STATUS markdown and date-stamped history snapshot from journal + state.

Outputs:
- docs/STATUS.md (latest)
- docs/history/status_YYYYMMDD.md (snapshot)
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def summarize_journal(j: List[Dict[str, Any]], days: int = 7) -> Dict[str, Any]:
    if not j:
        return {'events_last_7d': {}, 'recent': []}
    now = datetime.utcnow()
    recent = []
    for rec in j:
        ts = rec.get('utc_ts') or rec.get('timestamp')
        try:
            dt = datetime.fromisoformat(ts) if isinstance(ts, str) else None
        except Exception:
            dt = None
        if dt and (now - dt) <= timedelta(days=days):
            recent.append(rec)
    events = Counter([r.get('event','unknown') for r in recent])
    return {'events_last_7d': dict(events), 'recent': recent}


def load_model_summary() -> Dict[str, Any]:
    dep = ROOT / 'state' / 'models' / 'deployed' / 'meta_current.json'
    if not dep.exists():
        return {}
    try:
        return json.loads(dep.read_text(encoding='utf-8'))
    except Exception:
        return {}


def render_status(today: datetime, journal_sum: Dict[str, Any]) -> str:
    date_tag = today.strftime('%Y-%m-%d')
    # Current artifacts
    picks = ROOT / 'logs' / 'daily_picks.csv'
    totd = ROOT / 'logs' / 'trade_of_day.csv'
    mrep = ROOT / 'reports' / f'morning_report_{today.strftime("%Y%m%d")}.html'
    mchk = ROOT / 'reports' / 'morning_check.json'
    eod = ROOT / 'reports' / f'eod_report_{today.strftime("%Y%m%d")}.html'
    model = load_model_summary()

    lines: List[str] = []
    lines.append(f"# Kobe81 Status — {date_tag}")
    lines.append("")
    lines.append("## Overview")
    lines.append("- Strategies: Donchian Breakout (trend) + ICT Turtle Soup (mean reversion)")
    lines.append("- Universe: 900 optionable/liquid US equities, 10y coverage")
    lines.append("- Decisioning: ML meta-model + sentiment blending; confidence-gated TOTD")
    lines.append("")
    lines.append("## Today's Artifacts")
    lines.append(f"- Morning Report: {'exists' if mrep.exists() else 'pending'} ({mrep.name})")
    lines.append(f"- Morning Check: {'exists' if mchk.exists() else 'pending'}")
    lines.append(f"- Top-3 Picks: {'exists' if picks.exists() else 'pending'}")
    lines.append(f"- Trade of the Day: {'exists' if totd.exists() else 'pending'}")
    lines.append(f"- EOD Report: {'exists' if eod.exists() else 'pending'} ({eod.name})")
    lines.append("")
    if model:
        lines.append("## Model (Deployed Summary)")
        df = pd.DataFrame([
            {'strategy': k, **v} for k, v in model.items()
        ]) if isinstance(model, dict) else pd.DataFrame()
        if not df.empty:
            # Show key metrics
            cols = [c for c in ['strategy','accuracy','brier','wr','pf','sharpe','test_rows'] if c in df.columns]
            if cols:
                show = df[cols].copy()
                lines.append(show.to_markdown(index=False))
                lines.append("")
    lines.append("## Recent Journal (last 7 days)")
    for k, v in sorted(journal_sum.get('events_last_7d', {}).items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Goals & Next Steps")
    lines.append("- Maintain confidence calibration; monitor Brier/WR/PF/Sharpe on holdout")
    lines.append("- Enforce liquidity/spread gates for live execution; expand ADV/spread checks")
    lines.append("- Weekly retrain/promote with promotion gates; rollback on drift/perf drop")
    lines.append("- Extend features (breadth, dispersion) and add SHAP insights to morning report")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description='Update Kobe STATUS.md and daily history from journal/state')
    ap.add_argument('--days', type=int, default=7)
    args = ap.parse_args()

    today = datetime.utcnow()
    journal = load_jsonl(ROOT / 'state' / 'journal.jsonl')
    js = summarize_journal(journal, days=args.days)
    md = render_status(today, js)

    out = ROOT / 'docs' / 'STATUS.md'
    hist_dir = ROOT / 'docs' / 'history'
    hist_dir.mkdir(parents=True, exist_ok=True)
    snap = hist_dir / f'status_{today.strftime("%Y%m%d")}.md'
    out.write_text(md, encoding='utf-8')
    snap.write_text(md, encoding='utf-8')
    print('Updated:', out)
    print('Snapshot:', snap)


if __name__ == '__main__':
    main()

