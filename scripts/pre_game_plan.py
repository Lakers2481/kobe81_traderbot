#!/usr/bin/env python3
from __future__ import annotations

"""
Pre-Game Plan

Summarizes regime, sentiment snapshot, and produces pre-market Top-3 plan
without submitting orders. Uses prior trading day by default.
Outputs: reports/pre_game_plan_YYYYMMDD.html and logs/pre_market_picks.csv
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args: list[str]) -> int:
    return subprocess.run(args, cwd=str(ROOT)).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description='Generate pre-game plan and pre-market picks')
    ap.add_argument('--universe', type=str, default='data/universe/optionable_liquid_800.csv')
    ap.add_argument('--cap', type=int, default=300)
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--date', type=str, default=None, help='Plan date (YYYY-MM-DD); default: prior business day UTC')
    args = ap.parse_args()

    if args.date:
        day = args.date
    else:
        day = (datetime.utcnow().date() - timedelta(days=1)).isoformat()

    # Ensure sentiment cache exists for this day
    run_cmd([sys.executable, str(ROOT / 'scripts/update_sentiment_cache.py'), '--universe', args.universe, '--date', day, '--dotenv', args.dotenv])

    # Produce pre-market Top-3 (no submit), write to logs/pre_market_picks.csv
    picks_out = ROOT / 'logs' / 'pre_market_picks.csv'
    run_cmd([
        sys.executable, str(ROOT / 'scripts/scan.py'), '--dotenv', args.dotenv, '--strategy', 'all', '--cap', str(args.cap),
        '--top3', '--ml', '--ensure-top3', '--date', day, '--out-picks', str(picks_out), '--out-totd', str(ROOT / 'logs' / 'pre_market_totd.csv')
    ])

    # Generate lightweight HTML plan
    plan_path = ROOT / 'reports' / f'pre_game_plan_{day.replace("-", "")}.html'
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    picks_df = None
    if picks_out.exists():
        try:
            picks_df = pd.read_csv(picks_out)
        except Exception:
            picks_df = None
    # CT|ET stamp for display
    try:
        from core.clock.tz_utils import fmt_ct, now_et
        _now = now_et()
        stamp = f"Display: {fmt_ct(_now)} | {_now.strftime('%I:%M %p').lstrip('0')} ET"
    except Exception:
        stamp = ''
    html = ['<html><head><meta charset="utf-8"><title>Kobe Pre-Game Plan</title>',
            '<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3}</style>',
            '</head><body>', f'<h1>Pre-Game Plan - {day}</h1>', f'<p><em>{stamp}</em></p>']
    if picks_df is not None and not picks_df.empty:
        html.append('<h2>Top-3 (Pre-Market Plan)</h2>')
        html.append(picks_df[['strategy','symbol','side','entry_price','stop_loss','take_profit','conf_score']].to_html(index=False))
    else:
        html.append('<p>No picks generated.</p>')
    html.append('</body></html>')
    plan_path.write_text('\n'.join(html), encoding='utf-8')
    print('Pre-game plan written:', plan_path)


if __name__ == '__main__':
    main()
