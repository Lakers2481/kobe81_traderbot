#!/usr/bin/env python3
from __future__ import annotations

"""
Shadow Runner

Runs the daily scan/selection exactly like production but never submits orders.
Emits 'shadow' picks and TOTD CSVs and logs an event for divergence checks.

Outputs (for a given --date):
  logs/shadow/daily_picks_YYYYMMDD.csv
  logs/shadow/trade_of_day_YYYYMMDD.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args: list[str]) -> int:
    p = subprocess.run(args, cwd=str(ROOT))
    return p.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description='Kobe Shadow Runner (no submissions)')
    ap.add_argument('--dotenv', type=str, default=str(ROOT / '.env'))
    ap.add_argument('--universe', type=str, default=str(ROOT / 'data' / 'universe' / 'optionable_liquid_900.csv'))
    ap.add_argument('--cap', type=int, default=900)
    ap.add_argument('--date', type=str, default=None, help='YYYY-MM-DD (default: yesterday, ET)')
    ap.add_argument('--min-conf', type=float, default=0.60)
    ap.add_argument('--min-adv-usd', type=float, default=5_000_000.0)
    args = ap.parse_args()

    # Determine date default (yesterday) to avoid partial bar
    if not args.date:
        try:
            from zoneinfo import ZoneInfo
            ET = ZoneInfo('America/New_York')
            ymd = (datetime.now(ET) - timedelta(days=1)).date().isoformat()
        except Exception:
            ymd = (datetime.utcnow().date() - timedelta(days=1)).isoformat()
    else:
        ymd = args.date

    outdir = ROOT / 'logs' / 'shadow'
    outdir.mkdir(parents=True, exist_ok=True)
    picks = outdir / f'daily_picks_{ymd}.csv'
    totd = outdir / f'trade_of_day_{ymd}.csv'

    cmd = [
        sys.executable, str(ROOT / 'scripts' / 'scan.py'),
        '--dotenv', args.dotenv,
        '--universe', args.universe,
        '--cap', str(args.cap),
        '--date', ymd,
        '--top3', '--ml', '--ensure-top3',
        '--min-conf', str(args.min_conf),
        '--min-adv-usd', str(args.min_adv_usd),
        '--out-picks', str(picks),
        '--out-totd', str(totd),
    ]
    rc = run_cmd(cmd)
    print('Shadow scan rc=', rc)
    print('Shadow picks:', picks)
    print('Shadow TOTD :', totd)


if __name__ == '__main__':
    main()

