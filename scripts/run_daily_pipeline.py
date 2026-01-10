#!/usr/bin/env python3
from __future__ import annotations

"""
Daily ML + Sentiment Pipeline

1) Update sentiment cache for the scan date
2) Scan with ML + sentiment blending, write Top-3 and TOTD
3) Submit TOTD (confidence gated)
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.journal import append_journal


def run_cmd(args: list[str]) -> int:
    p = subprocess.run(args, cwd=str(ROOT))
    return p.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description='Run daily ML+sentiment pipeline for Top-3 and TOTD')
    ap.add_argument('--universe', type=str, default='data/universe/optionable_liquid_800.csv')
    ap.add_argument('--cap', type=int, default=900)
    ap.add_argument('--date', type=str, default=None, help='Scan date (YYYY-MM-DD); default: last business day UTC')
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--min-conf', type=float, default=0.60)
    ap.add_argument('--ensure-top3', action='store_true')
    args = ap.parse_args()

    # Determine scan date
    if args.date:
        scan_date = args.date
    else:
        # default to yesterday (to avoid partial bar day)
        scan_date = (datetime.utcnow().date() - timedelta(days=1)).isoformat()

    # 1) Update sentiment cache
    rc = run_cmd([sys.executable, str(ROOT / 'scripts/update_sentiment_cache.py'),
                  '--universe', args.universe, '--date', scan_date, '--dotenv', args.dotenv])
    if rc != 0:
        print('WARN: sentiment cache update failed (continuing)')

    # 2) Scan with ML + sentiment; write Top-3 and TOTD
    scan_args = [
        sys.executable, str(ROOT / 'scripts/scan.py'),
        '--dotenv', args.dotenv,
        '--strategy', 'all',
        '--cap', str(args.cap),
        '--top3', '--ml',
        '--date', scan_date,
        '--min-conf', str(args.min_conf),
    ]
    if args.ensure_top3:
        scan_args.append('--ensure-top3')
    rc = run_cmd(scan_args)
    if rc != 0:
        print('ERROR: scan failed')
        sys.exit(rc)

    # 3) Submit TOTD
    rc = run_cmd([sys.executable, str(ROOT / 'scripts/submit_totd.py'), '--dotenv', args.dotenv])
    if rc != 0:
        print('WARN: submit TOTD returned', rc)
    # Journal daily artifacts
    try:
        append_journal('daily_pipeline', {
            'date': scan_date,
            'picks_csv': str(ROOT / 'logs' / 'daily_picks.csv'),
            'totd_csv': str(ROOT / 'logs' / 'trade_of_day.csv'),
        })
    except Exception:
        pass
    # Update STATUS.md and history snapshot
    run_cmd([sys.executable, str(ROOT / 'scripts/update_status_md.py')])


if __name__ == '__main__':
    main()
