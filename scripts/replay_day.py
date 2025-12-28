#!/usr/bin/env python3
from __future__ import annotations

"""
Replay Day: Re-run scan for a given date and compare to stored picks/TOTD.

Checks bit-for-bit equality on Top-3 and TOTD CSVs if provided.
"""

import argparse
import sys
from pathlib import Path
import subprocess
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(day: str, dotenv: str, universe: str, cap: int, picks_path: Path, totd_path: Path) -> int:
    tmpdir = ROOT / 'logs' / 'replay'
    tmpdir.mkdir(parents=True, exist_ok=True)
    rpicks = tmpdir / f'picks_{day}.csv'
    rtotd = tmpdir / f'totd_{day}.csv'
    rc = subprocess.run([
        sys.executable, str(ROOT / 'scripts' / 'scan.py'),
        '--dotenv', dotenv, '--universe', universe, '--cap', str(cap), '--date', day,
        '--top3', '--ml', '--ensure-top3', '--out-picks', str(rpicks), '--out-totd', str(rtotd)
    ], cwd=str(ROOT)).returncode
    if rc != 0:
        print('Replay scan failed rc=', rc)
        return rc
    def _eq(a: Path, b: Path) -> bool:
        try:
            if (not a.exists()) and (not b.exists()):
                return True
            if a.exists() != b.exists():
                return False
            da = pd.read_csv(a); db = pd.read_csv(b)
            da = da.fillna('').astype(str); db = db.fillna('').astype(str)
            return da.equals(db)
        except Exception:
            return False
    ok1 = _eq(picks_path, rpicks)
    ok2 = _eq(totd_path, rtotd)
    print('Replay Top-3 match:', ok1)
    print('Replay TOTD  match:', ok2)
    return 0 if (ok1 and ok2) else 2


def main() -> None:
    ap = argparse.ArgumentParser(description='Replay a trading day decisions and compare')
    ap.add_argument('--date', type=str, required=True)
    ap.add_argument('--dotenv', type=str, default=str(ROOT / '.env'))
    ap.add_argument('--universe', type=str, default=str(ROOT / 'data' / 'universe' / 'optionable_liquid_900.csv'))
    ap.add_argument('--cap', type=int, default=900)
    ap.add_argument('--picks', type=str, default=str(ROOT / 'logs' / 'daily_picks.csv'))
    ap.add_argument('--totd', type=str, default=str(ROOT / 'logs' / 'trade_of_day.csv'))
    args = ap.parse_args()
    sys.exit(run(args.date, args.dotenv, args.universe, args.cap, Path(args.picks), Path(args.totd)))


if __name__ == '__main__':
    main()

