#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timedelta, time as dtime
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.structured_log import jlog
from config.env_loader import load_env


STATE_FILE = ROOT / 'state' / 'runner_last.json'


def parse_times(csv: str) -> list[dtime]:
    out: list[dtime] = []
    for part in (csv or '').split(','):
        part = part.strip()
        if not part:
            continue
        hh, mm = part.split(':')
        out.append(dtime(hour=int(hh), minute=int(mm)))
    return out


def already_ran(tag: str, today: str) -> bool:
    if not STATE_FILE.exists():
        return False
    try:
        data = json.loads(STATE_FILE.read_text(encoding='utf-8'))
    except Exception:
        return False
    return data.get(tag) == today


def mark_ran(tag: str, today: str) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(STATE_FILE.read_text(encoding='utf-8')) if STATE_FILE.exists() else {}
    except Exception:
        data = {}
    data[tag] = today
    STATE_FILE.write_text(json.dumps(data, indent=2))


def within_market_day(now: datetime) -> bool:
    # Simple weekday filter (Mon-Fri). Extend with holiday calendar if needed.
    return now.weekday() < 5


def run_submit(mode: str, universe: Path, cap: int, start_days: int, dotenv: Path) -> int:
    end = datetime.utcnow().date().isoformat()
    start = (datetime.utcnow().date() - timedelta(days=start_days)).isoformat()
    base_cmd = [sys.executable]
    if mode == 'paper':
        script = ROOT / 'scripts' / 'run_paper_trade.py'
    else:
        script = ROOT / 'scripts' / 'run_live_trade_micro.py'
    cmd = [*base_cmd, str(script), '--universe', str(universe), '--start', start, '--end', end, '--cap', str(cap), '--dotenv', str(dotenv)]
    jlog('runner_execute', mode=mode, script=str(script), universe=str(universe), start=start, end=end, cap=cap)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout:
        print(p.stdout)
    if p.stderr:
        print(p.stderr, file=sys.stderr)
    return p.returncode


def main():
    ap = argparse.ArgumentParser(description='Kobe 24/7 Runner (submit on schedule)')
    ap.add_argument('--mode', type=str, choices=['paper','live'], default='paper')
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--cap', type=int, default=50)
    ap.add_argument('--scan-times', type=str, default='09:35,10:30,15:55', help='Local HH:MM times, comma separated')
    ap.add_argument('--lookback-days', type=int, default=540)
    ap.add_argument('--dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
    ap.add_argument('--once', action='store_true', help='Run once immediately and exit')
    args = ap.parse_args()

    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    universe = Path(args.universe)
    times = parse_times(args.scan_times)
    if args.once:
        run_submit(args.mode, universe, args.cap, args.lookback_days, dotenv)
        return

    jlog('runner_start', mode=args.mode, scan_times=args.scan_times, universe=str(universe))
    while True:
        now = datetime.now()
        if within_market_day(now):
            today_str = now.date().isoformat()
            for t in times:
                tag = f"{args.mode}_{t.strftime('%H%M')}"
                target_dt = datetime.combine(now.date(), t)
                if now >= target_dt and not already_ran(tag, today_str):
                    rc = run_submit(args.mode, universe, args.cap, args.lookback_days, dotenv)
                    mark_ran(tag, today_str)
                    jlog('runner_done', mode=args.mode, schedule=tag, returncode=rc)
        time.sleep(30)


if __name__ == '__main__':
    main()
