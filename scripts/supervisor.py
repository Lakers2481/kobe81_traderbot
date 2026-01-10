#!/usr/bin/env python3
from __future__ import annotations

"""
Lightweight supervisor: ensures heartbeat freshness and scheduler liveness.

Intended to be run every 5-10 minutes via Task Scheduler.
If heartbeat is stale > 3 minutes, attempts to (re)start the scheduler.
"""

import subprocess
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args: list[str]) -> int:
    return subprocess.run(args, cwd=str(ROOT)).returncode


def load_last_heartbeat() -> dict:
    p = ROOT / 'logs' / 'heartbeat.jsonl'
    if not p.exists():
        return {}
    try:
        last = None
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    last = line
        return json.loads(last) if last else {}
    except Exception:
        return {}


def main() -> None:
    hb = load_last_heartbeat()
    stale = True
    if hb:
        ts = hb.get('ts_utc')
        if ts:
            try:
                t = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                age = (datetime.now(timezone.utc) - t).total_seconds()
                stale = age > 180  # > 3 minutes
            except Exception:
                stale = True

    if stale:
        print('Heartbeat stale or missing. Attempting to start scheduler...')
        rc = run_cmd([sys.executable, str(ROOT / 'scripts' / 'scheduler_kobe.py'), '--dotenv', str(ROOT / '.env'), '--telegram'])
        print('Scheduler start rc=', rc)
    else:
        print('Heartbeat fresh. No action.')


if __name__ == '__main__':
    main()

