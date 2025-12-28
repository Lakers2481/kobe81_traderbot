#!/usr/bin/env python3
from __future__ import annotations

"""
System heartbeat (minute-by-minute snapshot).

Writes a compact JSONL record of key runtime indicators every minute. Intended
to be scheduled via Windows Task Scheduler with a 1-minute repetition.
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]

LOG_JSONL = ROOT / 'logs' / 'heartbeat.jsonl'
LOG_TXT = ROOT / 'logs' / 'heartbeat_latest.txt'
STATE_SCHED = ROOT / 'state' / 'scheduler_master.json'


def load_env_file(p: Path) -> None:
    if not p.exists():
        return
    for line in p.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def file_info(p: Path) -> dict:
    try:
        if not p.exists():
            return {"exists": False}
        st = p.stat()
        return {"exists": True, "bytes": int(st.st_size), "mtime": datetime.utcfromtimestamp(st.st_mtime).isoformat() + 'Z'}
    except Exception:
        return {"exists": False}


def main() -> None:
    ap = argparse.ArgumentParser(description='Kobe heartbeat (minute-by-minute log)')
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--telegram', action='store_true', help='Send Telegram updates on anomalies')
    args = ap.parse_args()

    load_env_file(Path(args.dotenv))

    ET = ZoneInfo('America/New_York')
    now_utc = datetime.utcnow().isoformat() + 'Z'
    now_et = datetime.now(ET).isoformat()

    ks = ROOT / 'state' / 'KILL_SWITCH'
    picks = ROOT / 'logs' / 'daily_picks.csv'
    totd = ROOT / 'logs' / 'trade_of_day.csv'
    mrep = ROOT / 'reports' / f"morning_report_{datetime.now(ET).strftime('%Y%m%d')}.html"
    mchk = ROOT / 'reports' / 'morning_check.json'
    events = ROOT / 'logs' / 'events.jsonl'

    # Disk
    try:
        total, used, free = shutil.disk_usage(ROOT)
        disk = {"total": total, "used": used, "free": free}
    except Exception:
        disk = {}

    payload = {
        'ts_utc': now_utc,
        'ts_et': now_et,
        'kill_switch': ks.exists(),
        'files': {
            'daily_picks': file_info(picks),
            'totd': file_info(totd),
            'morning_report': file_info(mrep),
            'morning_check': file_info(mchk),
            'events_log': file_info(events),
        },
        'scheduler_state': json.loads(STATE_SCHED.read_text(encoding='utf-8')) if STATE_SCHED.exists() else {},
        'disk': disk,
    }

    LOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with LOG_JSONL.open('a', encoding='utf-8') as f:
        f.write(json.dumps(payload) + '\n')

    # Write compact text line for quick view
    txt = (
        f"{now_et} | KS={'ON' if payload['kill_switch'] else 'off'} "
        f"| picks={'Y' if payload['files']['daily_picks'].get('exists') else 'n'} "
        f"| totd={'Y' if payload['files']['totd'].get('exists') else 'n'}"
    )
    LOG_TXT.write_text(txt + '\n', encoding='utf-8')

    # Optional Telegram on anomalies: kill switch ON or missing picks by 10:00 ET
    if args.telegram:
        send = None
        try:
            from core.alerts import send_telegram as _send
            send = _send
        except Exception:
            send = None
        if send:
            try:
                etnow = datetime.now(ET)
                # Build CT|ET stamp
                try:
                    from core.clock.tz_utils import fmt_ct
                    stamp = f"{fmt_ct(etnow)} | {etnow.strftime('%I:%M %p').lstrip('0')} ET"
                except Exception:
                    stamp = ''
                if payload['kill_switch']:
                    send(f"[HEARTBEAT] Kill switch is ON {('['+stamp+']') if stamp else ''}")
                if etnow.hour >= 10 and not payload['files']['daily_picks'].get('exists'):
                    send(f"[HEARTBEAT] Missing daily_picks.csv after 10:00 ET {('['+stamp+']') if stamp else ''}")
            except Exception:
                pass


if __name__ == '__main__':
    main()
