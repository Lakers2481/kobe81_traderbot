#!/usr/bin/env python3
from __future__ import annotations

"""
Kobe Master Scheduler (24/7)

Implements a production schedule inspired by 2K28 MASTER_CONFIG with
America/New_York times and runs Kobe commands accordingly.

Daily schedule (ET):
- 05:30 DB_BACKUP (placeholder)
- 06:00 DATA_UPDATE (optional, placeholder)
- 08:00 PRE_GAME (update sentiment)
- 09:00 MARKET_NEWS (update sentiment)
- 09:45 FIRST_SCAN (Top‑3 + TOTD; ML+sentiment; confidence gated)
- 12:00 HALF_TIME (refresh Top‑3; no submit)
- 14:30 AFTERNOON_SCAN (refresh Top‑3; no submit)
- 15:30 SWING_SCANNER (refresh Top‑3; no submit)
- 16:00 POST_GAME (placeholder)
- 16:05 EOD_REPORT (placeholder)
- 17:00 EOD_LEARNING (weekly: dataset/train/promote)
- 21:00 OVERNIGHT_ANALYSIS (placeholder)

The scheduler stores last-run markers per tag/date to prevent duplicates.
"""

import argparse
import os
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo
from core.clock.tz_utils import fmt_ct, now_et
from core.journal import append_journal

ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / 'state' / 'scheduler_master.json'
ET = ZoneInfo('America/New_York')


@dataclass(frozen=True)
class ScheduleEntry:
    tag: str
    time: dtime  # ET


SCHEDULE: List[ScheduleEntry] = [
    ScheduleEntry('DB_BACKUP', dtime(5, 30)),
    ScheduleEntry('DATA_UPDATE', dtime(6, 0)),
    ScheduleEntry('MORNING_REPORT', dtime(6, 30)),
    ScheduleEntry('MORNING_CHECK', dtime(6, 45)),
    ScheduleEntry('PRE_GAME', dtime(8, 0)),
    ScheduleEntry('MARKET_NEWS', dtime(9, 0)),
    ScheduleEntry('PREMARKET_SCAN', dtime(9, 15)),
    ScheduleEntry('FIRST_SCAN', dtime(9, 45)),
    ScheduleEntry('HALF_TIME', dtime(12, 0)),
    ScheduleEntry('AFTERNOON_SCAN', dtime(14, 30)),
    ScheduleEntry('SWING_SCANNER', dtime(15, 30)),
    ScheduleEntry('POST_GAME', dtime(16, 0)),
    ScheduleEntry('EOD_REPORT', dtime(16, 5)),
    ScheduleEntry('EOD_LEARNING', dtime(17, 0)),
    ScheduleEntry('OVERNIGHT_ANALYSIS', dtime(21, 0)),
]


def load_state() -> Dict[str, str]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding='utf-8'))
    except Exception:
        return {}


def save_state(state: Dict[str, str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding='utf-8')


def mark_ran(state: Dict[str, str], tag: str, yyyymmdd: str) -> None:
    state[f'{tag}'] = yyyymmdd


def already_ran(state: Dict[str, str], tag: str, yyyymmdd: str) -> bool:
    return state.get(tag) == yyyymmdd


def run_cmd(args: List[str]) -> int:
    p = subprocess.run(args, cwd=str(ROOT))
    return p.returncode


def do_pregame(universe: str, dotenv: str, scan_date: str) -> None:
    run_cmd([sys.executable, str(ROOT / 'scripts/update_sentiment_cache.py'), '--universe', universe, '--date', scan_date, '--dotenv', dotenv])


def do_first_scan(universe: str, dotenv: str, cap: int, min_conf: float, scan_date: str) -> None:
    run_cmd([sys.executable, str(ROOT / 'scripts/run_daily_pipeline.py'), '--universe', universe, '--cap', str(cap), '--date', scan_date, '--dotenv', dotenv, '--min-conf', str(min_conf), '--ensure-top3'])


def do_refresh_top3(universe: str, dotenv: str, cap: int, scan_date: str) -> None:
    args = [
        sys.executable, str(ROOT / 'scripts/scan.py'),
        '--dotenv', dotenv,
        '--strategy', 'all',
        '--cap', str(cap),
        '--top3', '--ml',
        '--date', scan_date,
        '--ensure-top3'
    ]
    run_cmd(args)


def do_weekly_learning(wfdir: str, dotenv: str, min_delta: float, min_test: int) -> None:
    # Only run on Fridays
    if datetime.now(ET).weekday() == 4:
        run_cmd([sys.executable, str(ROOT / 'scripts/run_weekly_training.py'), '--wfdir', wfdir, '--dotenv', dotenv, '--min-delta', str(min_delta), '--min-test', str(min_test)])


def now_et_date_time() -> Tuple[str, dtime]:
    now = now_et()
    return now.date().isoformat(), dtime(now.hour, now.minute)


def main() -> None:
    ap = argparse.ArgumentParser(description='Kobe Master Scheduler (24/7)')
    ap.add_argument('--universe', type=str, default='data/universe/optionable_liquid_900.csv')
    ap.add_argument('--cap', type=int, default=900)
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--min-conf', type=float, default=0.60)
    ap.add_argument('--wfdir', type=str, default='wf_outputs')
    ap.add_argument('--min-delta', type=float, default=0.01)
    ap.add_argument('--min-test', type=int, default=100)
    ap.add_argument('--tick-seconds', type=int, default=20)
    ap.add_argument('--telegram', action='store_true', help='Send Telegram notifications for scheduler events')
    ap.add_argument('--telegram-dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env', help='Optional .env file for Telegram keys')
    args = ap.parse_args()

    # Bootstrap Telegram env if requested
    send_fn = None
    if args.telegram:
        # Load local .env first
        try:
            from config.env_loader import load_env
            de = Path(args.dotenv)
            if de.exists():
                load_env(de)
        except Exception:
            pass
        # Load 2K28 .env as fallback for TELEGRAM_* keys
        tdot = Path(args.telegram_dotenv)
        if tdot.exists():
            try:
                for line in tdot.read_text(encoding='utf-8').splitlines():
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    k = k.strip(); v = v.strip().strip('"').strip("'")
                    if k.startswith('TELEGRAM_') and not os.getenv(k):
                        os.environ[k] = v
            except Exception:
                pass
        try:
            from core.alerts import send_telegram as _send
            send_fn = _send
        except Exception:
            send_fn = None

    state = load_state()
    try:
        from core.clock.tz_utils import fmt_ct, now_et
        now = now_et()
        start_msg = f"Kobe Master Scheduler started. Display: {fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
    except Exception:
        start_msg = 'Kobe Master Scheduler started. Times in America/New_York (ET).'
    print(start_msg)
    if send_fn:
        try:
            from core.clock.tz_utils import fmt_ct
            now = now_et()
            send_fn(f"<b>Kobe Scheduler</b> started • {fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET")
        except Exception:
            pass
    try:
        append_journal('scheduler_started', {'telegram': bool(send_fn)})
    except Exception:
        pass

    while True:
        ymd, hhmm = now_et_date_time()
        for entry in SCHEDULE:
            try:
                if hhmm.hour == entry.time.hour and hhmm.minute == entry.time.minute:
                    if not already_ran(state, entry.tag, ymd):
                        print(f"[{fmt_ct(now_et())} | {hhmm.strftime('%I:%M %p').lstrip('0')} ET] Running {entry.tag}...")
                        # Determine scan_date default (yesterday to avoid partial bar unless pre/post market)
                        # For FIRST_SCAN and later intraday slots, use previous business day by default
                        scan_date = (datetime.now(ET) - timedelta(days=1)).date().isoformat()
                        if entry.tag in ('PRE_GAME','MARKET_NEWS'):
                            scan_date = datetime.now(ET).date().isoformat()

                        if entry.tag == 'MORNING_REPORT':
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/morning_report.py')])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et()
                                    stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")
                        elif entry.tag == 'MORNING_CHECK':
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/morning_check.py'), '--dotenv', args.dotenv])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")
                        elif entry.tag == 'PRE_GAME':
                            do_pregame(args.universe, args.dotenv, scan_date)
                            # Generate pre-game plan document (legacy)
                            run_cmd([sys.executable, str(ROOT / 'scripts/pre_game_plan.py'), '--universe', args.universe, '--cap', str(args.cap), '--dotenv', args.dotenv, '--date', scan_date])
                            # Generate comprehensive PRE_GAME briefing with LLM/ML analysis
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/generate_briefing.py'),
                                     '--phase', 'pregame', '--universe', args.universe, '--cap', str(args.cap),
                                     '--dotenv', args.dotenv, '--date', scan_date, '--telegram'])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")
                        elif entry.tag == 'MARKET_NEWS':
                            do_pregame(args.universe, args.dotenv, scan_date)
                        elif entry.tag == 'PREMARKET_SCAN':
                            # Build plan without submitting
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/pre_game_plan.py'), '--universe', args.universe, '--cap', str(args.cap), '--dotenv', args.dotenv, '--date', scan_date])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")
                        elif entry.tag == 'FIRST_SCAN':
                            do_first_scan(args.universe, args.dotenv, args.cap, args.min_conf, scan_date)
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] completed")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> completed")
                        elif entry.tag == 'HALF_TIME':
                            # Refresh Top-3 signals
                            do_refresh_top3(args.universe, args.dotenv, args.cap, scan_date)
                            # Generate comprehensive HALF_TIME briefing with position analysis
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/generate_briefing.py'),
                                     '--phase', 'halftime', '--dotenv', args.dotenv, '--telegram'])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if (rc is None or rc == 0) else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if (rc is None or rc == 0) else 'failed'}")
                        elif entry.tag in ('AFTERNOON_SCAN','SWING_SCANNER'):
                            rc = do_refresh_top3(args.universe, args.dotenv, args.cap, scan_date)
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if (rc is None or rc == 0) else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if (rc is None or rc == 0) else 'failed'}")
                        elif entry.tag == 'DB_BACKUP':
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/backup_state.py')])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")
                        elif entry.tag == 'DATA_UPDATE':
                            # Prefetch last 730 days to warm the cache
                            start = (datetime.now(ET).date() - timedelta(days=730)).isoformat()
                            end = datetime.now(ET).date().isoformat()
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/prefetch_polygon_universe.py'),
                                     '--universe', args.universe, '--start', start, '--end', end, '--concurrency', '3', '--dotenv', args.dotenv])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")
                        elif entry.tag == 'EOD_REPORT':
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/eod_report.py')])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")
                        elif entry.tag == 'EOD_LEARNING':
                            do_weekly_learning(args.wfdir, args.dotenv, args.min_delta, args.min_test)
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] completed (weekly promote)")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> completed (weekly promote)")
                        elif entry.tag == 'POST_GAME':
                            # Generate comprehensive POST_GAME briefing with full day analysis
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/generate_briefing.py'),
                                     '--phase', 'postgame', '--dotenv', args.dotenv, '--telegram'])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")
                        # Placeholders: DB_BACKUP, DATA_UPDATE, OVERNIGHT_ANALYSIS

                        mark_ran(state, entry.tag, ymd)
                        save_state(state)
                        try:
                            append_journal('scheduler_job', {'tag': entry.tag, 'date': ymd})
                        except Exception:
                            pass
            except Exception as e:
                print(f'Error in schedule {entry.tag}: {e}')
                if send_fn:
                    try:
                        try:
                            from core.clock.tz_utils import fmt_ct
                            now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                            send_fn(f"<b>{entry.tag}</b> [{stamp}] failed: {e}")
                        except Exception:
                            send_fn(f"<b>{entry.tag}</b> failed: {e}")
                    except Exception:
                        pass
                try:
                    append_journal('scheduler_job_failed', {'tag': entry.tag, 'error': str(e)})
                except Exception:
                    pass
        time.sleep(max(5, int(args.tick_seconds)))


if __name__ == '__main__':
    main()



