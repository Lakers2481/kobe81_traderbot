#!/usr/bin/env python3
from __future__ import annotations

"""
Run a single scheduled job by tag.

This is a one-shot runner used by Windows Task Scheduler to execute discrete
jobs (e.g., MORNING_REPORT, FIRST_SCAN) without the long-running master loop.
"""

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args: list[str]) -> int:
    p = subprocess.run(args, cwd=str(ROOT))
    return p.returncode


def _load_env_file(p: Path, only_telegram: bool = False) -> None:
    if not p.exists():
        return
    try:
        for line in p.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip(); v = v.strip().strip('"').strip("'")
            if only_telegram and not k.startswith('TELEGRAM_'):
                continue
            import os as _os
            if not _os.getenv(k):
                _os.environ[k] = v
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a single Kobe scheduler job by tag")
    ap.add_argument("--tag", type=str, required=True,
                    choices=[
                        "DB_BACKUP","DATA_UPDATE","MORNING_REPORT","MORNING_CHECK",
                        "PRE_GAME","MARKET_NEWS","PREMARKET_SCAN","FIRST_SCAN",
                        "HALF_TIME","AFTERNOON_SCAN","SWING_SCANNER","POST_GAME",
                        "EOD_REPORT","EOD_LEARNING","OVERNIGHT_ANALYSIS"
                    ])
    ap.add_argument("--dotenv", type=str, default="./.env")
    ap.add_argument("--universe", type=str, default=str(ROOT / 'data' / 'universe' / 'optionable_liquid_800.csv'))
    ap.add_argument("--cap", type=int, default=900)
    ap.add_argument("--min-conf", type=float, default=0.60)
    ap.add_argument("--telegram", action="store_true", help="Send Telegram notifications for job result")
    ap.add_argument("--telegram-dotenv", type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env', help="Optional .env to source TELEGRAM_* keys")
    args = ap.parse_args()

    # Determine a default scan_date (yesterday to avoid partial bar),
    # except for premarket/news where we use today.
    from zoneinfo import ZoneInfo
    ET = ZoneInfo('America/New_York')
    CT = ZoneInfo('America/Chicago')

    # Telegram wiring
    send_fn = None
    if args.telegram:
        # Load project env first, then fallback to 2K28 for TELEGRAM_*
        _load_env_file(Path(args.dotenv))
        _load_env_file(Path(args.telegram_dotenv), only_telegram=True)
        try:
            from core.alerts import send_telegram as _send
            send_fn = _send
        except Exception:
            send_fn = None
    now = datetime.now(ET)
    ymd = (now - timedelta(days=1)).date().isoformat()
    if args.tag in ("PRE_GAME","MARKET_NEWS"):
        ymd = now.date().isoformat()

    rc = 0
    msg = None
    def _stamp() -> str:
        try:
            ct = datetime.now(CT).strftime('%I:%M %p').lstrip('0') + ' CT'
            et = datetime.now(ET).strftime('%I:%M %p').lstrip('0') + ' ET'
            return f"{ct} | {et}"
        except Exception:
            return ''
    if args.tag == "MORNING_REPORT":
        rc = run_cmd([sys.executable, str(ROOT / 'scripts/morning_report.py')])
        msg = f"<b>{args.tag}</b> [{_stamp()}] {'completed' if rc==0 else 'failed'}"
    elif args.tag == "MORNING_CHECK":
        rc = run_cmd([sys.executable, str(ROOT / 'scripts/morning_check.py'), '--dotenv', args.dotenv])
        msg = f"<b>{args.tag}</b> [{_stamp()}] {'completed' if rc==0 else 'failed'}"
    elif args.tag == "PRE_GAME":
        # Update sentiment cache, then generate PRE_GAME briefing
        rc1 = run_cmd([sys.executable, str(ROOT / 'scripts/update_sentiment_cache.py'), '--universe', args.universe, '--date', ymd, '--dotenv', args.dotenv])
        rc2 = run_cmd([sys.executable, str(ROOT / 'scripts/generate_briefing.py'), '--phase', 'pregame', '--universe', args.universe, '--cap', str(args.cap), '--dotenv', args.dotenv])
        rc = rc1 or rc2
        msg = f"<b>{args.tag}</b> [{_stamp()}] {'completed' if rc==0 else 'failed'}"
    elif args.tag == "MARKET_NEWS":
        rc = run_cmd([sys.executable, str(ROOT / 'scripts/update_sentiment_cache.py'), '--universe', args.universe, '--date', ymd, '--dotenv', args.dotenv])
        msg = f"<b>{args.tag}</b> [{_stamp()}] {'completed' if rc==0 else 'failed'}"
    elif args.tag == "PREMARKET_SCAN":
        rc = run_cmd([sys.executable, str(ROOT / 'scripts/pre_game_plan.py'), '--universe', args.universe, '--cap', str(args.cap), '--dotenv', args.dotenv, '--date', ymd])
        msg = f"<b>{args.tag}</b> [{_stamp()}] {'completed' if rc==0 else 'failed'}"
    elif args.tag == "FIRST_SCAN":
        # Readiness gate
        try:
            rc_ready = run_cmd([sys.executable, str(ROOT / 'scripts' / 'readiness_check.py'), '--dotenv', args.dotenv, '--universe', args.universe])
            if rc_ready != 0:
                msg = "<b>FIRST_SCAN</b> readiness check failed"
                print(msg)
                if send_fn:
                    try:
                        send_fn(msg)
                    except Exception:
                        pass
                sys.exit(rc_ready)
        except Exception:
            pass
        # Dynamic confidence policy
        min_conf = args.min_conf
        try:
            from ml_meta.conf_policy import compute as dynamic_conf
            eff, _ = dynamic_conf(min_conf_base=float(args.min_conf))
            min_conf = float(eff)
        except Exception:
            min_conf = float(args.min_conf)
        rc = run_cmd([sys.executable, str(ROOT / 'scripts' / 'run_daily_pipeline.py'), '--universe', args.universe, '--cap', str(args.cap), '--date', ymd, '--dotenv', args.dotenv, '--min-conf', str(min_conf), '--ensure-top3'])
        msg = f"<b>{args.tag}</b> [{_stamp()}] {'completed' if rc==0 else 'failed'}"
    elif args.tag == "HALF_TIME":
        # Generate HALF_TIME briefing with position analysis
        rc = run_cmd([sys.executable, str(ROOT / 'scripts/generate_briefing.py'), '--phase', 'halftime', '--dotenv', args.dotenv])
        msg = f"<b>{args.tag}</b> [{_stamp()}] {'completed' if rc==0 else 'failed'}"
    elif args.tag in ("AFTERNOON_SCAN","SWING_SCANNER"):
        rc = run_cmd([
            sys.executable, str(ROOT / 'scripts/scan.py'), '--dotenv', args.dotenv,
            '--strategy', 'all', '--cap', str(args.cap), '--top3', '--ml', '--date', ymd, '--ensure-top3'
        ])
        msg = f"<b>{args.tag}</b> [{_stamp()}] {'completed' if rc==0 else 'failed'}"
    elif args.tag == "DB_BACKUP":
        rc = run_cmd([sys.executable, str(ROOT / 'scripts/backup_state.py')])
        msg = f"<b>{args.tag}</b> [{_stamp()}] {'completed' if rc==0 else 'failed'}"
    elif args.tag == "DATA_UPDATE":
        start = (now.date() - timedelta(days=730)).isoformat()
        end = now.date().isoformat()
        rc = run_cmd([sys.executable, str(ROOT / 'scripts/prefetch_polygon_universe.py'), '--universe', args.universe, '--start', start, '--end', end, '--concurrency', '3', '--dotenv', args.dotenv])
        msg = f"<b>{args.tag}</b> [{_stamp()}] {'completed' if rc==0 else 'failed'}"
    elif args.tag == "EOD_REPORT":
        rc = run_cmd([sys.executable, str(ROOT / 'scripts/eod_report.py')])
        msg = f"<b>{args.tag}</b> [{_stamp()}] {'completed' if rc==0 else 'failed'}"
    elif args.tag == "EOD_LEARNING":
        rc = run_cmd([sys.executable, str(ROOT / 'scripts/run_weekly_training.py'), '--wfdir', 'wf_outputs', '--dotenv', args.dotenv])
        msg = f"<b>{args.tag}</b> [{_stamp()}] {'completed' if rc==0 else 'failed'}"
    elif args.tag == "POST_GAME":
        # Generate POST_GAME briefing with performance analysis and lessons
        rc = run_cmd([sys.executable, str(ROOT / 'scripts/generate_briefing.py'), '--phase', 'postgame', '--dotenv', args.dotenv])
        msg = f"<b>{args.tag}</b> [{_stamp()}] {'completed' if rc==0 else 'failed'}"
    elif args.tag == "OVERNIGHT_ANALYSIS":
        rc = 0
        msg = f"<b>{args.tag}</b> [{_stamp()}] skipped (placeholder)"
    else:
        print(f"Unknown tag: {args.tag}")
        rc = 1
        msg = f"<b>{args.tag}</b> [{_stamp()}] unknown job"

    # Telegram notify
    if args.telegram and send_fn and msg:
        try:
            send_fn(msg)
        except Exception:
            pass

    sys.exit(rc)


if __name__ == '__main__':
    main()
