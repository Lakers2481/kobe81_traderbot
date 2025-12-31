#!/usr/bin/env python3
from __future__ import annotations

"""
Kobe Master Scheduler v2.0 (24/7)

Enhanced production scheduler with position management and divergence monitoring.
Implements America/New_York times and runs Kobe commands accordingly.

Daily Schedule (ET):
=============================================================================
PRE-MARKET (5:30 - 9:30)
- 05:30 DB_BACKUP          - State backup
- 06:00 DATA_UPDATE        - Warm data cache
- 06:30 MORNING_REPORT     - Generate morning summary
- 06:45 PREMARKET_CHECK    - Data staleness, splits, missing bars
- 08:00 PRE_GAME           - AI Briefing (evidence-locked)
- 09:00 MARKET_NEWS        - Update sentiment
- 09:15 PREMARKET_SCAN     - Build plan (portfolio-aware)

MARKET HOURS (9:30 - 16:00)
- 09:45 FIRST_SCAN         - ENTRY WINDOW - Submit orders
- 09:50+ POSITION_MANAGER  - Every 15 min: stops, exits, P&L
- 10:00+ DIVERGENCE        - Every 30 min: sync validation
- 12:00 HALF_TIME          - AI Briefing + position review
- 12:30 RECONCILE_MIDDAY   - Full broker-OMS reconciliation
- 14:30 AFTERNOON_SCAN     - Refresh Top-3 (portfolio-aware)
- 15:30 SWING_SCANNER      - Swing setups
- 15:55 POSITION_CLOSE_CHECK - Enforce time stops before close

POST-MARKET (16:00 - 21:00)
- 16:00 POST_GAME          - AI Briefing + lessons
- 16:05 EOD_REPORT         - Performance report
- 16:15 RECONCILE_EOD      - Full reconciliation + report
- 17:00 EOD_LEARNING       - Weekly ML training (Fridays)
- 18:00 EOD_FINALIZE       - Finalize EOD data after provider delay
- 21:00 OVERNIGHT_ANALYSIS - Overnight analysis

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

# Ensure project root is in Python path for imports BEFORE core imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.clock.tz_utils import fmt_ct, now_et
from core.journal import append_journal

STATE_FILE = ROOT / 'state' / 'scheduler_master.json'
ET = ZoneInfo('America/New_York')


@dataclass(frozen=True)
class ScheduleEntry:
    tag: str
    time: dtime  # ET


# =============================================================================
# SCHEDULE v2.0 - Enhanced with Position Manager & Divergence Monitor
# =============================================================================
# Pre-market: 05:30-09:30 (data, briefing, planning)
# Market hours: 09:30-16:00 (entries, position management, divergence checks)
# Post-market: 16:00-21:00 (reconciliation, learning, overnight)
# =============================================================================

SCHEDULE: List[ScheduleEntry] = [
    # === PRE-MARKET (5:30 - 9:30 ET) ===
    ScheduleEntry('DB_BACKUP', dtime(5, 30)),
    ScheduleEntry('DATA_UPDATE', dtime(6, 0)),
    ScheduleEntry('MORNING_REPORT', dtime(6, 30)),
    ScheduleEntry('PREMARKET_CHECK', dtime(6, 45)),  # Data staleness, splits check
    ScheduleEntry('PRE_GAME', dtime(8, 0)),          # AI Briefing (evidence-locked)
    ScheduleEntry('MARKET_NEWS', dtime(9, 0)),       # Update sentiment
    ScheduleEntry('PREMARKET_SCAN', dtime(9, 15)),   # Build plan (portfolio-aware)

    # === MARKET OPEN - ENTRY WINDOW (9:45) ===
    ScheduleEntry('FIRST_SCAN', dtime(9, 45)),       # ENTRY WINDOW - Submit orders

    # === POSITION MANAGER (every 15 min during market hours) ===
    ScheduleEntry('POSITION_MANAGER_1', dtime(9, 50)),
    ScheduleEntry('POSITION_MANAGER_2', dtime(10, 5)),
    ScheduleEntry('POSITION_MANAGER_3', dtime(10, 20)),
    ScheduleEntry('POSITION_MANAGER_4', dtime(10, 35)),
    ScheduleEntry('POSITION_MANAGER_5', dtime(10, 50)),
    ScheduleEntry('POSITION_MANAGER_6', dtime(11, 5)),
    ScheduleEntry('POSITION_MANAGER_7', dtime(11, 20)),
    ScheduleEntry('POSITION_MANAGER_8', dtime(11, 35)),
    ScheduleEntry('POSITION_MANAGER_9', dtime(11, 50)),

    # === MID-DAY ===
    ScheduleEntry('HALF_TIME', dtime(12, 0)),        # AI Briefing + position review
    ScheduleEntry('RECONCILE_MIDDAY', dtime(12, 30)), # Full broker-OMS reconciliation

    # === AFTERNOON POSITION MANAGER ===
    ScheduleEntry('POSITION_MANAGER_10', dtime(12, 45)),
    ScheduleEntry('POSITION_MANAGER_11', dtime(13, 0)),
    ScheduleEntry('POSITION_MANAGER_12', dtime(13, 15)),
    ScheduleEntry('POSITION_MANAGER_13', dtime(13, 30)),
    ScheduleEntry('POSITION_MANAGER_14', dtime(13, 45)),
    ScheduleEntry('POSITION_MANAGER_15', dtime(14, 0)),
    ScheduleEntry('POSITION_MANAGER_16', dtime(14, 15)),

    ScheduleEntry('AFTERNOON_SCAN', dtime(14, 30)),  # Refresh Top-3 (portfolio-aware)

    ScheduleEntry('POSITION_MANAGER_17', dtime(14, 45)),
    ScheduleEntry('POSITION_MANAGER_18', dtime(15, 0)),
    ScheduleEntry('POSITION_MANAGER_19', dtime(15, 15)),

    ScheduleEntry('SWING_SCANNER', dtime(15, 30)),   # Swing setups

    ScheduleEntry('POSITION_MANAGER_20', dtime(15, 45)),
    ScheduleEntry('POSITION_CLOSE_CHECK', dtime(15, 55)), # Enforce time stops before close

    # === MARKET CLOSE (16:00) ===
    ScheduleEntry('POST_GAME', dtime(16, 0)),        # AI Briefing + lessons
    ScheduleEntry('EOD_REPORT', dtime(16, 5)),
    ScheduleEntry('RECONCILE_EOD', dtime(16, 15)),   # Full reconciliation + report

    # === POST-MARKET ===
    ScheduleEntry('EOD_LEARNING', dtime(17, 0)),     # Weekly ML training (Fridays)
    ScheduleEntry('EOD_FINALIZE', dtime(18, 0)),     # Finalize EOD data after provider delay
    ScheduleEntry('OVERNIGHT_ANALYSIS', dtime(21, 0)),

    # === DIVERGENCE MONITOR (runs alongside position manager) ===
    ScheduleEntry('DIVERGENCE_1', dtime(10, 0)),
    ScheduleEntry('DIVERGENCE_2', dtime(10, 30)),
    ScheduleEntry('DIVERGENCE_3', dtime(11, 0)),
    ScheduleEntry('DIVERGENCE_4', dtime(11, 30)),
    ScheduleEntry('DIVERGENCE_5', dtime(12, 15)),
    ScheduleEntry('DIVERGENCE_6', dtime(13, 0)),
    ScheduleEntry('DIVERGENCE_7', dtime(13, 30)),
    ScheduleEntry('DIVERGENCE_8', dtime(14, 0)),
    ScheduleEntry('DIVERGENCE_9', dtime(14, 45)),
    ScheduleEntry('DIVERGENCE_10', dtime(15, 15)),
    ScheduleEntry('DIVERGENCE_11', dtime(15, 45)),

    # === INTRADAY SCANS (every 15 min, 8:00 AM - 3:45 PM ET) ===
    # Lightweight scans to refresh Top-3 picks for manual review - NO auto-execution
    # These supplement the main scans (FIRST_SCAN, AFTERNOON_SCAN, SWING_SCANNER)
    ScheduleEntry('INTRADAY_SCAN_0800', dtime(8, 0)),
    ScheduleEntry('INTRADAY_SCAN_0815', dtime(8, 15)),
    ScheduleEntry('INTRADAY_SCAN_0830', dtime(8, 30)),
    ScheduleEntry('INTRADAY_SCAN_0845', dtime(8, 45)),
    ScheduleEntry('INTRADAY_SCAN_0900', dtime(9, 0)),
    ScheduleEntry('INTRADAY_SCAN_0930', dtime(9, 30)),
    ScheduleEntry('INTRADAY_SCAN_1000', dtime(10, 0)),
    ScheduleEntry('INTRADAY_SCAN_1015', dtime(10, 15)),
    ScheduleEntry('INTRADAY_SCAN_1030', dtime(10, 30)),
    ScheduleEntry('INTRADAY_SCAN_1045', dtime(10, 45)),
    ScheduleEntry('INTRADAY_SCAN_1100', dtime(11, 0)),
    ScheduleEntry('INTRADAY_SCAN_1115', dtime(11, 15)),
    ScheduleEntry('INTRADAY_SCAN_1130', dtime(11, 30)),
    ScheduleEntry('INTRADAY_SCAN_1145', dtime(11, 45)),
    ScheduleEntry('INTRADAY_SCAN_1200', dtime(12, 0)),
    ScheduleEntry('INTRADAY_SCAN_1215', dtime(12, 15)),
    ScheduleEntry('INTRADAY_SCAN_1230', dtime(12, 30)),
    ScheduleEntry('INTRADAY_SCAN_1245', dtime(12, 45)),
    ScheduleEntry('INTRADAY_SCAN_1300', dtime(13, 0)),
    ScheduleEntry('INTRADAY_SCAN_1315', dtime(13, 15)),
    ScheduleEntry('INTRADAY_SCAN_1330', dtime(13, 30)),
    ScheduleEntry('INTRADAY_SCAN_1345', dtime(13, 45)),
    ScheduleEntry('INTRADAY_SCAN_1400', dtime(14, 0)),
    ScheduleEntry('INTRADAY_SCAN_1415', dtime(14, 15)),
    ScheduleEntry('INTRADAY_SCAN_1445', dtime(14, 45)),
    ScheduleEntry('INTRADAY_SCAN_1500', dtime(15, 0)),
    ScheduleEntry('INTRADAY_SCAN_1515', dtime(15, 15)),
    ScheduleEntry('INTRADAY_SCAN_1545', dtime(15, 45)),
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


def do_refresh_top3(universe: str, dotenv: str, cap: int, scan_date: str) -> int:
    """Refresh top-3 picks using the scanner with deterministic mode."""
    args = [
        sys.executable, str(ROOT / 'scripts/scan.py'),
        '--dotenv', dotenv,
        '--strategy', 'all',
        '--cap', str(cap),
        '--top3', '--ml',
        '--date', scan_date,
        '--ensure-top3',
        '--deterministic'  # Required for reproducible results per STATUS.md
    ]
    return run_cmd(args)


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
    ap.add_argument('--telegram-dotenv', type=str, default=str(ROOT / '.env'), help='Optional .env file for Telegram keys (uses project .env by default)')
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
                        # Determine scan_date based on market phase:
                        # - Pre-market (before 9:30): use today for planning
                        # - Market hours (9:30-16:00): use today for live data
                        # - Post-market (after 16:00): use today (finalized data)
                        now_time = datetime.now(ET).time()
                        market_open = dtime(9, 30)
                        market_close = dtime(16, 0)
                        pre_market_jobs = ('PRE_GAME', 'MARKET_NEWS', 'PREMARKET_SCAN', 'PREMARKET_CHECK', 'MORNING_REPORT')

                        if market_open <= now_time <= market_close:
                            # During market hours: always use today's data
                            scan_date = datetime.now(ET).date().isoformat()
                        elif entry.tag in pre_market_jobs:
                            # Pre-market jobs: use today for planning
                            scan_date = datetime.now(ET).date().isoformat()
                        else:
                            # Post-market: use today (data should be finalized)
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

                        # === POSITION MANAGER (v2.0) ===
                        elif entry.tag.startswith('POSITION_MANAGER_') or entry.tag == 'POSITION_CLOSE_CHECK':
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/position_manager.py'), '--dotenv', args.dotenv])
                            # Only send Telegram on issues (rc != 0)
                            if send_fn and rc != 0:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>POSITION_MANAGER</b> [{stamp}] exit(s) executed or issues detected")
                                except Exception:
                                    pass

                        # === DIVERGENCE MONITOR (v2.0) ===
                        elif entry.tag.startswith('DIVERGENCE_'):
                            rc = run_cmd([sys.executable, str(ROOT / 'monitor/divergence_monitor.py'), '--dotenv', args.dotenv])
                            # Only send Telegram on critical issues (rc == 2)
                            if send_fn and rc == 2:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>DIVERGENCE ALERT</b> [{stamp}] Critical divergence detected!")
                                except Exception:
                                    pass

                        # === RECONCILIATION (v2.0) ===
                        elif entry.tag in ('RECONCILE_MIDDAY', 'RECONCILE_EOD'):
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/reconcile_alpaca.py'), '--dotenv', args.dotenv])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'discrepancies found'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'discrepancies found'}")

                        # === PREMARKET CHECK (v2.0) ===
                        elif entry.tag == 'PREMARKET_CHECK':
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/premarket_check.py'), '--dotenv', args.dotenv])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    status = 'PASS' if rc == 0 else ('WARN' if rc == 1 else 'FAIL')
                                    send_fn(f"<b>PREMARKET_CHECK</b> [{stamp}] {status}")
                                except Exception:
                                    send_fn(f"<b>PREMARKET_CHECK</b> {'PASS' if rc == 0 else 'issues detected'}")

                        # === EOD FINALIZE (v2.0) ===
                        elif entry.tag == 'EOD_FINALIZE':
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/eod_finalize.py'), '--dotenv', args.dotenv, '--max-wait', '45'])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>EOD_FINALIZE</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>EOD_FINALIZE</b> {'completed' if rc == 0 else 'failed'}")

                        # === INTRADAY SCANS (v2.1) - Every 15 min refresh ===
                        # Lightweight scans to find new setups throughout the day
                        # Outputs to logs/daily_picks.csv for manual review - NO auto-execution
                        elif entry.tag.startswith('INTRADAY_SCAN_'):
                            rc = do_refresh_top3(args.universe, args.dotenv, args.cap, scan_date)
                            # Only send Telegram notification on errors (to avoid spam)
                            if send_fn and rc != 0:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>INTRADAY_SCAN</b> [{stamp}] failed (rc={rc})")
                                except Exception:
                                    pass

                        # === OVERNIGHT_ANALYSIS (v2.0) - Placeholder ===
                        # Reserved for future overnight analysis features:
                        # - Overnight gap analysis
                        # - Pre-market movers detection
                        # - Earnings surprise analysis
                        elif entry.tag == 'OVERNIGHT_ANALYSIS':
                            # Intentionally no action - slot reserved for future implementation
                            pass

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



