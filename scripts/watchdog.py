#!/usr/bin/env python3
"""
watchdog.py - Monitor and restart the Kobe Master Scheduler if it crashes.

This script is designed to run every 5 minutes via Windows Task Scheduler.
It checks if the scheduler is alive by examining the state file modification time.
If stale (>10 min), it restarts the scheduler.

Usage:
    python scripts/watchdog.py                    # Check status only
    python scripts/watchdog.py --restart-if-dead  # Restart if scheduler is dead
    python scripts/watchdog.py --status           # Show detailed status
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Exponential backoff for restarts (prevents restart storms)
try:
    from core.restart_backoff import get_restart_backoff, RestartBackoffConfig
    BACKOFF_AVAILABLE = True
except ImportError:
    BACKOFF_AVAILABLE = False

STATE_FILE = ROOT / "state" / "scheduler_master.json"
HEARTBEAT_FILE = ROOT / "state" / "heartbeat.json"
WATCHDOG_LOG = ROOT / "logs" / "watchdog.jsonl"
STALE_THRESHOLD_MINUTES = 10


def log_event(event: str, details: dict = None) -> None:
    """Append an event to the watchdog log."""
    WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "event": event,
        **(details or {})
    }
    with open(WATCHDOG_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def get_scheduler_pid() -> int | None:
    """Find running scheduler_kobe.py process."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["wmic", "process", "where",
                 "CommandLine like '%scheduler_kobe.py%' and Name='python.exe'",
                 "get", "ProcessId"],
                capture_output=True, text=True, timeout=15
            )
            lines = [l.strip() for l in result.stdout.strip().split('\n')
                     if l.strip() and l.strip().isdigit()]
            if lines:
                return int(lines[0])
        except Exception:
            pass
    else:
        try:
            result = subprocess.run(
                ["pgrep", "-f", "scheduler_kobe.py"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip().split()[0])
        except Exception:
            pass
    return None


def get_state_age_minutes() -> float | None:
    """Get age of state file in minutes."""
    if not STATE_FILE.exists():
        return None
    try:
        mtime = STATE_FILE.stat().st_mtime
        age_seconds = time.time() - mtime
        return age_seconds / 60.0
    except Exception:
        return None


def get_heartbeat_age_minutes() -> float | None:
    """Get age of heartbeat file in minutes."""
    if not HEARTBEAT_FILE.exists():
        return None
    try:
        mtime = HEARTBEAT_FILE.stat().st_mtime
        age_seconds = time.time() - mtime
        return age_seconds / 60.0
    except Exception:
        return None


def is_scheduler_alive() -> tuple[bool, str]:
    """
    Check if scheduler is alive.
    Returns (is_alive, reason).
    """
    pid = get_scheduler_pid()
    if pid:
        return True, f"Process running (PID {pid})"

    # No process found - check if it ran recently (might be between ticks)
    state_age = get_state_age_minutes()
    heartbeat_age = get_heartbeat_age_minutes()

    # If state file updated recently, scheduler might be in a sleep cycle
    if state_age is not None and state_age < STALE_THRESHOLD_MINUTES:
        return True, f"State file updated {state_age:.1f} min ago (scheduler in tick sleep)"

    if heartbeat_age is not None and heartbeat_age < STALE_THRESHOLD_MINUTES:
        return True, f"Heartbeat updated {heartbeat_age:.1f} min ago"

    # Scheduler is dead
    if state_age is not None:
        return False, f"State file stale ({state_age:.1f} min old)"

    return False, "No state file found - scheduler never ran"


def start_scheduler() -> bool:
    """Start the scheduler in background."""
    dotenv = ROOT / ".env"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "scheduler_kobe.py"),
        "--dotenv", str(dotenv),
        "--universe", "data/universe/optionable_liquid_900.csv",
        "--cap", "900",
        "--tick-seconds", "20",
        "--telegram"
    ]

    try:
        if sys.platform == "win32":
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            subprocess.Popen(
                cmd, cwd=str(ROOT),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            )
        else:
            subprocess.Popen(
                cmd, cwd=str(ROOT),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        return True
    except Exception as e:
        log_event("start_failed", {"error": str(e)})
        return False


def cmd_status(args) -> int:
    """Show detailed status."""
    print("=" * 50)
    print("  KOBE WATCHDOG STATUS")
    print("=" * 50)
    print()

    is_alive, reason = is_scheduler_alive()
    pid = get_scheduler_pid()
    state_age = get_state_age_minutes()
    heartbeat_age = get_heartbeat_age_minutes()

    print(f"Scheduler:    {'ALIVE' if is_alive else 'DEAD'}")
    print(f"Reason:       {reason}")
    print(f"PID:          {pid or 'Not found'}")
    print(f"State Age:    {state_age:.1f} min" if state_age else "State Age:    N/A")
    print(f"Heartbeat:    {heartbeat_age:.1f} min" if heartbeat_age else "Heartbeat:    N/A")
    print(f"Threshold:    {STALE_THRESHOLD_MINUTES} min")
    print()

    # Show backoff status
    if BACKOFF_AVAILABLE:
        print()
        print("Restart Backoff:")
        try:
            backoff = get_restart_backoff()
            status = backoff.get_status()
            print(f"  Attempts:     {status['attempt_count']}/{status['max_attempts_per_hour']}")
            print(f"  Last Restart: {status['last_restart_time'] or 'Never'}")
            print(f"  Next Allowed: {status['next_restart_allowed_at'] or 'Now'}")
            print(f"  Base Delay:   {status['base_delay_seconds']}s")
        except Exception as e:
            print(f"  (error: {e})")
    else:
        print()
        print("Restart Backoff: Not available")

    # Show recent watchdog events
    print()
    if WATCHDOG_LOG.exists():
        print("Recent Watchdog Events:")
        try:
            with open(WATCHDOG_LOG, "r", encoding="utf-8") as f:
                lines = f.readlines()[-5:]
                for line in lines:
                    data = json.loads(line.strip())
                    ts = data.get("ts", "")[:19]
                    event = data.get("event", "unknown")
                    print(f"  [{ts}] {event}")
        except Exception:
            print("  (unable to read log)")

    return 0 if is_alive else 1


def cmd_check(args) -> int:
    """Check and optionally restart."""
    is_alive, reason = is_scheduler_alive()

    if is_alive:
        if not args.quiet:
            print(f"OK: {reason}")
        return 0

    print(f"DEAD: {reason}")
    log_event("scheduler_dead", {"reason": reason})

    if args.restart_if_dead:
        # Apply exponential backoff to prevent restart storms
        if BACKOFF_AVAILABLE:
            backoff = get_restart_backoff()
            allowed, delay, backoff_reason = backoff.should_restart()

            if not allowed:
                print(f"Restart BLOCKED: {backoff_reason}")
                log_event("restart_blocked", {"reason": backoff_reason})
                return 1

            if delay > 0:
                print(f"Backoff: waiting {delay:.1f}s before restart...")
                log_event("restart_delayed", {"delay_seconds": delay})
                time.sleep(delay)

        print("Restarting scheduler...")
        if start_scheduler():
            log_event("scheduler_restarted", {"trigger": "watchdog"})

            # Record successful restart for backoff tracking
            if BACKOFF_AVAILABLE:
                backoff.record_success()

            time.sleep(3)

            # Verify restart
            pid = get_scheduler_pid()
            if pid:
                print(f"Scheduler restarted (PID {pid})")
                return 0
            else:
                print("Restart may have failed - no PID found")
                log_event("restart_verification_failed")
                if BACKOFF_AVAILABLE:
                    backoff.record_failure("no_pid_after_restart")
                return 1
        else:
            print("Failed to start scheduler")
            if BACKOFF_AVAILABLE:
                backoff.record_failure("start_scheduler_failed")
            return 1

    return 1


def main():
    parser = argparse.ArgumentParser(description="Kobe Scheduler Watchdog")
    parser.add_argument("--restart-if-dead", action="store_true",
                        help="Restart scheduler if found dead")
    parser.add_argument("--status", action="store_true",
                        help="Show detailed status")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress output when OK")
    args = parser.parse_args()

    if args.status:
        sys.exit(cmd_status(args))
    else:
        sys.exit(cmd_check(args))


if __name__ == "__main__":
    main()
