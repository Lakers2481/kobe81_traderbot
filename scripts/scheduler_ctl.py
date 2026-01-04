#!/usr/bin/env python3
"""
scheduler_ctl.py - Manage the Kobe Master Scheduler

Usage:
    python scripts/scheduler_ctl.py status      # Check scheduler status
    python scripts/scheduler_ctl.py start       # Start scheduler (manual)
    python scripts/scheduler_ctl.py stop        # Stop scheduler
    python scripts/scheduler_ctl.py register    # Register Windows Task
    python scripts/scheduler_ctl.py unregister  # Remove Windows Task
    python scripts/scheduler_ctl.py logs        # Tail scheduler logs
    python scripts/scheduler_ctl.py test        # Run scheduler for 60 seconds
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASK_NAME = "Kobe_Master_Scheduler"
STATE_FILE = ROOT / "state" / "scheduler_master.json"
LOGS_DIR = ROOT / "logs"


def get_scheduler_pid() -> Optional[int]:
    """Find running scheduler_kobe.py process."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["wmic", "process", "where",
                 "CommandLine like '%scheduler_kobe.py%' and Name='python.exe'",
                 "get", "ProcessId"],
                capture_output=True, text=True, timeout=10
            )
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip() and l.strip().isdigit()]
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


def get_task_status() -> dict:
    """Get Windows Task Scheduler status."""
    status = {"registered": False, "state": None, "last_run": None, "next_run": None}
    if sys.platform != "win32":
        return status

    try:
        result = subprocess.run(
            ["schtasks", "/query", "/tn", TASK_NAME, "/fo", "LIST", "/v"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            status["registered"] = True
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if "status" in key:
                        status["state"] = value
                    elif "last run time" in key:
                        status["last_run"] = value
                    elif "next run time" in key:
                        status["next_run"] = value
    except Exception:
        pass
    return status


def get_state_info() -> dict:
    """Read scheduler state file."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def cmd_status(args):
    """Show scheduler status."""
    print("=" * 50)
    print("  KOBE MASTER SCHEDULER STATUS")
    print("=" * 50)
    print()

    # Check if process is running
    pid = get_scheduler_pid()
    if pid:
        print(f"Process:   RUNNING (PID {pid})")
    else:
        print("Process:   NOT RUNNING")
    print()

    # Check Windows Task Scheduler
    task = get_task_status()
    if task["registered"]:
        print(f"Task:      {TASK_NAME}")
        print(f"State:     {task['state'] or 'Unknown'}")
        print(f"Last Run:  {task['last_run'] or 'Never'}")
        print(f"Next Run:  {task['next_run'] or 'Not scheduled'}")
    else:
        print("Task:      NOT REGISTERED")
        print("           Run: python scripts/scheduler_ctl.py register")
    print()

    # Check state file
    state = get_state_info()
    if state:
        today = datetime.now().strftime("%Y-%m-%d")
        jobs_today = [k for k, v in state.items() if v == today]
        print(f"State:     {STATE_FILE}")
        print(f"Jobs run today ({today}): {len(jobs_today)}")
        if jobs_today:
            # Show last 5 jobs
            for job in jobs_today[-5:]:
                print(f"           - {job}")
            if len(jobs_today) > 5:
                print(f"           ... and {len(jobs_today) - 5} more")
    else:
        print("State:     No state file (scheduler hasn't run yet)")
    print()


def cmd_start(args):
    """Start scheduler manually."""
    pid = get_scheduler_pid()
    if pid:
        print(f"Scheduler already running (PID {pid})")
        return 1

    print("Starting scheduler_kobe.py...")

    dotenv = args.dotenv or str(ROOT / ".env")
    cmd = [
        sys.executable, str(ROOT / "scripts" / "scheduler_kobe.py"),
        "--dotenv", dotenv,
        "--universe", "data/universe/optionable_liquid_900.csv",
        "--cap", "900",
        "--tick-seconds", "20",
    ]
    if args.telegram:
        cmd.append("--telegram")

    if args.foreground:
        # Run in foreground
        print("Running in foreground (Ctrl+C to stop)...")
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(ROOT))
    else:
        # Run in background
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
        time.sleep(2)
        pid = get_scheduler_pid()
        if pid:
            print(f"Scheduler started (PID {pid})")
        else:
            print("Scheduler may have failed to start. Check logs.")
            return 1
    return 0


def cmd_stop(args):
    """Stop scheduler."""
    pid = get_scheduler_pid()
    if not pid:
        print("Scheduler is not running.")
        return 0

    print(f"Stopping scheduler (PID {pid})...")

    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    else:
        import signal
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    time.sleep(1)
    if get_scheduler_pid():
        print("Failed to stop scheduler.")
        return 1
    print("Scheduler stopped.")
    return 0


def cmd_register(args):
    """Register Windows Task."""
    if sys.platform != "win32":
        print("Windows Task Scheduler only available on Windows.")
        return 1

    script = ROOT / "ops" / "windows" / "register_master_scheduler.ps1"
    if not script.exists():
        print(f"Registration script not found: {script}")
        return 1

    cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    if args.start:
        cmd.append("-Start")

    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


def cmd_unregister(args):
    """Unregister Windows Task."""
    if sys.platform != "win32":
        print("Windows Task Scheduler only available on Windows.")
        return 1

    script = ROOT / "ops" / "windows" / "register_master_scheduler.ps1"
    cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script), "-Unregister"]
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


def cmd_logs(args):
    """Tail scheduler logs."""
    events_log = LOGS_DIR / "events.jsonl"

    if not events_log.exists():
        print(f"No log file found: {events_log}")
        return 1

    print(f"Tailing {events_log} (Ctrl+C to stop)...")
    print("-" * 50)

    # Show last 20 lines then follow
    try:
        with open(events_log, "r", encoding="utf-8") as f:
            # Go to end and read last 20 lines
            f.seek(0, 2)
            file_size = f.tell()

            # Read last ~10KB to get recent lines
            f.seek(max(0, file_size - 10000))
            lines = f.readlines()

            # Print last 20
            for line in lines[-20:]:
                try:
                    data = json.loads(line.strip())
                    ts = data.get("ts", "")[:19]
                    event = data.get("event", "unknown")
                    print(f"[{ts}] {event}")
                except json.JSONDecodeError:
                    print(line.strip())

            print("-" * 50)
            print("Waiting for new events...")

            # Follow mode
            while True:
                line = f.readline()
                if line:
                    try:
                        data = json.loads(line.strip())
                        ts = data.get("ts", "")[:19]
                        event = data.get("event", "unknown")
                        print(f"[{ts}] {event}")
                    except json.JSONDecodeError:
                        print(line.strip())
                else:
                    time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def cmd_test(args):
    """Run scheduler for a short test period."""
    print("Running scheduler in test mode (60 seconds)...")

    dotenv = args.dotenv or str(ROOT / ".env")
    cmd = [
        sys.executable, str(ROOT / "scripts" / "scheduler_kobe.py"),
        "--dotenv", dotenv,
        "--universe", "data/universe/optionable_liquid_900.csv",
        "--cap", "900",
        "--tick-seconds", "5",  # Fast tick for testing
    ]

    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)

    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT))
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        print("\n" + "-" * 50)
        print("Test period complete. Stopping...")
        proc.terminate()
        proc.wait(timeout=5)
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping...")
        proc.terminate()
        proc.wait(timeout=5)

    print("Test complete.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Manage the Kobe Master Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  status       Show scheduler status
  start        Start scheduler manually
  stop         Stop running scheduler
  register     Register Windows Task Scheduler
  unregister   Remove Windows Task
  logs         Tail scheduler logs
  test         Run scheduler for 60 seconds (test mode)

Examples:
  python scripts/scheduler_ctl.py status
  python scripts/scheduler_ctl.py start --telegram
  python scripts/scheduler_ctl.py register --start
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # status
    status_parser = subparsers.add_parser("status", help="Show scheduler status")
    status_parser.set_defaults(func=cmd_status)

    # start
    start_parser = subparsers.add_parser("start", help="Start scheduler")
    start_parser.add_argument("--dotenv", type=str, help="Path to .env file")
    start_parser.add_argument("--telegram", action="store_true", help="Enable Telegram notifications")
    start_parser.add_argument("--foreground", action="store_true", help="Run in foreground")
    start_parser.set_defaults(func=cmd_start)

    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop scheduler")
    stop_parser.set_defaults(func=cmd_stop)

    # register
    register_parser = subparsers.add_parser("register", help="Register Windows Task")
    register_parser.add_argument("--start", action="store_true", help="Start immediately after registration")
    register_parser.set_defaults(func=cmd_register)

    # unregister
    unregister_parser = subparsers.add_parser("unregister", help="Remove Windows Task")
    unregister_parser.set_defaults(func=cmd_unregister)

    # logs
    logs_parser = subparsers.add_parser("logs", help="Tail scheduler logs")
    logs_parser.set_defaults(func=cmd_logs)

    # test
    test_parser = subparsers.add_parser("test", help="Run scheduler for 60s test")
    test_parser.add_argument("--dotenv", type=str, help="Path to .env file")
    test_parser.set_defaults(func=cmd_test)

    args = parser.parse_args()

    if not args.command:
        # Default to status
        args.func = cmd_status

    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
