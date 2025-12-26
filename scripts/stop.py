#!/usr/bin/env python3
"""
stop.py - Graceful shutdown of the Kobe trading system.

Usage:
    python scripts/stop.py
    python scripts/stop.py --force
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from core.structured_log import jlog

STATE_DIR = ROOT / "state"
PID_DIR = STATE_DIR
HEALTH_PID_FILE = PID_DIR / "health.pid"
RUNNER_PID_FILE = PID_DIR / "runner.pid"
STARTUP_STATE_FILE = STATE_DIR / "startup_state.json"


def get_pid_from_file(pid_file: Path) -> Optional[int]:
    """Read PID from file."""
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None


def is_process_running(pid: int) -> bool:
    """Check if a process is running."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True
            )
            return str(pid) in result.stdout
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def stop_process(pid: int, name: str, force: bool = False, timeout: int = 10) -> bool:
    """Stop a process gracefully or forcefully."""
    if not is_process_running(pid):
        print(f"  {name}: Not running (PID {pid})")
        return True

    print(f"  {name}: Stopping (PID {pid})...")

    if sys.platform == "win32":
        # Windows: use taskkill
        if force:
            cmd = ["taskkill", "/F", "/PID", str(pid)]
        else:
            cmd = ["taskkill", "/PID", str(pid)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            if not force:
                # Try force kill
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    else:
        # Unix: send SIGTERM, then SIGKILL if needed
        try:
            os.kill(pid, signal.SIGTERM)

            # Wait for process to exit
            for _ in range(timeout):
                time.sleep(1)
                if not is_process_running(pid):
                    break
            else:
                if force or not is_process_running(pid):
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(1)
        except (OSError, ProcessLookupError):
            pass

    # Verify stopped
    if is_process_running(pid):
        print(f"  {name}: Failed to stop (PID {pid})")
        return False
    else:
        print(f"  {name}: Stopped")
        return True


def save_state_snapshot() -> None:
    """Save current state before shutdown."""
    snapshot_file = STATE_DIR / "shutdown_snapshot.json"
    snapshot = {
        "shutdown_at": datetime.utcnow().isoformat(),
        "state_files": [],
    }

    # List state files
    if STATE_DIR.exists():
        for f in STATE_DIR.iterdir():
            if f.is_file() and f.suffix == ".json":
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    snapshot["state_files"].append({
                        "name": f.name,
                        "content_summary": str(data)[:200]
                    })
                except Exception:
                    snapshot["state_files"].append({"name": f.name, "error": "Could not read"})

    try:
        snapshot_file.write_text(json.dumps(snapshot, indent=2))
    except Exception as e:
        print(f"Warning: Could not save shutdown snapshot: {e}")


def cleanup_pid_files() -> None:
    """Remove PID files."""
    for pid_file in [HEALTH_PID_FILE, RUNNER_PID_FILE]:
        if pid_file.exists():
            try:
                pid_file.unlink()
            except Exception:
                pass


def get_running_status() -> Dict[str, Any]:
    """Get current running status."""
    status = {
        "health": {"running": False, "pid": None},
        "runner": {"running": False, "pid": None},
    }

    health_pid = get_pid_from_file(HEALTH_PID_FILE)
    if health_pid:
        status["health"]["pid"] = health_pid
        status["health"]["running"] = is_process_running(health_pid)

    runner_pid = get_pid_from_file(RUNNER_PID_FILE)
    if runner_pid:
        status["runner"]["pid"] = runner_pid
        status["runner"]["running"] = is_process_running(runner_pid)

    # Read startup state
    if STARTUP_STATE_FILE.exists():
        try:
            startup = json.loads(STARTUP_STATE_FILE.read_text())
            status["startup_state"] = startup
        except Exception:
            pass

    return status


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Graceful shutdown of the Kobe trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/stop.py
    python scripts/stop.py --force
    python scripts/stop.py --status
        """
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file"
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force kill processes immediately"
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="Show running status without stopping"
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout in seconds for graceful shutdown (default: 10)"
    )
    ap.add_argument(
        "--no-save-state",
        action="store_true",
        help="Don't save state snapshot before shutdown"
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output status in JSON format"
    )

    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    print("=" * 60)
    print("KOBE TRADING SYSTEM - SHUTDOWN")
    print(f"Time: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

    # Get status
    status = get_running_status()

    if args.status or args.json:
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("\n=== SYSTEM STATUS ===")
            print(f"Health server: {'Running' if status['health']['running'] else 'Not running'}")
            if status['health']['pid']:
                print(f"  PID: {status['health']['pid']}")
            print(f"Runner: {'Running' if status['runner']['running'] else 'Not running'}")
            if status['runner']['pid']:
                print(f"  PID: {status['runner']['pid']}")
            if 'startup_state' in status:
                print(f"\nStarted at: {status['startup_state'].get('started_at', '?')}")
                print(f"Mode: {status['startup_state'].get('mode', '?')}")
        return

    # Check if anything is running
    if not status["health"]["running"] and not status["runner"]["running"]:
        print("\nNo processes are running.")
        cleanup_pid_files()
        return

    # Save state before shutdown
    if not args.no_save_state:
        print("\n=== SAVING STATE ===")
        save_state_snapshot()
        print("State snapshot saved")

    # Stop processes
    print("\n=== STOPPING PROCESSES ===")

    stopped_all = True

    # Stop runner first
    if status["runner"]["pid"]:
        if not stop_process(status["runner"]["pid"], "Runner", args.force, args.timeout):
            stopped_all = False

    # Stop health server
    if status["health"]["pid"]:
        if not stop_process(status["health"]["pid"], "Health server", args.force, args.timeout):
            stopped_all = False

    # Cleanup PID files
    cleanup_pid_files()

    # Log shutdown
    jlog("system_stopped", force=args.force, success=stopped_all)

    # Summary
    print("\n" + "=" * 60)
    if stopped_all:
        print("KOBE TRADING SYSTEM STOPPED")
        print("=" * 60)
        print("\nAll processes stopped successfully.")
    else:
        print("SHUTDOWN INCOMPLETE")
        print("=" * 60)
        print("\nSome processes may still be running.")
        print("Try 'python scripts/stop.py --force' to force kill.")

    # Remove startup state
    if STARTUP_STATE_FILE.exists():
        try:
            STARTUP_STATE_FILE.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
