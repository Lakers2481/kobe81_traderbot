#!/usr/bin/env python3
"""
kill.py - Emergency halt of the Kobe trading system.

Creates a KILL_SWITCH file that prevents all order submissions.
Use this in emergencies to immediately halt trading.

Usage:
    python scripts/kill.py
    python scripts/kill.py --reason "Market flash crash"
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.env_loader import load_env
from core.structured_log import jlog

STATE_DIR = ROOT / "state"
KILL_SWITCH_FILE = STATE_DIR / "KILL_SWITCH"


def send_telegram_alert(message: str) -> bool:
    """
    Send alert via Telegram if configured.
    Returns True if alert was sent successfully.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        return False

    try:
        import urllib.request
        import urllib.parse

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }).encode()

        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except Exception as e:
        print(f"Warning: Failed to send Telegram alert: {e}")
        return False


def create_kill_switch(reason: Optional[str] = None) -> bool:
    """
    Create the kill switch file.
    Returns True if created successfully.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    kill_data = {
        "activated_at": datetime.utcnow().isoformat(),
        "reason": reason or "Emergency halt",
        "activated_by": "kill.py",
    }

    try:
        # Write JSON data to kill switch file
        KILL_SWITCH_FILE.write_text(json.dumps(kill_data, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        print(f"Error creating kill switch file: {e}")
        return False


def get_running_processes() -> dict:
    """Get status of running processes."""
    processes = {"health": None, "runner": None}

    health_pid_file = STATE_DIR / "health.pid"
    runner_pid_file = STATE_DIR / "runner.pid"

    if health_pid_file.exists():
        try:
            processes["health"] = int(health_pid_file.read_text().strip())
        except (ValueError, OSError):
            pass

    if runner_pid_file.exists():
        try:
            processes["runner"] = int(runner_pid_file.read_text().strip())
        except (ValueError, OSError):
            pass

    return processes


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Emergency halt - creates KILL_SWITCH to stop all order submissions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
!!! EMERGENCY USE ONLY !!!

This script creates a KILL_SWITCH file that immediately prevents all
order submissions. Use this when you need to halt trading immediately.

Examples:
    python scripts/kill.py
    python scripts/kill.py --reason "Market crash detected"
    python scripts/kill.py --reason "Account breach suspected" --no-alert

To resume trading after kill switch:
    python scripts/resume.py --confirm
        """
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file"
    )
    ap.add_argument(
        "--reason",
        "-r",
        type=str,
        help="Reason for emergency halt"
    )
    ap.add_argument(
        "--no-alert",
        action="store_true",
        help="Don't send Telegram alert"
    )
    ap.add_argument(
        "--stop-processes",
        action="store_true",
        help="Also stop running processes (health server, runner)"
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="Show current kill switch status"
    )

    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # Status check
    if args.status:
        print("=" * 60)
        print("KOBE TRADING SYSTEM - KILL SWITCH STATUS")
        print("=" * 60)

        if KILL_SWITCH_FILE.exists():
            print("\nStatus: ACTIVE - Trading is HALTED")
            try:
                data = json.loads(KILL_SWITCH_FILE.read_text(encoding="utf-8"))
                print(f"Activated at: {data.get('activated_at', '?')}")
                print(f"Reason: {data.get('reason', '?')}")
            except Exception:
                pass
        else:
            print("\nStatus: INACTIVE - Trading is allowed")
        return

    # Banner
    print("!" * 60)
    print("!!! EMERGENCY KILL SWITCH !!!")
    print("!" * 60)
    print(f"Time: {datetime.utcnow().isoformat()}Z")

    # Check if already active
    if KILL_SWITCH_FILE.exists():
        print("\n!!! KILL SWITCH IS ALREADY ACTIVE !!!")
        try:
            data = json.loads(KILL_SWITCH_FILE.read_text(encoding="utf-8"))
            print(f"Activated at: {data.get('activated_at', '?')}")
            print(f"Reason: {data.get('reason', '?')}")
        except Exception:
            pass
        print("\nNo action needed - trading is already halted.")
        return

    # Create kill switch
    print("\n=== ACTIVATING KILL SWITCH ===")
    reason = args.reason or "Emergency halt"
    print(f"Reason: {reason}")

    if not create_kill_switch(reason):
        print("\nFAILED TO CREATE KILL SWITCH!")
        jlog("kill_switch_failed", reason=reason, level="ERROR")
        sys.exit(1)

    print(f"\nKill switch created: {KILL_SWITCH_FILE}")

    # Log the event
    jlog("kill_switch_activated", reason=reason, level="WARN")

    # Send alert
    if not args.no_alert:
        print("\n=== SENDING ALERTS ===")
        alert_msg = (
            "<b>KOBE TRADING SYSTEM</b>\n"
            "KILL SWITCH ACTIVATED\n\n"
            f"<b>Time:</b> {datetime.utcnow().isoformat()}Z\n"
            f"<b>Reason:</b> {reason}\n\n"
            "All order submissions are now BLOCKED.\n"
            "Use resume.py --confirm to deactivate."
        )

        if send_telegram_alert(alert_msg):
            print("Telegram alert sent")
        else:
            print("Telegram alert not configured or failed")

    # Optionally stop processes
    if args.stop_processes:
        print("\n=== STOPPING PROCESSES ===")
        stop_script = ROOT / "scripts" / "stop.py"
        if stop_script.exists():
            subprocess.run([sys.executable, str(stop_script), "--force"])
        else:
            print("Warning: stop.py not found")

    # Summary
    print("\n" + "!" * 60)
    print("KILL SWITCH ACTIVATED")
    print("!" * 60)
    print("\nAll order submissions are now BLOCKED.")
    print("Existing positions will NOT be liquidated automatically.")
    print("\nTo resume trading:")
    print("  1. Investigate and resolve the issue")
    print("  2. Run: python scripts/resume.py --confirm")

    # Show running processes as warning
    processes = get_running_processes()
    if processes["health"] or processes["runner"]:
        print("\nNote: Processes are still running:")
        if processes["health"]:
            print(f"  - Health server (PID: {processes['health']})")
        if processes["runner"]:
            print(f"  - Runner (PID: {processes['runner']})")
        print("They will respect the kill switch but won't submit new orders.")
        print("Use --stop-processes or stop.py to stop them.")


if __name__ == "__main__":
    main()
