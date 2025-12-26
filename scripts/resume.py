#!/usr/bin/env python3
"""
resume.py - Deactivate the kill switch and resume trading.

Requires explicit --confirm flag as a safety measure.

Usage:
    python scripts/resume.py --confirm
"""
from __future__ import annotations

import argparse
import json
import os
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
KILL_SWITCH_HISTORY_FILE = STATE_DIR / "kill_switch_history.jsonl"


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


def run_safety_checks() -> dict:
    """
    Run safety checks before allowing resume.
    Returns dict with check results.
    """
    checks = {
        "all_passed": True,
        "results": []
    }

    # Check 1: Environment variables
    required_env = ["ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"]
    missing_env = [k for k in required_env if not os.getenv(k)]
    if missing_env:
        checks["results"].append({
            "name": "Environment Variables",
            "passed": False,
            "message": f"Missing: {', '.join(missing_env)}"
        })
        checks["all_passed"] = False
    else:
        checks["results"].append({
            "name": "Environment Variables",
            "passed": True,
            "message": "All required variables present"
        })

    # Check 2: Broker connectivity (optional, just warn)
    try:
        import requests
        base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
        r = requests.get(
            f"{base}/v2/account",
            headers={
                "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY_ID", ""),
                "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET_KEY", "")
            },
            timeout=5
        )
        if r.status_code == 200:
            checks["results"].append({
                "name": "Broker Connectivity",
                "passed": True,
                "message": "Alpaca API responding"
            })
        else:
            checks["results"].append({
                "name": "Broker Connectivity",
                "passed": False,
                "message": f"Alpaca returned status {r.status_code}"
            })
            # Don't fail on broker check, just warn
    except ImportError:
        checks["results"].append({
            "name": "Broker Connectivity",
            "passed": None,
            "message": "requests module not available (skipped)"
        })
    except Exception as e:
        checks["results"].append({
            "name": "Broker Connectivity",
            "passed": False,
            "message": f"Error: {str(e)[:50]}"
        })

    # Check 3: State directory writable
    try:
        test_file = STATE_DIR / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        checks["results"].append({
            "name": "State Directory Writable",
            "passed": True,
            "message": str(STATE_DIR)
        })
    except Exception as e:
        checks["results"].append({
            "name": "State Directory Writable",
            "passed": False,
            "message": f"Error: {str(e)[:50]}"
        })
        checks["all_passed"] = False

    # Check 4: Check how long kill switch has been active
    if KILL_SWITCH_FILE.exists():
        try:
            data = json.loads(KILL_SWITCH_FILE.read_text(encoding="utf-8"))
            activated_at = datetime.fromisoformat(data.get("activated_at", ""))
            duration = datetime.utcnow() - activated_at
            hours = duration.total_seconds() / 3600

            if hours < 0.1:  # Less than 6 minutes
                checks["results"].append({
                    "name": "Cool-down Period",
                    "passed": True,
                    "message": f"Active for {duration.total_seconds():.0f} seconds"
                })
            else:
                checks["results"].append({
                    "name": "Cool-down Period",
                    "passed": True,
                    "message": f"Active for {hours:.1f} hours"
                })
        except Exception:
            checks["results"].append({
                "name": "Cool-down Period",
                "passed": None,
                "message": "Could not determine duration"
            })

    return checks


def archive_kill_switch() -> None:
    """Archive the kill switch data to history file."""
    if not KILL_SWITCH_FILE.exists():
        return

    try:
        data = json.loads(KILL_SWITCH_FILE.read_text(encoding="utf-8"))
        data["deactivated_at"] = datetime.utcnow().isoformat()

        with open(KILL_SWITCH_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"Warning: Could not archive kill switch: {e}")


def remove_kill_switch() -> bool:
    """Remove the kill switch file."""
    if not KILL_SWITCH_FILE.exists():
        return True

    try:
        KILL_SWITCH_FILE.unlink()
        return True
    except Exception as e:
        print(f"Error removing kill switch: {e}")
        return False


def get_kill_switch_info() -> Optional[dict]:
    """Get information about the current kill switch."""
    if not KILL_SWITCH_FILE.exists():
        return None

    try:
        return json.loads(KILL_SWITCH_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"activated_at": "unknown", "reason": "unknown"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Deactivate kill switch and resume trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
!!! SAFETY CHECK REQUIRED !!!

This script removes the KILL_SWITCH file and allows trading to resume.
Requires explicit --confirm flag as a safety measure.

Examples:
    python scripts/resume.py --status
    python scripts/resume.py --confirm
    python scripts/resume.py --confirm --no-alert

Before resuming:
    1. Investigate why the kill switch was activated
    2. Verify the issue has been resolved
    3. Check broker connectivity
    4. Review any pending positions
        """
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file"
    )
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="Required flag to confirm resume (safety measure)"
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="Show current kill switch status"
    )
    ap.add_argument(
        "--no-alert",
        action="store_true",
        help="Don't send Telegram alert"
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Skip safety checks (use with caution)"
    )
    ap.add_argument(
        "--history",
        action="store_true",
        help="Show kill switch history"
    )

    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    print("=" * 60)
    print("KOBE TRADING SYSTEM - RESUME")
    print(f"Time: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

    # Show history
    if args.history:
        print("\n=== KILL SWITCH HISTORY ===")
        if not KILL_SWITCH_HISTORY_FILE.exists():
            print("No history found")
        else:
            for line in KILL_SWITCH_HISTORY_FILE.read_text().strip().split("\n"):
                if line:
                    try:
                        data = json.loads(line)
                        activated = data.get("activated_at", "?")
                        deactivated = data.get("deactivated_at", "?")
                        reason = data.get("reason", "?")
                        print(f"  {activated} -> {deactivated}")
                        print(f"    Reason: {reason}")
                    except Exception:
                        print(f"  (unparseable entry)")
        return

    # Check current status
    kill_info = get_kill_switch_info()

    if args.status:
        print("\n=== KILL SWITCH STATUS ===")
        if kill_info:
            print("Status: ACTIVE - Trading is HALTED")
            print(f"Activated at: {kill_info.get('activated_at', '?')}")
            print(f"Reason: {kill_info.get('reason', '?')}")

            # Calculate duration
            try:
                activated_at = datetime.fromisoformat(kill_info.get("activated_at", ""))
                duration = datetime.utcnow() - activated_at
                hours = duration.total_seconds() / 3600
                print(f"Duration: {hours:.1f} hours")
            except Exception:
                pass
        else:
            print("Status: INACTIVE - Trading is allowed")
        return

    # Check if kill switch is active
    if not kill_info:
        print("\nKill switch is not active. Trading is already allowed.")
        return

    # Show current status
    print("\n=== CURRENT KILL SWITCH ===")
    print(f"Activated at: {kill_info.get('activated_at', '?')}")
    print(f"Reason: {kill_info.get('reason', '?')}")

    # Check for --confirm flag
    if not args.confirm:
        print("\n" + "!" * 60)
        print("!!! CONFIRMATION REQUIRED !!!")
        print("!" * 60)
        print("\nTo resume trading, run:")
        print("  python scripts/resume.py --confirm")
        print("\nBefore resuming, verify:")
        print("  1. The issue that triggered the kill switch has been resolved")
        print("  2. Broker connectivity is working")
        print("  3. Market conditions are acceptable")
        return

    # Run safety checks
    if not args.force:
        print("\n=== SAFETY CHECKS ===")
        checks = run_safety_checks()

        for check in checks["results"]:
            status = "PASS" if check["passed"] else ("SKIP" if check["passed"] is None else "FAIL")
            print(f"  [{status}] {check['name']}: {check['message']}")

        if not checks["all_passed"]:
            print("\n!!! SAFETY CHECKS FAILED !!!")
            print("Cannot resume trading until all checks pass.")
            print("Use --force to override (not recommended)")
            return
    else:
        print("\n=== SAFETY CHECKS SKIPPED (--force) ===")

    # Archive kill switch
    print("\n=== DEACTIVATING KILL SWITCH ===")
    archive_kill_switch()

    # Remove kill switch
    if not remove_kill_switch():
        print("\nFAILED TO REMOVE KILL SWITCH!")
        jlog("kill_switch_resume_failed", level="ERROR")
        sys.exit(1)

    print("Kill switch removed successfully")

    # Log the event
    jlog("kill_switch_deactivated", reason=kill_info.get("reason", "unknown"))

    # Send alert
    if not args.no_alert:
        print("\n=== SENDING ALERTS ===")
        alert_msg = (
            "<b>KOBE TRADING SYSTEM</b>\n"
            "KILL SWITCH DEACTIVATED\n\n"
            f"<b>Time:</b> {datetime.utcnow().isoformat()}Z\n"
            f"<b>Original reason:</b> {kill_info.get('reason', 'unknown')}\n\n"
            "Trading can now resume.\n"
            "Use start.py to restart the system."
        )

        if send_telegram_alert(alert_msg):
            print("Telegram alert sent")
        else:
            print("Telegram alert not configured or failed")

    # Summary
    print("\n" + "=" * 60)
    print("KILL SWITCH DEACTIVATED")
    print("=" * 60)
    print("\nTrading can now resume.")
    print("\nNext steps:")
    print("  1. Start the system: python scripts/start.py --mode paper")
    print("  2. Monitor logs: python scripts/logs.py --tail 50")
    print("  3. Check positions: python scripts/state.py --positions")


if __name__ == "__main__":
    main()
