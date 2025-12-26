#!/usr/bin/env python3
"""
restart.py - Gracefully restart the Kobe trading system.

Usage:
    python restart.py                  # Restart in paper mode (default)
    python restart.py --mode paper     # Restart paper trading
    python restart.py --mode live      # Restart live trading
    python restart.py --mode all       # Restart all services
    python restart.py --no-confirm     # Skip confirmation prompts
    python restart.py --dotenv PATH    # Load environment from .env file
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env

# Process tracking file
PROCESS_FILE = ROOT / "state" / "running_processes.json"
KILL_SWITCH = ROOT / "state" / "KILL_SWITCH"

# Known trading processes
TRADING_PROCESSES = {
    "paper": {
        "script": ROOT / "scripts" / "run_paper_trade.py",
        "name": "Paper Trading",
    },
    "live": {
        "script": ROOT / "scripts" / "run_live_trade_micro.py",
        "name": "Live Trading",
    },
    "runner": {
        "script": ROOT / "scripts" / "runner.py",
        "name": "24/7 Runner",
    },
    "health": {
        "script": ROOT / "scripts" / "start_health.py",
        "name": "Health Monitor",
    },
}


def log_event(event: str, **fields: Any) -> None:
    """Log event to structured log."""
    try:
        from core.structured_log import jlog
        jlog(event, **fields)
    except Exception:
        ts = datetime.utcnow().isoformat()
        print(f"[{ts}] {event}: {fields}")


def load_process_info() -> Dict[str, Any]:
    """Load running process information."""
    if not PROCESS_FILE.exists():
        return {}
    try:
        return json.loads(PROCESS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return {}


def save_process_info(info: Dict[str, Any]) -> None:
    """Save process information."""
    PROCESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROCESS_FILE.write_text(json.dumps(info, indent=2), encoding="utf-8")


def find_running_processes() -> List[Dict[str, Any]]:
    """Find currently running trading processes."""
    running: List[Dict[str, Any]] = []

    try:
        import psutil

        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                cmdline_str = " ".join(cmdline)

                for mode, config in TRADING_PROCESSES.items():
                    script_name = config["script"].name
                    if script_name in cmdline_str:
                        running.append({
                            "mode": mode,
                            "name": config["name"],
                            "pid": proc.info["pid"],
                            "started": datetime.fromtimestamp(
                                proc.info["create_time"]
                            ).isoformat(),
                            "cmdline": cmdline_str[:100],
                        })
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    except ImportError:
        # Fallback without psutil
        print("[WARN] psutil not installed - process detection limited")

        # Try tasklist on Windows
        try:
            result = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.strip().split("\n"):
                if "python" in line.lower():
                    parts = line.strip('"').split('","')
                    if len(parts) >= 2:
                        running.append({
                            "mode": "unknown",
                            "name": "Python Process",
                            "pid": int(parts[1]) if parts[1].isdigit() else 0,
                            "cmdline": line[:100],
                        })
        except Exception:
            pass

    return running


def activate_kill_switch() -> None:
    """Activate the kill switch to stop trading."""
    KILL_SWITCH.parent.mkdir(parents=True, exist_ok=True)
    KILL_SWITCH.write_text(
        json.dumps({
            "activated": datetime.utcnow().isoformat(),
            "reason": "restart initiated",
        }),
        encoding="utf-8",
    )
    log_event("kill_switch_activated", reason="restart")
    print("[KILL SWITCH] Activated")


def deactivate_kill_switch() -> None:
    """Deactivate the kill switch."""
    if KILL_SWITCH.exists():
        KILL_SWITCH.unlink()
        log_event("kill_switch_deactivated")
        print("[KILL SWITCH] Deactivated")


def stop_process(pid: int, graceful_timeout: int = 10) -> bool:
    """Stop a process gracefully, then forcefully if needed."""
    try:
        import psutil
        proc = psutil.Process(pid)

        # Try graceful termination first
        print(f"  Sending SIGTERM to PID {pid}...")
        proc.terminate()

        try:
            proc.wait(timeout=graceful_timeout)
            print(f"  [STOPPED] PID {pid} terminated gracefully")
            return True
        except psutil.TimeoutExpired:
            print(f"  [WARN] PID {pid} did not stop, sending SIGKILL...")
            proc.kill()
            proc.wait(timeout=5)
            print(f"  [KILLED] PID {pid} force killed")
            return True

    except ImportError:
        # Fallback without psutil
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(graceful_timeout)
            try:
                os.kill(pid, 0)  # Check if still running
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            return True
        except OSError as e:
            print(f"  [ERROR] Failed to stop PID {pid}: {e}")
            return False

    except Exception as e:
        print(f"  [ERROR] Failed to stop PID {pid}: {e}")
        return False


def stop_trading(mode: str = "all", confirm: bool = True) -> bool:
    """Stop trading processes."""
    print("=" * 60)
    print("  STOPPING TRADING SYSTEM")
    print("=" * 60)

    # Activate kill switch first
    activate_kill_switch()
    time.sleep(1)  # Give processes time to see the kill switch

    # Find running processes
    running = find_running_processes()

    if mode != "all":
        running = [p for p in running if p["mode"] == mode]

    if not running:
        print("\n[INFO] No matching trading processes found")
        return True

    print("\n--- Running Processes ---")
    for proc in running:
        print(f"  {proc['name']} (PID: {proc['pid']})")
        if "started" in proc:
            print(f"    Started: {proc['started']}")

    if confirm:
        try:
            response = input("\nStop these processes? [y/N]: ")
            if response.lower() not in ("y", "yes"):
                print("Aborted.")
                deactivate_kill_switch()
                return False
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            deactivate_kill_switch()
            return False

    # Stop each process
    print("\n--- Stopping Processes ---")
    success_count = 0
    for proc in running:
        pid = proc["pid"]
        if stop_process(pid):
            success_count += 1
            log_event("process_stopped", mode=proc["mode"], pid=pid)

    print(f"\n[RESULT] Stopped {success_count}/{len(running)} processes")
    return success_count == len(running)


def wait_for_confirmation(timeout: int = 30) -> bool:
    """Wait for confirmation before starting."""
    print(f"\n[WAIT] Waiting {timeout}s before restart...")
    print("       Press Enter to continue immediately, or Ctrl+C to abort")

    try:
        import select
        import sys

        if sys.platform == "win32":
            # Windows doesn't support select on stdin
            for i in range(timeout):
                time.sleep(1)
                print(f"\r       {timeout - i - 1}s remaining...", end="", flush=True)
            print()
            return True
        else:
            # Unix systems
            for i in range(timeout):
                if select.select([sys.stdin], [], [], 1)[0]:
                    sys.stdin.readline()
                    return True
                print(f"\r       {timeout - i - 1}s remaining...", end="", flush=True)
            print()
            return True

    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        return False


def start_trading(
    mode: str,
    universe: Optional[str] = None,
    cap: int = 50,
    dotenv: Path = None,
) -> Optional[int]:
    """Start a trading process."""
    print("=" * 60)
    print(f"  STARTING {mode.upper()} TRADING")
    print("=" * 60)

    # Deactivate kill switch before starting
    deactivate_kill_switch()

    if mode not in TRADING_PROCESSES:
        print(f"[ERROR] Unknown mode: {mode}")
        return None

    config = TRADING_PROCESSES[mode]
    script = config["script"]

    if not script.exists():
        print(f"[ERROR] Script not found: {script}")
        return None

    # Build command
    cmd = [sys.executable, str(script)]

    if mode in ("paper", "live", "runner"):
        if not universe:
            # Try to find a universe file
            universe_dir = ROOT / "data" / "universe"
            if universe_dir.exists():
                universes = list(universe_dir.glob("*.csv"))
                if universes:
                    universe = str(universes[0])
                    print(f"[INFO] Using universe: {universe}")

        if universe:
            cmd.extend(["--universe", universe])

        cmd.extend(["--cap", str(cap)])

    if dotenv:
        cmd.extend(["--dotenv", str(dotenv)])

    print(f"\nCommand: {' '.join(cmd[:5])}...")

    try:
        # Start process in background
        if sys.platform == "win32":
            # Windows
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            # Unix
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        pid = proc.pid
        print(f"\n[STARTED] {config['name']} with PID {pid}")

        # Save process info
        info = load_process_info()
        info[mode] = {
            "pid": pid,
            "started": datetime.utcnow().isoformat(),
            "script": str(script),
            "universe": universe,
        }
        save_process_info(info)

        log_event("process_started", mode=mode, pid=pid, universe=universe)

        return pid

    except Exception as e:
        print(f"[ERROR] Failed to start: {e}")
        log_event("process_start_failed", mode=mode, error=str(e))
        return None


def restart_trading(
    mode: str = "paper",
    universe: Optional[str] = None,
    cap: int = 50,
    dotenv: Optional[Path] = None,
    confirm: bool = True,
    wait_time: int = 5,
) -> bool:
    """Perform a full restart: stop, wait, start."""
    print("=" * 60)
    print("  KOBE TRADING SYSTEM - RESTART")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Time: {datetime.now().isoformat()}")

    # Step 1: Stop
    print("\n[STEP 1/3] Stopping current processes...")
    if not stop_trading(mode=mode, confirm=confirm):
        print("[ABORT] Failed to stop processes")
        return False

    # Step 2: Wait
    print(f"\n[STEP 2/3] Waiting {wait_time} seconds...")
    if confirm:
        if not wait_for_confirmation(wait_time):
            return False
    else:
        time.sleep(wait_time)

    # Step 3: Start
    print("\n[STEP 3/3] Starting processes...")

    if mode == "all":
        # Start paper trading by default
        pid = start_trading("paper", universe=universe, cap=cap, dotenv=dotenv)
    else:
        pid = start_trading(mode, universe=universe, cap=cap, dotenv=dotenv)

    if pid:
        print("\n" + "=" * 60)
        print("  RESTART COMPLETE")
        print("=" * 60)
        print(f"  Mode: {mode}")
        print(f"  PID:  {pid}")
        print(f"  Time: {datetime.now().isoformat()}")
        return True
    else:
        print("\n[ERROR] Restart failed")
        return False


def show_status() -> None:
    """Show current trading system status."""
    print("=" * 60)
    print("  TRADING SYSTEM STATUS")
    print("=" * 60)

    # Check kill switch
    if KILL_SWITCH.exists():
        print("\n[KILL SWITCH] ACTIVE")
        try:
            data = json.loads(KILL_SWITCH.read_text(encoding="utf-8"))
            print(f"  Activated: {data.get('activated', 'unknown')}")
            print(f"  Reason: {data.get('reason', 'unknown')}")
        except Exception:
            pass
    else:
        print("\n[KILL SWITCH] Inactive")

    # Find running processes
    running = find_running_processes()

    if running:
        print("\n--- Running Processes ---")
        for proc in running:
            print(f"\n  {proc['name']}")
            print(f"    PID: {proc['pid']}")
            if "started" in proc:
                print(f"    Started: {proc['started']}")
            if "mode" in proc:
                print(f"    Mode: {proc['mode']}")
    else:
        print("\n[INFO] No trading processes detected")

    # Check saved process info
    saved_info = load_process_info()
    if saved_info:
        print("\n--- Last Known Processes ---")
        for mode, info in saved_info.items():
            print(f"\n  {mode}:")
            print(f"    PID: {info.get('pid', 'unknown')}")
            print(f"    Started: {info.get('started', 'unknown')}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Kobe Trading System - Restart Utility"
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["paper", "live", "runner", "health", "all"],
        default="paper",
        help="Trading mode to restart",
    )
    ap.add_argument(
        "--stop",
        action="store_true",
        help="Only stop (don't restart)",
    )
    ap.add_argument(
        "--start",
        action="store_true",
        help="Only start (don't stop first)",
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="Show current status",
    )
    ap.add_argument(
        "--universe",
        type=str,
        help="Universe file path",
    )
    ap.add_argument(
        "--cap",
        type=int,
        default=50,
        help="Universe cap",
    )
    ap.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompts",
    )
    ap.add_argument(
        "--wait",
        type=int,
        default=5,
        help="Wait time between stop and start (seconds)",
    )
    args = ap.parse_args()

    # Load environment if specified
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # Execute requested action
    if args.status:
        show_status()
    elif args.stop:
        stop_trading(mode=args.mode, confirm=not args.no_confirm)
    elif args.start:
        start_trading(
            mode=args.mode,
            universe=args.universe,
            cap=args.cap,
            dotenv=dotenv,
        )
    else:
        # Default: full restart
        restart_trading(
            mode=args.mode,
            universe=args.universe,
            cap=args.cap,
            dotenv=dotenv,
            confirm=not args.no_confirm,
            wait_time=args.wait,
        )


if __name__ == "__main__":
    main()
