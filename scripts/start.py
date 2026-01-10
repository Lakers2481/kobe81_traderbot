#!/usr/bin/env python3
"""
start.py - Start the Kobe trading system.

Usage:
    python scripts/start.py --mode paper
    python scripts/start.py --mode live
    python scripts/start.py --mode paper --skip-preflight
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from core.structured_log import jlog

STATE_DIR = ROOT / "state"
PID_DIR = STATE_DIR
HEALTH_PID_FILE = PID_DIR / "health.pid"
RUNNER_PID_FILE = PID_DIR / "runner.pid"
KILL_SWITCH_FILE = STATE_DIR / "KILL_SWITCH"

DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "optionable_liquid_800.csv"
DEFAULT_CAP = 50


def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)


def check_kill_switch() -> bool:
    """Check if kill switch is active."""
    if KILL_SWITCH_FILE.exists():
        print("\n!!! KILL SWITCH ACTIVE !!!")
        print(f"File: {KILL_SWITCH_FILE}")
        try:
            content = KILL_SWITCH_FILE.read_text(encoding="utf-8").strip()
            if content:
                print(f"Reason: {content}")
        except Exception:
            pass
        print("\nCannot start while kill switch is active.")
        print("Use 'python scripts/resume.py --confirm' to deactivate.")
        return True
    return False


def check_already_running() -> Dict[str, bool]:
    """Check if processes are already running."""
    status = {"health": False, "runner": False}

    if HEALTH_PID_FILE.exists():
        try:
            pid = int(HEALTH_PID_FILE.read_text().strip())
            # Check if process exists
            if sys.platform == "win32":
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}"],
                    capture_output=True, text=True
                )
                status["health"] = str(pid) in result.stdout
            else:
                os.kill(pid, 0)
                status["health"] = True
        except (ValueError, OSError, ProcessLookupError):
            # Process not running, clean up stale PID file
            HEALTH_PID_FILE.unlink(missing_ok=True)

    if RUNNER_PID_FILE.exists():
        try:
            pid = int(RUNNER_PID_FILE.read_text().strip())
            if sys.platform == "win32":
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}"],
                    capture_output=True, text=True
                )
                status["runner"] = str(pid) in result.stdout
            else:
                os.kill(pid, 0)
                status["runner"] = True
        except (ValueError, OSError, ProcessLookupError):
            RUNNER_PID_FILE.unlink(missing_ok=True)

    return status


def run_preflight(dotenv: Path) -> bool:
    """Run preflight checks."""
    print("\n=== PREFLIGHT CHECKS ===")
    preflight_script = ROOT / "scripts" / "preflight.py"

    if not preflight_script.exists():
        print(f"Warning: Preflight script not found: {preflight_script}")
        return True

    cmd = [sys.executable, str(preflight_script), "--dotenv", str(dotenv)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print("\nPreflight checks FAILED")
        return False

    print("Preflight checks PASSED")
    return True


def start_health_server(port: int, dotenv: Path) -> Optional[int]:
    """Start the health check server."""
    print(f"\n=== STARTING HEALTH SERVER (port {port}) ===")

    health_script = ROOT / "scripts" / "start_health.py"
    if not health_script.exists():
        print(f"Warning: Health script not found: {health_script}")
        return None

    # Start as background process
    if sys.platform == "win32":
        # Windows: use subprocess with CREATE_NEW_PROCESS_GROUP
        import subprocess
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200

        cmd = [sys.executable, str(health_script), "--port", str(port)]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        )
        pid = process.pid
    else:
        # Unix: fork and exec
        cmd = [sys.executable, str(health_script), "--port", str(port)]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        pid = process.pid

    # Save PID
    HEALTH_PID_FILE.write_text(str(pid))
    print(f"Health server started (PID: {pid})")
    jlog("health_server_started", pid=pid, port=port)

    return pid


def start_runner(
    mode: str,
    universe: Path,
    cap: int,
    scan_times: str,
    dotenv: Path,
    once: bool = False
) -> Optional[int]:
    """Start the trading runner."""
    print(f"\n=== STARTING RUNNER ({mode.upper()} mode) ===")

    runner_script = ROOT / "scripts" / "runner.py"
    if not runner_script.exists():
        print(f"Error: Runner script not found: {runner_script}")
        return None

    if not universe.exists():
        print(f"Error: Universe file not found: {universe}")
        return None

    cmd = [
        sys.executable,
        str(runner_script),
        "--mode", mode,
        "--universe", str(universe),
        "--cap", str(cap),
        "--scan-times", scan_times,
        "--dotenv", str(dotenv),
    ]

    if once:
        cmd.append("--once")
        # Run synchronously for --once
        print("Running single scan...")
        subprocess.run(cmd, capture_output=False)
        return None

    # Start as background process
    if sys.platform == "win32":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        )
        pid = process.pid
    else:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        pid = process.pid

    # Save PID
    RUNNER_PID_FILE.write_text(str(pid))
    print(f"Runner started (PID: {pid})")
    print(f"  Mode: {mode}")
    print(f"  Universe: {universe}")
    print(f"  Cap: {cap}")
    print(f"  Scan times: {scan_times}")

    jlog("runner_started", pid=pid, mode=mode, universe=str(universe), cap=cap, scan_times=scan_times)

    return pid


def save_startup_state(mode: str, health_pid: Optional[int], runner_pid: Optional[int]) -> None:
    """Save startup state to file."""
    state = {
        "started_at": datetime.utcnow().isoformat(),
        "mode": mode,
        "health_pid": health_pid,
        "runner_pid": runner_pid,
    }
    state_file = STATE_DIR / "startup_state.json"
    state_file.write_text(json.dumps(state, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Start the Kobe trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/start.py --mode paper
    python scripts/start.py --mode live
    python scripts/start.py --mode paper --skip-preflight
    python scripts/start.py --mode paper --once
    python scripts/start.py --mode paper --cap 100 --scan-times "09:35,15:55"
        """
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file"
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (default: paper)"
    )
    ap.add_argument(
        "--universe",
        type=str,
        default=str(DEFAULT_UNIVERSE),
        help=f"Path to universe file (default: {DEFAULT_UNIVERSE})"
    )
    ap.add_argument(
        "--cap",
        type=int,
        default=DEFAULT_CAP,
        help=f"Max symbols to trade (default: {DEFAULT_CAP})"
    )
    ap.add_argument(
        "--scan-times",
        type=str,
        default="09:35,10:30,15:55",
        help="Scan times in HH:MM format, comma-separated"
    )
    ap.add_argument(
        "--health-port",
        type=int,
        default=8000,
        help="Health server port (default: 8000)"
    )
    ap.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip preflight checks"
    )
    ap.add_argument(
        "--no-health",
        action="store_true",
        help="Don't start health server"
    )
    ap.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (don't start background runner)"
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force start even if processes already running"
    )

    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    print("=" * 60)
    print("KOBE TRADING SYSTEM - STARTUP")
    print(f"Time: {datetime.utcnow().isoformat()}Z")
    print(f"Mode: {args.mode.upper()}")
    print("=" * 60)

    # Ensure directories exist
    ensure_directories()

    # Check kill switch
    if check_kill_switch():
        sys.exit(1)

    # Check if already running
    status = check_already_running()
    if (status["health"] or status["runner"]) and not args.force:
        print("\n!!! PROCESSES ALREADY RUNNING !!!")
        if status["health"]:
            pid = HEALTH_PID_FILE.read_text().strip()
            print(f"  Health server: PID {pid}")
        if status["runner"]:
            pid = RUNNER_PID_FILE.read_text().strip()
            print(f"  Runner: PID {pid}")
        print("\nUse --force to start anyway, or use stop.py first.")
        sys.exit(1)

    # Live mode warning
    if args.mode == "live":
        print("\n" + "!" * 60)
        print("!!! LIVE TRADING MODE - REAL MONEY !!!")
        print("!" * 60)
        print("\nThis will execute real trades with real money.")
        print("Press Ctrl+C within 5 seconds to cancel...")
        try:
            import time
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nStartup cancelled.")
            sys.exit(0)
        print("\nProceeding with LIVE trading...")

    # Run preflight
    if not args.skip_preflight:
        if not run_preflight(dotenv):
            print("\nStartup aborted due to preflight failure.")
            sys.exit(2)
    else:
        print("\nSkipping preflight checks (--skip-preflight)")

    # Start health server
    health_pid = None
    if not args.no_health and not args.once:
        health_pid = start_health_server(args.health_port, dotenv)

    # Start runner
    runner_pid = start_runner(
        mode=args.mode,
        universe=Path(args.universe),
        cap=args.cap,
        scan_times=args.scan_times,
        dotenv=dotenv,
        once=args.once
    )

    # Save state
    if not args.once:
        save_startup_state(args.mode, health_pid, runner_pid)

    # Summary
    print("\n" + "=" * 60)
    if args.once:
        print("Single scan completed.")
    else:
        print("KOBE TRADING SYSTEM STARTED")
        print("=" * 60)
        print(f"Mode: {args.mode.upper()}")
        if health_pid:
            print(f"Health server: http://localhost:{args.health_port} (PID: {health_pid})")
        if runner_pid:
            print(f"Runner PID: {runner_pid}")
        print("\nUse 'python scripts/stop.py' to stop the system")
        print("Use 'python scripts/kill.py' for emergency halt")

    jlog("system_started", mode=args.mode, health_pid=health_pid, runner_pid=runner_pid)


if __name__ == "__main__":
    main()
