#!/usr/bin/env python3
"""
Run Guardian 24/7 - Autonomous Trading Oversight

This script runs the Guardian system continuously, monitoring all
trading systems, making autonomous decisions, and learning from outcomes.

Usage:
    python scripts/run_guardian.py              # Run 24/7
    python scripts/run_guardian.py --status     # Check status
    python scripts/run_guardian.py --once       # Single cycle (testing)
    python scripts/run_guardian.py --stop       # Stop guardian

Author: Kobe Trading System
Created: 2026-01-04
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from guardian import (
    get_guardian,
    get_system_monitor,
    get_self_learner,
    get_alert_manager,
    AlertPriority,
)


def show_status():
    """Show current Guardian status."""
    print("=" * 50)
    print("        GUARDIAN STATUS")
    print("=" * 50)
    print()

    # System health
    monitor = get_system_monitor()
    health = monitor.check_all()

    print(f"System Health: {health.overall_status.value.upper()}")
    print(f"  Healthy: {health.healthy_count}")
    print(f"  Degraded: {health.degraded_count}")
    print(f"  Unhealthy: {health.unhealthy_count}")
    print()

    # Components
    print("Components:")
    for name, comp in sorted(health.components.items()):
        icon = "[OK]" if comp.status.value == "healthy" else "[!!]" if comp.status.value == "degraded" else "[XX]"
        print(f"  {icon} {name}: {comp.message}")
    print()

    # Self-learner
    learner = get_self_learner()
    diagnosis = learner.diagnose_system()
    print(f"Self-Learner:")
    print(f"  Lessons: {diagnosis['lessons_learned']}")
    print(f"  Known Fixes: {diagnosis['known_fixes']}")
    print()

    # Guardian state
    guardian = get_guardian()
    state = guardian.get_state()
    print(f"Guardian:")
    print(f"  Mode: {state.mode.value}")
    print(f"  Running: {'YES' if state.is_running else 'NO'}")
    print(f"  Trading Allowed: {'YES' if state.trading_allowed else 'NO'}")
    print(f"  Check Count: {state.check_count}")
    print(f"  Uptime: {state.uptime_hours:.1f} hours")
    print("=" * 50)


def run_once():
    """Run a single Guardian cycle."""
    print("Running single Guardian cycle...")
    guardian = get_guardian()
    guardian.run_once()
    show_status()
    guardian.stop()


def stop_guardian():
    """Stop the Guardian."""
    print("Stopping Guardian...")
    guardian = get_guardian()
    guardian.stop()
    print("Guardian stopped.")


def run_24_7():
    """Run Guardian 24/7."""
    print("=" * 50)
    print("     KOBE GUARDIAN - 24/7 AUTONOMOUS MODE")
    print("=" * 50)
    print()
    print("Starting Guardian...")
    print("Press Ctrl+C to stop")
    print()

    # Record start in self-learner
    learner = get_self_learner()
    from guardian.self_learner import ChangeType
    learner.record_change(
        change_type=ChangeType.FEATURE_ADD,
        description="Guardian started in 24/7 mode",
        why="User requested continuous autonomous trading oversight",
        what_changed="Guardian process started with continuous monitoring",
        files_affected=["scripts/run_guardian.py"],
    )

    # Send startup alert
    alert_manager = get_alert_manager()
    alert_manager.send_quick(
        AlertPriority.INFO,
        "Guardian 24/7 Started",
        "Autonomous trading oversight is now running continuously",
        "startup",
    )

    # Run guardian
    guardian = get_guardian()
    try:
        guardian.run()
    except KeyboardInterrupt:
        print("\nGuardian interrupted by user")
    finally:
        guardian.stop()
        alert_manager.send_quick(
            AlertPriority.HIGH,
            "Guardian 24/7 Stopped",
            "Autonomous trading oversight has stopped",
            "shutdown",
        )


def main():
    parser = argparse.ArgumentParser(description="Run Kobe Guardian 24/7")
    parser.add_argument("--status", action="store_true", help="Show Guardian status")
    parser.add_argument("--once", action="store_true", help="Run single cycle")
    parser.add_argument("--stop", action="store_true", help="Stop Guardian")

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.once:
        run_once()
    elif args.stop:
        stop_guardian()
    else:
        run_24_7()


if __name__ == "__main__":
    main()
