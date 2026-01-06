#!/usr/bin/env python3
"""
Run the Comprehensive Autonomous Brain
=======================================
This is the REAL 24/7 brain that wires EVERYTHING:
- ICT pattern discovery
- External scrapers (Reddit, GitHub)
- Curiosity engine
- Reflection engine
- Research experiments
- Learning from trades
- ALL tasks run without restrictions

Usage:
    python scripts/run_comprehensive_brain.py              # Run forever (60s cycles)
    python scripts/run_comprehensive_brain.py --once       # Single cycle
    python scripts/run_comprehensive_brain.py --status     # Show status
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autonomous.comprehensive_brain import ComprehensiveBrain


def show_banner():
    """Show startup banner."""
    print("=" * 70)
    print("""
    K O B E   C O M P R E H E N S I V E   B R A I N   v2.0

    THE REAL 24/7 AUTONOMOUS TRADING BRAIN

    Always Learning | Never Stopping | Fully Wired
    """)
    print("=" * 70)
    print()
    print("Components Wired:")
    print("  [x] Trading: scan, pregame blueprint, swing scan, watchlist")
    print("  [x] ICT Patterns: turtle soup, order blocks, fair value gaps")
    print("  [x] Research: random params, PF optimization, feature analysis")
    print("  [x] Scrapers: Reddit, GitHub (for external ideas)")
    print("  [x] Cognitive: curiosity engine, reflection engine")
    print("  [x] Memory: episodic memory, pattern rhymes")
    print("  [x] Learning: trade analysis, daily reflection")
    print("  [x] Maintenance: data quality, health checks")
    print()
    print("  NO PHASE RESTRICTIONS - Tasks run 24/7!")
    print("=" * 70)
    print()


def show_status(brain: ComprehensiveBrain):
    """Show current brain status."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    ET = ZoneInfo("America/New_York")
    now = datetime.now(ET)

    print("\n=== COMPREHENSIVE BRAIN STATUS ===\n")
    print(f"Version: {brain.VERSION}")
    print(f"Current Time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Market Phase: {brain._get_phase(now)}")
    print(f"Cycles Completed: {brain.cycles_completed}")

    # Show uptime
    if brain.started_at:
        uptime = (now - brain.started_at).total_seconds() / 3600
        print(f"Uptime: {uptime:.1f} hours")

    # Show last run times
    print("\nLast Run Times:")
    priority_tasks = [
        "scan_signals", "pregame_blueprint", "ict_pattern_discovery",
        "curiosity_engine", "backtest_random_params"
    ]

    for task_id in priority_tasks:
        last = brain.last_run.get(task_id)
        if last:
            elapsed = (now - last).total_seconds() / 60
            print(f"  {task_id}: {elapsed:.0f} min ago")
        else:
            print(f"  {task_id}: never")

    # Show cooldowns
    print("\nCooldowns (minutes):")
    for task_id in priority_tasks:
        cooldown = brain.cooldowns.get(task_id, 0)
        can = brain.can_run(task_id)
        status = "READY" if can else "cooldown"
        print(f"  {task_id}: {cooldown} min ({status})")

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Kobe Comprehensive Brain")
    parser.add_argument("--cycle", type=int, default=60, help="Cycle interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run single cycle and exit")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--quiet", action="store_true", help="Minimal logging")
    args = parser.parse_args()

    # Configure logging
    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    brain = ComprehensiveBrain()

    if args.status:
        show_status(brain)
        return

    if args.once:
        show_banner()
        print("Running single cycle...\n")
        result = brain.run_cycle()
        print("\n=== CYCLE RESULT ===")
        print(f"Cycle: {result['cycle']}")
        print(f"Tasks Run: {result['tasks_run']}")
        print(f"Tasks Skipped (cooldown): {result['tasks_skipped']}")
        print("\nResults:")
        for task_id, res in result['results'].items():
            status = res.get('status', res.get('skipped', 'unknown'))
            print(f"  {task_id}: {status}")
        return

    # Run forever
    show_banner()
    brain.run_forever(cycle_seconds=args.cycle)


if __name__ == "__main__":
    main()
