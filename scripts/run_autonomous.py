#!/usr/bin/env python3
"""
Run Kobe's Autonomous Brain 24/7.

This script starts the autonomous brain that:
- Always knows what time/day/season it is
- Never stops working
- Continuously improves through research and learning
- Runs backtests with new strategies
- Learns from every trade

Usage:
    python scripts/run_autonomous.py                    # Run forever (60s cycles)
    python scripts/run_autonomous.py --cycle 30        # 30-second cycles
    python scripts/run_autonomous.py --status          # Show status
    python scripts/run_autonomous.py --once            # Single cycle (testing)
    python scripts/run_autonomous.py --daemon          # Run as background daemon
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autonomous.brain import AutonomousBrain
from autonomous.awareness import get_context


def setup_logging(log_level: str = "INFO", log_file: bool = True):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"autonomous_{datetime.now().strftime('%Y%m%d')}.log"
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def print_banner():
    """Print startup banner."""
    print("""
========================================================================

    K O B E   2 4 / 7   A U T O N O M O U S   B R A I N

    Always Aware | Always Learning | Never Stops

========================================================================
""")


def show_awareness():
    """Show current awareness state."""
    ctx = get_context()
    print(f"""
Current Awareness
========================================================
Time:           {ctx.timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}
Day:            {ctx.day_of_week}
Market Phase:   {ctx.phase.value}
Season:         {ctx.season.value}
Work Mode:      {ctx.work_mode.value}

Market State
--------------------------------------------------------
Market Open:    {'YES' if ctx.is_market_open else 'NO'}
Trading OK:     {'YES' if ctx.trading_allowed else 'NO'}
Weekend:        {'YES' if ctx.is_weekend else 'NO'}
Holiday:        {'YES' if ctx.is_holiday else 'NO'}
FOMC Day:       {'YES' if ctx.is_fomc_day else 'NO'}
OpEx Day:       {'YES' if ctx.is_opex_day else 'NO'}

{f'Minutes to Open:  {ctx.minutes_to_open}' if ctx.minutes_to_open else ''}
{f'Minutes to Close: {ctx.minutes_to_close}' if ctx.minutes_to_close else ''}

Recommended Actions
--------------------------------------------------------""")
    for action in ctx.recommended_actions:
        print(f"  - {action}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Kobe Autonomous Brain - 24/7 Self-Improving Trading System"
    )
    parser.add_argument(
        "--cycle", type=int, default=60,
        help="Cycle interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run single cycle and exit (for testing)"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current status and exit"
    )
    parser.add_argument(
        "--awareness", action="store_true",
        help="Show current awareness state and exit"
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Run as background daemon"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Show awareness and exit
    if args.awareness:
        show_awareness()
        return

    # Setup logging
    setup_logging(args.log_level)
    logging.getLogger(__name__)

    # Create brain
    brain = AutonomousBrain()

    # Status mode
    if args.status:
        status = brain.get_status()
        print(json.dumps(status, indent=2))
        return

    # Single cycle mode
    if args.once:
        print_banner()
        show_awareness()
        print("Running single cycle...")
        result = brain.run_single_cycle()
        print("\nResult:")
        print(json.dumps(result, indent=2))
        return

    # Daemon mode
    if args.daemon:
        # TODO: Implement proper daemonization for Windows
        print("Daemon mode not yet implemented for Windows")
        print("Use: start /B python scripts/run_autonomous.py")
        return

    # Full run mode
    print_banner()
    show_awareness()

    print(f"""
Starting Autonomous Brain
========================================================
Cycle Interval: {args.cycle} seconds
Log Level:      {args.log_level}

The brain will now run continuously, always:
  - Knowing what time/day/season it is
  - Deciding what to work on based on context
  - Running experiments and backtests
  - Learning from trades
  - Improving itself

Press Ctrl+C to stop gracefully.
========================================================
""")

    try:
        brain.run_forever(cycle_seconds=args.cycle)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        print("Autonomous brain stopped.")


if __name__ == "__main__":
    main()
