#!/usr/bin/env python3
"""
Unified CLI Entrypoint for Kobe Autonomous Trading System.

This is the main entry point for all autonomous operations.
Run with: python -m autonomous.run [command]

Commands:
    --start         Start 24/7 autonomous operation
    --stop          Graceful shutdown
    --status        Show current system status
    --demo          Run 5-minute interview demo
    --weekend       Run weekend deep research mode
    --awareness     Show current time/market awareness
    --research      Show research status and discoveries
    --health        Health check (exit 0 if healthy)
    --tour          Interactive system tour
    --stage N       Run with N-stock universe (default 50)
    --full          Run with full 900-stock universe

Examples:
    python -m autonomous.run --status
    python -m autonomous.run --demo
    python -m autonomous.run --start
    python -m autonomous.run --weekend

Author: Kobe Trading System
Version: 1.0.0
Last Updated: 2026-01-03
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import safety first
from safety import PAPER_ONLY, get_trading_mode, log_safety_status

# Import autonomous components
from autonomous.brain import AutonomousBrain
from autonomous.awareness import MarketCalendarAwareness as MarketAwareness, get_context
from autonomous.scheduler import AutonomousScheduler as TaskScheduler


class KobeRunner:
    """Unified runner for Kobe autonomous operations."""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.state_dir = self.project_root / "state" / "autonomous"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # PID file for single instance
        self.pid_file = self.state_dir / "kobe.pid"

        # Shutdown flag
        self._shutdown_requested = False

    def _check_already_running(self) -> bool:
        """Check if another instance is running."""
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
                # Check if process exists (Windows compatible)
                if sys.platform == "win32":
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    handle = kernel32.OpenProcess(1, 0, pid)
                    if handle:
                        kernel32.CloseHandle(handle)
                        return True
                else:
                    os.kill(pid, 0)
                    return True
            except (ValueError, OSError, FileNotFoundError):
                # Process doesn't exist, remove stale PID file
                self.pid_file.unlink(missing_ok=True)
        return False

    def _write_pid(self):
        """Write current PID to file."""
        self.pid_file.write_text(str(os.getpid()))

    def _remove_pid(self):
        """Remove PID file."""
        self.pid_file.unlink(missing_ok=True)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n[SHUTDOWN] Signal received, initiating graceful shutdown...")
        self._shutdown_requested = True

    def cmd_status(self) -> int:
        """Show current system status."""
        print("=" * 60)
        print("KOBE AUTONOMOUS TRADING SYSTEM - STATUS")
        print("=" * 60)

        # Safety status
        mode = get_trading_mode()
        print(f"\n[SAFETY]")
        print(f"  Mode: {mode['mode_str'].upper()}")
        print(f"  Paper Only: {mode['paper_only']}")
        print(f"  Kill Switch: {'ACTIVE' if mode['kill_switch'] else 'inactive'}")

        # Heartbeat status
        heartbeat_file = self.state_dir / "heartbeat.json"
        if heartbeat_file.exists():
            try:
                hb = json.loads(heartbeat_file.read_text())
                print(f"\n[BRAIN]")
                print(f"  Alive: {hb.get('alive', False)}")
                print(f"  Phase: {hb.get('phase', 'unknown')}")
                print(f"  Work Mode: {hb.get('work_mode', 'unknown')}")
                print(f"  Cycles: {hb.get('cycles', 0)}")
                print(f"  Uptime: {hb.get('uptime_hours', 0):.1f} hours")
                print(f"  Last Update: {hb.get('timestamp', 'unknown')}")
            except Exception as e:
                print(f"\n[BRAIN] Error reading heartbeat: {e}")
        else:
            print(f"\n[BRAIN] Not running (no heartbeat file)")

        # Brain state
        brain_state_file = self.state_dir / "brain_state.json"
        if brain_state_file.exists():
            try:
                state = json.loads(brain_state_file.read_text())
                print(f"\n[STATE]")
                print(f"  Total Cycles: {state.get('cycles_completed', 0)}")
                print(f"  Last Task: {state.get('last_task', 'none')}")
                print(f"  Errors Today: {state.get('errors_today', 0)}")
            except Exception:
                pass

        # Research status
        research_file = self.state_dir / "research" / "research_state.json"
        if research_file.exists():
            try:
                research = json.loads(research_file.read_text())
                print(f"\n[RESEARCH]")
                print(f"  Experiments Run: {research.get('experiments_run', 0)}")
                print(f"  Discoveries: {research.get('discoveries_count', 0)}")
                print(f"  Best Improvement: {research.get('best_improvement', 0):.1%}")
            except Exception:
                pass

        # Market awareness
        try:
            context = get_context()
            print(f"\n[AWARENESS]")
            print(f"  Phase: {context.phase.value if hasattr(context, 'phase') else 'unknown'}")
            print(f"  Work Mode: {context.work_mode.value if hasattr(context, 'work_mode') else 'unknown'}")
            print(f"  Day Type: {context.day_type if hasattr(context, 'day_type') else 'unknown'}")
            print(f"  Season: {context.season.value if hasattr(context, 'season') else 'normal'}")
            print(f"  Market Open: {context.market_open if hasattr(context, 'market_open') else False}")
        except Exception as e:
            print(f"\n[AWARENESS] Error: {e}")

        print("\n" + "=" * 60)
        return 0

    def cmd_awareness(self) -> int:
        """Show detailed market awareness."""
        print("=" * 60)
        print("KOBE MARKET AWARENESS")
        print("=" * 60)

        try:
            context = get_context()
            now = datetime.now()

            print(f"\n[TIME]")
            print(f"  Current: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Timezone: America/New_York")

            print(f"\n[PHASE]")
            print(f"  Current Phase: {context.phase.value if hasattr(context, 'phase') else 'unknown'}")
            print(f"  Work Mode: {context.work_mode.value if hasattr(context, 'work_mode') else 'unknown'}")
            print(f"  Day Type: {context.day_type if hasattr(context, 'day_type') else 'unknown'}")

            print(f"\n[MARKET]")
            print(f"  Market Open: {context.market_open if hasattr(context, 'market_open') else False}")
            print(f"  Trading Allowed: {context.trading_allowed if hasattr(context, 'trading_allowed') else False}")
            print(f"  Kill Zone: {context.kill_zone if hasattr(context, 'kill_zone') else 'none'}")

            print(f"\n[SEASON]")
            print(f"  Current: {context.season.value if hasattr(context, 'season') else 'normal'}")
            print(f"  Is Weekend: {context.is_weekend if hasattr(context, 'is_weekend') else False}")

            print(f"\n[TASKS]")
            print(f"  Context available for task selection")

        except Exception as e:
            print(f"Error getting awareness: {e}")
            return 1

        print("\n" + "=" * 60)
        return 0

    def cmd_research(self) -> int:
        """Show research status and discoveries."""
        print("=" * 60)
        print("KOBE RESEARCH STATUS")
        print("=" * 60)

        # Research state
        research_file = self.state_dir / "research" / "research_state.json"
        if research_file.exists():
            try:
                research = json.loads(research_file.read_text())
                print(f"\n[EXPERIMENTS]")
                print(f"  Total Run: {research.get('experiments_run', 0)}")
                print(f"  In Progress: {research.get('experiments_in_progress', 0)}")
                print(f"  Success Rate: {research.get('experiment_success_rate', 0):.1%}")

                print(f"\n[DISCOVERIES]")
                print(f"  Total: {research.get('discoveries_count', 0)}")
                print(f"  Validated: {research.get('validated_discoveries', 0)}")
                print(f"  Promoted: {research.get('promoted_discoveries', 0)}")

                print(f"\n[BEST RESULTS]")
                print(f"  Best Improvement: {research.get('best_improvement', 0):.1%}")
                print(f"  Best Win Rate: {research.get('best_win_rate', 0):.1%}")
                print(f"  Best Profit Factor: {research.get('best_profit_factor', 0):.2f}")

                # Recent experiments
                recent = research.get('recent_experiments', [])[:5]
                if recent:
                    print(f"\n[RECENT EXPERIMENTS]")
                    for exp in recent:
                        print(f"  - {exp.get('name', 'unknown')}: {exp.get('result', 'N/A')}")

            except Exception as e:
                print(f"Error reading research state: {e}")
        else:
            print("\nNo research state found. Brain may not have run experiments yet.")

        # Discovered patterns
        patterns_dir = self.project_root / "reports"
        pattern_files = list(patterns_dir.glob("quant_pattern_analysis_*.json"))
        if pattern_files:
            latest = max(pattern_files, key=lambda p: p.stat().st_mtime)
            try:
                patterns = json.loads(latest.read_text())
                top_patterns = patterns.get('top_50_patterns', [])[:10]
                if top_patterns:
                    print(f"\n[TOP PATTERNS]")
                    for p in top_patterns:
                        print(f"  {p['symbol']}: {p['streak']}d streak, "
                              f"{p['bounce_5d']:.0f}% bounce, +{p['avg_5d']:.1f}% avg")
            except Exception:
                pass

        print("\n" + "=" * 60)
        return 0

    def cmd_demo(self, duration_minutes: int = 5) -> int:
        """Run interview demo."""
        print("=" * 60)
        print("KOBE AUTONOMOUS TRADING SYSTEM - DEMO MODE")
        print(f"Duration: {duration_minutes} minutes")
        print("=" * 60)

        steps = [
            ("System Status", self.cmd_status),
            ("Market Awareness", self.cmd_awareness),
            ("Research Status", self.cmd_research),
        ]

        for i, (name, func) in enumerate(steps, 1):
            print(f"\n[DEMO STEP {i}/{len(steps)}] {name}")
            print("-" * 40)
            func()
            time.sleep(2)

        # Run a quick scan if available
        print("\n[DEMO] Running quick scan...")
        scan_script = self.project_root / "scripts" / "scan.py"
        if scan_script.exists():
            import subprocess
            try:
                result = subprocess.run(
                    [sys.executable, str(scan_script), "--cap", "50", "--deterministic", "--top3"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    print(result.stdout)
                else:
                    print(f"Scan failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                print("Scan timed out")
            except Exception as e:
                print(f"Scan error: {e}")

        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        return 0

    def cmd_start(self, universe_cap: int = 50) -> int:
        """Start 24/7 autonomous operation."""
        print("=" * 60)
        print("KOBE AUTONOMOUS BRAIN - STARTING")
        print("=" * 60)

        # Check if already running
        if self._check_already_running():
            print("[ERROR] Another instance is already running!")
            print(f"PID file: {self.pid_file}")
            return 1

        # Log safety status
        print(log_safety_status())

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Write PID
        self._write_pid()

        try:
            # Initialize brain
            brain = AutonomousBrain(universe_cap=universe_cap)

            print(f"\n[START] Brain initialized with {universe_cap}-stock universe")
            print("[START] Press Ctrl+C for graceful shutdown")
            print("=" * 60)

            # Run forever
            while not self._shutdown_requested:
                brain.think()
                time.sleep(60)  # 60-second cycles

        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Keyboard interrupt received")
        except Exception as e:
            print(f"\n[ERROR] Brain crashed: {e}")
            return 1
        finally:
            self._remove_pid()
            print("[SHUTDOWN] Kobe brain stopped")

        return 0

    def cmd_stop(self) -> int:
        """Stop running brain instance."""
        if not self.pid_file.exists():
            print("[INFO] No running instance found")
            return 0

        try:
            pid = int(self.pid_file.read_text().strip())
            print(f"[STOP] Sending SIGTERM to PID {pid}")

            if sys.platform == "win32":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(1, 0, pid)
                if handle:
                    kernel32.TerminateProcess(handle, 0)
                    kernel32.CloseHandle(handle)
            else:
                os.kill(pid, signal.SIGTERM)

            # Wait for shutdown
            for _ in range(30):
                if not self.pid_file.exists():
                    print("[STOP] Brain stopped successfully")
                    return 0
                time.sleep(1)

            print("[WARN] Brain did not stop within 30 seconds")
            return 1

        except Exception as e:
            print(f"[ERROR] Failed to stop: {e}")
            return 1

    def cmd_health(self) -> int:
        """Health check (exit 0 if healthy)."""
        heartbeat_file = self.state_dir / "heartbeat.json"

        if not heartbeat_file.exists():
            print("[UNHEALTHY] No heartbeat file")
            return 1

        try:
            hb = json.loads(heartbeat_file.read_text())
            if not hb.get('alive', False):
                print("[UNHEALTHY] Brain not alive")
                return 1

            # Check heartbeat age
            ts = datetime.fromisoformat(hb['timestamp'].replace('Z', '+00:00'))
            age_minutes = (datetime.now(ts.tzinfo) - ts).total_seconds() / 60

            if age_minutes > 5:
                print(f"[UNHEALTHY] Stale heartbeat ({age_minutes:.1f} min old)")
                return 1

            print("[HEALTHY] Brain is alive and responsive")
            return 0

        except Exception as e:
            print(f"[UNHEALTHY] Error checking health: {e}")
            return 1

    def cmd_weekend(self) -> int:
        """Run weekend deep research mode."""
        print("=" * 60)
        print("KOBE WEEKEND DEEP RESEARCH MODE")
        print("=" * 60)

        # Check safety
        mode = get_trading_mode()
        print(f"\n[SAFETY] Mode: {mode['mode_str']}, Paper Only: {mode['paper_only']}")

        # Initialize with weekend config
        try:
            brain = AutonomousBrain(
                universe_cap=900,
                force_mode="deep_research"
            )

            print("[WEEKEND] Starting deep research cycle...")
            print("[WEEKEND] Tasks: extended backtests, ML retraining, pattern discovery")
            print("-" * 60)

            # Run extended research cycle
            brain.run_weekend_research()

            print("\n" + "=" * 60)
            print("WEEKEND RESEARCH COMPLETE")
            print("=" * 60)

            return 0

        except Exception as e:
            print(f"[ERROR] Weekend research failed: {e}")
            return 1

    def cmd_tour(self) -> int:
        """Interactive system tour."""
        print("=" * 60)
        print("KOBE AUTONOMOUS TRADING SYSTEM - TOUR")
        print("=" * 60)

        sections = [
            ("1. ARCHITECTURE", [
                "autonomous/brain.py     - 24/7 orchestrator (542 lines)",
                "autonomous/scheduler.py - Task queue (43 tasks)",
                "autonomous/awareness.py - Market context (9 phases, 8 seasons)",
                "autonomous/handlers.py  - 35+ task handlers",
            ]),
            ("2. AGENTS", [
                "agents/scout_agent.py   - IdeaCard discovery",
                "agents/auditor_agent.py - Bias detection",
                "agents/risk_agent.py    - 5-gate validation",
                "agents/orchestrator.py  - Pipeline coordination",
            ]),
            ("3. QUANT GATES", [
                "Gate 0: Sanity    - Lookahead/leakage detection",
                "Gate 1: Baseline  - Min 50% WR, 1.0 PF",
                "Gate 2: Robustness- Train/test correlation",
                "Gate 3: Risk      - Max 25% DD, 100+ trades",
                "Gate 4: Multiple  - FDR < 10%",
            ]),
            ("4. STRATEGIES", [
                "DualStrategyScanner - Production scanner",
                "  - IBS+RSI (59.9% WR, 1.46 PF)",
                "  - Turtle Soup (61.0% WR, 1.37 PF)",
                "  - 0.3 ATR sweep filter",
            ]),
            ("5. SAFETY", [
                "PAPER_ONLY = True   - Hardcoded constant",
                "Kill switch file    - state/KILL_SWITCH",
                "Risk limits         - 2% per trade, 20% daily",
                "Kill zones          - No trades 9:30-10:00",
            ]),
        ]

        for title, items in sections:
            print(f"\n{title}")
            print("-" * 40)
            for item in items:
                print(f"  {item}")
            time.sleep(1)

        print("\n" + "=" * 60)
        print("TOUR COMPLETE - For more: see docs/ARCHITECTURE.md")
        print("=" * 60)
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Kobe Autonomous Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m autonomous.run --status     Show system status
  python -m autonomous.run --demo       Run 5-minute demo
  python -m autonomous.run --start      Start 24/7 operation
  python -m autonomous.run --weekend    Weekend deep research
        """
    )

    # Commands
    parser.add_argument("--start", action="store_true", help="Start 24/7 operation")
    parser.add_argument("--stop", action="store_true", help="Stop running brain")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--demo", action="store_true", help="Run interview demo")
    parser.add_argument("--weekend", action="store_true", help="Weekend deep research")
    parser.add_argument("--deepverify", action="store_true", help="Alias for --weekend")
    parser.add_argument("--awareness", action="store_true", help="Show market awareness")
    parser.add_argument("--research", action="store_true", help="Show research status")
    parser.add_argument("--health", action="store_true", help="Health check")
    parser.add_argument("--tour", action="store_true", help="System tour")

    # Options
    parser.add_argument("--stage", type=int, metavar="N", help="Universe cap (default 50)")
    parser.add_argument("--full", action="store_true", help="Full 900-stock universe")

    args = parser.parse_args()

    # Determine universe cap
    universe_cap = 50
    if args.stage:
        universe_cap = args.stage
    elif args.full:
        universe_cap = 900

    # Create runner
    runner = KobeRunner()

    # Execute command
    if args.stop:
        return runner.cmd_stop()
    elif args.status:
        return runner.cmd_status()
    elif args.demo:
        return runner.cmd_demo()
    elif args.weekend or args.deepverify:
        return runner.cmd_weekend()
    elif args.awareness:
        return runner.cmd_awareness()
    elif args.research:
        return runner.cmd_research()
    elif args.health:
        return runner.cmd_health()
    elif args.tour:
        return runner.cmd_tour()
    elif args.start:
        return runner.cmd_start(universe_cap)
    else:
        # Default to status
        return runner.cmd_status()


if __name__ == "__main__":
    sys.exit(main())
