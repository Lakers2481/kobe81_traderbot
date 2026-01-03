"""
The Autonomous Brain - Kobe's 24/7 Self-Aware Core.

This is the central orchestrator that:
- Always knows what time/day/season it is
- Decides what to work on based on context
- Never stops working
- Continuously improves itself
"""

import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from zoneinfo import ZoneInfo

from .awareness import ContextBuilder, MarketContext, WorkMode, MarketPhase
from .scheduler import AutonomousScheduler, Task, TaskPriority, TaskCategory
from .research import ResearchEngine
from .learning import LearningEngine

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class AutonomousBrain:
    """
    Kobe's Autonomous Brain - Always On, Always Learning.

    The brain orchestrates all autonomous activities:
    - Context awareness (time, day, season, market state)
    - Task scheduling and execution
    - Self-improvement through research
    - Learning from experience
    - 24/7 operation without human intervention
    """

    VERSION = "1.0.0"

    def __init__(self, state_dir: Optional[Path] = None):
        if state_dir is None:
            state_dir = Path("state/autonomous")
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self.context_builder = ContextBuilder()
        self.scheduler = AutonomousScheduler(state_dir)
        self.research = ResearchEngine(state_dir / "research")
        self.learning = LearningEngine(state_dir / "learning")

        # State
        self.running = False
        self.started_at: Optional[datetime] = None
        self.cycles_completed = 0
        self.last_task_time: Optional[datetime] = None

        # Heartbeat file for external monitoring
        self.heartbeat_file = state_dir / "heartbeat.json"

        # Load state
        self._load_state()

        logger.info(f"Autonomous Brain v{self.VERSION} initialized")

    def _load_state(self):
        """Load brain state from disk."""
        state_file = self.state_dir / "brain_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                self.cycles_completed = data.get("cycles_completed", 0)
            except Exception as e:
                logger.warning(f"Could not load brain state: {e}")

    def save_state(self):
        """Save brain state to disk."""
        state_file = self.state_dir / "brain_state.json"
        data = {
            "version": self.VERSION,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "cycles_completed": self.cycles_completed,
            "last_task_time": self.last_task_time.isoformat() if self.last_task_time else None,
            "updated_at": datetime.now(ET).isoformat(),
        }
        state_file.write_text(json.dumps(data, indent=2))

    def update_heartbeat(self):
        """Update heartbeat file for external monitoring."""
        context = self.get_context()
        data = {
            "alive": True,
            "timestamp": datetime.now(ET).isoformat(),
            "phase": context.phase.value,
            "work_mode": context.work_mode.value,
            "cycles": self.cycles_completed,
            "uptime_hours": self._get_uptime_hours(),
        }
        self.heartbeat_file.write_text(json.dumps(data, indent=2))

    def _get_uptime_hours(self) -> float:
        """Get uptime in hours."""
        if self.started_at is None:
            return 0.0
        delta = datetime.now(ET) - self.started_at
        return delta.total_seconds() / 3600

    def get_context(self) -> MarketContext:
        """Get current market context."""
        return self.context_builder.get_context()

    def think(self) -> Dict[str, Any]:
        """
        Main thinking loop - decide what to do.

        Returns information about what Kobe is thinking/doing.
        """
        context = self.get_context()

        # Log current awareness
        logger.debug(
            f"Awareness: {context.day_of_week} {context.timestamp.strftime('%H:%M')} ET | "
            f"Phase: {context.phase.value} | Mode: {context.work_mode.value}"
        )

        # Get next task
        task = self.scheduler.get_next_task()

        result = {
            "timestamp": context.timestamp.isoformat(),
            "phase": context.phase.value,
            "work_mode": context.work_mode.value,
            "recommended_actions": context.recommended_actions,
            "task": None,
            "task_result": None,
        }

        if task:
            result["task"] = task.name
            logger.info(f"Executing: {task.name}")
            task_result = self.scheduler.execute_task(task)
            result["task_result"] = task_result
            self.last_task_time = datetime.now(ET)
        else:
            # No task eligible - do background work based on mode
            background_result = self._do_background_work(context)
            result["background_work"] = background_result

        self.cycles_completed += 1
        self.update_heartbeat()

        return result

    def _do_background_work(self, context: MarketContext) -> Dict[str, Any]:
        """Do background work when no tasks are eligible."""
        work_mode = context.work_mode

        if work_mode == WorkMode.DEEP_RESEARCH:
            # Weekend/holiday: Run experiments
            logger.info("Deep research mode: Running experiment")
            return self.research.backtest_random_params()

        elif work_mode == WorkMode.RESEARCH:
            # Regular research time
            logger.info("Research mode: Exploring parameters")
            return self.research.backtest_random_params()

        elif work_mode == WorkMode.LEARNING:
            # Learning time: Analyze trades
            logger.info("Learning mode: Analyzing trades")
            return self.learning.analyze_trades()

        elif work_mode == WorkMode.OPTIMIZATION:
            # Night: Run optimizations
            logger.info("Optimization mode: Feature analysis")
            return self.research.analyze_features()

        else:
            # Monitoring or active trading - just wait
            return {"status": "waiting", "mode": work_mode.value}

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive brain status."""
        context = self.get_context()
        research_summary = self.research.get_research_summary()
        learning_summary = self.learning.get_learning_summary()

        return {
            "version": self.VERSION,
            "running": self.running,
            "uptime_hours": self._get_uptime_hours(),
            "cycles_completed": self.cycles_completed,

            "awareness": {
                "timestamp": context.timestamp.isoformat(),
                "day": context.day_of_week,
                "phase": context.phase.value,
                "season": context.season.value,
                "work_mode": context.work_mode.value,
                "market_open": context.is_market_open,
                "trading_allowed": context.trading_allowed,
            },

            "scheduler": {
                "pending_tasks": self.scheduler.queue.get_pending_count(),
            },

            "research": research_summary,
            "learning": learning_summary,

            "recommended_actions": context.recommended_actions,
        }

    def run_forever(self, cycle_seconds: int = 60):
        """
        Run the brain forever.

        This is the main loop that keeps Kobe alive 24/7.
        """
        self.running = True
        self.started_at = datetime.now(ET)

        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Shutdown signal received")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("=" * 60)
        logger.info("AUTONOMOUS BRAIN STARTING")
        logger.info(f"Version: {self.VERSION}")
        logger.info(f"Cycle interval: {cycle_seconds}s")
        logger.info("=" * 60)

        # Initial status
        status = self.get_status()
        logger.info(f"Current phase: {status['awareness']['phase']}")
        logger.info(f"Work mode: {status['awareness']['work_mode']}")
        logger.info(f"Actions: {status['recommended_actions']}")

        try:
            while self.running:
                try:
                    # Think and act
                    result = self.think()

                    if result.get("task"):
                        logger.info(f"Completed: {result['task']}")

                    # Save state periodically
                    if self.cycles_completed % 10 == 0:
                        self.save_state()

                    # Log status every hour
                    if self.cycles_completed % 60 == 0:
                        status = self.get_status()
                        logger.info(
                            f"Hourly status: {self.cycles_completed} cycles, "
                            f"uptime {status['uptime_hours']:.1f}h, "
                            f"mode: {status['awareness']['work_mode']}"
                        )

                except Exception as e:
                    logger.error(f"Error in think cycle: {e}")

                # Wait for next cycle
                time.sleep(cycle_seconds)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        finally:
            self.running = False
            self.save_state()
            logger.info("Autonomous brain stopped")

    def run_single_cycle(self) -> Dict[str, Any]:
        """Run a single think cycle (for testing)."""
        return self.think()


def run_brain():
    """Entry point for running the autonomous brain."""
    import argparse

    parser = argparse.ArgumentParser(description="Kobe Autonomous Brain")
    parser.add_argument(
        "--cycle", type=int, default=60,
        help="Cycle interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run single cycle and exit"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show status and exit"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    brain = AutonomousBrain()

    if args.status:
        status = brain.get_status()
        print(json.dumps(status, indent=2))
        return

    if args.once:
        result = brain.run_single_cycle()
        print(json.dumps(result, indent=2))
        return

    # Run forever
    brain.run_forever(cycle_seconds=args.cycle)


if __name__ == "__main__":
    run_brain()
