"""
Autonomous Task Scheduler for Kobe.

Manages the queue of tasks Kobe should be working on 24/7.
Tasks are prioritized based on time, context, and importance.
"""

import json
import logging
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from zoneinfo import ZoneInfo

from .awareness import (
    MarketPhase, WorkMode, MarketContext, ContextBuilder
)

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1      # Must run immediately (kill switch, reconciliation)
    HIGH = 2          # Important, run soon (trading, position management)
    NORMAL = 3        # Standard tasks (backtesting, research)
    LOW = 4           # Nice to have (cleanup, optimization)
    BACKGROUND = 5    # Run when nothing else to do


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskCategory(Enum):
    """Categories of tasks."""
    TRADING = "trading"           # Active trading activities
    MONITORING = "monitoring"     # Position/market watching
    RESEARCH = "research"         # Backtesting, analysis
    LEARNING = "learning"         # Cognitive updates
    OPTIMIZATION = "optimization" # Parameter tuning
    MAINTENANCE = "maintenance"   # System upkeep
    DATA = "data"                # Data updates
    DISCOVERY = "discovery"       # Finding new strategies


@dataclass
class Task:
    """A task for Kobe to execute."""
    id: str
    name: str
    category: TaskCategory
    priority: TaskPriority
    description: str

    # Execution
    handler: Optional[str] = None  # Module.function to call
    params: Dict[str, Any] = field(default_factory=dict)

    # Scheduling
    scheduled_time: Optional[datetime] = None
    valid_phases: List[MarketPhase] = field(default_factory=list)
    valid_modes: List[WorkMode] = field(default_factory=list)
    cooldown_minutes: int = 0  # Minimum time between runs

    # State
    status: TaskStatus = TaskStatus.PENDING
    last_run: Optional[datetime] = None
    run_count: int = 0
    last_result: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None

    # Recurrence
    recurring: bool = False
    recurrence_minutes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "priority": self.priority.value,
            "description": self.description,
            "handler": self.handler,
            "params": self.params,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "valid_phases": [p.value for p in self.valid_phases],
            "valid_modes": [m.value for m in self.valid_modes],
            "cooldown_minutes": self.cooldown_minutes,
            "status": self.status.value,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
            "recurring": self.recurring,
        }

    def can_run(self, context: MarketContext) -> bool:
        """Check if task can run in current context."""
        # Check phase restrictions
        if self.valid_phases and context.phase not in self.valid_phases:
            return False

        # Check mode restrictions
        if self.valid_modes and context.work_mode not in self.valid_modes:
            return False

        # Check cooldown
        if self.last_run and self.cooldown_minutes > 0:
            elapsed = (context.timestamp - self.last_run).total_seconds() / 60
            if elapsed < self.cooldown_minutes:
                return False

        # Check scheduled time
        if self.scheduled_time and context.timestamp < self.scheduled_time:
            return False

        return True


class TaskQueue:
    """Priority queue of tasks."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.tasks: Dict[str, Task] = {}
        self.state_file = state_dir / "task_queue.json"
        self._load_state()

    def _load_state(self):
        """Load task state from disk."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                # Restore task metadata (not full tasks)
                for task_data in data.get("tasks", []):
                    task_id = task_data.get("id")
                    if task_id in self.tasks:
                        if task_data.get("last_run"):
                            dt = datetime.fromisoformat(task_data["last_run"])
                            # Ensure timezone-aware (FIX 2026-01-05)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=ET)
                            self.tasks[task_id].last_run = dt
                        else:
                            self.tasks[task_id].last_run = None
                        self.tasks[task_id].run_count = task_data.get("run_count", 0)
            except Exception as e:
                logger.warning(f"Could not load task state: {e}")

    def save_state(self):
        """Save task state to disk."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "updated_at": datetime.now(ET).isoformat(),
            "tasks": [t.to_dict() for t in self.tasks.values()],
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    def add_task(self, task: Task):
        """Add a task to the queue."""
        self.tasks[task.id] = task
        logger.debug(f"Added task: {task.name}")

    def remove_task(self, task_id: str):
        """Remove a task from the queue."""
        if task_id in self.tasks:
            del self.tasks[task_id]

    def get_next_task(self, context: MarketContext) -> Optional[Task]:
        """Get the next task to run based on context and priority."""
        eligible = []

        for task in self.tasks.values():
            if task.status == TaskStatus.RUNNING:
                continue  # Skip running tasks
            if task.can_run(context):
                eligible.append(task)

        if not eligible:
            return None

        # Sort by priority (lower = more important)
        # Use timezone-aware datetime.min (FIX 2026-01-05)
        min_dt = datetime.min.replace(tzinfo=ET)
        eligible.sort(key=lambda t: (t.priority.value, t.last_run or min_dt))

        return eligible[0]

    def get_tasks_by_category(self, category: TaskCategory) -> List[Task]:
        """Get all tasks in a category."""
        return [t for t in self.tasks.values() if t.category == category]

    def get_pending_count(self) -> int:
        """Get count of pending tasks."""
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)


class AutonomousScheduler:
    """
    The main scheduler that keeps Kobe working 24/7.

    This scheduler:
    - Maintains a queue of tasks
    - Selects appropriate tasks based on time/context
    - Tracks task history and results
    - Never stops working
    """

    def __init__(self, state_dir: Optional[Path] = None):
        if state_dir is None:
            state_dir = Path("state/autonomous")
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.context_builder = ContextBuilder()
        self.queue = TaskQueue(state_dir)
        self.running = False

        # Task registry - maps handler names to functions
        self._handlers: Dict[str, Callable] = {}

        # Initialize default tasks
        self._init_default_tasks()

    def _init_default_tasks(self):
        """Initialize the default task set."""
        default_tasks = [
            # === TRADING TASKS ===
            Task(
                id="scan_signals",
                name="Scan for Trading Signals",
                category=TaskCategory.TRADING,
                priority=TaskPriority.HIGH,
                description="Run the scanner to find trading opportunities",
                handler="scripts.scan:run_scan",
                valid_phases=[MarketPhase.MARKET_MORNING, MarketPhase.MARKET_AFTERNOON],
                valid_modes=[WorkMode.ACTIVE_TRADING],
                cooldown_minutes=30,
                recurring=True,
            ),
            Task(
                id="check_positions",
                name="Check Position P&L",
                category=TaskCategory.MONITORING,
                priority=TaskPriority.HIGH,
                description="Monitor open positions and P&L",
                handler="scripts.positions:check_positions",
                valid_modes=[WorkMode.ACTIVE_TRADING, WorkMode.MONITORING],
                cooldown_minutes=5,
                recurring=True,
            ),
            Task(
                id="reconcile_broker",
                name="Reconcile Broker Positions",
                category=TaskCategory.MONITORING,
                priority=TaskPriority.CRITICAL,
                description="Ensure local and broker positions match",
                handler="scripts.reconcile_alpaca:reconcile",
                cooldown_minutes=60,
                recurring=True,
            ),

            # === RESEARCH TASKS ===
            Task(
                id="backtest_random_params",
                name="Backtest Random Parameters",
                category=TaskCategory.RESEARCH,
                priority=TaskPriority.NORMAL,
                description="Test random parameter variations to find improvements",
                handler="autonomous.research:backtest_random_params",
                valid_modes=[WorkMode.RESEARCH, WorkMode.DEEP_RESEARCH],
                cooldown_minutes=120,
                recurring=True,
            ),
            Task(
                id="feature_importance",
                name="Analyze Feature Importance",
                category=TaskCategory.RESEARCH,
                priority=TaskPriority.NORMAL,
                description="Run SHAP analysis on recent predictions",
                handler="autonomous.research:analyze_features",
                valid_modes=[WorkMode.RESEARCH, WorkMode.OPTIMIZATION],
                cooldown_minutes=240,
                recurring=True,
            ),
            Task(
                id="strategy_discovery",
                name="Discover New Strategies",
                category=TaskCategory.DISCOVERY,
                priority=TaskPriority.LOW,
                description="Search for new trading patterns and strategies",
                handler="autonomous.research:discover_strategies",
                valid_modes=[WorkMode.DEEP_RESEARCH],
                cooldown_minutes=360,
                recurring=True,
            ),

            # === LEARNING TASKS ===
            # FIXED: Allow learning during weekend/deep_research modes
            Task(
                id="analyze_trades",
                name="Analyze Recent Trades",
                category=TaskCategory.LEARNING,
                priority=TaskPriority.NORMAL,
                description="Learn from recent trade outcomes",
                handler="autonomous.learning:analyze_trades",
                valid_modes=[WorkMode.LEARNING, WorkMode.DEEP_RESEARCH, WorkMode.RESEARCH],
                cooldown_minutes=60,
                recurring=True,
            ),
            Task(
                id="update_memory",
                name="Update Episodic Memory",
                category=TaskCategory.LEARNING,
                priority=TaskPriority.NORMAL,
                description="Store new experiences in cognitive memory",
                handler="autonomous.learning:update_memory",
                valid_modes=[WorkMode.LEARNING, WorkMode.DEEP_RESEARCH, WorkMode.RESEARCH],
                cooldown_minutes=30,
                recurring=True,
            ),
            Task(
                id="reflect_on_day",
                name="Daily Reflection",
                category=TaskCategory.LEARNING,
                priority=TaskPriority.NORMAL,
                description="Generate daily performance reflection",
                handler="autonomous.learning:daily_reflection",
                valid_phases=[MarketPhase.AFTER_HOURS, MarketPhase.WEEKEND, MarketPhase.NIGHT],
                valid_modes=[WorkMode.LEARNING, WorkMode.DEEP_RESEARCH, WorkMode.RESEARCH],
                cooldown_minutes=480,  # Every 8 hours to allow weekend reflections
                recurring=True,
            ),

            # === OPTIMIZATION TASKS ===
            Task(
                id="optimize_pf",
                name="Profit Factor Optimization",
                category=TaskCategory.OPTIMIZATION,
                priority=TaskPriority.NORMAL,
                description="Focus on improving Profit Factor via exits and filters",
                handler="autonomous.research:optimize_pf",
                valid_modes=[WorkMode.OPTIMIZATION, WorkMode.RESEARCH, WorkMode.DEEP_RESEARCH],
                cooldown_minutes=30,  # Run frequently during optimization mode
                recurring=True,
            ),
            Task(
                id="walk_forward",
                name="Walk-Forward Optimization",
                category=TaskCategory.OPTIMIZATION,
                priority=TaskPriority.LOW,
                description="Run walk-forward backtest for parameter validation",
                handler="autonomous.optimization:walk_forward",
                valid_modes=[WorkMode.OPTIMIZATION, WorkMode.DEEP_RESEARCH],
                cooldown_minutes=720,  # Every 12 hours max
                recurring=True,
            ),
            Task(
                id="retrain_models",
                name="Retrain ML Models",
                category=TaskCategory.OPTIMIZATION,
                priority=TaskPriority.LOW,
                description="Retrain ensemble and LSTM models with recent data",
                handler="autonomous.optimization:retrain_models",
                valid_modes=[WorkMode.OPTIMIZATION, WorkMode.DEEP_RESEARCH],
                valid_phases=[MarketPhase.NIGHT, MarketPhase.WEEKEND],
                cooldown_minutes=1440,  # Once per day
                recurring=True,
            ),

            # === MAINTENANCE TASKS ===
            Task(
                id="data_quality",
                name="Check Data Quality",
                category=TaskCategory.MAINTENANCE,
                priority=TaskPriority.NORMAL,
                description="Validate data integrity and freshness",
                handler="autonomous.maintenance:check_data",
                cooldown_minutes=180,
                recurring=True,
            ),
            Task(
                id="cleanup_logs",
                name="Clean Up Old Logs",
                category=TaskCategory.MAINTENANCE,
                priority=TaskPriority.BACKGROUND,
                description="Remove old log files and free disk space",
                handler="autonomous.maintenance:cleanup",
                cooldown_minutes=1440,  # Once per day
                recurring=True,
            ),
            Task(
                id="health_check",
                name="System Health Check",
                category=TaskCategory.MAINTENANCE,
                priority=TaskPriority.NORMAL,
                description="Verify all system components are healthy",
                handler="autonomous.maintenance:health_check",
                cooldown_minutes=30,
                recurring=True,
            ),
            # FIX (2026-01-05): Proactive LLM budget reset at midnight
            Task(
                id="reset_llm_budget",
                name="Reset Daily LLM Budget",
                category=TaskCategory.MAINTENANCE,
                priority=TaskPriority.CRITICAL,
                description="Reset token/USD limits for LLM API calls at midnight ET",
                handler="llm.token_budget:reset_daily_budget",
                valid_phases=[MarketPhase.NIGHT],  # Run during night phase (after midnight)
                cooldown_minutes=1440,  # Once per day
                recurring=True,
            ),

            # === DATA TASKS ===
            Task(
                id="update_universe",
                name="Update Stock Universe",
                category=TaskCategory.DATA,
                priority=TaskPriority.LOW,
                description="Refresh the 900-stock universe",
                handler="autonomous.data:update_universe",
                valid_phases=[MarketPhase.NIGHT, MarketPhase.WEEKEND],
                cooldown_minutes=10080,  # Weekly
                recurring=True,
            ),
            Task(
                id="fetch_new_data",
                name="Fetch Latest Market Data",
                category=TaskCategory.DATA,
                priority=TaskPriority.NORMAL,
                description="Download latest EOD data",
                handler="autonomous.data:fetch_data",
                valid_phases=[MarketPhase.AFTER_HOURS],
                cooldown_minutes=1440,
                recurring=True,
            ),

            # === WATCHLIST TASKS (Professional Flow) ===
            # FIXED: Allow weekend so Monday watchlist can be built
            Task(
                id="build_overnight_watchlist",
                name="Build Overnight Watchlist",
                category=TaskCategory.TRADING,
                priority=TaskPriority.HIGH,
                description="Build Top 5 watchlist for next trading day",
                handler="scripts.overnight_watchlist:build",
                scheduled_time=None,
                valid_phases=[MarketPhase.MARKET_CLOSE, MarketPhase.WEEKEND, MarketPhase.AFTER_HOURS],
                valid_modes=[WorkMode.DEEP_RESEARCH, WorkMode.RESEARCH, WorkMode.MONITORING],
                cooldown_minutes=480,  # Every 8 hours during weekend
                recurring=True,
            ),
            Task(
                id="premarket_validation",
                name="Premarket Watchlist Validation",
                category=TaskCategory.MONITORING,
                priority=TaskPriority.HIGH,
                description="Validate overnight watchlist for gaps and news",
                handler="scripts.premarket_validator:validate",
                valid_phases=[MarketPhase.PRE_MARKET_ACTIVE, MarketPhase.PRE_MARKET_EARLY],
                cooldown_minutes=1440,
                recurring=True,
            ),
            # FORCE watchlist builder - can run anytime
            Task(
                id="force_build_watchlist",
                name="Force Build Monday Watchlist",
                category=TaskCategory.TRADING,
                priority=TaskPriority.HIGH,
                description="Force build watchlist - runs anytime during weekend",
                handler="autonomous.handlers:force_build_watchlist",
                valid_phases=[MarketPhase.WEEKEND],
                valid_modes=[WorkMode.DEEP_RESEARCH, WorkMode.RESEARCH],
                cooldown_minutes=240,  # Every 4 hours
                recurring=True,
            ),

            # === SELF-IMPROVEMENT TASKS (Always getting smarter) ===
            Task(
                id="check_goals",
                name="Check Goal Progress",
                category=TaskCategory.LEARNING,
                priority=TaskPriority.NORMAL,
                description="Track progress toward trading goals",
                handler="autonomous.research:check_goals",
                cooldown_minutes=60,
                recurring=True,
            ),
            Task(
                id="review_discoveries",
                name="Review Discoveries",
                category=TaskCategory.LEARNING,
                priority=TaskPriority.NORMAL,
                description="Validate and review new discoveries",
                handler="autonomous.brain:review_discoveries",
                cooldown_minutes=120,
                recurring=True,
            ),
            Task(
                id="consolidate_learnings",
                name="Consolidate Learnings",
                category=TaskCategory.LEARNING,
                priority=TaskPriority.NORMAL,
                description="Turn discoveries into actionable insights",
                handler="autonomous.brain:consolidate_learnings",
                cooldown_minutes=180,
                recurring=True,
            ),
            Task(
                id="check_data_quality_deep",
                name="Deep Data Quality Check",
                category=TaskCategory.DATA,
                priority=TaskPriority.NORMAL,
                description="Thorough data quality analysis",
                handler="autonomous.research:check_data_quality",
                cooldown_minutes=240,
                recurring=True,
            ),

            # === EXTERNAL RESEARCH TASKS (24/7 Learning from External Sources) ===
            Task(
                id="scrape_github",
                name="Scrape GitHub Strategies",
                category=TaskCategory.DISCOVERY,
                priority=TaskPriority.NORMAL,
                description="Fetch trading strategy ideas from GitHub repositories",
                handler="autonomous.scrapers:scrape_github",
                valid_phases=[MarketPhase.NIGHT, MarketPhase.WEEKEND, MarketPhase.AFTER_HOURS],
                valid_modes=[WorkMode.DEEP_RESEARCH, WorkMode.RESEARCH],
                cooldown_minutes=360,  # Every 6 hours
                recurring=True,
            ),
            Task(
                id="scrape_reddit",
                name="Scrape Reddit Ideas",
                category=TaskCategory.DISCOVERY,
                priority=TaskPriority.NORMAL,
                description="Fetch trading ideas from r/algotrading and related subs",
                handler="autonomous.scrapers:scrape_reddit",
                valid_phases=[MarketPhase.NIGHT, MarketPhase.WEEKEND, MarketPhase.AFTER_HOURS],
                valid_modes=[WorkMode.DEEP_RESEARCH, WorkMode.RESEARCH],
                cooldown_minutes=360,  # Every 6 hours
                recurring=True,
            ),
            Task(
                id="scrape_youtube",
                name="Scrape YouTube Strategies",
                category=TaskCategory.DISCOVERY,
                priority=TaskPriority.LOW,
                description="Extract strategy ideas from trading video transcripts",
                handler="autonomous.scrapers:scrape_youtube",
                valid_phases=[MarketPhase.NIGHT, MarketPhase.WEEKEND],
                valid_modes=[WorkMode.DEEP_RESEARCH],
                cooldown_minutes=720,  # Every 12 hours (less frequent)
                recurring=True,
            ),
            Task(
                id="scrape_arxiv",
                name="Fetch arXiv Papers",
                category=TaskCategory.DISCOVERY,
                priority=TaskPriority.LOW,
                description="Fetch quantitative finance research papers from arXiv",
                handler="autonomous.scrapers:scrape_arxiv",
                valid_phases=[MarketPhase.NIGHT, MarketPhase.WEEKEND],
                valid_modes=[WorkMode.DEEP_RESEARCH],
                cooldown_minutes=720,  # Every 12 hours
                recurring=True,
            ),
            Task(
                id="fetch_all_external",
                name="Fetch All External Ideas",
                category=TaskCategory.DISCOVERY,
                priority=TaskPriority.NORMAL,
                description="Run all scrapers to find new strategy ideas",
                handler="autonomous.scrapers:fetch_all",
                valid_phases=[MarketPhase.NIGHT, MarketPhase.WEEKEND],
                valid_modes=[WorkMode.DEEP_RESEARCH],
                cooldown_minutes=480,  # Every 8 hours
                recurring=True,
            ),
            Task(
                id="validate_external_ideas",
                name="Validate External Ideas",
                category=TaskCategory.RESEARCH,
                priority=TaskPriority.HIGH,
                description="Test external strategy ideas with REAL backtest data",
                handler="autonomous.scrapers:validate_ideas",
                valid_modes=[WorkMode.RESEARCH, WorkMode.DEEP_RESEARCH, WorkMode.OPTIMIZATION],
                cooldown_minutes=180,  # Every 3 hours
                recurring=True,
            ),
            Task(
                id="source_credibility",
                name="Source Credibility Report",
                category=TaskCategory.LEARNING,
                priority=TaskPriority.LOW,
                description="Track which external sources provide validated ideas",
                handler="autonomous.scrapers:source_credibility",
                cooldown_minutes=480,  # Every 8 hours
                recurring=True,
            ),

            # Weekend Morning Report (8:30 AM Central = 9:30 AM ET)
            Task(
                id="weekend_morning_report",
                name="Weekend Morning Game Plan",
                category=TaskCategory.TRADING,
                priority=TaskPriority.HIGH,
                description="Generate comprehensive weekend morning report at 8:30 AM Central",
                handler="autonomous.handlers:weekend_morning_report",
                valid_phases=[MarketPhase.WEEKEND],
                valid_modes=[WorkMode.DEEP_RESEARCH, WorkMode.RESEARCH],
                cooldown_minutes=60,  # Run every hour during weekend morning
                recurring=True,
            ),

            # Pattern Rhymes Tasks (History Rhymes)
            Task(
                id="analyze_seasonality",
                name="Analyze Seasonality Patterns",
                category=TaskCategory.RESEARCH,
                priority=TaskPriority.LOW,
                description="Analyze monthly/quarterly patterns in historical data",
                handler="autonomous.patterns:analyze_seasonality",
                valid_phases=[MarketPhase.NIGHT, MarketPhase.WEEKEND],
                valid_modes=[WorkMode.DEEP_RESEARCH],
                cooldown_minutes=720,  # Every 12 hours
                recurring=True,
            ),
            Task(
                id="mean_reversion_timing",
                name="Mean Reversion Timing Analysis",
                category=TaskCategory.RESEARCH,
                priority=TaskPriority.NORMAL,
                description="Analyze how long extreme moves take to revert",
                handler="autonomous.patterns:mean_reversion_timing",
                valid_phases=[MarketPhase.NIGHT, MarketPhase.WEEKEND],
                valid_modes=[WorkMode.DEEP_RESEARCH, WorkMode.RESEARCH],
                cooldown_minutes=480,  # Every 8 hours
                recurring=True,
            ),
            Task(
                id="sector_correlations",
                name="Sector Correlation Analysis",
                category=TaskCategory.RESEARCH,
                priority=TaskPriority.LOW,
                description="Find highly correlated stocks for diversification",
                handler="autonomous.patterns:sector_correlations",
                valid_phases=[MarketPhase.NIGHT, MarketPhase.WEEKEND],
                valid_modes=[WorkMode.DEEP_RESEARCH],
                cooldown_minutes=720,  # Every 12 hours
                recurring=True,
            ),

            # UNIQUE Pattern Discovery - PLTR-style insights from REAL data
            Task(
                id="discover_unique_patterns",
                name="Discover Unique Patterns",
                category=TaskCategory.DISCOVERY,
                priority=TaskPriority.HIGH,
                description="Find UNIQUE patterns like 'AMAT: 23x 5+ down days -> 83% bounce' from REAL data",
                handler="autonomous.patterns:discover_unique",
                valid_phases=[MarketPhase.NIGHT, MarketPhase.WEEKEND, MarketPhase.AFTER_HOURS],
                valid_modes=[WorkMode.DEEP_RESEARCH, WorkMode.RESEARCH, WorkMode.OPTIMIZATION],
                cooldown_minutes=120,  # Every 2 hours - find new insights frequently
                recurring=True,
            ),

            # =================================================================
            # NEW: WEEKLY GAME PLAN & STRATEGY ROTATION
            # =================================================================
            Task(
                id="weekly_game_plan",
                name="Generate Weekly Game Plan",
                category=TaskCategory.TRADING,
                priority=TaskPriority.HIGH,
                description="Comprehensive weekly plan with strategy research, experiments, and risk reminders",
                handler="autonomous.handlers:weekly_game_plan",
                valid_phases=[MarketPhase.WEEKEND],
                valid_modes=[WorkMode.DEEP_RESEARCH, WorkMode.RESEARCH],
                cooldown_minutes=480,  # Every 8 hours during weekend
                recurring=True,
            ),
            Task(
                id="strategy_rotation_report",
                name="Strategy Rotation Report",
                category=TaskCategory.LEARNING,
                priority=TaskPriority.NORMAL,
                description="Track ICT vs Basic vs Complex strategy research rotation",
                handler="autonomous.handlers:strategy_rotation_report",
                valid_phases=[MarketPhase.WEEKEND, MarketPhase.NIGHT],
                valid_modes=[WorkMode.DEEP_RESEARCH, WorkMode.RESEARCH],
                cooldown_minutes=360,  # Every 6 hours
                recurring=True,
            ),
            Task(
                id="discoveries_dashboard",
                name="Generate Discoveries Dashboard",
                category=TaskCategory.LEARNING,
                priority=TaskPriority.NORMAL,
                description="Dashboard showing all discoveries for user visibility",
                handler="autonomous.handlers:discoveries_dashboard",
                valid_phases=[MarketPhase.WEEKEND, MarketPhase.NIGHT, MarketPhase.AFTER_HOURS],
                valid_modes=[WorkMode.DEEP_RESEARCH, WorkMode.RESEARCH, WorkMode.LEARNING],
                cooldown_minutes=120,  # Every 2 hours
                recurring=True,
            ),
        ]

        for task in default_tasks:
            self.queue.add_task(task)

    def register_handler(self, name: str, handler: Callable):
        """Register a task handler function."""
        self._handlers[name] = handler

    def get_context(self) -> MarketContext:
        """Get current market context."""
        return self.context_builder.get_context()

    def get_next_task(self) -> Optional[Task]:
        """Get next task to execute."""
        context = self.get_context()
        return self.queue.get_next_task(context)

    def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task."""
        logger.info(f"Executing task: {task.name}")
        task.status = TaskStatus.RUNNING

        try:
            # Find handler
            if task.handler and task.handler in self._handlers:
                handler = self._handlers[task.handler]
                result = handler(**task.params)
            else:
                # Default: just log
                logger.info(f"Task {task.name} has no registered handler")
                result = {"status": "no_handler", "task": task.name}

            task.status = TaskStatus.COMPLETED
            task.last_result = result
            task.last_error = None

        except Exception as e:
            logger.error(f"Task {task.name} failed: {e}")
            task.status = TaskStatus.FAILED
            task.last_error = str(e)
            result = {"status": "error", "error": str(e)}

        task.last_run = datetime.now(ET)
        task.run_count += 1

        # Reset status for recurring tasks
        if task.recurring:
            task.status = TaskStatus.PENDING

        self.queue.save_state()
        return result

    def run_one_cycle(self) -> Optional[Dict[str, Any]]:
        """Run one scheduling cycle."""
        context = self.get_context()
        task = self.queue.get_next_task(context)

        if task:
            return {
                "task": task.name,
                "context": context.to_dict(),
                "result": self.execute_task(task),
            }
        else:
            logger.debug("No tasks eligible to run")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        context = self.get_context()
        return {
            "context": context.to_dict(),
            "pending_tasks": self.queue.get_pending_count(),
            "tasks": [t.to_dict() for t in self.queue.tasks.values()],
        }


# Convenience functions
def get_scheduler() -> AutonomousScheduler:
    """Get the autonomous scheduler."""
    return AutonomousScheduler()


if __name__ == "__main__":
    # Demo
    scheduler = get_scheduler()
    status = scheduler.get_status()

    print("Scheduler Status:")
    print(f"  Current Phase: {status['context']['phase']}")
    print(f"  Work Mode: {status['context']['work_mode']}")
    print(f"  Pending Tasks: {status['pending_tasks']}")
    print("\nRecommended Actions:")
    for action in status['context']['recommended_actions']:
        print(f"  - {action}")

    print("\nNext Task: ", end="")
    task = scheduler.get_next_task()
    if task:
        print(f"{task.name} ({task.priority.name})")
    else:
        print("None eligible")
