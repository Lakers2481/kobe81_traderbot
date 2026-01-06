#!/usr/bin/env python3
"""
SYNC ALL TASKS - Unified Scheduler Fix
=======================================
This script:
1. Reloads ALL tasks from autonomous/scheduler.py into task_queue.json
2. Adds missing time-critical tasks from scheduler_kobe.py
3. Removes restrictive phase/mode limitations that block execution
4. Ensures the brain runs ALL tasks 24/7

Run this to fix the broken scheduler!
"""

import json
from pathlib import Path
from datetime import datetime
import pytz

# Paths
STATE_DIR = Path("state/autonomous")
TASK_QUEUE_FILE = STATE_DIR / "task_queue.json"

def create_unified_task_queue():
    """Create unified task queue with ALL tasks."""

    et = pytz.timezone("US/Eastern")
    now = datetime.now(et)

    # UNIFIED TASK LIST - ALL tasks that should run 24/7
    tasks = [
        # === CRITICAL TRADING TASKS (NO PHASE RESTRICTIONS) ===
        {
            "id": "scan_signals",
            "name": "Scan for Trading Signals",
            "category": "trading",
            "priority": 1,  # CRITICAL
            "description": "Run 900-stock scanner for opportunities",
            "handler": "scripts.scan:run_scan",
            "params": {"cap": 900, "deterministic": True, "no_quality_gate": True, "workers": 1},
            "scheduled_time": None,
            "valid_phases": [],  # NO RESTRICTIONS - run anytime
            "valid_modes": [],   # NO RESTRICTIONS
            "cooldown_minutes": 30,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "generate_pregame_blueprint",
            "name": "Generate Pre-Game Blueprint",
            "category": "trading",
            "priority": 1,  # CRITICAL
            "description": "Generate full 15-section analysis for Top 2 trades",
            "handler": "autonomous.handlers:generate_pregame_blueprint",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],  # NO RESTRICTIONS
            "valid_modes": [],
            "cooldown_minutes": 60,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "swing_scanner",
            "name": "Swing Scanner (2:45 PM CT / 3:45 PM ET)",
            "category": "trading",
            "priority": 1,
            "description": "End of day swing setup scanner",
            "handler": "scripts.scan:run_scan",
            "params": {"cap": 900, "deterministic": True, "top3": True},
            "scheduled_time": "15:45",  # 3:45 PM ET = 2:45 PM CT
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 1440,  # Once per day
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "build_overnight_watchlist",
            "name": "Build Overnight Watchlist",
            "category": "trading",
            "priority": 1,
            "description": "Build Top 5 watchlist for next trading day",
            "handler": "scripts.overnight_watchlist:build",
            "params": {},
            "scheduled_time": "15:30",  # 3:30 PM ET
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 480,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "premarket_validation",
            "name": "Premarket Watchlist Validation",
            "category": "monitoring",
            "priority": 1,
            "description": "Validate overnight watchlist for gaps and news",
            "handler": "scripts.premarket_validator:validate",
            "params": {},
            "scheduled_time": "08:00",
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 1440,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },

        # === POSITION MANAGEMENT (Every 5 min during market) ===
        {
            "id": "check_positions",
            "name": "Check Position P&L",
            "category": "monitoring",
            "priority": 2,
            "description": "Monitor open positions and P&L",
            "handler": "scripts.positions:check_positions",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 5,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "reconcile_broker",
            "name": "Reconcile Broker Positions",
            "category": "monitoring",
            "priority": 1,
            "description": "Ensure local and broker positions match",
            "handler": "scripts.reconcile_alpaca:reconcile",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 60,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },

        # === RESEARCH TASKS ===
        {
            "id": "backtest_random_params",
            "name": "Backtest Random Parameters",
            "category": "research",
            "priority": 3,
            "description": "Test random parameter variations",
            "handler": "autonomous.research:backtest_random_params",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 120,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "feature_importance",
            "name": "Analyze Feature Importance",
            "category": "research",
            "priority": 3,
            "description": "Run SHAP analysis on predictions",
            "handler": "autonomous.research:analyze_features",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 240,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "strategy_discovery",
            "name": "Discover New Strategies",
            "category": "discovery",
            "priority": 4,
            "description": "Search for new trading patterns",
            "handler": "autonomous.research:discover_strategies",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 360,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "optimize_pf",
            "name": "Profit Factor Optimization",
            "category": "optimization",
            "priority": 3,
            "description": "Improve Profit Factor via exits/filters",
            "handler": "autonomous.research:optimize_pf",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 30,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },

        # === LEARNING TASKS ===
        {
            "id": "analyze_trades",
            "name": "Analyze Recent Trades",
            "category": "learning",
            "priority": 3,
            "description": "Learn from recent trade outcomes",
            "handler": "autonomous.learning:analyze_trades",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 60,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "update_memory",
            "name": "Update Episodic Memory",
            "category": "learning",
            "priority": 3,
            "description": "Store experiences in cognitive memory",
            "handler": "autonomous.learning:update_memory",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 30,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "reflect_on_day",
            "name": "Daily Reflection",
            "category": "learning",
            "priority": 3,
            "description": "Generate daily performance reflection",
            "handler": "autonomous.learning:daily_reflection",
            "params": {},
            "scheduled_time": "17:00",
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 480,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "check_goals",
            "name": "Check Goal Progress",
            "category": "learning",
            "priority": 3,
            "description": "Track progress toward trading goals",
            "handler": "autonomous.research:check_goals",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 60,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "review_discoveries",
            "name": "Review Discoveries",
            "category": "learning",
            "priority": 3,
            "description": "Validate and review new discoveries",
            "handler": "autonomous.brain:review_discoveries",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 120,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "consolidate_learnings",
            "name": "Consolidate Learnings",
            "category": "learning",
            "priority": 3,
            "description": "Turn discoveries into insights",
            "handler": "autonomous.brain:consolidate_learnings",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 180,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },

        # === OPTIMIZATION TASKS ===
        {
            "id": "walk_forward",
            "name": "Walk-Forward Optimization",
            "category": "optimization",
            "priority": 4,
            "description": "Run walk-forward backtest",
            "handler": "autonomous.optimization:walk_forward",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 720,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "retrain_models",
            "name": "Retrain ML Models",
            "category": "optimization",
            "priority": 4,
            "description": "Retrain ensemble and LSTM models",
            "handler": "autonomous.optimization:retrain_models",
            "params": {},
            "scheduled_time": "02:00",  # 2 AM ET
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 1440,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },

        # === MAINTENANCE TASKS ===
        {
            "id": "data_quality",
            "name": "Check Data Quality",
            "category": "maintenance",
            "priority": 3,
            "description": "Validate data integrity",
            "handler": "autonomous.maintenance:check_data",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 180,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "cleanup_logs",
            "name": "Clean Up Old Logs",
            "category": "maintenance",
            "priority": 5,
            "description": "Remove old log files",
            "handler": "autonomous.maintenance:cleanup",
            "params": {},
            "scheduled_time": "04:00",
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 1440,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "health_check",
            "name": "System Health Check",
            "category": "maintenance",
            "priority": 3,
            "description": "Verify all components healthy",
            "handler": "autonomous.maintenance:health_check",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 30,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },

        # === DATA TASKS ===
        {
            "id": "update_universe",
            "name": "Update Stock Universe",
            "category": "data",
            "priority": 4,
            "description": "Refresh 900-stock universe",
            "handler": "autonomous.data:update_universe",
            "params": {},
            "scheduled_time": "03:00",
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 10080,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "fetch_new_data",
            "name": "Fetch Latest Market Data",
            "category": "data",
            "priority": 3,
            "description": "Download latest EOD data",
            "handler": "autonomous.data:fetch_data",
            "params": {},
            "scheduled_time": "18:00",
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 1440,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "check_data_quality_deep",
            "name": "Deep Data Quality Check",
            "category": "data",
            "priority": 3,
            "description": "Thorough data quality analysis",
            "handler": "autonomous.research:check_data_quality",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 240,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },

        # === DISCOVERY/SCRAPING TASKS ===
        {
            "id": "scrape_github",
            "name": "Scrape GitHub Strategies",
            "category": "discovery",
            "priority": 3,
            "description": "Fetch trading ideas from GitHub",
            "handler": "autonomous.scrapers:scrape_github",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 360,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "scrape_reddit",
            "name": "Scrape Reddit Ideas",
            "category": "discovery",
            "priority": 3,
            "description": "Fetch ideas from r/algotrading",
            "handler": "autonomous.scrapers:scrape_reddit",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 360,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "discover_unique_patterns",
            "name": "Discover Unique Patterns",
            "category": "discovery",
            "priority": 2,
            "description": "Find unique patterns from REAL data",
            "handler": "autonomous.patterns:discover_unique",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 120,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "validate_external_ideas",
            "name": "Validate External Ideas",
            "category": "research",
            "priority": 2,
            "description": "Test external strategies with REAL backtest",
            "handler": "autonomous.scrapers:validate_ideas",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 180,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },

        # === WEEKEND TASKS ===
        {
            "id": "weekend_morning_report",
            "name": "Weekend Morning Game Plan",
            "category": "trading",
            "priority": 2,
            "description": "Comprehensive weekend morning report",
            "handler": "autonomous.handlers:weekend_morning_report",
            "params": {},
            "scheduled_time": "09:30",
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 60,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
        {
            "id": "force_build_watchlist",
            "name": "Force Build Monday Watchlist",
            "category": "trading",
            "priority": 2,
            "description": "Force build watchlist anytime",
            "handler": "autonomous.handlers:force_build_watchlist",
            "params": {},
            "scheduled_time": None,
            "valid_phases": [],
            "valid_modes": [],
            "cooldown_minutes": 240,
            "status": "pending",
            "last_run": None,
            "run_count": 0,
            "recurring": True
        },
    ]

    # Create the unified queue
    queue = {
        "updated_at": now.isoformat(),
        "tasks": tasks
    }

    return queue


def main():
    print("=" * 70)
    print("SYNC ALL TASKS - Unified Scheduler Fix")
    print("=" * 70)
    print()

    # Backup existing
    if TASK_QUEUE_FILE.exists():
        backup = TASK_QUEUE_FILE.with_suffix(".json.backup")
        import shutil
        shutil.copy(TASK_QUEUE_FILE, backup)
        print(f"Backed up existing queue to: {backup}")

    # Create unified queue
    queue = create_unified_task_queue()

    # Save
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TASK_QUEUE_FILE, 'w') as f:
        json.dump(queue, f, indent=2, default=str)

    print(f"\nCreated unified task queue with {len(queue['tasks'])} tasks:")
    print()

    # Show summary by category
    from collections import Counter
    categories = Counter(t['category'] for t in queue['tasks'])
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} tasks")

    print()
    print("KEY TASKS NOW INCLUDED:")
    print("  - scan_signals (NO phase restrictions)")
    print("  - generate_pregame_blueprint (was MISSING)")
    print("  - swing_scanner @ 3:45 PM ET (was MISSING)")
    print("  - build_overnight_watchlist @ 3:30 PM ET")
    print("  - premarket_validation @ 8:00 AM ET")
    print()
    print("ALL PHASE/MODE RESTRICTIONS REMOVED - Tasks run 24/7!")
    print()
    print("Restart the brain to pick up new tasks:")
    print("  python scripts/run_autonomous.py --once")
    print()


if __name__ == "__main__":
    main()
