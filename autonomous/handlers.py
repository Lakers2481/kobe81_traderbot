"""
Task Handlers for the Autonomous Brain.

This module wires up all the task handlers to actual functions.
Each handler is a function that executes a specific task.
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def run_script(script_path: str, args: list = None, timeout: int = 300) -> Dict[str, Any]:
    """Run a Python script and return results."""
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent,
        )
        return {
            "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-500:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"Script timed out after {timeout}s"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# TRADING HANDLERS
# =============================================================================

def scan_signals(**kwargs) -> Dict[str, Any]:
    """Run the daily scanner."""
    logger.info("Running signal scanner...")
    return run_script("scripts/scan.py", ["--cap", "100", "--deterministic"], timeout=180)


def check_positions(**kwargs) -> Dict[str, Any]:
    """Check current positions and P&L."""
    logger.info("Checking positions...")
    try:
        from oms.order_state import OrderStateManager

        manager = OrderStateManager()
        positions = manager.get_open_positions()

        return {
            "status": "success",
            "open_positions": len(positions),
            "positions": [
                {"symbol": p.symbol, "side": p.side, "entry_price": p.entry_price}
                for p in positions[:10]
            ],
        }
    except Exception as e:
        logger.error(f"Position check failed: {e}")
        return {"status": "error", "error": str(e)}


def reconcile_broker(**kwargs) -> Dict[str, Any]:
    """Reconcile positions with broker."""
    logger.info("Reconciling with broker...")
    return run_script("scripts/reconcile_alpaca.py", timeout=60)


# =============================================================================
# RESEARCH HANDLERS
# =============================================================================

def backtest_random_params(**kwargs) -> Dict[str, Any]:
    """Run random parameter experiment."""
    logger.info("Running parameter experiment...")
    from autonomous.research import ResearchEngine

    engine = ResearchEngine()
    return engine.backtest_random_params()


def analyze_features(**kwargs) -> Dict[str, Any]:
    """Analyze feature importance."""
    logger.info("Analyzing features...")
    from autonomous.research import ResearchEngine

    engine = ResearchEngine()
    return engine.analyze_features()


def discover_strategies(**kwargs) -> Dict[str, Any]:
    """Discover new trading patterns."""
    logger.info("Discovering strategies...")
    from autonomous.research import ResearchEngine

    engine = ResearchEngine()
    return engine.discover_strategies()


# =============================================================================
# LEARNING HANDLERS
# =============================================================================

def analyze_trades(**kwargs) -> Dict[str, Any]:
    """Analyze recent trades for lessons."""
    logger.info("Analyzing trades...")
    from autonomous.learning import LearningEngine

    engine = LearningEngine()
    return engine.analyze_trades()


def update_memory(**kwargs) -> Dict[str, Any]:
    """Update episodic memory with lessons."""
    logger.info("Updating memory...")
    from autonomous.learning import LearningEngine

    engine = LearningEngine()
    return engine.update_memory()


def daily_reflection(**kwargs) -> Dict[str, Any]:
    """Generate daily reflection."""
    logger.info("Generating daily reflection...")
    from autonomous.learning import LearningEngine

    engine = LearningEngine()
    return engine.daily_reflection()


# =============================================================================
# OPTIMIZATION HANDLERS
# =============================================================================

def walk_forward(**kwargs) -> Dict[str, Any]:
    """Run walk-forward optimization."""
    logger.info("Running walk-forward optimization...")
    # This is a long-running task, limit to smaller universe
    return run_script(
        "scripts/run_wf_polygon.py",
        ["--universe", "data/universe/optionable_liquid_900.csv",
         "--start", "2022-01-01", "--end", "2024-12-31",
         "--train-days", "252", "--test-days", "63", "--cap", "20"],
        timeout=600,
    )


def retrain_models(**kwargs) -> Dict[str, Any]:
    """Retrain ML models with recent data."""
    logger.info("Retraining models...")
    results = {}

    # Train ensemble
    result = run_script("scripts/train_ensemble.py", timeout=300)
    results["ensemble"] = result.get("status", "unknown")

    # Train HMM if ensemble succeeded
    if result.get("status") == "success":
        result = run_script("scripts/train_hmm_regime.py", timeout=120)
        results["hmm"] = result.get("status", "unknown")

    return {"status": "success", "models": results}


# =============================================================================
# MAINTENANCE HANDLERS
# =============================================================================

def check_data(**kwargs) -> Dict[str, Any]:
    """Check data quality."""
    logger.info("Checking data quality...")
    from autonomous.maintenance import MaintenanceEngine

    engine = MaintenanceEngine()
    return engine.check_data()


def cleanup(**kwargs) -> Dict[str, Any]:
    """Clean up old files."""
    logger.info("Running cleanup...")
    from autonomous.maintenance import MaintenanceEngine

    engine = MaintenanceEngine()
    return engine.cleanup()


def health_check(**kwargs) -> Dict[str, Any]:
    """Run health check."""
    logger.info("Running health check...")
    from autonomous.maintenance import MaintenanceEngine

    engine = MaintenanceEngine()
    return engine.health_check()


# =============================================================================
# DATA HANDLERS
# =============================================================================

def update_universe(**kwargs) -> Dict[str, Any]:
    """Update stock universe."""
    logger.info("Updating universe...")
    return run_script(
        "scripts/build_universe_polygon.py",
        ["--cidates", "data/universe/optionable_liquid_cidates.csv",
         "--start", "2015-01-01", "--end", "2024-12-31",
         "--min-years", "10", "--cap", "900", "--concurrency", "3"],
        timeout=600,
    )


def fetch_data(**kwargs) -> Dict[str, Any]:
    """Fetch latest market data."""
    logger.info("Fetching data...")
    return run_script(
        "scripts/prefetch_polygon_universe.py",
        ["--universe", "data/universe/optionable_liquid_900.csv",
         "--start", "2024-01-01", "--end", "2024-12-31"],
        timeout=600,
    )


# =============================================================================
# WATCHLIST HANDLERS (Professional Flow)
# =============================================================================

def build_overnight_watchlist(**kwargs) -> Dict[str, Any]:
    """Build overnight watchlist for next day."""
    logger.info("Building overnight watchlist...")
    return run_script("scripts/overnight_watchlist.py", timeout=180)


def premarket_validation(**kwargs) -> Dict[str, Any]:
    """Validate premarket watchlist."""
    logger.info("Running premarket validation...")
    return run_script("scripts/premarket_validator.py", timeout=120)


# =============================================================================
# HANDLER REGISTRY
# =============================================================================

HANDLERS = {
    # Trading
    "scripts.scan:run_scan": scan_signals,
    "scripts.positions:check_positions": check_positions,
    "scripts.reconcile_alpaca:reconcile": reconcile_broker,

    # Research
    "autonomous.research:backtest_random_params": backtest_random_params,
    "autonomous.research:analyze_features": analyze_features,
    "autonomous.research:discover_strategies": discover_strategies,

    # Learning
    "autonomous.learning:analyze_trades": analyze_trades,
    "autonomous.learning:update_memory": update_memory,
    "autonomous.learning:daily_reflection": daily_reflection,

    # Optimization
    "autonomous.optimization:walk_forward": walk_forward,
    "autonomous.optimization:retrain_models": retrain_models,

    # Maintenance
    "autonomous.maintenance:check_data": check_data,
    "autonomous.maintenance:cleanup": cleanup,
    "autonomous.maintenance:health_check": health_check,

    # Data
    "autonomous.data:update_universe": update_universe,
    "autonomous.data:fetch_data": fetch_data,

    # Watchlist
    "scripts.overnight_watchlist:build": build_overnight_watchlist,
    "scripts.premarket_validator:validate": premarket_validation,
}


def register_all_handlers(scheduler):
    """Register all handlers with the scheduler."""
    for name, handler in HANDLERS.items():
        scheduler.register_handler(name, handler)
    logger.info(f"Registered {len(HANDLERS)} task handlers")


def get_handler(name: str):
    """Get a handler by name."""
    return HANDLERS.get(name)
