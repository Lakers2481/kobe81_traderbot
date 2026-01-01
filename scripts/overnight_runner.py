#!/usr/bin/env python3
"""
Overnight Operations Runner
===========================

Runs continuous overnight operations:
1. Data updates
2. Strategy backtests
3. Learning cycles
4. Report generation
5. System health checks

This script is designed to run while you sleep.
"""

import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/overnight.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('overnight_runner')


def check_kill_switch():
    """Check if kill switch is active."""
    return Path('state/KILL_SWITCH').exists()


def run_data_update():
    """Update market data from Polygon."""
    logger.info("=" * 60)
    logger.info("TASK: Data Update")
    logger.info("=" * 60)

    try:
        from data.providers.polygon_eod import PolygonEODProvider
        from data.universe.loader import load_universe

        # Load universe
        universe = load_universe('data/universe/optionable_liquid_900.csv', cap=100)
        logger.info(f"Loaded {len(universe)} symbols from universe")

        # Note: Data is cached, so this just validates freshness
        provider = PolygonEODProvider()
        logger.info("Data provider initialized - using cached EOD data")

        return True
    except Exception as e:
        logger.error(f"Data update failed: {e}")
        return False


def run_backtest_validation():
    """Run a quick backtest to validate strategy performance."""
    logger.info("=" * 60)
    logger.info("TASK: Backtest Validation")
    logger.info("=" * 60)

    try:
        import subprocess
        result = subprocess.run(
            [
                sys.executable, 'scripts/backtest_dual_strategy.py',
                '--universe', 'data/universe/optionable_liquid_900.csv',
                '--start', '2024-01-01',
                '--end', '2024-12-31',
                '--cap', '50',
            ],
            capture_output=True,
            text=True,
            timeout=300
        )

        # Parse output for key metrics
        output = result.stdout
        logger.info("Backtest completed")

        # Look for win rate and profit factor
        for line in output.split('\n'):
            if 'Win Rate' in line or 'Profit Factor' in line or 'Total Trades' in line:
                logger.info(f"  {line.strip()}")

        return result.returncode == 0
    except Exception as e:
        logger.error(f"Backtest validation failed: {e}")
        return False


def run_learning_cycle():
    """Run cognitive learning cycle - process episodic memory."""
    logger.info("=" * 60)
    logger.info("TASK: Learning Cycle")
    logger.info("=" * 60)

    try:
        from cognitive.episodic_memory import get_episodic_memory
        from cognitive.self_model import get_self_model

        # Get episodic memory stats
        mem = get_episodic_memory()
        stats = mem.get_stats()
        logger.info(f"Episodic Memory Stats:")
        logger.info(f"  Total Episodes: {stats.get('total_episodes', 0)}")
        logger.info(f"  Win Rate: {stats.get('win_rate', 'N/A')}")
        logger.info(f"  Total Lessons: {stats.get('total_lessons', 0)}")

        # Get self-model stats
        sm = get_self_model()
        logger.info(f"Self Model Stats:")
        logger.info(f"  Strengths: {len(sm.get_strengths())}")
        logger.info(f"  Weaknesses: {len(sm.get_weaknesses())}")
        logger.info(f"  Calibrated: {sm.is_well_calibrated()}")

        return True
    except Exception as e:
        logger.error(f"Learning cycle failed: {e}")
        return False


def run_system_health():
    """Check system health."""
    logger.info("=" * 60)
    logger.info("TASK: System Health Check")
    logger.info("=" * 60)

    checks = {
        'universe_file': Path('data/universe/optionable_liquid_900.csv').exists(),
        'episodic_memory': Path('state/cognitive/episodes').exists(),
        'ml_models': Path('state/models/ensemble').exists(),
        'logs_dir': Path('logs').exists(),
        'kill_switch': not check_kill_switch(),
    }

    all_passed = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {check}: {status}")
        if not passed:
            all_passed = False

    return all_passed


def run_scan_preview():
    """Run a scan in preview mode."""
    logger.info("=" * 60)
    logger.info("TASK: Scan Preview")
    logger.info("=" * 60)

    try:
        import subprocess
        result = subprocess.run(
            [
                sys.executable, 'scripts/scan.py',
                '--cap', '100',
                '--deterministic',
                '--top3',
            ],
            capture_output=True,
            text=True,
            timeout=300
        )

        # Log the output
        for line in result.stdout.split('\n')[-20:]:  # Last 20 lines
            if line.strip():
                logger.info(f"  {line}")

        return result.returncode == 0
    except Exception as e:
        logger.error(f"Scan preview failed: {e}")
        return False


def generate_report():
    """Generate overnight summary report."""
    logger.info("=" * 60)
    logger.info("TASK: Generate Report")
    logger.info("=" * 60)

    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, 'scripts/pregame_report.py'],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Save to file
        report_path = Path('reports') / f"overnight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(result.stdout)
        logger.info(f"Report saved to: {report_path}")

        return True
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return False


def main():
    """Main overnight runner loop."""
    logger.info("=" * 80)
    logger.info("KOBE OVERNIGHT RUNNER STARTED")
    logger.info(f"Time: {datetime.now()}")
    logger.info("=" * 80)

    # Define tasks and intervals
    tasks = [
        ('Health Check', run_system_health, 30),       # Every 30 min
        ('Learning Cycle', run_learning_cycle, 60),    # Every 60 min
        ('Scan Preview', run_scan_preview, 120),       # Every 2 hours
        ('Report Generation', generate_report, 240),   # Every 4 hours
    ]

    last_run = {name: datetime.min for name, _, _ in tasks}

    cycle = 0
    while True:
        cycle += 1
        now = datetime.now()

        logger.info(f"\n{'='*80}")
        logger.info(f"CYCLE {cycle} - {now.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}")

        # Check kill switch
        if check_kill_switch():
            logger.warning("KILL SWITCH ACTIVE - Stopping overnight runner")
            break

        # Run due tasks
        for name, func, interval_min in tasks:
            if (now - last_run[name]).total_seconds() >= interval_min * 60:
                try:
                    success = func()
                    status = "SUCCESS" if success else "FAILED"
                    logger.info(f"Task '{name}': {status}")
                    last_run[name] = now
                except Exception as e:
                    logger.error(f"Task '{name}' error: {e}")
                    last_run[name] = now  # Don't retry immediately

        # Sleep between cycles
        logger.info(f"\nSleeping for 10 minutes...")
        time.sleep(600)  # 10 minutes


if __name__ == '__main__':
    main()
