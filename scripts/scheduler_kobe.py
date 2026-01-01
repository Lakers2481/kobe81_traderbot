#!/usr/bin/env python3
from __future__ import annotations

"""
Kobe Master Scheduler v2.0 (24/7)

Enhanced production scheduler with position management and divergence monitoring.
Implements America/New_York times and runs Kobe commands accordingly.

Daily Schedule (ET):
=============================================================================
PRE-MARKET (5:30 - 9:30)
- 05:30 DB_BACKUP          - State backup
- 06:00 DATA_UPDATE        - Warm data cache
- 06:30 MORNING_REPORT     - Generate morning summary
- 06:45 PREMARKET_CHECK    - Data staleness, splits, missing bars
- 08:00 PRE_GAME           - AI Briefing (evidence-locked)
- 09:00 MARKET_NEWS        - Update sentiment
- 09:15 PREMARKET_SCAN     - Build plan (portfolio-aware)

MARKET HOURS (9:30 - 16:00)
- 09:45 FIRST_SCAN         - ENTRY WINDOW - Submit orders
- 09:50+ POSITION_MANAGER  - Every 15 min: stops, exits, P&L
- 10:00+ DIVERGENCE        - Every 30 min: sync validation
- 12:00 HALF_TIME          - AI Briefing + position review
- 12:30 RECONCILE_MIDDAY   - Full broker-OMS reconciliation
- 14:30 AFTERNOON_SCAN     - Refresh Top-3 (portfolio-aware)
- 15:30 SWING_SCANNER      - Swing setups
- 15:55 POSITION_CLOSE_CHECK - Enforce time stops before close

POST-MARKET (16:00 - 22:00)
- 16:00 POST_GAME          - AI Briefing + lessons
- 16:05 EOD_REPORT         - Performance report
- 16:15 RECONCILE_EOD      - Full reconciliation + report
- 17:00 EOD_LEARNING       - Weekly ML training (Fridays)
- 17:15 COGNITIVE_LEARN    - Daily hypothesis testing & edge discovery
- 17:30 LEARN_ANALYSIS     - Daily trade learning analysis
- 18:00 EOD_FINALIZE       - Finalize EOD data after provider delay
SATURDAY MORNING (8:30 AM CT = 9:30 AM ET) - Also runs on holidays!
- 08:30 RESEARCH_DISCOVER   - Pattern discovery
- 09:00 ALPHA_SCREEN_WEEKLY - Alpha screening
- 09:30 WEEKEND_WATCHLIST   - Build watchlist for next trading day

HOLIDAY/WEEKEND SCHEDULE (36 tasks for market-closed days):
- Robot learns, adapts, researches instead of idling on closed days
- Runs on ALL market-closed days (weekends + NYSE holidays)

Early Morning (5:30-6:30 AM ET):
- 05:30 HOLIDAY_BACKUP         - State backup
- 06:00 HOLIDAY_HEALTH_CHECK   - Full system health
- 06:15 HOLIDAY_LOG_CLEANUP    - Purge old logs

Morning (6:30-8:00 AM ET):
- 06:30 HOLIDAY_DATA_INTEGRITY   - Missing bars, duplicates, outliers
- 06:45 HOLIDAY_CORPORATE_ACTIONS- Splits, dividends sync
- 07:00 HOLIDAY_UNIVERSE_REFRESH - Delistings, halted tickers
- 07:30 HOLIDAY_BROKER_TEST      - Broker connectivity test

Research Phase (8:00-10:00 AM ET):
- 08:00 HOLIDAY_RESEARCH_START  - Start research session
- 08:30 HOLIDAY_PATTERN_SCAN    - Scan for new patterns
- 09:00 HOLIDAY_ALPHA_DISCOVERY - Alpha screening
- 09:30 HOLIDAY_EDGE_ANALYSIS   - Edge discovery analysis

Backtesting Phase (10:00 AM-12:00 PM ET):
- 10:00 HOLIDAY_BACKTEST_QUICK  - Quick backtest validation
- 10:30 HOLIDAY_WF_TEST         - Walk-forward test
- 11:00 HOLIDAY_STRATEGY_COMPARE- Strategy comparison
- 11:30 HOLIDAY_PARAM_DRIFT     - Parameter drift check

Optimization & Tuning (12:00-2:00 PM ET):
- 12:00 HOLIDAY_OPTIMIZE_START  - Start optimization
- 12:30 HOLIDAY_GRID_SEARCH     - Grid search parameters
- 13:00 HOLIDAY_THRESHOLD_TUNE  - Tune confidence thresholds
- 13:30 HOLIDAY_RISK_CALIBRATE  - Calibrate risk limits

ML Training (2:00-4:00 PM ET):
- 14:00 HOLIDAY_ML_TRAIN        - ML model training
- 14:30 HOLIDAY_META_RETRAIN    - Meta model retrain
- 15:00 HOLIDAY_ENSEMBLE_UPDATE - Ensemble update
- 15:30 HOLIDAY_HMM_REGIME      - HMM regime recalibration

Cognitive Learning (4:00-6:00 PM ET):
- 16:00 HOLIDAY_COGNITIVE_REFLECT - Cognitive reflection
- 16:30 HOLIDAY_HYPOTHESIS_TEST   - Test hypotheses
- 17:00 HOLIDAY_MEMORY_CONSOLIDATE- Memory consolidation
- 17:30 HOLIDAY_SELF_CALIBRATE    - Self-model calibration

Simulation & Stress Test (6:00-8:00 PM ET):
- 18:00 HOLIDAY_MONTE_CARLO      - Monte Carlo simulation
- 18:30 HOLIDAY_STRESS_TEST      - Stress testing
- 19:00 HOLIDAY_VAR_CALC         - VaR recalculation
- 19:30 HOLIDAY_DRAWDOWN_ANALYSIS- Drawdown analysis

Next Day Prep (8:00-10:00 PM ET):
- 20:00 HOLIDAY_NEXT_DAY_PREP   - Prepare for next day
- 20:30 HOLIDAY_WATCHLIST_BUILD - Build watchlist
- 21:00 HOLIDAY_PREVIEW_SCAN    - Preview mode scan
- 21:30 HOLIDAY_FINAL_BACKUP    - Final backup
- 22:00 HOLIDAY_COMPLETE        - Holiday schedule complete

The scheduler stores last-run markers per tag/date to prevent duplicates.
"""

import argparse
import os
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

# Ensure project root is in Python path for imports BEFORE core imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.clock.tz_utils import fmt_ct, now_et
from core.clock.equities_calendar import EquitiesCalendar
from core.journal import append_journal


def is_early_close_day(date: datetime = None) -> bool:
    """Check if given date is an early close day (1 PM close).

    Early close days per NYSE calendar:
    - July 3 (day before Independence Day)
    - Black Friday (day after Thanksgiving)
    - Christmas Eve (Dec 24)

    Note: Dec 31 (New Year's Eve) is NOT an early close for stocks.
    """
    if date is None:
        date = now_et()
    cal = EquitiesCalendar()
    is_early, _ = cal.is_early_close(date.date() if hasattr(date, 'date') else date)
    return is_early


def is_market_closed_day(date: datetime = None) -> bool:
    """Check if given date is a market-closed day (weekend or holiday).

    Returns True for:
    - Saturdays and Sundays
    - NYSE holidays (New Year's, MLK Day, Presidents Day, Good Friday,
      Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving,
      Christmas, etc.)
    """
    if date is None:
        date = now_et()
    cal = EquitiesCalendar()
    d = date.date() if hasattr(date, 'date') else date
    return not cal.is_trading_day(d)


def get_market_close_time(date: datetime = None) -> dtime:
    """Get market close time for a given date (handles early closes).

    Returns 13:00 for early close days, 16:00 for normal days.
    """
    if date is None:
        date = now_et()
    cal = EquitiesCalendar()
    _, close_time = cal.get_market_hours(date.date() if hasattr(date, 'date') else date)
    return close_time

STATE_FILE = ROOT / 'state' / 'scheduler_master.json'
ET = ZoneInfo('America/New_York')


@dataclass(frozen=True)
class ScheduleEntry:
    tag: str
    time: dtime  # ET


# =============================================================================
# SCHEDULE v2.0 - Enhanced with Position Manager & Divergence Monitor
# =============================================================================
# Pre-market: 05:30-09:30 (data, briefing, planning)
# Market hours: 09:30-16:00 (entries, position management, divergence checks)
# Post-market: 16:00-21:00 (reconciliation, learning, overnight)
# =============================================================================

SCHEDULE: List[ScheduleEntry] = [
    # === PRE-MARKET (5:30 - 9:30 ET) ===
    ScheduleEntry('DB_BACKUP', dtime(5, 30)),
    ScheduleEntry('DATA_UPDATE', dtime(6, 0)),
    ScheduleEntry('MORNING_REPORT', dtime(6, 30)),
    ScheduleEntry('PREMARKET_CHECK', dtime(6, 45)),  # Data staleness, splits check
    ScheduleEntry('PRE_GAME', dtime(8, 0)),          # AI Briefing (evidence-locked)
    ScheduleEntry('MARKET_NEWS', dtime(9, 0)),       # Update sentiment
    ScheduleEntry('PREMARKET_SCAN', dtime(9, 15)),   # Build plan (portfolio-aware)

    # === MARKET OPEN - ENTRY WINDOW (9:45) ===
    ScheduleEntry('FIRST_SCAN', dtime(9, 45)),       # ENTRY WINDOW - Submit orders

    # === POSITION MANAGER (every 15 min during market hours) ===
    ScheduleEntry('POSITION_MANAGER_1', dtime(9, 50)),
    ScheduleEntry('POSITION_MANAGER_2', dtime(10, 5)),
    ScheduleEntry('POSITION_MANAGER_3', dtime(10, 20)),
    ScheduleEntry('POSITION_MANAGER_4', dtime(10, 35)),
    ScheduleEntry('POSITION_MANAGER_5', dtime(10, 50)),
    ScheduleEntry('POSITION_MANAGER_6', dtime(11, 5)),
    ScheduleEntry('POSITION_MANAGER_7', dtime(11, 20)),
    ScheduleEntry('POSITION_MANAGER_8', dtime(11, 35)),
    ScheduleEntry('POSITION_MANAGER_9', dtime(11, 50)),

    # === MID-DAY ===
    ScheduleEntry('HALF_TIME', dtime(12, 0)),        # AI Briefing + position review
    ScheduleEntry('RECONCILE_MIDDAY', dtime(12, 30)), # Full broker-OMS reconciliation

    # === AFTERNOON POSITION MANAGER ===
    ScheduleEntry('POSITION_MANAGER_10', dtime(12, 45)),
    ScheduleEntry('POSITION_MANAGER_11', dtime(13, 0)),
    ScheduleEntry('POSITION_MANAGER_12', dtime(13, 15)),
    ScheduleEntry('POSITION_MANAGER_13', dtime(13, 30)),
    ScheduleEntry('POSITION_MANAGER_14', dtime(13, 45)),
    ScheduleEntry('POSITION_MANAGER_15', dtime(14, 0)),
    ScheduleEntry('POSITION_MANAGER_16', dtime(14, 15)),

    ScheduleEntry('AFTERNOON_SCAN', dtime(14, 30)),  # Refresh Top-3 (portfolio-aware)

    ScheduleEntry('POSITION_MANAGER_17', dtime(14, 45)),
    ScheduleEntry('POSITION_MANAGER_18', dtime(15, 0)),
    ScheduleEntry('POSITION_MANAGER_19', dtime(15, 15)),

    ScheduleEntry('SWING_SCANNER', dtime(15, 30)),   # Swing setups (normal days)

    ScheduleEntry('POSITION_MANAGER_20', dtime(15, 45)),
    ScheduleEntry('POSITION_CLOSE_CHECK', dtime(15, 55)), # Enforce time stops before close

    # === EARLY CLOSE DAY SCHEDULE (1 PM close days: July 3, Black Friday, Dec 24) ===
    # These run ONLY on early close days, ~15 min before 1 PM close
    ScheduleEntry('SWING_SCANNER_EARLY', dtime(12, 45)),      # Swing setups (early close)
    ScheduleEntry('POSITION_CLOSE_CHECK_EARLY', dtime(12, 50)), # Time stops (early close)

    # === MARKET CLOSE (16:00) ===
    ScheduleEntry('POST_GAME', dtime(16, 0)),        # AI Briefing + lessons
    ScheduleEntry('EOD_REPORT', dtime(16, 5)),
    ScheduleEntry('RECONCILE_EOD', dtime(16, 15)),   # Full reconciliation + report

    # === POST-MARKET ===
    ScheduleEntry('EOD_LEARNING', dtime(17, 0)),     # Weekly ML training (Fridays)
    ScheduleEntry('COGNITIVE_LEARN', dtime(17, 15)), # Daily cognitive consolidation (hypothesis testing)
    ScheduleEntry('LEARN_ANALYSIS', dtime(17, 30)),  # Daily trade learning analysis
    ScheduleEntry('EOD_FINALIZE', dtime(18, 0)),     # Finalize EOD data after provider delay

    # === SATURDAY MORNING (Weekend Work) ===
    ScheduleEntry('RESEARCH_DISCOVER', dtime(8, 30)),   # Pattern discovery (Saturdays 7:30 AM CT)
    ScheduleEntry('ALPHA_SCREEN_WEEKLY', dtime(9, 0)),  # Alpha screening (Saturdays 8:00 AM CT)
    ScheduleEntry('WEEKEND_WATCHLIST', dtime(9, 30)),   # Build watchlist (Saturdays 8:30 AM CT) - FINAL

    # === DIVERGENCE MONITOR (runs alongside position manager) ===
    ScheduleEntry('DIVERGENCE_1', dtime(10, 0)),
    ScheduleEntry('DIVERGENCE_2', dtime(10, 30)),
    ScheduleEntry('DIVERGENCE_3', dtime(11, 0)),
    ScheduleEntry('DIVERGENCE_4', dtime(11, 30)),
    ScheduleEntry('DIVERGENCE_5', dtime(12, 15)),
    ScheduleEntry('DIVERGENCE_6', dtime(13, 0)),
    ScheduleEntry('DIVERGENCE_7', dtime(13, 30)),
    ScheduleEntry('DIVERGENCE_8', dtime(14, 0)),
    ScheduleEntry('DIVERGENCE_9', dtime(14, 45)),
    ScheduleEntry('DIVERGENCE_10', dtime(15, 15)),
    ScheduleEntry('DIVERGENCE_11', dtime(15, 45)),

    # === INTRADAY SCANS (every 15 min, 8:00 AM - 3:45 PM ET) ===
    # Lightweight scans to refresh Top-3 picks for manual review - NO auto-execution
    # These supplement the main scans (FIRST_SCAN, AFTERNOON_SCAN, SWING_SCANNER)
    ScheduleEntry('INTRADAY_SCAN_0800', dtime(8, 0)),
    ScheduleEntry('INTRADAY_SCAN_0815', dtime(8, 15)),
    ScheduleEntry('INTRADAY_SCAN_0830', dtime(8, 30)),
    ScheduleEntry('INTRADAY_SCAN_0845', dtime(8, 45)),
    ScheduleEntry('INTRADAY_SCAN_0900', dtime(9, 0)),
    ScheduleEntry('INTRADAY_SCAN_0930', dtime(9, 30)),
    ScheduleEntry('INTRADAY_SCAN_1000', dtime(10, 0)),
    ScheduleEntry('INTRADAY_SCAN_1015', dtime(10, 15)),
    ScheduleEntry('INTRADAY_SCAN_1030', dtime(10, 30)),
    ScheduleEntry('INTRADAY_SCAN_1045', dtime(10, 45)),
    ScheduleEntry('INTRADAY_SCAN_1100', dtime(11, 0)),
    ScheduleEntry('INTRADAY_SCAN_1115', dtime(11, 15)),
    ScheduleEntry('INTRADAY_SCAN_1130', dtime(11, 30)),
    ScheduleEntry('INTRADAY_SCAN_1145', dtime(11, 45)),
    ScheduleEntry('INTRADAY_SCAN_1200', dtime(12, 0)),
    ScheduleEntry('INTRADAY_SCAN_1215', dtime(12, 15)),
    ScheduleEntry('INTRADAY_SCAN_1230', dtime(12, 30)),
    ScheduleEntry('INTRADAY_SCAN_1245', dtime(12, 45)),
    ScheduleEntry('INTRADAY_SCAN_1300', dtime(13, 0)),
    ScheduleEntry('INTRADAY_SCAN_1315', dtime(13, 15)),
    ScheduleEntry('INTRADAY_SCAN_1330', dtime(13, 30)),
    ScheduleEntry('INTRADAY_SCAN_1345', dtime(13, 45)),
    ScheduleEntry('INTRADAY_SCAN_1400', dtime(14, 0)),
    ScheduleEntry('INTRADAY_SCAN_1415', dtime(14, 15)),
    ScheduleEntry('INTRADAY_SCAN_1445', dtime(14, 45)),
    ScheduleEntry('INTRADAY_SCAN_1500', dtime(15, 0)),
    ScheduleEntry('INTRADAY_SCAN_1515', dtime(15, 15)),
    ScheduleEntry('INTRADAY_SCAN_1545', dtime(15, 45)),

    # =============================================================================
    # HOLIDAY/WEEKEND SCHEDULE (36 tasks for market-closed days)
    # =============================================================================
    # These run ONLY on market-closed days (holidays + weekends)
    # Robot learns, adapts, researches, optimizes instead of idling
    # =============================================================================

    # === EARLY MORNING: System Health & Backup (5:30-6:30 AM ET) ===
    ScheduleEntry('HOLIDAY_BACKUP', dtime(5, 30)),           # State backup
    ScheduleEntry('HOLIDAY_HEALTH_CHECK', dtime(6, 0)),      # Full system health
    ScheduleEntry('HOLIDAY_LOG_CLEANUP', dtime(6, 15)),      # Purge old logs

    # === MORNING: Data Integrity & Prep (6:30-8:00 AM ET) ===
    ScheduleEntry('HOLIDAY_DATA_INTEGRITY', dtime(6, 30)),   # Missing bars, duplicates, outliers
    ScheduleEntry('HOLIDAY_CORPORATE_ACTIONS', dtime(6, 45)),# Splits, dividends sync
    ScheduleEntry('HOLIDAY_UNIVERSE_REFRESH', dtime(7, 0)),  # Delistings, halted tickers
    ScheduleEntry('HOLIDAY_BROKER_TEST', dtime(7, 30)),      # Broker connectivity test

    # === RESEARCH PHASE: Pattern Discovery (8:00-10:00 AM ET) ===
    ScheduleEntry('HOLIDAY_RESEARCH_START', dtime(8, 0)),    # Start research session
    ScheduleEntry('HOLIDAY_PATTERN_SCAN', dtime(8, 30)),     # Scan for new patterns
    ScheduleEntry('HOLIDAY_ALPHA_DISCOVERY', dtime(9, 0)),   # Alpha screening
    ScheduleEntry('HOLIDAY_EDGE_ANALYSIS', dtime(9, 30)),    # Edge discovery analysis

    # === BACKTESTING PHASE: Validation (10:00 AM-12:00 PM ET) ===
    ScheduleEntry('HOLIDAY_BACKTEST_QUICK', dtime(10, 0)),   # Quick backtest validation
    ScheduleEntry('HOLIDAY_WF_TEST', dtime(10, 30)),         # Walk-forward test
    ScheduleEntry('HOLIDAY_STRATEGY_COMPARE', dtime(11, 0)), # Strategy comparison
    ScheduleEntry('HOLIDAY_PARAM_DRIFT', dtime(11, 30)),     # Parameter drift check

    # === MIDDAY: Optimization & Tuning (12:00-2:00 PM ET) ===
    ScheduleEntry('HOLIDAY_OPTIMIZE_START', dtime(12, 0)),   # Start optimization
    ScheduleEntry('HOLIDAY_GRID_SEARCH', dtime(12, 30)),     # Grid search parameters
    ScheduleEntry('HOLIDAY_THRESHOLD_TUNE', dtime(13, 0)),   # Tune confidence thresholds
    ScheduleEntry('HOLIDAY_RISK_CALIBRATE', dtime(13, 30)),  # Calibrate risk limits

    # === AFTERNOON: ML Training (2:00-4:00 PM ET) ===
    ScheduleEntry('HOLIDAY_ML_TRAIN', dtime(14, 0)),         # ML model training
    ScheduleEntry('HOLIDAY_META_RETRAIN', dtime(14, 30)),    # Meta model retrain
    ScheduleEntry('HOLIDAY_ENSEMBLE_UPDATE', dtime(15, 0)),  # Ensemble update
    ScheduleEntry('HOLIDAY_HMM_REGIME', dtime(15, 30)),      # HMM regime recalibration

    # === EVENING: Cognitive Learning (4:00-6:00 PM ET) ===
    ScheduleEntry('HOLIDAY_COGNITIVE_REFLECT', dtime(16, 0)),# Cognitive reflection
    ScheduleEntry('HOLIDAY_HYPOTHESIS_TEST', dtime(16, 30)), # Test hypotheses
    ScheduleEntry('HOLIDAY_MEMORY_CONSOLIDATE', dtime(17, 0)),# Memory consolidation
    ScheduleEntry('HOLIDAY_SELF_CALIBRATE', dtime(17, 30)),  # Self-model calibration

    # === NIGHT: Simulation & Stress Test (6:00-8:00 PM ET) ===
    ScheduleEntry('HOLIDAY_MONTE_CARLO', dtime(18, 0)),      # Monte Carlo simulation
    ScheduleEntry('HOLIDAY_STRESS_TEST', dtime(18, 30)),     # Stress testing
    ScheduleEntry('HOLIDAY_VAR_CALC', dtime(19, 0)),         # VaR recalculation
    ScheduleEntry('HOLIDAY_DRAWDOWN_ANALYSIS', dtime(19, 30)),# Drawdown analysis

    # === LATE NIGHT: Prep for Next Trading Day (8:00-10:00 PM ET) ===
    ScheduleEntry('HOLIDAY_NEXT_DAY_PREP', dtime(20, 0)),    # Prepare for next day
    ScheduleEntry('HOLIDAY_WATCHLIST_BUILD', dtime(20, 30)), # Build watchlist
    ScheduleEntry('HOLIDAY_PREVIEW_SCAN', dtime(21, 0)),     # Preview mode scan
    ScheduleEntry('HOLIDAY_FINAL_BACKUP', dtime(21, 30)),    # Final backup
    ScheduleEntry('HOLIDAY_COMPLETE', dtime(22, 0)),         # Holiday schedule complete
]


def load_state() -> Dict[str, str]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding='utf-8'))
    except Exception:
        return {}


def save_state(state: Dict[str, str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding='utf-8')


def mark_ran(state: Dict[str, str], tag: str, yyyymmdd: str) -> None:
    state[f'{tag}'] = yyyymmdd


def already_ran(state: Dict[str, str], tag: str, yyyymmdd: str) -> bool:
    return state.get(tag) == yyyymmdd


def run_cmd(args: List[str]) -> int:
    p = subprocess.run(args, cwd=str(ROOT))
    return p.returncode


def do_pregame(universe: str, dotenv: str, scan_date: str) -> None:
    run_cmd([sys.executable, str(ROOT / 'scripts/update_sentiment_cache.py'), '--universe', universe, '--date', scan_date, '--dotenv', dotenv])


def do_first_scan(universe: str, dotenv: str, cap: int, min_conf: float, scan_date: str) -> None:
    run_cmd([sys.executable, str(ROOT / 'scripts/run_daily_pipeline.py'), '--universe', universe, '--cap', str(cap), '--date', scan_date, '--dotenv', dotenv, '--min-conf', str(min_conf), '--ensure-top3'])


def do_refresh_top3(universe: str, dotenv: str, cap: int, scan_date: str) -> int:
    """Refresh top-3 picks using the scanner with deterministic mode."""
    args = [
        sys.executable, str(ROOT / 'scripts/scan.py'),
        '--dotenv', dotenv,
        '--strategy', 'all',
        '--cap', str(cap),
        '--top3', '--ml',
        '--date', scan_date,
        '--ensure-top3',
        '--deterministic'  # Required for reproducible results per STATUS.md
    ]
    return run_cmd(args)


def do_weekly_learning(wfdir: str, dotenv: str, min_delta: float, min_test: int) -> None:
    # Only run on Fridays
    if datetime.now(ET).weekday() == 4:
        run_cmd([sys.executable, str(ROOT / 'scripts/run_weekly_training.py'), '--wfdir', wfdir, '--dotenv', dotenv, '--min-delta', str(min_delta), '--min-test', str(min_test)])


def now_et_date_time() -> Tuple[str, dtime]:
    now = now_et()
    return now.date().isoformat(), dtime(now.hour, now.minute)


# =============================================================================
# CONTINUOUS WORK SYSTEM - Never idle, always learning
# =============================================================================
# When no scheduled task is running, the robot picks up continuous work:
# - Stock-by-stock analysis (cycles through 900 stocks)
# - Pattern discovery on random stocks
# - Mini-backtests on random symbols
# - Alpha research and edge discovery
# - Data quality checks
# - Feature engineering experiments
# =============================================================================

import random

CONTINUOUS_WORK_QUEUE = [
    'ANALYZE_STOCK',        # Analyze single stock from universe
    'PATTERN_HUNT',         # Hunt for patterns in random stock
    'MINI_BACKTEST',        # Quick backtest on random symbol
    'ALPHA_RESEARCH',       # Research alpha factors
    'DATA_QUALITY',         # Check data quality for random stocks
    'FEATURE_EXPERIMENT',   # Test feature engineering ideas
    'CORRELATION_CHECK',    # Check correlations between stocks
    'REGIME_ANALYSIS',      # Analyze current regime
    'VOLATILITY_STUDY',     # Study volatility patterns
    'SECTOR_ROTATION',      # Analyze sector rotation
]

# Track which stock index we're on for systematic analysis
_stock_analysis_index = 0


def load_universe_symbols(universe_path: str) -> List[str]:
    """Load symbols from universe file."""
    import pandas as pd
    try:
        df = pd.read_csv(universe_path)
        return df['symbol'].tolist() if 'symbol' in df.columns else df.iloc[:, 0].tolist()
    except Exception:
        return []


def do_continuous_work(universe: str, dotenv: str, cap: int, send_fn=None) -> None:
    """Execute continuous work when no scheduled task is running.

    Never idle - always learning, discovering, analyzing.
    """
    global _stock_analysis_index

    # Pick a random work type
    work_type = random.choice(CONTINUOUS_WORK_QUEUE)
    symbols = load_universe_symbols(universe)

    if not symbols:
        return

    # Get current stock for systematic analysis
    current_symbol = symbols[_stock_analysis_index % len(symbols)]
    random_symbol = random.choice(symbols)

    try:
        if work_type == 'ANALYZE_STOCK':
            # Systematic stock-by-stock analysis
            rc = run_cmd([sys.executable, str(ROOT / 'scripts/analyze_stock.py'),
                         '--symbol', current_symbol, '--dotenv', dotenv])
            _stock_analysis_index += 1  # Move to next stock
            if send_fn and rc == 0:
                progress = f"{_stock_analysis_index}/{len(symbols)}"
                send_fn(f"<b>CONTINUOUS</b> Analyzed {current_symbol} ({progress})")

        elif work_type == 'PATTERN_HUNT':
            # Hunt for patterns in random stock
            rc = run_cmd([sys.executable, str(ROOT / 'scripts/pattern_hunt.py'),
                         '--symbol', random_symbol, '--dotenv', dotenv])

        elif work_type == 'MINI_BACKTEST':
            # Quick backtest on random symbol
            rc = run_cmd([sys.executable, str(ROOT / 'scripts/mini_backtest.py'),
                         '--symbol', random_symbol, '--days', '90', '--dotenv', dotenv])

        elif work_type == 'ALPHA_RESEARCH':
            # Research alpha factors
            rc = run_cmd([sys.executable, str(ROOT / 'scripts/alpha_research.py'),
                         '--symbol', random_symbol, '--dotenv', dotenv])

        elif work_type == 'DATA_QUALITY':
            # Check data quality for random stocks
            batch = random.sample(symbols, min(10, len(symbols)))
            rc = run_cmd([sys.executable, str(ROOT / 'scripts/check_data_quality.py'),
                         '--symbols', ','.join(batch), '--dotenv', dotenv])

        elif work_type == 'FEATURE_EXPERIMENT':
            # Test feature engineering ideas
            rc = run_cmd([sys.executable, str(ROOT / 'scripts/feature_experiment.py'),
                         '--symbol', random_symbol, '--dotenv', dotenv])

        elif work_type == 'CORRELATION_CHECK':
            # Check correlations between stocks
            batch = random.sample(symbols, min(20, len(symbols)))
            rc = run_cmd([sys.executable, str(ROOT / 'scripts/correlation_check.py'),
                         '--symbols', ','.join(batch), '--dotenv', dotenv])

        elif work_type == 'REGIME_ANALYSIS':
            # Analyze current regime
            rc = run_cmd([sys.executable, str(ROOT / 'scripts/regime_analysis.py'),
                         '--dotenv', dotenv])

        elif work_type == 'VOLATILITY_STUDY':
            # Study volatility patterns
            rc = run_cmd([sys.executable, str(ROOT / 'scripts/volatility_study.py'),
                         '--symbol', random_symbol, '--dotenv', dotenv])

        elif work_type == 'SECTOR_ROTATION':
            # Analyze sector rotation
            rc = run_cmd([sys.executable, str(ROOT / 'scripts/sector_rotation.py'),
                         '--dotenv', dotenv])

    except Exception as e:
        # Log error but don't crash - continuous work is best-effort
        print(f"Continuous work error ({work_type}): {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description='Kobe Master Scheduler (24/7)')
    ap.add_argument('--universe', type=str, default='data/universe/optionable_liquid_900.csv')
    ap.add_argument('--cap', type=int, default=900)
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--min-conf', type=float, default=0.60)
    ap.add_argument('--wfdir', type=str, default='wf_outputs')
    ap.add_argument('--min-delta', type=float, default=0.01)
    ap.add_argument('--min-test', type=int, default=100)
    ap.add_argument('--tick-seconds', type=int, default=20)
    ap.add_argument('--telegram', action='store_true', help='Send Telegram notifications for scheduler events')
    ap.add_argument('--telegram-dotenv', type=str, default=str(ROOT / '.env'), help='Optional .env file for Telegram keys (uses project .env by default)')
    args = ap.parse_args()

    # Bootstrap Telegram env if requested
    send_fn = None
    if args.telegram:
        # Load local .env first
        try:
            from config.env_loader import load_env
            de = Path(args.dotenv)
            if de.exists():
                load_env(de)
        except Exception:
            pass
        # Load 2K28 .env as fallback for TELEGRAM_* keys
        tdot = Path(args.telegram_dotenv)
        if tdot.exists():
            try:
                for line in tdot.read_text(encoding='utf-8').splitlines():
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    k = k.strip(); v = v.strip().strip('"').strip("'")
                    if k.startswith('TELEGRAM_') and not os.getenv(k):
                        os.environ[k] = v
            except Exception:
                pass
        try:
            from core.alerts import send_telegram as _send
            send_fn = _send
        except Exception:
            send_fn = None

    state = load_state()
    try:
        from core.clock.tz_utils import fmt_ct, now_et
        now = now_et()
        start_msg = f"Kobe Master Scheduler started. Display: {fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
    except Exception:
        start_msg = 'Kobe Master Scheduler started. Times in America/New_York (ET).'
    print(start_msg)
    if send_fn:
        try:
            from core.clock.tz_utils import fmt_ct
            now = now_et()
            send_fn(f"<b>Kobe Scheduler</b> started • {fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET")
        except Exception:
            pass
    try:
        append_journal('scheduler_started', {'telegram': bool(send_fn)})
    except Exception:
        pass

    while True:
        ymd, hhmm = now_et_date_time()
        for entry in SCHEDULE:
            try:
                if hhmm.hour == entry.time.hour and hhmm.minute == entry.time.minute:
                    if not already_ran(state, entry.tag, ymd):
                        print(f"[{fmt_ct(now_et())} | {hhmm.strftime('%I:%M %p').lstrip('0')} ET] Running {entry.tag}...")
                        # Determine scan_date based on market phase:
                        # - Pre-market (before 9:30): use today for planning
                        # - Market hours (9:30-16:00): use today for live data
                        # - Post-market (after 16:00): use today (finalized data)
                        now_time = datetime.now(ET).time()
                        market_open = dtime(9, 30)
                        market_close = dtime(16, 0)
                        pre_market_jobs = ('PRE_GAME', 'MARKET_NEWS', 'PREMARKET_SCAN', 'PREMARKET_CHECK', 'MORNING_REPORT')

                        if market_open <= now_time <= market_close:
                            # During market hours: always use today's data
                            scan_date = datetime.now(ET).date().isoformat()
                        elif entry.tag in pre_market_jobs:
                            # Pre-market jobs: use today for planning
                            scan_date = datetime.now(ET).date().isoformat()
                        else:
                            # Post-market: use today (data should be finalized)
                            scan_date = datetime.now(ET).date().isoformat()

                        if entry.tag == 'MORNING_REPORT':
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/morning_report.py')])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et()
                                    stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")
                        elif entry.tag == 'PRE_GAME':
                            do_pregame(args.universe, args.dotenv, scan_date)
                            # Generate pre-game plan document (legacy)
                            run_cmd([sys.executable, str(ROOT / 'scripts/pre_game_plan.py'), '--universe', args.universe, '--cap', str(args.cap), '--dotenv', args.dotenv, '--date', scan_date])
                            # Generate comprehensive PRE_GAME briefing with LLM/ML analysis
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/generate_briefing.py'),
                                     '--phase', 'pregame', '--universe', args.universe, '--cap', str(args.cap),
                                     '--dotenv', args.dotenv, '--date', scan_date, '--telegram'])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")
                        elif entry.tag == 'MARKET_NEWS':
                            do_pregame(args.universe, args.dotenv, scan_date)
                        elif entry.tag == 'PREMARKET_SCAN':
                            # Build plan without submitting
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/pre_game_plan.py'), '--universe', args.universe, '--cap', str(args.cap), '--dotenv', args.dotenv, '--date', scan_date])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")
                        elif entry.tag == 'FIRST_SCAN':
                            do_first_scan(args.universe, args.dotenv, args.cap, args.min_conf, scan_date)
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] completed")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> completed")
                        elif entry.tag == 'HALF_TIME':
                            # Refresh Top-3 signals
                            do_refresh_top3(args.universe, args.dotenv, args.cap, scan_date)
                            # Generate comprehensive HALF_TIME briefing with position analysis
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/generate_briefing.py'),
                                     '--phase', 'halftime', '--dotenv', args.dotenv, '--telegram'])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if (rc is None or rc == 0) else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if (rc is None or rc == 0) else 'failed'}")
                        elif entry.tag == 'AFTERNOON_SCAN':
                            rc = do_refresh_top3(args.universe, args.dotenv, args.cap, scan_date)
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if (rc is None or rc == 0) else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if (rc is None or rc == 0) else 'failed'}")
                        elif entry.tag == 'SWING_SCANNER':
                            # Skip on early close days (use SWING_SCANNER_EARLY instead)
                            if is_early_close_day():
                                print(f"  Skipping SWING_SCANNER - early close day (use SWING_SCANNER_EARLY)")
                            else:
                                rc = do_refresh_top3(args.universe, args.dotenv, args.cap, scan_date)
                                if send_fn:
                                    try:
                                        from core.clock.tz_utils import fmt_ct
                                        now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                        send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if (rc is None or rc == 0) else 'failed'}")
                                    except Exception:
                                        send_fn(f"<b>{entry.tag}</b> {'completed' if (rc is None or rc == 0) else 'failed'}")
                        elif entry.tag == 'SWING_SCANNER_EARLY':
                            # Only run on early close days (1 PM close)
                            if is_early_close_day():
                                rc = do_refresh_top3(args.universe, args.dotenv, args.cap, scan_date)
                                if send_fn:
                                    try:
                                        from core.clock.tz_utils import fmt_ct
                                        now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                        send_fn(f"<b>SWING_SCANNER_EARLY</b> [{stamp}] Early close day - swing scan complete")
                                    except Exception:
                                        send_fn(f"<b>SWING_SCANNER_EARLY</b> Early close day - swing scan complete")
                            else:
                                pass  # Skip on normal days
                        elif entry.tag == 'POSITION_CLOSE_CHECK_EARLY':
                            # Only run on early close days (1 PM close)
                            if is_early_close_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/position_manager.py'),
                                              '--dotenv', args.dotenv, '--force-time-stop'])
                                if send_fn:
                                    try:
                                        from core.clock.tz_utils import fmt_ct
                                        now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                        send_fn(f"<b>POSITION_CLOSE_CHECK_EARLY</b> [{stamp}] Early close day - time stops checked")
                                    except Exception:
                                        send_fn(f"<b>POSITION_CLOSE_CHECK_EARLY</b> Early close day - time stops checked")
                            else:
                                pass  # Skip on normal days
                        elif entry.tag == 'DB_BACKUP':
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/backup_state.py')])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")
                        elif entry.tag == 'DATA_UPDATE':
                            # Prefetch last 730 days to warm the cache
                            start = (datetime.now(ET).date() - timedelta(days=730)).isoformat()
                            end = datetime.now(ET).date().isoformat()
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/prefetch_polygon_universe.py'),
                                     '--universe', args.universe, '--start', start, '--end', end, '--concurrency', '3', '--dotenv', args.dotenv])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")
                        elif entry.tag == 'EOD_REPORT':
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/eod_report.py')])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")
                        elif entry.tag == 'EOD_LEARNING':
                            do_weekly_learning(args.wfdir, args.dotenv, args.min_delta, args.min_test)
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] completed (weekly promote)")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> completed (weekly promote)")

                        # === COGNITIVE LEARNING (Daily) ===
                        elif entry.tag == 'COGNITIVE_LEARN':
                            # Run daily cognitive consolidation - hypothesis testing & edge discovery
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/cognitive_learn.py'),
                                         '--dotenv', args.dotenv, '--consolidate', 'daily'])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'} (hypothesis testing)")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")

                        elif entry.tag == 'LEARN_ANALYSIS':
                            # Daily trade learning analysis - what Kobe learned
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/learn.py'), '--period', '1'])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'} (trade analysis)")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")

                        # === SATURDAY/HOLIDAY MORNING TASKS (Weekend + Holiday Work) ===
                        # Now runs on ANY market-closed day (weekends + holidays)
                        elif entry.tag == 'RESEARCH_DISCOVER':
                            # Pattern discovery - runs on all market-closed days
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/research_discover.py'),
                                             '--cap', str(args.cap)])
                                if send_fn:
                                    try:
                                        from core.clock.tz_utils import fmt_ct
                                        now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                        send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'} (pattern discovery)")
                                    except Exception:
                                        send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")

                        elif entry.tag == 'ALPHA_SCREEN_WEEKLY':
                            # Alpha screening - runs on all market-closed days
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/run_alpha_screener.py'),
                                             '--universe', args.universe, '--top', '20'])
                                if send_fn:
                                    try:
                                        from core.clock.tz_utils import fmt_ct
                                        now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                        send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'} (alpha screening)")
                                    except Exception:
                                        send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")

                        elif entry.tag == 'WEEKEND_WATCHLIST':
                            # Build watchlist for next trading day - runs on all market-closed days
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/scan.py'),
                                             '--cap', str(args.cap), '--deterministic', '--top3',
                                             '--dotenv', args.dotenv])
                                if send_fn:
                                    try:
                                        from core.clock.tz_utils import fmt_ct
                                        now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                        # Find next trading day
                                        cal = EquitiesCalendar()
                                        next_day = datetime.now(ET).date() + timedelta(days=1)
                                        while not cal.is_trading_day(next_day):
                                            next_day += timedelta(days=1)
                                        send_fn(f"<b>WATCHLIST READY</b> [{stamp}] Prepared for {next_day}!")
                                    except Exception:
                                        send_fn(f"<b>WATCHLIST READY</b> Prepared for next trading day!")

                        elif entry.tag == 'POST_GAME':
                            # Generate comprehensive POST_GAME briefing with full day analysis
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/generate_briefing.py'),
                                     '--phase', 'postgame', '--dotenv', args.dotenv, '--telegram'])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'failed'}")

                        # === POSITION MANAGER (v2.0) ===
                        elif entry.tag.startswith('POSITION_MANAGER_'):
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/position_manager.py'), '--dotenv', args.dotenv])
                            # Only send Telegram on issues (rc != 0)
                            if send_fn and rc != 0:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>POSITION_MANAGER</b> [{stamp}] exit(s) executed or issues detected")
                                except Exception:
                                    pass
                        elif entry.tag == 'POSITION_CLOSE_CHECK':
                            # Skip on early close days (use POSITION_CLOSE_CHECK_EARLY instead)
                            if is_early_close_day():
                                print(f"  Skipping POSITION_CLOSE_CHECK - early close day (use POSITION_CLOSE_CHECK_EARLY)")
                            else:
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/position_manager.py'), '--dotenv', args.dotenv])
                                if send_fn and rc != 0:
                                    try:
                                        from core.clock.tz_utils import fmt_ct
                                        now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                        send_fn(f"<b>POSITION_CLOSE_CHECK</b> [{stamp}] exit(s) executed or issues detected")
                                    except Exception:
                                        pass

                        # === DIVERGENCE MONITOR (v2.0) ===
                        elif entry.tag.startswith('DIVERGENCE_'):
                            rc = run_cmd([sys.executable, str(ROOT / 'monitor/divergence_monitor.py'), '--dotenv', args.dotenv])
                            # Only send Telegram on critical issues (rc == 2)
                            if send_fn and rc == 2:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>DIVERGENCE ALERT</b> [{stamp}] Critical divergence detected!")
                                except Exception:
                                    pass

                        # === RECONCILIATION (v2.0) ===
                        elif entry.tag in ('RECONCILE_MIDDAY', 'RECONCILE_EOD'):
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/reconcile_alpaca.py'), '--dotenv', args.dotenv])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>{entry.tag}</b> [{stamp}] {'completed' if rc == 0 else 'discrepancies found'}")
                                except Exception:
                                    send_fn(f"<b>{entry.tag}</b> {'completed' if rc == 0 else 'discrepancies found'}")

                        # === PREMARKET CHECK (v2.0) ===
                        elif entry.tag == 'PREMARKET_CHECK':
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/premarket_check.py'), '--dotenv', args.dotenv])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    status = 'PASS' if rc == 0 else ('WARN' if rc == 1 else 'FAIL')
                                    send_fn(f"<b>PREMARKET_CHECK</b> [{stamp}] {status}")
                                except Exception:
                                    send_fn(f"<b>PREMARKET_CHECK</b> {'PASS' if rc == 0 else 'issues detected'}")

                        # === EOD FINALIZE (v2.0) ===
                        elif entry.tag == 'EOD_FINALIZE':
                            rc = run_cmd([sys.executable, str(ROOT / 'scripts/eod_finalize.py'), '--dotenv', args.dotenv, '--max-wait', '45'])
                            if send_fn:
                                try:
                                    from core.clock.tz_utils import fmt_ct
                                    now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                    send_fn(f"<b>EOD_FINALIZE</b> [{stamp}] {'completed' if rc == 0 else 'failed'}")
                                except Exception:
                                    send_fn(f"<b>EOD_FINALIZE</b> {'completed' if rc == 0 else 'failed'}")

                        # === INTRADAY SCANS (v2.2) - Every 15 min comprehensive check ===
                        # 1. Refresh Top-3 picks for manual review
                        # 2. Check/manage open positions (stops, exits)
                        # 3. Health status validation
                        elif entry.tag.startswith('INTRADAY_SCAN_'):
                            # Step 1: Refresh Top-3 signals
                            rc_scan = do_refresh_top3(args.universe, args.dotenv, args.cap, scan_date)

                            # Step 2: Position management (check stops, time exits, P&L)
                            rc_pos = run_cmd([sys.executable, str(ROOT / 'scripts/position_manager.py'), '--dotenv', args.dotenv])

                            # Step 3: Health/divergence check (broker sync validation)
                            rc_health = run_cmd([sys.executable, str(ROOT / 'monitor/divergence_monitor.py'), '--dotenv', args.dotenv])

                            # Notify on any issues
                            if send_fn:
                                issues = []
                                if rc_scan != 0:
                                    issues.append(f"scan:{rc_scan}")
                                if rc_pos != 0:
                                    issues.append(f"positions:{rc_pos}")
                                if rc_health == 2:  # Critical divergence
                                    issues.append("DIVERGENCE!")

                                if issues:
                                    try:
                                        from core.clock.tz_utils import fmt_ct
                                        now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                                        send_fn(f"<b>INTRADAY_CHECK</b> [{stamp}] issues: {', '.join(issues)}")
                                    except Exception:
                                        pass

                        # =============================================================================
                        # HOLIDAY/WEEKEND SCHEDULE HANDLERS (36 tasks)
                        # =============================================================================
                        # All holiday tasks check is_market_closed_day() before running
                        # Robot learns, adapts, researches instead of idling on closed days
                        # =============================================================================

                        # === EARLY MORNING: System Health & Backup ===
                        elif entry.tag == 'HOLIDAY_BACKUP':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/backup_state.py')])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_BACKUP</b> State backup {'completed' if rc == 0 else 'failed'}")

                        elif entry.tag == 'HOLIDAY_HEALTH_CHECK':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/preflight.py'), '--dotenv', args.dotenv])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_HEALTH_CHECK</b> System health {'OK' if rc == 0 else 'ISSUES'}")

                        elif entry.tag == 'HOLIDAY_LOG_CLEANUP':
                            if is_market_closed_day():
                                # Purge logs older than 30 days
                                rc = run_cmd([sys.executable, '-c',
                                    'import os, time; from pathlib import Path; '
                                    '[f.unlink() for f in Path("logs").glob("*.log") '
                                    'if time.time() - f.stat().st_mtime > 30*86400]'])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_LOG_CLEANUP</b> Old logs purged")

                        # === MORNING: Data Integrity & Prep ===
                        elif entry.tag == 'HOLIDAY_DATA_INTEGRITY':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/data_integrity_check.py'),
                                             '--universe', args.universe, '--dotenv', args.dotenv])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_DATA_INTEGRITY</b> {'PASS' if rc == 0 else 'issues found'}")

                        elif entry.tag == 'HOLIDAY_CORPORATE_ACTIONS':
                            if is_market_closed_day():
                                # Sync corporate actions (splits, dividends)
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/sync_corporate_actions.py'),
                                             '--universe', args.universe, '--dotenv', args.dotenv])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_CORPORATE_ACTIONS</b> {'synced' if rc == 0 else 'failed'}")

                        elif entry.tag == 'HOLIDAY_UNIVERSE_REFRESH':
                            if is_market_closed_day():
                                # Check for delistings, halted tickers
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/validate_universe.py'),
                                             '--universe', args.universe, '--dotenv', args.dotenv])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_UNIVERSE_REFRESH</b> {'OK' if rc == 0 else 'updates needed'}")

                        elif entry.tag == 'HOLIDAY_BROKER_TEST':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/preflight.py'),
                                             '--dotenv', args.dotenv, '--broker-only'])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_BROKER_TEST</b> Broker {'connected' if rc == 0 else 'OFFLINE'}")

                        # === RESEARCH PHASE: Pattern Discovery ===
                        elif entry.tag == 'HOLIDAY_RESEARCH_START':
                            if is_market_closed_day():
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_RESEARCH</b> Starting research session...")
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/research_discover.py'),
                                             '--cap', str(args.cap)])

                        elif entry.tag == 'HOLIDAY_PATTERN_SCAN':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/pattern_scanner.py'),
                                             '--universe', args.universe, '--top', '50'])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_PATTERN_SCAN</b> Pattern scan complete")

                        elif entry.tag == 'HOLIDAY_ALPHA_DISCOVERY':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/run_alpha_screener.py'),
                                             '--universe', args.universe, '--top', '20'])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_ALPHA_DISCOVERY</b> Alpha screening complete")

                        elif entry.tag == 'HOLIDAY_EDGE_ANALYSIS':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/cognitive_learn.py'),
                                             '--dotenv', args.dotenv, '--edge-discovery'])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_EDGE_ANALYSIS</b> Edge analysis complete")

                        # === BACKTESTING PHASE: Validation ===
                        elif entry.tag == 'HOLIDAY_BACKTEST_QUICK':
                            if is_market_closed_day():
                                # Quick validation backtest (last 3 months)
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/backtest_dual_strategy.py'),
                                             '--universe', args.universe, '--cap', '100',
                                             '--start', (datetime.now(ET) - timedelta(days=90)).strftime('%Y-%m-%d'),
                                             '--end', datetime.now(ET).strftime('%Y-%m-%d')])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_BACKTEST_QUICK</b> Quick validation complete")

                        elif entry.tag == 'HOLIDAY_WF_TEST':
                            if is_market_closed_day():
                                # Walk-forward test (last year)
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/run_wf_polygon.py'),
                                             '--universe', args.universe, '--cap', '150',
                                             '--start', (datetime.now(ET) - timedelta(days=365)).strftime('%Y-%m-%d'),
                                             '--end', datetime.now(ET).strftime('%Y-%m-%d'),
                                             '--train-days', '126', '--test-days', '21'])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_WF_TEST</b> Walk-forward test complete")

                        elif entry.tag == 'HOLIDAY_STRATEGY_COMPARE':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/strategy_showdown.py'),
                                             '--universe', args.universe, '--cap', '100'])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_STRATEGY_COMPARE</b> Strategy comparison complete")

                        elif entry.tag == 'HOLIDAY_PARAM_DRIFT':
                            if is_market_closed_day():
                                # Check for parameter drift
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/param_drift_check.py'),
                                             '--dotenv', args.dotenv])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_PARAM_DRIFT</b> Drift check complete")

                        # === MIDDAY: Optimization & Tuning ===
                        elif entry.tag == 'HOLIDAY_OPTIMIZE_START':
                            if is_market_closed_day():
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_OPTIMIZE</b> Starting optimization phase...")

                        elif entry.tag == 'HOLIDAY_GRID_SEARCH':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/grid_search_params.py'),
                                             '--universe', args.universe, '--cap', '100'])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_GRID_SEARCH</b> Grid search complete")

                        elif entry.tag == 'HOLIDAY_THRESHOLD_TUNE':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/tune_thresholds.py'),
                                             '--dotenv', args.dotenv])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_THRESHOLD_TUNE</b> Threshold tuning complete")

                        elif entry.tag == 'HOLIDAY_RISK_CALIBRATE':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/calibrate_risk.py'),
                                             '--dotenv', args.dotenv])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_RISK_CALIBRATE</b> Risk calibration complete")

                        # === AFTERNOON: ML Training ===
                        elif entry.tag == 'HOLIDAY_ML_TRAIN':
                            if is_market_closed_day():
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_ML_TRAIN</b> Starting ML training...")
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/train_meta.py'),
                                             '--wfdir', args.wfdir])

                        elif entry.tag == 'HOLIDAY_META_RETRAIN':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/train_meta.py'),
                                             '--wfdir', args.wfdir, '--retrain'])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_META_RETRAIN</b> Meta model retrained")

                        elif entry.tag == 'HOLIDAY_ENSEMBLE_UPDATE':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/update_ensemble.py'),
                                             '--dotenv', args.dotenv])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_ENSEMBLE_UPDATE</b> Ensemble updated")

                        elif entry.tag == 'HOLIDAY_HMM_REGIME':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/recalibrate_hmm.py'),
                                             '--dotenv', args.dotenv])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_HMM_REGIME</b> HMM regime recalibrated")

                        # === EVENING: Cognitive Learning ===
                        elif entry.tag == 'HOLIDAY_COGNITIVE_REFLECT':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/cognitive_learn.py'),
                                             '--dotenv', args.dotenv, '--consolidate', 'weekly'])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_COGNITIVE_REFLECT</b> Cognitive reflection complete")

                        elif entry.tag == 'HOLIDAY_HYPOTHESIS_TEST':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/cognitive_learn.py'),
                                             '--dotenv', args.dotenv, '--test-hypotheses'])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_HYPOTHESIS_TEST</b> Hypothesis testing complete")

                        elif entry.tag == 'HOLIDAY_MEMORY_CONSOLIDATE':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/cognitive_learn.py'),
                                             '--dotenv', args.dotenv, '--consolidate', 'memory'])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_MEMORY_CONSOLIDATE</b> Memory consolidated")

                        elif entry.tag == 'HOLIDAY_SELF_CALIBRATE':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/cognitive_learn.py'),
                                             '--dotenv', args.dotenv, '--self-calibrate'])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_SELF_CALIBRATE</b> Self-model calibrated")

                        # === NIGHT: Simulation & Stress Test ===
                        elif entry.tag == 'HOLIDAY_MONTE_CARLO':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/monte_carlo_sim.py'),
                                             '--universe', args.universe, '--simulations', '10000'])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_MONTE_CARLO</b> Monte Carlo simulation complete")

                        elif entry.tag == 'HOLIDAY_STRESS_TEST':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/stress_test.py'),
                                             '--dotenv', args.dotenv])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_STRESS_TEST</b> Stress testing complete")

                        elif entry.tag == 'HOLIDAY_VAR_CALC':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/calculate_var.py'),
                                             '--dotenv', args.dotenv])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_VAR_CALC</b> VaR recalculated")

                        elif entry.tag == 'HOLIDAY_DRAWDOWN_ANALYSIS':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/drawdown_analysis.py'),
                                             '--dotenv', args.dotenv])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_DRAWDOWN_ANALYSIS</b> Drawdown analysis complete")

                        # === LATE NIGHT: Prep for Next Trading Day ===
                        elif entry.tag == 'HOLIDAY_NEXT_DAY_PREP':
                            if is_market_closed_day():
                                # Find next trading day and prep
                                cal = EquitiesCalendar()
                                next_day = datetime.now(ET).date() + timedelta(days=1)
                                while not cal.is_trading_day(next_day):
                                    next_day += timedelta(days=1)
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_NEXT_DAY_PREP</b> Next trading day: {next_day}")

                        elif entry.tag == 'HOLIDAY_WATCHLIST_BUILD':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/scan.py'),
                                             '--cap', str(args.cap), '--deterministic', '--top3',
                                             '--dotenv', args.dotenv, '--preview'])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_WATCHLIST_BUILD</b> Watchlist ready")

                        elif entry.tag == 'HOLIDAY_PREVIEW_SCAN':
                            if is_market_closed_day():
                                rc = do_refresh_top3(args.universe, args.dotenv, args.cap,
                                                     datetime.now(ET).strftime('%Y-%m-%d'))
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_PREVIEW_SCAN</b> Preview scan complete")

                        elif entry.tag == 'HOLIDAY_FINAL_BACKUP':
                            if is_market_closed_day():
                                rc = run_cmd([sys.executable, str(ROOT / 'scripts/backup_state.py')])
                                if send_fn:
                                    send_fn(f"<b>HOLIDAY_FINAL_BACKUP</b> Final backup complete")

                        elif entry.tag == 'HOLIDAY_COMPLETE':
                            if is_market_closed_day():
                                if send_fn:
                                    cal = EquitiesCalendar()
                                    next_day = datetime.now(ET).date() + timedelta(days=1)
                                    while not cal.is_trading_day(next_day):
                                        next_day += timedelta(days=1)
                                    send_fn(f"<b>HOLIDAY LEARNING COMPLETE</b> Robot ready for {next_day}!")

                        mark_ran(state, entry.tag, ymd)
                        save_state(state)
                        try:
                            append_journal('scheduler_job', {'tag': entry.tag, 'date': ymd})
                        except Exception:
                            pass
            except Exception as e:
                print(f'Error in schedule {entry.tag}: {e}')
                if send_fn:
                    try:
                        try:
                            from core.clock.tz_utils import fmt_ct
                            now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                            send_fn(f"<b>{entry.tag}</b> [{stamp}] failed: {e}")
                        except Exception:
                            send_fn(f"<b>{entry.tag}</b> failed: {e}")
                    except Exception:
                        pass
                try:
                    append_journal('scheduler_job_failed', {'tag': entry.tag, 'error': str(e)})
                except Exception:
                    pass

        # =================================================================
        # CONTINUOUS WORK - Never idle, always learning
        # =================================================================
        # Instead of sleeping, do continuous work when no scheduled task ran
        # This ensures the robot is ALWAYS doing something productive:
        # - Analyzing stocks from the 900 universe
        # - Hunting for patterns
        # - Running mini-backtests
        # - Researching alpha factors
        # - Checking data quality
        # =================================================================
        try:
            do_continuous_work(args.universe, args.dotenv, args.cap, send_fn)
        except Exception as e:
            print(f"Continuous work error: {e}")

        # Brief pause to prevent CPU hogging, but much shorter than before
        time.sleep(2)  # 2 seconds between continuous work cycles


if __name__ == '__main__':
    main()



