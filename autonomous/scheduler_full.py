#!/usr/bin/env python3
"""
KOBE FULL SCHEDULER - ALL 150+ TASKS WITH SPECIFIC TIMES
==========================================================
Every task has a specific time. Every task logs when it runs.
If you don't see it in the log, something is wrong.

VISIBILITY IS ACCOUNTABILITY.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ScheduledTask:
    """A task scheduled for a specific time."""
    time: str  # HH:MM format
    name: str
    description: str
    function: str  # Function name to call
    priority: int = 5  # 1=highest, 10=lowest
    enabled: bool = True
    weekdays_only: bool = False  # Only run Mon-Fri
    weekends_only: bool = False  # Only run Sat-Sun
    saturday_only: bool = False  # Only run Saturday
    sunday_only: bool = False  # Only run Sunday


# =============================================================================
# MASTER SCHEDULE - ALL TASKS WITH SPECIFIC TIMES
# =============================================================================
MASTER_SCHEDULE = [
    # =========================================================================
    # PRE-MARKET (4:00 AM - 9:30 AM ET)
    # =========================================================================

    # 4:00 AM - System startup
    ScheduledTask("04:00", "system_health_check", "Full system health check", "health_check", priority=1, weekdays_only=True),
    ScheduledTask("04:05", "data_integrity_check", "Verify all data files intact", "data_integrity", priority=1, weekdays_only=True),
    ScheduledTask("04:10", "broker_connection_test", "Test Alpaca connection", "broker_connect", priority=1, weekdays_only=True),

    # 5:00 AM - Data refresh
    ScheduledTask("05:00", "polygon_data_refresh", "Refresh Polygon EOD cache", "refresh_polygon", priority=2, weekdays_only=True),
    ScheduledTask("05:15", "universe_validation", "Validate 800-stock universe", "validate_universe", priority=2, weekdays_only=True),
    ScheduledTask("05:30", "indicator_precalc", "Pre-calculate indicators", "precalc_indicators", priority=2, weekdays_only=True),

    # 6:00 AM - Economic data
    ScheduledTask("06:00", "fred_vix_fetch", "Fetch VIX from FRED", "fetch_vix", priority=2, weekdays_only=True),
    ScheduledTask("06:05", "fred_treasury_fetch", "Fetch 10Y Treasury", "fetch_treasury", priority=3, weekdays_only=True),
    ScheduledTask("06:10", "fear_greed_fetch", "Fetch Fear & Greed Index", "fetch_fear_greed", priority=3, weekdays_only=True),
    ScheduledTask("06:15", "market_regime_detect", "Detect current market regime", "detect_regime", priority=2, weekdays_only=True),

    # 7:00 AM - ML models
    ScheduledTask("07:00", "lstm_confidence_run", "Run LSTM confidence model", "run_lstm", priority=3, weekdays_only=True),
    ScheduledTask("07:15", "hmm_regime_update", "Update HMM regime state", "update_hmm", priority=3, weekdays_only=True),
    ScheduledTask("07:30", "ensemble_weights_check", "Check ensemble model weights", "check_ensemble", priority=3, weekdays_only=True),

    # 8:00 AM - Pre-market validation
    ScheduledTask("08:00", "premarket_gap_check", "Check for overnight gaps", "check_gaps", priority=1, weekdays_only=True),
    ScheduledTask("08:15", "watchlist_validation", "Validate overnight watchlist", "validate_watchlist", priority=1, weekdays_only=True),
    ScheduledTask("08:30", "news_sentiment_scan", "Scan news for watchlist stocks", "scan_news", priority=2, weekdays_only=True),
    ScheduledTask("08:45", "pregame_blueprint_gen", "Generate Pre-Game Blueprint", "generate_pregame", priority=1, weekdays_only=True),

    # 9:00 AM - Final prep
    ScheduledTask("09:00", "final_preflight_check", "Final preflight before market", "preflight", priority=1, weekdays_only=True),
    ScheduledTask("09:15", "position_sizing_calc", "Calculate position sizes", "calc_position_sizes", priority=1, weekdays_only=True),
    ScheduledTask("09:25", "kill_zone_status", "Log kill zone status", "log_kill_zone", priority=1, weekdays_only=True),

    # =========================================================================
    # MARKET OPEN - OBSERVE ONLY (9:30 AM - 10:00 AM ET)
    # =========================================================================
    ScheduledTask("09:30", "market_open_log", "*** MARKET OPEN - OBSERVE ONLY ***", "log_market_open", priority=1, weekdays_only=True),
    ScheduledTask("09:30", "opening_price_capture", "Capture all opening prices", "capture_opens", priority=1, weekdays_only=True),
    ScheduledTask("09:31", "opening_gap_analysis", "Analyze opening gaps", "analyze_gaps", priority=1, weekdays_only=True),
    ScheduledTask("09:32", "volume_surge_detect", "Detect volume surges", "detect_volume", priority=2, weekdays_only=True),
    ScheduledTask("09:33", "watchlist_price_update", "Update watchlist prices", "update_watchlist_prices", priority=2, weekdays_only=True),
    ScheduledTask("09:35", "opening_range_record", "Record opening range", "record_opening", priority=1, weekdays_only=True),
    ScheduledTask("09:36", "momentum_direction", "Detect momentum direction", "detect_momentum", priority=2, weekdays_only=True),
    ScheduledTask("09:37", "sector_strength_check", "Check sector strength", "check_sectors", priority=2, weekdays_only=True),
    ScheduledTask("09:38", "spy_correlation", "Check SPY correlation", "check_spy", priority=2, weekdays_only=True),
    ScheduledTask("09:40", "opening_range_5min", "5-min opening range", "range_5min", priority=1, weekdays_only=True),
    ScheduledTask("09:42", "vix_intraday_check", "Check intraday VIX", "check_vix", priority=2, weekdays_only=True),
    ScheduledTask("09:45", "opening_range_update", "Update opening range", "update_opening", priority=1, weekdays_only=True),
    ScheduledTask("09:47", "false_breakout_detect", "Detect false breakouts", "detect_false_breakouts", priority=2, weekdays_only=True),
    ScheduledTask("09:50", "opening_range_15min", "15-min opening range", "range_15min", priority=1, weekdays_only=True),
    ScheduledTask("09:52", "institutional_flow", "Check institutional flow", "check_flow", priority=2, weekdays_only=True),
    ScheduledTask("09:55", "opening_range_final", "Finalize opening range", "finalize_opening", priority=1, weekdays_only=True),
    ScheduledTask("09:57", "morning_bias_determine", "Determine morning bias", "determine_bias", priority=1, weekdays_only=True),
    ScheduledTask("09:58", "final_observe_log", "Final observation before trading", "final_observe", priority=1, weekdays_only=True),

    # =========================================================================
    # MORNING SESSION (10:00 AM - 11:30 AM ET) - PRIMARY TRADING WINDOW
    # =========================================================================
    ScheduledTask("10:00", "morning_scan_start", "*** PRIMARY WINDOW OPEN - SCANNING ***", "log_primary_open", priority=1, weekdays_only=True),
    ScheduledTask("10:00", "full_universe_scan", "Scan 800 stocks for signals", "scan_universe", priority=1, weekdays_only=True),
    ScheduledTask("10:01", "signal_ranking", "Rank signals by confidence", "rank_signals", priority=1, weekdays_only=True),
    ScheduledTask("10:02", "historical_pattern_check", "Check historical patterns", "check_patterns", priority=1, weekdays_only=True),
    ScheduledTask("10:03", "consecutive_day_analysis", "Analyze consecutive day patterns", "analyze_consecutive", priority=2, weekdays_only=True),
    ScheduledTask("10:05", "watchlist_signal_check", "Check watchlist for triggers", "check_watchlist_signals", priority=1, weekdays_only=True),
    ScheduledTask("10:06", "support_resistance_check", "Check S/R levels", "check_sr", priority=2, weekdays_only=True),
    ScheduledTask("10:07", "expected_move_calc", "Calculate expected moves", "calc_expected_move", priority=2, weekdays_only=True),
    ScheduledTask("10:08", "news_impact_check", "Check news impact", "check_news", priority=2, weekdays_only=True),
    ScheduledTask("10:10", "signal_quality_gate", "Apply quality gate to signals", "quality_gate", priority=1, weekdays_only=True),
    ScheduledTask("10:11", "risk_reward_calc", "Calculate R:R ratios", "calc_rr", priority=1, weekdays_only=True),
    ScheduledTask("10:12", "position_size_calc", "Calculate position sizes", "calc_size", priority=1, weekdays_only=True),
    ScheduledTask("10:15", "trade_execution_check", "Execute qualified trades", "execute_trades", priority=1, weekdays_only=True),
    ScheduledTask("10:16", "order_confirmation", "Confirm order fills", "confirm_orders", priority=1, weekdays_only=True),
    ScheduledTask("10:17", "fill_price_log", "Log fill prices", "log_fills", priority=1, weekdays_only=True),
    ScheduledTask("10:20", "position_monitor_quick", "Quick position check", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("10:25", "pnl_update_1", "Update P&L", "update_pnl", priority=2, weekdays_only=True),
    ScheduledTask("10:30", "position_monitor_1", "Monitor open positions", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("10:30", "fallback_scan", "Fallback scan if watchlist empty", "fallback_scan", priority=2, weekdays_only=True),
    ScheduledTask("10:31", "fallback_quality_gate", "Quality gate for fallback signals", "fallback_gate", priority=2, weekdays_only=True),
    ScheduledTask("10:35", "stop_distance_check", "Check stop distances", "check_stop_distance", priority=2, weekdays_only=True),
    ScheduledTask("10:40", "pnl_update_2", "Update P&L", "update_pnl", priority=2, weekdays_only=True),
    ScheduledTask("10:45", "mid_morning_check", "Mid-morning position check", "position_check", priority=2, weekdays_only=True),
    ScheduledTask("10:46", "unrealized_pnl_log", "Log unrealized P&L", "log_unrealized", priority=2, weekdays_only=True),
    ScheduledTask("10:50", "position_monitor_2", "Position monitor 2", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("10:55", "pnl_update_3", "Update P&L", "update_pnl", priority=2, weekdays_only=True),
    ScheduledTask("11:00", "hourly_pnl_log", "Log hourly P&L", "log_pnl", priority=2, weekdays_only=True),
    ScheduledTask("11:00", "drift_detection", "Check for price drift", "check_drift", priority=2, weekdays_only=True),
    ScheduledTask("11:01", "cross_validation_prices", "Cross-validate prices", "cross_validate", priority=2, weekdays_only=True),
    ScheduledTask("11:05", "position_monitor_3", "Position monitor 3", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("11:10", "pnl_update_4", "Update P&L", "update_pnl", priority=2, weekdays_only=True),
    ScheduledTask("11:15", "stop_loss_check", "Check stop losses", "check_stops", priority=1, weekdays_only=True),
    ScheduledTask("11:16", "target_check", "Check profit targets", "check_targets", priority=1, weekdays_only=True),
    ScheduledTask("11:17", "time_stop_check", "Check time stops", "check_time_stops", priority=1, weekdays_only=True),
    ScheduledTask("11:20", "position_monitor_4", "Position monitor 4", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("11:25", "morning_window_close", "*** PRIMARY WINDOW CLOSING ***", "log_primary_close", priority=1, weekdays_only=True),
    ScheduledTask("11:26", "morning_summary", "Morning session summary", "morning_summary", priority=2, weekdays_only=True),
    ScheduledTask("11:27", "trades_executed_count", "Count trades executed", "count_trades", priority=2, weekdays_only=True),
    ScheduledTask("11:28", "morning_pnl_snapshot", "Morning P&L snapshot", "pnl_snapshot", priority=2, weekdays_only=True),

    # =========================================================================
    # LUNCH SESSION (11:30 AM - 2:00 PM ET) - RESEARCH TIME
    # =========================================================================
    ScheduledTask("11:30", "lunch_session_start", "*** LUNCH SESSION - NO NEW TRADES ***", "log_lunch_start", priority=1, weekdays_only=True),
    ScheduledTask("11:30", "lunch_position_monitor", "Monitor positions during lunch", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("11:35", "lunch_pnl_update", "Update P&L", "update_pnl", priority=2, weekdays_only=True),
    ScheduledTask("11:40", "lunch_stop_check", "Check stop losses", "check_stops", priority=2, weekdays_only=True),
    ScheduledTask("11:45", "lunch_position_monitor_2", "Position monitor", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("11:50", "cognitive_quick_check", "Quick cognitive check", "cognitive_check", priority=3, weekdays_only=True),
    ScheduledTask("11:55", "lunch_pnl_update_2", "Update P&L", "update_pnl", priority=2, weekdays_only=True),

    ScheduledTask("12:00", "midday_pnl_log", "Log midday P&L", "log_pnl", priority=2, weekdays_only=True),
    ScheduledTask("12:00", "research_experiment_1", "Run parameter experiment", "run_experiment", priority=4, weekdays_only=True),
    ScheduledTask("12:05", "position_monitor_noon", "Noon position check", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("12:10", "curiosity_scan", "Run curiosity engine scan", "curiosity_scan", priority=3, weekdays_only=True),
    ScheduledTask("12:15", "hypothesis_check", "Check existing hypotheses", "check_hypotheses", priority=3, weekdays_only=True),
    ScheduledTask("12:20", "noon_pnl_update", "Update P&L", "update_pnl", priority=2, weekdays_only=True),
    ScheduledTask("12:25", "noon_stop_check", "Check stop losses", "check_stops", priority=2, weekdays_only=True),
    ScheduledTask("12:30", "ict_pattern_discovery", "Discover ICT patterns", "discover_ict", priority=3, weekdays_only=True),
    ScheduledTask("12:35", "order_flow_analysis", "Analyze order flow", "analyze_flow", priority=3, weekdays_only=True),
    ScheduledTask("12:40", "sector_rotation_check", "Check sector rotation", "check_rotation", priority=3, weekdays_only=True),
    ScheduledTask("12:45", "position_monitor_1245", "Position monitor", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("12:50", "research_hypothesis_gen", "Generate hypotheses", "gen_hypotheses", priority=3, weekdays_only=True),
    ScheduledTask("12:55", "pnl_update_1255", "Update P&L", "update_pnl", priority=2, weekdays_only=True),

    ScheduledTask("13:00", "hourly_pnl_log_2", "Log hourly P&L", "log_pnl", priority=2, weekdays_only=True),
    ScheduledTask("13:00", "scrape_reddit", "Scrape Reddit for ideas", "scrape_reddit", priority=4, weekdays_only=True),
    ScheduledTask("13:05", "position_monitor_1305", "Position monitor", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("13:10", "ml_model_check", "Check ML model performance", "check_ml", priority=3, weekdays_only=True),
    ScheduledTask("13:15", "hmm_regime_check", "Check HMM regime", "check_hmm", priority=3, weekdays_only=True),
    ScheduledTask("13:20", "lstm_confidence_check", "Check LSTM confidence", "check_lstm", priority=3, weekdays_only=True),
    ScheduledTask("13:25", "position_monitor_1325", "Position monitor", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("13:30", "scrape_github", "Scrape GitHub for code", "scrape_github", priority=4, weekdays_only=True),
    ScheduledTask("13:35", "scrape_hackernews", "Scrape HackerNews", "scrape_hn", priority=4, weekdays_only=True),
    ScheduledTask("13:40", "scrape_arxiv_midday", "Scrape arXiv papers", "scrape_arxiv", priority=4, weekdays_only=True),
    ScheduledTask("13:45", "knowledge_integration", "Integrate scraped knowledge", "integrate_knowledge", priority=3, weekdays_only=True),
    ScheduledTask("13:50", "position_monitor_1350", "Position monitor", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("13:55", "pre_power_hour_prep", "Pre-power hour preparation", "pre_power", priority=2, weekdays_only=True),

    # =========================================================================
    # AFTERNOON SESSION (2:00 PM - 3:30 PM ET) - POWER HOUR
    # =========================================================================
    ScheduledTask("14:00", "power_hour_prep", "*** POWER HOUR PREP ***", "log_power_prep", priority=1, weekdays_only=True),
    ScheduledTask("14:00", "afternoon_scan", "Afternoon universe scan", "scan_universe", priority=2, weekdays_only=True),
    ScheduledTask("14:01", "afternoon_signal_rank", "Rank afternoon signals", "rank_signals", priority=2, weekdays_only=True),
    ScheduledTask("14:02", "afternoon_pattern_check", "Check afternoon patterns", "check_patterns", priority=2, weekdays_only=True),
    ScheduledTask("14:05", "position_monitor_1405", "Position monitor", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("14:10", "pnl_update_1410", "Update P&L", "update_pnl", priority=2, weekdays_only=True),
    ScheduledTask("14:15", "position_adjustment", "Adjust positions if needed", "adjust_positions", priority=2, weekdays_only=True),
    ScheduledTask("14:16", "stop_tightening_check", "Check if stops should tighten", "check_tighten", priority=2, weekdays_only=True),
    ScheduledTask("14:20", "position_monitor_1420", "Position monitor", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("14:25", "pnl_update_1425", "Update P&L", "update_pnl", priority=2, weekdays_only=True),
    ScheduledTask("14:28", "power_hour_countdown", "Power hour in 2 min", "log_countdown", priority=1, weekdays_only=True),

    ScheduledTask("14:30", "power_hour_start", "*** POWER HOUR START ***", "log_power_start", priority=1, weekdays_only=True),
    ScheduledTask("14:30", "power_hour_scan", "Power hour signal scan", "scan_universe", priority=1, weekdays_only=True),
    ScheduledTask("14:31", "power_hour_signal_rank", "Rank power hour signals", "rank_signals", priority=1, weekdays_only=True),
    ScheduledTask("14:32", "power_hour_quality_gate", "Power hour quality gate", "quality_gate", priority=1, weekdays_only=True),
    ScheduledTask("14:35", "position_monitor_1435", "Position monitor", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("14:40", "pnl_update_1440", "Update P&L", "update_pnl", priority=2, weekdays_only=True),
    ScheduledTask("14:45", "WATCHLIST_NEXT_DAY", "*** BUILD NEXT DAY WATCHLIST ***", "build_watchlist", priority=1, weekdays_only=True),
    ScheduledTask("14:46", "watchlist_top5_select", "Select Top 5 for watchlist", "select_top5", priority=1, weekdays_only=True),
    ScheduledTask("14:47", "watchlist_totd_select", "Select Trade of the Day", "select_totd", priority=1, weekdays_only=True),
    ScheduledTask("14:48", "watchlist_save", "Save watchlist to JSON", "save_watchlist", priority=1, weekdays_only=True),
    ScheduledTask("14:50", "position_monitor_1450", "Position monitor", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("14:55", "pnl_update_1455", "Update P&L", "update_pnl", priority=2, weekdays_only=True),

    ScheduledTask("15:00", "hourly_pnl_log_3", "Log hourly P&L", "log_pnl", priority=2, weekdays_only=True),
    ScheduledTask("15:00", "position_monitor_1500", "Monitor positions", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("15:01", "institutional_selling_check", "Check for institutional selling", "check_selling", priority=2, weekdays_only=True),
    ScheduledTask("15:05", "position_monitor_1505", "Position monitor", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("15:10", "pnl_update_1510", "Update P&L", "update_pnl", priority=2, weekdays_only=True),
    ScheduledTask("15:15", "exit_evaluation", "Evaluate exit conditions", "evaluate_exits", priority=1, weekdays_only=True),
    ScheduledTask("15:16", "partial_profit_check", "Check for partial profit taking", "check_partial", priority=2, weekdays_only=True),
    ScheduledTask("15:20", "position_monitor_1520", "Position monitor", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("15:25", "power_hour_close", "*** POWER HOUR CLOSING ***", "log_power_close", priority=1, weekdays_only=True),
    ScheduledTask("15:26", "power_hour_summary", "Power hour summary", "power_summary", priority=2, weekdays_only=True),
    ScheduledTask("15:27", "eod_prep_start", "Prepare for EOD", "eod_prep", priority=1, weekdays_only=True),

    # =========================================================================
    # MARKET CLOSE (3:30 PM - 4:00 PM ET)
    # =========================================================================
    ScheduledTask("15:30", "close_prep", "*** MARKET CLOSE PREP - NO NEW TRADES ***", "log_close_prep", priority=1, weekdays_only=True),
    ScheduledTask("15:30", "final_position_check", "Final position check", "final_position_check", priority=1, weekdays_only=True),
    ScheduledTask("15:31", "close_stop_check", "Final stop check", "check_stops", priority=1, weekdays_only=True),
    ScheduledTask("15:32", "close_target_check", "Final target check", "check_targets", priority=1, weekdays_only=True),
    ScheduledTask("15:35", "position_monitor_1535", "Position monitor", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("15:38", "late_day_selling", "Check late day selling", "check_selling", priority=2, weekdays_only=True),
    ScheduledTask("15:40", "position_monitor_1540", "Position monitor", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("15:42", "moc_imbalance_check", "Check MOC imbalances", "check_moc", priority=2, weekdays_only=True),
    ScheduledTask("15:45", "eod_exit_check", "Check for EOD exits", "check_eod_exits", priority=1, weekdays_only=True),
    ScheduledTask("15:46", "eod_exit_execute", "Execute EOD exits", "execute_eod_exits", priority=1, weekdays_only=True),
    ScheduledTask("15:48", "position_monitor_1548", "Position monitor", "monitor_positions", priority=2, weekdays_only=True),
    ScheduledTask("15:50", "final_pnl_update", "Final P&L update", "update_pnl", priority=1, weekdays_only=True),
    ScheduledTask("15:52", "overnight_risk_check", "Check overnight risk", "check_overnight_risk", priority=1, weekdays_only=True),
    ScheduledTask("15:55", "pre_close_log", "Log pre-close status", "log_pre_close", priority=1, weekdays_only=True),
    ScheduledTask("15:57", "final_position_snapshot", "Final position snapshot", "snapshot_positions", priority=1, weekdays_only=True),
    ScheduledTask("15:58", "market_close_countdown", "Market closes in 2 min", "log_countdown", priority=1, weekdays_only=True),
    ScheduledTask("15:59", "last_second_check", "Last second check", "last_check", priority=1, weekdays_only=True),
    ScheduledTask("16:00", "market_close_log", "*** MARKET CLOSED ***", "log_market_close", priority=1, weekdays_only=True),

    # =========================================================================
    # POST-MARKET (4:00 PM - 8:00 PM ET) - LEARNING
    # =========================================================================
    ScheduledTask("16:00", "daily_pnl_calc", "Calculate daily P&L", "calc_daily_pnl", priority=1, weekdays_only=True),
    ScheduledTask("16:01", "realized_pnl_log", "Log realized P&L", "log_realized", priority=1, weekdays_only=True),
    ScheduledTask("16:02", "unrealized_pnl_log", "Log unrealized P&L", "log_unrealized", priority=1, weekdays_only=True),
    ScheduledTask("16:03", "win_loss_calc", "Calculate win/loss ratio", "calc_win_loss", priority=1, weekdays_only=True),
    ScheduledTask("16:05", "trade_log_update", "Update trade log", "update_trade_log", priority=1, weekdays_only=True),
    ScheduledTask("16:06", "trade_count_log", "Log trade count", "log_trade_count", priority=2, weekdays_only=True),
    ScheduledTask("16:07", "avg_win_loss_calc", "Calculate avg win/loss", "calc_avg", priority=2, weekdays_only=True),
    ScheduledTask("16:10", "broker_reconcile", "Reconcile with broker", "reconcile_broker", priority=1, weekdays_only=True),
    ScheduledTask("16:11", "position_reconcile", "Reconcile positions", "reconcile_positions", priority=1, weekdays_only=True),
    ScheduledTask("16:12", "cash_reconcile", "Reconcile cash", "reconcile_cash", priority=1, weekdays_only=True),
    ScheduledTask("16:15", "data_validation_eod", "EOD data validation", "validate_eod", priority=2, weekdays_only=True),
    ScheduledTask("16:20", "polygon_data_check", "Check Polygon data", "check_polygon", priority=2, weekdays_only=True),
    ScheduledTask("16:25", "cross_validate_eod", "Cross-validate EOD prices", "cross_validate", priority=2, weekdays_only=True),

    ScheduledTask("16:30", "trade_analysis", "Analyze today's trades", "analyze_trades", priority=2, weekdays_only=True),
    ScheduledTask("16:31", "entry_analysis", "Analyze entries", "analyze_entries", priority=2, weekdays_only=True),
    ScheduledTask("16:32", "exit_analysis", "Analyze exits", "analyze_exits", priority=2, weekdays_only=True),
    ScheduledTask("16:35", "slippage_analysis", "Analyze slippage", "analyze_slippage", priority=2, weekdays_only=True),
    ScheduledTask("16:40", "execution_quality", "Check execution quality", "check_execution", priority=2, weekdays_only=True),
    ScheduledTask("16:45", "lesson_extraction", "Extract lessons from trades", "extract_lessons", priority=2, weekdays_only=True),
    ScheduledTask("16:46", "mistake_detection", "Detect mistakes", "detect_mistakes", priority=2, weekdays_only=True),
    ScheduledTask("16:47", "improvement_ideas", "Generate improvement ideas", "gen_improvements", priority=2, weekdays_only=True),
    ScheduledTask("16:50", "rule_check", "Check trading rules followed", "check_rules", priority=2, weekdays_only=True),
    ScheduledTask("16:55", "self_assessment", "Run self-assessment", "self_assess", priority=2, weekdays_only=True),

    ScheduledTask("17:00", "cognitive_reflection", "Run cognitive reflection", "run_reflection", priority=2, weekdays_only=True),
    ScheduledTask("17:05", "what_went_well", "Identify what went well", "went_well", priority=2, weekdays_only=True),
    ScheduledTask("17:10", "what_went_wrong", "Identify what went wrong", "went_wrong", priority=2, weekdays_only=True),
    ScheduledTask("17:15", "episodic_memory_update", "Update episodic memory", "update_episodic", priority=3, weekdays_only=True),
    ScheduledTask("17:20", "experience_store", "Store experience in memory", "store_experience", priority=3, weekdays_only=True),
    ScheduledTask("17:25", "pattern_recognition_update", "Update pattern recognition", "update_patterns", priority=3, weekdays_only=True),
    ScheduledTask("17:30", "semantic_memory_update", "Update semantic memory", "update_semantic", priority=3, weekdays_only=True),
    ScheduledTask("17:35", "rule_update", "Update trading rules", "update_rules", priority=3, weekdays_only=True),
    ScheduledTask("17:40", "knowledge_consolidate", "Consolidate knowledge", "consolidate", priority=3, weekdays_only=True),
    ScheduledTask("17:45", "curiosity_update", "Update curiosity engine", "update_curiosity", priority=3, weekdays_only=True),
    ScheduledTask("17:50", "hypothesis_update", "Update hypotheses", "update_hypotheses", priority=3, weekdays_only=True),
    ScheduledTask("17:55", "edge_discovery_check", "Check for edge discovery", "check_edges", priority=3, weekdays_only=True),

    ScheduledTask("18:00", "daily_report_gen", "Generate daily report", "generate_report", priority=2, weekdays_only=True),
    ScheduledTask("18:05", "performance_metrics", "Calculate performance metrics", "calc_metrics", priority=2, weekdays_only=True),
    ScheduledTask("18:10", "sharpe_update", "Update Sharpe ratio", "update_sharpe", priority=2, weekdays_only=True),
    ScheduledTask("18:15", "profit_factor_update", "Update profit factor", "update_pf", priority=2, weekdays_only=True),
    ScheduledTask("18:20", "win_rate_update", "Update win rate", "update_wr", priority=2, weekdays_only=True),
    ScheduledTask("18:25", "drawdown_update", "Update drawdown", "update_dd", priority=2, weekdays_only=True),
    ScheduledTask("18:30", "watchlist_finalize", "Finalize next day watchlist", "finalize_watchlist", priority=1, weekdays_only=True),
    ScheduledTask("18:31", "pregame_prep", "Start pre-game prep", "start_pregame", priority=1, weekdays_only=True),
    ScheduledTask("18:35", "news_scan_watchlist", "Scan news for watchlist", "scan_news", priority=2, weekdays_only=True),
    ScheduledTask("18:40", "earnings_check", "Check earnings dates", "check_earnings", priority=2, weekdays_only=True),
    ScheduledTask("18:45", "catalyst_check", "Check for catalysts", "check_catalysts", priority=2, weekdays_only=True),
    ScheduledTask("18:50", "sector_analysis", "Analyze sector strength", "analyze_sectors", priority=2, weekdays_only=True),
    ScheduledTask("18:55", "market_breadth", "Check market breadth", "check_breadth", priority=2, weekdays_only=True),

    ScheduledTask("19:00", "scrape_arxiv", "Scrape arXiv for papers", "scrape_arxiv", priority=4, weekdays_only=True),
    ScheduledTask("19:05", "scrape_ssrn", "Scrape SSRN papers", "scrape_ssrn", priority=4, weekdays_only=True),
    ScheduledTask("19:10", "scrape_semanticscholar", "Scrape Semantic Scholar", "scrape_ss", priority=4, weekdays_only=True),
    ScheduledTask("19:15", "research_digest", "Create research digest", "research_digest", priority=3, weekdays_only=True),
    ScheduledTask("19:20", "quant_forum_scan", "Scan quant forums", "scan_forums", priority=4, weekdays_only=True),
    ScheduledTask("19:25", "twitter_scan", "Scan Twitter/X for alpha", "scan_twitter", priority=4, weekdays_only=True),
    ScheduledTask("19:30", "scrape_stackoverflow", "Scrape StackOverflow", "scrape_stackoverflow", priority=4, weekdays_only=True),
    ScheduledTask("19:35", "scrape_reddit_wsb", "Scrape Reddit WSB", "scrape_wsb", priority=4, weekdays_only=True),
    ScheduledTask("19:40", "scrape_reddit_algotrading", "Scrape Reddit algotrading", "scrape_algo", priority=4, weekdays_only=True),
    ScheduledTask("19:45", "knowledge_integrate", "Integrate all scraped knowledge", "integrate_all", priority=3, weekdays_only=True),
    ScheduledTask("19:50", "discovery_summary", "Summary of discoveries", "discovery_summary", priority=3, weekdays_only=True),
    ScheduledTask("19:55", "research_state_save", "Save research state", "save_research", priority=2, weekdays_only=True),

    # =========================================================================
    # OVERNIGHT (8:00 PM - 4:00 AM ET) - OPTIMIZATION
    # =========================================================================
    ScheduledTask("20:00", "overnight_start", "*** OVERNIGHT SESSION START ***", "log_overnight_start", priority=1, weekdays_only=True),
    ScheduledTask("20:00", "full_data_validation", "Full data validation", "full_validation", priority=2, weekdays_only=True),

    ScheduledTask("21:00", "ml_model_retrain_check", "Check if ML models need retrain", "check_retrain", priority=3, weekdays_only=True),
    # FIX (2026-01-07): Fix 3 - Actually trigger retraining if drift detected
    ScheduledTask("21:15", "nightly_ml_retrain", "Retrain ML models if drift detected", "full_retrain", priority=3, weekdays_only=True),
    ScheduledTask("21:30", "walk_forward_mini", "Mini walk-forward test", "mini_wf", priority=3, weekdays_only=True),

    ScheduledTask("22:00", "research_experiment_2", "Run overnight experiment", "run_experiment", priority=4, weekdays_only=True),
    ScheduledTask("22:30", "curiosity_engine", "Run curiosity engine", "run_curiosity", priority=3, weekdays_only=True),

    ScheduledTask("23:00", "hypothesis_generation", "Generate new hypotheses", "generate_hypotheses", priority=3, weekdays_only=True),
    ScheduledTask("23:30", "knowledge_consolidation", "Consolidate knowledge base", "consolidate_knowledge", priority=3, weekdays_only=True),

    ScheduledTask("00:00", "midnight_health_check", "Midnight system health", "health_check", priority=2, weekdays_only=True),
    ScheduledTask("01:00", "deep_backtest", "Run deep backtest", "deep_backtest", priority=4, weekdays_only=True),
    ScheduledTask("02:00", "parameter_optimization", "Parameter optimization run", "optimize_params", priority=4, weekdays_only=True),
    ScheduledTask("03:00", "data_cleanup", "Clean up old cache/logs", "cleanup", priority=3, weekdays_only=True),
    ScheduledTask("03:30", "system_backup", "Backup system state", "backup_state", priority=2, weekdays_only=True),

    # =========================================================================
    # SATURDAY - WATCHLIST BY 9:30 AM ET (8:30 AM Central)
    # Reports first, then full watchlist with all components ready early
    # =========================================================================

    # --- 06:00 AM ET - Saturday System Startup ---
    ScheduledTask("06:00", "sat_start", "*** SATURDAY - REPORTS & WATCHLIST DAY ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("06:02", "sat_health_check", "Full system health check", "full_health", priority=1, saturday_only=True),
    ScheduledTask("06:04", "sat_data_integrity", "Check all data files", "data_integrity", priority=1, saturday_only=True),
    ScheduledTask("06:06", "sat_broker_check", "Verify broker connection", "broker_connect", priority=1, saturday_only=True),
    ScheduledTask("06:08", "sat_cache_check", "Check Polygon cache health", "check_polygon", priority=2, saturday_only=True),
    ScheduledTask("06:10", "sat_universe_validate", "Validate 800-stock universe", "validate_universe", priority=2, saturday_only=True),
    ScheduledTask("06:12", "sat_data_freshness", "Check data freshness", "full_validation", priority=2, saturday_only=True),

    # --- 06:15 AM - Quick Weekly P&L Summary ---
    ScheduledTask("06:15", "sat_pnl_start", "*** WEEKLY P&L REPORTS ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("06:17", "sat_weekly_pnl", "Calculate weekly P&L", "calc_weekly_perf", priority=1, saturday_only=True),
    ScheduledTask("06:19", "sat_realized_pnl", "Calculate realized P&L", "log_realized", priority=1, saturday_only=True),
    ScheduledTask("06:21", "sat_unrealized_pnl", "Calculate unrealized P&L", "log_unrealized", priority=1, saturday_only=True),
    ScheduledTask("06:23", "sat_total_pnl", "Calculate total P&L", "calc_daily_pnl", priority=1, saturday_only=True),
    ScheduledTask("06:25", "sat_pnl_by_strategy", "P&L by strategy breakdown", "compare_strategies", priority=1, saturday_only=True),
    ScheduledTask("06:27", "sat_pnl_by_symbol", "P&L by symbol breakdown", "analyze_trades", priority=1, saturday_only=True),
    ScheduledTask("06:29", "sat_pnl_report_gen", "Generate P&L report", "gen_weekly_report", priority=1, saturday_only=True),

    # --- 06:30 AM - Quick Performance Metrics ---
    ScheduledTask("06:30", "sat_metrics_start", "*** PERFORMANCE METRICS ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("06:32", "sat_win_rate", "Calculate weekly win rate", "calc_win_loss", priority=1, saturday_only=True),
    ScheduledTask("06:34", "sat_profit_factor", "Calculate profit factor", "update_pf", priority=1, saturday_only=True),
    ScheduledTask("06:36", "sat_sharpe_ratio", "Calculate Sharpe ratio", "update_sharpe", priority=1, saturday_only=True),
    ScheduledTask("06:38", "sat_sortino_ratio", "Calculate Sortino ratio", "update_sharpe", priority=1, saturday_only=True),
    ScheduledTask("06:40", "sat_max_drawdown", "Calculate max drawdown", "update_dd", priority=1, saturday_only=True),
    ScheduledTask("06:42", "sat_expectancy", "Calculate expectancy", "calc_metrics", priority=1, saturday_only=True),
    ScheduledTask("06:44", "sat_metrics_report", "Generate metrics report", "gen_weekly_report", priority=1, saturday_only=True),

    # --- 06:45 AM - Quick Trade Review ---
    ScheduledTask("06:45", "sat_trade_review_start", "*** QUICK TRADE REVIEW ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("06:47", "sat_all_trades_review", "Review all trades this week", "analyze_trades", priority=1, saturday_only=True),
    ScheduledTask("06:49", "sat_winning_trades", "Analyze winning trades", "analyze_trades", priority=1, saturday_only=True),
    ScheduledTask("06:51", "sat_losing_trades", "Analyze losing trades", "analyze_trades", priority=1, saturday_only=True),
    ScheduledTask("06:53", "sat_lessons_quick", "Quick lessons learned", "extract_lessons", priority=1, saturday_only=True),
    ScheduledTask("06:55", "sat_reports_save", "Save all reports", "save_research", priority=1, saturday_only=True),

    # ===========================================================================
    # 07:00 AM - BUILD MONDAY WATCHLIST (Scan 800 stocks)
    # ===========================================================================
    ScheduledTask("07:00", "sat_watchlist_start", "*** BUILD MONDAY WATCHLIST - SCAN 800 ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("07:02", "sat_scan_universe", "Scan 800 stocks for setups", "scan_universe", priority=1, saturday_only=True),
    ScheduledTask("07:05", "sat_scan_ibs_rsi", "Find IBS+RSI patterns", "scan_universe", priority=1, saturday_only=True),
    ScheduledTask("07:08", "sat_scan_turtle", "Find Turtle Soup patterns", "scan_universe", priority=1, saturday_only=True),
    ScheduledTask("07:11", "sat_scan_dual", "Find Dual Strategy patterns", "scan_universe", priority=1, saturday_only=True),
    ScheduledTask("07:14", "sat_rank_all_signals", "Rank all signals by confidence", "rank_signals", priority=1, saturday_only=True),
    ScheduledTask("07:17", "sat_quality_gate", "Apply quality gate (70+ score)", "quality_gate", priority=1, saturday_only=True),
    ScheduledTask("07:20", "sat_select_top10", "Select top 10 candidates", "select_top5", priority=1, saturday_only=True),
    ScheduledTask("07:23", "sat_check_earnings", "Filter out earnings stocks", "check_earnings", priority=1, saturday_only=True),
    ScheduledTask("07:26", "sat_check_news", "Check news for candidates", "scan_news", priority=1, saturday_only=True),
    ScheduledTask("07:29", "sat_check_catalysts", "Check catalysts (FOMC, etc)", "check_catalysts", priority=1, saturday_only=True),

    # --- 07:30 AM - Select Top 5 + TOTD ---
    ScheduledTask("07:30", "sat_top5_start", "*** SELECT TOP 5 + TOTD ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("07:32", "sat_select_top5", "Select top 5 watchlist", "select_top5", priority=1, saturday_only=True),
    ScheduledTask("07:34", "sat_select_totd", "Select Trade of the Day", "select_totd", priority=1, saturday_only=True),
    ScheduledTask("07:36", "sat_totd_validate", "Validate TOTD setup", "validate_watchlist", priority=1, saturday_only=True),
    ScheduledTask("07:38", "sat_top5_validate", "Validate Top 5 setups", "validate_watchlist", priority=1, saturday_only=True),
    ScheduledTask("07:40", "sat_save_top5", "Save Top 5 watchlist", "finalize_watchlist", priority=1, saturday_only=True),

    # ===========================================================================
    # 07:45 AM - HISTORICAL PATTERN ANALYSIS (All 15 Pre-Game Components)
    # ===========================================================================
    ScheduledTask("07:45", "sat_historical_start", "*** HISTORICAL PATTERN ANALYSIS ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("07:47", "sat_hist_totd", "TOTD: Historical patterns (consecutive days)", "analyze_historical", priority=1, saturday_only=True),
    ScheduledTask("07:49", "sat_hist_totd_samples", "TOTD: Sample size & win rate", "analyze_historical", priority=1, saturday_only=True),
    ScheduledTask("07:51", "sat_hist_totd_bounce", "TOTD: Bounce stats (avg, min, max)", "analyze_historical", priority=1, saturday_only=True),
    ScheduledTask("07:53", "sat_hist_top2", "Top 2: Historical patterns", "analyze_historical", priority=1, saturday_only=True),
    ScheduledTask("07:55", "sat_hist_top3", "Top 3: Historical patterns", "analyze_historical", priority=1, saturday_only=True),
    ScheduledTask("07:57", "sat_hist_top4", "Top 4: Historical patterns", "analyze_historical", priority=1, saturday_only=True),
    ScheduledTask("07:59", "sat_hist_top5", "Top 5: Historical patterns", "analyze_historical", priority=1, saturday_only=True),

    # --- 08:00 AM - EXPECTED MOVE & VOLATILITY ---
    ScheduledTask("08:00", "sat_em_start", "*** EXPECTED MOVE ANALYSIS ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("08:02", "sat_em_totd", "TOTD: Weekly expected move", "calc_expected_move", priority=1, saturday_only=True),
    ScheduledTask("08:04", "sat_em_totd_vol", "TOTD: 20-day realized volatility", "calc_expected_move", priority=1, saturday_only=True),
    ScheduledTask("08:06", "sat_em_totd_room", "TOTD: Room up/down analysis", "calc_expected_move", priority=1, saturday_only=True),
    ScheduledTask("08:08", "sat_em_top2", "Top 2: Expected move", "calc_expected_move", priority=1, saturday_only=True),
    ScheduledTask("08:10", "sat_em_top3", "Top 3: Expected move", "calc_expected_move", priority=1, saturday_only=True),
    ScheduledTask("08:12", "sat_em_top4", "Top 4: Expected move", "calc_expected_move", priority=1, saturday_only=True),
    ScheduledTask("08:14", "sat_em_top5", "Top 5: Expected move", "calc_expected_move", priority=1, saturday_only=True),

    # --- 08:15 AM - SUPPORT & RESISTANCE LEVELS ---
    ScheduledTask("08:15", "sat_sr_start", "*** SUPPORT/RESISTANCE LEVELS ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("08:17", "sat_sr_totd", "TOTD: Key S/R levels", "check_sr", priority=1, saturday_only=True),
    ScheduledTask("08:19", "sat_sr_totd_pivots", "TOTD: Pivot points", "check_sr", priority=1, saturday_only=True),
    ScheduledTask("08:21", "sat_sr_top2", "Top 2: S/R levels", "check_sr", priority=1, saturday_only=True),
    ScheduledTask("08:23", "sat_sr_top3", "Top 3: S/R levels", "check_sr", priority=1, saturday_only=True),
    ScheduledTask("08:25", "sat_sr_top5", "Top 4-5: S/R levels", "check_sr", priority=1, saturday_only=True),

    # --- 08:27 AM - NEWS & CATALYSTS ---
    ScheduledTask("08:27", "sat_news_start", "*** NEWS & CATALYSTS (Last 7 Days) ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("08:29", "sat_news_totd", "TOTD: Recent headlines", "scan_news", priority=1, saturday_only=True),
    ScheduledTask("08:31", "sat_news_totd_sentiment", "TOTD: Sentiment scores", "scan_news", priority=1, saturday_only=True),
    ScheduledTask("08:33", "sat_news_top2", "Top 2: Headlines & sentiment", "scan_news", priority=1, saturday_only=True),
    ScheduledTask("08:35", "sat_news_top5", "Top 3-5: Headlines", "scan_news", priority=1, saturday_only=True),

    # --- 08:37 AM - CONGRESSIONAL & INSIDER ACTIVITY ---
    ScheduledTask("08:37", "sat_political_start", "*** POLITICAL & INSIDER ACTIVITY ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("08:39", "sat_congress_totd", "TOTD: Congressional trades (90 days)", "check_political", priority=1, saturday_only=True),
    ScheduledTask("08:41", "sat_insider_totd", "TOTD: Insider activity (30 days)", "check_insider", priority=1, saturday_only=True),
    ScheduledTask("08:43", "sat_political_top5", "Top 2-5: Political/insider", "check_political", priority=1, saturday_only=True),

    # --- 08:45 AM - SECTOR & MARKET CONTEXT ---
    ScheduledTask("08:45", "sat_sector_start", "*** SECTOR & MARKET CONTEXT ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("08:47", "sat_sector_totd", "TOTD: Sector ETF mapping", "analyze_sectors", priority=1, saturday_only=True),
    ScheduledTask("08:49", "sat_sector_totd_rs", "TOTD: Relative strength vs sector", "analyze_sectors", priority=1, saturday_only=True),
    ScheduledTask("08:51", "sat_sector_totd_beta", "TOTD: Beta calculation", "analyze_sectors", priority=1, saturday_only=True),
    ScheduledTask("08:53", "sat_sector_top5", "Top 2-5: Sector context", "analyze_sectors", priority=1, saturday_only=True),
    ScheduledTask("08:55", "sat_market_regime", "Market regime (SPY vs SMA50, VIX)", "detect_regime", priority=1, saturday_only=True),

    # --- 08:57 AM - VOLUME ANALYSIS ---
    ScheduledTask("08:57", "sat_volume_start", "*** VOLUME ANALYSIS ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("08:59", "sat_vol_totd", "TOTD: Volume analysis (20/50-day avg)", "check_volume", priority=1, saturday_only=True),
    ScheduledTask("09:01", "sat_vol_top5", "Top 2-5: Volume analysis", "check_volume", priority=1, saturday_only=True),

    # ===========================================================================
    # 09:03 AM - ENTRY/STOP/TARGET WITH FULL JUSTIFICATION
    # ===========================================================================
    ScheduledTask("09:03", "sat_levels_start", "*** ENTRY/STOP/TARGET LEVELS ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("09:05", "sat_entry_totd", "TOTD: Entry price + reasoning", "calc_entry", priority=1, saturday_only=True),
    ScheduledTask("09:07", "sat_stop_totd", "TOTD: Stop loss + technical basis", "calc_stop", priority=1, saturday_only=True),
    ScheduledTask("09:09", "sat_target_totd", "TOTD: Target(s) + historical backing", "calc_target", priority=1, saturday_only=True),
    ScheduledTask("09:11", "sat_rr_totd", "TOTD: R:R ratio (all scenarios)", "calc_rr", priority=1, saturday_only=True),
    ScheduledTask("09:13", "sat_levels_top2", "Top 2: Entry/Stop/Target/R:R", "calc_entry", priority=1, saturday_only=True),
    ScheduledTask("09:15", "sat_levels_top5", "Top 3-5: Entry/Stop/Target/R:R", "calc_entry", priority=1, saturday_only=True),

    # --- 09:17 AM - BULL/BEAR CASES & RISKS ---
    ScheduledTask("09:17", "sat_thesis_start", "*** BULL/BEAR CASES & RISKS ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("09:19", "sat_bull_totd", "TOTD: Bull case (5 reasons)", "gen_bull_case", priority=1, saturday_only=True),
    ScheduledTask("09:21", "sat_bear_totd", "TOTD: Bear case (5 reasons)", "gen_bear_case", priority=1, saturday_only=True),
    ScheduledTask("09:23", "sat_risk_totd", "TOTD: What could go wrong (5 risks)", "gen_risks", priority=1, saturday_only=True),
    ScheduledTask("09:25", "sat_thesis_top5", "Top 2-5: Bull/Bear/Risks", "gen_bull_case", priority=1, saturday_only=True),

    # ===========================================================================
    # 09:27 AM - FINALIZE & SAVE (All Components Ready by 9:30 AM ET / 8:30 AM CT)
    # ===========================================================================
    ScheduledTask("09:27", "sat_finalize_start", "*** FINALIZE WATCHLIST (ALL 15 COMPONENTS) ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("09:28", "sat_pregame_totd", "Generate FULL Pre-Game Blueprint TOTD", "generate_pregame", priority=1, saturday_only=True),
    ScheduledTask("09:29", "sat_pregame_top2", "Generate FULL Pre-Game Blueprint Top 2", "generate_pregame", priority=1, saturday_only=True),
    ScheduledTask("09:30", "sat_pregame_save", "*** SAVE ALL - WATCHLIST READY 9:30 AM ET ***", "finalize_watchlist", priority=1, saturday_only=True),
    ScheduledTask("09:31", "sat_pregame_top3", "Generate Pre-Game Blueprint Top 3", "generate_pregame", priority=1, saturday_only=True),
    ScheduledTask("09:32", "sat_pregame_top4", "Generate Pre-Game Blueprint Top 4", "generate_pregame", priority=1, saturday_only=True),
    ScheduledTask("09:33", "sat_pregame_top5", "Generate Pre-Game Blueprint Top 5", "generate_pregame", priority=1, saturday_only=True),
    ScheduledTask("09:35", "sat_position_sizes", "Calculate position sizes (dual-cap)", "calc_position_sizes", priority=1, saturday_only=True),
    ScheduledTask("09:37", "sat_watchlist_backup", "Backup watchlist", "backup_state", priority=1, saturday_only=True),
    ScheduledTask("09:40", "sat_monday_ready", "*** MONDAY WATCHLIST COMPLETE ***", "log_weekend_start", priority=1, saturday_only=True),

    # ===========================================================================
    # REST OF SATURDAY - Deep Analysis & Review (After watchlist is ready)
    # ===========================================================================

    # --- 10:00 AM - Detailed Trade Analysis ---
    ScheduledTask("10:00", "sat_trade_analysis_start", "*** DETAILED TRADE ANALYSIS ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("10:05", "sat_entry_quality", "Analyze entry quality", "analyze_entries", priority=1, saturday_only=True),
    ScheduledTask("10:15", "sat_exit_quality", "Analyze exit quality", "analyze_exits", priority=1, saturday_only=True),
    ScheduledTask("10:25", "sat_slippage_analysis", "Analyze slippage", "analyze_slippage", priority=1, saturday_only=True),
    ScheduledTask("10:35", "sat_execution_quality", "Analyze execution quality", "check_execution", priority=1, saturday_only=True),
    ScheduledTask("10:45", "sat_timing_analysis", "Analyze trade timing", "analyze_trades", priority=2, saturday_only=True),
    ScheduledTask("10:55", "sat_trade_report", "Generate detailed trade report", "gen_weekly_report", priority=1, saturday_only=True),

    # --- 11:00 AM - Mistake Analysis & Lessons ---
    ScheduledTask("11:00", "sat_lessons_start", "*** LESSONS LEARNED ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("11:10", "sat_mistake_review", "Review all mistakes", "detect_mistakes", priority=1, saturday_only=True),
    ScheduledTask("11:20", "sat_rule_violations", "Check rule violations", "check_rules", priority=1, saturday_only=True),
    ScheduledTask("11:30", "sat_emotional_trades", "Identify emotional trades", "detect_mistakes", priority=1, saturday_only=True),
    ScheduledTask("11:40", "sat_missed_exits", "Identify missed exits", "analyze_exits", priority=1, saturday_only=True),
    ScheduledTask("11:50", "sat_lessons_extract", "Extract lessons learned", "extract_lessons", priority=1, saturday_only=True),

    # --- 12:00 PM - Strategy Performance ---
    ScheduledTask("12:00", "sat_strategy_start", "*** STRATEGY COMPARISON ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("12:10", "sat_ibs_rsi_perf", "IBS+RSI performance", "compare_strategies", priority=1, saturday_only=True),
    ScheduledTask("12:20", "sat_turtle_soup_perf", "Turtle Soup performance", "compare_strategies", priority=1, saturday_only=True),
    ScheduledTask("12:30", "sat_dual_strategy_perf", "Dual Strategy performance", "compare_strategies", priority=1, saturday_only=True),
    ScheduledTask("12:40", "sat_strategy_ranking", "Rank strategies by performance", "rank_signals", priority=1, saturday_only=True),
    ScheduledTask("12:50", "sat_strategy_report", "Generate strategy report", "gen_weekly_report", priority=1, saturday_only=True),

    # --- 13:00 PM - Market Analysis ---
    ScheduledTask("13:00", "sat_market_start", "*** MARKET ANALYSIS ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("13:10", "sat_spy_analysis", "SPY weekly analysis", "analyze_trades", priority=1, saturday_only=True),
    ScheduledTask("13:20", "sat_sector_performance", "Sector performance review", "analyze_sectors", priority=1, saturday_only=True),
    ScheduledTask("13:30", "sat_vix_analysis", "VIX analysis", "fetch_vix", priority=1, saturday_only=True),
    ScheduledTask("13:40", "sat_breadth_analysis", "Market breadth analysis", "check_breadth", priority=1, saturday_only=True),
    ScheduledTask("13:50", "sat_market_report", "Generate market report", "gen_weekly_report", priority=1, saturday_only=True),

    # --- 14:00 PM - Next Week Calendar ---
    ScheduledTask("14:00", "sat_calendar_start", "*** NEXT WEEK CALENDAR ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("14:10", "sat_earnings_calendar", "Check earnings calendar", "check_earnings", priority=1, saturday_only=True),
    ScheduledTask("14:20", "sat_fomc_check", "Check FOMC schedule", "check_catalysts", priority=1, saturday_only=True),
    ScheduledTask("14:30", "sat_economic_data", "Check economic data releases", "check_catalysts", priority=1, saturday_only=True),
    ScheduledTask("14:40", "sat_options_expiry", "Check options expiry", "check_catalysts", priority=1, saturday_only=True),
    ScheduledTask("14:50", "sat_calendar_save", "Save calendar events", "save_research", priority=1, saturday_only=True),

    # --- 15:00 PM - Weekly Summary Report ---
    ScheduledTask("15:00", "sat_summary_start", "*** WEEKLY SUMMARY REPORT ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("15:10", "sat_compile_reports", "Compile all reports", "gen_weekly_report", priority=1, saturday_only=True),
    ScheduledTask("15:20", "sat_executive_summary", "Generate executive summary", "gen_weekly_report", priority=1, saturday_only=True),
    ScheduledTask("15:30", "sat_highlights", "Identify key highlights", "went_well", priority=1, saturday_only=True),
    ScheduledTask("15:40", "sat_concerns", "Identify concerns", "went_wrong", priority=1, saturday_only=True),
    ScheduledTask("15:50", "sat_summary_save", "Save weekly summary", "save_research", priority=1, saturday_only=True),

    # --- 16:00 PM - Cognitive Reflection ---
    ScheduledTask("16:00", "sat_reflection_start", "*** SATURDAY REFLECTION ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("16:15", "sat_what_worked", "What worked this week", "went_well", priority=1, saturday_only=True),
    ScheduledTask("16:30", "sat_what_failed", "What failed this week", "went_wrong", priority=1, saturday_only=True),
    ScheduledTask("16:45", "sat_self_assessment", "Self-assessment", "self_assess", priority=1, saturday_only=True),

    # --- 17:00 PM - Memory Updates ---
    ScheduledTask("17:00", "sat_memory_start", "*** MEMORY UPDATES ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("17:15", "sat_update_rules", "Update trading rules", "update_rules", priority=1, saturday_only=True),
    ScheduledTask("17:30", "sat_update_episodic", "Update episodic memory", "update_episodic", priority=1, saturday_only=True),
    ScheduledTask("17:45", "sat_lessons_save", "Save lessons to memory", "update_episodic", priority=1, saturday_only=True),

    # --- 18:00 PM - System Maintenance ---
    ScheduledTask("18:00", "sat_maint_start", "*** SATURDAY MAINTENANCE ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("18:15", "sat_cleanup_logs", "Clean up old logs", "cleanup", priority=2, saturday_only=True),
    ScheduledTask("18:30", "sat_cleanup_cache", "Clean up old cache", "cleanup", priority=2, saturday_only=True),
    ScheduledTask("18:45", "sat_backup_all", "Full system backup", "backup_state", priority=1, saturday_only=True),

    # --- 19:00 PM - Final Saturday Checks ---
    ScheduledTask("19:00", "sat_final_start", "*** SATURDAY FINAL CHECKS ***", "log_weekend_start", priority=1, saturday_only=True),
    ScheduledTask("19:15", "sat_final_health", "Final health check", "full_health", priority=1, saturday_only=True),
    ScheduledTask("19:30", "sat_backup_verify", "Verify backup integrity", "data_integrity", priority=1, saturday_only=True),
    ScheduledTask("19:45", "sat_all_saved", "Verify all saved", "save_research", priority=1, saturday_only=True),
    ScheduledTask("20:00", "sat_end", "*** SATURDAY SESSION END ***", "log_weekend_end", priority=1, saturday_only=True),

    # =========================================================================
    # SUNDAY - LEARNING, DISCOVERY, BACKTESTING, RESEARCH
    # This is the day we learn, experiment, and improve the system
    # =========================================================================

    # --- 08:00 AM - Sunday System Startup ---
    ScheduledTask("08:00", "sun_start", "*** SUNDAY - LEARNING DAY ***", "log_weekend_start", priority=1, sunday_only=True),
    ScheduledTask("08:05", "sun_health_check", "System health check", "full_health", priority=1, sunday_only=True),
    ScheduledTask("08:10", "sun_data_check", "Data integrity check", "data_integrity", priority=1, sunday_only=True),
    ScheduledTask("08:15", "sun_saturday_review", "Review Saturday outputs", "gen_weekly_report", priority=1, sunday_only=True),

    # --- 09:00 AM - Deep Backtesting ---
    ScheduledTask("09:00", "sun_backtest_start", "*** DEEP BACKTESTING ***", "log_deep_research", priority=1, sunday_only=True),
    ScheduledTask("09:05", "sun_backtest_ibs", "Backtest IBS+RSI strategy", "deep_backtest", priority=2, sunday_only=True),
    ScheduledTask("09:25", "sun_backtest_turtle", "Backtest Turtle Soup strategy", "deep_backtest", priority=2, sunday_only=True),
    ScheduledTask("09:45", "sun_backtest_dual", "Backtest Dual Strategy", "deep_backtest", priority=2, sunday_only=True),

    # --- 10:00 AM - Walk-Forward Analysis ---
    ScheduledTask("10:00", "sun_wf_start", "*** WALK-FORWARD ANALYSIS ***", "log_deep_research", priority=1, sunday_only=True),
    ScheduledTask("10:05", "sun_wf_run", "Run full walk-forward", "full_wf", priority=2, sunday_only=True),
    ScheduledTask("10:30", "sun_wf_analyze", "Analyze WF results", "analyze_trades", priority=2, sunday_only=True),
    ScheduledTask("10:45", "sun_wf_stability", "Check parameter stability", "review_params", priority=2, sunday_only=True),

    # --- 11:00 AM - Parameter Optimization ---
    ScheduledTask("11:00", "sun_param_start", "*** PARAMETER OPTIMIZATION ***", "log_deep_research", priority=1, sunday_only=True),
    ScheduledTask("11:05", "sun_grid_search", "Run parameter grid search", "optimize_params", priority=2, sunday_only=True),
    ScheduledTask("11:25", "sun_sensitivity", "Parameter sensitivity test", "optimize_params", priority=2, sunday_only=True),
    ScheduledTask("11:45", "sun_robustness", "Robustness check", "review_params", priority=2, sunday_only=True),

    # --- 12:00 PM - Knowledge Scraping ---
    ScheduledTask("12:00", "sun_scrape_start", "*** KNOWLEDGE SCRAPING ***", "log_deep_research", priority=1, sunday_only=True),
    ScheduledTask("12:05", "sun_scrape_arxiv", "Scrape arXiv papers", "scrape_arxiv", priority=2, sunday_only=True),
    ScheduledTask("12:15", "sun_scrape_ssrn", "Scrape SSRN papers", "scrape_ssrn", priority=2, sunday_only=True),
    ScheduledTask("12:25", "sun_scrape_github", "Scrape GitHub repos", "scrape_github", priority=2, sunday_only=True),
    ScheduledTask("12:35", "sun_scrape_reddit", "Scrape Reddit", "scrape_reddit", priority=2, sunday_only=True),
    ScheduledTask("12:45", "sun_scrape_so", "Scrape StackOverflow", "scrape_stackoverflow", priority=2, sunday_only=True),
    ScheduledTask("12:55", "sun_scrape_all", "Scrape all sources", "scrape_all", priority=2, sunday_only=True),

    # --- 13:00 PM - Knowledge Integration ---
    ScheduledTask("13:00", "sun_knowledge_start", "*** KNOWLEDGE INTEGRATION ***", "log_deep_research", priority=1, sunday_only=True),
    ScheduledTask("13:05", "sun_parse_scraped", "Parse scraped content", "integrate_knowledge", priority=2, sunday_only=True),
    ScheduledTask("13:20", "sun_evaluate_discoveries", "Evaluate discoveries", "check_hypotheses", priority=2, sunday_only=True),
    ScheduledTask("13:35", "sun_rank_discoveries", "Rank by potential", "rank_signals", priority=2, sunday_only=True),
    ScheduledTask("13:50", "sun_integrate_knowledge", "Integrate into system", "deep_integrate", priority=2, sunday_only=True),

    # --- 14:00 PM - ML Model Training ---
    ScheduledTask("14:00", "sun_ml_start", "*** ML MODEL TRAINING ***", "log_deep_research", priority=1, sunday_only=True),
    ScheduledTask("14:05", "sun_ml_data_prep", "Prepare training data", "data_integrity", priority=2, sunday_only=True),
    ScheduledTask("14:15", "sun_ml_features", "Feature engineering", "precalc_indicators", priority=2, sunday_only=True),
    ScheduledTask("14:30", "sun_train_lstm", "Train LSTM model", "full_retrain", priority=2, sunday_only=True),
    ScheduledTask("14:45", "sun_train_hmm", "Train HMM regime model", "full_retrain", priority=2, sunday_only=True),
    ScheduledTask("15:00", "sun_train_xgboost", "Train XGBoost model", "full_retrain", priority=2, sunday_only=True),
    ScheduledTask("15:15", "sun_train_lightgbm", "Train LightGBM model", "full_retrain", priority=2, sunday_only=True),
    ScheduledTask("15:30", "sun_ensemble_update", "Update ensemble weights", "check_ensemble", priority=2, sunday_only=True),
    ScheduledTask("15:45", "sun_ml_validate", "Validate ML models", "check_ml", priority=2, sunday_only=True),

    # --- 16:00 PM - Hypothesis Testing ---
    ScheduledTask("16:00", "sun_hypothesis_start", "*** HYPOTHESIS TESTING ***", "log_deep_research", priority=1, sunday_only=True),
    ScheduledTask("16:05", "sun_review_hypotheses", "Review existing hypotheses", "check_hypotheses", priority=2, sunday_only=True),
    ScheduledTask("16:20", "sun_test_hypothesis_1", "Test hypothesis 1", "run_experiment", priority=2, sunday_only=True),
    ScheduledTask("16:35", "sun_test_hypothesis_2", "Test hypothesis 2", "run_experiment", priority=2, sunday_only=True),
    ScheduledTask("16:50", "sun_test_hypothesis_3", "Test hypothesis 3", "run_experiment", priority=2, sunday_only=True),

    # --- 17:00 PM - Curiosity Engine & Discovery ---
    ScheduledTask("17:00", "sun_curiosity_start", "*** CURIOSITY & DISCOVERY ***", "log_deep_research", priority=1, sunday_only=True),
    ScheduledTask("17:05", "sun_curiosity_run", "Run curiosity engine", "run_curiosity", priority=2, sunday_only=True),
    ScheduledTask("17:20", "sun_generate_hypotheses", "Generate new hypotheses", "generate_hypotheses", priority=2, sunday_only=True),
    ScheduledTask("17:35", "sun_edge_discovery", "Edge discovery scan", "check_edges", priority=2, sunday_only=True),
    ScheduledTask("17:50", "sun_pattern_discovery", "Pattern discovery", "discover_ict", priority=2, sunday_only=True),

    # --- 18:00 PM - Universe & Parameter Review ---
    ScheduledTask("18:00", "sun_review_start", "*** UNIVERSE & PARAM REVIEW ***", "log_deep_research", priority=1, sunday_only=True),
    ScheduledTask("18:05", "sun_universe_review", "Review 800-stock universe", "review_universe", priority=2, sunday_only=True),
    ScheduledTask("18:20", "sun_universe_cleanup", "Remove dead/delisted", "cleanup", priority=2, sunday_only=True),
    ScheduledTask("18:35", "sun_param_review", "Review frozen parameters", "review_params", priority=2, sunday_only=True),
    ScheduledTask("18:50", "sun_param_drift", "Check parameter drift", "check_drift", priority=2, sunday_only=True),

    # --- 19:00 PM - Cognitive Learning ---
    ScheduledTask("19:00", "sun_learning_start", "*** COGNITIVE LEARNING ***", "log_deep_research", priority=1, sunday_only=True),
    ScheduledTask("19:05", "sun_update_episodic", "Update episodic memory", "update_episodic", priority=2, sunday_only=True),
    ScheduledTask("19:15", "sun_update_semantic", "Update semantic memory", "update_semantic", priority=2, sunday_only=True),
    ScheduledTask("19:25", "sun_consolidate", "Consolidate knowledge", "consolidate", priority=2, sunday_only=True),
    ScheduledTask("19:35", "sun_update_patterns", "Update pattern recognition", "update_patterns", priority=2, sunday_only=True),
    ScheduledTask("19:45", "sun_learning_save", "Save learning state", "save_research", priority=2, sunday_only=True),

    # --- 20:00 PM - Final Sunday Tasks ---
    ScheduledTask("20:00", "sun_final_start", "*** SUNDAY FINAL TASKS ***", "log_deep_research", priority=1, sunday_only=True),
    ScheduledTask("20:10", "sun_review_watchlist", "Review Monday watchlist", "validate_watchlist", priority=1, sunday_only=True),
    ScheduledTask("20:20", "sun_final_pregame", "Final pre-game check", "generate_pregame", priority=1, sunday_only=True),
    ScheduledTask("20:30", "sun_monday_prep", "Monday preparation", "next_week_prep", priority=1, sunday_only=True),
    ScheduledTask("20:40", "sun_backup", "Backup all state", "backup_state", priority=1, sunday_only=True),
    ScheduledTask("20:50", "sun_health_final", "Final health check", "full_health", priority=1, sunday_only=True),
    ScheduledTask("21:00", "sun_end", "*** SUNDAY SESSION END ***", "log_weekend_end", priority=1, sunday_only=True),
]


class FullScheduler:
    """
    Full scheduler with all 150+ tasks at specific times.
    Every task logs when it runs. Full visibility.
    """

    def __init__(self, state_dir: Optional[Path] = None):
        if state_dir is None:
            state_dir = Path("state/autonomous/scheduler")
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Task execution log
        self.execution_log: List[Dict] = []
        self.last_execution: Dict[str, datetime] = {}

        # Load state
        self._load_state()

        logger.info(f"FullScheduler initialized with {len(MASTER_SCHEDULE)} scheduled tasks")

    def _load_state(self):
        """Load scheduler state."""
        state_file = self.state_dir / "scheduler_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                for name, ts in data.get("last_execution", {}).items():
                    self.last_execution[name] = datetime.fromisoformat(ts)
            except Exception as e:
                logger.warning(f"Could not load scheduler state: {e}")

    def save_state(self):
        """Save scheduler state."""
        state_file = self.state_dir / "scheduler_state.json"
        data = {
            "last_execution": {k: v.isoformat() for k, v in self.last_execution.items()},
            "total_tasks": len(MASTER_SCHEDULE),
            "updated_at": datetime.now(ET).isoformat(),
        }
        state_file.write_text(json.dumps(data, indent=2))

    def get_tasks_for_time(self, now: datetime) -> List[ScheduledTask]:
        """Get all tasks scheduled for the current time."""
        current_time = now.strftime("%H:%M")
        weekday = now.weekday()  # 0=Mon, 5=Sat, 6=Sun
        is_weekend = weekday >= 5
        is_saturday = weekday == 5
        is_sunday = weekday == 6

        tasks = []
        for task in MASTER_SCHEDULE:
            if not task.enabled:
                continue
            if task.time != current_time:
                continue
            # Weekday filter
            if task.weekdays_only and is_weekend:
                continue
            # Weekend filter (Sat or Sun)
            if task.weekends_only and not is_weekend:
                continue
            # Saturday-only filter
            if task.saturday_only and not is_saturday:
                continue
            # Sunday-only filter
            if task.sunday_only and not is_sunday:
                continue
            tasks.append(task)

        # Sort by priority
        tasks.sort(key=lambda t: t.priority)
        return tasks

    def get_upcoming_tasks(self, now: datetime, hours: int = 1) -> List[ScheduledTask]:
        """Get tasks coming up in the next N hours."""
        current_time = now.time()
        end_time = (now + timedelta(hours=hours)).time()
        is_weekend = now.weekday() >= 5

        tasks = []
        for task in MASTER_SCHEDULE:
            if not task.enabled:
                continue
            if task.weekdays_only and is_weekend:
                continue
            if task.weekends_only and not is_weekend:
                continue

            task_time = datetime.strptime(task.time, "%H:%M").time()
            if current_time <= task_time <= end_time:
                tasks.append(task)

        tasks.sort(key=lambda t: t.time)
        return tasks

    def should_run_task(self, task: ScheduledTask, now: datetime) -> bool:
        """Check if a task should run (not run today yet)."""
        last = self.last_execution.get(task.name)
        if last is None:
            return True

        # Check if already ran today
        return last.date() < now.date()

    def mark_task_executed(self, task: ScheduledTask, status: TaskStatus, result: Any = None):
        """Mark a task as executed."""
        now = datetime.now(ET)
        self.last_execution[task.name] = now

        log_entry = {
            "timestamp": now.isoformat(),
            "time": task.time,
            "name": task.name,
            "description": task.description,
            "status": status.value,
            "result": str(result) if result else None,
        }
        self.execution_log.append(log_entry)

        # Log with visibility
        status_emoji = {
            TaskStatus.SUCCESS: "✓",
            TaskStatus.FAILED: "✗",
            TaskStatus.SKIPPED: "⊘",
            TaskStatus.RUNNING: "→",
        }.get(status, "?")

        logger.info(f"  [{task.time}] {status_emoji} {task.name}: {task.description}")

        self.save_state()

    def get_daily_summary(self) -> Dict[str, Any]:
        """Get summary of today's task execution."""
        today = datetime.now(ET).date()

        today_log = [
            e for e in self.execution_log
            if datetime.fromisoformat(e["timestamp"]).date() == today
        ]

        summary = {
            "date": str(today),
            "total_scheduled": len([t for t in MASTER_SCHEDULE if t.enabled]),
            "executed": len(today_log),
            "success": len([e for e in today_log if e["status"] == "success"]),
            "failed": len([e for e in today_log if e["status"] == "failed"]),
            "skipped": len([e for e in today_log if e["status"] == "skipped"]),
        }

        return summary

    def print_schedule(self, for_date: Optional[datetime] = None):
        """Print the full schedule for a day."""
        if for_date is None:
            for_date = datetime.now(ET)

        is_weekend = for_date.weekday() >= 5
        day_type = "WEEKEND" if is_weekend else "WEEKDAY"

        print(f"\n{'='*70}")
        print(f"KOBE FULL SCHEDULE - {for_date.strftime('%A %Y-%m-%d')} ({day_type})")
        print(f"{'='*70}\n")

        current_hour = -1
        for task in sorted(MASTER_SCHEDULE, key=lambda t: t.time):
            if not task.enabled:
                continue
            if task.weekdays_only and is_weekend:
                continue
            if task.weekends_only and not is_weekend:
                continue

            hour = int(task.time.split(":")[0])
            if hour != current_hour:
                current_hour = hour
                print(f"\n--- {hour:02d}:00 ---")

            priority_marker = "*" * (6 - task.priority) if task.priority <= 3 else ""
            print(f"  [{task.time}] {task.name:30} {priority_marker}")

        print(f"\n{'='*70}")
        print(f"Total tasks for {day_type}: {len([t for t in MASTER_SCHEDULE if t.enabled and ((not t.weekdays_only and not t.weekends_only) or (t.weekdays_only and not is_weekend) or (t.weekends_only and is_weekend))])}")
        print(f"{'='*70}\n")


# Singleton
_scheduler = None

def get_scheduler() -> FullScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = FullScheduler()
    return _scheduler


if __name__ == "__main__":
    # Print schedule for today
    scheduler = get_scheduler()
    scheduler.print_schedule()
