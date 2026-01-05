#!/usr/bin/env python3
"""
Update Slippage Defaults from TCA Data
=======================================

Weekly job that analyzes historical TCA/slippage data and updates
the default slippage thresholds based on actual performance.

FIX (2026-01-05): Added for TCA/slippage feedback loop.

The algorithm:
1. Load last 7 days of slippage history
2. Compute median and standard deviation
3. Set alert threshold = median + 2σ (capped at 10-50 bps)
4. Update config and log the change

Usage:
    python scripts/update_slippage_defaults.py

Called by:
    - scripts/run_weekly_training.py
    - autonomous/scheduler.py (weekly task)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import statistics

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.structured_log import get_logger

logger = get_logger(__name__)


# Config file for slippage settings
SLIPPAGE_CONFIG_FILE = Path("config/slippage_thresholds.json")
SLIPPAGE_HISTORY_FILE = Path("state/execution/slippage_history.json")
AUDIT_LOG_FILE = Path("state/execution/slippage_audit.jsonl")

# Default values if no config exists
DEFAULT_ALERT_THRESHOLD_BPS = 25.0
DEFAULT_AVG_ALERT_THRESHOLD_BPS = 15.0

# Bounds for auto-tuned thresholds
MIN_THRESHOLD_BPS = 10.0
MAX_THRESHOLD_BPS = 50.0
MIN_TRADES_FOR_UPDATE = 10


def load_slippage_history(days: int = 7) -> list[Dict[str, Any]]:
    """
    Load slippage history from the last N days.

    Args:
        days: Number of days to look back

    Returns:
        List of slippage records
    """
    if not SLIPPAGE_HISTORY_FILE.exists():
        logger.warning(f"Slippage history file not found: {SLIPPAGE_HISTORY_FILE}")
        return []

    try:
        with open(SLIPPAGE_HISTORY_FILE, "r") as f:
            data = json.load(f)
            all_trades = data.get("trades", [])

        # Filter to last N days
        cutoff = datetime.now() - timedelta(days=days)
        recent_trades = []
        for trade in all_trades:
            try:
                ts = datetime.fromisoformat(trade["timestamp"])
                if ts >= cutoff:
                    recent_trades.append(trade)
            except (KeyError, ValueError):
                continue

        logger.info(f"Loaded {len(recent_trades)} trades from last {days} days")
        return recent_trades

    except Exception as e:
        logger.error(f"Failed to load slippage history: {e}")
        return []


def compute_recommended_thresholds(trades: list[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute recommended slippage thresholds from historical data.

    Algorithm:
    - Alert threshold = median + 2σ (for single-trade alerts)
    - Avg alert threshold = median + 1σ (for rolling average alerts)

    Args:
        trades: List of trade slippage records

    Returns:
        Dict with recommended thresholds
    """
    if len(trades) < MIN_TRADES_FOR_UPDATE:
        logger.info(f"Not enough trades ({len(trades)}) for threshold update")
        return {}

    # Extract slippage values
    slippages = [t.get("slippage_bps", 0) for t in trades if t.get("slippage_bps") is not None]

    if not slippages:
        return {}

    # Compute statistics
    median_bps = statistics.median(slippages)
    try:
        std_bps = statistics.stdev(slippages)
    except statistics.StatisticsError:
        std_bps = 0.0

    # Compute recommended thresholds
    # Single trade alert: median + 2σ (anomaly detection)
    alert_threshold = median_bps + 2 * std_bps
    # Rolling avg alert: median + 1σ (systemic issue detection)
    avg_alert_threshold = median_bps + 1 * std_bps

    # Apply bounds
    alert_threshold = max(MIN_THRESHOLD_BPS, min(MAX_THRESHOLD_BPS, alert_threshold))
    avg_alert_threshold = max(MIN_THRESHOLD_BPS, min(MAX_THRESHOLD_BPS, avg_alert_threshold))

    # Ensure alert > avg_alert
    if avg_alert_threshold >= alert_threshold:
        avg_alert_threshold = alert_threshold * 0.75

    return {
        "alert_threshold_bps": round(alert_threshold, 1),
        "avg_alert_threshold_bps": round(avg_alert_threshold, 1),
        "median_bps": round(median_bps, 2),
        "std_bps": round(std_bps, 2),
        "sample_size": len(slippages),
    }


def load_current_config() -> Dict[str, float]:
    """Load current slippage thresholds from config."""
    if SLIPPAGE_CONFIG_FILE.exists():
        try:
            with open(SLIPPAGE_CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load slippage config: {e}")

    # Return defaults if no config
    return {
        "alert_threshold_bps": DEFAULT_ALERT_THRESHOLD_BPS,
        "avg_alert_threshold_bps": DEFAULT_AVG_ALERT_THRESHOLD_BPS,
        "updated_at": None,
    }


def save_config(config: Dict[str, Any]) -> None:
    """Save slippage config to file."""
    SLIPPAGE_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config["updated_at"] = datetime.now().isoformat()

    with open(SLIPPAGE_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved slippage config: {SLIPPAGE_CONFIG_FILE}")


def log_audit(old_config: Dict, new_config: Dict, stats: Dict) -> None:
    """Log the threshold change for audit trail."""
    AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": "threshold_update",
        "old_alert_threshold_bps": old_config.get("alert_threshold_bps"),
        "new_alert_threshold_bps": new_config.get("alert_threshold_bps"),
        "old_avg_alert_threshold_bps": old_config.get("avg_alert_threshold_bps"),
        "new_avg_alert_threshold_bps": new_config.get("avg_alert_threshold_bps"),
        "median_bps": stats.get("median_bps"),
        "std_bps": stats.get("std_bps"),
        "sample_size": stats.get("sample_size"),
    }

    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(json.dumps(audit_entry) + "\n")

    logger.info(f"Audit log updated: {AUDIT_LOG_FILE}")


def update_slippage_defaults(lookback_days: int = 7, dry_run: bool = False) -> Dict[str, Any]:
    """
    Main function to update slippage thresholds from TCA data.

    Args:
        lookback_days: Days of history to analyze
        dry_run: If True, don't actually update config

    Returns:
        Result dict with old and new thresholds
    """
    logger.info(f"=== Updating slippage defaults (lookback={lookback_days}d, dry_run={dry_run}) ===")

    # Load history
    trades = load_slippage_history(lookback_days)

    if not trades:
        logger.info("No trades found, skipping update")
        return {"status": "skipped", "reason": "no_trades"}

    # Compute recommended thresholds
    stats = compute_recommended_thresholds(trades)

    if not stats:
        logger.info("Insufficient data for threshold update")
        return {"status": "skipped", "reason": "insufficient_data"}

    # Load current config
    old_config = load_current_config()

    # Check if update is needed (threshold changed by > 10%)
    old_alert = old_config.get("alert_threshold_bps", DEFAULT_ALERT_THRESHOLD_BPS)
    new_alert = stats["alert_threshold_bps"]

    change_pct = abs(new_alert - old_alert) / old_alert * 100 if old_alert > 0 else 100

    if change_pct < 10:
        logger.info(f"Threshold change too small ({change_pct:.1f}%), skipping update")
        return {
            "status": "skipped",
            "reason": "change_too_small",
            "change_pct": round(change_pct, 1),
        }

    # Prepare new config
    new_config = {
        "alert_threshold_bps": stats["alert_threshold_bps"],
        "avg_alert_threshold_bps": stats["avg_alert_threshold_bps"],
        "median_bps": stats["median_bps"],
        "std_bps": stats["std_bps"],
        "sample_size": stats["sample_size"],
        "lookback_days": lookback_days,
    }

    if not dry_run:
        # Save config
        save_config(new_config)

        # Log audit
        log_audit(old_config, new_config, stats)

    logger.info(
        f"Slippage thresholds updated: "
        f"alert {old_alert} → {new_alert} bps, "
        f"avg_alert {old_config.get('avg_alert_threshold_bps')} → {stats['avg_alert_threshold_bps']} bps"
    )

    return {
        "status": "updated" if not dry_run else "dry_run",
        "old_config": old_config,
        "new_config": new_config,
        "stats": stats,
        "change_pct": round(change_pct, 1),
    }


def get_slippage_thresholds() -> Dict[str, float]:
    """
    Get current slippage thresholds for use by SlippageTracker.

    This is the interface that SlippageTracker should use instead of
    hardcoded class constants.

    Returns:
        Dict with alert_threshold_bps and avg_alert_threshold_bps
    """
    config = load_current_config()
    return {
        "alert_threshold_bps": config.get("alert_threshold_bps", DEFAULT_ALERT_THRESHOLD_BPS),
        "avg_alert_threshold_bps": config.get("avg_alert_threshold_bps", DEFAULT_AVG_ALERT_THRESHOLD_BPS),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update slippage thresholds from TCA data")
    parser.add_argument("--days", type=int, default=7, help="Lookback days (default: 7)")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually update")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    result = update_slippage_defaults(lookback_days=args.days, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
