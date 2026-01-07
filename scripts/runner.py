#!/usr/bin/env python3
"""
Kobe 24/7 Runner - Multi-Asset Scheduled Trading Execution.

Runs paper or live trading at configurable times throughout the trading day.
Supports equities (NYSE hours), crypto (24/7), and options (event-driven).

Features:
- Multi-asset scheduling (equities, crypto, options)
- Single-instance enforcement via file locking
- Heartbeat tracking for process monitoring
- Graceful shutdown on SIGTERM/SIGINT
- Kill switch integration
- Live trading safety (requires LIVE_TRADING_APPROVED=YES + --approve-live)
- Decision packet artifact generation
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
from pathlib import Path
from datetime import datetime, timedelta, time as dtime
import subprocess
import sys
import time

# Add project root to path BEFORE importing local modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

# noqa: E402 - Local imports must come after sys.path setup
from core.clock.tz_utils import fmt_ct, now_et, ET  # noqa: E402

from core.structured_log import jlog
from monitor.health_endpoints import start_health_server, update_request_counter
from monitor.heartbeat import HeartbeatWriter
from ops.locks import FileLock, LockError, is_another_instance_running
from market_calendar import is_market_closed, get_market_hours
from core.alerts import send_telegram
from core.kill_switch import is_kill_switch_active, get_kill_switch_info, activate_kill_switch
from config.env_loader import load_env
from portfolio.state_manager import get_state_manager

# Exit manager import for catch-up logic
try:
    from exit_manager import catch_up_missed_exits
    EXIT_MANAGER_AVAILABLE = True
except ImportError:
    EXIT_MANAGER_AVAILABLE = False

# Drift detection imports
try:
    from monitor.drift_detector import (
        get_drift_detector,
        check_drift,
        get_position_scale,
        DriftSeverity,
        PerformanceSnapshot,
    )
    DRIFT_DETECTION_AVAILABLE = True
except ImportError:
    DRIFT_DETECTION_AVAILABLE = False

# Multi-asset clock imports
try:
    from core.clock import MarketClock, AssetType, SessionType
    MULTI_ASSET_AVAILABLE = True
except ImportError:
    MULTI_ASSET_AVAILABLE = False

# Decision packet imports
try:
    from explainability.decision_packet import build_decision_packet
    DECISION_PACKET_AVAILABLE = True
except ImportError:
    DECISION_PACKET_AVAILABLE = False

# Global shutdown flag
_shutdown_requested = False

# Global lock and heartbeat instances
_lock: FileLock | None = None
_heartbeat: HeartbeatWriter | None = None

# Drift → Kill-Switch Escalation (Codex/Gemini 2026-01-04)
# After N consecutive CRITICAL drift detections, escalate to kill-switch
DRIFT_ESCALATION_THRESHOLD = 3  # Consecutive critical drifts before kill-switch
_consecutive_critical_drift_count = 0


STATE_FILE = ROOT / 'state' / 'runner_last.json'
LOCK_FILE = ROOT / 'state' / 'kobe_runner.lock'


def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    jlog('runner_signal_received', level='INFO', signal=sig_name)
    _shutdown_requested = True


def _cleanup():
    """Cleanup on shutdown."""
    global _lock, _heartbeat
    jlog('runner_cleanup', level='INFO')

    if _heartbeat:
        _heartbeat.update("shutting_down")
        _heartbeat.stop()
        _heartbeat = None

    if _lock:
        _lock.release()
        _lock = None


def _setup_signal_handlers():
    """Register signal handlers for graceful shutdown."""
    # SIGINT (Ctrl+C) and SIGTERM (termination request)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Register cleanup on exit
    atexit.register(_cleanup)


def parse_times(csv: str) -> list[dtime]:
    out: list[dtime] = []
    for part in (csv or '').split(','):
        part = part.strip()
        if not part:
            continue
        hh, mm = part.split(':')
        out.append(dtime(hour=int(hh), minute=int(mm)))
    return out


def already_ran(tag: str, today: str) -> bool:
    if not STATE_FILE.exists():
        return False
    try:
        data = json.loads(STATE_FILE.read_text(encoding='utf-8'))
    except Exception:
        return False
    return data.get(tag) == today


def mark_ran(tag: str, today: str) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(STATE_FILE.read_text(encoding='utf-8')) if STATE_FILE.exists() else {}
    except Exception:
        data = {}
    data[tag] = today
    STATE_FILE.write_text(json.dumps(data, indent=2))


def within_market_day(now: datetime) -> bool:
    # Respect US market calendar (holidays and early closes)
    closed, _reason = is_market_closed(now)
    return not closed


def reconcile_positions(dotenv: Path) -> dict:
    """
    Run position reconciliation against broker.

    Returns:
        Dictionary with reconciliation results
    """
    jlog('reconcile_start', level='INFO')
    result = {
        'success': False,
        'positions': [],
        'orders_open': [],
        'discrepancies': [],
    }

    try:
        base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
        key = os.getenv('ALPACA_API_KEY_ID', '')
        sec = os.getenv('ALPACA_API_SECRET_KEY', '')

        if not key or not sec:
            jlog('reconcile_no_credentials', level='WARNING')
            return result

        import requests
        hdr = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': sec}

        # Get positions from broker
        r = requests.get(f"{base}/v2/positions", headers=hdr, timeout=10)
        r.raise_for_status()
        positions = r.json()
        result['positions'] = positions

        # Get open orders
        r = requests.get(f"{base}/v2/orders?status=open", headers=hdr, timeout=10)
        r.raise_for_status()
        orders_open = r.json()
        result['orders_open'] = orders_open

        # Save reconciliation snapshot
        out_dir = ROOT / 'state' / 'reconcile'
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        (out_dir / f'positions_{timestamp}.json').write_text(json.dumps(positions, indent=2))
        (out_dir / f'orders_open_{timestamp}.json').write_text(json.dumps(orders_open, indent=2))

        result['success'] = True
        jlog('reconcile_complete', level='INFO',
             positions_count=len(positions),
             open_orders_count=len(orders_open))

        # Log position summary
        for pos in positions:
            jlog('reconcile_position', level='INFO',
                 symbol=pos.get('symbol'),
                 qty=pos.get('qty'),
                 market_value=pos.get('market_value'),
                 unrealized_pl=pos.get('unrealized_pl'))

    except Exception as e:
        jlog('reconcile_failed', level='ERROR', error=str(e))
        result['error'] = str(e)

    return result


def reconcile_and_fix(dotenv: Path, auto_fix: bool = True) -> dict:
    """
    Enhanced reconciliation that detects AND fixes discrepancies.

    FIX (2026-01-06): Gap #1 - Reconciliation now actively fixes issues.

    Detects:
    - Positions at broker not tracked locally (orphans)
    - Local positions not at broker (stale state)
    - Positions without stop loss orders

    Fixes:
    - Syncs local state to match broker (source of truth)
    - Adds stop losses for positions without them
    - Logs all discrepancies for audit trail

    Args:
        dotenv: Path to .env file
        auto_fix: If True, automatically apply fixes

    Returns:
        Dictionary with reconciliation results and fixes applied
    """
    jlog('reconcile_and_fix_start', level='INFO', auto_fix=auto_fix)

    result = {
        'success': False,
        'broker_positions': [],
        'broker_orders': [],
        'discrepancies': [],
        'fixes_applied': [],
        'alerts': [],
    }

    try:
        import requests

        base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
        key = os.getenv('ALPACA_API_KEY_ID', '')
        sec = os.getenv('ALPACA_API_SECRET_KEY', '')

        if not key or not sec:
            jlog('reconcile_no_credentials', level='WARNING')
            return result

        hdr = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': sec}

        # =====================================================================
        # STEP 1: Get broker state (source of truth)
        # =====================================================================
        r = requests.get(f"{base}/v2/positions", headers=hdr, timeout=10)
        r.raise_for_status()
        broker_positions = r.json()
        result['broker_positions'] = broker_positions

        r = requests.get(f"{base}/v2/orders?status=open", headers=hdr, timeout=10)
        r.raise_for_status()
        broker_orders = r.json()
        result['broker_orders'] = broker_orders

        broker_symbols = {p['symbol'] for p in broker_positions}

        # Build map of symbols with stop orders AND any pending sell orders
        # FIX (2026-01-07): Check ALL sell orders, not just stops, to avoid 403 errors
        # The 403 "insufficient qty available" happens when position already has orders attached
        symbols_with_stops = set()
        symbols_with_pending_sells: dict[str, int] = {}  # symbol -> total sell qty pending
        for order in broker_orders:
            sym = order.get('symbol')
            order_side = order.get('side', '').lower()
            order_type = order.get('type', '').lower()
            order_qty = int(float(order.get('qty', 0)))

            # Track stop orders specifically
            if order_type in ('stop', 'stop_limit'):
                symbols_with_stops.add(sym)

            # Track ALL pending sell orders (to calculate remaining qty available)
            if order_side == 'sell' and order_qty > 0:
                symbols_with_pending_sells[sym] = symbols_with_pending_sells.get(sym, 0) + order_qty

        # =====================================================================
        # STEP 2: Get local state
        # =====================================================================
        sm = get_state_manager()
        local_positions = sm.get_positions()
        local_symbols = set(local_positions.keys())

        # =====================================================================
        # STEP 3: Detect discrepancies
        # =====================================================================

        # 3a. Positions at broker but not tracked locally (orphans)
        orphan_symbols = broker_symbols - local_symbols
        for sym in orphan_symbols:
            broker_pos = next(p for p in broker_positions if p['symbol'] == sym)
            disc = {
                'type': 'ORPHAN_POSITION',
                'symbol': sym,
                'broker_qty': broker_pos.get('qty'),
                'broker_value': broker_pos.get('market_value'),
                'description': f"Position at broker not tracked locally: {sym}",
            }
            result['discrepancies'].append(disc)
            jlog('reconcile_discrepancy', level='WARNING', **disc)

        # 3b. Local positions not at broker (stale state)
        stale_symbols = local_symbols - broker_symbols
        for sym in stale_symbols:
            disc = {
                'type': 'STALE_LOCAL',
                'symbol': sym,
                'local_data': local_positions.get(sym),
                'description': f"Local position not at broker (closed?): {sym}",
            }
            result['discrepancies'].append(disc)
            jlog('reconcile_discrepancy', level='WARNING', **disc)

        # 3c. Positions without stop losses
        positions_without_stops = broker_symbols - symbols_with_stops
        for sym in positions_without_stops:
            broker_pos = next(p for p in broker_positions if p['symbol'] == sym)
            disc = {
                'type': 'NO_STOP_LOSS',
                'symbol': sym,
                'broker_qty': broker_pos.get('qty'),
                'description': f"Position without stop loss order: {sym}",
            }
            result['discrepancies'].append(disc)
            jlog('reconcile_discrepancy', level='WARNING', **disc)

        # =====================================================================
        # STEP 4: Apply fixes if enabled
        # =====================================================================
        if auto_fix and result['discrepancies']:
            jlog('reconcile_applying_fixes', level='INFO',
                 discrepancy_count=len(result['discrepancies']))

            # 4a. Sync local state to broker (broker is source of truth)
            new_local_state = {}
            for pos in broker_positions:
                sym = pos['symbol']
                new_local_state[sym] = {
                    'symbol': sym,
                    'qty': int(float(pos.get('qty', 0))),
                    'entry_price': float(pos.get('avg_entry_price', 0)),
                    'market_value': float(pos.get('market_value', 0)),
                    'unrealized_pl': float(pos.get('unrealized_pl', 0)),
                    'synced_at': now_et().isoformat(),
                    'source': 'broker_reconcile',
                }

                # Preserve local metadata if it exists
                if sym in local_positions:
                    for key in ['strategy', 'entry_date', 'decision_id']:
                        if key in local_positions[sym]:
                            new_local_state[sym][key] = local_positions[sym][key]

            # Atomic update of local state
            sm.set_positions(new_local_state)

            fix = {
                'type': 'SYNC_LOCAL_STATE',
                'symbols_synced': list(broker_symbols),
                'symbols_removed': list(stale_symbols),
            }
            result['fixes_applied'].append(fix)
            jlog('reconcile_fix_applied', level='INFO', **fix)

            # 4b. Add stop losses for positions without them
            # FIX (2026-01-07): Only place stop if there's available qty after pending sells
            for sym in positions_without_stops:
                broker_pos = next(p for p in broker_positions if p['symbol'] == sym)
                entry_price = float(broker_pos.get('avg_entry_price', 0))
                position_qty = int(float(broker_pos.get('qty', 0)))
                pending_sell_qty = symbols_with_pending_sells.get(sym, 0)

                # Calculate remaining quantity available for new orders
                remaining_qty = position_qty - pending_sell_qty

                if entry_price <= 0 or position_qty <= 0:
                    continue

                # Skip if all shares already have pending sell orders
                if remaining_qty <= 0:
                    jlog('reconcile_skip_stop_already_covered', level='INFO',
                         symbol=sym, position_qty=position_qty, pending_sell_qty=pending_sell_qty,
                         reason='All shares already have pending sell orders')
                    continue

                # Calculate ATR-based stop (2 ATR default, ~4% fallback)
                stop_pct = 0.04  # 4% default stop
                stop_price = round(entry_price * (1 - stop_pct), 2)

                try:
                    # Place stop order for REMAINING qty only
                    order_data = {
                        'symbol': sym,
                        'qty': str(remaining_qty),  # Only order for available qty
                        'side': 'sell',
                        'type': 'stop',
                        'time_in_force': 'gtc',
                        'stop_price': str(stop_price),
                    }
                    r = requests.post(
                        f"{base}/v2/orders",
                        headers={**hdr, 'Content-Type': 'application/json'},
                        json=order_data,
                        timeout=10
                    )

                    if r.status_code in (200, 201):
                        order_result = r.json()
                        fix = {
                            'type': 'ADDED_STOP_LOSS',
                            'symbol': sym,
                            'stop_price': stop_price,
                            'qty': remaining_qty,
                            'order_id': order_result.get('id'),
                        }
                        result['fixes_applied'].append(fix)
                        jlog('reconcile_fix_applied', level='INFO', **fix)

                        # Alert
                        alert_msg = f"RECONCILE FIX: Added stop loss for {sym} ({remaining_qty} shares) at ${stop_price:.2f}"
                        result['alerts'].append(alert_msg)
                    elif r.status_code == 403:
                        # 403 = insufficient qty - likely bracket order or OTO attached
                        jlog('reconcile_stop_order_403', level='WARNING',
                             symbol=sym, status=r.status_code,
                             reason='Position likely has bracket/OTO orders attached',
                             response=r.text[:200] if r.text else 'No response')
                    else:
                        jlog('reconcile_stop_order_failed', level='ERROR',
                             symbol=sym, status=r.status_code, response=r.text)

                except Exception as e:
                    jlog('reconcile_stop_order_error', level='ERROR',
                         symbol=sym, error=str(e))

        # =====================================================================
        # STEP 5: Save reconciliation report
        # =====================================================================
        out_dir = ROOT / 'state' / 'reconcile'
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        report = {
            'timestamp': timestamp,
            'broker_positions': len(broker_positions),
            'broker_orders': len(broker_orders),
            'discrepancies': result['discrepancies'],
            'fixes_applied': result['fixes_applied'],
        }
        (out_dir / f'reconcile_report_{timestamp}.json').write_text(
            json.dumps(report, indent=2, default=str)
        )

        # Send alerts if any
        if result['alerts']:
            now = now_et()
            stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
            for alert in result['alerts']:
                try:
                    send_telegram(f"{alert} [{stamp}]")
                except Exception:
                    pass

        result['success'] = True
        jlog('reconcile_and_fix_complete', level='INFO',
             discrepancies=len(result['discrepancies']),
             fixes=len(result['fixes_applied']))

    except Exception as e:
        jlog('reconcile_and_fix_failed', level='ERROR', error=str(e))
        result['error'] = str(e)

    return result


def check_performance_drift() -> dict:
    """
    Check for performance drift and get position scaling recommendation.

    Returns:
        Dictionary with drift check results and position scale
    """
    if not DRIFT_DETECTION_AVAILABLE:
        return {'available': False, 'position_scale': 1.0}

    result = {
        'available': True,
        'position_scale': 1.0,
        'severity': 'NONE',
        'has_drift': False,
        'message': '',
    }

    try:
        # Try to load recent trade performance from logs
        trades_file = ROOT / 'logs' / 'trade_history.jsonl'
        if trades_file.exists():
            trades = []
            with open(trades_file, 'r') as f:
                for line in f:
                    try:
                        trades.append(json.loads(line))
                    except Exception:
                        continue

            if trades:
                # Calculate recent metrics (last 30 trades)
                recent = trades[-30:] if len(trades) > 30 else trades
                wins = sum(1 for t in recent if t.get('pnl', 0) > 0)
                total = len(recent)
                wr = wins / total if total > 0 else 0.5

                gross_profit = sum(t['pnl'] for t in recent if t.get('pnl', 0) > 0)
                gross_loss = abs(sum(t['pnl'] for t in recent if t.get('pnl', 0) < 0))
                pf = gross_profit / gross_loss if gross_loss > 0 else 1.0

                metrics = {
                    'wr': wr,
                    'pf': pf,
                    'sharpe': 0.0,  # Simplified
                    'accuracy': wr,
                }

                drift_result = check_drift(metrics)
                result['position_scale'] = drift_result.position_scale
                result['severity'] = drift_result.severity.name
                result['has_drift'] = drift_result.has_drift
                result['message'] = drift_result.message

                if drift_result.has_drift:
                    jlog('performance_drift_detected', level='WARNING',
                         severity=drift_result.severity.name,
                         position_scale=drift_result.position_scale,
                         message=drift_result.message)

                    # Send alert for significant drift
                    if drift_result.severity in (DriftSeverity.SEVERE, DriftSeverity.CRITICAL):
                        try:
                            now = now_et()
                            stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                            msg = f"DRIFT ALERT [{drift_result.severity.name}]: {drift_result.message} [{stamp}]"
                            send_telegram(msg)
                        except Exception:
                            pass
        else:
            jlog('drift_check_no_history', level='DEBUG', message='No trade history file found')
            result['message'] = 'No trade history available'

    except Exception as e:
        jlog('drift_check_error', level='ERROR', error=str(e))
        result['error'] = str(e)

    return result


def get_drift_scaled_position_size(base_size: int) -> int:
    """
    Apply drift-based position scaling to a base size.

    Args:
        base_size: Original position size

    Returns:
        Scaled position size (may be smaller if drift detected)
    """
    if not DRIFT_DETECTION_AVAILABLE:
        return base_size

    scale = get_position_scale()
    scaled_size = int(base_size * scale)

    if scale < 1.0:
        jlog('position_size_scaled', level='INFO',
             base_size=base_size,
             scale=scale,
             scaled_size=scaled_size)

    return max(1, scaled_size)  # Ensure at least 1 if trading


def check_live_trading_approved(require_flag: bool = True) -> tuple[bool, str]:
    """
    Check if live trading is approved.

    Returns (approved, reason).

    CRITICAL SAFETY: Live trading requires BOTH:
    1. LIVE_TRADING_APPROVED=YES environment variable
    2. --approve-live CLI flag (if require_flag=True)
    """
    env_approved = os.getenv('LIVE_TRADING_APPROVED', '').upper() == 'YES'

    if not env_approved:
        return False, "LIVE_TRADING_APPROVED environment variable not set to YES"

    return True, "Live trading approved"


# ==============================================================================
# UNIFIED MEGA SCHEDULER - ALL 200+ TASKS COMBINED
# ==============================================================================
# Combines:
#   - runner.py original (4 tasks)
#   - scheduler_full.py (150+ tasks)
#   - scheduler_kobe.py (100+ tasks)
#
# This is the ONE schedule the robot follows. Every task has a specific time.
# Every task logs when it runs. VISIBILITY IS ACCOUNTABILITY.
#
# SCHEDULE OVERVIEW (ET):
# ============================================================================
# PRE-MARKET (4:00-9:30 AM)
#   04:00-05:00  System health, data integrity, broker test
#   05:00-06:00  Data refresh, universe validation
#   06:00-07:00  Economic data (VIX, Treasury), regime detect
#   07:00-08:00  ML models (LSTM, HMM, ensemble)
#   08:00-09:00  Pre-game blueprint, watchlist validation, news
#   09:00-09:30  Final preflight, position sizing, kill zone status
#
# OPENING RANGE (9:30-10:00 AM) - OBSERVE ONLY, NO TRADES
#   09:30-10:00  Capture opens, record range, detect momentum
#
# MORNING SESSION (10:00-11:30 AM) - PRIMARY TRADING WINDOW
#   10:00        FIRST SCAN + Execute qualified trades
#   10:30        Fallback scan (if watchlist fails)
#   10:00-11:30  Position monitoring every 15 min
#
# LUNCH SESSION (11:30-14:00) - RESEARCH, NO NEW TRADES
#   11:30-14:00  Position monitor, ML checks, research experiments
#
# AFTERNOON SESSION (14:00-15:30) - POWER HOUR
#   14:30        Power hour scan
#   14:45        Build next day watchlist
#   14:00-15:30  Position monitor, exit evaluation
#
# MARKET CLOSE (15:30-16:00) - NO NEW TRADES, MANAGE ONLY
#   15:30-16:00  Final position check, EOD exits, overnight risk
#
# POST-MARKET (16:00-20:00) - LEARNING
#   16:00-17:00  P&L calc, reconciliation, trade analysis
#   17:00-18:00  Cognitive reflection, memory updates
#   18:00-20:00  Reports, research, knowledge scraping
#
# OVERNIGHT (20:00-04:00) - OPTIMIZATION
#   20:00-00:00  Data validation, ML retrain check
#   00:00-04:00  Deep backtest, parameter optimization, cleanup
#
# SATURDAY - Reports first, then Monday watchlist by 9:30 AM ET
# SUNDAY - Learning, backtesting, ML training
# HOLIDAYS - Same as weekend schedule
# ============================================================================

# Main daily cycle scripts
DAILY_CYCLE_TIMES = {
    # === PRE-MARKET EARLY (4:00-6:00 AM) - System & Data ===
    'system_health_check': dtime(4, 0),        # Full system health check
    'data_integrity_check': dtime(4, 5),       # Verify all data files
    'broker_connection_test': dtime(4, 10),    # Test Alpaca connection
    'polygon_data_refresh': dtime(5, 0),       # Refresh Polygon EOD cache
    'universe_validation': dtime(5, 15),       # Validate 900-stock universe
    'indicator_precalc': dtime(5, 30),         # Pre-calculate indicators
    'db_backup': dtime(5, 30),                 # State backup

    # === PRE-MARKET MID (6:00-7:00 AM) - Economic & Regime ===
    'data_update': dtime(6, 0),                # Warm data cache
    'fred_vix_fetch': dtime(6, 0),             # Fetch VIX from FRED
    'fred_treasury_fetch': dtime(6, 5),        # Fetch 10Y Treasury
    'fear_greed_fetch': dtime(6, 10),          # Fetch Fear & Greed Index
    'market_regime_detect': dtime(6, 15),      # Detect market regime
    'morning_report': dtime(6, 30),            # Generate morning summary
    'premarket_data_check': dtime(6, 45),      # Data staleness, splits check

    # === PRE-MARKET LATE (7:00-8:00 AM) - ML Models ===
    'lstm_confidence_run': dtime(7, 0),        # Run LSTM confidence model
    'hmm_regime_update': dtime(7, 15),         # Update HMM regime state
    'ensemble_weights_check': dtime(7, 30),    # Check ensemble model weights
    'weekday_game_plan': dtime(7, 30),         # Daily game plan

    # === PREMARKET (8:00-9:00 AM) - Validation & News ===
    'premarket_validator': dtime(8, 0),        # Validate overnight watchlist (gaps, news)
    'premarket_gap_check': dtime(8, 0),        # Check for overnight gaps
    'pregame_blueprint': dtime(8, 15),         # Comprehensive Pre-Game Blueprint
    'watchlist_validation': dtime(8, 15),      # Validate overnight watchlist
    'news_sentiment_scan': dtime(8, 30),       # Scan news for watchlist stocks

    # === FINAL PREP (9:00-9:30 AM) ===
    'market_news': dtime(9, 0),                # Update sentiment
    'final_preflight_check': dtime(9, 0),      # Final preflight before market
    'premarket_scan': dtime(9, 15),            # Build plan (portfolio-aware)
    'position_sizing_calc': dtime(9, 15),      # Calculate position sizes
    'kill_zone_status': dtime(9, 25),          # Log kill zone status

    # === OPENING RANGE (9:30-10:00) - OBSERVE ONLY ===
    'opening_range_1': dtime(9, 30),           # First observation
    'opening_price_capture': dtime(9, 30),     # Capture all opening prices
    'opening_gap_analysis': dtime(9, 31),      # Analyze opening gaps
    'volume_surge_detect': dtime(9, 32),       # Detect volume surges
    'watchlist_price_update': dtime(9, 33),    # Update watchlist prices
    'opening_range_5min': dtime(9, 35),        # 5-min opening range
    'opening_range_2': dtime(9, 45),           # Second observation
    'opening_range_15min': dtime(9, 50),       # 15-min opening range
    'morning_bias_determine': dtime(9, 57),    # Determine morning bias

    # === MORNING SESSION (10:00-11:30) - PRIMARY TRADING WINDOW ===
    'first_scan': dtime(10, 0),                # PRIMARY SCAN - Execute qualified trades
    'watchlist_signal_check': dtime(10, 5),    # Check watchlist for triggers
    'signal_quality_gate': dtime(10, 10),      # Apply quality gate to signals
    'position_monitor_1': dtime(10, 20),       # Position monitor
    'position_monitor_2': dtime(10, 35),       # Position monitor
    'fallback_scan': dtime(10, 30),            # Fallback scan if watchlist empty
    'position_monitor_3': dtime(10, 50),       # Position monitor
    'position_monitor_4': dtime(11, 5),        # Position monitor
    'stop_loss_check_1': dtime(11, 15),        # Check stop losses
    'position_monitor_5': dtime(11, 20),       # Position monitor
    'morning_window_close': dtime(11, 25),     # PRIMARY WINDOW CLOSING

    # === LUNCH SESSION (11:30-14:00) - RESEARCH, NO NEW TRADES ===
    'lunch_position_monitor': dtime(11, 30),   # Monitor positions during lunch
    'cognitive_quick_check': dtime(11, 50),    # Quick cognitive check
    'midday_pnl_log': dtime(12, 0),            # Log midday P&L
    'reconcile_midday': dtime(12, 0),          # Half time AI Briefing
    'research_experiment': dtime(12, 0),       # Run parameter experiment
    'curiosity_scan': dtime(12, 10),           # Run curiosity engine scan
    'ict_pattern_discovery': dtime(12, 30),    # Discover ICT patterns
    'position_monitor_6': dtime(12, 45),       # Position monitor
    'hourly_pnl_log': dtime(13, 0),            # Log hourly P&L
    'ml_model_check': dtime(13, 10),           # Check ML model performance
    'hmm_regime_check': dtime(13, 15),         # Check HMM regime
    'lstm_confidence_check': dtime(13, 20),    # Check LSTM confidence
    'position_monitor_7': dtime(13, 30),       # Position monitor
    'knowledge_integration': dtime(13, 45),    # Integrate scraped knowledge
    'pre_power_hour_prep': dtime(13, 55),      # Pre-power hour preparation

    # === AFTERNOON SESSION (14:00-15:30) - POWER HOUR ===
    'afternoon_scan': dtime(14, 0),            # Afternoon universe scan
    'position_monitor_8': dtime(14, 5),        # Position monitor
    'position_adjustment': dtime(14, 15),      # Adjust positions if needed
    'position_monitor_9': dtime(14, 20),       # Position monitor
    'power_hour_scan': dtime(14, 30),          # POWER HOUR SCAN
    'watchlist_next_day': dtime(14, 45),       # BUILD NEXT DAY WATCHLIST
    'position_monitor_10': dtime(14, 50),      # Position monitor
    'hourly_pnl_log_2': dtime(15, 0),          # Log hourly P&L
    'position_monitor_11': dtime(15, 5),       # Position monitor
    'exit_evaluation': dtime(15, 15),          # Evaluate exit conditions
    'position_monitor_12': dtime(15, 20),      # Position monitor
    'power_hour_close': dtime(15, 25),         # POWER HOUR CLOSING

    # === MARKET CLOSE (15:30-16:00) - NO NEW TRADES ===
    'overnight_watchlist': dtime(15, 30),      # Build tomorrow's Top 5
    'swing_scanner': dtime(15, 30),            # Swing setups
    'final_position_check': dtime(15, 30),     # Final position check
    'close_stop_check': dtime(15, 31),         # Final stop check
    'position_monitor_13': dtime(15, 35),      # Position monitor
    'eod_exit_check': dtime(15, 45),           # Check for EOD exits
    'final_pnl_update': dtime(15, 50),         # Final P&L update
    'overnight_risk_check': dtime(15, 52),     # Check overnight risk
    'position_close_check': dtime(15, 55),     # Enforce time stops before close

    # === POST-MARKET (16:00-18:00) - LEARNING ===
    'daily_pnl_calc': dtime(16, 0),            # Calculate daily P&L
    'post_game': dtime(16, 0),                 # AI Briefing + lessons
    'cache_refresh_eod': dtime(16, 2),         # CRITICAL: Refresh polygon_cache with today's close
    'eod_report': dtime(16, 5),                # EOD performance report
    'broker_reconcile': dtime(16, 10),         # Reconcile with broker
    'reconcile_eod': dtime(16, 15),            # Full reconciliation + report
    'trade_analysis': dtime(16, 30),           # Analyze today's trades
    'lesson_extraction': dtime(16, 45),        # Extract lessons from trades
    'self_assessment': dtime(16, 55),          # Run self-assessment

    # === COGNITIVE LEARNING (17:00-18:00) ===
    'cognitive_reflection': dtime(17, 0),      # Run cognitive reflection
    'eod_learning': dtime(17, 0),              # Weekly ML training (Fridays)
    'cognitive_learn': dtime(17, 15),          # Daily cognitive consolidation
    'episodic_memory_update': dtime(17, 15),   # Update episodic memory
    'experience_store': dtime(17, 20),         # Store experience in memory
    'pattern_recognition_update': dtime(17, 25), # Update pattern recognition
    'learn_analysis': dtime(17, 30),           # Daily trade learning analysis
    'semantic_memory_update': dtime(17, 30),   # Update semantic memory
    'knowledge_consolidate': dtime(17, 40),    # Consolidate knowledge
    'curiosity_update': dtime(17, 45),         # Update curiosity engine
    'edge_discovery_check': dtime(17, 55),     # Check for edge discovery

    # === REPORTS & RESEARCH (18:00-20:00) ===
    'daily_report_gen': dtime(18, 0),          # Generate daily report
    'eod_finalize': dtime(18, 0),              # Finalize EOD data
    'performance_metrics': dtime(18, 5),       # Calculate performance metrics
    'watchlist_finalize': dtime(18, 30),       # Finalize next day watchlist
    'news_scan_watchlist': dtime(18, 35),      # Scan news for watchlist
    'earnings_check': dtime(18, 40),           # Check earnings dates
    'catalyst_check': dtime(18, 45),           # Check for catalysts
    'sector_analysis': dtime(18, 50),          # Analyze sector strength
    'market_breadth': dtime(18, 55),           # Check market breadth
    'scrape_arxiv': dtime(19, 0),              # Scrape arXiv for papers
    'scrape_ssrn': dtime(19, 5),               # Scrape SSRN papers
    'research_digest': dtime(19, 15),          # Create research digest
    'scrape_reddit': dtime(19, 35),            # Scrape Reddit
    'knowledge_integrate': dtime(19, 45),      # Integrate all scraped knowledge
    'research_state_save': dtime(19, 55),      # Save research state

    # === OVERNIGHT (20:00-04:00) - OPTIMIZATION ===
    'overnight_start': dtime(20, 0),           # OVERNIGHT SESSION START
    'full_data_validation': dtime(20, 0),      # Full data validation
    'ml_model_retrain_check': dtime(21, 0),    # Check if ML models need retrain
    'nightly_ml_retrain': dtime(21, 15),       # Retrain ML models if drift
    'walk_forward_mini': dtime(21, 30),        # Mini walk-forward test
    'research_experiment_2': dtime(22, 0),     # Run overnight experiment
    'curiosity_engine': dtime(22, 30),         # Run curiosity engine
    'hypothesis_generation': dtime(23, 0),     # Generate new hypotheses
    'knowledge_consolidation': dtime(23, 30),  # Consolidate knowledge base
    'midnight_health_check': dtime(0, 0),      # Midnight system health
    'deep_backtest': dtime(1, 0),              # Run deep backtest
    'parameter_optimization': dtime(2, 0),     # Parameter optimization run
    'data_cleanup': dtime(3, 0),               # Clean up old cache/logs
    'system_backup': dtime(3, 30),             # Backup system state
}

# Weekend/Holiday schedule (runs on market-closed days)
WEEKEND_CYCLE_TIMES = {
    # === SATURDAY MORNING - Reports & Weekly Analysis ===
    'sat_health_check': dtime(6, 0),           # Full system health check
    'sat_data_integrity': dtime(6, 4),         # Check all data files
    'sat_weekly_pnl': dtime(6, 17),            # Calculate weekly P&L
    'sat_win_rate': dtime(6, 32),              # Calculate weekly win rate
    'sat_trade_review': dtime(6, 47),          # Review all trades this week

    # === SATURDAY WATCHLIST (by 9:30 AM ET) ===
    'sat_scan_universe': dtime(7, 2),          # Scan 900 stocks for setups
    'sat_quality_gate': dtime(7, 17),          # Apply quality gate
    'sat_select_top5': dtime(7, 32),           # Select top 5 watchlist
    'sat_historical_patterns': dtime(7, 47),   # Historical pattern analysis
    'sat_expected_move': dtime(8, 2),          # Expected move analysis
    'sat_support_resistance': dtime(8, 17),    # S/R levels
    'sat_news_totd': dtime(8, 29),             # TOTD headlines
    'sat_political_activity': dtime(8, 39),    # Congressional/insider activity
    'sat_sector_context': dtime(8, 47),        # Sector context
    'sat_volume_analysis': dtime(8, 59),       # Volume analysis
    'sat_entry_levels': dtime(9, 5),           # Entry/Stop/Target levels
    'sat_thesis': dtime(9, 17),                # Bull/Bear cases
    'sat_pregame_save': dtime(9, 30),          # SAVE ALL - WATCHLIST READY

    # === SATURDAY AFTERNOON - Deep Analysis ===
    'sat_trade_analysis': dtime(10, 0),        # Detailed trade analysis
    'sat_lessons': dtime(11, 0),               # Lessons learned
    'sat_strategy_compare': dtime(12, 0),      # Strategy comparison
    'sat_market_analysis': dtime(13, 0),       # Market analysis
    'sat_calendar': dtime(14, 0),              # Next week calendar
    'sat_summary': dtime(15, 0),               # Weekly summary report
    'sat_reflection': dtime(16, 0),            # Cognitive reflection
    'sat_memory_update': dtime(17, 0),         # Memory updates
    'sat_maintenance': dtime(18, 0),           # System maintenance
    'sat_final_checks': dtime(19, 0),          # Final Saturday checks
    'sat_end': dtime(20, 0),                   # Saturday session end

    # === SUNDAY - Learning, Discovery, Research ===
    'sun_start': dtime(8, 0),                  # Sunday system startup
    'sun_saturday_review': dtime(8, 15),       # Review Saturday outputs
    'sun_backtest': dtime(9, 0),               # Deep backtesting
    'sun_walk_forward': dtime(10, 0),          # Walk-forward analysis
    'sun_param_optimization': dtime(11, 0),    # Parameter optimization
    'sun_knowledge_scrape': dtime(12, 0),      # Knowledge scraping
    'sun_knowledge_integrate': dtime(13, 0),   # Knowledge integration
    'sun_ml_training': dtime(14, 0),           # ML model training
    'sun_hypothesis_test': dtime(16, 0),       # Hypothesis testing
    'sun_curiosity': dtime(17, 0),             # Curiosity & discovery
    'sun_universe_review': dtime(18, 0),       # Universe & param review
    'sun_cognitive_learning': dtime(19, 0),    # Cognitive learning
    'sun_monday_prep': dtime(20, 0),           # Monday preparation
    'sun_end': dtime(21, 0),                   # Sunday session end

    # === HOLIDAY SCHEDULE (Same as weekend) ===
    'holiday_backup': dtime(5, 30),            # State backup
    'holiday_health_check': dtime(6, 0),       # Full system health
    'holiday_log_cleanup': dtime(6, 15),       # Purge old logs
    'holiday_data_integrity': dtime(6, 30),    # Missing bars, duplicates
    'holiday_universe_refresh': dtime(7, 0),   # Delistings, halted tickers
    'holiday_broker_test': dtime(7, 30),       # Broker connectivity test
    'holiday_research_start': dtime(8, 0),     # Start research session
    'holiday_pattern_scan': dtime(8, 30),      # Scan for new patterns
    'holiday_alpha_discovery': dtime(9, 0),    # Alpha screening
    'holiday_edge_analysis': dtime(9, 30),     # Edge discovery analysis
    'holiday_backtest_quick': dtime(10, 0),    # Quick backtest validation
    'holiday_wf_test': dtime(10, 30),          # Walk-forward test
    'holiday_strategy_compare': dtime(11, 0),  # Strategy comparison
    'holiday_param_drift': dtime(11, 30),      # Parameter drift check
    'holiday_optimize_start': dtime(12, 0),    # Start optimization
    'holiday_grid_search': dtime(12, 30),      # Grid search parameters
    'holiday_threshold_tune': dtime(13, 0),    # Tune confidence thresholds
    'holiday_risk_calibrate': dtime(13, 30),   # Calibrate risk limits
    'holiday_ml_train': dtime(14, 0),          # ML model training
    'holiday_meta_retrain': dtime(14, 30),     # Meta model retrain
    'holiday_ensemble_update': dtime(15, 0),   # Ensemble update
    'holiday_hmm_regime': dtime(15, 30),       # HMM regime recalibration
    'holiday_cognitive_reflect': dtime(16, 0), # Cognitive reflection
    'holiday_hypothesis_test': dtime(16, 30),  # Test hypotheses
    'holiday_memory_consolidate': dtime(17, 0),# Memory consolidation
    'holiday_self_calibrate': dtime(17, 30),   # Self-model calibration
    'holiday_monte_carlo': dtime(18, 0),       # Monte Carlo simulation
    'holiday_stress_test': dtime(18, 30),      # Stress testing
    'holiday_var_calc': dtime(19, 0),          # VaR recalculation
    'holiday_drawdown_analysis': dtime(19, 30),# Drawdown analysis
    'holiday_next_day_prep': dtime(20, 0),     # Prepare for next day
    'holiday_watchlist_build': dtime(20, 30),  # Build watchlist
    'holiday_preview_scan': dtime(21, 0),      # Preview mode scan
    'holiday_final_backup': dtime(21, 30),     # Final backup
    'holiday_complete': dtime(22, 0),          # Holiday schedule complete
}

# Position monitor runs every N minutes during market hours
POSITION_MONITOR_INTERVAL_MINUTES = 15  # Check positions every 15 minutes

# Track last position monitor run
_last_position_monitor: datetime | None = None

# Track last crypto scan run
_last_crypto_scan: datetime | None = None
CRYPTO_SCAN_CADENCE_HOURS = 4  # Default, can be overridden by --crypto-cadence

# Track last options scan run
_last_options_scan: datetime | None = None
OPTIONS_SCAN_CADENCE_HOURS = 2  # Default, can be overridden by --options-cadence


def run_position_monitor(dotenv: Path) -> int:
    """
    Run position monitor to check time-based exits (3-bar rule for Turtle Soup, 7-bar for IBS+RSI).

    Returns: returncode from subprocess
    """
    global _last_position_monitor

    # Check kill switch
    if is_kill_switch_active():
        info = get_kill_switch_info()
        jlog('position_monitor_blocked_by_kill_switch', level='WARNING',
             reason=info.get('reason') if info else 'Unknown')
        return -1

    script_path = ROOT / 'scripts' / 'exit_manager.py'
    if not script_path.exists():
        jlog('position_monitor_script_not_found', level='WARNING', path=str(script_path))
        return -1

    cmd = [sys.executable, str(script_path), '--check-time-exits']
    if dotenv.exists():
        cmd.extend(['--dotenv', str(dotenv)])

    jlog('position_monitor_execute', cmd=' '.join(cmd))

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if p.stdout:
            # Only print if there's something interesting
            if 'EXIT' in p.stdout or 'TIME STOP' in p.stdout or 'STOP LOSS' in p.stdout:
                print(f"[position_monitor] {p.stdout}")
        if p.stderr:
            print(f"[position_monitor] {p.stderr}", file=sys.stderr)

        _last_position_monitor = now_et()
        jlog('position_monitor_complete', returncode=p.returncode)
        return p.returncode

    except subprocess.TimeoutExpired:
        jlog('position_monitor_timeout', level='ERROR')
        return -1
    except Exception as e:
        jlog('position_monitor_error', level='ERROR', error=str(e))
        return -1


def should_run_position_monitor() -> bool:
    """Check if it's time to run position monitor."""
    global _last_position_monitor

    if _last_position_monitor is None:
        return True

    now = now_et()
    minutes_since_last = (now - _last_position_monitor).total_seconds() / 60

    return minutes_since_last >= POSITION_MONITOR_INTERVAL_MINUTES


def run_crypto_scan(dotenv: Path, cadence_hours: int = 4) -> int:
    """
    Run crypto scanner (24/7, independent of market hours).

    Crypto trades 24/7, so this can run any time.
    Uses the same DualStrategyScanner as equities.

    Returns: returncode from subprocess, or number of signals found
    """
    global _last_crypto_scan

    # Check kill switch
    if is_kill_switch_active():
        info = get_kill_switch_info()
        jlog('crypto_scan_blocked_by_kill_switch', level='WARNING',
             reason=info.get('reason') if info else 'Unknown')
        return -1

    jlog('crypto_scan_start', cadence_hours=cadence_hours)

    try:
        # Import crypto scanner
        from scanner.crypto_signals import scan_crypto, CRYPTO_DATA_AVAILABLE

        if not CRYPTO_DATA_AVAILABLE:
            jlog('crypto_scan_unavailable', level='WARNING', reason='Crypto data provider not available')
            return -1

        # Run the scan
        signals = scan_crypto(cap=8, max_signals=3, verbose=False)

        _last_crypto_scan = now_et()

        if signals.empty:
            jlog('crypto_scan_complete', signals_found=0)
            return 0

        # Log signals found
        signal_count = len(signals)
        jlog('crypto_scan_complete', signals_found=signal_count)

        # Save to state file for visibility
        state_dir = ROOT / 'state' / 'watchlist'
        state_dir.mkdir(parents=True, exist_ok=True)
        crypto_file = state_dir / 'crypto_signals.json'

        signals_list = signals.to_dict(orient='records')
        with open(crypto_file, 'w') as f:
            json.dump({
                'scan_time': _last_crypto_scan.isoformat(),
                'signals_count': signal_count,
                'signals': signals_list,
            }, f, indent=2, default=str)

        jlog('crypto_scan_saved', file=str(crypto_file), signals=signal_count)

        # Send alert if signals found
        if signal_count > 0:
            try:
                now2 = now_et()
                stamp2 = f"{fmt_ct(now2)} | {now2.strftime('%I:%M %p').lstrip('0')} ET"
                symbols = ', '.join(signals['symbol'].tolist()[:3])
                send_telegram(f"Kobe CRYPTO: {signal_count} signal(s) found: {symbols} [{stamp2}]")
            except Exception:
                pass

        return signal_count

    except ImportError as e:
        jlog('crypto_scan_import_error', level='ERROR', error=str(e))
        return -1
    except Exception as e:
        jlog('crypto_scan_error', level='ERROR', error=str(e))
        return -1


def should_run_crypto_scan(cadence_hours: int = 4) -> bool:
    """Check if it's time to run crypto scan."""
    global _last_crypto_scan

    if _last_crypto_scan is None:
        return True

    now = now_et()
    hours_since_last = (now - _last_crypto_scan).total_seconds() / 3600

    return hours_since_last >= cadence_hours


def run_options_scan(dotenv: Path, universe: Path, cap: int, cadence_hours: int = 2) -> int:
    """
    Run options scanner - generates options signals from equity signals.

    Options scanning runs during market hours and generates CALL/PUT signals
    from the top equity signals using 30-delta targeting and Black-Scholes pricing.

    Returns: number of options signals found, or -1 on error
    """
    global _last_options_scan

    # Check kill switch
    if is_kill_switch_active():
        info = get_kill_switch_info()
        jlog('options_scan_blocked_by_kill_switch', level='WARNING',
             reason=info.get('reason') if info else 'Unknown')
        return -1

    # Check if market is open (options only trade during market hours)
    now = now_et()
    closed, reason = is_market_closed(now)
    if closed:
        jlog('options_scan_skipped_market_closed', level='DEBUG', reason=reason)
        return 0

    jlog('options_scan_start', cadence_hours=cadence_hours)

    try:
        # Import options signal generator
        from scanner.options_signals import generate_options_signals, OPTIONS_AVAILABLE

        if not OPTIONS_AVAILABLE:
            jlog('options_scan_unavailable', level='WARNING', reason='Options modules not available')
            return -1

        # First, run a quick equity scan to get signals to convert
        from strategies.dual_strategy import DualStrategyScanner, DualStrategyParams
        from data.providers.polygon_eod import fetch_daily_bars_polygon
        from data.universe.loader import load_universe
        import pandas as pd

        # Load universe and get top signals
        symbols = load_universe(str(universe), cap=min(cap, 200))  # Cap at 200 for speed

        scanner = DualStrategyScanner(DualStrategyParams())
        equity_signals = []
        all_price_data = []

        for sym in symbols[:100]:  # Scan top 100 for options conversion
            try:
                df = fetch_daily_bars_polygon(sym, start='2024-06-01', cache_dir=None)
                if df is None or len(df) < 50:
                    continue
                df['symbol'] = sym
                all_price_data.append(df)

                signals = scanner.generate_signals(df)
                if not signals.empty:
                    for _, sig in signals.iterrows():
                        equity_signals.append(sig.to_dict())
            except Exception:
                continue

        if not equity_signals:
            jlog('options_scan_no_equity_signals', level='INFO')
            _last_options_scan = now_et()
            return 0

        equity_df = pd.DataFrame(equity_signals)
        price_df = pd.concat(all_price_data, ignore_index=True) if all_price_data else pd.DataFrame()

        # Sort by conf_score and take top 5
        if 'conf_score' in equity_df.columns:
            equity_df = equity_df.sort_values('conf_score', ascending=False).head(5)
        else:
            equity_df = equity_df.head(5)

        # Generate options signals (CALL + PUT for each)
        options_signals = generate_options_signals(
            equity_df,
            price_df,
            max_signals=10,  # 5 calls + 5 puts
            target_delta=0.30,
            target_dte=21,
        )

        _last_options_scan = now_et()

        if options_signals.empty:
            jlog('options_scan_complete', signals_found=0)
            return 0

        signal_count = len(options_signals)
        jlog('options_scan_complete', signals_found=signal_count)

        # Save to state file
        state_dir = ROOT / 'state' / 'watchlist'
        state_dir.mkdir(parents=True, exist_ok=True)
        options_file = state_dir / 'options_signals.json'

        signals_list = options_signals.to_dict(orient='records')
        with open(options_file, 'w') as f:
            json.dump({
                'scan_time': _last_options_scan.isoformat(),
                'signals_count': signal_count,
                'signals': signals_list,
            }, f, indent=2, default=str)

        jlog('options_scan_saved', file=str(options_file), signals=signal_count)

        # Send alert if signals found
        if signal_count > 0:
            try:
                now2 = now_et()
                stamp2 = f"{fmt_ct(now2)} | {now2.strftime('%I:%M %p').lstrip('0')} ET"
                # Show top 2 calls and top 2 puts
                calls = [s for s in signals_list if s.get('option_type') == 'CALL'][:2]
                puts = [s for s in signals_list if s.get('option_type') == 'PUT'][:2]
                call_syms = ', '.join([c['symbol'] for c in calls]) if calls else 'none'
                put_syms = ', '.join([p['symbol'] for p in puts]) if puts else 'none'
                send_telegram(f"Kobe OPTIONS: {signal_count} signals - CALLS: {call_syms} | PUTS: {put_syms} [{stamp2}]")
            except Exception:
                pass

        return signal_count

    except ImportError as e:
        jlog('options_scan_import_error', level='ERROR', error=str(e))
        return -1
    except Exception as e:
        jlog('options_scan_error', level='ERROR', error=str(e))
        return -1


def should_run_options_scan(cadence_hours: int = 2) -> bool:
    """Check if it's time to run options scan."""
    global _last_options_scan

    if _last_options_scan is None:
        return True

    now = now_et()
    hours_since_last = (now - _last_options_scan).total_seconds() / 3600

    return hours_since_last >= cadence_hours


def run_unified_multi_asset_scan(dotenv: Path, universe: Path, cap: int, mode: str = 'paper') -> int:
    """
    Run UNIFIED multi-asset scan - equities + crypto + options in ONE pass.

    This is the professional quant approach:
    1. Scan 900 equities
    2. Scan 8 crypto pairs
    3. Generate options from top equity signals
    4. Rank ALL signals together by conf_score/EV
    5. Pick Top 5 to watch, Top 2 to trade
    6. Execute trades for the Top 2 (regardless of asset class)

    Returns: number of trades executed, or -1 on error
    """
    # Check kill switch
    if is_kill_switch_active():
        info = get_kill_switch_info()
        jlog('unified_scan_blocked_by_kill_switch', level='WARNING',
             reason=info.get('reason') if info else 'Unknown')
        return -1

    jlog('unified_scan_start', mode=mode)

    try:
        # Import unified scanner
        from scripts.unified_multi_asset_scan import scan_all_assets

        # This runs the full unified scan and saves to state/watchlist/multi_asset_scan.json
        scan_all_assets()

        # Load results
        results_file = ROOT / 'state' / 'watchlist' / 'multi_asset_scan.json'
        if not results_file.exists():
            jlog('unified_scan_no_results', level='WARNING')
            return 0

        with open(results_file) as f:
            results = json.load(f)

        top2 = results.get('top2', [])
        jlog('unified_scan_complete',
             total_signals=results.get('total_signals', 0),
             top2_count=len(top2),
             by_class=results.get('by_asset_class', {}))

        if not top2:
            jlog('unified_scan_no_signals', level='INFO')
            return 0

        # Send alert
        try:
            now2 = now_et()
            stamp2 = f"{fmt_ct(now2)} | {now2.strftime('%I:%M %p').lstrip('0')} ET"
            top2_symbols = ', '.join([t['symbol'] for t in top2])
            by_class = results.get('by_asset_class', {})
            send_telegram(
                f"Kobe UNIFIED SCAN: {results.get('total_signals', 0)} signals "
                f"(E:{by_class.get('equity', 0)} C:{by_class.get('crypto', 0)} O:{by_class.get('options', 0)})\n"
                f"Top 2: {top2_symbols} [{stamp2}]"
            )
        except Exception:
            pass

        # For now, unified scan is read-only (generates signals, doesn't execute)
        # TODO: Add execution logic for Top 2 trades across asset classes
        jlog('incomplete_feature_warning', feature='unified_multi_asset_execution', status='not_implemented')
        return len(top2)

    except ImportError as e:
        jlog('unified_scan_import_error', level='ERROR', error=str(e))
        return -1
    except Exception as e:
        jlog('unified_scan_error', level='ERROR', error=str(e))
        return -1


def run_daily_cycle_script(script_name: str, universe: Path, cap: int, dotenv: Path) -> int:
    """
    Run a daily cycle script (overnight watchlist, premarket validator, opening range observer).

    Returns: returncode from subprocess
    """
    # Check kill switch before running
    if is_kill_switch_active():
        info = get_kill_switch_info()
        jlog('daily_cycle_blocked_by_kill_switch', level='WARNING',
             script=script_name, reason=info.get('reason') if info else 'Unknown')
        return -1

    script_map = {
        'overnight_watchlist': ROOT / 'scripts' / 'overnight_watchlist.py',
        'premarket_validator': ROOT / 'scripts' / 'premarket_validator.py',
        'opening_range_1': ROOT / 'scripts' / 'opening_range_observer.py',
        'opening_range_2': ROOT / 'scripts' / 'opening_range_observer.py',
        'cache_refresh_eod': ROOT / 'scripts' / 'refresh_polygon_cache.py',
    }

    script_path = script_map.get(script_name)
    if not script_path or not script_path.exists():
        jlog('daily_cycle_script_not_found', level='WARNING', script=script_name)
        return -1

    # Build command based on script
    cmd = [sys.executable, str(script_path)]

    if script_name == 'overnight_watchlist':
        cmd.extend(['--cap', str(cap)])
    elif script_name == 'premarket_validator':
        pass  # No special args
    elif script_name.startswith('opening_range'):
        pass  # No special args
    elif script_name == 'cache_refresh_eod':
        # Refresh cache for all 900 symbols with latest EOD data
        cmd.extend(['--universe', str(universe)])

    if dotenv.exists():
        cmd.extend(['--dotenv', str(dotenv)])

    # Longer timeouts for scripts that process 900 stocks
    if script_name == 'cache_refresh_eod':
        timeout = 1800  # 30 min - refreshes 900 symbols
    elif script_name == 'overnight_watchlist':
        timeout = 600   # 10 min - scans 900 stocks
    else:
        timeout = 300   # 5 min default

    jlog('daily_cycle_execute', script=script_name, cmd=' '.join(cmd), timeout=timeout)

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.stdout:
            print(f"[{script_name}] {p.stdout}")
        if p.stderr:
            print(f"[{script_name}] {p.stderr}", file=sys.stderr)

        jlog('daily_cycle_complete', script=script_name, returncode=p.returncode)
        return p.returncode

    except subprocess.TimeoutExpired:
        jlog('daily_cycle_timeout', level='ERROR', script=script_name, timeout=timeout)
        return -1
    except Exception as e:
        jlog('daily_cycle_error', level='ERROR', script=script_name, error=str(e))
        return -1


def run_submit(mode: str, universe: Path, cap: int, start_days: int, dotenv: Path, scan_time: dtime = None) -> int:
    """
    Run trading script with appropriate flags based on scan time.

    Scan Time Strategy:
    - 10:00 AM: --watchlist-only (trade from validated watchlist only)
    - 10:30 AM: --fallback-enabled (fallback to full scan if watchlist fails)
    - 14:30 PM: Normal scan (power hour)
    """
    # Check kill switch before submission
    if is_kill_switch_active():
        info = get_kill_switch_info()
        jlog('runner_blocked_by_kill_switch', level='WARNING',
             reason=info.get('reason') if info else 'Unknown')
        return -1

    end = datetime.utcnow().date().isoformat()
    start = (datetime.utcnow().date() - timedelta(days=start_days)).isoformat()
    base_cmd = [sys.executable]
    if mode == 'paper':
        script = ROOT / 'scripts' / 'run_paper_trade.py'
    else:
        script = ROOT / 'scripts' / 'run_live_trade_micro.py'

    cmd = [*base_cmd, str(script), '--universe', str(universe), '--start', start, '--end', end, '--cap', str(cap), '--dotenv', str(dotenv)]

    # Add flags based on scan time (Professional Execution Flow)
    if scan_time:
        if scan_time.hour == 10 and scan_time.minute == 0:
            # 10:00 AM - Primary window: Trade from validated watchlist only
            cmd.append('--watchlist-only')
            jlog('runner_using_watchlist_only', scan_time='10:00')
        elif scan_time.hour == 10 and scan_time.minute == 30:
            # 10:30 AM - Fallback window: If watchlist fails, scan full universe with higher bar
            cmd.append('--fallback-enabled')
            jlog('runner_using_fallback', scan_time='10:30')
        # 14:30 PM - Power hour: Normal scan (no special flags)

    jlog('runner_execute', mode=mode, script=str(script), universe=str(universe), start=start, end=end, cap=cap, scan_time=scan_time.strftime('%H:%M') if scan_time else None)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout:
        print(p.stdout)
    if p.stderr:
        print(p.stderr, file=sys.stderr)
    return p.returncode


def main():
    global _lock, _heartbeat, _shutdown_requested

    ap = argparse.ArgumentParser(description='Kobe 24/7 Multi-Asset Runner')
    ap.add_argument('--mode', type=str, choices=['paper','live'], default='paper')
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--cap', type=int, default=50)
    ap.add_argument('--scan-times', type=str, default='10:00,10:30,14:30', help='Local HH:MM times (ET), comma separated. Default respects kill zones.')
    ap.add_argument('--lookback-days', type=int, default=540)
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--once', action='store_true', help='Run once immediately and exit')
    ap.add_argument('--dry-run', action='store_true', help='Dry run - produce artifacts without placing orders')
    ap.add_argument('--skip-reconcile', action='store_true', help='Skip position reconciliation on startup')
    ap.add_argument('--skip-lock', action='store_true', help='Skip single-instance lock (for debugging)')
    # Live trading safety
    ap.add_argument('--approve-live', action='store_true',
                    help='Explicitly approve live trading (ALSO requires LIVE_TRADING_APPROVED=YES env)')
    # Multi-asset options
    ap.add_argument('--enable-crypto', action='store_true', help='Enable crypto scanning (24/7)')
    ap.add_argument('--crypto-cadence', type=int, default=4, help='Crypto scan cadence in hours')
    ap.add_argument('--crypto-universe', type=str, help='Crypto universe file (optional)')
    ap.add_argument('--enable-options', action='store_true', help='Enable options event scanning')
    ap.add_argument('--options-cadence', type=int, default=2, help='Options scan cadence in hours (default: 2)')
    # Unified multi-asset scanning (recommended - runs all asset classes together)
    ap.add_argument('--unified', action='store_true',
                    help='Use unified multi-asset scanning (equities + crypto + options in one scan)')
    args = ap.parse_args()

    # CRITICAL: Live trading safety check
    if args.mode == 'live':
        approved, reason = check_live_trading_approved()
        if not approved:
            jlog('live_trading_blocked', level='CRITICAL', reason=reason)
            print(f"BLOCKED: {reason}")
            sys.exit(1)
        if not args.approve_live:
            jlog('live_trading_blocked', level='CRITICAL', reason='--approve-live flag not provided')
            print("BLOCKED: Live trading requires --approve-live flag")
            print("This is a safety measure to prevent accidental live trading.")
            sys.exit(1)
        jlog('live_trading_approved', level='WARNING')

    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # CRITICAL: Validate strategy imports at startup
    # This ensures we NEVER use deprecated standalone strategies
    try:
        from strategies.registry import validate_strategy_import, assert_no_deprecated_strategies
        validate_strategy_import()  # Warn about any bad imports
        jlog('strategy_validation', level='INFO', status='passed', message='Using correct DualStrategyScanner')
    except ImportError:
        jlog('strategy_validation', level='WARNING', message='Registry not available, skipping validation')

    universe = Path(args.universe)
    times = parse_times(args.scan_times)

    # Setup signal handlers for graceful shutdown
    _setup_signal_handlers()

    # Single-instance lock (unless skipped)
    if not args.skip_lock:
        if is_another_instance_running(LOCK_FILE):
            jlog('runner_already_running', level='ERROR')
            print("ERROR: Another runner instance is already running.")
            print(f"Lock file: {LOCK_FILE}")
            sys.exit(1)

        try:
            _lock = FileLock(LOCK_FILE)
            _lock.acquire(blocking=False)
            jlog('runner_lock_acquired', lock_file=str(LOCK_FILE))
        except LockError as e:
            jlog('runner_lock_failed', level='ERROR', error=str(e))
            print(f"ERROR: Could not acquire lock: {e}")
            sys.exit(1)

    # Start heartbeat writer
    _heartbeat = HeartbeatWriter(
        heartbeat_path=ROOT / 'state' / 'heartbeat.json',
        interval=60,
        mode=args.mode,
    )
    _heartbeat.start()

    # Start health server (if enabled in config/base.yaml)
    try:
        from config.settings_loader import get_setting
        if bool(get_setting('health.enabled', True)):
            port = int(get_setting('health.port', 8081))
            start_health_server(port)
            jlog('health_server_started', port=port)
    except Exception as e:
        jlog('health_server_start_failed', level='ERROR', error=str(e))

    # Position reconciliation on startup (enhanced: detects AND fixes discrepancies)
    if not args.skip_reconcile:
        jlog('runner_startup_reconcile', level='INFO')
        _heartbeat.update("reconciling_positions")
        reconcile_result = reconcile_and_fix(dotenv, auto_fix=True)
        if reconcile_result.get('discrepancies'):
            jlog('runner_reconcile_discrepancies', level='WARNING',
                 discrepancies=reconcile_result['discrepancies'])
            # Log fixes applied
            if reconcile_result.get('fixes_applied'):
                jlog('runner_reconcile_fixes', level='INFO',
                     fixes=reconcile_result['fixes_applied'])
        if not reconcile_result.get('success'):
            msg = f"Kobe reconcile failed: {reconcile_result.get('error','unknown')}"
            try:
                now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                msg = f"{msg} [{stamp}]"
            except Exception:
                pass
            send_telegram(msg)

        # FIX (2026-01-06): Gap #5 - Exit manager catch-up for missed exits
        if EXIT_MANAGER_AVAILABLE:
            jlog('runner_startup_exit_catchup', level='INFO')
            _heartbeat.update("catching_up_exits")
            catchup_result = catch_up_missed_exits(dotenv_path=str(dotenv), execute=True)
            if catchup_result.get('exits_executed'):
                jlog('runner_startup_exits_executed', level='INFO',
                     symbols=catchup_result['exits_executed'])
            if not catchup_result.get('success'):
                jlog('runner_startup_exit_catchup_failed', level='WARNING',
                     errors=catchup_result.get('errors'))

        # Check performance drift on startup
        if DRIFT_DETECTION_AVAILABLE:
            jlog('runner_startup_drift_check', level='INFO')
            _heartbeat.update("checking_drift")
            drift_result = check_performance_drift()
            if drift_result.get('has_drift'):
                jlog('runner_startup_drift_detected', level='WARNING',
                     severity=drift_result.get('severity'),
                     position_scale=drift_result.get('position_scale'),
                     message=drift_result.get('message'))
                try:
                    now = now_et()
                    stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                    msg = f"Kobe startup drift check: {drift_result.get('severity')} - scale={drift_result.get('position_scale')} [{stamp}]"
                    send_telegram(msg)
                except Exception:
                    pass

    # Check kill switch on startup
    if is_kill_switch_active():
        info = get_kill_switch_info()
        jlog('runner_kill_switch_active', level='CRITICAL',
             reason=info.get('reason') if info else 'Unknown')
        print("ERROR: Kill switch is active. Trading halted.")
        try:
            now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
            send_telegram(f"Kobe halted by kill switch: {info.get('reason') if info else 'Unknown'} [{stamp}]")
        except Exception:
            send_telegram(f"Kobe halted by kill switch: {info.get('reason') if info else 'Unknown'}")
        print(f"Reason: {info.get('reason') if info else 'Unknown'}")
        print("Use /resume skill to deactivate after investigation.")
        _cleanup()
        return

    if args.once:
        _heartbeat.update("running_once")
        run_submit(args.mode, universe, args.cap, args.lookback_days, dotenv)
        _cleanup()
        return

    jlog('runner_start', mode=args.mode, scan_times=args.scan_times, universe=str(universe))
    _heartbeat.update("started")
    last_reconcile_date = now_et().date()

    while not _shutdown_requested:
        now = now_et()

        # Update heartbeat
        _heartbeat.update("monitoring_" + fmt_ct(now, include_tz=False))

        # Touch lock to prevent stale detection
        if _lock:
            _lock.touch()

        # Check kill switch periodically
        if is_kill_switch_active():
            info = get_kill_switch_info()
            jlog('runner_halted_by_kill_switch', level='WARNING',
                 reason=info.get('reason') if info else 'Unknown')
            _heartbeat.update("halted_by_kill_switch")
            time.sleep(60)  # Check again in 1 minute
            continue

        # Daily reconciliation (run once per day at start) - enhanced with auto-fix
        if now.date() != last_reconcile_date and within_market_day(now):
            jlog('runner_daily_reconcile', level='INFO')
            _heartbeat.update("daily_reconcile")
            reconcile_and_fix(dotenv, auto_fix=True)

            # Check performance drift after reconciliation
            if DRIFT_DETECTION_AVAILABLE:
                _heartbeat.update("checking_drift")
                drift_result = check_performance_drift()
                if drift_result.get('has_drift'):
                    jlog('runner_drift_detected', level='WARNING',
                         severity=drift_result.get('severity'),
                         position_scale=drift_result.get('position_scale'),
                         message=drift_result.get('message'))

                    # Critical drift should halt trading
                    if drift_result.get('severity') == 'CRITICAL':
                        global _consecutive_critical_drift_count
                        _consecutive_critical_drift_count += 1
                        jlog('runner_critical_drift_halt', level='CRITICAL',
                             message='Critical drift detected',
                             consecutive_count=_consecutive_critical_drift_count,
                             threshold=DRIFT_ESCALATION_THRESHOLD)

                        # Escalate to kill-switch after N consecutive critical drifts
                        if _consecutive_critical_drift_count >= DRIFT_ESCALATION_THRESHOLD:
                            reason = (
                                f"AUTO-ESCALATION: {_consecutive_critical_drift_count} consecutive CRITICAL drift detections. "
                                f"Message: {drift_result.get('message', 'Unknown')}"
                            )
                            jlog('runner_drift_kill_switch_activated', level='CRITICAL',
                                 reason=reason, consecutive_count=_consecutive_critical_drift_count)
                            try:
                                now2 = now_et(); stamp2 = f"{fmt_ct(now2)} | {now2.strftime('%I:%M %p').lstrip('0')} ET"
                                send_telegram(
                                    f"KILL SWITCH ACTIVATED - DRIFT ESCALATION\n"
                                    f"{_consecutive_critical_drift_count} consecutive critical drifts detected\n"
                                    f"{drift_result.get('message', 'Unknown')}\n"
                                    f"[{stamp2}]"
                                )
                            except Exception:
                                pass
                            activate_kill_switch(reason)
                    else:
                        # Reset counter on non-critical drift
                        _consecutive_critical_drift_count = 0
                else:
                    # No drift detected - reset counter
                    _consecutive_critical_drift_count = 0

            last_reconcile_date = now.date()

        # ==================================================================
        # DAILY CYCLE SCRIPTS (Professional Execution Flow)
        # ==================================================================
        # These run at specific times regardless of trading (except overnight)
        today_str = now.date().isoformat()

        # Premarket Validator (8:00 AM) - Run on market days
        if within_market_day(now):
            premarket_tag = 'premarket_validator'
            premarket_time = DAILY_CYCLE_TIMES['premarket_validator']
            premarket_dt = datetime.combine(now.date(), premarket_time, tzinfo=ET)
            if now >= premarket_dt and not already_ran(premarket_tag, today_str):
                _heartbeat.update(f"running_{premarket_tag}")
                rc = run_daily_cycle_script(premarket_tag, universe, args.cap, dotenv)
                mark_ran(premarket_tag, today_str)
                jlog('daily_cycle_done', script=premarket_tag, returncode=rc)

        # Opening Range Observer (9:30 AM, 9:45 AM) - Run on market days
        if within_market_day(now):
            for obs_tag in ['opening_range_1', 'opening_range_2']:
                if _shutdown_requested:
                    break
                obs_time = DAILY_CYCLE_TIMES[obs_tag]
                obs_dt = datetime.combine(now.date(), obs_time, tzinfo=ET)
                if now >= obs_dt and not already_ran(obs_tag, today_str):
                    _heartbeat.update(f"running_{obs_tag}")
                    rc = run_daily_cycle_script(obs_tag, universe, args.cap, dotenv)
                    mark_ran(obs_tag, today_str)
                    jlog('daily_cycle_done', script=obs_tag, returncode=rc)

        # Overnight Watchlist (3:30 PM) - Run on market days to build next day's watchlist
        if within_market_day(now):
            overnight_tag = 'overnight_watchlist'
            overnight_time = DAILY_CYCLE_TIMES['overnight_watchlist']
            overnight_dt = datetime.combine(now.date(), overnight_time, tzinfo=ET)
            if now >= overnight_dt and not already_ran(overnight_tag, today_str):
                _heartbeat.update(f"running_{overnight_tag}")
                rc = run_daily_cycle_script(overnight_tag, universe, args.cap, dotenv)
                mark_ran(overnight_tag, today_str)
                jlog('daily_cycle_done', script=overnight_tag, returncode=rc)
                try:
                    now2 = now_et()
                    stamp2 = f"{fmt_ct(now2)} | {now2.strftime('%I:%M %p').lstrip('0')} ET"
                    send_telegram(f"Kobe overnight watchlist built for tomorrow [{stamp2}]")
                except Exception:
                    pass

        # ==================================================================
        # TRADING EXECUTION (Scan + Trade)
        # ==================================================================
        if within_market_day(now):
            # Respect early close: skip schedule items after close
            _open, early_close = get_market_hours(now)
            for t in times:
                if _shutdown_requested:
                    break
                tag = f"{args.mode}_{t.strftime('%H%M')}"
                target_dt = datetime.combine(now.date(), t, tzinfo=ET)
                if early_close and t > early_close:
                    continue
                if now >= target_dt and not already_ran(tag, today_str):
                    _heartbeat.update(f"running_{tag}")

                    # UNIFIED MODE: Scan all asset classes together
                    if getattr(args, 'unified', False):
                        jlog('runner_unified_scan_start', tag=tag)
                        rc = run_unified_multi_asset_scan(dotenv, universe, args.cap, args.mode)
                        scan_type = 'unified'
                    else:
                        # LEGACY MODE: Scan equities only at scheduled times
                        rc = run_submit(args.mode, universe, args.cap, args.lookback_days, dotenv, scan_time=t)
                        scan_type = 'equity'

                    update_request_counter('total', 1)
                    mark_ran(tag, today_str)
                    jlog('runner_done', mode=args.mode, schedule=tag, scan_type=scan_type, returncode=rc)
                    try:
                        now2 = now_et(); stamp2 = f"{fmt_ct(now2)} | {now2.strftime('%I:%M %p').lstrip('0')} ET"
                        send_telegram(f"Kobe {scan_type} run {tag} completed rc={rc} [{stamp2}]")
                    except Exception:
                        send_telegram(f"Kobe {scan_type} run {tag} completed rc={rc}")

        # ==================================================================
        # POSITION MONITOR (Time-Based Exits)
        # ==================================================================
        # Run every 15 minutes during market hours to check:
        # - 3-bar time stop for Turtle Soup
        # - 7-bar time stop for IBS+RSI
        # - Trailing stop updates
        if within_market_day(now) and should_run_position_monitor():
            # Only run during trading hours (10:00 AM - 4:00 PM)
            market_open_dt = datetime.combine(now.date(), dtime(10, 0), tzinfo=ET)
            market_close_dt = datetime.combine(now.date(), dtime(16, 0), tzinfo=ET)
            if market_open_dt <= now <= market_close_dt:
                _heartbeat.update("running_position_monitor")
                rc = run_position_monitor(dotenv)
                if rc == 0:
                    jlog('position_monitor_done', returncode=rc)

        # ==================================================================
        # CRYPTO SCAN (24/7 - Independent of Market Hours)
        # ==================================================================
        # Run every N hours (default 4) regardless of time/day
        # Crypto markets never close
        # SKIP if unified mode is enabled (unified handles crypto)
        if not getattr(args, 'unified', False):
            if args.enable_crypto and should_run_crypto_scan(args.crypto_cadence):
                _heartbeat.update("running_crypto_scan")
                crypto_result = run_crypto_scan(dotenv, args.crypto_cadence)
                jlog('crypto_scan_done', signals_found=crypto_result)

        # ==================================================================
        # OPTIONS SCAN (Market Hours Only - Event Driven)
        # ==================================================================
        # Run every N hours (default 2) during market hours
        # Generates CALL/PUT signals from top equity signals
        # SKIP if unified mode is enabled (unified handles options)
        if not getattr(args, 'unified', False):
            if args.enable_options and should_run_options_scan(args.options_cadence):
                # Only run during market hours (options don't trade after hours)
                closed, _ = is_market_closed(now)
                if not closed:
                    _heartbeat.update("running_options_scan")
                    options_result = run_options_scan(dotenv, universe, args.cap, args.options_cadence)
                    jlog('options_scan_done', signals_found=options_result)

        # Sleep in small intervals for faster shutdown response
        for _ in range(30):
            if _shutdown_requested:
                break
            time.sleep(1)

    # Graceful shutdown
    jlog('runner_shutdown', level='INFO')
    _cleanup()


if __name__ == '__main__':
    main()
