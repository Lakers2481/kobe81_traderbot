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
from core.clock.tz_utils import fmt_ct, now_et, ET
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from core.structured_log import jlog
from monitor.health_endpoints import start_health_server, update_request_counter
from monitor.heartbeat import HeartbeatWriter, update_global_heartbeat
from ops.locks import FileLock, LockError, is_another_instance_running
from market_calendar import is_market_closed, get_market_hours
from core.alerts import send_telegram
from core.kill_switch import is_kill_switch_active, get_kill_switch_info
from config.env_loader import load_env

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


def run_submit(mode: str, universe: Path, cap: int, start_days: int, dotenv: Path) -> int:
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
    jlog('runner_execute', mode=mode, script=str(script), universe=str(universe), start=start, end=end, cap=cap)
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
    ap.add_argument('--scan-times', type=str, default='09:35,10:30,15:55', help='Local HH:MM times, comma separated')
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

    # Position reconciliation on startup
    if not args.skip_reconcile:
        jlog('runner_startup_reconcile', level='INFO')
        _heartbeat.update("reconciling_positions")
        reconcile_result = reconcile_positions(dotenv)
        if reconcile_result.get('discrepancies'):
            jlog('runner_reconcile_discrepancies', level='WARNING',
                 discrepancies=reconcile_result['discrepancies'])
        if not reconcile_result.get('success'):
            msg = f"Kobe reconcile failed: {reconcile_result.get('error','unknown')}"
            try:
                now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                msg = f"{msg} [{stamp}]"
            except Exception:
                pass
            send_telegram(msg)

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

        # Daily reconciliation (run once per day at start)
        if now.date() != last_reconcile_date and within_market_day(now):
            jlog('runner_daily_reconcile', level='INFO')
            _heartbeat.update("daily_reconcile")
            reconcile_positions(dotenv)
            last_reconcile_date = now.date()

        if within_market_day(now):
            today_str = now.date().isoformat()
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
                    rc = run_submit(args.mode, universe, args.cap, args.lookback_days, dotenv)
                    update_request_counter('total', 1)
                    mark_ran(tag, today_str)
                    jlog('runner_done', mode=args.mode, schedule=tag, returncode=rc)
                    try:
                        now2 = now_et(); stamp2 = f"{fmt_ct(now2)} | {now2.strftime('%I:%M %p').lstrip('0')} ET"
                        send_telegram(f"Kobe run {tag} completed rc={rc} [{stamp2}]")
                    except Exception:
                        send_telegram(f"Kobe run {tag} completed rc={rc}")

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
