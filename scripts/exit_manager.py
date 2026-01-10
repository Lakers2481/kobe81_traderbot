#!/usr/bin/env python3
"""
Kobe Exit Manager - Time-Based Position Exit Monitoring
=========================================================

Monitors open positions and enforces time-based exit rules:
- IBS+RSI: 7-bar time stop (exit after 7 trading days)
- Turtle Soup: 3-bar time stop (faster mean reversion)

This script should run every 30 minutes during market hours via Task Scheduler.

Usage:
    python scripts/exit_manager.py --check-time-exits
    python scripts/exit_manager.py --check-time-exits --execute
    python scripts/exit_manager.py --check-time-exits --execute --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio


# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from core.structured_log import jlog
from core.alerts import send_telegram
from core.clock.tz_utils import now_et, fmt_ct
from portfolio.state_manager import get_state_manager
from integration.learning_hub import get_learning_hub, TradeOutcomeEvent

# Time stop rules per strategy
TIME_STOP_BARS = {
    'IBS_RSI': 7,       # Exit after 7 trading days
    'TURTLESOUP': 3,    # Exit after 3 trading days (faster)
    'DEFAULT': 5,       # Default if strategy unknown
}

# Legacy file (kept for migration)
POSITION_STATE_FILE = ROOT / 'state' / 'position_tracker.json'
# FIX (2026-01-08): Use trades.jsonl (broker_alpaca.py logs) not trade_history.jsonl
TRADE_LOG_FILE = ROOT / 'logs' / 'trades.jsonl'


def load_position_state() -> Dict[str, Any]:
    """
    Load position tracking state using StateManager (atomic, locked).

    FIX (2026-01-06): Gap #2 - Now uses StateManager for atomic access.
    """
    sm = get_state_manager()

    # First try StateManager (new location)
    positions = sm.get_positions()
    if positions:
        return {'positions': positions, 'last_sync': None}

    # Migration: If legacy file exists, migrate it
    if POSITION_STATE_FILE.exists():
        try:
            legacy_data = json.loads(POSITION_STATE_FILE.read_text())
            legacy_positions = legacy_data.get('positions', {})
            if legacy_positions:
                # Migrate to StateManager
                sm.set_positions(legacy_positions)
                jlog('position_state_migrated', count=len(legacy_positions))
                # Rename legacy file
                POSITION_STATE_FILE.rename(POSITION_STATE_FILE.with_suffix('.json.bak'))
            return {'positions': legacy_positions}
        except Exception as e:
            jlog('position_state_migration_error', error=str(e), level='WARN')

    return {'positions': {}}


def save_position_state(state: Dict[str, Any]) -> None:
    """
    Save position tracking state using StateManager (atomic, locked).

    FIX (2026-01-06): Gap #2 - Atomic writes with file locking.
    """
    sm = get_state_manager()
    positions = state.get('positions', {})
    sm.set_positions(positions)


def get_broker_positions() -> List[Dict]:
    """Get current positions from Alpaca broker."""
    import requests

    base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
    key = os.getenv('ALPACA_API_KEY_ID', '')
    sec = os.getenv('ALPACA_API_SECRET_KEY', '')

    if not key or not sec:
        jlog('exit_manager_no_credentials', level='ERROR')
        return []

    headers = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': sec}

    try:
        resp = requests.get(f"{base}/v2/positions", headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        jlog('exit_manager_positions_error', error=str(e), level='ERROR')
        return []


def close_position(symbol: str, qty: int, side: str = 'sell', dry_run: bool = False) -> Dict:
    """
    Close a position via market order.

    Args:
        symbol: Stock symbol
        qty: Number of shares (positive)
        side: 'sell' for long positions, 'buy' for short positions
        dry_run: If True, don't actually place the order

    Returns:
        Order result dictionary
    """
    import requests

    if dry_run:
        jlog('exit_manager_dry_run', symbol=symbol, qty=qty, side=side)
        return {'status': 'DRY_RUN', 'symbol': symbol, 'qty': qty}

    base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
    key = os.getenv('ALPACA_API_KEY_ID', '')
    sec = os.getenv('ALPACA_API_SECRET_KEY', '')

    headers = {
        'APCA-API-KEY-ID': key,
        'APCA-API-SECRET-KEY': sec,
        'Content-Type': 'application/json'
    }

    # Use Alpaca's close position endpoint (cleaner)
    try:
        resp = requests.delete(
            f"{base}/v2/positions/{symbol}",
            headers=headers,
            timeout=10
        )

        if resp.status_code == 200:
            result = resp.json()
            jlog('exit_manager_position_closed', symbol=symbol,
                 qty=qty, order_id=result.get('id'))
            return {'status': 'CLOSED', 'symbol': symbol, 'order': result}
        else:
            jlog('exit_manager_close_failed', symbol=symbol,
                 status=resp.status_code, response=resp.text, level='ERROR')
            return {'status': 'FAILED', 'symbol': symbol, 'error': resp.text}

    except Exception as e:
        jlog('exit_manager_close_error', symbol=symbol, error=str(e), level='ERROR')
        return {'status': 'ERROR', 'symbol': symbol, 'error': str(e)}


def get_trading_days_since(entry_date: datetime) -> int:
    """
    Calculate number of trading days since entry.

    FIX (2026-01-08): Proper NYSE calendar-based calculation.
    Uses NYSE calendar to count actual trading days, excluding weekends and holidays.
    """
    today = now_et().date()
    entry_date_obj = entry_date.date() if hasattr(entry_date, 'date') else entry_date

    try:
        # Try to use market_calendar module (same directory)
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from market_calendar import get_trading_days_between, is_market_closed
        trading_days = get_trading_days_between(entry_date_obj, today)

        # Debug logging for bars counter verification
        jlog('bars_counter_calculation', level='DEBUG',
             entry_date=str(entry_date_obj),
             today=str(today),
             trading_days_held=trading_days,
             calendar_days=(today - entry_date_obj).days)

        return trading_days

    except Exception as e:
        # Fallback: Better estimate that counts actual days excluding weekends
        from datetime import timedelta
        count = 0
        current = entry_date_obj + timedelta(days=1)

        while current <= today:
            # Skip weekends (5=Saturday, 6=Sunday)
            if current.weekday() < 5:
                count += 1
            current += timedelta(days=1)

        jlog('bars_counter_fallback', level='WARNING',
             entry_date=str(entry_date_obj),
             today=str(today),
             trading_days_estimate=count,
             reason=str(e))

        return count


def sync_positions_with_broker(state: Dict[str, Any], broker_positions: List[Dict]) -> Dict[str, Any]:
    """
    Sync local position state with broker positions.

    - Remove positions that are no longer open
    - Add new positions with current timestamp
    """
    broker_symbols = {p['symbol'] for p in broker_positions}
    local_positions = state.get('positions', {})

    # Remove closed positions
    closed = [sym for sym in local_positions if sym not in broker_symbols]
    for sym in closed:
        jlog('exit_manager_position_removed', symbol=sym, reason='no_longer_open')
        del local_positions[sym]

    # Add new positions
    for pos in broker_positions:
        sym = pos['symbol']
        if sym not in local_positions:
            # New position - record entry time
            local_positions[sym] = {
                'symbol': sym,
                'entry_date': now_et().isoformat(),
                'entry_price': float(pos.get('avg_entry_price', 0)),
                'qty': int(float(pos.get('qty', 0))),
                'side': 'long' if float(pos.get('qty', 0)) > 0 else 'short',
                'strategy': 'UNKNOWN',  # Will be updated from trade log
            }
            jlog('exit_manager_position_added', symbol=sym,
                 entry_date=local_positions[sym]['entry_date'])

    # Try to enrich with strategy info from trade log
    if TRADE_LOG_FILE.exists():
        try:
            trades = []
            with open(TRADE_LOG_FILE, 'r') as f:
                for line in f:
                    try:
                        trades.append(json.loads(line))
                    except (json.JSONDecodeError, ValueError):
                        continue

            # Find most recent entry for each symbol
            for sym in local_positions:
                if local_positions[sym].get('strategy') == 'UNKNOWN':
                    for trade in reversed(trades):
                        if trade.get('symbol') == sym and trade.get('action') == 'ENTRY':
                            local_positions[sym]['strategy'] = trade.get('strategy', 'UNKNOWN')
                            break
        except Exception as e:
            jlog('exit_manager_trade_log_error', error=str(e), level='WARN')

    state['positions'] = local_positions
    state['last_sync'] = now_et().isoformat()
    return state


def check_time_exits(state: Dict[str, Any], execute: bool = False, dry_run: bool = False) -> List[Dict]:
    """
    Check all positions for time-based exit conditions.

    Args:
        state: Position tracking state
        execute: If True, actually close positions
        dry_run: If True with execute, simulate but don't place orders

    Returns:
        List of positions that need/triggered exits
    """
    exits_needed = []
    positions = state.get('positions', {})

    for sym, pos_info in positions.items():
        entry_date_str = pos_info.get('entry_date')
        if not entry_date_str:
            continue

        try:
            entry_date = datetime.fromisoformat(entry_date_str.replace('Z', '+00:00'))
        except ValueError:
            continue

        strategy = pos_info.get('strategy', 'DEFAULT')
        max_bars = TIME_STOP_BARS.get(strategy.upper(), TIME_STOP_BARS['DEFAULT'])

        bars_held = get_trading_days_since(entry_date)

        if bars_held >= max_bars:
            exit_info = {
                'symbol': sym,
                'strategy': strategy,
                'entry_date': entry_date_str,
                'bars_held': bars_held,
                'max_bars': max_bars,
                'qty': pos_info.get('qty', 0),
                'side': pos_info.get('side', 'long'),
                'reason': f'TIME_STOP_{max_bars}BAR',
            }
            exits_needed.append(exit_info)

            jlog('exit_manager_time_stop_triggered', **exit_info)
            print(f"TIME STOP: {sym} held {bars_held} bars (max {max_bars}) - EXIT NEEDED")

            if execute:
                # Close the position
                close_side = 'sell' if pos_info.get('side') == 'long' else 'buy'
                result = close_position(sym, abs(pos_info.get('qty', 0)), close_side, dry_run)
                exit_info['close_result'] = result

                if result.get('status') in ('CLOSED', 'DRY_RUN'):
                    # LEARNING HUB INTEGRATION (FIX 1)
                    try:
                        hub = get_learning_hub()
                        order_details = result.get('order', {})
                        
                        entry_price = pos_info.get('entry_price', 0)
                        exit_price = float(order_details.get('filled_avg_price', 0))
                        shares = abs(pos_info.get('qty', 0))
                        
                        if entry_price > 0 and exit_price > 0 and shares > 0:
                            pnl = (exit_price - entry_price) * shares
                            if pos_info.get('side') == 'short':
                                pnl = -pnl
                            pnl_pct = (pnl / (entry_price * shares))
                            
                            exit_time_str = order_details.get('filled_at')
                            exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00')) if exit_time_str else now_et()

                            trade_outcome = TradeOutcomeEvent(
                                symbol=sym,
                                side=pos_info.get('side', 'long'),
                                entry_price=entry_price,
                                exit_price=exit_price,
                                shares=shares,
                                entry_time=entry_date,
                                exit_time=exit_time,
                                pnl=pnl,
                                pnl_pct=pnl_pct,
                                won=pnl > 0,
                                pattern_type=pos_info.get('strategy', 'UNKNOWN'),
                                exit_reason='time_stop',
                                trade_id=f"trade_{sym}_{entry_date.strftime('%Y%m%d')}"
                            )
                            
                            jlog('learning_hub_feed_start', trade_id=trade_outcome.trade_id, pnl_pct=pnl_pct)
                            asyncio.run(hub.process_trade_outcome(trade_outcome))
                            jlog('learning_hub_feed_complete', trade_id=trade_outcome.trade_id)

                    except Exception as e:
                        jlog('learning_hub_integration_error', error=str(e), level='ERROR')

                    # Remove from tracking
                    if sym in state['positions']:
                        del state['positions'][sym]

                    # Send alert
                    now = now_et()
                    stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                    msg = f"TIME EXIT: {sym} closed after {bars_held} bars ({strategy}) [{stamp}]"
                    try:
                        send_telegram(msg)
                    except Exception:
                        pass
        else:
            # Log position status
            bars_remaining = max_bars - bars_held
            print(f"  {sym}: {bars_held}/{max_bars} bars ({bars_remaining} remaining)")

    return exits_needed


def catch_up_missed_exits(dotenv_path: Optional[str] = None, execute: bool = True) -> dict:
    """
    Catch-up logic for missed exits on startup/restart.

    FIX (2026-01-06): Gap #5 - Exit manager now checks for missed exits on startup.

    This function should be called when the runner starts to ensure any positions
    that exceeded their time limits during downtime are properly closed.

    Args:
        dotenv_path: Path to .env file
        execute: If True, actually close positions (default: True for catch-up)

    Returns:
        Dict with catch-up results
    """
    result = {
        'success': False,
        'positions_checked': 0,
        'exits_executed': [],
        'exits_needed': [],
        'errors': [],
    }

    try:
        # Load environment if provided
        if dotenv_path:
            dotenv = Path(dotenv_path)
            if dotenv.exists():
                load_env(dotenv)

        jlog('exit_manager_catchup_start', level='INFO')

        # Get broker positions
        broker_positions = get_broker_positions()
        if not broker_positions:
            jlog('exit_manager_catchup_no_positions', level='INFO')
            result['success'] = True
            return result

        result['positions_checked'] = len(broker_positions)

        # Load and sync state
        state = load_position_state()
        state = sync_positions_with_broker(state, broker_positions)
        save_position_state(state)

        # Check for overdue exits
        exits = check_time_exits(state, execute=execute, dry_run=False)

        # Save updated state
        save_position_state(state)

        # Record results
        result['exits_needed'] = [
            {'symbol': e['symbol'], 'bars_held': e['bars_held'], 'max_bars': e['max_bars']}
            for e in exits
        ]
        result['exits_executed'] = [
            e['symbol'] for e in exits
            if e.get('close_result', {}).get('status') == 'CLOSED'
        ]

        if result['exits_executed']:
            jlog('exit_manager_catchup_exits', level='INFO',
                 symbols=result['exits_executed'],
                 count=len(result['exits_executed']))

            # Send alert
            now = now_et()
            stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
            msg = f"EXIT MANAGER CATCH-UP: Closed {len(result['exits_executed'])} overdue positions [{stamp}]"
            for sym in result['exits_executed']:
                msg += f"\n  - {sym}"
            try:
                send_telegram(msg)
            except Exception:
                pass

        result['success'] = True
        jlog('exit_manager_catchup_complete', level='INFO',
             positions_checked=result['positions_checked'],
             exits_executed=len(result['exits_executed']))

    except Exception as e:
        jlog('exit_manager_catchup_error', level='ERROR', error=str(e))
        result['errors'].append(str(e))

    return result


def main():
    ap = argparse.ArgumentParser(description='Kobe Exit Manager - Time-based position exits')
    ap.add_argument('--check-time-exits', action='store_true', help='Check for time-based exits')
    ap.add_argument('--execute', action='store_true', help='Actually close positions (not just report)')
    ap.add_argument('--dry-run', action='store_true', help='Simulate execution without placing orders')
    ap.add_argument('--sync-only', action='store_true', help='Only sync positions, no exit checks')
    ap.add_argument('--dotenv', type=str, default='./.env', help='Path to .env file')
    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    print("=" * 60)
    print("KOBE EXIT MANAGER - Time-Based Position Monitoring")
    print("=" * 60)
    print(f"Time: {fmt_ct(now_et())} ET")

    # Load state
    state = load_position_state()

    # Get broker positions
    print("\nFetching broker positions...")
    broker_positions = get_broker_positions()
    print(f"  Found {len(broker_positions)} open positions")

    if not broker_positions:
        print("  No open positions to monitor.")
        return

    # Sync state
    print("\nSyncing position state...")
    state = sync_positions_with_broker(state, broker_positions)
    save_position_state(state)

    if args.sync_only:
        print("\nSync complete (--sync-only mode)")
        return

    # Check time exits
    if args.check_time_exits:
        print("\n" + "-" * 60)
        print("CHECKING TIME-BASED EXITS")
        print("-" * 60)
        print(f"Rules: IBS_RSI={TIME_STOP_BARS['IBS_RSI']} bars, "
              f"TURTLESOUP={TIME_STOP_BARS['TURTLESOUP']} bars, "
              f"DEFAULT={TIME_STOP_BARS['DEFAULT']} bars")
        print()

        exits = check_time_exits(state, execute=args.execute, dry_run=args.dry_run)

        # Save updated state
        save_position_state(state)

        print("\n" + "-" * 60)
        if exits:
            print(f"EXITS TRIGGERED: {len(exits)}")
            for ex in exits:
                status = ex.get('close_result', {}).get('status', 'NOT_EXECUTED')
                print(f"  {ex['symbol']}: {ex['reason']} -> {status}")
        else:
            print("No time-based exits needed.")
        print("-" * 60)

    print("\nExit manager complete.")


if __name__ == '__main__':
    main()
