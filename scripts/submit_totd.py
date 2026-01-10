#!/usr/bin/env python3
from __future__ import annotations

"""
Submit Trade of the Day (TOTD) as an IOC LIMIT order via Alpaca.

Reads logs/trade_of_day.csv produced by scripts/scan.py --top3 [--ml]
and submits the single highest-confidence trade, subject to PolicyGate.
"""

import argparse
from pathlib import Path
import math
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from config.settings_loader import load_settings as load_config
from data.providers.alpaca_live import get_market_clock
from execution.broker_alpaca import get_best_ask, get_best_bid, construct_decision, place_ioc_limit
from risk.policy_gate import PolicyGate, RiskLimits
from core.structured_log import jlog
from core.hash_chain import append_block
from core.config_pin import sha256_file
from core.clock.macro_events import MacroEventCalendar
from core.signal_freshness import validate_signal_file


def get_open_position_count() -> int:
    """Get count of open positions from Alpaca."""
    import os
    import urllib.request
    import json

    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    api_key = os.getenv('ALPACA_API_KEY_ID', '')
    api_secret = os.getenv('ALPACA_API_SECRET_KEY', '')

    if not api_key or not api_secret:
        return 0  # Can't check, allow submission

    url = f"{base_url}/v2/positions"
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': api_secret,
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            positions = json.loads(resp.read().decode('utf-8'))
            return len(positions) if isinstance(positions, list) else 0
    except Exception:
        return 0  # On error, allow submission


def main() -> None:
    ap = argparse.ArgumentParser(description='Submit Trade of the Day (IOC LIMIT)')
    ap.add_argument('--totd', type=str, default='logs/trade_of_day.csv', help='Path to trade_of_day.csv')
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--max-order', type=float, default=75.0, help='Max $ per order')
    ap.add_argument('--max-spread-pct', type=float, default=0.02, help='Max bid/ask spread as fraction of mid (default 2%%)')
    ap.add_argument('--allow-closed', action='store_true', help='Allow submission when market is closed')
    # Feature toggle flags (runtime overrides for ML/AI components)
    ap.add_argument('--calibration', action='store_true', help='Enable probability calibration for ML confidence')
    ap.add_argument('--conformal', action='store_true', help='Enable conformal prediction for position sizing')
    ap.add_argument('--exec-bandit', action='store_true', help='Enable execution bandit for order routing')
    ap.add_argument('--intraday-trigger', action='store_true', help='Enable intraday entry trigger (VWAP reclaim)')
    ap.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    ap.add_argument('--allow-stale', action='store_true', help='DANGER: Allow submission of stale signals (not recommended)')
    args = ap.parse_args()

    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)
    else:
        # Fallback to project .env if available
        local_env = ROOT / '.env'
        if local_env.exists():
            load_env(local_env)

    # Load config for feature flags
    config = load_config()

    # === Apply CLI feature flags to config (runtime overrides) ===
    if args.calibration or args.conformal or args.exec_bandit or args.intraday_trigger:
        # Override calibration setting
        if args.calibration:
            if 'ml' not in config:
                config['ml'] = {}
            if 'calibration' not in config['ml']:
                config['ml']['calibration'] = {}
            config['ml']['calibration']['enabled'] = True
            if args.verbose:
                print("[CLI] --calibration: Probability calibration ENABLED")

        # Override conformal setting
        if args.conformal:
            if 'ml' not in config:
                config['ml'] = {}
            if 'conformal' not in config['ml']:
                config['ml']['conformal'] = {}
            config['ml']['conformal']['enabled'] = True
            if args.verbose:
                print("[CLI] --conformal: Conformal prediction ENABLED")

        # Override execution bandit setting
        if args.exec_bandit:
            if 'execution' not in config:
                config['execution'] = {}
            config['execution']['bandit_enabled'] = True
            if args.verbose:
                print("[CLI] --exec-bandit: Execution bandit ENABLED")

        # Override intraday trigger setting
        if args.intraday_trigger:
            if 'execution' not in config:
                config['execution'] = {}
            if 'intraday_trigger' not in config['execution']:
                config['execution']['intraday_trigger'] = {}
            config['execution']['intraday_trigger']['enabled'] = True
            if args.verbose:
                print("[CLI] --intraday-trigger: Intraday entry trigger ENABLED")

    # Quick market-open guard (skip unless explicitly allowed)
    clk = get_market_clock()
    if clk and not clk.get('is_open', False) and not args.allow_closed:
        print('Market is CLOSED; use --allow-closed to override (order likely rejected).')
        return

    # Macro blackout gate (FOMC, NFP, CPI days)
    macro_blackout_enabled = config.get('risk', {}).get('macro_blackout_enabled', True)
    if macro_blackout_enabled:
        calendar = MacroEventCalendar()
        should_reduce, reason = calendar.should_reduce_exposure()
        if should_reduce:
            jlog('totd_macro_blackout', reason=reason)
            print(f'MACRO BLACKOUT: {reason} - Skipping new entries today.')
            return

    # One-at-a-time mode (limit concurrent trades)
    max_concurrent = config.get('risk', {}).get('max_concurrent_trades', 3)
    open_positions = get_open_position_count()
    if open_positions >= max_concurrent:
        jlog('totd_max_concurrent', open_positions=open_positions, max_concurrent=max_concurrent)
        print(f'MAX CONCURRENT: {open_positions}/{max_concurrent} positions open - Skipping new entry.')
        return

    p = Path(args.totd)
    if not p.exists():
        print('No TOTD file found:', p)
        return
    df = pd.read_csv(p)
    if df.empty:
        print('TOTD file empty - skipping submission.')
        return

    # === SIGNAL FRESHNESS VALIDATION ===
    # Critical safety check: prevent trading stale signals from previous days
    all_fresh, freshness_result, _ = validate_signal_file(p, timestamp_column='timestamp', max_age_days=1)
    if not all_fresh and not args.allow_stale:
        jlog('totd_stale_signal',
             signal_date=str(freshness_result.signal_date),
             expected_date=str(freshness_result.expected_date),
             days_old=freshness_result.days_old,
             reason=freshness_result.reason)
        print(f'STALE SIGNAL BLOCKED: {freshness_result.reason}')
        print(f'  Signal date: {freshness_result.signal_date}')
        print(f'  Expected:    {freshness_result.expected_date}')
        print(f'  Days old:    {freshness_result.days_old}')
        print('Run a fresh scan before submitting: python scripts/scan.py --top3')
        print('Or use --allow-stale to force (not recommended)')
        return
    elif not all_fresh and args.allow_stale:
        print(f'WARNING: Submitting stale signal ({freshness_result.days_old} days old) - --allow-stale flag used')
        jlog('totd_stale_signal_forced',
             signal_date=str(freshness_result.signal_date),
             days_old=freshness_result.days_old)

    row = df.iloc[0]
    symbol = str(row.get('symbol', '')).upper()
    side = str(row.get('side', 'long')).lower()
    if not symbol:
        print('Invalid TOTD: missing symbol')
        return

    # Intraday entry trigger gate (VWAP reclaim, first-hour retest)
    intraday_cfg = config.get('execution', {}).get('intraday_trigger', {})
    if intraday_cfg.get('enabled', False):
        from execution.intraday_trigger import check_entry_trigger
        trigger_mode = intraday_cfg.get('mode', 'vwap_reclaim')
        result = check_entry_trigger(symbol, side, mode=trigger_mode)
        if not result.triggered:
            jlog('totd_trigger_not_met', symbol=symbol, side=side, reason=result.reason,
                 price=result.price, vwap=result.vwap)
            print(f'TRIGGER NOT MET: {result.reason} - Skipping entry.')
            return
        else:
            jlog('totd_trigger_confirmed', symbol=symbol, side=side, reason=result.reason,
                 price=result.price, vwap=result.vwap)
            print(f'TRIGGER CONFIRMED: {result.reason}')

    ask = get_best_ask(symbol)
    bid = get_best_bid(symbol)
    if ask is None or bid is None or math.isnan(ask) or ask <= 0 or bid <= 0:
        jlog('totd_no_best_ask', symbol=symbol)
        print(f'No best ask for {symbol}; skipping.')
        return
    mid = (ask + bid) / 2.0
    spread = (ask - bid) / mid if mid > 0 else 1.0
    if spread > float(args.max_spread_pct):
        jlog('totd_skip_wide_spread', symbol=symbol, spread=spread)
        print(f'Skip {symbol}: spread {spread:.2%} > max {args.max_spread_pct:.2%}')
        return
    limit_px = round(float(ask) * 1.001, 2)
    qty = max(1, int(args.max_order // limit_px))

    policy = PolicyGate(RiskLimits(max_notional_per_order=args.max_order, max_daily_notional=1000.0, min_price=3.0, allow_shorts=False))
    ok, reason = policy.check(symbol, 'long' if side == 'long' else 'short', limit_px, qty)
    if not ok:
        jlog('totd_policy_veto', symbol=symbol, reason=reason, price=limit_px, qty=qty)
        print(f'Policy veto for {symbol}: {reason}')
        return

    # Telegram trade confirmation workflow (human-in-the-loop)
    telegram_cfg = config.get('telegram', {})
    if telegram_cfg.get('confirm_enabled', False):
        from alerts.telegram_commander import get_commander
        timeout = telegram_cfg.get('confirm_timeout_minutes', 30)
        commander = get_commander(timeout_minutes=timeout)

        # Build signal dict for trade card
        signal = {
            'symbol': symbol,
            'side': side,
            'entry_price': limit_px,
            'stop_loss': row.get('stop_loss', limit_px * 0.95),
            'take_profit': row.get('take_profit', limit_px * 1.10),
            'confidence': row.get('confidence', row.get('score', 0)),
            'strategy': row.get('strategy', 'TOTD'),
            'spread_pct': spread,
        }

        confirm_id = commander.send_trade_card(signal)
        jlog('totd_confirm_sent', symbol=symbol, confirm_id=confirm_id)
        print(f'Trade card sent to Telegram. Confirm ID: {confirm_id}')
        print(f'Waiting {timeout} minutes for confirmation...')

        confirmed = commander.wait_for_confirmation(confirm_id, timeout)

        if not confirmed:
            auto_exec = telegram_cfg.get('auto_execute_after_timeout', False)
            if not auto_exec:
                jlog('totd_not_confirmed', symbol=symbol, confirm_id=confirm_id)
                print(f'Trade NOT CONFIRMED for {symbol} - Skipping.')
                commander.send_confirmation_result(confirm_id, executed=False, details='User did not confirm')
                return
            else:
                jlog('totd_auto_execute_timeout', symbol=symbol, confirm_id=confirm_id)
                print(f'Timeout reached, auto-executing {symbol}...')
        else:
            jlog('totd_confirmed', symbol=symbol, confirm_id=confirm_id)
            print(f'Trade CONFIRMED for {symbol}!')

    decision = construct_decision(symbol, 'long' if side == 'long' else 'short', qty, ask)
    rec = place_ioc_limit(decision)
    config_pin = sha256_file('config/settings.json') if Path('config/settings.json').exists() else None
    append_block({
        'decision_id': rec.decision_id,
        'symbol': symbol,
        'side': 'BUY' if side == 'long' else 'SELL',
        'qty': qty,
        'limit_price': limit_px,
        'config_pin': config_pin,
        'status': str(rec.status),
        'notes': rec.notes,
    })
    jlog('totd_order_submit', symbol=symbol, status=str(rec.status), qty=qty, price=limit_px, decision_id=rec.decision_id)
    print(f'TOTD {symbol} -> {rec.status} @ {limit_px} qty {qty} note={rec.notes}')


if __name__ == '__main__':
    main()
