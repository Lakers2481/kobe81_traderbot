#!/usr/bin/env python3
from __future__ import annotations

"""
Submit Top-3 daily picks (2 ICT + 1 IBS_RSI) as IOC LIMIT orders via Alpaca (paper by default).

Flow:
- Load .env for credentials
- If logs/daily_picks.csv exists, read; else run scan.py --top3 to generate
- Apply kill switch and PolicyGate checks
- For each pick, fetch best ask, clamp limit, size to budget, and submit IOC LIMIT
- Append to hash chain and structured logs
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import math
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from config.env_loader import load_env
from data.providers.alpaca_live import get_market_clock
from execution.broker_alpaca import get_best_ask, construct_decision, place_ioc_limit, place_bracket_order
from risk.policy_gate import PolicyGate, RiskLimits
from core.hash_chain import append_block
from core.structured_log import jlog
from core.kill_switch import is_kill_switch_active, get_kill_switch_info


def run_scan(dotenv: Path, cap: int | None = None) -> Path:
    """Run scan.py --top3 to produce daily_picks.csv if missing."""
    cmd = [sys.executable, str(ROOT / 'scripts' / 'scan.py'), '--top3', '--dotenv', str(dotenv)]
    if cap:
        cmd += ['--cap', str(cap)]
    subprocess.run(cmd, check=False, cwd=str(ROOT))
    return ROOT / 'logs' / 'daily_picks.csv'


def main() -> int:
    ap = argparse.ArgumentParser(description='Trade Top-3 daily picks (IOC LIMIT via Alpaca)')
    ap.add_argument('--dotenv', type=str, default='./.env', help='Path to .env file (default: ./.env)')
    ap.add_argument('--ensure-scan', action='store_true', help='Run scan.py --top3 if picks CSV missing')
    ap.add_argument('--cap', type=int, default=None, help='Universe cap for scan (optional)')
    ap.add_argument('--budget-per-order', type=float, default=75.0)
    ap.add_argument('--daily-budget', type=float, default=1000.0)
    ap.add_argument('--min-price', type=float, default=5.0)
    ap.add_argument('--allow-shorts', action='store_true')
    ap.add_argument('--allow-closed', action='store_true', help='Allow submission when market is closed')
    ap.add_argument('--bracket', action='store_true', help='Use bracket orders (entry + SL + TP) instead of IOC LIMIT')
    args = ap.parse_args()

    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)
    else:
        # Fallback to local .env
        local_env = ROOT / '.env'
        if local_env.exists():
            load_env(local_env)

    # Default to Alpaca paper
    os.environ.setdefault('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    if is_kill_switch_active():
        info = get_kill_switch_info()
        jlog('trade_top3_kill_switch', level='WARNING', info=info)
        print('KILL SWITCH active; aborting.')
        return 2

    # Market-open guard unless overridden
    clk = get_market_clock()
    if clk and not clk.get('is_open', False) and not args.allow_closed:
        print('Market is CLOSED; use --allow-closed to override (orders may be rejected).')
        return 2

    picks_path = ROOT / 'logs' / 'daily_picks.csv'
    if not picks_path.exists() and args.ensure_scan:
        picks_path = run_scan(dotenv, cap=args.cap)

    if not picks_path.exists():
        print(f'Missing picks file: {picks_path}. Run scan.py --top3 first or use --ensure-scan.')
        return 1

    try:
        df = pd.read_csv(picks_path)
    except Exception as e:
        print(f'Failed to read picks: {e}')
        return 1

    if df.empty:
        print('No picks to trade.')
        return 0

    policy = PolicyGate(RiskLimits(
        max_notional_per_order=float(args.budget_per_order),
        max_daily_notional=float(args.daily_budget),
        min_price=float(args.min_price),
        allow_shorts=bool(args.allow_shorts),
    ))

    submitted = 0
    for _, row in df.iterrows():
        sym = str(row.get('symbol', '')).upper()
        side_raw = str(row.get('side', 'long')).lower()
        side = 'BUY' if side_raw == 'long' else 'SELL'
        ask = get_best_ask(sym)
        if ask is None or math.isnan(ask) or ask <= 0:
            jlog('trade_top3_no_quote', symbol=sym)
            print(f'Skip {sym}: no best ask')
            continue
        # Sizing under budget
        limit_hint = ask * 1.001
        qty = max(1, int(float(args.budget_per_order) // max(limit_hint, 0.01)))
        ok, reason = policy.check(sym, 'long' if side == 'BUY' else 'short', float(limit_hint), int(qty))
        if not ok:
            jlog('trade_top3_policy_veto', symbol=sym, reason=reason)
            print(f'VETO {sym}: {reason}')
            continue
        # Construct and submit
        if args.bracket:
            # Use bracket order (entry + stop-loss + take-profit)
            stop_loss = float(row.get('stop_loss', 0))
            take_profit = float(row.get('take_profit', 0))
            if stop_loss <= 0 or take_profit <= 0:
                jlog('trade_top3_bracket_missing_levels', symbol=sym)
                print(f'Skip {sym}: missing stop_loss/take_profit for bracket order')
                continue
            result = place_bracket_order(
                symbol=sym,
                side='long' if side == 'BUY' else 'short',
                qty=int(qty),
                limit_price=limit_hint,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            rec = result.order
            order_type = 'BRACKET'
        else:
            # Use IOC LIMIT order
            decision = construct_decision(sym, 'long' if side == 'BUY' else 'short', int(qty), ask)
            broker_result = place_ioc_limit(decision)
            rec = broker_result.order
            order_type = 'IOC_LIMIT'

        append_block({
            'decision_id': rec.decision_id,
            'symbol': sym,
            'side': side,
            'qty': int(qty),
            'limit_price': float(rec.limit_price),
            'status': str(rec.status),
            'notes': rec.notes,
            'order_type': order_type,
            'ts': datetime.utcnow().isoformat(),
        })
        jlog('trade_top3_submit', symbol=sym, status=str(rec.status), qty=int(qty), price=float(rec.limit_price), order_type=order_type)
        print(f'{sym} -> {rec.status} @ {rec.limit_price} qty {qty} type={order_type} note={rec.notes}')
        submitted += 1

    order_desc = 'bracket' if args.bracket else 'IOC LIMIT'
    print(f'Submitted {submitted} {order_desc} order(s).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

