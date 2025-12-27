#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time
import os
import math

import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from config.settings_loader import is_earnings_filter_enabled
from core.earnings_filter import filter_signals_by_earnings
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon
from strategies.donchian.strategy import DonchianBreakoutStrategy
from strategies.ict.turtle_soup import TurtleSoupStrategy
from execution.broker_alpaca import get_best_ask, construct_decision, place_ioc_limit
from risk.policy_gate import PolicyGate, RiskLimits
from oms.order_state import OrderStatus
from core.hash_chain import append_block
from core.structured_log import jlog
from monitor.health_endpoints import update_request_counter
from core.config_pin import sha256_file


def main():
    ap = argparse.ArgumentParser(description='Kobe paper trading runner (IOC LIMIT only)')
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--start', type=str, required=True)
    ap.add_argument('--end', type=str, required=True)
    ap.add_argument('--cap', type=int, default=50)
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--cache', type=str, default='data/cache')
    ap.add_argument('--kill-switch', type=str, default='state/KILL_SWITCH')
    args = ap.parse_args()

    # Env
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # Mode: ensure paper endpoint
    os.environ.setdefault('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    # Universe
    symbols = load_universe(Path(args.universe), cap=args.cap)
    cache_dir = Path(args.cache)

    # Strategies (Donchian + ICT Turtle Soup)
    don = DonchianBreakoutStrategy()
    ict = TurtleSoupStrategy()

    # Risk/Policy
    policy = PolicyGate(RiskLimits(max_notional_per_order=75.0, max_daily_notional=1000.0, min_price=3.0, allow_shorts=False))

    # Fetch latest bars (daily)
    frames = []
    for s in symbols:
        df = fetch_daily_bars_polygon(s, args.start, args.end, cache_dir=cache_dir)
        if not df.empty:
            if 'symbol' not in df:
                df = df.copy(); df['symbol'] = s
            frames.append(df)
    if not frames:
        print('No data fetched; abort.')
        return
    data = pd.concat(frames, ignore_index=True).sort_values(['symbol','timestamp'])

    # Generate Donchian and ICT signals; select only current bar
    a = don.scan_signals_over_time(data)
    b = ict.scan_signals_over_time(data)
    if a.empty and b.empty:
        print('No signals today (Donchian/ICT).')
        return
    # Reduce to last date only to avoid replaying history
    last_ts = max([x['timestamp'].max() for x in [a, b] if not x.empty])
    cols = ['timestamp','symbol','side','entry_price','stop_loss','take_profit','reason']
    a = a[a['timestamp'] == last_ts][cols].copy() if not a.empty else pd.DataFrame(columns=cols)
    a['strategy'] = 'DONCHIAN'
    b = b[b['timestamp'] == last_ts][cols].copy() if not b.empty else pd.DataFrame(columns=cols)
    b['strategy'] = 'TURTLE_SOUP'
    todays = pd.concat([a, b], ignore_index=True)
    # Apply earnings filter if enabled
    if not todays.empty and is_earnings_filter_enabled():
        todays = pd.DataFrame(filter_signals_by_earnings(todays.to_dict('records')))

    # Kill switch check
    if Path(args.kill_switch).exists():
        jlog('kill_switch_active', level='WARN', path=str(args.kill_switch))
        print('KILL SWITCH active; aborting submissions.')
        return

    # Submit orders with IOC LIMIT at best ask + 0.1%
    submitted = 0
    config_pin = sha256_file('config/settings.json') if Path('config/settings.json').exists() else None
    for _, row in todays.iterrows():
        sym = row['symbol']
        side = 'BUY' if str(row['side']).lower() == 'long' else 'SELL'
        # Get best ask for limit
        ask = get_best_ask(sym)
        if ask is None or math.isnan(ask):
            jlog('skip_no_best_ask', symbol=sym)
            print(f"Skip {sym}: no best ask")
            continue
        # Sizing: fit under per-order budget
        limit_px = round(ask * 1.001, 2)
        max_qty = max(1, int(75.0 // limit_px))
        ok, reason = policy.check(sym, 'long' if side=='BUY' else 'short', limit_px, max_qty)
        if not ok:
            jlog('policy_veto', symbol=sym, reason=reason, price=limit_px, qty=max_qty)
            print(f"VETO {sym}: {reason}")
            continue
        decision = construct_decision(sym, 'long' if side=='BUY' else 'short', max_qty, ask)
        rec = place_ioc_limit(decision)
        # Audit block
        append_block({
            'decision_id': rec.decision_id,
            'symbol': sym,
            'side': side,
            'qty': max_qty,
            'limit_price': limit_px,
            'config_pin': config_pin,
            'status': str(rec.status),
            'notes': rec.notes,
        })
        jlog('order_submit', symbol=sym, status=str(rec.status), qty=max_qty, price=limit_px, decision_id=rec.decision_id)
        print(f"{sym} -> {rec.status} @ {limit_px} qty {max_qty} note={rec.notes}")
        if str(rec.status).upper().endswith('SUBMITTED'):
            update_request_counter('orders_submitted', 1)
        elif str(rec.status).upper().endswith('REJECTED'):
            update_request_counter('orders_rejected', 1)
        submitted += 1
    print(f"Submitted {submitted} IOC LIMIT orders.")


if __name__ == '__main__':
    main()
