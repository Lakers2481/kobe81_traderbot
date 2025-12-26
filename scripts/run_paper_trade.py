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
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon
from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
from strategies.ibs.strategy import IBSStrategy
from execution.broker_alpaca import get_best_ask, construct_decision, place_ioc_limit
from risk.policy_gate import PolicyGate, RiskLimits
from oms.order_state import OrderStatus
from core.hash_chain import append_block
from core.structured_log import jlog
from core.config_pin import sha256_file


def main():
    ap = argparse.ArgumentParser(description='Kobe paper trading runner (IOC LIMIT only)')
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--start', type=str, required=True)
    ap.add_argument('--end', type=str, required=True)
    ap.add_argument('--cap', type=int, default=50)
    ap.add_argument('--dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
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

    # Strategies
    rsi2 = ConnorsRSI2Strategy()
    ibs = IBSStrategy()

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

    # Generate AND signals (same-bar concurrence)
    a = rsi2.scan_signals_over_time(data)
    b = ibs.scan_signals_over_time(data)
    sigs = pd.merge(a, b, on=['timestamp','symbol','side'], suffixes=('_rsi2','_ibs')) if not a.empty and not b.empty else pd.DataFrame()
    if sigs.empty:
        print('No AND signals today.')
        return
    # Reduce to last date only (today's bar) to avoid replaying history
    last_ts = sigs['timestamp'].max()
    todays = sigs[sigs['timestamp'] == last_ts].copy()

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
        submitted += 1
    print(f"Submitted {submitted} IOC LIMIT orders.")


if __name__ == '__main__':
    main()
