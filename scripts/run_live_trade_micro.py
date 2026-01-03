#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import os

import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from config.settings_loader import is_earnings_filter_enabled
from core.earnings_filter import filter_signals_by_earnings
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon
from strategies.dual_strategy import DualStrategyScanner, DualStrategyParams
from execution.broker_alpaca import get_best_ask, get_best_bid, construct_decision, place_ioc_limit
from risk.policy_gate import PolicyGate, RiskLimits
from risk.position_limit_gate import PositionLimitGate, PositionLimits
from core.hash_chain import append_block
from core.structured_log import jlog
from monitor.health_endpoints import update_request_counter
from core.config_pin import sha256_file


def main():
    ap = argparse.ArgumentParser(description='Kobe live micro trading (IOC LIMIT only)')
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--start', type=str, required=True)
    ap.add_argument('--end', type=str, required=True)
    ap.add_argument('--cap', type=int, default=10)
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--cache', type=str, default='data/cache')
    ap.add_argument('--max-spread-pct', type=float, default=0.02, help='Max bid/ask spread as fraction of mid (default 2%)')
    args = ap.parse_args()

    # Env
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # Mode: live endpoint expected in env
    # os.environ['ALPACA_BASE_URL'] must point to live trading URL if using live

    # Universe
    symbols = load_universe(Path(args.universe), cap=args.cap)
    cache_dir = Path(args.cache)

    # Strategies: Dual scanner (IBS+RSI + Turtle Soup)
    scanner = DualStrategyScanner(DualStrategyParams())

    policy = PolicyGate(RiskLimits(max_notional_per_order=75.0, max_daily_notional=1000.0, min_price=3.0, allow_shorts=False))
    position_gate = PositionLimitGate(PositionLimits(max_positions=5, max_per_symbol=1))

    # Check current position count before proceeding
    pos_status = position_gate.get_status()
    print(f"Current positions: {pos_status['open_positions']}/{pos_status['max_positions']} "
          f"(available: {pos_status['positions_available']})")
    if pos_status['open_symbols']:
        print(f"  Open: {', '.join(pos_status['open_symbols'])}")

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

    # Generate combined signals over time; select last date only
    sigs = scanner.generate_signals(data)
    if sigs.empty:
        print('No signals today (IBS+RSI/ICT).')
        return
    last_ts = sigs['timestamp'].max()
    cols = ['timestamp','symbol','side','entry_price','stop_loss','take_profit','reason']
    todays = sigs[sigs['timestamp'] == last_ts][cols].copy()
    if not todays.empty and is_earnings_filter_enabled():
        todays = pd.DataFrame(filter_signals_by_earnings(todays.to_dict('records')))

    config_pin = sha256_file('config/settings.json') if Path('config/settings.json').exists() else None
    submitted = 0
    for _, row in todays.iterrows():
        sym = row['symbol']
        ask = get_best_ask(sym)
        bid = get_best_bid(sym)
        if ask is None or bid is None or ask <= 0 or bid <= 0:
            jlog('skip_no_best_quote', symbol=sym)
            print(f'Skip {sym}: no valid bid/ask')
            continue
        mid = (ask + bid) / 2.0
        spread = (ask - bid) / mid if mid > 0 else 1.0
        if spread > float(args.max_spread_pct):
            jlog('skip_wide_spread', symbol=sym, spread=spread)
            print(f'Skip {sym}: spread {spread:.2%} > max {args.max_spread_pct:.2%}')
            continue
        limit_px = round(ask * 1.001, 2)
        qty = max(1, int(75.0 // limit_px))
        ok, reason = policy.check(sym, 'long', limit_px, qty)
        if not ok:
            jlog('policy_veto', symbol=sym, reason=reason, price=limit_px, qty=qty)
            print(f'VETO {sym}: {reason}')
            continue
        # Position limit check
        pos_ok, pos_reason = position_gate.check(sym, 'long', limit_px, qty)
        if not pos_ok:
            jlog('position_limit_veto', symbol=sym, reason=pos_reason)
            print(f'VETO {sym}: {pos_reason}')
            continue
        decision = construct_decision(sym, 'long', qty, ask)
        rec = place_ioc_limit(decision)
        append_block({
            'decision_id': rec.decision_id,
            'symbol': sym,
            'side': 'BUY',
            'qty': qty,
            'limit_price': limit_px,
            'config_pin': config_pin,
            'status': str(rec.status),
            'notes': rec.notes,
        })
        jlog('order_submit', symbol=sym, status=str(rec.status), qty=qty, price=limit_px, decision_id=rec.decision_id)
        print(f"{sym} -> {rec.status} @ {limit_px} qty {qty} note={rec.notes}")
        if str(rec.status).upper().endswith('SUBMITTED'):
            update_request_counter('orders_submitted', 1)
        elif str(rec.status).upper().endswith('REJECTED'):
            update_request_counter('orders_rejected', 1)
        submitted += 1
    print(f'Submitted {submitted} IOC LIMIT orders (live micro).')


if __name__ == '__main__':
    main()



