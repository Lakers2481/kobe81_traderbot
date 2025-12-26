#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import os

import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.env_loader import load_env
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon
from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
from strategies.ibs.strategy import IBSStrategy
from execution.broker_alpaca import get_best_ask, construct_decision, place_ioc_limit
from risk.policy_gate import PolicyGate, RiskLimits
from core.hash_chain import append_block
from core.structured_log import jlog
from core.config_pin import sha256_file


def main():
    ap = argparse.ArgumentParser(description='Kobe live micro trading (IOC LIMIT only)')
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--start', type=str, required=True)
    ap.add_argument('--end', type=str, required=True)
    ap.add_argument('--cap', type=int, default=10)
    ap.add_argument('--dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
    ap.add_argument('--cache', type=str, default='data/cache')
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

    # Strategies
    rsi2 = ConnorsRSI2Strategy()
    ibs = IBSStrategy()

    policy = PolicyGate(RiskLimits(max_notional_per_order=75.0, max_daily_notional=1000.0, min_price=3.0, allow_shorts=False))

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

    a = rsi2.scan_signals_over_time(data)
    b = ibs.scan_signals_over_time(data)
    sigs = pd.merge(a, b, on=['timestamp','symbol','side'], suffixes=('_rsi2','_ibs')) if not a.empty and not b.empty else pd.DataFrame()
    if sigs.empty:
        print('No AND signals today.')
        return
    last_ts = sigs['timestamp'].max()
    todays = sigs[sigs['timestamp'] == last_ts].copy()

    config_pin = sha256_file('configs/settings.json') if Path('configs/settings.json').exists() else None
    submitted = 0
    for _, row in todays.iterrows():
        sym = row['symbol']
        ask = get_best_ask(sym)
        if ask is None:
            jlog('skip_no_best_ask', symbol=sym)
            print(f'Skip {sym}: no best ask')
            continue
        limit_px = round(ask * 1.001, 2)
        qty = max(1, int(75.0 // limit_px))
        ok, reason = policy.check(sym, 'long', limit_px, qty)
        if not ok:
            jlog('policy_veto', symbol=sym, reason=reason, price=limit_px, qty=qty)
            print(f'VETO {sym}: {reason}')
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
        submitted += 1
    print(f'Submitted {submitted} IOC LIMIT orders (live micro).')


if __name__ == '__main__':
    main()
