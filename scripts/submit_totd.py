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
from data.providers.alpaca_live import get_market_clock
from execution.broker_alpaca import get_best_ask, get_best_bid, construct_decision, place_ioc_limit
from risk.policy_gate import PolicyGate, RiskLimits
from core.structured_log import jlog
from core.hash_chain import append_block
from core.config_pin import sha256_file


def main() -> None:
    ap = argparse.ArgumentParser(description='Submit Trade of the Day (IOC LIMIT)')
    ap.add_argument('--totd', type=str, default='logs/trade_of_day.csv', help='Path to trade_of_day.csv')
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--max-order', type=float, default=75.0, help='Max $ per order')
    ap.add_argument('--max-spread-pct', type=float, default=0.02, help='Max bid/ask spread as fraction of mid (default 2%)')
    ap.add_argument('--allow-closed', action='store_true', help='Allow submission when market is closed')
    args = ap.parse_args()

    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)
    else:
        # Fallback to project .env if available
        local_env = ROOT / '.env'
        if local_env.exists():
            load_env(local_env)

    # Quick market-open guard (skip unless explicitly allowed)
    clk = get_market_clock()
    if clk and not clk.get('is_open', False) and not args.allow_closed:
        print('Market is CLOSED; use --allow-closed to override (order likely rejected).')
        return

    p = Path(args.totd)
    if not p.exists():
        print('No TOTD file found:', p)
        return
    df = pd.read_csv(p)
    if df.empty:
        print('TOTD file empty - skipping submission.')
        return

    row = df.iloc[0]
    symbol = str(row.get('symbol', '')).upper()
    side = str(row.get('side', 'long')).lower()
    if not symbol:
        print('Invalid TOTD: missing symbol')
        return

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
