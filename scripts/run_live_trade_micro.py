#!/usr/bin/env python3
"""
DEPRECATED: This script is deprecated. Use run_paper_trade.py --live instead.

This script lacks critical safeguards present in run_paper_trade.py:
- No weekly exposure gate
- No unified enrichment pipeline (Kelly sizing, regime, VIX)
- No cognitive brain evaluation
- No learning hub integration
- No kill zone gate

Use the unified script for live trading:
  python scripts/run_paper_trade.py --live --confirm-live I_UNDERSTAND_REAL_MONEY --universe ...
"""
from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

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
from risk.signal_quality_gate import filter_to_best_signals
from core.hash_chain import append_block
from core.structured_log import jlog
from monitor.health_endpoints import update_request_counter
from core.config_pin import sha256_file

# FIX 4 (2026-01-08): Wire learning hub for episodic memory
try:
    from integration.learning_hub import get_learning_hub
    LEARNING_HUB_AVAILABLE = True
except ImportError:
    LEARNING_HUB_AVAILABLE = False


def main():
    # ==========================================================================
    # DEPRECATION WARNING
    # ==========================================================================
    warnings.warn(
        "run_live_trade_micro.py is DEPRECATED. "
        "Use: python scripts/run_paper_trade.py --live --confirm-live I_UNDERSTAND_REAL_MONEY",
        DeprecationWarning,
        stacklevel=2
    )
    print("=" * 70)
    print("WARNING: This script is DEPRECATED")
    print("=" * 70)
    print()
    print("This script lacks critical safeguards. Use the unified script instead:")
    print("  python scripts/run_paper_trade.py --live --confirm-live I_UNDERSTAND_REAL_MONEY")
    print()
    print("Missing safeguards in this script:")
    print("  - Weekly exposure gate (40% cap)")
    print("  - Unified enrichment pipeline (100+ fields)")
    print("  - Cognitive brain evaluation")
    print("  - Kill zone gate (10:00-11:30, 14:30-15:30)")
    print("  - Decision cards audit trail")
    print()
    print("Proceeding with REDUCED SAFEGUARDS in 5 seconds...")
    print("=" * 70)
    import time
    time.sleep(5)
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

    # FIX (2026-01-08): Apply quality gate for parity with scan.py (Score >= 70, Conf >= 0.60)
    if not todays.empty:
        before_count = len(todays)
        filtered_signals = filter_to_best_signals(todays.to_dict('records'), max_signals=10)
        todays = pd.DataFrame(filtered_signals) if filtered_signals else pd.DataFrame()
        jlog('quality_gate_applied', before=before_count, after=len(todays))
        if todays.empty:
            print('No signals passed quality gate (Score >= 70, Confidence >= 0.60).')
            return

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
            # FIX 4 (2026-01-08): Record entry with learning hub for feedback loop
            if LEARNING_HUB_AVAILABLE:
                try:
                    hub = get_learning_hub()
                    stop_loss = row.get('stop_loss', limit_px * 0.95) if hasattr(row, 'get') else limit_px * 0.95
                    strategy_name = row.get('strategy', 'DUAL_STRATEGY') if hasattr(row, 'get') else 'DUAL_STRATEGY'
                    hub.record_trade_entry({
                        'symbol': sym,
                        'side': 'long',
                        'entry_price': limit_px,
                        'shares': qty,
                        'strategy': strategy_name,
                        'trade_id': rec.decision_id,
                        'entry_time': datetime.utcnow().isoformat(),
                        'stop_loss': float(stop_loss) if stop_loss else limit_px * 0.95,
                        'take_profit': limit_px * 1.05,  # 5% target for micro
                        'signal_score': row.get('conf_score', 0) if hasattr(row, 'get') else 0,
                        'regime': 'unknown',  # Live micro doesn't have enrichment
                    })
                    jlog('learning_hub_entry_recorded', symbol=sym, trade_id=rec.decision_id)
                except Exception as e:
                    jlog('learning_hub_entry_error', symbol=sym, error=str(e), level='WARN')
        elif str(rec.status).upper().endswith('REJECTED'):
            update_request_counter('orders_rejected', 1)
        submitted += 1
    print(f'Submitted {submitted} IOC LIMIT orders (live micro).')


if __name__ == '__main__':
    main()



