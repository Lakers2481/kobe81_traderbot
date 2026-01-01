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
from strategies.dual_strategy import DualStrategyScanner, DualStrategyParams
from execution.broker_alpaca import get_best_ask, get_best_bid, construct_decision, place_ioc_limit
from risk.policy_gate import PolicyGate, RiskLimits, load_limits_from_config
from risk.position_limit_gate import PositionLimitGate, PositionLimits
from oms.order_state import OrderStatus
from core.hash_chain import append_block
from core.structured_log import jlog
from monitor.health_endpoints import update_request_counter
from core.config_pin import sha256_file

# Decision Card logging
try:
    from trade_logging.decision_card_logger import (
        get_card_logger, TradePlan, SignalDriver, RiskCheck, ModelInfo
    )
    DECISION_CARDS_AVAILABLE = True
except ImportError:
    DECISION_CARDS_AVAILABLE = False

# Cognitive system (optional)
try:
    from cognitive.signal_processor import get_signal_processor
    COGNITIVE_AVAILABLE = True
except ImportError:
    COGNITIVE_AVAILABLE = False


def main():
    ap = argparse.ArgumentParser(description='Kobe paper trading runner (IOC LIMIT only)')
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--start', type=str, required=True)
    ap.add_argument('--end', type=str, required=True)
    ap.add_argument('--cap', type=int, default=50)
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--cache', type=str, default='data/cache')
    ap.add_argument('--kill-switch', type=str, default='state/KILL_SWITCH')
    ap.add_argument('--max-spread-pct', type=float, default=0.02, help='Max bid/ask spread as fraction of mid (default 2%)')
    ap.add_argument('--cognitive', action='store_true', default=True, help='Enable cognitive brain (ON by default)')
    ap.add_argument('--no-cognitive', action='store_true', help='Disable cognitive brain')
    ap.add_argument('--cognitive-min-conf', type=float, default=0.5, help='Min cognitive confidence to trade')
    args = ap.parse_args()

    # Handle --no-cognitive override
    if args.no_cognitive:
        args.cognitive = False

    # Env
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # Mode: ensure paper endpoint
    os.environ.setdefault('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    # Universe
    symbols = load_universe(Path(args.universe), cap=args.cap)
    cache_dir = Path(args.cache)

    # Strategies (IBS+RSI + ICT Turtle Soup) via dual scanner
    scanner = DualStrategyScanner(DualStrategyParams())

    # Risk/Policy - LOAD FROM CONFIG (respects trading_mode: micro/paper/real)
    risk_limits = load_limits_from_config()
    policy = PolicyGate(risk_limits)
    position_gate = PositionLimitGate(PositionLimits(max_positions=risk_limits.max_positions, max_per_symbol=1))

    print(f"Trading Mode: {risk_limits.mode_name.upper()}")
    print(f"  Max Notional/Order: ${risk_limits.max_notional_per_order:,.0f}")
    print(f"  Max Daily Notional: ${risk_limits.max_daily_notional:,.0f}")
    print(f"  Risk per Trade: {risk_limits.risk_per_trade_pct*100:.1f}%")

    # Check current position count before proceeding
    pos_status = position_gate.get_status()
    print(f"Current positions: {pos_status['open_positions']}/{pos_status['max_positions']} "
          f"(available: {pos_status['positions_available']})")
    if pos_status['open_symbols']:
        print(f"  Open: {', '.join(pos_status['open_symbols'])}")

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

    # Generate combined signals; select only current bar
    sigs = scanner.generate_signals(data)
    if sigs.empty:
        print('No signals today (IBS+RSI/ICT).')
        return
    # Reduce to last date only to avoid replaying history
    last_ts = sigs['timestamp'].max()
    cols = ['timestamp','symbol','side','entry_price','stop_loss','take_profit','reason','strategy']
    todays = sigs[sigs['timestamp'] == last_ts][cols].copy()
    # Apply earnings filter if enabled
    if not todays.empty and is_earnings_filter_enabled():
        todays = pd.DataFrame(filter_signals_by_earnings(todays.to_dict('records')))

    # Cognitive brain evaluation (optional)
    cognitive_processor = None
    cognitive_decisions = {}  # symbol -> (episode_id, size_multiplier)
    if args.cognitive and COGNITIVE_AVAILABLE and not todays.empty:
        jlog('cognitive_eval_start', count=len(todays))
        print(f"Running cognitive brain evaluation on {len(todays)} signals...")
        try:
            cognitive_processor = get_signal_processor()
            cognitive_processor.min_confidence = args.cognitive_min_conf

            # Evaluate signals through cognitive system
            approved_df, evaluated = cognitive_processor.evaluate_signals(
                signals=todays,
                market_data=data,
            )

            # Track cognitive decisions for outcome learning
            for ev in evaluated:
                sym = ev.original_signal.get('symbol', '')
                strat = ev.original_signal.get('strategy', '')
                if ev.approved:
                    decision_id = f"{sym}_{strat}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                    cognitive_decisions[sym] = {
                        'episode_id': ev.episode_id,
                        'decision_id': decision_id,
                        'size_multiplier': ev.size_multiplier,
                        'confidence': ev.cognitive_confidence,
                    }

            # Filter to only approved signals
            if not approved_df.empty:
                todays = approved_df
                jlog('cognitive_eval_complete', approved=len(approved_df), total=len(evaluated))
                print(f"  Cognitive: {len(evaluated)} evaluated -> {len(approved_df)} approved")
            else:
                jlog('cognitive_all_rejected', total=len(evaluated))
                print("  Cognitive: All signals rejected (low confidence)")
                todays = pd.DataFrame()

        except Exception as e:
            jlog('cognitive_eval_error', error=str(e), level='WARN')
            print(f"  [WARN] Cognitive evaluation failed: {e}")
    elif args.cognitive and not COGNITIVE_AVAILABLE:
        print("  [WARN] Cognitive system not available")

    # SELECT ONLY THE BEST SIGNAL (Trade of the Day)
    # Sort by cognitive confidence (or conf_score) and take top 1
    if not todays.empty:
        # Get confidence column
        if 'cognitive_confidence' in todays.columns:
            conf_col = 'cognitive_confidence'
        elif 'conf_score' in todays.columns:
            conf_col = 'conf_score'
        else:
            conf_col = None

        if conf_col and len(todays) > 1:
            todays = todays.sort_values(conf_col, ascending=False).head(1)
            best_sym = todays.iloc[0]['symbol']
            best_conf = todays.iloc[0].get(conf_col, 'N/A')
            print(f"  TRADE OF THE DAY: {best_sym} (conf={best_conf})")
            jlog('trade_of_day_selected', symbol=best_sym, confidence=best_conf)
        elif len(todays) == 1:
            print(f"  Single signal: {todays.iloc[0]['symbol']}")

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
        bid = get_best_bid(sym)
        if ask is None or bid is None or math.isnan(ask) or ask <= 0 or bid <= 0:
            jlog('skip_no_best_quote', symbol=sym)
            print(f"Skip {sym}: no valid bid/ask")
            continue
        mid = (ask + bid) / 2.0
        spread = (ask - bid) / mid if mid > 0 else 1.0
        if spread > float(args.max_spread_pct):
            jlog('skip_wide_spread', symbol=sym, spread=spread)
            print(f"Skip {sym}: spread {spread:.2%} > max {args.max_spread_pct:.2%}")
            continue
        # RISK-AWARE SIZING (Critical fix: enforce $50 max risk cap)
        limit_px = round(ask * 1.001, 2)

        # Get stop loss from signal (fallback to 5% if missing)
        stop_loss = row.get('stop_loss')
        if stop_loss is None or pd.isna(stop_loss):
            stop_loss = limit_px * 0.95  # 5% fallback
        risk_per_share = abs(limit_px - float(stop_loss))

        # Base max risk = max_notional × risk_pct (e.g., $2,500 × 2% = $50)
        base_max_risk = risk_limits.max_notional_per_order * risk_limits.risk_per_trade_pct

        # Get cognitive multiplier and confidence
        size_multiplier = 1.0
        cognitive_conf = None
        if sym in cognitive_decisions:
            cog = cognitive_decisions[sym]
            size_multiplier = cog.get('size_multiplier', 1.0)
            cognitive_conf = cog.get('confidence')

        # Apply cognitive multiplier to RISK BUDGET (not qty!)
        # This ensures hard risk cap always holds
        effective_risk = base_max_risk * size_multiplier  # e.g., $50 × 0.5 = $25

        # Calculate max qty by risk (primary constraint)
        max_qty_by_risk = int(effective_risk / risk_per_share) if risk_per_share > 0.01 else 1

        # Calculate max qty by notional (hard cap)
        max_qty_by_notional = int(risk_limits.max_notional_per_order // limit_px)

        # Take the LESSER of the two (enforce both caps)
        max_qty = max(1, min(max_qty_by_risk, max_qty_by_notional))

        # Log computed risk for audit trail
        computed_risk = max_qty * risk_per_share
        jlog('risk_sizing', symbol=sym, entry=limit_px, stop=float(stop_loss),
             risk_per_share=round(risk_per_share, 2), effective_risk_budget=round(effective_risk, 2),
             qty=max_qty, computed_risk=round(computed_risk, 2), multiplier=size_multiplier)
        ok, reason = policy.check(sym, 'long' if side=='BUY' else 'short', limit_px, max_qty, float(stop_loss))
        if not ok:
            jlog('policy_veto', symbol=sym, reason=reason, price=limit_px, qty=max_qty)
            print(f"VETO {sym}: {reason}")
            continue
        # Position limit check
        pos_ok, pos_reason = position_gate.check(sym, 'long' if side=='BUY' else 'short')
        if not pos_ok:
            jlog('position_limit_veto', symbol=sym, reason=pos_reason)
            print(f"VETO {sym}: {pos_reason}")
            continue
        decision = construct_decision(sym, 'long' if side=='BUY' else 'short', max_qty, ask)
        rec = place_ioc_limit(decision)
        # Build audit block with cognitive data if available
        audit_data = {
            'decision_id': rec.decision_id,
            'symbol': sym,
            'side': side,
            'qty': max_qty,
            'limit_price': limit_px,
            'config_pin': config_pin,
            'status': str(rec.status),
            'notes': rec.notes,
        }
        if cognitive_conf is not None:
            audit_data['cognitive_confidence'] = cognitive_conf
            audit_data['cognitive_size_mult'] = size_multiplier
            if sym in cognitive_decisions:
                audit_data['cognitive_episode_id'] = cognitive_decisions[sym].get('episode_id')

        append_block(audit_data)
        jlog('order_submit', symbol=sym, status=str(rec.status), qty=max_qty, price=limit_px,
             decision_id=rec.decision_id, cognitive_conf=cognitive_conf)

        # Create Decision Card for audit trail
        if DECISION_CARDS_AVAILABLE:
            try:
                card_logger = get_card_logger(environment='paper')
                take_profit = row.get('take_profit')
                card = card_logger.create_card(
                    symbol=sym,
                    side='long' if side == 'BUY' else 'short',
                    strategy=row.get('strategy', 'IBS_RSI'),
                    plan=TradePlan(
                        entry_price=limit_px,
                        target_price=float(take_profit) if take_profit else limit_px * 1.05,
                        stop_loss=float(stop_loss),
                        qty=max_qty,
                    ),
                    signals=[
                        SignalDriver(name='conf_score', value=row.get('conf_score', 0), contribution=0.5),
                        SignalDriver(name='rsi2', value=row.get('rsi2', 0), contribution=0.3),
                        SignalDriver(name='ibs', value=row.get('ibs', 0), contribution=0.2),
                    ],
                    risk_checks=[
                        RiskCheck(name='policy_gate', passed=True, details='notional_ok'),
                        RiskCheck(name='position_limit', passed=True, details='within_limits'),
                        RiskCheck(name='risk_sizing', passed=True, details=f'risk=${computed_risk:.2f}'),
                    ],
                    model_info=ModelInfo(
                        ml_confidence=cognitive_conf if cognitive_conf else 0.0,
                        regime=str(row.get('regime', 'UNKNOWN')),
                    ),
                    config_hash=config_pin or '',
                    metadata={
                        'decision_id': rec.decision_id,
                        'size_multiplier': size_multiplier,
                        'oversold_tier': row.get('oversold_tier', 'UNKNOWN'),
                    },
                )
                card_logger.save_card(card)
                jlog('decision_card_created', card_id=card.card_id, symbol=sym)
            except Exception as e:
                jlog('decision_card_error', symbol=sym, error=str(e), level='WARN')

        # Display with cognitive info if available
        cog_info = f" cog={cognitive_conf:.2f}" if cognitive_conf is not None else ""
        print(f"{sym} -> {rec.status} @ {limit_px} qty {max_qty}{cog_info} note={rec.notes}")
        if str(rec.status).upper().endswith('SUBMITTED'):
            update_request_counter('orders_submitted', 1)
        elif str(rec.status).upper().endswith('REJECTED'):
            update_request_counter('orders_rejected', 1)
        submitted += 1
    print(f"Submitted {submitted} IOC LIMIT orders.")


if __name__ == '__main__':
    main()
