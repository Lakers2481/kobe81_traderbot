#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
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
from execution.broker_alpaca import get_best_ask, get_best_bid, construct_decision, place_bracket_order
from risk.policy_gate import PolicyGate, load_limits_from_config
from risk.position_limit_gate import PositionLimitGate, PositionLimits
from risk.equity_sizer import calculate_position_size, get_account_equity, format_size_summary
from risk.weekly_exposure_gate import get_weekly_exposure_gate
from risk.kill_zone_gate import get_kill_zone_gate
from risk.signal_quality_gate import filter_to_best_signals
from risk.advanced import check_portfolio_var
from risk.advanced.correlation_limits import EnhancedCorrelationLimits
from core.hash_chain import append_block
from core.structured_log import jlog
from monitor.health_endpoints import update_request_counter
from core.config_pin import sha256_file
from integration.learning_hub import get_learning_hub

# Import unified enrichment pipeline (CRITICAL: This adds 100+ fields for proper position sizing)
try:
    from pipelines.unified_signal_enrichment import run_full_enrichment
    ENRICHMENT_AVAILABLE = True
except ImportError:
    ENRICHMENT_AVAILABLE = False

# CRITICAL FIX (2026-01-08): Fake data detection before trading
try:
    from validation.fake_data_detector import validate_signals_before_trading, FakeDataError
    FAKE_DATA_DETECTION_AVAILABLE = True
except ImportError:
    FAKE_DATA_DETECTION_AVAILABLE = False
    FakeDataError = Exception

import json

# Position state file for position_manager.py to track time-based exits
POSITION_STATE_FILE = ROOT / 'state' / 'position_state.json'


def register_position_entry(
    symbol: str,
    entry_price: float,
    qty: int,
    side: str,
    stop_loss: float,
    strategy: str,
) -> None:
    """
    Register a new position in position_state.json for time-based exit tracking.

    This allows position_manager.py to know:
    - Entry date (for bars held calculation)
    - Strategy (for correct time stop: 3-bar for Turtle Soup, 7-bar for IBS+RSI)
    - Stop loss (for price stop checking)
    """
    POSITION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load existing state
    state = {}
    if POSITION_STATE_FILE.exists():
        try:
            with open(POSITION_STATE_FILE, 'r') as f:
                state = json.load(f)
        except Exception:
            state = {}

    # Add/update position
    state[symbol] = {
        'symbol': symbol,
        'entry_date': datetime.utcnow().strftime('%Y-%m-%d'),
        'entry_price': entry_price,
        'qty': qty,
        'side': side,
        'stop_loss': stop_loss,
        'initial_stop': stop_loss,
        'strategy': strategy,
        'bars_held': 0,
        'last_check': datetime.utcnow().isoformat(),
        'current_price': None,
        'unrealized_pnl': None,
        'r_multiple': 0.0,
        'stop_state': 'initial',
        'should_exit': False,
        'exit_reason': None,
    }

    # Save state
    try:
        with open(POSITION_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        jlog('position_state_registered', symbol=symbol, strategy=strategy, stop_loss=stop_loss)
    except Exception as e:
        jlog('position_state_register_error', symbol=symbol, error=str(e), level='WARN')

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
    ap.add_argument('--max-spread-pct', type=float, default=0.02, help='Max bid/ask spread as fraction of mid (default 0.02)')
    ap.add_argument('--cognitive', action='store_true', default=True, help='Enable cognitive brain (ON by default)')
    ap.add_argument('--no-cognitive', action='store_true', help='Disable cognitive brain')
    ap.add_argument('--cognitive-min-conf', type=float, default=0.5, help='Min cognitive confidence to trade')
    ap.add_argument('--watchlist-only', action='store_true', help='Only trade stocks from validated watchlist')
    ap.add_argument('--watchlist-path', type=str, default='state/watchlist/today_validated.json', help='Path to validated watchlist')
    ap.add_argument('--fallback-enabled', action='store_true', help='Enable fallback scan if watchlist fails')
    ap.add_argument('--fallback-min-score', type=int, default=75, help='Min quality score for fallback trades (default 75)')
    ap.add_argument('--skip-wf-check', action='store_true', help='Skip walk-forward validation check (NOT RECOMMENDED)')
    ap.add_argument('--live', action='store_true', help='Enable LIVE trading (REAL MONEY - requires confirmation)')
    ap.add_argument('--confirm-live', type=str, default='', help='Confirmation token for live trading (must be "I_UNDERSTAND_REAL_MONEY")')
    args = ap.parse_args()

    # Handle --no-cognitive override
    if args.no_cognitive:
        args.cognitive = False

    # Env
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # ==========================================================================
    # LIVE MODE SAFETY GATE
    # ==========================================================================
    if args.live:
        # Require explicit confirmation
        if args.confirm_live != "I_UNDERSTAND_REAL_MONEY":
            print("=" * 70)
            print("LIVE TRADING REQUIRES EXPLICIT CONFIRMATION")
            print("=" * 70)
            print()
            print("You are about to trade with REAL MONEY.")
            print("This will use your LIVE Alpaca account.")
            print()
            print("To proceed, add this flag:")
            print("  --confirm-live I_UNDERSTAND_REAL_MONEY")
            print()
            print("Example:")
            print("  python scripts/run_paper_trade.py --live --confirm-live I_UNDERSTAND_REAL_MONEY --universe ...")
            print("=" * 70)
            return

        # Check NO_LIVE_ORDERS safety gate
        try:
            from safety.execution_choke import NO_LIVE_ORDERS
            if NO_LIVE_ORDERS:
                print("=" * 70)
                print("ERROR: Live trading is BLOCKED by NO_LIVE_ORDERS flag")
                print("=" * 70)
                print("Edit safety/execution_choke.py and set NO_LIVE_ORDERS = False to enable")
                print("=" * 70)
                return
        except ImportError:
            print("WARNING: Could not import safety.execution_choke - proceeding with caution")

        # Set LIVE API endpoint
        os.environ['ALPACA_BASE_URL'] = 'https://api.alpaca.markets'
        print("\n" + "!" * 70)
        print("!!! LIVE TRADING MODE - REAL MONEY !!!")
        print("!" * 70 + "\n")
        jlog('live_mode_enabled', confirmed=True, confirmation_token='I_UNDERSTAND_REAL_MONEY')
    else:
        # Mode: ensure paper endpoint (DEFAULT - safe)
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

    # Weekly Exposure Gate - Professional Portfolio Allocation
    weekly_gate = get_weekly_exposure_gate()

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

    # Weekly budget status - Professional Portfolio Allocation
    weekly_status = weekly_gate.get_status()
    print("\nWeekly Budget Status:")
    print(f"  Week: {weekly_status['week']}")
    print(f"  Current Exposure: {weekly_status['exposure']['current_pct']} of {weekly_status['exposure']['max_weekly_pct']} max")
    print(f"  Daily Entries Today: {weekly_status['daily']['entries_today']}/{weekly_status['daily']['max_per_day']}")
    print(f"  Weekly Positions: {weekly_status['positions']['total_this_week']}")
    can_enter, block_reason = weekly_gate.check_can_enter('TEST', 0, get_account_equity())
    if not can_enter:
        print(f"  WARNING: Cannot enter new positions - {block_reason}")

    # === KILL ZONE GATE - Professional Time-Based Trade Blocking ===
    kill_zone_gate = get_kill_zone_gate()
    kz_status = kill_zone_gate.check_can_trade()
    print("\nKill Zone Status:")
    print(f"  Current Zone: {kz_status.current_zone.value}")
    print(f"  Can Trade: {kz_status.can_trade}")
    print(f"  Reason: {kz_status.reason}")
    if kz_status.next_window_opens:
        print(f"  Next Window: {kz_status.next_window_opens}")
        if kz_status.minutes_until_window:
            print(f"  Minutes Until: {kz_status.minutes_until_window}")

    if not kz_status.can_trade:
        jlog('kill_zone_block', zone=kz_status.current_zone.value, reason=kz_status.reason)
        print(f"\n{'='*60}")
        print(f"TRADING BLOCKED: {kz_status.reason}")
        print(f"{'='*60}")
        print("No trades will be executed during this kill zone.")
        if kz_status.next_window_opens:
            print(f"Next trading window opens at {kz_status.next_window_opens}")
        return

    # === WALK-FORWARD VALIDATION GATE ===
    # MNIST analogy: Don't deploy a model without testing on held-out data first
    # This ensures strategy has been validated before trading real money
    if not args.skip_wf_check:
        print("\n" + "="*60)
        print("WALK-FORWARD VALIDATION CHECK")
        print("="*60)

        wf_dir = ROOT / 'wf_outputs'
        wf_valid = False
        wf_reason = "No walk-forward results found"

        if wf_dir.exists():
            # Look for the most recent summary file
            summary_files = list(wf_dir.glob('**/wf_summary.json')) + list(wf_dir.glob('**/summary.json'))

            if summary_files:
                # Get most recent
                latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
                age_days = (datetime.now().timestamp() - latest_summary.stat().st_mtime) / 86400

                try:
                    import json
                    with open(latest_summary) as f:
                        wf_results = json.load(f)

                    # Check minimum criteria (like MNIST test accuracy threshold)
                    win_rate = wf_results.get('win_rate', wf_results.get('oos_win_rate', 0))
                    profit_factor = wf_results.get('profit_factor', wf_results.get('oos_profit_factor', 0))

                    print(f"  Latest WF results: {latest_summary.name}")
                    print(f"  Age: {age_days:.1f} days")
                    print(f"  Win Rate: {win_rate:.1%}")
                    print(f"  Profit Factor: {profit_factor:.2f}")

                    # Minimum thresholds (like requiring >80% MNIST accuracy before deployment)
                    if age_days > 30:
                        wf_reason = f"Walk-forward results too old ({age_days:.0f} days > 30 day limit)"
                    elif win_rate < 0.50:
                        wf_reason = f"Walk-forward win rate too low ({win_rate:.1%} < 50% minimum)"
                    elif profit_factor < 1.0:
                        wf_reason = f"Walk-forward profit factor too low ({profit_factor:.2f} < 1.0 minimum)"
                    else:
                        wf_valid = True
                        print("  [PASS] Walk-forward validation passed!")

                except Exception as e:
                    wf_reason = f"Error reading WF results: {e}"

        if not wf_valid:
            jlog('wf_validation_failed', reason=wf_reason)
            print(f"\n  [FAIL] {wf_reason}")
            print("\n  This is the EVAL step in the MNIST loop.")
            print("  Run walk-forward validation first:")
            print("    python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_900.csv --start 2023-01-01 --end 2024-12-31")
            print("\n  Or use --skip-wf-check to bypass (NOT RECOMMENDED)")
            print("="*60)
            return
    else:
        print("\n  [WARN] Walk-forward validation check SKIPPED (--skip-wf-check)")
        jlog('wf_validation_skipped', warning='User bypassed WF check')

    # === WATCHLIST-ONLY MODE ===
    # If --watchlist-only is set, restrict to validated watchlist symbols
    watchlist_symbols = []
    watchlist_data = {}
    if args.watchlist_only:
        watchlist_file = ROOT / args.watchlist_path
        if watchlist_file.exists():
            try:
                import json
                with open(watchlist_file) as f:
                    watchlist_data = json.load(f)
                watchlist_stocks = watchlist_data.get('watchlist', [])
                watchlist_symbols = [s['symbol'] for s in watchlist_stocks if s.get('symbol')]
                print(f"\n{'='*60}")
                print("WATCHLIST-ONLY MODE")
                print(f"{'='*60}")
                print(f"Validated watchlist: {len(watchlist_symbols)} stocks")
                for s in watchlist_stocks[:5]:
                    status = s.get('validation_status', 'UNKNOWN')
                    sym = s.get('symbol', '?')
                    print(f"  [{status}] {sym}")
                # Filter universe to only watchlist symbols
                symbols = [s for s in symbols if s in watchlist_symbols]
                if not symbols:
                    print("WARNING: No watchlist symbols found in universe!")
                    if args.fallback_enabled:
                        print("Fallback mode enabled - will scan full universe with higher bar")
                        symbols = load_universe(Path(args.universe), cap=args.cap)
                    else:
                        print("No watchlist and fallback disabled. Exiting.")
                        return
                jlog('watchlist_mode', symbols=watchlist_symbols, filtered_count=len(symbols))
            except Exception as e:
                jlog('watchlist_load_error', error=str(e), level='WARN')
                print(f"WARNING: Could not load watchlist: {e}")
                if not args.fallback_enabled:
                    print("Fallback disabled. Exiting.")
                    return
        else:
            jlog('watchlist_not_found', path=str(watchlist_file), level='WARN')
            print(f"WARNING: No validated watchlist found at {watchlist_file}")
            if not args.fallback_enabled:
                print("Fallback disabled. Exiting.")
                return
            print("Fallback mode enabled - will scan full universe with higher bar")

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
    todays = sigs[sigs['timestamp'] == last_ts].copy()

    # === CRITICAL FIX (2026-01-08): Run unified enrichment pipeline ===
    # This adds 100+ fields: kelly_optimal_pct, regime, vix_level, final_conf_score, etc.
    # Without this, position sizing uses DEFAULTS instead of calculated values!
    if ENRICHMENT_AVAILABLE and not todays.empty:
        jlog('enrichment_start', count=len(todays))
        print(f"[ENRICHMENT] Running unified pipeline on {len(todays)} signals...")
        try:
            # Get SPY and VIX data for market context
            spy_bars = None
            vix_df = None
            try:
                spy_bars = fetch_daily_bars_polygon('SPY', args.start, args.end, cache_dir=cache_dir)
            except Exception:
                pass

            # Try to get VIX from FRED or use default
            try:
                from data.vix_data import get_current_vix
                vix_level = get_current_vix()
                vix_df = pd.DataFrame({'timestamp': [datetime.now()], 'close': [vix_level], 'vix': [vix_level]})
            except Exception:
                vix_df = pd.DataFrame({'timestamp': [datetime.now()], 'close': [20.0], 'vix': [20.0]})

            # Run enrichment
            enriched_signals, _ = run_full_enrichment(
                signals=todays,
                price_data=data,
                spy_data=spy_bars,
                vix_data=vix_df,
                verbose=False,
            )

            if enriched_signals:
                # Convert enriched signals to DataFrame with ALL fields
                todays = pd.DataFrame([s.to_dict() for s in enriched_signals])
                jlog('enrichment_complete', count=len(todays), columns=len(todays.columns))
                print(f"[ENRICHMENT] Complete: {len(todays)} signals with {len(todays.columns)} fields")
            else:
                jlog('enrichment_no_signals', level='WARN')
                print("[ENRICHMENT] No signals returned from pipeline")
        except Exception as e:
            jlog('enrichment_error', error=str(e), level='WARN')
            print(f"[ENRICHMENT] Error: {e} - continuing with raw signals")
    elif not ENRICHMENT_AVAILABLE:
        jlog('enrichment_unavailable', level='WARN')
        print("[WARN] Enrichment pipeline not available - using raw signals with DEFAULTS")

    # Apply earnings filter if enabled
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

    # CRITICAL FIX (2026-01-08): Validate for fake/hardcoded data BEFORE trading
    if FAKE_DATA_DETECTION_AVAILABLE and not todays.empty:
        try:
            alerts = validate_signals_before_trading(todays, halt_on_critical=True)
            if alerts:
                print(f"[FAKE DATA WARNING] {len(alerts)} non-critical alerts detected")
                for alert in alerts:
                    jlog('fake_data_warning', field=alert.field, description=alert.description)
        except FakeDataError as e:
            print(f"\n{'='*60}")
            print(f"TRADING HALTED: FAKE DATA DETECTED")
            print(f"{'='*60}")
            print(e.get_summary())
            print(f"\nFix the data issues before trading. Aborting.")
            jlog('trading_halted', reason='fake_data_detected', alerts=len(e.alerts))
            return

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

    # SELECT BEST 2 SIGNALS (Professional Portfolio Allocation)
    # Sort by cognitive confidence (or conf_score) and take top 2
    max_trades_per_day = 2  # Professional limit
    if not todays.empty:
        # Get confidence column
        if 'cognitive_confidence' in todays.columns:
            conf_col = 'cognitive_confidence'
        elif 'conf_score' in todays.columns:
            conf_col = 'conf_score'
        else:
            conf_col = None

        if conf_col and len(todays) > max_trades_per_day:
            todays = todays.sort_values(conf_col, ascending=False).head(max_trades_per_day)
            print(f"\n  TOP {len(todays)} TRADES FOR TODAY:")
            for i, (_, sig) in enumerate(todays.iterrows(), 1):
                print(f"    {i}. {sig['symbol']} (conf={sig.get(conf_col, 'N/A')})")
            jlog('top_trades_selected', count=len(todays),
                 symbols=[r['symbol'] for _, r in todays.iterrows()])
        elif len(todays) == 1:
            print(f"  Single signal: {todays.iloc[0]['symbol']}")
        else:
            print(f"\n  TOP {len(todays)} TRADES FOR TODAY:")
            for i, (_, sig) in enumerate(todays.iterrows(), 1):
                conf_val = sig.get(conf_col, 'N/A') if conf_col else 'N/A'
                print(f"    {i}. {sig['symbol']} (conf={conf_val})")

    # Check weekly budget before proceeding
    if not todays.empty:
        account_equity = get_account_equity()
        available_daily = weekly_gate.get_available_daily_budget(account_equity)
        available_weekly = weekly_gate.get_available_weekly_budget(account_equity)
        print(f"\n  Available Budget: ${available_daily:,.0f} daily / ${available_weekly:,.0f} weekly")

        if available_daily <= 0:
            jlog('daily_budget_exhausted', level='WARN')
            print("  WARNING: Daily budget exhausted - no new positions today")
            todays = pd.DataFrame()
        elif available_weekly <= 0:
            jlog('weekly_budget_exhausted', level='WARN')
            print("  WARNING: Weekly budget exhausted - no new positions this week")
            todays = pd.DataFrame()

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
        # EQUITY-BASED POSITION SIZING WITH FULL FACTOR INTEGRATION
        # FIX (2026-01-08): Wire Kelly, Regime, VIX, Confidence multipliers
        limit_px = round(ask * 1.001, 2)

        # Get stop loss from signal (fallback to 5% if missing)
        stop_loss = row.get('stop_loss')
        if stop_loss is None or pd.isna(stop_loss):
            stop_loss = limit_px * 0.95  # 5% fallback

        # Get cognitive multiplier and confidence
        size_multiplier = 1.0
        cognitive_conf = None
        if sym in cognitive_decisions:
            cog = cognitive_decisions[sym]
            size_multiplier = cog.get('size_multiplier', 1.0)
            cognitive_conf = cog.get('confidence')

        # === TIER 1.1: KELLY SIZING ===
        # Use kelly_optimal_pct from signal enrichment, cap at 5%
        kelly_pct = row.get('kelly_optimal_pct', risk_limits.risk_per_trade_pct)
        if pd.isna(kelly_pct) or kelly_pct <= 0:
            kelly_pct = risk_limits.risk_per_trade_pct
        risk_pct = min(kelly_pct, 0.05)  # Cap at 5% max
        jlog('kelly_applied', symbol=sym, kelly=kelly_pct, capped=risk_pct)

        # === TIER 1.2: REGIME MULTIPLIER ===
        regime = row.get('regime', 'NEUTRAL')
        if pd.isna(regime):
            regime = 'NEUTRAL'
        regime_mult = {'BULL': 1.5, 'NEUTRAL': 1.0, 'BEAR': 0.5}.get(str(regime).upper(), 1.0)
        risk_pct *= regime_mult
        jlog('regime_applied', symbol=sym, regime=regime, multiplier=regime_mult, new_risk_pct=risk_pct)

        # === TIER 1.3: VIX ADJUSTMENT ===
        vix_level = row.get('vix_level', 20)
        if pd.isna(vix_level):
            vix_level = 20
        if vix_level > 30:
            vix_mult = 0.5  # High fear = half size
        elif vix_level > 25:
            vix_mult = 0.75
        else:
            vix_mult = 1.0
        risk_pct *= vix_mult
        jlog('vix_applied', symbol=sym, vix=vix_level, multiplier=vix_mult, new_risk_pct=risk_pct)

        # === TIER 1.4: CONFIDENCE MULTIPLIER ===
        # NOTE: EnrichedSignal uses 'final_conf_score' not 'final_confidence'
        final_confidence = row.get('final_conf_score', row.get('final_confidence', 0.5))
        if pd.isna(final_confidence):
            final_confidence = 0.5
        conf_mult = 0.5 + (final_confidence * 0.5)  # Range: 0.5 to 1.0
        risk_pct *= conf_mult
        jlog('confidence_applied', symbol=sym, confidence=final_confidence, multiplier=conf_mult, final_risk_pct=risk_pct)

        # Ensure risk_pct stays within bounds (min 0.5%, max 5%)
        risk_pct = max(0.005, min(risk_pct, 0.05))
        jlog('final_risk_pct', symbol=sym, risk_pct=risk_pct)

        # Calculate position size with all factors applied
        pos_size = calculate_position_size(
            entry_price=limit_px,
            stop_loss=float(stop_loss),
            risk_pct=risk_pct,  # Now uses Kelly * Regime * VIX * Confidence
            cognitive_multiplier=size_multiplier,
            max_notional_pct=0.10,  # Max 10% of account per position
        )
        max_qty = pos_size.shares

        # Log computed risk for audit trail
        jlog('risk_sizing', symbol=sym, entry=limit_px, stop=float(stop_loss),
             risk_per_share=pos_size.risk_per_share,
             risk_dollars=pos_size.risk_dollars,
             account_equity=pos_size.account_equity,
             risk_pct=pos_size.risk_pct,
             qty=max_qty, notional=pos_size.notional,
             multiplier=size_multiplier, capped=pos_size.capped)
        print(f"  {format_size_summary(pos_size, sym)}")
        ok, reason = policy.check(sym, 'long' if side=='BUY' else 'short', limit_px, max_qty, float(stop_loss))
        if not ok:
            jlog('policy_veto', symbol=sym, reason=reason, price=limit_px, qty=max_qty)
            print(f"VETO {sym}: {reason}")
            continue
        # Position limit check
        pos_ok, pos_reason = position_gate.check(sym, 'long' if side=='BUY' else 'short', limit_px, max_qty)
        if not pos_ok:
            jlog('position_limit_veto', symbol=sym, reason=pos_reason)
            print(f"VETO {sym}: {pos_reason}")
            continue

        # === TIER 1.5: VaR GATE ===
        # Check portfolio VaR before adding new position
        try:
            # Get current positions for VaR calculation
            current_positions = position_gate.get_open_symbols()
            var_ok, var_result = check_portfolio_var(positions=current_positions)
            if not var_ok:
                jlog('var_gate_blocked', symbol=sym, var_pct=var_result.get('var_pct', 0), limit=0.05)
                print(f"VETO {sym}: VaR limit exceeded ({var_result.get('var_pct', 0):.1%} > 5%)")
                continue
            jlog('var_gate_passed', symbol=sym, var_pct=var_result.get('var_pct', 0))
        except Exception as e:
            jlog('var_check_error', symbol=sym, error=str(e), level='WARN')
            # Continue anyway if VaR check fails

        # === TIER 4.2: NEWS SENTIMENT FILTER ===
        news_sentiment = row.get('news_sentiment', 0)
        if pd.notna(news_sentiment) and news_sentiment < -0.5:
            jlog('news_filter_blocked', symbol=sym, sentiment=news_sentiment)
            print(f"VETO {sym}: Very negative news sentiment ({news_sentiment:.2f})")
            continue

        # === TIER 4.3: SECTOR STRENGTH CHECK ===
        # NOTE: EnrichedSignal uses 'sector_relative_strength' not 'sector_strength'
        sector_strength = row.get('sector_relative_strength', row.get('sector_strength', 0))
        if pd.notna(sector_strength) and sector_strength < -0.3:
            # Don't block, but reduce size
            sector_reduction = 0.75
            max_qty = int(max_qty * sector_reduction)
            if max_qty < 1:
                max_qty = 1
            jlog('sector_adjustment', symbol=sym, sector_strength=sector_strength, reduction=sector_reduction)
            print(f"  Sector weak ({sector_strength:.2f}), reducing qty to {max_qty}")

        # === TIER 4.4: CORRELATION GATE ===
        try:
            corr_checker = EnhancedCorrelationLimits()
            current_positions_for_corr = position_gate.get_open_symbols()
            corr_ok, corr_reason = corr_checker.check_new_position(sym, current_positions_for_corr)
            if not corr_ok:
                jlog('correlation_blocked', symbol=sym, reason=corr_reason)
                print(f"VETO {sym}: {corr_reason}")
                continue
            jlog('correlation_passed', symbol=sym)
        except Exception as e:
            jlog('correlation_check_error', symbol=sym, error=str(e), level='WARN')
            # Continue anyway if correlation check fails

        # Weekly exposure gate check (Professional Portfolio Allocation)
        notional = max_qty * limit_px
        weekly_ok, weekly_reason = weekly_gate.check_can_enter(sym, notional, pos_size.account_equity)
        if not weekly_ok:
            jlog('weekly_gate_veto', symbol=sym, reason=weekly_reason, notional=notional)
            print(f"VETO {sym}: {weekly_reason}")
            continue

        construct_decision(sym, 'long' if side=='BUY' else 'short', max_qty, ask)

        # Calculate take profit (2R target)
        risk_per_share = limit_px - float(stop_loss)
        take_profit_px = round(limit_px + (risk_per_share * 2), 2)  # 2:1 R:R

        # Use bracket order with automatic stop loss and take profit
        bracket_result = place_bracket_order(
            symbol=sym,
            side='buy' if side == 'BUY' else 'sell',
            qty=max_qty,
            limit_price=limit_px,
            stop_loss=float(stop_loss),
            take_profit=take_profit_px,
            time_in_force='gtc',  # Good til canceled for swing trades
        )
        rec = bracket_result.order

        jlog('bracket_order_placed', symbol=sym, entry=limit_px, stop=float(stop_loss),
             target=take_profit_px, qty=max_qty, stop_order_id=bracket_result.stop_order_id,
             tp_order_id=bracket_result.profit_order_id)
        print(f"  BRACKET ORDER: entry=${limit_px:.2f}, stop=${float(stop_loss):.2f}, target=${take_profit_px:.2f}")

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

        # Record entry with weekly exposure gate (Professional Portfolio Allocation)
        if str(rec.status).upper().endswith('SUBMITTED') or str(rec.status).upper() == 'ACCEPTED':
            weekly_gate.record_entry(
                symbol=sym,
                notional=notional,
                account_equity=pos_size.account_equity,
            )
            jlog('weekly_entry_recorded', symbol=sym, notional=notional,
                 new_exposure_pct=weekly_gate.get_status()['current_exposure_pct'])

            # Register position for time-based exit tracking (position_manager.py)
            # This ensures correct time stops: 3-bar for TurtleSoup, 7-bar for IBS_RSI
            strategy_name = row.get('strategy', 'IBS_RSI')
            register_position_entry(
                symbol=sym,
                entry_price=limit_px,
                qty=max_qty,
                side='long' if side == 'BUY' else 'short',
                stop_loss=float(stop_loss),
                strategy=strategy_name,
            )

            # FIX (2026-01-07): Wire Learning Hub to trade entry for feedback loop
            try:
                hub = get_learning_hub()
                hub.record_trade_entry({
                    'symbol': sym,
                    'side': 'long' if side == 'BUY' else 'short',
                    'entry_price': limit_px,
                    'shares': max_qty,
                    'strategy': strategy_name,
                    'trade_id': rec.decision_id,
                    'entry_time': datetime.utcnow().isoformat(),
                    'stop_loss': float(stop_loss),
                    'take_profit': take_profit_px,
                    'signal_score': row.get('conf_score', 0),
                    'regime': 'unknown',  # Would be set by regime detector
                })
                jlog('learning_hub_entry_recorded', symbol=sym, trade_id=rec.decision_id)
            except Exception as e:
                jlog('learning_hub_entry_error', symbol=sym, error=str(e), level='WARN')

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
                        RiskCheck(name='risk_sizing', passed=True, details=f'risk=${pos_size.risk_dollars:.2f}'),
                        RiskCheck(name='weekly_exposure', passed=True,
                                  details=f'exp={weekly_gate.get_status()["exposure"]["current_pct"]}'),
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
