#!/usr/bin/env python3
"""
Kobe Daily Scheduler - Orchestrates the daily trading cycle.

Daily Cycle:
1. EOD Watchlist (3:30 PM ET): Scan 800 stocks -> Top 5 watchlist -> Top 2 trades for NEXT day
2. Morning Validation (9:45 AM ET): Validate watchlist with fresh 15-min data
3. Trading Window (10:00 AM ET): Execute validated Top 2 trades

Schedule (ET):
- 15:30 (3:30 PM) - EOD scan, generate next day's watchlist
- 09:45 (9:45 AM) - Morning validation with intraday data
- 10:00 (10:00 AM) - Execute trades from validated watchlist

This runs continuously, checking the schedule every minute.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import time
from datetime import datetime, date, timedelta
from typing import Dict, Any
import argparse

from dotenv import load_dotenv
load_dotenv()

from core.clock.tz_utils import now_et
from core.structured_log import jlog
from core.kill_switch import is_kill_switch_active
from market_calendar import is_market_closed

# Import scanning modules
from strategies.dual_strategy.combined import DualStrategyScanner, DualStrategyParams
from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.providers.alpaca_intraday import fetch_intraday_bars
from data.universe.loader import load_universe

import pandas as pd


STATE_DIR = ROOT / 'state' / 'scheduler'
STATE_DIR.mkdir(parents=True, exist_ok=True)


def get_state_file(name: str) -> Path:
    return STATE_DIR / f'{name}.json'


def already_ran_today(task: str) -> bool:
    """Check if task already ran today."""
    state_file = get_state_file(task)
    if not state_file.exists():
        return False
    try:
        data = json.loads(state_file.read_text())
        return data.get('date') == date.today().isoformat()
    except (json.JSONDecodeError, IOError):
        return False


def mark_ran(task: str, result: Dict[str, Any] = None):
    """Mark task as completed for today."""
    state_file = get_state_file(task)
    data = {
        'date': date.today().isoformat(),
        'time': datetime.now().isoformat(),
        'result': result or {}
    }
    state_file.write_text(json.dumps(data, indent=2, default=str))


def eod_watchlist_scan(universe_file: str, cap: int = 800) -> Dict[str, Any]:
    """
    EOD Scan: Generate watchlist for NEXT trading day.

    Runs at 3:30 PM ET using today's close data.
    """
    jlog('eod_scan_start', level='INFO')
    print('=' * 80)
    print('EOD WATCHLIST SCAN - For Next Trading Day')
    print(f'Time: {now_et().strftime("%Y-%m-%d %H:%M:%S")} ET')
    print('=' * 80)

    # Load universe
    symbols = load_universe(universe_file, cap=cap)
    print(f'Universe: {len(symbols)} stocks')

    # Initialize scanner
    params = DualStrategyParams()
    scanner = DualStrategyScanner(params)

    # Today's date for data
    today = date.today().isoformat()

    all_signals = []
    ibs_rsi_count = 0
    turtle_soup_count = 0

    print(f'Scanning with data through {today}...')

    for i, sym in enumerate(symbols):
        if (i+1) % 100 == 0:
            print(f'  Progress: {i+1}/{len(symbols)} | Signals: {len(all_signals)}')

        try:
            df = fetch_daily_bars_polygon(sym, start='2024-01-01', end=today, cache_dir=None)
            if df is None or len(df) < 50:
                continue

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            sig = scanner.generate_signals(df)

            if not sig.empty:
                r = sig.iloc[0]
                strategy = r.get('strategy', '')
                entry = float(r.get('entry_price', 0))
                stop = float(r.get('stop_loss', 0))
                target = float(r.get('take_profit', 0)) if pd.notna(r.get('take_profit')) else None

                risk = entry - stop if stop > 0 else 1
                reward = (target - entry) if target else 0
                rr = reward / risk if risk > 0 else 0

                all_signals.append({
                    'symbol': sym,
                    'strategy': strategy,
                    'entry_price': entry,
                    'stop_loss': stop,
                    'take_profit': target,
                    'rr_ratio': rr,
                    'score': float(r.get('score', 50)),
                    'reason': str(r.get('reason', ''))[:60]
                })

                if 'IBS' in strategy:
                    ibs_rsi_count += 1
                elif 'Turtle' in strategy:
                    turtle_soup_count += 1

        except Exception:
            continue

    print()
    print(f'Total signals: {len(all_signals)} (IBS_RSI: {ibs_rsi_count}, TurtleSoup: {turtle_soup_count})')

    # Sort by score and select Top 5
    all_signals.sort(key=lambda x: x['score'], reverse=True)
    top5 = all_signals[:5]

    # Determine TOTD (Trade of the Day)
    totd = top5[0] if top5 else None

    # Calculate next trading day
    next_day = date.today() + timedelta(days=1)
    if next_day.weekday() == 5:  # Saturday
        next_day += timedelta(days=2)
    elif next_day.weekday() == 6:  # Sunday
        next_day += timedelta(days=1)

    result = {
        'generated_at': datetime.now().isoformat(),
        'for_date': next_day.isoformat(),
        'universe_size': len(symbols),
        'signals_found': len(all_signals),
        'ibs_rsi_count': ibs_rsi_count,
        'turtle_soup_count': turtle_soup_count,
        'watchlist_size': len(top5),
        'totd': totd,
        'watchlist': [
            {
                'rank': i+1,
                **s
            }
            for i, s in enumerate(top5)
        ],
        'status': 'READY',
    }

    # Save to state
    watchlist_dir = ROOT / 'state' / 'watchlist'
    watchlist_dir.mkdir(parents=True, exist_ok=True)

    with open(watchlist_dir / 'next_day.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print()
    print('TOP 5 WATCHLIST:')
    print('-' * 80)
    for i, s in enumerate(top5):
        rr_str = f"{s['rr_ratio']:.2f}:1" if s['rr_ratio'] > 0 else 'N/A'
        print(f"  {i+1}. {s['symbol']:5} | {s['strategy']:12} | Score={s['score']:.1f} | R:R={rr_str}")

    print()
    print(f'Watchlist saved for {next_day.isoformat()}')
    jlog('eod_scan_complete', level='INFO', signals=len(all_signals), watchlist_size=len(top5))

    return result


def morning_validation_scan() -> Dict[str, Any]:
    """
    Morning Validation: Validate watchlist with fresh intraday data.

    Runs at 9:45 AM ET (after first 15-min candle closes).
    """
    jlog('morning_validation_start', level='INFO')
    print('=' * 80)
    print('MORNING VALIDATION - Fresh Intraday Data Check')
    print(f'Time: {now_et().strftime("%Y-%m-%d %H:%M:%S")} ET')
    print('=' * 80)

    # Load overnight watchlist
    watchlist_file = ROOT / 'state' / 'watchlist' / 'next_day.json'
    if not watchlist_file.exists():
        print('ERROR: No overnight watchlist found')
        return {'status': 'NO_WATCHLIST'}

    with open(watchlist_file) as f:
        watchlist_data = json.load(f)

    watchlist = watchlist_data.get('watchlist', [])
    print(f'Validating {len(watchlist)} stocks from overnight watchlist...')
    print()

    validated = []
    removed = []

    for item in watchlist:
        symbol = item['symbol']
        entry = item['entry_price']
        stop = item['stop_loss']
        target = item.get('take_profit')
        rr = item.get('rr_ratio', 0)

        try:
            bars = fetch_intraday_bars(symbol, timeframe='15Min', limit=5)
            if bars:
                live = bars[-1].close
                gap = ((live / entry) - 1) * 100 if entry > 0 else 0

                gap_ok = abs(gap) < 3
                rr_ok = rr >= 1.5

                item_result = {
                    **item,
                    'current_price': live,
                    'gap_pct': gap / 100,
                }

                if gap_ok and rr_ok:
                    item_result['validation_status'] = 'VALID'
                    item_result['validation_note'] = f'Gap {abs(gap):.1f}% OK, R:R {rr:.2f}:1 OK'
                    validated.append(item_result)
                    status = 'VALID'
                else:
                    reasons = []
                    if not gap_ok:
                        reasons.append(f'Gap {abs(gap):.1f}% exceeds 3%')
                    if not rr_ok:
                        reasons.append(f'R:R {rr:.2f}:1 below 1.5:1')
                    item_result['validation_status'] = 'GAP_INVALIDATED' if not gap_ok else 'RR_INVALIDATED'
                    item_result['validation_note'] = '; '.join(reasons)
                    removed.append(item_result)
                    status = 'FAIL'

                print(f"  {symbol:5} | Entry=${entry:.2f} | Live=${live:.2f} | Gap={gap:+.1f}% | R:R={rr:.2f}:1 | {status}")
            else:
                item['validation_status'] = 'NO_DATA'
                item['validation_note'] = 'Could not fetch intraday data'
                removed.append(item)
                print(f"  {symbol:5} | No intraday data available")
        except Exception as e:
            item['validation_status'] = 'ERROR'
            item['validation_note'] = str(e)[:50]
            removed.append(item)
            print(f"  {symbol:5} | Error: {str(e)[:40]}")

    # Select Top 2 from validated
    top2 = validated[:2]
    totd = top2[0] if top2 else None

    result = {
        'validated_at': datetime.now().isoformat(),
        'for_date': date.today().isoformat(),
        'overnight_generated': watchlist_data.get('generated_at'),
        'status': 'VALIDATED',
        'summary': {
            'original_count': len(watchlist),
            'validated_count': len(validated),
            'removed_count': len(removed),
            'gap_threshold': 0.03,
            'rr_threshold': 1.5,
        },
        'totd': totd,
        'watchlist': validated,
        'top2': top2,
        'removed': removed,
    }

    # Save validated watchlist
    with open(ROOT / 'state' / 'watchlist' / 'today_validated.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print()
    print('=' * 80)

    if top2:
        print(f'VALIDATED: {len(validated)} | TOP 2 TRADEABLE:')
        print('=' * 80)
        for i, t in enumerate(top2):
            print(f"\nTRADE {i+1}: {t['symbol']} ({t['strategy']})")
            print(f"  Entry:  ${t['entry_price']:.2f}")
            print(f"  Stop:   ${t['stop_loss']:.2f}")
            print(f"  Live:   ${t['current_price']:.2f}")
            print(f"  R:R:    {t.get('rr_ratio', 0):.2f}:1")
    else:
        print('NO VALID TRADES - All signals failed validation')
        print('  Wait for EOD scan at 3:30 PM ET')

    jlog('morning_validation_complete', level='INFO', validated=len(validated), top2=len(top2))

    return result


def run_scheduler(universe_file: str, cap: int = 800, dry_run: bool = False):
    """
    Main scheduler loop - runs continuously checking the schedule.

    Schedule (ET):
    - 09:45 - Morning validation
    - 15:30 - EOD watchlist scan
    """
    print('=' * 80)
    print('KOBE DAILY SCHEDULER')
    print('=' * 80)
    print('Schedule (ET):')
    print('  09:45 - Morning validation (15 min after open)')
    print('  15:30 - EOD watchlist scan (before close)')
    print()
    print(f'Universe: {universe_file}')
    print(f'Cap: {cap}')
    print(f'Dry run: {dry_run}')
    print('=' * 80)
    print()

    while True:
        try:
            # Check kill switch
            if is_kill_switch_active():
                print('Kill switch active - pausing scheduler')
                time.sleep(60)
                continue

            # Check market status
            now = now_et()
            closed, reason = is_market_closed(now)

            if closed:
                print(f'\r[{now.strftime("%H:%M:%S")}] Market closed: {reason}', end='', flush=True)
                time.sleep(60)
                continue

            hour_min = now.hour * 100 + now.minute

            # 09:45 - Morning validation
            if 945 <= hour_min <= 950 and not already_ran_today('morning_validation'):
                print(f'\n[{now.strftime("%H:%M:%S")}] Running morning validation...')
                if not dry_run:
                    result = morning_validation_scan()
                    mark_ran('morning_validation', result)
                else:
                    print('  DRY RUN - skipping execution')
                    mark_ran('morning_validation', {'dry_run': True})

            # 15:30 - EOD watchlist scan
            elif 1530 <= hour_min <= 1535 and not already_ran_today('eod_watchlist'):
                print(f'\n[{now.strftime("%H:%M:%S")}] Running EOD watchlist scan...')
                if not dry_run:
                    result = eod_watchlist_scan(universe_file, cap)
                    mark_ran('eod_watchlist', result)
                else:
                    print('  DRY RUN - skipping execution')
                    mark_ran('eod_watchlist', {'dry_run': True})

            else:
                # Status display
                next_task = 'morning_validation (09:45)' if hour_min < 945 else 'eod_watchlist (15:30)' if hour_min < 1530 else 'tomorrow'
                print(f'\r[{now.strftime("%H:%M:%S")}] Waiting... Next: {next_task}', end='', flush=True)

            time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            print('\nScheduler stopped by user')
            break
        except Exception as e:
            jlog('scheduler_error', level='ERROR', error=str(e))
            print(f'\nError: {e}')
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description='Kobe Daily Scheduler')
    parser.add_argument('--universe', type=str, default='data/universe/optionable_liquid_800.csv')
    parser.add_argument('--cap', type=int, default=800)
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--eod-now', action='store_true', help='Run EOD scan immediately')
    parser.add_argument('--validate-now', action='store_true', help='Run morning validation immediately')
    parser.add_argument('--once', action='store_true', help='Run once and exit')

    args = parser.parse_args()

    if args.eod_now:
        eod_watchlist_scan(args.universe, args.cap)
    elif args.validate_now:
        morning_validation_scan()
    elif args.once:
        # Run both immediately
        morning_validation_scan()
    else:
        run_scheduler(args.universe, args.cap, args.dry_run)


if __name__ == '__main__':
    main()
