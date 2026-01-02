#!/usr/bin/env python3
"""
Overnight Watchlist Builder - Generates next day's Top 5 watchlist.

Runs at 3:30 PM to:
1. Scan 900 stocks for NEXT DAY setups
2. Generate Top 5 watchlist + TOTD
3. Save to state/watchlist/next_day.json
4. Optionally prefetch data for Top 5 only (saves API calls)

Usage:
    python scripts/overnight_watchlist.py --cap 900
    python scripts/overnight_watchlist.py --cap 900 --prefetch
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
import pytz

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def build_overnight_watchlist(
    universe_path: str,
    cap: int = 900,
    top_n: int = 5,
    dotenv_path: str = "./.env",
    prefetch: bool = False,
) -> Dict:
    """
    Build the overnight watchlist for next trading day.

    Args:
        universe_path: Path to universe CSV
        cap: Max stocks to scan
        top_n: Number of stocks for watchlist
        dotenv_path: Path to .env file
        prefetch: Whether to prefetch data for watchlist stocks

    Returns:
        Dict with watchlist data
    """
    load_dotenv(dotenv_path)

    ET = pytz.timezone('America/New_York')
    now = datetime.now(ET)

    # Determine next trading day
    from data.calendar import get_next_trading_day
    try:
        next_day = get_next_trading_day(now.date())
    except:
        # Fallback: assume next weekday
        next_day = now.date() + timedelta(days=1)
        while next_day.weekday() >= 5:  # Skip weekends
            next_day += timedelta(days=1)

    logger.info(f"Building overnight watchlist for {next_day}")

    # Import scanner
    from strategies.dual_strategy import DualStrategyScanner, DualStrategyParams
    from data.universe.loader import load_universe
    from data.providers.polygon_eod import PolygonEODProvider

    # Load universe
    symbols = load_universe(universe_path, cap=cap)
    logger.info(f"Loaded {len(symbols)} symbols from universe")

    # Initialize scanner with frozen params
    params = DualStrategyParams()
    scanner = DualStrategyScanner(params)
    provider = PolygonEODProvider()

    # Scan for setups
    all_signals = []
    scanned = 0
    errors = 0

    for symbol in symbols:
        try:
            # Get recent data (last 60 days for indicators)
            end_date = now.strftime('%Y-%m-%d')
            start_date = (now - timedelta(days=90)).strftime('%Y-%m-%d')

            df = provider.get_bars(symbol, start_date, end_date)
            if df is None or len(df) < 30:
                continue

            # Generate signals
            signals = scanner.scan_signals_over_time(df)
            if signals is not None and len(signals) > 0:
                # Get the most recent signal
                latest = signals.iloc[-1].to_dict()
                latest['symbol'] = symbol
                all_signals.append(latest)

            scanned += 1
            if scanned % 100 == 0:
                logger.info(f"Scanned {scanned}/{len(symbols)} stocks, {len(all_signals)} signals")

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.debug(f"Error scanning {symbol}: {e}")

    logger.info(f"Scan complete: {scanned} stocks, {len(all_signals)} signals, {errors} errors")

    # Score and rank signals
    scored_signals = []
    for sig in all_signals:
        # Calculate quality score
        score = calculate_overnight_score(sig)
        sig['overnight_score'] = score
        scored_signals.append(sig)

    # Sort by score descending
    scored_signals.sort(key=lambda x: x.get('overnight_score', 0), reverse=True)

    # Take top N
    watchlist = scored_signals[:top_n]

    # TOTD is the highest scoring
    totd = watchlist[0] if watchlist else None

    # Build result
    result = {
        'generated_at': now.isoformat(),
        'for_date': str(next_day),
        'universe_size': len(symbols),
        'signals_found': len(all_signals),
        'watchlist_size': len(watchlist),
        'totd': {
            'symbol': totd['symbol'],
            'score': totd.get('overnight_score', 0),
            'strategy': totd.get('strategy', 'unknown'),
            'entry_price': totd.get('entry_price', 0),
            'stop_loss': totd.get('stop_loss', 0),
            'take_profit': totd.get('take_profit', 0),
        } if totd else None,
        'watchlist': [
            {
                'rank': i + 1,
                'symbol': sig['symbol'],
                'score': sig.get('overnight_score', 0),
                'strategy': sig.get('strategy', 'unknown'),
                'entry_price': sig.get('entry_price', 0),
                'stop_loss': sig.get('stop_loss', 0),
                'take_profit': sig.get('take_profit', 0),
                'reason': sig.get('reason', ''),
            }
            for i, sig in enumerate(watchlist)
        ],
        'status': 'READY',
        'notes': f"Scanned {scanned} stocks, found {len(all_signals)} setups",
    }

    # Save to state file
    state_dir = ROOT / 'state' / 'watchlist'
    state_dir.mkdir(parents=True, exist_ok=True)

    output_path = state_dir / 'next_day.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"Saved watchlist to {output_path}")

    # Optionally prefetch data for watchlist stocks
    if prefetch and watchlist:
        logger.info("Prefetching data for watchlist stocks...")
        watchlist_symbols = [s['symbol'] for s in watchlist]
        prefetch_watchlist_data(watchlist_symbols, provider)

    return result


def calculate_overnight_score(signal: Dict) -> float:
    """
    Calculate overnight score for ranking signals.

    Higher score = better setup for next day.
    """
    score = 50.0  # Base score

    # Quality score if available
    quality = signal.get('quality_score', signal.get('score', 0))
    score += quality * 0.3

    # Confidence
    confidence = signal.get('confidence', 0.5)
    score += confidence * 20

    # Risk/Reward
    entry = signal.get('entry_price', 0)
    stop = signal.get('stop_loss', 0)
    target = signal.get('take_profit', 0)

    if entry and stop and target and entry != stop:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = reward / risk if risk > 0 else 0
        if rr >= 2.0:
            score += 15
        elif rr >= 1.5:
            score += 10
        elif rr >= 1.0:
            score += 5

    # Strategy bonus
    strategy = signal.get('strategy', '').lower()
    if 'turtle' in strategy or 'ict' in strategy:
        score += 5  # ICT strategies get slight bonus
    if 'ibs' in strategy:
        score += 3  # Mean reversion

    # Sweep strength (for turtle soup)
    sweep = signal.get('sweep_strength', 0)
    if sweep >= 0.5:
        score += 10
    elif sweep >= 0.3:
        score += 5

    return round(score, 2)


def prefetch_watchlist_data(symbols: List[str], provider) -> None:
    """Prefetch and cache data for watchlist symbols."""
    from datetime import datetime, timedelta
    import pytz

    ET = pytz.timezone('America/New_York')
    now = datetime.now(ET)

    end_date = now.strftime('%Y-%m-%d')
    start_date = (now - timedelta(days=60)).strftime('%Y-%m-%d')

    for symbol in symbols:
        try:
            df = provider.get_bars(symbol, start_date, end_date)
            if df is not None:
                logger.debug(f"Prefetched {len(df)} bars for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to prefetch {symbol}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Build overnight watchlist for next trading day')
    parser.add_argument('--universe', type=str, default='data/universe/optionable_liquid_900.csv')
    parser.add_argument('--cap', type=int, default=900)
    parser.add_argument('--top', type=int, default=5, help='Number of stocks for watchlist')
    parser.add_argument('--dotenv', type=str, default='./.env')
    parser.add_argument('--prefetch', action='store_true', help='Prefetch data for watchlist stocks')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    result = build_overnight_watchlist(
        universe_path=args.universe,
        cap=args.cap,
        top_n=args.top,
        dotenv_path=args.dotenv,
        prefetch=args.prefetch,
    )

    # Print summary
    print()
    print("=" * 60)
    print(f"OVERNIGHT WATCHLIST FOR {result['for_date']}")
    print("=" * 60)
    print()

    if result['totd']:
        print(f"TRADE OF THE DAY: {result['totd']['symbol']}")
        print(f"  Score: {result['totd']['score']}")
        print(f"  Strategy: {result['totd']['strategy']}")
        print()

    print("TOP 5 WATCHLIST:")
    for stock in result['watchlist']:
        print(f"  {stock['rank']}. {stock['symbol']} (score: {stock['score']:.1f})")
        print(f"     Entry: ${stock['entry_price']:.2f} | Stop: ${stock['stop_loss']:.2f}")

    print()
    print(f"Generated: {result['generated_at']}")
    print(f"Saved to: state/watchlist/next_day.json")

    return 0


if __name__ == '__main__':
    sys.exit(main())
