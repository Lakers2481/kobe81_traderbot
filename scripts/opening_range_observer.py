#!/usr/bin/env python3
"""
Opening Range Observer - Observes and logs watchlist behavior during opening range.

Runs at 9:30 and 9:45 AM to:
1. Log opening prices for watchlist stocks
2. Track which stocks show strength/weakness
3. Note any breaking out of opening range
4. DO NOT execute any trades - observe only

This data feeds into the 10:00 AM primary scan for better entry decisions.

Usage:
    python scripts/opening_range_observer.py
    python scripts/opening_range_observer.py --snapshot  # Just take a snapshot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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


def observe_opening_range(dotenv_path: str = "./.env") -> Dict:
    """
    Observe watchlist stocks during opening range.

    This function:
    - Loads validated watchlist
    - Gets current prices
    - Logs strength/weakness
    - Saves observations for later analysis

    NO TRADES ARE EXECUTED.

    Returns:
        Dict with observations
    """
    load_dotenv(dotenv_path)

    ET = pytz.timezone('America/New_York')
    now = datetime.now(ET)
    timestamp = now.strftime('%H:%M:%S')

    logger.info(f"Opening Range Observer - {timestamp} ET")
    logger.info("=" * 50)
    logger.info("OBSERVE ONLY - NO TRADES DURING OPENING RANGE")
    logger.info("=" * 50)

    # Load validated watchlist
    watchlist_path = ROOT / 'state' / 'watchlist' / 'today_validated.json'
    if not watchlist_path.exists():
        logger.warning("No validated watchlist found")
        return {'status': 'NO_WATCHLIST', 'observations': []}

    with open(watchlist_path) as f:
        validated = json.load(f)

    watchlist = validated.get('watchlist', [])
    if not watchlist:
        logger.warning("Validated watchlist is empty")
        return {'status': 'EMPTY_WATCHLIST', 'observations': []}

    logger.info(f"Observing {len(watchlist)} watchlist stocks")

    # Get current prices and calculate opening range behavior
    from execution.broker_alpaca import get_best_bid, get_best_ask

    observations = []

    for stock in watchlist:
        symbol = stock['symbol']
        premarket_price = stock.get('current_price', stock.get('entry_price', 0))

        try:
            bid = get_best_bid(symbol)
            ask = get_best_ask(symbol)

            if bid and ask:
                current_price = (bid + ask) / 2
                spread = (ask - bid) / current_price if current_price > 0 else 0
            elif bid:
                current_price = bid
                spread = 0
            elif ask:
                current_price = ask
                spread = 0
            else:
                current_price = premarket_price
                spread = 0

            # Calculate change since premarket validation
            if premarket_price > 0:
                change_pct = (current_price - premarket_price) / premarket_price
            else:
                change_pct = 0

            # Determine strength/weakness
            if change_pct > 0.01:
                strength = "STRONG"
                signal = "Showing strength, potential breakout"
            elif change_pct < -0.01:
                strength = "WEAK"
                signal = "Showing weakness, may need to wait"
            else:
                strength = "NEUTRAL"
                signal = "Holding near premarket levels"

            obs = {
                'symbol': symbol,
                'timestamp': timestamp,
                'premarket_price': premarket_price,
                'current_price': current_price,
                'change_pct': change_pct,
                'spread': spread,
                'strength': strength,
                'signal': signal,
                'original_entry': stock.get('entry_price', 0),
                'stop_loss': stock.get('stop_loss', 0),
            }
            observations.append(obs)

            # Log observation
            icon = "🟢" if strength == "STRONG" else "🔴" if strength == "WEAK" else "⚪"
            # Use ASCII for Windows compatibility
            icon_ascii = "[+]" if strength == "STRONG" else "[-]" if strength == "WEAK" else "[ ]"
            logger.info(f"  {icon_ascii} {symbol}: ${current_price:.2f} ({change_pct:+.1%}) - {strength}")

        except Exception as e:
            logger.warning(f"  Error observing {symbol}: {e}")
            observations.append({
                'symbol': symbol,
                'timestamp': timestamp,
                'error': str(e),
                'strength': 'UNKNOWN',
            })

    # Load existing opening range data or create new
    or_path = ROOT / 'state' / 'watchlist' / 'opening_range.json'
    if or_path.exists():
        with open(or_path) as f:
            or_data = json.load(f)
    else:
        or_data = {
            'date': now.strftime('%Y-%m-%d'),
            'snapshots': [],
        }

    # Add this snapshot
    snapshot = {
        'timestamp': now.isoformat(),
        'time': timestamp,
        'observations': observations,
    }
    or_data['snapshots'].append(snapshot)

    # Analyze overall market sentiment from watchlist
    strong = sum(1 for o in observations if o.get('strength') == 'STRONG')
    weak = sum(1 for o in observations if o.get('strength') == 'WEAK')
    neutral = sum(1 for o in observations if o.get('strength') == 'NEUTRAL')

    or_data['latest_summary'] = {
        'timestamp': timestamp,
        'strong_count': strong,
        'weak_count': weak,
        'neutral_count': neutral,
        'bias': 'BULLISH' if strong > weak else 'BEARISH' if weak > strong else 'NEUTRAL',
    }

    # Save observations
    with open(or_path, 'w') as f:
        json.dump(or_data, f, indent=2, default=str)

    logger.info(f"")
    logger.info(f"Opening Range Summary: {strong} strong, {neutral} neutral, {weak} weak")
    logger.info(f"Market Bias: {or_data['latest_summary']['bias']}")
    logger.info(f"")
    logger.info(f"Saved to: {or_path}")
    logger.info(f"")
    logger.info("⏳ Primary execution window opens at 10:00 AM")

    return {
        'status': 'OBSERVED',
        'timestamp': now.isoformat(),
        'observations': observations,
        'summary': or_data['latest_summary'],
    }


def get_opening_range_insights() -> Dict:
    """
    Get insights from opening range observations.

    Called at 10:00 AM to inform primary scan.
    """
    or_path = ROOT / 'state' / 'watchlist' / 'opening_range.json'
    if not or_path.exists():
        return {'status': 'NO_DATA'}

    with open(or_path) as f:
        or_data = json.load(f)

    # Get latest observations for each symbol
    latest = {}
    for snapshot in or_data.get('snapshots', []):
        for obs in snapshot.get('observations', []):
            symbol = obs.get('symbol')
            if symbol:
                latest[symbol] = obs

    # Rank by strength
    ranked = sorted(
        latest.values(),
        key=lambda x: (
            0 if x.get('strength') == 'STRONG' else
            1 if x.get('strength') == 'NEUTRAL' else 2
        )
    )

    return {
        'status': 'OK',
        'summary': or_data.get('latest_summary', {}),
        'ranked_symbols': [o.get('symbol') for o in ranked],
        'observations': latest,
    }


def main():
    parser = argparse.ArgumentParser(description='Observe opening range (NO TRADES)')
    parser.add_argument('--dotenv', type=str, default='./.env')
    parser.add_argument('--snapshot', action='store_true', help='Just take a snapshot')
    parser.add_argument('--insights', action='store_true', help='Get insights from observations')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.insights:
        insights = get_opening_range_insights()
        print(json.dumps(insights, indent=2, default=str))
        return 0

    result = observe_opening_range(dotenv_path=args.dotenv)

    print()
    print("=" * 60)
    print("OPENING RANGE OBSERVATION COMPLETE")
    print("=" * 60)
    print()
    print("Remember: NO TRADES until 10:00 AM")
    print("This observation informs the primary scan.")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
