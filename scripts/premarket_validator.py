#!/usr/bin/env python3
"""
Premarket Validator - Validates overnight watchlist based on morning conditions.

Runs at 8:00 AM to:
1. Load overnight watchlist
2. Check each stock for gap > 3%, news, corporate actions
3. Flag stocks as VALID, GAP_INVALIDATED, NEWS_RISK, etc.
4. Generate validated watchlist for today's trading

Usage:
    python scripts/premarket_validator.py
    python scripts/premarket_validator.py --gap-threshold 0.03
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from enum import Enum

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


class ValidationStatus(Enum):
    """Status of watchlist stock after validation."""
    VALID = "VALID"                     # Setup still valid, cleared for trading
    GAP_INVALIDATED = "GAP_INVALIDATED"  # Gap > threshold, setup broken
    NEWS_RISK = "NEWS_RISK"             # News event, proceed with caution
    IMPROVED = "IMPROVED"               # Setup actually improved overnight
    DEGRADED = "DEGRADED"               # Setup weakened but still tradeable
    REMOVED = "REMOVED"                 # Removed from watchlist entirely


def validate_premarket_watchlist(
    dotenv_path: str = "./.env",
    gap_threshold: float = 0.03,
    check_news: bool = True,
) -> Dict:
    """
    Validate the overnight watchlist against premarket conditions.

    Args:
        dotenv_path: Path to .env file
        gap_threshold: Gap percentage that invalidates setup (default 3%)
        check_news: Whether to check for news events

    Returns:
        Dict with validated watchlist
    """
    load_dotenv(dotenv_path)

    ET = pytz.timezone('America/New_York')
    now = datetime.now(ET)
    today = now.strftime('%Y-%m-%d')

    logger.info(f"Validating premarket watchlist for {today}")

    # Load overnight watchlist
    watchlist_path = ROOT / 'state' / 'watchlist' / 'next_day.json'
    if not watchlist_path.exists():
        logger.warning("No overnight watchlist found")
        return {
            'validated_at': now.isoformat(),
            'for_date': today,
            'status': 'NO_WATCHLIST',
            'watchlist': [],
            'totd': None,
        }

    with open(watchlist_path) as f:
        overnight = json.load(f)

    watchlist = overnight.get('watchlist', [])
    if not watchlist:
        logger.warning("Overnight watchlist is empty")
        return {
            'validated_at': now.isoformat(),
            'for_date': today,
            'status': 'EMPTY_WATCHLIST',
            'watchlist': [],
            'totd': None,
        }

    logger.info(f"Loaded {len(watchlist)} stocks from overnight watchlist")

    # Get current prices and validate each stock
    from execution.broker_alpaca import get_best_bid, get_best_ask

    validated = []
    removed = []

    for stock in watchlist:
        symbol = stock['symbol']
        overnight_entry = stock.get('entry_price', 0)

        try:
            # Get current premarket price
            bid = get_best_bid(symbol)
            ask = get_best_ask(symbol)

            if bid and ask:
                current_price = (bid + ask) / 2
            elif bid:
                current_price = bid
            elif ask:
                current_price = ask
            else:
                # No premarket quote, keep as valid
                stock['validation_status'] = ValidationStatus.VALID.value
                stock['validation_note'] = "No premarket quote, using overnight analysis"
                stock['current_price'] = overnight_entry
                validated.append(stock)
                continue

            # Calculate gap percentage
            if overnight_entry > 0:
                gap_pct = (current_price - overnight_entry) / overnight_entry
            else:
                gap_pct = 0

            stock['current_price'] = current_price
            stock['gap_pct'] = gap_pct

            # Check for gap invalidation
            if abs(gap_pct) > gap_threshold:
                stock['validation_status'] = ValidationStatus.GAP_INVALIDATED.value
                stock['validation_note'] = f"Gap {gap_pct:.1%} exceeds {gap_threshold:.0%} threshold"
                removed.append(stock)
                logger.info(f"{symbol}: GAP_INVALIDATED ({gap_pct:+.1%})")
                continue

            # Check for news (if enabled)
            if check_news:
                has_news, news_note = check_stock_news(symbol, dotenv_path)
                if has_news:
                    stock['validation_status'] = ValidationStatus.NEWS_RISK.value
                    stock['validation_note'] = news_note
                    stock['news_flag'] = True
                    # Still include but flagged
                    validated.append(stock)
                    logger.info(f"{symbol}: NEWS_RISK - {news_note}")
                    continue

            # Check if setup improved or degraded
            if gap_pct > 0.01:
                # Gapped up slightly - setup may be weaker for longs
                stock['validation_status'] = ValidationStatus.DEGRADED.value
                stock['validation_note'] = f"Small gap up {gap_pct:+.1%}, entry less favorable"
            elif gap_pct < -0.01:
                # Gapped down - better entry for longs
                stock['validation_status'] = ValidationStatus.IMPROVED.value
                stock['validation_note'] = f"Gap down {gap_pct:+.1%}, better entry opportunity"
            else:
                # Minimal gap - setup unchanged
                stock['validation_status'] = ValidationStatus.VALID.value
                stock['validation_note'] = "Setup unchanged from overnight analysis"

            validated.append(stock)
            logger.info(f"{symbol}: {stock['validation_status']} ({gap_pct:+.1%})")

        except Exception as e:
            logger.warning(f"Error validating {symbol}: {e}")
            stock['validation_status'] = ValidationStatus.VALID.value
            stock['validation_note'] = f"Validation error, using overnight analysis: {e}"
            validated.append(stock)

    # Sort by rank and validation status
    # IMPROVED > VALID > DEGRADED > NEWS_RISK
    status_priority = {
        ValidationStatus.IMPROVED.value: 0,
        ValidationStatus.VALID.value: 1,
        ValidationStatus.DEGRADED.value: 2,
        ValidationStatus.NEWS_RISK.value: 3,
    }

    validated.sort(key=lambda x: (
        status_priority.get(x.get('validation_status', 'VALID'), 99),
        x.get('rank', 99)
    ))

    # Determine TOTD from validated list
    totd = validated[0] if validated else None

    # Build result
    result = {
        'validated_at': now.isoformat(),
        'for_date': today,
        'overnight_generated': overnight.get('generated_at'),
        'status': 'VALIDATED',
        'summary': {
            'original_count': len(watchlist),
            'validated_count': len(validated),
            'removed_count': len(removed),
            'gap_threshold': gap_threshold,
        },
        'totd': {
            'symbol': totd['symbol'],
            'score': totd.get('score', totd.get('overnight_score', 0)),
            'validation_status': totd.get('validation_status'),
            'entry_price': totd.get('entry_price', 0),
            'current_price': totd.get('current_price', 0),
            'stop_loss': totd.get('stop_loss', 0),
        } if totd else None,
        'watchlist': validated,
        'removed': removed,
    }

    # Save validated watchlist
    output_path = ROOT / 'state' / 'watchlist' / 'today_validated.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"Saved validated watchlist to {output_path}")
    logger.info(f"Result: {len(validated)} valid, {len(removed)} removed")

    return result


def check_stock_news(symbol: str, dotenv_path: str) -> Tuple[bool, str]:
    """
    Check if stock has significant news that could affect the trade.

    Returns:
        Tuple of (has_significant_news, note)
    """
    try:
        from cognitive.news_processor import NewsProcessor

        processor = NewsProcessor()
        articles = processor.fetch_news([symbol], hours=12)

        if not articles:
            return (False, "")

        # Check for significant news
        significant_keywords = [
            'earnings', 'guidance', 'fda', 'sec', 'lawsuit',
            'acquisition', 'merger', 'bankruptcy', 'downgrade',
            'upgrade', 'target', 'analyst'
        ]

        for article in articles[:5]:
            headline = article.get('headline', '').lower()
            for keyword in significant_keywords:
                if keyword in headline:
                    return (True, f"News: {article.get('headline', '')[:50]}...")

        return (False, "")

    except Exception as e:
        logger.debug(f"News check failed for {symbol}: {e}")
        return (False, "")


def main():
    parser = argparse.ArgumentParser(description='Validate overnight watchlist for today')
    parser.add_argument('--dotenv', type=str, default='./.env')
    parser.add_argument('--gap-threshold', type=float, default=0.03, help='Gap threshold (default 3%%)')
    parser.add_argument('--no-news', action='store_true', help='Skip news check')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    result = validate_premarket_watchlist(
        dotenv_path=args.dotenv,
        gap_threshold=args.gap_threshold,
        check_news=not args.no_news,
    )

    # Print summary
    print()
    print("=" * 60)
    print(f"PREMARKET VALIDATION - {result['for_date']}")
    print("=" * 60)
    print()

    summary = result.get('summary', {})
    print(f"Original: {summary.get('original_count', 0)} stocks")
    print(f"Validated: {summary.get('validated_count', 0)} stocks")
    print(f"Removed: {summary.get('removed_count', 0)} stocks")
    print()

    if result['totd']:
        totd = result['totd']
        print(f"TRADE OF THE DAY: {totd['symbol']}")
        print(f"  Status: {totd['validation_status']}")
        print(f"  Entry: ${totd['entry_price']:.2f} | Current: ${totd['current_price']:.2f}")
        print()

    print("VALIDATED WATCHLIST:")
    for stock in result.get('watchlist', []):
        status = stock.get('validation_status', 'UNKNOWN')
        gap = stock.get('gap_pct', 0)
        marker = "+" if status == 'IMPROVED' else "-" if status in ('DEGRADED', 'NEWS_RISK') else " "
        print(f"  {marker} {stock['symbol']}: {status} (gap: {gap:+.1%})")

    if result.get('removed'):
        print()
        print("REMOVED (Gap Invalidated):")
        for stock in result['removed']:
            gap = stock.get('gap_pct', 0)
            print(f"  X {stock['symbol']}: {gap:+.1%} gap")

    print()
    print("Saved to: state/watchlist/today_validated.json")

    return 0


if __name__ == '__main__':
    sys.exit(main())
