#!/usr/bin/env python3
"""
Cognitive Learning Script
=========================

Processes closed trades and feeds outcomes back to the cognitive system
for learning and self-improvement.

This script:
1. Fetches closed orders from Alpaca
2. Matches them with cognitive episodes
3. Calculates P&L and R-multiples
4. Records outcomes for cognitive learning
5. Optionally runs daily/weekly consolidation

Usage:
    python scripts/cognitive_learn.py --days 1
    python scripts/cognitive_learn.py --consolidate daily
    python scripts/cognitive_learn.py --introspect
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from core.structured_log import jlog


def fetch_closed_orders(days: int = 1) -> List[Dict[str, Any]]:
    """Fetch closed orders from Alpaca."""
    import requests

    base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
    key = os.getenv('ALPACA_API_KEY_ID', '')
    sec = os.getenv('ALPACA_API_SECRET_KEY', '')

    if not key or not sec:
        print("Error: Alpaca credentials not set")
        return []

    headers = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': sec}

    # Fetch orders from the last N days
    after = (datetime.utcnow() - timedelta(days=days)).isoformat() + 'Z'

    try:
        r = requests.get(
            f"{base}/v2/orders",
            headers=headers,
            params={'status': 'closed', 'after': after, 'limit': 500},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching orders: {e}")
        return []


def fetch_account_activities(days: int = 1) -> List[Dict[str, Any]]:
    """Fetch account activities (fills, dividends, etc.) from Alpaca."""
    import requests

    base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
    key = os.getenv('ALPACA_API_KEY_ID', '')
    sec = os.getenv('ALPACA_API_SECRET_KEY', '')

    if not key or not sec:
        return []

    headers = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': sec}

    after = (datetime.utcnow() - timedelta(days=days)).isoformat() + 'Z'

    try:
        r = requests.get(
            f"{base}/v2/account/activities",
            headers=headers,
            params={'activity_types': 'FILL', 'after': after},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching activities: {e}")
        return []


def load_hash_chain_entries(days: int = 1) -> List[Dict[str, Any]]:
    """Load entries from hash chain with cognitive data."""
    chain_file = ROOT / 'state' / 'hash_chain.jsonl'
    entries = []

    if not chain_file.exists():
        return entries

    cutoff = datetime.utcnow() - timedelta(days=days)

    with open(chain_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                ts = entry.get('ts') or entry.get('timestamp')
                if ts:
                    entry_time = datetime.fromisoformat(ts.replace('Z', ''))
                    if entry_time >= cutoff:
                        entries.append(entry)
            except:
                continue

    return entries


def match_orders_to_episodes(
    orders: List[Dict],
    chain_entries: List[Dict],
) -> List[Dict[str, Any]]:
    """Match closed orders with cognitive episodes from hash chain."""
    matched = []

    # Build lookup from chain entries
    episode_lookup = {}
    for entry in chain_entries:
        if 'cognitive_episode_id' in entry:
            sym = entry.get('symbol', '')
            episode_lookup[sym] = entry

    for order in orders:
        sym = order.get('symbol', '')
        side = order.get('side', '')
        filled_qty = float(order.get('filled_qty', 0))
        filled_avg_price = float(order.get('filled_avg_price', 0))

        if filled_qty == 0:
            continue

        match = {
            'symbol': sym,
            'side': side,
            'filled_qty': filled_qty,
            'filled_price': filled_avg_price,
            'order_id': order.get('id'),
            'created_at': order.get('created_at'),
            'filled_at': order.get('filled_at'),
        }

        # Look for matching episode
        if sym in episode_lookup:
            ep_data = episode_lookup[sym]
            match['episode_id'] = ep_data.get('cognitive_episode_id')
            match['entry_confidence'] = ep_data.get('cognitive_confidence')
            match['entry_price'] = ep_data.get('limit_price')
            match['decision_id'] = ep_data.get('decision_id')

        matched.append(match)

    return matched


def calculate_trade_outcome(
    entry_price: float,
    exit_price: float,
    stop_loss: Optional[float],
    side: str,
) -> Dict[str, Any]:
    """Calculate trade outcome metrics."""
    if side.upper() == 'BUY':
        pnl = exit_price - entry_price
        won = pnl > 0
    else:
        pnl = entry_price - exit_price
        won = pnl > 0

    # Calculate R-multiple if stop loss is known
    r_multiple = None
    if stop_loss and stop_loss != entry_price:
        risk = abs(entry_price - stop_loss)
        if risk > 0:
            r_multiple = pnl / risk

    return {
        'won': won,
        'pnl': pnl,
        'r_multiple': r_multiple,
        'pnl_pct': (pnl / entry_price) * 100 if entry_price > 0 else 0,
    }


def process_outcomes(matched_trades: List[Dict], dry_run: bool = False) -> Dict[str, Any]:
    """Process trade outcomes and record in cognitive system."""
    from cognitive.signal_processor import get_signal_processor
    from cognitive.cognitive_brain import get_cognitive_brain

    processor = get_signal_processor()
    brain = get_cognitive_brain()

    results = {
        'processed': 0,
        'learned': 0,
        'errors': 0,
        'trades': [],
    }

    for trade in matched_trades:
        episode_id = trade.get('episode_id')
        if not episode_id:
            continue

        entry_price = trade.get('entry_price', trade.get('filled_price', 0))
        exit_price = trade.get('filled_price', 0)
        side = trade.get('side', 'BUY')

        outcome = calculate_trade_outcome(entry_price, exit_price, None, side)

        results['processed'] += 1

        trade_summary = {
            'symbol': trade.get('symbol'),
            'episode_id': episode_id,
            'entry_confidence': trade.get('entry_confidence'),
            **outcome,
        }
        results['trades'].append(trade_summary)

        if dry_run:
            print(f"  [DRY RUN] Would record: {trade.get('symbol')} "
                  f"won={outcome['won']} pnl={outcome['pnl']:.2f}")
            continue

        try:
            brain.learn_from_outcome(
                episode_id=episode_id,
                outcome={
                    'won': outcome['won'],
                    'pnl': outcome['pnl'],
                    'r_multiple': outcome['r_multiple'],
                },
            )
            results['learned'] += 1
            jlog('cognitive_learn', symbol=trade.get('symbol'), **outcome)
            print(f"  Learned: {trade.get('symbol')} won={outcome['won']} "
                  f"pnl={outcome['pnl']:.2f} R={outcome['r_multiple']}")
        except Exception as e:
            results['errors'] += 1
            print(f"  Error learning {trade.get('symbol')}: {e}")

    return results


def run_consolidation(scope: str) -> Dict[str, Any]:
    """Run cognitive consolidation."""
    from cognitive.cognitive_brain import get_cognitive_brain

    brain = get_cognitive_brain()

    if scope == 'daily':
        result = brain.daily_consolidation()
        print("Daily consolidation complete:")
    elif scope == 'weekly':
        result = brain.weekly_consolidation()
        print("Weekly consolidation complete:")
    else:
        print(f"Unknown scope: {scope}")
        return {}

    for key, value in result.items():
        print(f"  {key}: {value}")

    return result


def show_introspection():
    """Show cognitive system introspection."""
    from cognitive.cognitive_brain import get_cognitive_brain
    from cognitive.signal_processor import get_signal_processor

    brain = get_cognitive_brain()
    processor = get_signal_processor()

    print("=" * 60)
    print("COGNITIVE SYSTEM INTROSPECTION")
    print("=" * 60)
    print()

    # Brain status
    status = brain.get_status()
    print("--- Brain Status ---")
    print(f"Initialized: {status.get('initialized')}")
    print(f"Decision count: {status.get('decision_count')}")
    print(f"Min confidence: {status.get('min_confidence_to_act')}")
    print()

    # Component status
    print("--- Components ---")
    for comp_name, comp_status in status.get('components', {}).items():
        if isinstance(comp_status, dict):
            count = comp_status.get('total_episodes') or comp_status.get('total_rules') or 0
            print(f"  {comp_name}: {count} entries")
        else:
            print(f"  {comp_name}: {comp_status}")
    print()

    # Processor status
    proc_status = processor.get_cognitive_status()
    print("--- Signal Processor ---")
    print(f"Active: {proc_status.get('processor_active')}")
    print(f"Min confidence: {proc_status.get('min_confidence')}")
    print(f"Active episodes: {proc_status.get('active_episodes')}")
    print()

    # Full introspection
    print("--- Full Introspection ---")
    print(brain.introspect()[:2000])  # Limit output


def main():
    ap = argparse.ArgumentParser(description='Cognitive Learning - Process trade outcomes')
    ap.add_argument('--days', type=int, default=1, help='Days of history to process')
    ap.add_argument('--dotenv', type=str, default='./.env', help='Path to .env file')
    ap.add_argument('--dry-run', action='store_true', help='Show what would be learned')
    ap.add_argument('--consolidate', type=str, choices=['daily', 'weekly'],
                    help='Run consolidation (daily or weekly)')
    ap.add_argument('--introspect', action='store_true', help='Show cognitive introspection')
    ap.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # Show introspection if requested
    if args.introspect:
        show_introspection()
        return 0

    # Run consolidation if requested
    if args.consolidate:
        run_consolidation(args.consolidate)
        return 0

    # Process trade outcomes
    print(f"Fetching closed orders from last {args.days} day(s)...")
    orders = fetch_closed_orders(args.days)
    print(f"  Found {len(orders)} closed orders")

    if not orders:
        print("No closed orders to process.")
        return 0

    print(f"\nLoading hash chain entries...")
    chain_entries = load_hash_chain_entries(args.days)
    print(f"  Found {len(chain_entries)} chain entries")

    print(f"\nMatching orders to cognitive episodes...")
    matched = match_orders_to_episodes(orders, chain_entries)
    episodes_with_data = [m for m in matched if m.get('episode_id')]
    print(f"  Matched {len(episodes_with_data)} trades with cognitive episodes")

    if not episodes_with_data:
        print("No trades with cognitive episodes found.")
        return 0

    print(f"\nProcessing outcomes...")
    results = process_outcomes(episodes_with_data, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {results['processed']}")
    print(f"Learned:   {results['learned']}")
    print(f"Errors:    {results['errors']}")

    if args.verbose and results['trades']:
        print(f"\nTrades:")
        for t in results['trades']:
            status = "WIN" if t['won'] else "LOSS"
            print(f"  {t['symbol']}: {status} pnl={t['pnl']:.2f} R={t.get('r_multiple', 'N/A')}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
