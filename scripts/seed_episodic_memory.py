#!/usr/bin/env python3
"""
Seed Episodic Memory with Historical Trade Examples
====================================================

Populates the cognitive brain's episodic memory with historical trade examples
from the signal_dataset.parquet file. This enables the knowledge boundary to
recognize familiar trading contexts and reduce uncertainty penalties.

Usage:
    python scripts/seed_episodic_memory.py [--max-episodes 500] [--sample-ratio 0.5]
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def create_episode_dict(
    trade: Dict[str, Any],
    idx: int
) -> Dict[str, Any]:
    """Create an episode dictionary from a historical trade row."""

    # Parse timestamp
    ts = trade['timestamp']
    if isinstance(ts, pd.Timestamp):
        ts_str = ts.isoformat()
    else:
        ts_str = str(ts)

    # Create deterministic episode ID
    episode_id = f"hist_{ts_str[:10]}_{trade['symbol']}_{idx:06d}"
    episode_id = episode_id.replace("-", "").replace(":", "")

    # Determine outcome
    won = trade['label'] == 1
    pnl = float(trade['pnl']) if pd.notna(trade['pnl']) else 0.0

    if won:
        outcome = "win"
    elif pnl < 0:
        outcome = "loss"
    else:
        outcome = "breakeven"

    # Build context signature (must match episodic_memory.py logic)
    # key_elements = [regime, strategy, side]
    regime = "unknown"  # Historical data doesn't have regime
    strategy = trade.get('strategy', 'IBS_RSI')
    side = trade.get('side', 'long')

    # Feature data from trade
    features = {
        'atr14': float(trade.get('atr14', 0) or 0),
        'sma20_over_200': float(trade.get('sma20_over_200', 1) or 1),
        'rv20': float(trade.get('rv20', 0) or 0),
        'don20_width': float(trade.get('don20_width', 0) or 0),
        'pos_in_don20': float(trade.get('pos_in_don20', 0.5) or 0.5),
        'ret5': float(trade.get('ret5', 0) or 0),
        'log_vol': float(trade.get('log_vol', 0) or 0),
    }

    episode = {
        'episode_id': episode_id,
        'started_at': ts_str,
        'completed_at': ts_str,  # Historical trades are already complete

        # Context
        'market_context': {
            'regime': regime,
            'regime_confidence': 0.7,  # Assumed
        },
        'signal_context': {
            'strategy': strategy,
            'side': side,
            'symbol': trade['symbol'],
            'entry_price': float(trade.get('entry_price', 0) or 0),
            **features,
        },
        'portfolio_context': {},

        # Reasoning (minimal for historical)
        'reasoning_trace': ["Historical trade seeded from backtest data"],
        'confidence_levels': {'initial': 0.6},
        'alternatives_considered': [],
        'concerns_noted': [],

        # Action
        'action_taken': {
            'type': 'buy' if side == 'long' else 'sell',
            'symbol': trade['symbol'],
            'entry_price': float(trade.get('entry_price', 0) or 0),
        },
        'decision_mode': 'historical',

        # Outcome
        'outcome': outcome,
        'pnl': pnl,
        'r_multiple': 0.0,  # Not available in historical data
        'outcome_details': {
            'pnl': pnl,
            'won': won,
            'exit_price': float(trade.get('exit_price', 0) or 0),
        },

        # Reflection
        'postmortem': '',
        'lessons_learned': [],
        'mistakes_made': [],
        'what_to_repeat': [],
        'what_to_avoid': [],

        # Metadata
        'tags': ['historical', 'seeded', strategy],
        'importance': 0.5,
        'is_simulated': False,
        'simulation_source': None,
        'simulation_params': {},
    }

    return episode


def main():
    parser = argparse.ArgumentParser(description='Seed episodic memory with historical trades')
    parser.add_argument('--max-episodes', type=int, default=500,
                        help='Maximum number of episodes to create (default: 500)')
    parser.add_argument('--sample-ratio', type=float, default=0.5,
                        help='Ratio to sample from dataset (default: 0.5)')
    parser.add_argument('--balance-outcomes', action='store_true', default=True,
                        help='Balance wins and losses (default: True)')
    args = parser.parse_args()

    # Paths
    data_path = Path('data/ml/signal_dataset.parquet')
    episodes_dir = Path('state/cognitive/episodes')
    episodes_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading historical trades from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"  Loaded {len(df):,} trades")

    # Balance outcomes if requested
    if args.balance_outcomes:
        wins = df[df['label'] == 1]
        losses = df[df['label'] == 0]

        # Take equal samples from each
        n_each = min(len(wins), len(losses), args.max_episodes // 2)

        # Sample evenly across time (not random)
        wins_sample = wins.iloc[::max(1, len(wins) // n_each)][:n_each]
        losses_sample = losses.iloc[::max(1, len(losses) // n_each)][:n_each]

        df_sample = pd.concat([wins_sample, losses_sample]).sort_values('timestamp')
        print(f"  Balanced sample: {len(df_sample):,} trades ({n_each} wins, {n_each} losses)")
    else:
        # Just take a sample
        step = max(1, len(df) // args.max_episodes)
        df_sample = df.iloc[::step][:args.max_episodes]
        print(f"  Sampled {len(df_sample):,} trades")

    # Count existing episodes
    existing = list(episodes_dir.glob("*.json"))
    print(f"  Existing episodes in memory: {len(existing)}")

    # Create episodes
    created = 0
    skipped = 0

    for idx, (_, row) in enumerate(df_sample.iterrows()):
        episode = create_episode_dict(row.to_dict(), idx)

        # Save to file
        ep_file = episodes_dir / f"{episode['episode_id']}.json"

        if ep_file.exists():
            skipped += 1
            continue

        with open(ep_file, 'w') as f:
            json.dump(episode, f, indent=2, default=str)
        created += 1

    print("\nResults:")
    print(f"  Created: {created} episodes")
    print(f"  Skipped: {skipped} (already exist)")
    print(f"  Total in memory: {len(list(episodes_dir.glob('*.json')))}")

    # Show context distribution
    print("\nContext signatures seeded:")
    contexts = {}
    for ep_file in episodes_dir.glob("*.json"):
        with open(ep_file) as f:
            ep = json.load(f)

        regime = ep.get('market_context', {}).get('regime', 'unknown')
        strategy = ep.get('signal_context', {}).get('strategy', 'unknown')
        side = ep.get('signal_context', {}).get('side', 'unknown')
        key = f"{regime}|{strategy}|{side}"
        contexts[key] = contexts.get(key, 0) + 1

    for ctx, count in sorted(contexts.items()):
        print(f"  {ctx}: {count} episodes")

    print(f"\nDone! Episodic memory now has {len(list(episodes_dir.glob('*.json')))} episodes.")


if __name__ == '__main__':
    main()
