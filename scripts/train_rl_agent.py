#!/usr/bin/env python3
"""
Train RL Trading Agent with Sortino Reward
===========================================

Trains a PPO agent using Sortino ratio as the reward function.
Sortino > Sharpe because it only penalizes downside volatility.

Usage:
    python scripts/train_rl_agent.py --timesteps 100000
    python scripts/train_rl_agent.py --timesteps 50000 --algorithm A2C
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config.env_loader import load_env
from core.structured_log import jlog
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon


def load_training_data(
    universe_path: str,
    start_date: str,
    end_date: str,
    cap: int = 50,
) -> dict:
    """Load OHLCV data for training."""
    print(f"Loading universe from {universe_path}...")
    symbols = load_universe(universe_path, cap=cap)
    print(f"Loaded {len(symbols)} symbols")

    price_data = {}

    print(f"Fetching data from {start_date} to {end_date}...")
    for i, sym in enumerate(symbols):
        try:
            df = fetch_daily_bars_polygon(sym, start_date, end_date)
            if df is not None and len(df) >= 252:  # At least 1 year
                price_data[sym] = df
                if (i + 1) % 10 == 0:
                    print(f"  Loaded {i+1}/{len(symbols)} symbols...")
        except Exception:
            pass  # Skip failed symbols

    print(f"Loaded data for {len(price_data)} symbols")
    return price_data


def train_agent(
    price_data: dict,
    algorithm: str = 'PPO',
    timesteps: int = 100000,
    learning_rate: float = 0.0003,
) -> dict:
    """Train RL agent."""
    from ml.alpha_discovery.rl_agent.agent import RLTradingAgent, RLAgentConfig
    from ml.alpha_discovery.rl_agent.trading_env import TradingEnv

    # Combine all data for training
    all_dfs = list(price_data.values())
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined training data: {len(combined)} bars")

    # Create environment with Sortino reward
    env = TradingEnv(
        price_data=combined,
        reward_type='sortino',  # Key: Sortino for better risk adjustment
        initial_capital=100000.0,
        max_position_pct=0.10,
        transaction_cost_bps=5.0,
    )

    # Configure agent
    config = RLAgentConfig(
        algorithm=algorithm,
        learning_rate=learning_rate,
        total_timesteps=timesteps,
        gamma=0.99,
        batch_size=64,
    )

    print(f"\nTraining {algorithm} agent for {timesteps} timesteps...")
    print("  Reward type: SORTINO (downside volatility only)")
    print(f"  Learning rate: {learning_rate}")

    # Train
    agent = RLTradingAgent(config=config, env=env)

    checkpoint_dir = str(ROOT / 'models' / 'rl_agent')
    result = agent.train(checkpoint_dir=checkpoint_dir)

    return result


def main():
    parser = argparse.ArgumentParser(description='Train RL Trading Agent')
    parser.add_argument('--universe', type=str,
                       default='data/universe/optionable_liquid_800.csv',
                       help='Universe file path')
    parser.add_argument('--start', type=str, default='2020-01-01',
                       help='Training start date')
    parser.add_argument('--end', type=str, default='2024-12-31',
                       help='Training end date')
    parser.add_argument('--cap', type=int, default=50,
                       help='Number of symbols to use')
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'A2C', 'DQN'],
                       help='RL algorithm')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--lr', type=float, default=0.0003,
                       help='Learning rate')
    parser.add_argument('--dotenv', type=str, default='./.env',
                       help='Path to .env file')
    args = parser.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    print("=" * 60)
    print("RL TRADING AGENT TRAINING")
    print("=" * 60)
    print(f"Algorithm:  {args.algorithm}")
    print(f"Timesteps:  {args.timesteps}")
    print("Reward:     SORTINO (downside volatility)")
    print(f"Universe:   {args.cap} symbols")
    print(f"Period:     {args.start} to {args.end}")
    print("=" * 60)

    # Load data
    price_data = load_training_data(
        args.universe,
        args.start,
        args.end,
        args.cap,
    )

    if not price_data:
        print("ERROR: No price data loaded!")
        sys.exit(1)

    # Train
    result = train_agent(
        price_data,
        algorithm=args.algorithm,
        timesteps=args.timesteps,
        learning_rate=args.lr,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Algorithm:       {result.get('algorithm', 'N/A')}")
    print(f"Total timesteps: {result.get('total_timesteps', 'N/A')}")

    metrics = result.get('metrics', {})
    if isinstance(metrics, dict) and 'return_pct' in metrics:
        print("\nEvaluation Results:")
        print(f"  Return:    {metrics.get('return_pct', 0):.2f}%")
        print(f"  Win Rate:  {metrics.get('win_rate', 0)*100:.1f}%")
        print(f"  Trades:    {metrics.get('n_trades', 0)}")

    print(f"\nModel saved to: {result.get('model_path', 'N/A')}")

    jlog("rl_agent_trained",
         algorithm=result.get('algorithm'),
         timesteps=result.get('total_timesteps'),
         metrics=metrics)


if __name__ == '__main__':
    main()
