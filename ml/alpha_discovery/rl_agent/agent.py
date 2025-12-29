"""
Reinforcement Learning trading agent.
"""
from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

from .trading_env import TradingEnv

logger = logging.getLogger(__name__)

# Try to import stable-baselines3
try:
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


@dataclass
class RLAgentConfig:
    """Configuration for RL agent training."""
    algorithm: str = 'PPO'
    learning_rate: float = 0.0003
    gamma: float = 0.99
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.01
    total_timesteps: int = 100000
    eval_freq: int = 10000
    save_freq: int = 50000


class RLTradingAgent:
    """
    Reinforcement Learning agent for trading optimization.

    Supports PPO, DQN, and A2C algorithms via stable-baselines3.
    Falls back to a simple policy if SB3 not available.
    """

    def __init__(
        self,
        config: Optional[RLAgentConfig] = None,
        env: Optional[TradingEnv] = None,
    ):
        """
        Initialize RL agent.

        Args:
            config: Agent configuration
            env: Trading environment (optional, can be set later)
        """
        self.config = config or RLAgentConfig()
        self.env = env
        self.model = None
        self._trained = False
        self._training_history: List[Dict] = []

    def train(
        self,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
        validation_data: Optional[Dict[str, pd.DataFrame]] = None,
        checkpoint_dir: str = 'models/rl_checkpoints',
    ) -> Dict[str, Any]:
        """
        Train the RL agent.

        Args:
            price_data: Dict of symbol -> OHLCV DataFrame
            validation_data: Optional validation data
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training metrics dict
        """
        if not SB3_AVAILABLE:
            logger.warning("stable-baselines3 not available, using simple policy")
            return self._train_simple(price_data, checkpoint_dir)

        if self.env is None and price_data:
            # Create environment from first symbol's data
            first_symbol = list(price_data.keys())[0]
            self.env = TradingEnv(
                price_data=price_data[first_symbol],
                reward_type='sharpe',
            )

        if self.env is None:
            raise ValueError("No environment or price data provided")

        # Create vectorized environment
        def make_env():
            return self.env

        vec_env = DummyVecEnv([make_env])

        # Initialize model
        algorithm_cls = {
            'PPO': PPO,
            'DQN': DQN,
            'A2C': A2C,
        }.get(self.config.algorithm, PPO)

        self.model = algorithm_cls(
            'MlpPolicy',
            vec_env,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            verbose=1,
        )

        # Create checkpoint directory
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Train
        logger.info(f"Training {self.config.algorithm} for {self.config.total_timesteps} steps")
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            progress_bar=True,
        )

        self._trained = True

        # Evaluate
        metrics = self.evaluate(self.env.price_data if hasattr(self.env, 'price_data') else None)

        # Save final model
        final_path = Path(checkpoint_dir) / 'final_model'
        self.save(str(final_path))

        return {
            'algorithm': self.config.algorithm,
            'total_timesteps': self.config.total_timesteps,
            'metrics': metrics,
            'model_path': str(final_path),
        }

    def _train_simple(
        self,
        price_data: Optional[Dict[str, pd.DataFrame]],
        checkpoint_dir: str,
    ) -> Dict[str, Any]:
        """Simple training without SB3."""
        logger.info("Using simple rule-based policy (SB3 not available)")

        # Create simple policy weights
        self._simple_policy = {
            'rsi_threshold': 30,
            'ibs_threshold': 0.2,
            'trend_required': True,
        }

        self._trained = True

        return {
            'algorithm': 'simple_rules',
            'total_timesteps': 0,
            'metrics': {'type': 'rule_based'},
            'model_path': checkpoint_dir,
        }

    def predict(
        self,
        observation: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Predict action and confidence for current state.

        Args:
            observation: State observation

        Returns:
            Tuple of (action, confidence)
        """
        if not self._trained:
            logger.warning("Model not trained, returning HOLD action")
            return 0, 0.0

        if self.model is not None and SB3_AVAILABLE:
            action, _ = self.model.predict(observation, deterministic=True)
            return int(action), 0.8  # SB3 doesn't provide confidence

        # Simple policy
        if hasattr(self, '_simple_policy'):
            # Extract features from observation
            if len(observation) >= 10:
                rsi = observation[4] * 100  # Denormalize
                ibs = observation[5]

                if rsi < self._simple_policy['rsi_threshold'] and ibs < self._simple_policy['ibs_threshold']:
                    return 1, 0.6  # BUY
                elif rsi > 70:
                    return 2, 0.6  # SELL

        return 0, 0.5  # HOLD

    def evaluate(
        self,
        test_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Evaluate agent on test data.

        Args:
            test_data: Test OHLCV data

        Returns:
            Evaluation metrics dict
        """
        if test_data is None or test_data.empty:
            return {'error': 'no_test_data'}

        env = TradingEnv(price_data=test_data, reward_type='sharpe')
        obs = env.reset()

        total_reward = 0.0
        n_trades = 0
        wins = 0

        while True:
            action, _ = self.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                break

        n_trades = len(env.trades)
        wins = sum(1 for t in env.trades if t['pnl'] > 0)

        return {
            'total_reward': total_reward,
            'portfolio_value': env.portfolio_value,
            'return_pct': (env.portfolio_value / env.initial_capital - 1) * 100,
            'n_trades': n_trades,
            'win_rate': wins / max(n_trades, 1),
        }

    def save(self, path: str) -> None:
        """Save trained model."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.model is not None and SB3_AVAILABLE:
            self.model.save(save_path / 'model')

        # Save config
        config_data = {
            'algorithm': self.config.algorithm,
            'learning_rate': self.config.learning_rate,
            'gamma': self.config.gamma,
            'total_timesteps': self.config.total_timesteps,
            'trained': self._trained,
            'saved_at': datetime.utcnow().isoformat(),
        }

        if hasattr(self, '_simple_policy'):
            config_data['simple_policy'] = self._simple_policy

        with open(save_path / 'config.json', 'w') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Saved RL agent to {save_path}")

    def load(self, path: str) -> None:
        """Load trained model."""
        load_path = Path(path)

        # Load config
        config_file = load_path / 'config.json'
        if config_file.exists():
            with open(config_file) as f:
                config_data = json.load(f)

            if 'simple_policy' in config_data:
                self._simple_policy = config_data['simple_policy']

            self._trained = config_data.get('trained', False)

        # Load SB3 model if available
        model_file = load_path / 'model.zip'
        if model_file.exists() and SB3_AVAILABLE:
            algorithm_cls = {
                'PPO': PPO,
                'DQN': DQN,
                'A2C': A2C,
            }.get(self.config.algorithm, PPO)
            self.model = algorithm_cls.load(model_file)
            self._trained = True

        logger.info(f"Loaded RL agent from {load_path}")
