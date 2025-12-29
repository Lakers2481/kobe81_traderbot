"""
Custom Gym environment for trading simulation.
"""
from __future__ import annotations

import logging
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import gym
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        gym = None
        spaces = None


class TradingEnv:
    """
    Custom trading environment for RL training.

    Supports standard Gym interface even without gym installed.
    """

    # Action definitions
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3

    def __init__(
        self,
        price_data: pd.DataFrame,
        initial_capital: float = 100000.0,
        max_position_pct: float = 0.10,
        transaction_cost_bps: float = 5.0,
        reward_type: str = 'r_multiple',
        lookback: int = 50,
    ):
        """
        Initialize trading environment.

        Args:
            price_data: OHLCV DataFrame with timestamp, open, high, low, close, volume
            initial_capital: Starting capital
            max_position_pct: Maximum position as % of capital
            transaction_cost_bps: Transaction costs in basis points
            reward_type: Reward function type ('r_multiple', 'sharpe', 'profit_factor')
            lookback: Number of bars for observation
        """
        self.price_data = price_data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.transaction_cost_bps = transaction_cost_bps
        self.reward_type = reward_type
        self.lookback = lookback

        # Precompute features
        self._compute_features()

        # State
        self.current_step = 0
        self.position = 0  # -1, 0, +1
        self.position_price = 0.0
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.trades: List[Dict] = []
        self.returns: List[float] = []

        # Gym spaces
        self.n_features = self._features.shape[1] if hasattr(self, '_features') else 10
        self.observation_space_shape = (self.n_features + 3,)  # features + position info
        self.action_space_n = 4  # hold, buy, sell, close

    def _compute_features(self) -> None:
        """Compute technical features from price data."""
        df = self.price_data.copy()

        if 'close' not in df.columns:
            logger.warning("No 'close' column in price data")
            self._features = np.zeros((len(df), 10))
            return

        close = df['close'].values
        high = df['high'].values if 'high' in df.columns else close
        low = df['low'].values if 'low' in df.columns else close
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))

        features = []

        # Returns
        returns = np.zeros(len(close))
        returns[1:] = (close[1:] / close[:-1]) - 1
        features.append(returns)

        # 5-day returns
        ret5 = np.zeros(len(close))
        ret5[5:] = (close[5:] / close[:-5]) - 1
        features.append(ret5)

        # 20-day returns
        ret20 = np.zeros(len(close))
        ret20[20:] = (close[20:] / close[:-20]) - 1
        features.append(ret20)

        # Volatility (20-day)
        vol = pd.Series(returns).rolling(20).std().fillna(0).values
        features.append(vol)

        # RSI-like (simplified)
        delta = np.diff(close, prepend=close[0])
        up = np.clip(delta, 0, None)
        down = -np.clip(delta, None, 0)
        avg_up = pd.Series(up).rolling(14).mean().fillna(0).values
        avg_down = pd.Series(down).rolling(14).mean().fillna(1e-10).values
        rsi = 100 - (100 / (1 + avg_up / avg_down))
        rsi = np.nan_to_num(rsi, nan=50.0)
        features.append(rsi / 100)  # Normalize

        # IBS
        range_hl = high - low
        range_hl = np.where(range_hl < 1e-10, 1e-10, range_hl)
        ibs = (close - low) / range_hl
        features.append(ibs)

        # Volume ratio
        vol_sma = pd.Series(volume).rolling(20).mean().fillna(volume[0]).values
        vol_ratio = volume / (vol_sma + 1)
        features.append(np.clip(vol_ratio, 0, 5))

        # Price position in range
        high_20 = pd.Series(high).rolling(20).max().fillna(high[0]).values
        low_20 = pd.Series(low).rolling(20).min().fillna(low[0]).values
        range_20 = high_20 - low_20
        range_20 = np.where(range_20 < 1e-10, 1e-10, range_20)
        price_pos = (close - low_20) / range_20
        features.append(price_pos)

        # SMA distance
        sma_20 = pd.Series(close).rolling(20).mean().fillna(close[0]).values
        sma_dist = (close - sma_20) / (sma_20 + 1e-10)
        features.append(np.clip(sma_dist, -0.5, 0.5))

        # Trend (simple)
        sma_50 = pd.Series(close).rolling(50).mean().fillna(close[0]).values
        trend = (sma_20 > sma_50).astype(float)
        features.append(trend)

        self._features = np.column_stack(features)

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = self.lookback
        self.position = 0
        self.position_price = 0.0
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.trades = []
        self.returns = []
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_step >= len(self._features):
            return np.zeros(self.observation_space_shape)

        features = self._features[self.current_step]

        # Add position info
        position_info = np.array([
            self.position,  # Current position
            self.position_price / (self.price_data['close'].iloc[self.current_step] + 1e-10) if self.position != 0 else 0,
            self.portfolio_value / self.initial_capital - 1,  # Return so far
        ])

        return np.concatenate([features, position_info]).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0=hold, 1=buy, 2=sell, 3=close)

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.current_step >= len(self.price_data) - 1:
            return self._get_observation(), 0.0, True, {}

        current_price = self.price_data['close'].iloc[self.current_step]
        next_price = self.price_data['close'].iloc[self.current_step + 1]

        reward = 0.0
        position_before = self.position

        # Execute action
        if action == self.BUY and self.position <= 0:
            # Close short if any, go long
            if self.position < 0:
                reward += self._close_position(current_price)
            self._open_position(1, current_price)

        elif action == self.SELL and self.position >= 0:
            # Close long if any, go short
            if self.position > 0:
                reward += self._close_position(current_price)
            self._open_position(-1, current_price)

        elif action == self.CLOSE and self.position != 0:
            reward += self._close_position(current_price)

        # Update portfolio value
        if self.position != 0:
            position_value = self.max_position_pct * self.initial_capital
            pnl = position_value * self.position * (next_price / current_price - 1)
            self.portfolio_value += pnl

        # Track returns
        ret = (self.portfolio_value / self.initial_capital) - 1
        self.returns.append(ret)

        # Calculate reward based on type
        if self.reward_type == 'r_multiple' and position_before != 0:
            reward = self._calculate_r_multiple_reward(position_before, current_price, next_price)
        elif self.reward_type == 'sharpe' and len(self.returns) > 10:
            reward = self._calculate_sharpe_reward()
        else:
            reward = (self.portfolio_value / self.initial_capital) - 1

        self.current_step += 1
        done = self.current_step >= len(self.price_data) - 1

        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'trades': len(self.trades),
        }

        return self._get_observation(), reward, done, info

    def _open_position(self, direction: int, price: float) -> None:
        """Open a new position."""
        cost = self.max_position_pct * self.initial_capital * (self.transaction_cost_bps / 10000)
        self.cash -= cost
        self.position = direction
        self.position_price = price

    def _close_position(self, price: float) -> float:
        """Close current position and return P&L."""
        if self.position == 0:
            return 0.0

        position_value = self.max_position_pct * self.initial_capital
        pnl = position_value * self.position * (price / self.position_price - 1)

        # Transaction cost
        cost = position_value * (self.transaction_cost_bps / 10000)
        pnl -= cost

        self.trades.append({
            'entry_price': self.position_price,
            'exit_price': price,
            'direction': self.position,
            'pnl': pnl,
        })

        self.cash += pnl
        self.position = 0
        self.position_price = 0.0

        return pnl

    def _calculate_r_multiple_reward(
        self,
        position: int,
        current_price: float,
        next_price: float,
    ) -> float:
        """Calculate R-multiple based reward."""
        if position == 0 or self.position_price == 0:
            return 0.0

        # Risk is 2% of position
        risk = self.position_price * 0.02
        if risk < 0.01:
            return 0.0

        # Reward
        ret = position * (next_price - current_price)
        return ret / risk

    def _calculate_sharpe_reward(self) -> float:
        """Calculate Sharpe-based reward from recent returns."""
        if len(self.returns) < 10:
            return 0.0

        recent = np.array(self.returns[-20:])
        if np.std(recent) < 1e-10:
            return 0.0

        return float(np.mean(recent) / np.std(recent))

    def render(self, mode: str = 'human') -> None:
        """Render environment state."""
        print(f"Step: {self.current_step}, Position: {self.position}, "
              f"Portfolio: ${self.portfolio_value:.2f}, Trades: {len(self.trades)}")
