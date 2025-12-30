"""
Execution Bandit - Multi-Armed Bandit for Execution Strategy Selection
=======================================================================

Uses Thompson Sampling, UCB, or epsilon-greedy algorithms to learn the
optimal execution strategy (IOC, TWAP, VWAP, LIMIT) based on historical
slippage performance.

The bandit learns which execution strategy minimizes slippage for each
symbol or market condition, adapting over time as it collects more data.

Usage:
    from execution.execution_bandit import ExecutionBandit

    bandit = ExecutionBandit(strategies=["IOC", "TWAP", "VWAP"])

    # Select strategy for a trade
    strategy = bandit.select_strategy(symbol="AAPL", context={"volatility": 0.02})

    # After execution, update with observed slippage
    bandit.update(symbol="AAPL", strategy="TWAP", slippage=-0.0005)  # negative = good

    # Get arm statistics
    stats = bandit.get_stats()
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ArmStats:
    """Statistics for a single bandit arm (execution strategy)."""
    name: str
    n_pulls: int = 0
    total_reward: float = 0.0
    sum_squared_reward: float = 0.0

    # For Thompson Sampling (Beta distribution parameters)
    alpha: float = 1.0  # Successes + 1
    beta: float = 1.0   # Failures + 1

    @property
    def mean_reward(self) -> float:
        """Average reward (negative slippage is good)."""
        if self.n_pulls == 0:
            return 0.0
        return self.total_reward / self.n_pulls

    @property
    def std_reward(self) -> float:
        """Standard deviation of rewards."""
        if self.n_pulls < 2:
            return float('inf')
        variance = (self.sum_squared_reward / self.n_pulls) - (self.mean_reward ** 2)
        return math.sqrt(max(0, variance))

    @property
    def ucb_score(self) -> float:
        """Upper Confidence Bound score (for UCB algorithm)."""
        # Will be computed with total pulls context
        return self.mean_reward

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'n_pulls': self.n_pulls,
            'mean_reward': round(self.mean_reward, 6),
            'std_reward': round(self.std_reward, 6) if self.std_reward != float('inf') else 'inf',
            'alpha': self.alpha,
            'beta': self.beta,
        }


@dataclass
class BanditStats:
    """Overall bandit statistics."""
    total_pulls: int
    total_reward: float
    arm_stats: Dict[str, Dict[str, Any]]
    best_arm: str
    regret_estimate: float
    last_updated: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Bandit Algorithms
# =============================================================================

class ExecutionBandit:
    """
    Multi-armed bandit for execution strategy selection.

    Learns which execution strategy (IOC, TWAP, VWAP, LIMIT) minimizes
    slippage over time using one of several algorithms.

    Args:
        strategies: List of execution strategy names (arms)
        algorithm: "thompson", "ucb", or "epsilon_greedy"
        epsilon: Exploration rate for epsilon-greedy (default 0.1)
        ucb_c: Exploration constant for UCB (default 2.0)
        state_file: Path to persist bandit state
    """

    ALGORITHMS = ["thompson", "ucb", "epsilon_greedy"]

    def __init__(
        self,
        strategies: Optional[List[str]] = None,
        algorithm: str = "thompson",
        epsilon: float = 0.1,
        ucb_c: float = 2.0,
        state_file: str = "state/execution_bandit.json",
    ):
        if strategies is None:
            strategies = ["IOC", "TWAP", "VWAP", "LIMIT"]

        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use one of {self.ALGORITHMS}")

        self.strategies = strategies
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.ucb_c = ucb_c
        self.state_file = Path(state_file)

        # Initialize arms
        self._arms: Dict[str, ArmStats] = {
            name: ArmStats(name=name) for name in strategies
        }

        # Per-symbol arm stats (optional, for symbol-specific learning)
        self._symbol_arms: Dict[str, Dict[str, ArmStats]] = {}

        # Load existing state
        self._load_state()

        logger.info(f"ExecutionBandit initialized: algorithm={algorithm}, strategies={strategies}")

    def select_strategy(
        self,
        symbol: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Select an execution strategy using the bandit algorithm.

        Args:
            symbol: Optional symbol for symbol-specific learning
            context: Optional context dict (volatility, size, etc.) - future use

        Returns:
            Selected strategy name
        """
        arms = self._get_arms(symbol)

        if self.algorithm == "thompson":
            return self._thompson_select(arms)
        elif self.algorithm == "ucb":
            return self._ucb_select(arms)
        else:  # epsilon_greedy
            return self._epsilon_greedy_select(arms)

    def _thompson_select(self, arms: Dict[str, ArmStats]) -> str:
        """Thompson Sampling: Sample from Beta posterior and pick highest."""
        samples = {}
        for name, arm in arms.items():
            # Sample from Beta(alpha, beta) distribution
            # Higher alpha = more successes, higher expected value
            sample = np.random.beta(arm.alpha, arm.beta)
            samples[name] = sample

        selected = max(samples, key=samples.get)
        logger.debug(f"Thompson sampling: {samples}, selected={selected}")
        return selected

    def _ucb_select(self, arms: Dict[str, ArmStats]) -> str:
        """Upper Confidence Bound: Pick arm with highest UCB score."""
        total_pulls = sum(arm.n_pulls for arm in arms.values())

        if total_pulls == 0:
            # Random selection when no data
            return random.choice(list(arms.keys()))

        ucb_scores = {}
        for name, arm in arms.items():
            if arm.n_pulls == 0:
                # Unexplored arms get infinite UCB
                ucb_scores[name] = float('inf')
            else:
                # UCB formula: mean + c * sqrt(log(t) / n)
                exploration = self.ucb_c * math.sqrt(math.log(total_pulls) / arm.n_pulls)
                ucb_scores[name] = arm.mean_reward + exploration

        selected = max(ucb_scores, key=ucb_scores.get)
        logger.debug(f"UCB scores: {ucb_scores}, selected={selected}")
        return selected

    def _epsilon_greedy_select(self, arms: Dict[str, ArmStats]) -> str:
        """Epsilon-greedy: Exploit best arm (1-epsilon), explore randomly (epsilon)."""
        if random.random() < self.epsilon:
            # Explore: random selection
            selected = random.choice(list(arms.keys()))
            logger.debug(f"Epsilon-greedy: exploring, selected={selected}")
        else:
            # Exploit: pick best mean reward
            selected = max(arms, key=lambda k: arms[k].mean_reward)
            logger.debug(f"Epsilon-greedy: exploiting, selected={selected}")

        return selected

    def update(
        self,
        strategy: str,
        slippage: float,
        symbol: Optional[str] = None,
        success_threshold: float = 0.001,
    ) -> None:
        """
        Update bandit with observed execution result.

        Args:
            strategy: The execution strategy that was used
            slippage: Observed slippage (negative = better than expected)
            symbol: Optional symbol for symbol-specific learning
            success_threshold: Slippage below this is considered "success"
        """
        if strategy not in self.strategies:
            logger.warning(f"Unknown strategy: {strategy}")
            return

        # Convert slippage to reward (negative slippage = positive reward)
        reward = -slippage

        # Determine success/failure for Thompson Sampling
        is_success = slippage < success_threshold

        # Update global arms
        arm = self._arms[strategy]
        arm.n_pulls += 1
        arm.total_reward += reward
        arm.sum_squared_reward += reward ** 2

        # Update Beta distribution parameters for Thompson
        if is_success:
            arm.alpha += 1
        else:
            arm.beta += 1

        # Update symbol-specific arms if symbol provided
        if symbol:
            if symbol not in self._symbol_arms:
                self._symbol_arms[symbol] = {
                    name: ArmStats(name=name) for name in self.strategies
                }

            sym_arm = self._symbol_arms[symbol][strategy]
            sym_arm.n_pulls += 1
            sym_arm.total_reward += reward
            sym_arm.sum_squared_reward += reward ** 2
            if is_success:
                sym_arm.alpha += 1
            else:
                sym_arm.beta += 1

        # Persist state
        self._save_state()

        logger.info(
            f"Bandit updated: strategy={strategy}, slippage={slippage:.4f}, "
            f"reward={reward:.4f}, success={is_success}"
        )

    def _get_arms(self, symbol: Optional[str] = None) -> Dict[str, ArmStats]:
        """Get arm stats for symbol (if available) or global."""
        if symbol and symbol in self._symbol_arms:
            return self._symbol_arms[symbol]
        return self._arms

    def get_stats(self, symbol: Optional[str] = None) -> BanditStats:
        """Get bandit statistics."""
        arms = self._get_arms(symbol)

        total_pulls = sum(arm.n_pulls for arm in arms.values())
        total_reward = sum(arm.total_reward for arm in arms.values())

        # Best arm by mean reward
        best_arm = max(arms, key=lambda k: arms[k].mean_reward) if total_pulls > 0 else self.strategies[0]

        # Estimate regret (difference from optimal)
        best_mean = arms[best_arm].mean_reward
        regret = sum(
            arm.n_pulls * (best_mean - arm.mean_reward)
            for arm in arms.values()
        )

        return BanditStats(
            total_pulls=total_pulls,
            total_reward=round(total_reward, 6),
            arm_stats={name: arm.to_dict() for name, arm in arms.items()},
            best_arm=best_arm,
            regret_estimate=round(regret, 6),
            last_updated=datetime.utcnow().isoformat(),
        )

    def get_best_strategy(self, symbol: Optional[str] = None) -> Tuple[str, float]:
        """Get the current best strategy and its mean reward."""
        arms = self._get_arms(symbol)
        best_arm = max(arms, key=lambda k: arms[k].mean_reward)
        return best_arm, arms[best_arm].mean_reward

    def _save_state(self) -> None:
        """Persist bandit state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'algorithm': self.algorithm,
            'epsilon': self.epsilon,
            'ucb_c': self.ucb_c,
            'arms': {name: asdict(arm) for name, arm in self._arms.items()},
            'symbol_arms': {
                sym: {name: asdict(arm) for name, arm in arms.items()}
                for sym, arms in self._symbol_arms.items()
            },
            'last_updated': datetime.utcnow().isoformat(),
        }

        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save bandit state: {e}")

    def _load_state(self) -> None:
        """Load bandit state from file."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Restore global arms
            for name, arm_data in state.get('arms', {}).items():
                if name in self._arms:
                    self._arms[name] = ArmStats(**arm_data)

            # Restore symbol-specific arms
            for symbol, symbol_arms in state.get('symbol_arms', {}).items():
                self._symbol_arms[symbol] = {}
                for name, arm_data in symbol_arms.items():
                    self._symbol_arms[symbol][name] = ArmStats(**arm_data)

            logger.info(f"Loaded bandit state from {self.state_file}")

        except Exception as e:
            logger.warning(f"Failed to load bandit state: {e}")

    def reset(self) -> None:
        """Reset all arm statistics."""
        self._arms = {name: ArmStats(name=name) for name in self.strategies}
        self._symbol_arms = {}
        self._save_state()
        logger.info("Bandit state reset")


# =============================================================================
# Global Instance
# =============================================================================

_global_bandit: Optional[ExecutionBandit] = None


def get_execution_bandit(
    strategies: Optional[List[str]] = None,
    algorithm: str = "thompson",
    **kwargs,
) -> ExecutionBandit:
    """Get or create global execution bandit instance."""
    global _global_bandit

    if _global_bandit is None:
        _global_bandit = ExecutionBandit(
            strategies=strategies,
            algorithm=algorithm,
            **kwargs,
        )

    return _global_bandit


def set_execution_bandit(bandit: ExecutionBandit) -> None:
    """Set global execution bandit instance."""
    global _global_bandit
    _global_bandit = bandit
