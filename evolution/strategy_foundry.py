"""
Strategy Foundry - Genetic Programming for Strategy Discovery
==============================================================

Uses Genetic Programming (GP) to autonomously discover new trading
strategies from a universe of stocks. Evolves expression trees using
technical indicator primitives.

Key Phases:
1. Initialization - Generate random strategy population
2. Fitness Evaluation - Backtest each strategy
3. Selection - Tournament selection of fittest
4. Crossover/Mutation - Create offspring strategies
5. Repetition - Iterate until convergence

Usage:
    from evolution.strategy_foundry import StrategyFoundry

    foundry = StrategyFoundry(
        primitives=["SMA", "RSI", "IBS", "ATR"],
        population_size=100,
        generations=50,
    )

    # Evolve strategies on historical data
    best_strategies = foundry.evolve(data, n_best=5)

    # Export to SymbolicReasoner format
    foundry.export_rules(best_strategies, "config/evolved_rules.yaml")
"""

from __future__ import annotations

import copy
import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Expression Tree Nodes
# =============================================================================

class Node(ABC):
    """Abstract base class for expression tree nodes."""

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate node on dataframe, return boolean series."""
        pass

    @abstractmethod
    def to_string(self) -> str:
        """Convert to human-readable string."""
        pass

    @abstractmethod
    def depth(self) -> int:
        """Return depth of subtree."""
        pass

    @abstractmethod
    def copy(self) -> 'Node':
        """Deep copy of node."""
        pass


class TerminalNode(Node):
    """Terminal node - constant or indicator value."""

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate terminal (indicator or constant)."""
        if self.name == "CONST":
            return pd.Series(self.params.get("value", 0), index=df.index)

        # Technical indicators
        close = df['close'] if 'close' in df.columns else df['Close']
        high = df.get('high', df.get('High', close))
        low = df.get('low', df.get('Low', close))
        volume = df.get('volume', df.get('Volume', pd.Series(1, index=df.index)))

        if self.name == "SMA":
            period = self.params.get("period", 20)
            return close.rolling(window=period).mean()

        elif self.name == "EMA":
            period = self.params.get("period", 20)
            return close.ewm(span=period, adjust=False).mean()

        elif self.name == "RSI":
            period = self.params.get("period", 14)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, 1e-10)
            return 100 - (100 / (1 + rs))

        elif self.name == "IBS":
            # Internal Bar Strength
            range_hl = high - low
            return (close - low) / range_hl.replace(0, 1e-10)

        elif self.name == "ATR":
            period = self.params.get("period", 14)
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            return tr.rolling(window=period).mean()

        elif self.name == "VOLUME_SMA":
            period = self.params.get("period", 20)
            return volume.rolling(window=period).mean()

        elif self.name == "CLOSE":
            return close

        elif self.name == "HIGH":
            return high

        elif self.name == "LOW":
            return low

        elif self.name == "VOLUME":
            return volume

        elif self.name == "DONCHIAN_HIGH":
            period = self.params.get("period", 20)
            return high.rolling(window=period).max()

        elif self.name == "DONCHIAN_LOW":
            period = self.params.get("period", 20)
            return low.rolling(window=period).min()

        elif self.name == "BB_UPPER":
            period = self.params.get("period", 20)
            std_mult = self.params.get("std", 2)
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            return sma + std_mult * std

        elif self.name == "BB_LOWER":
            period = self.params.get("period", 20)
            std_mult = self.params.get("std", 2)
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            return sma - std_mult * std

        else:
            logger.warning(f"Unknown terminal: {self.name}")
            return pd.Series(0, index=df.index)

    def to_string(self) -> str:
        if self.params:
            param_str = ",".join(f"{k}={v}" for k, v in self.params.items())
            return f"{self.name}({param_str})"
        return self.name

    def depth(self) -> int:
        return 1

    def copy(self) -> 'TerminalNode':
        return TerminalNode(self.name, self.params.copy())


class OperatorNode(Node):
    """Operator node - combines child nodes."""

    OPERATORS = ["AND", "OR", "GT", "LT", "GTE", "LTE", "CROSS_ABOVE", "CROSS_BELOW"]

    def __init__(self, operator: str, left: Node, right: Node):
        if operator not in self.OPERATORS:
            raise ValueError(f"Unknown operator: {operator}")
        self.operator = operator
        self.left = left
        self.right = right

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate operator on child nodes."""
        left_val = self.left.evaluate(df)
        right_val = self.right.evaluate(df)

        if self.operator == "AND":
            return left_val & right_val

        elif self.operator == "OR":
            return left_val | right_val

        elif self.operator == "GT":
            return left_val > right_val

        elif self.operator == "LT":
            return left_val < right_val

        elif self.operator == "GTE":
            return left_val >= right_val

        elif self.operator == "LTE":
            return left_val <= right_val

        elif self.operator == "CROSS_ABOVE":
            return (left_val > right_val) & (left_val.shift(1) <= right_val.shift(1))

        elif self.operator == "CROSS_BELOW":
            return (left_val < right_val) & (left_val.shift(1) >= right_val.shift(1))

        else:
            return pd.Series(False, index=df.index)

    def to_string(self) -> str:
        return f"({self.left.to_string()} {self.operator} {self.right.to_string()})"

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def copy(self) -> 'OperatorNode':
        return OperatorNode(self.operator, self.left.copy(), self.right.copy())


# =============================================================================
# Strategy Individual
# =============================================================================

@dataclass
class StrategyIndividual:
    """An individual strategy (expression tree) in the population."""
    entry_rule: Node
    exit_rule: Optional[Node] = None
    fitness: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    generation: int = 0

    def evaluate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate entry signals from the strategy."""
        return self.entry_rule.evaluate(df).fillna(False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_rule': self.entry_rule.to_string(),
            'exit_rule': self.exit_rule.to_string() if self.exit_rule else None,
            'fitness': round(self.fitness, 4),
            'win_rate': round(self.win_rate, 4),
            'profit_factor': round(self.profit_factor, 4),
            'n_trades': self.n_trades,
            'generation': self.generation,
        }

    def copy(self) -> 'StrategyIndividual':
        return StrategyIndividual(
            entry_rule=self.entry_rule.copy(),
            exit_rule=self.exit_rule.copy() if self.exit_rule else None,
            fitness=self.fitness,
            win_rate=self.win_rate,
            profit_factor=self.profit_factor,
            n_trades=self.n_trades,
            generation=self.generation,
        )


# =============================================================================
# Strategy Foundry
# =============================================================================

class StrategyFoundry:
    """
    Genetic Programming engine for evolving trading strategies.

    Evolves a population of strategy expression trees using:
    - Tournament selection
    - Subtree crossover
    - Point and subtree mutation
    """

    # Default primitives
    DEFAULT_TERMINALS = [
        ("SMA", {"period": [5, 10, 20, 50, 200]}),
        ("EMA", {"period": [5, 10, 20, 50]}),
        ("RSI", {"period": [2, 5, 14, 21]}),
        ("IBS", {}),
        ("ATR", {"period": [7, 14, 21]}),
        ("DONCHIAN_HIGH", {"period": [10, 20, 55]}),
        ("DONCHIAN_LOW", {"period": [10, 20, 55]}),
        ("CLOSE", {}),
        ("CONST", {"value": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 5, 10, 20, 30, 50, 70, 80, 95]}),
    ]

    DEFAULT_OPERATORS = ["AND", "OR", "GT", "LT", "CROSS_ABOVE", "CROSS_BELOW"]

    def __init__(
        self,
        terminals: Optional[List[Tuple[str, Dict]]] = None,
        operators: Optional[List[str]] = None,
        population_size: int = 100,
        generations: int = 50,
        tournament_size: int = 5,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        max_depth: int = 6,
        min_trades: int = 10,
        fitness_metric: str = "sharpe",  # sharpe, profit_factor, win_rate
        elitism: int = 5,
    ):
        self.terminals = terminals or self.DEFAULT_TERMINALS
        self.operators = operators or self.DEFAULT_OPERATORS
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_depth = max_depth
        self.min_trades = min_trades
        self.fitness_metric = fitness_metric
        self.elitism = elitism

        self._population: List[StrategyIndividual] = []
        self._best_individuals: List[StrategyIndividual] = []
        self._generation = 0

        logger.info(
            f"StrategyFoundry initialized: pop_size={population_size}, "
            f"generations={generations}, fitness={fitness_metric}"
        )

    def _random_terminal(self) -> TerminalNode:
        """Generate random terminal node."""
        name, param_options = random.choice(self.terminals)
        params = {}
        for key, values in param_options.items():
            if isinstance(values, list):
                params[key] = random.choice(values)
            else:
                params[key] = values
        return TerminalNode(name, params)

    def _random_operator(self) -> str:
        """Select random operator."""
        return random.choice(self.operators)

    def _random_tree(self, depth: int = 0) -> Node:
        """Generate random expression tree."""
        if depth >= self.max_depth or (depth > 1 and random.random() < 0.3):
            return self._random_terminal()

        operator = self._random_operator()
        left = self._random_tree(depth + 1)
        right = self._random_tree(depth + 1)
        return OperatorNode(operator, left, right)

    def _initialize_population(self) -> None:
        """Create initial random population."""
        self._population = []
        for _ in range(self.population_size):
            entry_rule = self._random_tree()
            individual = StrategyIndividual(entry_rule=entry_rule, generation=0)
            self._population.append(individual)
        logger.info(f"Initialized population with {self.population_size} individuals")

    def _evaluate_fitness(
        self,
        individual: StrategyIndividual,
        df: pd.DataFrame,
    ) -> float:
        """
        Evaluate fitness of a strategy on historical data.

        Uses a simple backtesting approach:
        - Entry on signal
        - Exit after N bars or fixed target/stop
        """
        try:
            signals = individual.evaluate_signals(df)
            signals = signals.shift(1).fillna(False)  # No lookahead

            # Simple P&L calculation
            close = df['close'] if 'close' in df.columns else df['Close']
            returns = close.pct_change().shift(-1)  # Next bar return

            # Apply signals
            signal_returns = returns[signals]

            n_trades = len(signal_returns)
            if n_trades < self.min_trades:
                return -1.0  # Penalize strategies with too few trades

            wins = (signal_returns > 0).sum()
            (signal_returns <= 0).sum()
            win_rate = wins / n_trades if n_trades > 0 else 0

            total_gain = signal_returns[signal_returns > 0].sum()
            total_loss = abs(signal_returns[signal_returns <= 0].sum())
            profit_factor = total_gain / total_loss if total_loss > 0 else total_gain

            # Sharpe-like metric
            mean_ret = signal_returns.mean()
            std_ret = signal_returns.std()
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

            # Store metrics
            individual.win_rate = win_rate
            individual.profit_factor = profit_factor
            individual.n_trades = n_trades

            # Calculate fitness based on selected metric
            if self.fitness_metric == "sharpe":
                fitness = sharpe
            elif self.fitness_metric == "profit_factor":
                fitness = profit_factor
            elif self.fitness_metric == "win_rate":
                fitness = win_rate
            else:
                fitness = sharpe

            individual.fitness = fitness
            return fitness

        except Exception as e:
            logger.warning(f"Fitness evaluation error: {e}")
            return -1.0

    def _tournament_select(self) -> StrategyIndividual:
        """Tournament selection."""
        tournament = random.sample(self._population, min(self.tournament_size, len(self._population)))
        return max(tournament, key=lambda ind: ind.fitness)

    def _get_random_node(self, node: Node, depth: int = 0) -> Tuple[Node, int]:
        """Get a random node from the tree."""
        if isinstance(node, TerminalNode):
            return node, depth

        if random.random() < 0.3:
            return node, depth

        if random.random() < 0.5:
            return self._get_random_node(node.left, depth + 1)
        else:
            return self._get_random_node(node.right, depth + 1)

    def _crossover(
        self,
        parent1: StrategyIndividual,
        parent2: StrategyIndividual,
    ) -> Tuple[StrategyIndividual, StrategyIndividual]:
        """Subtree crossover between two parents."""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Simple swap of subtrees (simplified implementation)
        if random.random() < 0.5:
            child1.entry_rule, child2.entry_rule = child2.entry_rule, child1.entry_rule

        child1.generation = self._generation
        child2.generation = self._generation

        return child1, child2

    def _mutate(self, individual: StrategyIndividual) -> StrategyIndividual:
        """Mutate an individual."""
        mutant = individual.copy()

        # Replace entry rule with new random subtree
        if random.random() < 0.5:
            mutant.entry_rule = self._random_tree()
        else:
            # Point mutation - just regenerate
            mutant.entry_rule = self._random_tree()

        mutant.generation = self._generation
        return mutant

    def evolve(
        self,
        data: pd.DataFrame,
        n_best: int = 5,
        verbose: bool = True,
    ) -> List[StrategyIndividual]:
        """
        Run genetic programming evolution.

        Args:
            data: Historical OHLCV data
            n_best: Number of best strategies to return
            verbose: Print progress

        Returns:
            List of best evolved strategies
        """
        logger.info(f"Starting evolution: {self.generations} generations on {len(data)} bars")

        # Initialize
        self._initialize_population()

        for gen in range(self.generations):
            self._generation = gen

            # Evaluate fitness
            for individual in self._population:
                self._evaluate_fitness(individual, data)

            # Sort by fitness
            self._population.sort(key=lambda x: x.fitness, reverse=True)

            # Track best
            if self._population[0].fitness > 0:
                self._best_individuals.append(self._population[0].copy())

            if verbose and gen % 10 == 0:
                best = self._population[0]
                logger.info(
                    f"Gen {gen}: best_fitness={best.fitness:.3f}, "
                    f"wr={best.win_rate:.1%}, pf={best.profit_factor:.2f}, "
                    f"trades={best.n_trades}"
                )

            # Create next generation
            new_population = []

            # Elitism - keep best individuals
            for i in range(self.elitism):
                new_population.append(self._population[i].copy())

            # Fill rest with offspring
            while len(new_population) < self.population_size:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()

                if random.random() < self.crossover_prob:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                if random.random() < self.mutation_prob:
                    child1 = self._mutate(child1)
                if random.random() < self.mutation_prob:
                    child2 = self._mutate(child2)

                new_population.extend([child1, child2])

            self._population = new_population[:self.population_size]

        # Final evaluation
        for individual in self._population:
            self._evaluate_fitness(individual, data)

        self._population.sort(key=lambda x: x.fitness, reverse=True)

        best_strategies = self._population[:n_best]
        logger.info(f"Evolution complete. Best fitness: {best_strategies[0].fitness:.3f}")

        return best_strategies

    def export_rules(
        self,
        strategies: List[StrategyIndividual],
        output_path: str = "config/evolved_rules.yaml",
    ) -> None:
        """Export evolved strategies to YAML format for SymbolicReasoner."""
        import yaml

        rules = []
        for i, strategy in enumerate(strategies):
            rule = {
                'name': f"evolved_strategy_{i+1}",
                'condition': strategy.entry_rule.to_string(),
                'fitness': strategy.fitness,
                'win_rate': strategy.win_rate,
                'profit_factor': strategy.profit_factor,
                'n_trades': strategy.n_trades,
                'generation': strategy.generation,
            }
            rules.append(rule)

        output = {'evolved_strategies': rules}

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(output, f, default_flow_style=False)

        logger.info(f"Exported {len(strategies)} evolved strategies to {output_path}")

    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get statistics about the evolution run."""
        if not self._best_individuals:
            return {'status': 'not_run'}

        return {
            'generations_run': self._generation + 1,
            'population_size': self.population_size,
            'best_fitness': max(ind.fitness for ind in self._best_individuals),
            'best_win_rate': max(ind.win_rate for ind in self._best_individuals),
            'best_profit_factor': max(ind.profit_factor for ind in self._best_individuals),
            'n_best_tracked': len(self._best_individuals),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def run_strategy_foundry(
    data: pd.DataFrame,
    generations: int = 50,
    population_size: int = 100,
    n_best: int = 5,
) -> List[Dict[str, Any]]:
    """
    Run strategy foundry and return best strategies as dicts.

    Args:
        data: Historical OHLCV data
        generations: Number of generations to evolve
        population_size: Population size
        n_best: Number of best strategies to return

    Returns:
        List of strategy dictionaries
    """
    foundry = StrategyFoundry(
        generations=generations,
        population_size=population_size,
    )

    best = foundry.evolve(data, n_best=n_best)
    return [s.to_dict() for s in best]
