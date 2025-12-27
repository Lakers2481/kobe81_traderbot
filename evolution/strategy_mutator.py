"""
Strategy Mutator for Trading Systems
=====================================

Creates variations of existing trading strategies through
parameter perturbation, rule swapping, and structural mutations.

Enables exploration of strategy space while maintaining
core trading logic integrity.
"""

from __future__ import annotations

import random
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Type
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class MutationType(Enum):
    """Types of strategy mutations."""
    PARAMETER = "parameter"      # Adjust numeric parameters
    INDICATOR = "indicator"      # Swap/modify indicators
    FILTER = "filter"            # Add/remove/modify filters
    TIMING = "timing"            # Entry/exit timing changes
    SIZING = "sizing"            # Position sizing changes
    STOP_LOSS = "stop_loss"      # Stop loss modifications
    TAKE_PROFIT = "take_profit"  # Take profit modifications


@dataclass
class MutationRecord:
    """Record of a mutation applied."""
    mutation_type: MutationType
    original_value: Any
    new_value: Any
    parameter_name: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'type': self.mutation_type.value,
            'parameter': self.parameter_name,
            'original': str(self.original_value),
            'new': str(self.new_value),
            'description': self.description,
        }


@dataclass
class StrategyVariant:
    """A variant of a strategy created through mutation."""
    name: str
    params: Dict[str, Any]
    mutations: List[MutationRecord] = field(default_factory=list)
    parent_name: Optional[str] = None
    generation: int = 0
    fitness: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'params': self.params,
            'mutations': [m.to_dict() for m in self.mutations],
            'parent_name': self.parent_name,
            'generation': self.generation,
            'fitness': self.fitness,
        }


class StrategyMutator:
    """
    Creates variations of trading strategies through mutations.

    Supports multiple mutation types including parameter tweaks,
    indicator swaps, and structural changes.
    """

    def __init__(
        self,
        mutation_rate: float = 0.3,
        mutation_strength: float = 0.2,
        preserve_constraints: bool = True,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the strategy mutator.

        Args:
            mutation_rate: Probability of mutating each mutable element
            mutation_strength: Magnitude of mutations (0-1)
            preserve_constraints: Whether to enforce parameter constraints
            random_seed: Random seed for reproducibility
        """
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.preserve_constraints = preserve_constraints

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Parameter constraints (name -> (min, max))
        self.param_constraints: Dict[str, tuple] = {}

        # Indicator alternatives for swapping
        self.indicator_alternatives: Dict[str, List[str]] = {
            'rsi': ['stoch_rsi', 'williams_r', 'cci'],
            'sma': ['ema', 'wma', 'dema', 'tema'],
            'ema': ['sma', 'wma', 'dema'],
            'atr': ['true_range', 'average_range'],
            'macd': ['ppo', 'trix'],
            'bollinger': ['keltner', 'donchian'],
        }

        logger.info(
            f"StrategyMutator initialized with rate={mutation_rate}, "
            f"strength={mutation_strength}"
        )

    def add_constraint(self, param_name: str, min_val: float, max_val: float):
        """Add a constraint for a parameter."""
        self.param_constraints[param_name] = (min_val, max_val)

    def _mutate_numeric(
        self,
        value: float,
        param_name: str,
    ) -> tuple[float, str]:
        """Mutate a numeric parameter."""
        if param_name in self.param_constraints:
            min_val, max_val = self.param_constraints[param_name]
            range_size = max_val - min_val
        else:
            # Default: allow 50% deviation
            range_size = abs(value) * 0.5 if value != 0 else 1.0
            min_val = value - range_size
            max_val = value + range_size

        # Gaussian mutation
        delta = random.gauss(0, range_size * self.mutation_strength)
        new_value = value + delta

        # Apply constraints
        if self.preserve_constraints:
            new_value = max(min_val, min(max_val, new_value))

        # Preserve type
        if isinstance(value, int):
            new_value = int(round(new_value))

        description = f"{param_name}: {value} -> {new_value}"
        return new_value, description

    def _mutate_indicator(
        self,
        indicator: str,
    ) -> tuple[str, str]:
        """Swap an indicator for an alternative."""
        indicator_lower = indicator.lower()

        for base, alternatives in self.indicator_alternatives.items():
            if base in indicator_lower:
                new_indicator = random.choice(alternatives)
                # Preserve case pattern
                if indicator[0].isupper():
                    new_indicator = new_indicator.upper()
                description = f"Indicator swap: {indicator} -> {new_indicator}"
                return new_indicator, description

        # No swap available
        return indicator, ""

    def mutate_params(
        self,
        params: Dict[str, Any],
        mutation_types: Optional[List[MutationType]] = None,
    ) -> tuple[Dict[str, Any], List[MutationRecord]]:
        """
        Mutate strategy parameters.

        Args:
            params: Original parameters
            mutation_types: Types of mutations to apply (None = all)

        Returns:
            Tuple of (mutated_params, mutation_records)
        """
        if mutation_types is None:
            mutation_types = [MutationType.PARAMETER]

        mutated = copy.deepcopy(params)
        records = []

        for key, value in params.items():
            if random.random() > self.mutation_rate:
                continue

            # Numeric mutation
            if MutationType.PARAMETER in mutation_types:
                if isinstance(value, (int, float)):
                    new_value, desc = self._mutate_numeric(value, key)
                    if new_value != value:
                        records.append(MutationRecord(
                            mutation_type=MutationType.PARAMETER,
                            original_value=value,
                            new_value=new_value,
                            parameter_name=key,
                            description=desc,
                        ))
                        mutated[key] = new_value

            # Indicator swapping
            if MutationType.INDICATOR in mutation_types:
                if isinstance(value, str) and any(
                    ind in value.lower()
                    for ind in self.indicator_alternatives
                ):
                    new_value, desc = self._mutate_indicator(value)
                    if new_value != value and desc:
                        records.append(MutationRecord(
                            mutation_type=MutationType.INDICATOR,
                            original_value=value,
                            new_value=new_value,
                            parameter_name=key,
                            description=desc,
                        ))
                        mutated[key] = new_value

        return mutated, records

    def create_variant(
        self,
        base_params: Dict[str, Any],
        base_name: str = "strategy",
        generation: int = 0,
        mutation_types: Optional[List[MutationType]] = None,
    ) -> StrategyVariant:
        """
        Create a variant of a strategy through mutation.

        Args:
            base_params: Original strategy parameters
            base_name: Name of the base strategy
            generation: Generation number
            mutation_types: Types of mutations to apply

        Returns:
            StrategyVariant with mutated parameters
        """
        mutated_params, mutations = self.mutate_params(
            base_params,
            mutation_types,
        )

        variant_name = f"{base_name}_v{generation}_{len(mutations)}m"

        return StrategyVariant(
            name=variant_name,
            params=mutated_params,
            mutations=mutations,
            parent_name=base_name,
            generation=generation,
        )

    def create_variants(
        self,
        base_params: Dict[str, Any],
        count: int,
        base_name: str = "strategy",
        generation: int = 0,
    ) -> List[StrategyVariant]:
        """Create multiple variants of a strategy."""
        variants = []
        for i in range(count):
            variant = self.create_variant(
                base_params,
                f"{base_name}_{i}",
                generation,
            )
            variants.append(variant)
        return variants


def crossover_strategies(
    parent1: Dict[str, Any],
    parent2: Dict[str, Any],
    crossover_rate: float = 0.5,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Perform crossover between two strategy parameter sets.

    Args:
        parent1: First parent parameters
        parent2: Second parent parameters
        crossover_rate: Probability of swapping each parameter

    Returns:
        Tuple of two child parameter sets
    """
    child1 = {}
    child2 = {}

    # Get all keys from both parents
    all_keys = set(parent1.keys()) | set(parent2.keys())

    for key in all_keys:
        val1 = parent1.get(key)
        val2 = parent2.get(key)

        if val1 is None:
            child1[key] = val2
            child2[key] = val2
        elif val2 is None:
            child1[key] = val1
            child2[key] = val1
        elif random.random() < crossover_rate:
            child1[key] = val2
            child2[key] = val1
        else:
            child1[key] = val1
            child2[key] = val2

    return child1, child2


def mutate_strategy(
    params: Dict[str, Any],
    mutation_rate: float = 0.3,
    mutation_strength: float = 0.2,
) -> Dict[str, Any]:
    """
    Convenience function to mutate strategy parameters.

    Args:
        params: Original parameters
        mutation_rate: Probability of mutating each parameter
        mutation_strength: Magnitude of mutations

    Returns:
        Mutated parameters
    """
    mutator = StrategyMutator(
        mutation_rate=mutation_rate,
        mutation_strength=mutation_strength,
    )
    mutated, _ = mutator.mutate_params(params)
    return mutated


# Module-level mutator instance
_mutator: Optional[StrategyMutator] = None


def get_mutator() -> StrategyMutator:
    """Get or create the global mutator instance."""
    global _mutator
    if _mutator is None:
        _mutator = StrategyMutator()
    return _mutator
