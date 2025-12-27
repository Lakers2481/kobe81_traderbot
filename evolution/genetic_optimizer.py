"""
Genetic Optimizer for Trading Strategies
=========================================

Evolves trading strategy parameters using genetic algorithms.
Supports continuous and discrete parameter optimization with
elitism, crossover, and mutation operators.

Research shows GAs "can be used to continuously optimize trading
parameters to adjust to evolving market conditions."
"""

from __future__ import annotations

import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Callable, Any
from enum import Enum
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class ParamType(Enum):
    """Parameter type for optimization."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


@dataclass
class ParamSpec:
    """Specification for a parameter to optimize."""
    name: str
    param_type: ParamType
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    choices: Optional[List[Any]] = None
    step: Optional[float] = None  # For discrete params

    def sample(self) -> Any:
        """Sample a random value for this parameter."""
        if self.param_type == ParamType.CONTINUOUS:
            return random.uniform(self.min_val, self.max_val)
        elif self.param_type == ParamType.DISCRETE:
            if self.step:
                steps = int((self.max_val - self.min_val) / self.step)
                return self.min_val + random.randint(0, steps) * self.step
            return random.randint(int(self.min_val), int(self.max_val))
        elif self.param_type == ParamType.CATEGORICAL:
            return random.choice(self.choices)
        raise ValueError(f"Unknown param type: {self.param_type}")

    def mutate(self, value: Any, mutation_strength: float = 0.2) -> Any:
        """Mutate a parameter value."""
        if self.param_type == ParamType.CONTINUOUS:
            range_size = self.max_val - self.min_val
            delta = random.gauss(0, range_size * mutation_strength)
            new_val = value + delta
            return max(self.min_val, min(self.max_val, new_val))
        elif self.param_type == ParamType.DISCRETE:
            range_size = self.max_val - self.min_val
            delta = int(random.gauss(0, range_size * mutation_strength))
            new_val = value + delta
            if self.step:
                new_val = round(new_val / self.step) * self.step
            return max(self.min_val, min(self.max_val, new_val))
        elif self.param_type == ParamType.CATEGORICAL:
            if random.random() < mutation_strength:
                return random.choice(self.choices)
            return value
        return value


@dataclass
class Individual:
    """An individual in the genetic population (a strategy configuration)."""
    params: Dict[str, Any]
    fitness: Optional[float] = None
    sharpe: Optional[float] = None
    profit_factor: Optional[float] = None
    win_rate: Optional[float] = None
    total_trades: Optional[int] = None
    max_drawdown: Optional[float] = None
    generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'params': self.params,
            'fitness': self.fitness,
            'sharpe': self.sharpe,
            'profit_factor': self.profit_factor,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'max_drawdown': self.max_drawdown,
            'generation': self.generation,
        }


@dataclass
class Population:
    """A population of individuals."""
    individuals: List[Individual] = field(default_factory=list)
    generation: int = 0
    best_fitness_history: List[float] = field(default_factory=list)
    avg_fitness_history: List[float] = field(default_factory=list)

    def add(self, individual: Individual):
        """Add an individual to the population."""
        self.individuals.append(individual)

    def get_best(self, n: int = 1) -> List[Individual]:
        """Get the n best individuals by fitness."""
        sorted_pop = sorted(
            [i for i in self.individuals if i.fitness is not None],
            key=lambda x: x.fitness,
            reverse=True
        )
        return sorted_pop[:n]

    def get_stats(self) -> Dict[str, float]:
        """Get population statistics."""
        fitnesses = [i.fitness for i in self.individuals if i.fitness is not None]
        if not fitnesses:
            return {'best': 0, 'avg': 0, 'worst': 0, 'std': 0}
        return {
            'best': max(fitnesses),
            'avg': np.mean(fitnesses),
            'worst': min(fitnesses),
            'std': np.std(fitnesses),
        }


class GeneticOptimizer:
    """
    Genetic algorithm optimizer for trading strategy parameters.

    Uses tournament selection, crossover, and mutation to evolve
    strategy configurations that maximize a fitness function
    (typically Sharpe ratio or risk-adjusted returns).
    """

    def __init__(
        self,
        param_specs: List[ParamSpec],
        fitness_function: Callable[[Dict[str, Any]], float],
        population_size: int = 50,
        elite_count: int = 5,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.2,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        min_trades: int = 30,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the genetic optimizer.

        Args:
            param_specs: List of parameter specifications
            fitness_function: Function that evaluates params and returns fitness
            population_size: Number of individuals per generation
            elite_count: Number of best individuals to preserve
            mutation_rate: Probability of mutating each parameter
            mutation_strength: Magnitude of mutations (0-1)
            crossover_rate: Probability of crossover vs cloning
            tournament_size: Size of tournament for selection
            min_trades: Minimum trades required for valid fitness
            random_seed: Random seed for reproducibility
        """
        self.param_specs = {spec.name: spec for spec in param_specs}
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.min_trades = min_trades

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.population: Optional[Population] = None
        self.best_individual: Optional[Individual] = None

        logger.info(
            f"GeneticOptimizer initialized with {len(param_specs)} parameters, "
            f"population_size={population_size}"
        )

    def _create_random_individual(self, generation: int = 0) -> Individual:
        """Create a random individual."""
        params = {
            name: spec.sample()
            for name, spec in self.param_specs.items()
        }
        return Individual(params=params, generation=generation)

    def _initialize_population(self) -> Population:
        """Initialize a random population."""
        population = Population(generation=0)
        for _ in range(self.population_size):
            individual = self._create_random_individual(generation=0)
            population.add(individual)
        return population

    def _evaluate_individual(self, individual: Individual) -> Individual:
        """Evaluate an individual's fitness."""
        try:
            result = self.fitness_function(individual.params)

            # Handle both simple fitness and detailed results
            if isinstance(result, dict):
                individual.fitness = result.get('fitness', result.get('sharpe', 0))
                individual.sharpe = result.get('sharpe')
                individual.profit_factor = result.get('profit_factor')
                individual.win_rate = result.get('win_rate')
                individual.total_trades = result.get('total_trades')
                individual.max_drawdown = result.get('max_drawdown')

                # Penalize insufficient trades
                if individual.total_trades and individual.total_trades < self.min_trades:
                    individual.fitness *= (individual.total_trades / self.min_trades)
            else:
                individual.fitness = float(result)

        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            individual.fitness = -999  # Heavily penalize failures

        return individual

    def _evaluate_population(
        self,
        population: Population,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> Population:
        """Evaluate all individuals in the population."""
        if parallel:
            # Parallel evaluation (careful with shared state)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._evaluate_individual, ind): i
                    for i, ind in enumerate(population.individuals)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        population.individuals[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Parallel evaluation failed: {e}")
        else:
            # Sequential evaluation
            for i, individual in enumerate(population.individuals):
                population.individuals[i] = self._evaluate_individual(individual)

        return population

    def _tournament_select(self, population: Population) -> Individual:
        """Select an individual via tournament selection."""
        candidates = random.sample(
            [i for i in population.individuals if i.fitness is not None],
            min(self.tournament_size, len(population.individuals))
        )
        return max(candidates, key=lambda x: x.fitness or -999)

    def _crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        generation: int,
    ) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        child1_params = {}
        child2_params = {}

        for name in self.param_specs:
            if random.random() < 0.5:
                child1_params[name] = parent1.params[name]
                child2_params[name] = parent2.params[name]
            else:
                child1_params[name] = parent2.params[name]
                child2_params[name] = parent1.params[name]

        return (
            Individual(params=child1_params, generation=generation),
            Individual(params=child2_params, generation=generation),
        )

    def _mutate(self, individual: Individual) -> Individual:
        """Mutate an individual's parameters."""
        mutated_params = individual.params.copy()

        for name, spec in self.param_specs.items():
            if random.random() < self.mutation_rate:
                mutated_params[name] = spec.mutate(
                    mutated_params[name],
                    self.mutation_strength
                )

        individual.params = mutated_params
        individual.fitness = None  # Reset fitness after mutation
        return individual

    def _create_next_generation(self, population: Population) -> Population:
        """Create the next generation via selection, crossover, and mutation."""
        next_gen = Population(
            generation=population.generation + 1,
            best_fitness_history=population.best_fitness_history.copy(),
            avg_fitness_history=population.avg_fitness_history.copy(),
        )

        # Elitism: preserve best individuals
        elites = population.get_best(self.elite_count)
        for elite in elites:
            next_gen.add(Individual(
                params=elite.params.copy(),
                fitness=elite.fitness,
                sharpe=elite.sharpe,
                profit_factor=elite.profit_factor,
                win_rate=elite.win_rate,
                total_trades=elite.total_trades,
                max_drawdown=elite.max_drawdown,
                generation=next_gen.generation,
            ))

        # Fill rest of population
        while len(next_gen.individuals) < self.population_size:
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)

            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(
                    parent1, parent2, next_gen.generation
                )
            else:
                child1 = Individual(
                    params=parent1.params.copy(),
                    generation=next_gen.generation
                )
                child2 = Individual(
                    params=parent2.params.copy(),
                    generation=next_gen.generation
                )

            # Mutate children
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            next_gen.add(child1)
            if len(next_gen.individuals) < self.population_size:
                next_gen.add(child2)

        return next_gen

    def evolve(
        self,
        generations: int = 50,
        early_stop_generations: int = 10,
        target_fitness: Optional[float] = None,
        parallel: bool = False,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run the genetic algorithm evolution.

        Args:
            generations: Maximum number of generations
            early_stop_generations: Stop if no improvement for this many gens
            target_fitness: Stop if this fitness is achieved
            parallel: Use parallel evaluation
            verbose: Print progress

        Returns:
            Tuple of (best_params, best_fitness)
        """
        logger.info(f"Starting evolution for {generations} generations")

        # Initialize population
        self.population = self._initialize_population()
        self.population = self._evaluate_population(self.population, parallel)

        best_fitness = -float('inf')
        generations_without_improvement = 0

        for gen in range(generations):
            # Track statistics
            stats = self.population.get_stats()
            self.population.best_fitness_history.append(stats['best'])
            self.population.avg_fitness_history.append(stats['avg'])

            # Update best individual
            current_best = self.population.get_best(1)
            if current_best and current_best[0].fitness > best_fitness:
                best_fitness = current_best[0].fitness
                self.best_individual = current_best[0]
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if verbose:
                logger.info(
                    f"Gen {gen}: best={stats['best']:.4f}, "
                    f"avg={stats['avg']:.4f}, std={stats['std']:.4f}"
                )

            # Check stopping conditions
            if target_fitness and best_fitness >= target_fitness:
                logger.info(f"Target fitness {target_fitness} achieved!")
                break

            if generations_without_improvement >= early_stop_generations:
                logger.info(f"Early stopping: no improvement for {early_stop_generations} generations")
                break

            # Create next generation
            self.population = self._create_next_generation(self.population)
            self.population = self._evaluate_population(self.population, parallel)

        if self.best_individual:
            logger.info(
                f"Evolution complete. Best fitness: {self.best_individual.fitness:.4f}"
            )
            return self.best_individual.params, self.best_individual.fitness

        return {}, 0.0

    def get_convergence_history(self) -> Dict[str, List[float]]:
        """Get the fitness history over generations."""
        if self.population:
            return {
                'best': self.population.best_fitness_history,
                'avg': self.population.avg_fitness_history,
            }
        return {'best': [], 'avg': []}


# Convenience functions
_optimizer: Optional[GeneticOptimizer] = None


def get_optimizer() -> Optional[GeneticOptimizer]:
    """Get the global optimizer instance."""
    return _optimizer


def evolve_strategy(
    param_ranges: Dict[str, Tuple[float, float]],
    fitness_function: Callable[[Dict[str, Any]], float],
    generations: int = 50,
    population_size: int = 50,
    **kwargs,
) -> Tuple[Dict[str, Any], float]:
    """
    Convenience function to evolve strategy parameters.

    Args:
        param_ranges: Dict of param_name -> (min, max) tuples
        fitness_function: Function that takes params dict and returns fitness
        generations: Number of generations to evolve
        population_size: Size of population
        **kwargs: Additional GeneticOptimizer arguments

    Returns:
        Tuple of (best_params, best_fitness)
    """
    global _optimizer

    # Convert param ranges to ParamSpec objects
    param_specs = []
    for name, (min_val, max_val) in param_ranges.items():
        if isinstance(min_val, int) and isinstance(max_val, int):
            param_specs.append(ParamSpec(
                name=name,
                param_type=ParamType.DISCRETE,
                min_val=min_val,
                max_val=max_val,
            ))
        else:
            param_specs.append(ParamSpec(
                name=name,
                param_type=ParamType.CONTINUOUS,
                min_val=min_val,
                max_val=max_val,
            ))

    _optimizer = GeneticOptimizer(
        param_specs=param_specs,
        fitness_function=fitness_function,
        population_size=population_size,
        **kwargs,
    )

    return _optimizer.evolve(generations=generations)
