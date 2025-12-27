"""
Adaptive Strategy Evolution Module
===================================

Genetic algorithms and evolutionary optimization for trading strategies.
Automatically generates, mutates, and selects trading rules that adapt
to changing market conditions.

Components:
- GeneticOptimizer: Evolves strategy parameters via genetic algorithms
- StrategyMutator: Creates variations of existing strategies
- RuleGenerator: Generates new trading rules from patterns
- PromotionGate: Only promotes strategies passing walk-forward tests

Usage:
    from evolution import GeneticOptimizer, evolve_strategy

    optimizer = GeneticOptimizer(
        strategy_class=DonchianBreakoutStrategy,
        param_ranges={'lookback': (10, 50), 'atr_mult': (1.0, 3.0)}
    )

    best_params, fitness = optimizer.evolve(
        data=historical_data,
        generations=50,
        population_size=100
    )
"""

from .genetic_optimizer import (
    GeneticOptimizer,
    Individual,
    Population,
    evolve_strategy,
    get_optimizer,
)

from .strategy_mutator import (
    StrategyMutator,
    MutationType,
    mutate_strategy,
    crossover_strategies,
)

from .rule_generator import (
    RuleGenerator,
    TradingRule,
    RuleCondition,
    generate_rules,
    discover_patterns,
)

from .promotion_gate import (
    PromotionGate,
    PromotionResult,
    PromotionCriteria,
    check_promotion,
)

__all__ = [
    # Genetic Optimizer
    'GeneticOptimizer',
    'Individual',
    'Population',
    'evolve_strategy',
    'get_optimizer',
    # Strategy Mutator
    'StrategyMutator',
    'MutationType',
    'mutate_strategy',
    'crossover_strategies',
    # Rule Generator
    'RuleGenerator',
    'TradingRule',
    'RuleCondition',
    'generate_rules',
    'discover_patterns',
    # Promotion Gate
    'PromotionGate',
    'PromotionResult',
    'PromotionCriteria',
    'check_promotion',
]
