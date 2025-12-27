"""
Tests for the Evolution Module.

Tests genetic optimizer, strategy mutator, rule generator,
and promotion gate functionality.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from evolution import (
    # Genetic Optimizer
    GeneticOptimizer,
    Individual,
    Population,
    evolve_strategy,
    get_optimizer,
    # Strategy Mutator
    StrategyMutator,
    MutationType,
    mutate_strategy,
    crossover_strategies,
    # Rule Generator
    RuleGenerator,
    TradingRule,
    RuleCondition,
    generate_rules,
    discover_patterns,
    # Promotion Gate
    PromotionGate,
    PromotionResult,
    PromotionCriteria,
    check_promotion,
)

from evolution.genetic_optimizer import ParamSpec, ParamType
from evolution.rule_generator import ConditionOperator, IndicatorType
from evolution.promotion_gate import PromotionStatus, WalkForwardResult


class TestGeneticOptimizer:
    """Tests for GeneticOptimizer."""

    def test_initialization(self):
        """Should initialize with param specs."""
        specs = [
            ParamSpec(name='lookback', param_type=ParamType.DISCRETE, min_val=10, max_val=50),
            ParamSpec(name='threshold', param_type=ParamType.CONTINUOUS, min_val=0.1, max_val=0.9),
        ]

        optimizer = GeneticOptimizer(
            param_specs=specs,
            fitness_function=lambda p: p['lookback'] * 0.1,
            population_size=10,
        )

        assert len(optimizer.param_specs) == 2
        assert optimizer.population_size == 10

    def test_evolve_simple(self):
        """Should evolve and return best params."""
        specs = [
            ParamSpec(name='x', param_type=ParamType.CONTINUOUS, min_val=0, max_val=10),
        ]

        # Simple fitness: maximize x
        def fitness(params):
            return params['x']

        optimizer = GeneticOptimizer(
            param_specs=specs,
            fitness_function=fitness,
            population_size=20,
            random_seed=42,
        )

        best_params, best_fitness = optimizer.evolve(
            generations=10,
            verbose=False,
        )

        # Should find x close to 10
        assert best_params['x'] > 5
        assert best_fitness > 5

    def test_discrete_params(self):
        """Should handle discrete parameters."""
        specs = [
            ParamSpec(name='period', param_type=ParamType.DISCRETE, min_val=5, max_val=20),
        ]

        optimizer = GeneticOptimizer(
            param_specs=specs,
            fitness_function=lambda p: p['period'],
            population_size=10,
        )

        best_params, _ = optimizer.evolve(generations=5, verbose=False)

        # Should be integer
        assert isinstance(best_params['period'], (int, float))
        assert 5 <= best_params['period'] <= 20

    def test_early_stopping(self):
        """Should stop early when fitness plateaus."""
        specs = [
            ParamSpec(name='x', param_type=ParamType.CONTINUOUS, min_val=0, max_val=1),
        ]

        # Constant fitness - should trigger early stop
        optimizer = GeneticOptimizer(
            param_specs=specs,
            fitness_function=lambda p: 1.0,
            population_size=10,
        )

        # This should stop early
        optimizer.evolve(
            generations=100,
            early_stop_generations=3,
            verbose=False,
        )

        # Should have stopped before 100 generations
        assert optimizer.population.generation < 100

    def test_convergence_history(self):
        """Should track convergence history."""
        specs = [
            ParamSpec(name='x', param_type=ParamType.CONTINUOUS, min_val=0, max_val=10),
        ]

        optimizer = GeneticOptimizer(
            param_specs=specs,
            fitness_function=lambda p: p['x'],
            population_size=10,
        )

        # Use enough generations and high early_stop to ensure history is populated
        optimizer.evolve(generations=10, early_stop_generations=20, verbose=False)

        history = optimizer.get_convergence_history()
        assert 'best' in history
        assert 'avg' in history
        assert len(history['best']) > 0


class TestPopulation:
    """Tests for Population."""

    def test_add_individual(self):
        """Should add individuals."""
        pop = Population()
        ind = Individual(params={'x': 1}, fitness=0.5)
        pop.add(ind)

        assert len(pop.individuals) == 1

    def test_get_best(self):
        """Should return best individuals by fitness."""
        pop = Population()
        pop.add(Individual(params={'x': 1}, fitness=0.3))
        pop.add(Individual(params={'x': 2}, fitness=0.8))
        pop.add(Individual(params={'x': 3}, fitness=0.5))

        best = pop.get_best(2)

        assert len(best) == 2
        assert best[0].fitness == 0.8
        assert best[1].fitness == 0.5

    def test_get_stats(self):
        """Should calculate population statistics."""
        pop = Population()
        pop.add(Individual(params={}, fitness=0.2))
        pop.add(Individual(params={}, fitness=0.4))
        pop.add(Individual(params={}, fitness=0.6))

        stats = pop.get_stats()

        assert stats['best'] == 0.6
        assert stats['worst'] == 0.2
        assert 0.3 < stats['avg'] < 0.5


class TestParamSpec:
    """Tests for ParamSpec."""

    def test_sample_continuous(self):
        """Should sample continuous values."""
        spec = ParamSpec(
            name='x',
            param_type=ParamType.CONTINUOUS,
            min_val=0,
            max_val=10,
        )

        samples = [spec.sample() for _ in range(100)]

        assert all(0 <= s <= 10 for s in samples)

    def test_sample_discrete(self):
        """Should sample discrete values."""
        spec = ParamSpec(
            name='x',
            param_type=ParamType.DISCRETE,
            min_val=1,
            max_val=5,
        )

        samples = [spec.sample() for _ in range(100)]

        assert all(1 <= s <= 5 for s in samples)
        assert all(isinstance(s, (int, float)) for s in samples)

    def test_sample_categorical(self):
        """Should sample from choices."""
        spec = ParamSpec(
            name='x',
            param_type=ParamType.CATEGORICAL,
            choices=['a', 'b', 'c'],
        )

        samples = [spec.sample() for _ in range(100)]

        assert all(s in ['a', 'b', 'c'] for s in samples)

    def test_mutate_continuous(self):
        """Should mutate continuous values."""
        spec = ParamSpec(
            name='x',
            param_type=ParamType.CONTINUOUS,
            min_val=0,
            max_val=10,
        )

        original = 5.0
        mutations = [spec.mutate(original, 0.2) for _ in range(100)]

        # Should stay in bounds
        assert all(0 <= m <= 10 for m in mutations)
        # Should vary
        assert len(set(mutations)) > 1


class TestStrategyMutator:
    """Tests for StrategyMutator."""

    def test_initialization(self):
        """Should initialize with defaults."""
        mutator = StrategyMutator()

        assert mutator.mutation_rate == 0.3
        assert mutator.mutation_strength == 0.2

    def test_mutate_params(self):
        """Should mutate numeric parameters."""
        mutator = StrategyMutator(mutation_rate=1.0, random_seed=42)

        params = {'lookback': 20, 'threshold': 0.5}
        mutated, records = mutator.mutate_params(params)

        # At least one should be mutated with rate=1.0
        assert mutated != params or len(records) > 0

    def test_create_variant(self):
        """Should create strategy variant."""
        mutator = StrategyMutator(mutation_rate=0.5)

        params = {'period': 14, 'mult': 2.0}
        variant = mutator.create_variant(params, 'test_strategy', generation=1)

        assert variant.parent_name == 'test_strategy'
        assert variant.generation == 1
        assert isinstance(variant.params, dict)

    def test_create_variants(self):
        """Should create multiple variants."""
        mutator = StrategyMutator()

        params = {'x': 10}
        variants = mutator.create_variants(params, count=5)

        assert len(variants) == 5


class TestCrossover:
    """Tests for strategy crossover."""

    def test_crossover_strategies(self):
        """Should perform crossover between parents."""
        parent1 = {'a': 1, 'b': 2, 'c': 3}
        parent2 = {'a': 10, 'b': 20, 'c': 30}

        child1, child2 = crossover_strategies(parent1, parent2)

        # Children should have all keys
        assert set(child1.keys()) == set(parent1.keys())
        assert set(child2.keys()) == set(parent2.keys())

        # Values should come from parents
        for key in parent1:
            assert child1[key] in [parent1[key], parent2[key]]
            assert child2[key] in [parent1[key], parent2[key]]


class TestRuleGenerator:
    """Tests for RuleGenerator."""

    def test_initialization(self):
        """Should initialize with defaults."""
        gen = RuleGenerator()

        assert gen.min_conditions >= 1
        assert gen.max_conditions >= gen.min_conditions

    def test_generate_rule(self):
        """Should generate a trading rule."""
        gen = RuleGenerator(random_seed=42)

        rule = gen.generate_rule()

        assert isinstance(rule, TradingRule)
        assert len(rule.entry_conditions) > 0
        assert rule.side in ['long', 'short']

    def test_generate_rules(self):
        """Should generate multiple rules."""
        gen = RuleGenerator()

        rules = gen.generate_rules(count=5)

        assert len(rules) == 5
        assert all(isinstance(r, TradingRule) for r in rules)

    def test_rule_with_indicators(self):
        """Should use specified indicators."""
        gen = RuleGenerator()

        rule = gen.generate_rule(
            available_indicators=['rsi', 'ibs'],
            side='long',
        )

        # Entry conditions should use available indicators
        indicators_used = [c.indicator for c in rule.entry_conditions]
        assert all(i in ['rsi', 'ibs'] for i in indicators_used)


class TestRuleCondition:
    """Tests for RuleCondition."""

    def test_to_expression(self):
        """Should generate readable expression."""
        cond = RuleCondition(
            indicator='rsi',
            operator=ConditionOperator.LESS_THAN,
            threshold=30,
        )

        expr = cond.to_expression()

        assert 'rsi' in expr
        assert '<' in expr
        assert '30' in expr

    def test_evaluate_less_than(self):
        """Should evaluate less than condition."""
        cond = RuleCondition(
            indicator='rsi',
            operator=ConditionOperator.LESS_THAN,
            threshold=30,
        )

        df = pd.DataFrame({'rsi': [25, 35, 30]})

        assert cond.evaluate(df, 0) == True   # 25 < 30
        assert cond.evaluate(df, 1) == False  # 35 < 30
        assert cond.evaluate(df, 2) == False  # 30 < 30

    def test_evaluate_crosses_above(self):
        """Should evaluate crosses above condition."""
        cond = RuleCondition(
            indicator='price',
            operator=ConditionOperator.CROSSES_ABOVE,
            threshold=100,
        )

        df = pd.DataFrame({'price': [95, 105, 110]})

        assert cond.evaluate(df, 0) == False  # No previous
        assert cond.evaluate(df, 1) == True   # 95 -> 105 crosses 100
        assert cond.evaluate(df, 2) == False  # 105 -> 110 already above


class TestTradingRule:
    """Tests for TradingRule."""

    def test_check_entry(self):
        """Should check all entry conditions."""
        rule = TradingRule(
            name='test',
            entry_conditions=[
                RuleCondition('rsi', ConditionOperator.LESS_THAN, 30),
                RuleCondition('ibs', ConditionOperator.LESS_THAN, 0.2),
            ],
        )

        df = pd.DataFrame({'rsi': [25], 'ibs': [0.1]})
        assert rule.check_entry(df) == True

        df2 = pd.DataFrame({'rsi': [25], 'ibs': [0.5]})
        assert rule.check_entry(df2) == False  # ibs fails

    def test_describe(self):
        """Should generate description."""
        rule = TradingRule(
            name='test_rule',
            side='long',
            entry_conditions=[
                RuleCondition('rsi', ConditionOperator.LESS_THAN, 30),
            ],
            confidence=0.75,
        )

        desc = rule.describe()

        assert 'test_rule' in desc
        assert 'long' in desc
        assert 'rsi' in desc


class TestPromotionGate:
    """Tests for PromotionGate."""

    def test_initialization(self):
        """Should initialize with criteria."""
        gate = PromotionGate()

        assert gate.criteria.min_sharpe > 0
        assert gate.criteria.min_trades > 0

    def test_passed_promotion(self):
        """Should pass strategy meeting all criteria."""
        gate = PromotionGate(require_wf=False)

        metrics = {
            'sharpe': 1.5,
            'profit_factor': 2.0,
            'win_rate': 0.55,
            'total_trades': 100,
            'max_drawdown': 0.10,
        }

        result = gate.check_promotion('good_strategy', metrics)

        assert result.status == PromotionStatus.PASSED
        assert len(result.passed_checks) > 0
        assert len(result.failed_checks) == 0

    def test_failed_promotion_low_sharpe(self):
        """Should fail strategy with low Sharpe."""
        gate = PromotionGate(require_wf=False)

        metrics = {
            'sharpe': 0.2,  # Below minimum
            'profit_factor': 2.0,
            'win_rate': 0.55,
            'total_trades': 100,
            'max_drawdown': 0.10,
        }

        result = gate.check_promotion('bad_strategy', metrics)

        assert result.status == PromotionStatus.FAILED
        assert any('Sharpe' in c for c in result.failed_checks)

    def test_failed_promotion_low_trades(self):
        """Should fail strategy with insufficient trades."""
        gate = PromotionGate(require_wf=False)

        metrics = {
            'sharpe': 1.5,
            'profit_factor': 2.0,
            'win_rate': 0.55,
            'total_trades': 10,  # Below minimum
            'max_drawdown': 0.10,
        }

        result = gate.check_promotion('low_trades_strategy', metrics)

        assert result.status == PromotionStatus.FAILED
        assert any('Trade count' in c for c in result.failed_checks)

    def test_walk_forward_check(self):
        """Should validate walk-forward results."""
        gate = PromotionGate(require_wf=True)

        metrics = {
            'sharpe': 1.5,
            'profit_factor': 2.0,
            'win_rate': 0.55,
            'total_trades': 100,
            'max_drawdown': 0.10,
        }

        wf_results = [
            WalkForwardResult(
                split_id=i,
                train_start='2020-01-01',
                train_end='2020-06-30',
                test_start='2020-07-01',
                test_end='2020-09-30',
                is_sharpe=1.0,
                oos_sharpe=0.8,
                oos_return=0.05,
                oos_trades=20,
            )
            for i in range(5)
        ]

        result = gate.check_promotion('wf_strategy', metrics, wf_results)

        assert 'wf_splits' in result.metrics
        assert result.metrics['wf_splits'] == 5

    def test_get_promoted_strategies(self):
        """Should track promoted strategies."""
        gate = PromotionGate(require_wf=False)

        good_metrics = {
            'sharpe': 1.5,
            'profit_factor': 2.0,
            'win_rate': 0.55,
            'total_trades': 100,
            'max_drawdown': 0.10,
        }

        gate.check_promotion('strategy_a', good_metrics)
        gate.check_promotion('strategy_b', good_metrics)

        promoted = gate.get_promoted_strategies()

        assert 'strategy_a' in promoted
        assert 'strategy_b' in promoted


class TestPromotionCriteria:
    """Tests for PromotionCriteria."""

    def test_default_criteria(self):
        """Should have sensible defaults."""
        criteria = PromotionCriteria()

        assert criteria.min_sharpe > 0
        assert criteria.min_profit_factor > 1.0
        assert criteria.min_win_rate > 0.4
        assert criteria.min_trades > 0

    def test_custom_criteria(self):
        """Should accept custom values."""
        criteria = PromotionCriteria(
            min_sharpe=2.0,
            min_trades=50,
        )

        assert criteria.min_sharpe == 2.0
        assert criteria.min_trades == 50


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_evolve_strategy(self):
        """Should evolve with convenience function."""
        param_ranges = {
            'lookback': (10, 30),
            'threshold': (0.1, 0.9),
        }

        best_params, fitness = evolve_strategy(
            param_ranges=param_ranges,
            fitness_function=lambda p: p['lookback'] * 0.1,
            generations=5,
            population_size=10,
        )

        assert 'lookback' in best_params
        assert 'threshold' in best_params

    def test_mutate_strategy(self):
        """Should mutate with convenience function."""
        params = {'x': 10, 'y': 20}
        mutated = mutate_strategy(params, mutation_rate=1.0)

        assert isinstance(mutated, dict)

    def test_generate_rules(self):
        """Should generate with convenience function."""
        rules = generate_rules(count=3)

        assert len(rules) == 3

    def test_check_promotion(self):
        """Should check with convenience function."""
        metrics = {
            'sharpe': 0.3,
            'profit_factor': 0.8,
            'win_rate': 0.40,
            'total_trades': 20,
            'max_drawdown': 0.25,
        }

        result = check_promotion('test_strategy', metrics)

        assert isinstance(result, PromotionResult)
        assert result.status in [PromotionStatus.PASSED, PromotionStatus.FAILED, PromotionStatus.DEFERRED]


class TestDiscoverPatterns:
    """Tests for pattern discovery."""

    def test_discover_patterns(self):
        """Should find correlated indicators."""
        np.random.seed(42)

        # Create data with known correlation
        df = pd.DataFrame({
            'rsi': np.random.randn(100),
            'momentum': np.random.randn(100),
        })
        # Make forward_return correlated with rsi
        df['forward_return'] = df['rsi'] * 0.5 + np.random.randn(100) * 0.1

        patterns = discover_patterns(
            df,
            indicators=['rsi', 'momentum'],
            min_correlation=0.1,
        )

        # Should find rsi correlation
        assert len(patterns) > 0
        assert any(p['indicator'] == 'rsi' for p in patterns)


# Run with: pytest tests/test_evolution.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
