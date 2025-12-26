"""
Kobe Trading System - Bayesian Hyperparameter Optimization Module.

Provides intelligent hyperparameter optimization using Optuna:
- TPE (Tree-structured Parzen Estimator) sampler for efficient search
- Single-objective optimization (Sharpe, profit factor, win rate, etc.)
- Multi-objective Pareto optimization (Sharpe + low drawdown)
- Early stopping/pruning for bad trials
- Parameter importance analysis
"""

from .bayesian_hyperopt import (
    # Core classes
    StrategyOptimizer,
    MultiObjectiveOptimizer,

    # Data classes
    ParameterSpace,
    OptimizationResult,

    # Pre-defined parameter spaces
    CONNORS_RSI2_PARAM_SPACE,
    IBS_PARAM_SPACE,
    QUICK_PARAM_SPACE,

    # Convenience functions
    quick_optimize,
    multi_objective_optimize,

    # Constants
    OPTUNA_AVAILABLE,
)

__all__ = [
    # Core classes
    'StrategyOptimizer',
    'MultiObjectiveOptimizer',

    # Data classes
    'ParameterSpace',
    'OptimizationResult',

    # Pre-defined parameter spaces
    'CONNORS_RSI2_PARAM_SPACE',
    'IBS_PARAM_SPACE',
    'QUICK_PARAM_SPACE',

    # Convenience functions
    'quick_optimize',
    'multi_objective_optimize',

    # Constants
    'OPTUNA_AVAILABLE',
]
