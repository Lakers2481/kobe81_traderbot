"""
Kobe Trading System - Experiment Registry.

Tracks all backtesting experiments for reproducibility:
- Dataset ID and version
- Strategy parameters
- Random seeds
- Results hash for verification
- Git commit tracking

Every experiment can be reproduced exactly by loading its config.
"""

from .registry import (
    ExperimentRegistry,
    ExperimentConfig,
    ExperimentResults,
    Experiment,
    ExperimentStatus,
    get_registry,
    register_experiment,
    record_experiment_results,
    list_experiments,
)

__all__ = [
    'ExperimentRegistry',
    'ExperimentConfig',
    'ExperimentResults',
    'Experiment',
    'ExperimentStatus',
    'get_registry',
    'register_experiment',
    'record_experiment_results',
    'list_experiments',
]
