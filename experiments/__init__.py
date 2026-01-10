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

# MLflow integration (optional - requires mlflow)
try:
    from .mlflow_adapter import (
        MLflowAdapter,
        get_mlflow_adapter,
        sync_to_mlflow,
        query_best,
        log_backtest_to_mlflow,
        HAS_MLFLOW,
    )
except ImportError:
    HAS_MLFLOW = False

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
    # MLflow (optional)
    'MLflowAdapter',
    'get_mlflow_adapter',
    'sync_to_mlflow',
    'query_best',
    'log_backtest_to_mlflow',
    'HAS_MLFLOW',
]
