"""
Backtest module for Kobe Trading System.

Provides:
- Event-driven backtesting engine
- Walk-forward analysis
- High-performance vectorized backtester (vectorbt-inspired)
- Data versioning and reproducibility tracking
- Visualization (Plotly/Matplotlib)
- Monte Carlo simulation for confidence intervals
"""
from __future__ import annotations

from .engine import (
    Backtester,
    BacktestConfig,
    CommissionConfig,
    Trade,
    Position,
)
from .walk_forward import (
    WFSplit,
    generate_splits,
    run_walk_forward,
    summarize_results,
    train_start_to_date,
)
from .vectorized import (
    VectorizedBacktester,
    VectorConfig,
    VectorResults,
    run_parallel_backtests,
)
from .reproducibility import (
    DataVersioner,
    DataVersion,
    ExperimentTracker,
    ExperimentRun,
    ExperimentManifest,
    compute_file_checksum,
    compute_directory_checksum,
    create_reproducibility_report,
)
from .visualization import (
    BacktestPlotter,
    PlotConfig,
)
from .monte_carlo import (
    MonteCarloSimulator,
    MonteCarloConfig,
    MonteCarloResults,
    run_monte_carlo_analysis,
)

__all__ = [
    # Engine exports
    "Backtester",
    "BacktestConfig",
    "CommissionConfig",
    "Trade",
    "Position",
    # Walk-forward exports
    "WFSplit",
    "generate_splits",
    "run_walk_forward",
    "summarize_results",
    "train_start_to_date",
    # Vectorized backtester exports
    "VectorizedBacktester",
    "VectorConfig",
    "VectorResults",
    "run_parallel_backtests",
    # Reproducibility exports
    "DataVersioner",
    "DataVersion",
    "ExperimentTracker",
    "ExperimentRun",
    "ExperimentManifest",
    "compute_file_checksum",
    "compute_directory_checksum",
    "create_reproducibility_report",
    # Visualization exports
    "BacktestPlotter",
    "PlotConfig",
    # Monte Carlo exports
    "MonteCarloSimulator",
    "MonteCarloConfig",
    "MonteCarloResults",
    "run_monte_carlo_analysis",
]
