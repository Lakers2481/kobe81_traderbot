"""
Basic MLflow experiment tracking (graceful degradation if not installed).

This module provides a simple wrapper around MLflow for experiment tracking.
If MLflow is not installed, all functions become no-ops.

Usage:
    from ml.experiment_tracking import init_experiment, log_scan_run, log_backtest_run

    init_experiment("kobe_trading")
    log_scan_run(
        params={'universe': 'optionable_liquid_900', 'cap': 50},
        metrics={'signals_count': 3, 'top3_count': 3}
    )
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.debug("MLflow not installed - experiment tracking disabled")


def init_experiment(name: str = "kobe_trading") -> None:
    """
    Initialize MLflow experiment.

    Args:
        name: Experiment name (default: kobe_trading)
    """
    if not MLFLOW_AVAILABLE:
        return
    try:
        mlflow.set_experiment(name)
        logger.debug(f"MLflow experiment set: {name}")
    except Exception as e:
        logger.warning(f"MLflow init failed: {e}")


def log_scan_run(params: Dict[str, Any], metrics: Dict[str, Any]) -> Optional[str]:
    """
    Log a scan run to MLflow.

    Args:
        params: Run parameters (universe, cap, mode, etc.)
        metrics: Run metrics (signals_count, top3_count, etc.)

    Returns:
        Run ID if logged, None otherwise
    """
    if not MLFLOW_AVAILABLE:
        return None
    try:
        with mlflow.start_run() as run:
            # Log parameters (convert to strings)
            for k, v in params.items():
                mlflow.log_param(k, str(v) if v is not None else "")
            # Log metrics (must be numeric)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            return run.info.run_id
    except Exception as e:
        logger.warning(f"MLflow log_scan_run failed: {e}")
        return None


def log_backtest_run(
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    artifacts: Optional[List[str]] = None
) -> Optional[str]:
    """
    Log a backtest run to MLflow with optional artifacts.

    Args:
        params: Run parameters (strategy, universe, period, etc.)
        metrics: Run metrics (win_rate, profit_factor, sharpe, etc.)
        artifacts: List of file paths to log as artifacts

    Returns:
        Run ID if logged, None otherwise
    """
    if not MLFLOW_AVAILABLE:
        return None
    try:
        with mlflow.start_run() as run:
            # Log parameters
            for k, v in params.items():
                mlflow.log_param(k, str(v) if v is not None else "")
            # Log metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            # Log artifacts
            if artifacts:
                for path in artifacts:
                    if os.path.exists(path):
                        mlflow.log_artifact(path)
            return run.info.run_id
    except Exception as e:
        logger.warning(f"MLflow log_backtest_run failed: {e}")
        return None


def log_trade_decision(
    symbol: str,
    side: str,
    strategy: str,
    confidence: float,
    approved: bool,
    params: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a single trade decision as a metric/param set.

    This is lightweight - doesn't create a new run, just logs to active run.

    Args:
        symbol: Stock symbol
        side: Trade side (long/short)
        strategy: Strategy name
        confidence: Cognitive confidence score
        approved: Whether trade was approved
        params: Additional parameters
    """
    if not MLFLOW_AVAILABLE:
        return
    try:
        mlflow.log_params({
            'symbol': symbol,
            'side': side,
            'strategy': strategy,
        })
        mlflow.log_metrics({
            'confidence': confidence,
            'approved': 1 if approved else 0,
        })
        if params:
            for k, v in params.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
    except Exception:
        # Silently fail for individual trade logging
        pass


def is_available() -> bool:
    """Check if MLflow is available."""
    return MLFLOW_AVAILABLE
