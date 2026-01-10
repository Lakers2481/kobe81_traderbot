"""
MLflow Adapter for Kobe Experiment Tracking.

Syncs experiments from registry.py to MLflow for rich querying and comparison.
Also provides direct MLflow logging for new experiments.

Usage:
    from experiments.mlflow_adapter import MLflowAdapter, get_mlflow_adapter

    adapter = get_mlflow_adapter()

    # Sync existing experiment to MLflow
    adapter.sync_experiment(exp_id)

    # Sync all experiments
    adapter.sync_all()

    # Query best experiments
    results = adapter.query_best_experiments(metric="sharpe_ratio", min_value=1.5)

    # Compare strategies
    comparison = adapter.compare_strategies()

Features:
- Automatic sync from registry.py to MLflow
- SQL-like querying across all experiments
- Time-series metrics for walk-forward analysis
- Artifact storage (equity curves, trade lists)
- Model versioning for ML components

Author: Claude Opus 4.5
Date: 2026-01-07
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    import mlflow
    from mlflow.entities import ViewType
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None
    ViewType = None

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_TRACKING_URI = "sqlite:///mlruns.db"
DEFAULT_EXPERIMENT_PREFIX = "kobe"


def setup_mlflow(
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None,
) -> bool:
    """
    Setup MLflow tracking.

    Args:
        tracking_uri: MLflow tracking server URI (default: sqlite:///mlruns.db)
        artifact_location: Where to store artifacts (default: mlartifacts/)

    Returns:
        True if setup successful
    """
    if not HAS_MLFLOW:
        logger.error("MLflow not installed: pip install mlflow")
        return False

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)

    logger.info(f"MLflow tracking URI: {uri}")
    return True


# =============================================================================
# MLFLOW ADAPTER
# =============================================================================

class MLflowAdapter:
    """
    Adapter to sync Kobe experiments to MLflow.

    Provides:
    - Automatic sync from registry.py
    - Direct logging for new experiments
    - Rich querying via MLflow search API
    - Strategy comparison across experiments
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_prefix: str = DEFAULT_EXPERIMENT_PREFIX,
    ):
        """
        Initialize MLflow adapter.

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_prefix: Prefix for experiment names
        """
        if not HAS_MLFLOW:
            raise ImportError("MLflow required: pip install mlflow")

        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI",
            DEFAULT_TRACKING_URI
        )
        self.experiment_prefix = experiment_prefix

        mlflow.set_tracking_uri(self.tracking_uri)
        self._client = mlflow.tracking.MlflowClient()

        logger.info(f"MLflowAdapter initialized: {self.tracking_uri}")

    # -------------------------------------------------------------------------
    # Experiment Management
    # -------------------------------------------------------------------------

    def get_or_create_experiment(self, name: str) -> str:
        """
        Get or create an MLflow experiment.

        Args:
            name: Experiment name

        Returns:
            Experiment ID
        """
        full_name = f"{self.experiment_prefix}_{name}"

        experiment = mlflow.get_experiment_by_name(full_name)
        if experiment:
            return experiment.experiment_id

        return mlflow.create_experiment(full_name)

    def _get_experiment_name(self, strategy: str) -> str:
        """Get MLflow experiment name from strategy."""
        return f"{self.experiment_prefix}_{strategy}"

    # -------------------------------------------------------------------------
    # Sync from Registry
    # -------------------------------------------------------------------------

    def sync_experiment(self, exp_id: str) -> Optional[str]:
        """
        Sync a Kobe experiment to MLflow.

        Args:
            exp_id: Kobe experiment ID from registry

        Returns:
            MLflow run ID (or None if failed)
        """
        try:
            from experiments.registry import get_registry
            registry = get_registry()
        except ImportError:
            logger.error("Could not import registry")
            return None

        exp = registry.get_experiment(exp_id)
        if not exp:
            logger.warning(f"Experiment not found: {exp_id}")
            return None

        # Set experiment
        exp_name = self._get_experiment_name(exp.config.strategy)
        mlflow.set_experiment(exp_name)

        # Create run
        with mlflow.start_run(run_name=exp.name) as run:
            # Log parameters
            mlflow.log_param("kobe_exp_id", exp.id)
            mlflow.log_param("strategy", exp.config.strategy)
            mlflow.log_param("dataset_id", exp.config.dataset_id)
            mlflow.log_param("seed", exp.config.seed)
            mlflow.log_param("initial_equity", exp.config.initial_equity)
            mlflow.log_param("risk_pct", exp.config.risk_pct)
            mlflow.log_param("config_hash", exp.config.config_hash)

            # Log strategy params (flatten dict)
            for key, value in exp.config.params.items():
                # MLflow params must be strings, numbers, or booleans
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(f"param_{key}", value)
                else:
                    mlflow.log_param(f"param_{key}", str(value))

            # Log dates
            if exp.config.start_date:
                mlflow.log_param("start_date", exp.config.start_date)
            if exp.config.end_date:
                mlflow.log_param("end_date", exp.config.end_date)

            # Log git info
            if exp.git_commit:
                mlflow.log_param("git_commit", exp.git_commit)
            if exp.git_branch:
                mlflow.log_param("git_branch", exp.git_branch)

            # Set tags
            for tag in exp.tags:
                mlflow.set_tag(tag, "true")

            mlflow.set_tag("status", exp.status.value)
            mlflow.set_tag("source", "kobe_registry")

            # Log metrics (if completed)
            if exp.results:
                mlflow.log_metric("total_trades", exp.results.total_trades)
                mlflow.log_metric("win_rate", exp.results.win_rate)
                mlflow.log_metric("sharpe_ratio", exp.results.sharpe_ratio)
                mlflow.log_metric("profit_factor", exp.results.profit_factor)
                mlflow.log_metric("max_drawdown", exp.results.max_drawdown)
                mlflow.log_metric("total_pnl", exp.results.total_pnl)

                # Log artifacts
                if exp.results.equity_curve_path:
                    path = Path(exp.results.equity_curve_path)
                    if path.exists():
                        mlflow.log_artifact(str(path), artifact_path="equity_curves")

                if exp.results.trade_list_path:
                    path = Path(exp.results.trade_list_path)
                    if path.exists():
                        mlflow.log_artifact(str(path), artifact_path="trade_lists")

                if exp.results.report_path:
                    path = Path(exp.results.report_path)
                    if path.exists():
                        mlflow.log_artifact(str(path), artifact_path="reports")

            # Log failure info
            if exp.status.value == "failed" and exp.error_message:
                mlflow.set_tag("error", exp.error_message[:250])

            logger.info(f"Synced {exp_id} to MLflow run {run.info.run_id}")
            return run.info.run_id

    def sync_all(self, status: Optional[str] = None, limit: int = 1000) -> Dict[str, int]:
        """
        Sync all experiments to MLflow.

        Args:
            status: Only sync experiments with this status
            limit: Maximum experiments to sync

        Returns:
            Dict with counts: {"synced": N, "failed": M, "skipped": K}
        """
        try:
            from experiments.registry import get_registry, ExperimentStatus
            registry = get_registry()
        except ImportError:
            logger.error("Could not import registry")
            return {"synced": 0, "failed": 0, "skipped": 0}

        # Get experiments
        exp_status = ExperimentStatus(status) if status else None
        experiments = registry.list_experiments(status=exp_status, limit=limit)

        counts = {"synced": 0, "failed": 0, "skipped": 0}

        for exp in experiments:
            try:
                run_id = self.sync_experiment(exp.id)
                if run_id:
                    counts["synced"] += 1
                else:
                    counts["skipped"] += 1
            except Exception as e:
                logger.error(f"Failed to sync {exp.id}: {e}")
                counts["failed"] += 1

        logger.info(f"Sync complete: {counts}")
        return counts

    # -------------------------------------------------------------------------
    # Direct Logging
    # -------------------------------------------------------------------------

    def log_backtest(
        self,
        strategy: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        run_name: Optional[str] = None,
    ) -> str:
        """
        Log a backtest directly to MLflow.

        Args:
            strategy: Strategy name
            params: Backtest parameters
            metrics: Result metrics
            artifacts: Dict of {artifact_name: file_path}
            tags: Additional tags
            run_name: Custom run name

        Returns:
            MLflow run ID
        """
        exp_name = self._get_experiment_name(strategy)
        mlflow.set_experiment(exp_name)

        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_param("strategy", strategy)
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)
                else:
                    mlflow.log_param(key, str(value))

            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    if Path(path).exists():
                        mlflow.log_artifact(path, artifact_path=name)

            # Set tags
            mlflow.set_tag("source", "direct_logging")
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)

            return run.info.run_id

    def log_walk_forward(
        self,
        strategy: str,
        params: Dict[str, Any],
        split_results: List[Dict[str, float]],
        artifacts: Optional[Dict[str, str]] = None,
        run_name: Optional[str] = None,
    ) -> str:
        """
        Log walk-forward results with time-series metrics.

        Args:
            strategy: Strategy name
            params: Walk-forward parameters
            split_results: List of per-split metrics dicts
            artifacts: Dict of {artifact_name: file_path}
            run_name: Custom run name

        Returns:
            MLflow run ID
        """
        exp_name = self._get_experiment_name(f"{strategy}_walk_forward")
        mlflow.set_experiment(exp_name)

        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_param("strategy", strategy)
            mlflow.log_param("n_splits", len(split_results))
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)

            # Log per-split metrics (time-series)
            for i, split in enumerate(split_results, start=1):
                for metric_name, value in split.items():
                    mlflow.log_metric(f"split_{metric_name}", value, step=i)

            # Log aggregate metrics
            if split_results:
                for metric_name in split_results[0].keys():
                    values = [s.get(metric_name, 0) for s in split_results]
                    mlflow.log_metric(f"avg_{metric_name}", sum(values) / len(values))
                    mlflow.log_metric(f"min_{metric_name}", min(values))
                    mlflow.log_metric(f"max_{metric_name}", max(values))
                    mlflow.log_metric(f"std_{metric_name}", pd.Series(values).std())

            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    if Path(path).exists():
                        mlflow.log_artifact(path, artifact_path=name)

            mlflow.set_tag("type", "walk_forward")
            return run.info.run_id

    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------

    def query_best_experiments(
        self,
        metric: str = "sharpe_ratio",
        min_value: float = 1.5,
        limit: int = 10,
        strategy: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Query best experiments by metric.

        Args:
            metric: Metric name (win_rate, sharpe_ratio, profit_factor)
            min_value: Minimum metric value
            limit: Maximum results
            strategy: Filter by strategy (optional)

        Returns:
            DataFrame of matching runs
        """
        filter_parts = [f"metrics.{metric} > {min_value}"]

        if strategy:
            filter_parts.append(f"params.strategy = '{strategy}'")

        filter_string = " AND ".join(filter_parts)

        runs = mlflow.search_runs(
            filter_string=filter_string,
            order_by=[f"metrics.{metric} DESC"],
            max_results=limit,
        )

        return runs

    def query_experiments(
        self,
        filter_string: str = "",
        order_by: Optional[List[str]] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Query experiments with custom filter.

        Filter string examples:
            - "metrics.sharpe_ratio > 1.5"
            - "params.strategy = 'DualStrategy'"
            - "metrics.win_rate > 0.6 AND metrics.max_drawdown < 0.15"
            - "params.start_date >= '2023-01-01'"

        Args:
            filter_string: MLflow filter expression
            order_by: List of order expressions (e.g., ["metrics.sharpe_ratio DESC"])
            limit: Maximum results

        Returns:
            DataFrame of matching runs
        """
        return mlflow.search_runs(
            filter_string=filter_string,
            order_by=order_by,
            max_results=limit,
        )

    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare all strategies by average metrics.

        Returns:
            DataFrame with strategies as index and average metrics as columns
        """
        runs = mlflow.search_runs(filter_string="", max_results=10000)

        if runs.empty:
            return pd.DataFrame()

        # Filter to runs with strategy param
        if "params.strategy" not in runs.columns:
            return pd.DataFrame()

        runs = runs[runs["params.strategy"].notna()]

        if runs.empty:
            return pd.DataFrame()

        # Aggregate by strategy
        metric_cols = [c for c in runs.columns if c.startswith("metrics.")]

        if not metric_cols:
            return pd.DataFrame()

        agg_dict = {col: ["mean", "std", "count"] for col in metric_cols}

        comparison = runs.groupby("params.strategy").agg(agg_dict)

        # Flatten column names
        comparison.columns = ["_".join(col).strip() for col in comparison.columns]

        return comparison.sort_values(
            "metrics.sharpe_ratio_mean" if "metrics.sharpe_ratio_mean" in comparison.columns else comparison.columns[0],
            ascending=False
        )

    def get_run_artifacts(self, run_id: str, artifact_path: str = "") -> List[str]:
        """
        Get list of artifacts for a run.

        Args:
            run_id: MLflow run ID
            artifact_path: Path within artifacts (optional)

        Returns:
            List of artifact file names
        """
        artifacts = self._client.list_artifacts(run_id, artifact_path)
        return [a.path for a in artifacts]

    def download_artifact(self, run_id: str, artifact_path: str, dst_path: str) -> str:
        """
        Download an artifact from a run.

        Args:
            run_id: MLflow run ID
            artifact_path: Path to artifact within run
            dst_path: Local destination path

        Returns:
            Path to downloaded file
        """
        return self._client.download_artifacts(run_id, artifact_path, dst_path)

    # -------------------------------------------------------------------------
    # Reports
    # -------------------------------------------------------------------------

    def generate_summary_report(self) -> str:
        """
        Generate summary report of all experiments.

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "MLFLOW EXPERIMENT SUMMARY",
            "=" * 60,
            f"Tracking URI: {self.tracking_uri}",
            f"Generated: {datetime.now().isoformat()}",
            "",
        ]

        # List experiments
        experiments = mlflow.search_experiments()
        lines.append(f"Experiments: {len(experiments)}")

        for exp in experiments:
            if exp.name.startswith(self.experiment_prefix):
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                lines.append(f"  - {exp.name}: {len(runs)} runs")

        # Strategy comparison
        lines.append("")
        lines.append("STRATEGY COMPARISON:")
        lines.append("-" * 40)

        comparison = self.compare_strategies()
        if not comparison.empty:
            for strategy in comparison.index:
                row = comparison.loc[strategy]
                sharpe_col = "metrics.sharpe_ratio_mean"
                wr_col = "metrics.win_rate_mean"

                sharpe = row.get(sharpe_col, 0)
                wr = row.get(wr_col, 0)

                lines.append(f"  {strategy}:")
                lines.append(f"    Sharpe: {sharpe:.2f}")
                lines.append(f"    Win Rate: {wr:.1%}")
        else:
            lines.append("  No data available")

        # Top performers
        lines.append("")
        lines.append("TOP PERFORMERS (Sharpe > 1.5):")
        lines.append("-" * 40)

        top = self.query_best_experiments(metric="sharpe_ratio", min_value=1.5, limit=5)
        if not top.empty:
            for _, row in top.iterrows():
                name = row.get("tags.mlflow.runName", row.get("run_id", "unknown"))[:30]
                sharpe = row.get("metrics.sharpe_ratio", 0)
                wr = row.get("metrics.win_rate", 0)
                lines.append(f"  {name}: Sharpe={sharpe:.2f}, WR={wr:.1%}")
        else:
            lines.append("  No runs with Sharpe > 1.5")

        return "\n".join(lines)


# =============================================================================
# SINGLETON
# =============================================================================

_mlflow_adapter: Optional[MLflowAdapter] = None


def get_mlflow_adapter() -> MLflowAdapter:
    """Get singleton MLflowAdapter instance."""
    global _mlflow_adapter
    if _mlflow_adapter is None:
        _mlflow_adapter = MLflowAdapter()
    return _mlflow_adapter


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def sync_to_mlflow(exp_id: str) -> Optional[str]:
    """Sync experiment to MLflow."""
    return get_mlflow_adapter().sync_experiment(exp_id)


def query_best(metric: str = "sharpe_ratio", min_value: float = 1.5) -> pd.DataFrame:
    """Query best experiments."""
    return get_mlflow_adapter().query_best_experiments(metric, min_value)


def log_backtest_to_mlflow(
    strategy: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    **kwargs
) -> str:
    """Log backtest directly to MLflow."""
    return get_mlflow_adapter().log_backtest(strategy, params, metrics, **kwargs)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLflow adapter for Kobe experiments")
    parser.add_argument("--sync-all", action="store_true", help="Sync all experiments")
    parser.add_argument("--sync", type=str, help="Sync specific experiment ID")
    parser.add_argument("--query", type=str, help="Query filter string")
    parser.add_argument("--compare", action="store_true", help="Compare strategies")
    parser.add_argument("--report", action="store_true", help="Generate summary report")
    parser.add_argument("--server", action="store_true", help="Start MLflow server")
    parser.add_argument("--port", type=int, default=5000, help="Server port")

    args = parser.parse_args()

    if args.server:
        import subprocess
        print(f"Starting MLflow server at http://localhost:{args.port}")
        subprocess.run([
            "mlflow", "server",
            "--backend-store-uri", DEFAULT_TRACKING_URI,
            "--host", "127.0.0.1",
            "--port", str(args.port),
        ])

    elif args.sync_all:
        adapter = get_mlflow_adapter()
        counts = adapter.sync_all()
        print(f"Synced: {counts['synced']}, Failed: {counts['failed']}, Skipped: {counts['skipped']}")

    elif args.sync:
        adapter = get_mlflow_adapter()
        run_id = adapter.sync_experiment(args.sync)
        if run_id:
            print(f"Synced to run: {run_id}")
        else:
            print("Sync failed")

    elif args.query:
        adapter = get_mlflow_adapter()
        results = adapter.query_experiments(filter_string=args.query)
        print(results[["run_id", "params.strategy", "metrics.sharpe_ratio", "metrics.win_rate"]].to_string())

    elif args.compare:
        adapter = get_mlflow_adapter()
        comparison = adapter.compare_strategies()
        print(comparison.to_string())

    elif args.report:
        adapter = get_mlflow_adapter()
        print(adapter.generate_summary_report())

    else:
        parser.print_help()
