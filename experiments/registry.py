"""
Experiment Registry.

Tracks all backtesting experiments for reproducibility:
- Dataset ID used
- Strategy parameters
- Random seeds
- Results hash
- Git commit (optional)

Every experiment can be reproduced exactly by loading its config.

Usage:
    from experiments.registry import ExperimentRegistry

    registry = ExperimentRegistry()

    # Register a new experiment
    exp_id = registry.register(
        name="momentum_v2_test",
        dataset_id="stooq_1d_2015_2025_abc123",
        strategy="momentum",
        params={"lookback": 20, "threshold": 0.02},
        seed=42,
    )

    # Run and record results
    results = run_backtest(...)
    registry.record_results(exp_id, results)

    # List all experiments
    for exp in registry.list_experiments():
        print(f"{exp.id}: {exp.name} - {exp.status}")

    # Reproduce an experiment
    config = registry.get_config(exp_id)
    results = run_backtest(**config.to_backtest_args())
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment status."""
    REGISTERED = "registered"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    dataset_id: str
    strategy: str
    params: Dict[str, Any]
    seed: int = 42

    # Optional metadata
    universe_path: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_equity: float = 100_000
    risk_pct: float = 0.02

    # Versioning
    config_hash: str = ""

    def __post_init__(self):
        """Compute config hash after initialization."""
        if not self.config_hash:
            self.config_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute deterministic hash of config."""
        # Create canonical representation
        canonical = json.dumps({
            "dataset_id": self.dataset_id,
            "strategy": self.strategy,
            "params": self.params,
            "seed": self.seed,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_equity": self.initial_equity,
            "risk_pct": self.risk_pct,
        }, sort_keys=True)

        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_backtest_args(self) -> Dict[str, Any]:
        """Convert to backtest function arguments."""
        return {
            "dataset_id": self.dataset_id,
            "strategy": self.strategy,
            "params": self.params,
            "seed": self.seed,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_equity": self.initial_equity,
            "risk_pct": self.risk_pct,
        }


@dataclass
class ExperimentResults:
    """Results from an experiment."""
    total_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: float = 0.0
    results_hash: str = ""

    # Artifacts
    equity_curve_path: Optional[str] = None
    trade_list_path: Optional[str] = None
    report_path: Optional[str] = None

    def compute_hash(self) -> str:
        """Compute hash of results for reproducibility verification."""
        canonical = json.dumps({
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 6),
            "profit_factor": round(self.profit_factor, 6),
            "max_drawdown": round(self.max_drawdown, 6),
            "total_pnl": round(self.total_pnl, 2),
        }, sort_keys=True)

        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Experiment:
    """A registered experiment."""
    id: str
    name: str
    config: ExperimentConfig
    status: ExperimentStatus
    created_at: str
    updated_at: str

    # Optional
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Results (filled after completion)
    results: Optional[ExperimentResults] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "id": self.id,
            "name": self.name,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "description": self.description,
            "tags": self.tags,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
        }

        if self.results:
            d["results"] = self.results.to_dict()

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Experiment':
        """Create from dictionary."""
        config = ExperimentConfig(**d["config"])
        results = None
        if "results" in d and d["results"]:
            results = ExperimentResults(**d["results"])

        return cls(
            id=d["id"],
            name=d["name"],
            config=config,
            status=ExperimentStatus(d["status"]),
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            git_commit=d.get("git_commit"),
            git_branch=d.get("git_branch"),
            description=d.get("description"),
            tags=d.get("tags", []),
            results=results,
            completed_at=d.get("completed_at"),
            error_message=d.get("error_message"),
        )


class ExperimentRegistry:
    """
    Registry for tracking experiments.

    All experiments are stored in a JSON file for persistence.
    Each experiment can be reproduced by loading its config.
    """

    def __init__(
        self,
        registry_dir: str = "experiments",
        registry_file: str = "registry.json",
    ):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / registry_file
        self.artifacts_dir = self.registry_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)

        self._experiments: Dict[str, Experiment] = {}
        self._load()

        logger.info(f"ExperimentRegistry initialized with {len(self._experiments)} experiments")

    def _load(self):
        """Load registry from file."""
        if self.registry_file.exists():
            try:
                data = json.loads(self.registry_file.read_text())
                for exp_dict in data.get("experiments", []):
                    exp = Experiment.from_dict(exp_dict)
                    self._experiments[exp.id] = exp
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")

    def _save(self):
        """Save registry to file."""
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "experiments": [exp.to_dict() for exp in self._experiments.values()],
        }
        self.registry_file.write_text(json.dumps(data, indent=2, default=str))

    def _get_git_info(self) -> tuple:
        """Get current git commit and branch."""
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()[:8]

            branch = subprocess.check_output(
                ["git", "branch", "--show-current"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()

            return commit, branch
        except:
            return None, None

    def register(
        self,
        name: str,
        dataset_id: str,
        strategy: str,
        params: Dict[str, Any],
        seed: int = 42,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Register a new experiment.

        Args:
            name: Human-readable experiment name
            dataset_id: Frozen dataset ID
            strategy: Strategy name
            params: Strategy parameters
            seed: Random seed for reproducibility
            description: Optional description
            tags: Optional tags for filtering
            **kwargs: Additional config parameters

        Returns:
            Experiment ID
        """
        # Create config
        config = ExperimentConfig(
            dataset_id=dataset_id,
            strategy=strategy,
            params=params,
            seed=seed,
            **kwargs,
        )

        # Generate ID
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.config_hash[:6]}"

        # Get git info
        git_commit, git_branch = self._get_git_info()

        now = datetime.now().isoformat()

        experiment = Experiment(
            id=exp_id,
            name=name,
            config=config,
            status=ExperimentStatus.REGISTERED,
            created_at=now,
            updated_at=now,
            git_commit=git_commit,
            git_branch=git_branch,
            description=description,
            tags=tags or [],
        )

        self._experiments[exp_id] = experiment
        self._save()

        logger.info(f"Registered experiment: {exp_id} ({name})")
        return exp_id

    def start(self, exp_id: str):
        """Mark experiment as running."""
        if exp_id not in self._experiments:
            raise ValueError(f"Experiment not found: {exp_id}")

        exp = self._experiments[exp_id]
        exp.status = ExperimentStatus.RUNNING
        exp.updated_at = datetime.now().isoformat()
        self._save()

    def record_results(
        self,
        exp_id: str,
        results: Dict[str, Any],
        equity_curve_path: Optional[str] = None,
        trade_list_path: Optional[str] = None,
        report_path: Optional[str] = None,
    ):
        """
        Record results for an experiment.

        Args:
            exp_id: Experiment ID
            results: Results dictionary with metrics
            equity_curve_path: Path to equity curve CSV
            trade_list_path: Path to trade list CSV
            report_path: Path to report JSON
        """
        if exp_id not in self._experiments:
            raise ValueError(f"Experiment not found: {exp_id}")

        exp = self._experiments[exp_id]

        # Create results object
        exp_results = ExperimentResults(
            total_trades=results.get("total_trades", 0),
            win_rate=results.get("win_rate", 0.0),
            sharpe_ratio=results.get("sharpe_ratio", results.get("sharpe", 0.0)),
            profit_factor=results.get("profit_factor", 0.0),
            max_drawdown=results.get("max_drawdown", results.get("max_dd", 0.0)),
            total_pnl=results.get("total_pnl", results.get("pnl", 0.0)),
            equity_curve_path=equity_curve_path,
            trade_list_path=trade_list_path,
            report_path=report_path,
        )
        exp_results.results_hash = exp_results.compute_hash()

        exp.results = exp_results
        exp.status = ExperimentStatus.COMPLETED
        exp.completed_at = datetime.now().isoformat()
        exp.updated_at = datetime.now().isoformat()

        self._save()
        logger.info(f"Recorded results for {exp_id}: Sharpe={exp_results.sharpe_ratio:.2f}")

    def record_failure(self, exp_id: str, error_message: str):
        """Record experiment failure."""
        if exp_id not in self._experiments:
            raise ValueError(f"Experiment not found: {exp_id}")

        exp = self._experiments[exp_id]
        exp.status = ExperimentStatus.FAILED
        exp.error_message = error_message
        exp.updated_at = datetime.now().isoformat()
        self._save()

        logger.warning(f"Experiment {exp_id} failed: {error_message}")

    def get_experiment(self, exp_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self._experiments.get(exp_id)

    def get_config(self, exp_id: str) -> Optional[ExperimentConfig]:
        """Get experiment config for reproduction."""
        exp = self.get_experiment(exp_id)
        return exp.config if exp else None

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        strategy: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Experiment]:
        """
        List experiments with optional filtering.

        Args:
            status: Filter by status
            strategy: Filter by strategy name
            tags: Filter by tags (any match)
            limit: Maximum results

        Returns:
            List of experiments (most recent first)
        """
        experiments = list(self._experiments.values())

        # Filter
        if status:
            experiments = [e for e in experiments if e.status == status]

        if strategy:
            experiments = [e for e in experiments if e.config.strategy == strategy]

        if tags:
            experiments = [
                e for e in experiments
                if any(t in e.tags for t in tags)
            ]

        # Sort by created_at descending
        experiments.sort(key=lambda e: e.created_at, reverse=True)

        return experiments[:limit]

    def compare_experiments(
        self,
        exp_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments.

        Args:
            exp_ids: List of experiment IDs to compare

        Returns:
            Comparison summary
        """
        experiments = [self.get_experiment(eid) for eid in exp_ids]
        experiments = [e for e in experiments if e and e.results]

        if not experiments:
            return {"error": "No completed experiments found"}

        comparison = {
            "experiments": [],
            "best_sharpe": None,
            "best_win_rate": None,
            "best_profit_factor": None,
        }

        best_sharpe = -float('inf')
        best_win_rate = 0
        best_pf = 0

        for exp in experiments:
            entry = {
                "id": exp.id,
                "name": exp.name,
                "strategy": exp.config.strategy,
                "sharpe": exp.results.sharpe_ratio,
                "win_rate": exp.results.win_rate,
                "profit_factor": exp.results.profit_factor,
                "max_drawdown": exp.results.max_drawdown,
                "total_pnl": exp.results.total_pnl,
            }
            comparison["experiments"].append(entry)

            if exp.results.sharpe_ratio > best_sharpe:
                best_sharpe = exp.results.sharpe_ratio
                comparison["best_sharpe"] = exp.id

            if exp.results.win_rate > best_win_rate:
                best_win_rate = exp.results.win_rate
                comparison["best_win_rate"] = exp.id

            if exp.results.profit_factor > best_pf:
                best_pf = exp.results.profit_factor
                comparison["best_profit_factor"] = exp.id

        return comparison

    def archive(self, exp_id: str):
        """Archive an experiment."""
        if exp_id not in self._experiments:
            raise ValueError(f"Experiment not found: {exp_id}")

        exp = self._experiments[exp_id]
        exp.status = ExperimentStatus.ARCHIVED
        exp.updated_at = datetime.now().isoformat()
        self._save()

    def delete(self, exp_id: str):
        """Delete an experiment."""
        if exp_id in self._experiments:
            del self._experiments[exp_id]
            self._save()
            logger.info(f"Deleted experiment: {exp_id}")

    def export_experiment(self, exp_id: str, output_path: Optional[Path] = None) -> Path:
        """
        Export experiment config for sharing/reproduction.

        Args:
            exp_id: Experiment ID
            output_path: Output path (default: artifacts/<exp_id>.json)

        Returns:
            Path to exported file
        """
        exp = self.get_experiment(exp_id)
        if not exp:
            raise ValueError(f"Experiment not found: {exp_id}")

        if output_path is None:
            output_path = self.artifacts_dir / f"{exp_id}.json"

        output_path.write_text(json.dumps(exp.to_dict(), indent=2, default=str))
        return output_path

    def import_experiment(self, config_path: Path) -> str:
        """
        Import experiment from config file.

        Args:
            config_path: Path to experiment JSON

        Returns:
            New experiment ID
        """
        data = json.loads(config_path.read_text())

        # Create new experiment with fresh ID
        exp_id = self.register(
            name=f"imported_{data['name']}",
            dataset_id=data["config"]["dataset_id"],
            strategy=data["config"]["strategy"],
            params=data["config"]["params"],
            seed=data["config"]["seed"],
            description=f"Imported from {config_path.name}",
            tags=data.get("tags", []) + ["imported"],
        )

        return exp_id

    def verify_reproducibility(self, exp_id: str, new_results: Dict[str, Any]) -> bool:
        """
        Verify that new results match original experiment.

        Args:
            exp_id: Original experiment ID
            new_results: New results from re-running

        Returns:
            True if results match (within tolerance)
        """
        exp = self.get_experiment(exp_id)
        if not exp or not exp.results:
            return False

        new_exp_results = ExperimentResults(
            total_trades=new_results.get("total_trades", 0),
            win_rate=new_results.get("win_rate", 0.0),
            sharpe_ratio=new_results.get("sharpe_ratio", 0.0),
            profit_factor=new_results.get("profit_factor", 0.0),
            max_drawdown=new_results.get("max_drawdown", 0.0),
            total_pnl=new_results.get("total_pnl", 0.0),
        )
        new_hash = new_exp_results.compute_hash()

        match = new_hash == exp.results.results_hash
        if not match:
            logger.warning(
                f"Reproducibility check failed for {exp_id}: "
                f"original={exp.results.results_hash}, new={new_hash}"
            )

        return match


# Singleton instance
_registry: Optional[ExperimentRegistry] = None


def get_registry() -> ExperimentRegistry:
    """Get singleton registry instance."""
    global _registry
    if _registry is None:
        _registry = ExperimentRegistry()
    return _registry


# Convenience functions
def register_experiment(
    name: str,
    dataset_id: str,
    strategy: str,
    params: Dict[str, Any],
    **kwargs,
) -> str:
    """Register a new experiment."""
    return get_registry().register(name, dataset_id, strategy, params, **kwargs)


def record_experiment_results(exp_id: str, results: Dict[str, Any], **kwargs):
    """Record results for an experiment."""
    get_registry().record_results(exp_id, results, **kwargs)


def list_experiments(**kwargs) -> List[Experiment]:
    """List experiments."""
    return get_registry().list_experiments(**kwargs)
