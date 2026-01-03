"""
Reproducibility & Data Versioning Module
=========================================

Ensures research reproducibility through:
- Data versioning (DVC-compatible)
- Experiment manifests
- Code + data + config checksums
- Freezing/snapshotting data states

Based on best practices from:
- DVC (Data Version Control)
- MLflow experiment tracking
- Weights & Biases run management

Usage:
    from backtest.reproducibility import ExperimentTracker

    tracker = ExperimentTracker()
    with tracker.run("my_experiment") as run:
        run.log_params(strategy_params)
        results = backtest.run(...)
        run.log_metrics(results.metrics)
        run.log_artifact("equity_curve.csv", results.equity)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

logger = logging.getLogger(__name__)

# Root directory for experiments
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
DATA_DIR = ROOT / "data"


@dataclass
class DataVersion:
    """Represents a versioned snapshot of data."""
    version_id: str
    source_path: str
    checksum: str
    row_count: int
    created_at: str
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'DataVersion':
        return cls(**d)


@dataclass
class ExperimentManifest:
    """Complete manifest for reproducible experiments."""
    experiment_id: str
    name: str
    created_at: str
    git_commit: Optional[str]
    git_branch: Optional[str]
    git_dirty: bool
    python_version: str
    platform: str
    code_checksum: str
    config_checksum: str
    data_versions: Dict[str, DataVersion]
    parameters: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['data_versions'] = {k: v.to_dict() for k, v in self.data_versions.items()}
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'ExperimentManifest':
        d['data_versions'] = {
            k: DataVersion.from_dict(v)
            for k, v in d.get('data_versions', {}).items()
        }
        return cls(**d)


def compute_file_checksum(path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """Compute checksum of a file."""
    path = Path(path)
    if not path.exists():
        return ""

    hasher = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_directory_checksum(
    path: Union[str, Path],
    extensions: Optional[List[str]] = None,
) -> str:
    """Compute combined checksum of all files in a directory."""
    path = Path(path)
    if not path.exists():
        return ""

    extensions = extensions or ['.py', '.yaml', '.json']
    hasher = hashlib.sha256()

    for file_path in sorted(path.rglob('*')):
        if file_path.is_file() and file_path.suffix in extensions:
            # Include relative path in hash for reproducibility
            rel_path = file_path.relative_to(path)
            hasher.update(str(rel_path).encode())
            hasher.update(compute_file_checksum(file_path).encode())

    return hasher.hexdigest()


def get_git_info() -> Dict[str, Any]:
    """Get current git repository information."""
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()

        branch = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()

        dirty = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, cwd=ROOT
        ).stdout.strip() != ""

        return {
            'commit': commit,
            'branch': branch,
            'dirty': dirty,
        }
    except Exception as e:
        logger.warning(f"Could not get git info: {e}")
        return {'commit': None, 'branch': None, 'dirty': True}


class DataVersioner:
    """
    Manages data versioning for reproducible research.

    Features:
    - Checksum-based version tracking
    - Snapshot creation and restoration
    - DVC-compatible .dvc file generation
    - Version history management
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or DATA_DIR / ".versions"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.storage_dir / "versions.json"
        self._versions: Dict[str, DataVersion] = {}
        self._load_versions()

    def _load_versions(self):
        """Load existing versions from storage."""
        if self.versions_file.exists():
            try:
                data = json.loads(self.versions_file.read_text())
                self._versions = {
                    k: DataVersion.from_dict(v)
                    for k, v in data.items()
                }
            except Exception as e:
                logger.warning(f"Could not load versions: {e}")

    def _save_versions(self):
        """Save versions to storage."""
        data = {k: v.to_dict() for k, v in self._versions.items()}
        self.versions_file.write_text(json.dumps(data, indent=2))

    def register_data(
        self,
        name: str,
        path: Union[str, Path],
        description: str = "",
        metadata: Optional[Dict] = None,
    ) -> DataVersion:
        """
        Register a data file/directory and create a version record.

        Args:
            name: Unique name for this data source
            path: Path to the data file or directory
            description: Human-readable description
            metadata: Additional metadata

        Returns:
            DataVersion object
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {path}")

        # Compute checksum
        if path.is_file():
            checksum = compute_file_checksum(path)
            row_count = self._count_rows(path)
        else:
            checksum = compute_directory_checksum(path)
            row_count = sum(self._count_rows(f) for f in path.rglob('*.csv'))

        # Check if this exact version already exists
        for vid, ver in self._versions.items():
            if ver.source_path == str(path) and ver.checksum == checksum:
                logger.info(f"Data {name} already registered with same checksum")
                return ver

        # Create new version
        version = DataVersion(
            version_id=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{checksum[:8]}",
            source_path=str(path),
            checksum=checksum,
            row_count=row_count,
            created_at=datetime.now().isoformat(),
            description=description,
            metadata=metadata or {},
        )

        self._versions[version.version_id] = version
        self._save_versions()

        logger.info(f"Registered data version: {version.version_id}")
        return version

    def _count_rows(self, path: Path) -> int:
        """Count rows in a CSV file."""
        if not path.suffix == '.csv':
            return 0
        try:
            with open(path, 'r') as f:
                return sum(1 for _ in f) - 1  # Subtract header
        except:
            return 0

    def create_snapshot(
        self,
        name: str,
        path: Union[str, Path],
        description: str = "",
    ) -> DataVersion:
        """
        Create an immutable snapshot of data.

        Copies the data to versioned storage for future reproducibility.
        """
        path = Path(path)
        version = self.register_data(name, path, description)

        # Create snapshot directory
        snapshot_dir = self.storage_dir / "snapshots" / version.version_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Copy data to snapshot
        if path.is_file():
            shutil.copy2(path, snapshot_dir / path.name)
        else:
            shutil.copytree(path, snapshot_dir / path.name, dirs_exist_ok=True)

        logger.info(f"Created snapshot: {snapshot_dir}")
        return version

    def generate_dvc_file(
        self,
        name: str,
        path: Union[str, Path],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate a DVC-compatible .dvc file for the data.

        This allows integration with DVC for data versioning.
        """
        path = Path(path)
        version = self.register_data(name, path)

        dvc_content = {
            'md5': version.checksum,
            'outs': [
                {
                    'md5': version.checksum,
                    'size': path.stat().st_size if path.is_file() else 0,
                    'path': path.name,
                }
            ],
            'meta': {
                'kobe_version_id': version.version_id,
                'created_at': version.created_at,
                'description': version.description,
            },
        }

        dvc_path = output_path or path.with_suffix(path.suffix + '.dvc')
        dvc_path.write_text(json.dumps(dvc_content, indent=2))

        logger.info(f"Generated DVC file: {dvc_path}")
        return dvc_path

    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Get a specific data version."""
        return self._versions.get(version_id)

    def list_versions(self, name_filter: Optional[str] = None) -> List[DataVersion]:
        """List all registered versions."""
        versions = list(self._versions.values())
        if name_filter:
            versions = [v for v in versions if name_filter in v.version_id]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def verify_data(self, name: str, path: Union[str, Path]) -> bool:
        """Verify that data matches its registered version."""
        path = Path(path)
        if not path.exists():
            return False

        current_checksum = compute_file_checksum(path) if path.is_file() else compute_directory_checksum(path)

        for ver in self._versions.values():
            if ver.source_path == str(path):
                if ver.checksum == current_checksum:
                    return True
                else:
                    logger.warning(f"Data {name} has been modified since version {ver.version_id}")
                    return False

        logger.warning(f"No registered version found for {name}")
        return False


class ExperimentRun:
    """Context manager for tracking a single experiment run."""

    def __init__(self, tracker: 'ExperimentTracker', name: str):
        self.tracker = tracker
        self.name = name
        self.experiment_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.parameters: Dict[str, Any] = {}
        self.metrics: Dict[str, float] = {}
        self.artifacts: List[str] = []
        self.tags: List[str] = []
        self.notes: str = ""
        self.data_versions: Dict[str, DataVersion] = {}
        self._run_dir: Optional[Path] = None

    def __enter__(self) -> 'ExperimentRun':
        self._run_dir = self.tracker.experiments_dir / self.experiment_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Started experiment: {self.experiment_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Save manifest
        manifest = self._create_manifest()
        manifest_path = self._run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))
        logger.info(f"Experiment complete: {self.experiment_id}")

    def _create_manifest(self) -> ExperimentManifest:
        """Create experiment manifest."""
        git_info = get_git_info()

        return ExperimentManifest(
            experiment_id=self.experiment_id,
            name=self.name,
            created_at=datetime.now().isoformat(),
            git_commit=git_info['commit'],
            git_branch=git_info['branch'],
            git_dirty=git_info['dirty'],
            python_version=sys.version,
            platform=platform.platform(),
            code_checksum=compute_directory_checksum(ROOT, ['.py']),
            config_checksum=compute_directory_checksum(ROOT / 'config', ['.yaml', '.json']),
            data_versions=self.data_versions,
            parameters=self.parameters,
            metrics=self.metrics,
            artifacts=self.artifacts,
            tags=self.tags,
            notes=self.notes,
        )

    def log_params(self, params: Dict[str, Any]):
        """Log experiment parameters."""
        self.parameters.update(params)
        logger.debug(f"Logged params: {params}")

    def log_metrics(self, metrics: Dict[str, float]):
        """Log experiment metrics."""
        self.metrics.update(metrics)
        logger.debug(f"Logged metrics: {metrics}")

    def log_metric(self, name: str, value: float):
        """Log a single metric."""
        self.metrics[name] = value

    def log_artifact(self, name: str, data: Any, format: str = 'auto'):
        """
        Log an artifact (file, DataFrame, figure).

        Args:
            name: Artifact name (becomes filename)
            data: Data to save
            format: 'csv', 'json', 'pickle', or 'auto'
        """
        import pandas as pd

        if self._run_dir is None:
            return

        artifacts_dir = self._run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        # Determine format
        if format == 'auto':
            if isinstance(data, pd.DataFrame):
                format = 'csv'
            elif isinstance(data, (dict, list)):
                format = 'json'
            else:
                format = 'pickle'

        # Save artifact
        if format == 'csv' and isinstance(data, pd.DataFrame):
            path = artifacts_dir / f"{name}.csv"
            data.to_csv(path, index=True)
        elif format == 'json':
            path = artifacts_dir / f"{name}.json"
            path.write_text(json.dumps(data, indent=2, default=str))
        else:
            import pickle
            path = artifacts_dir / f"{name}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(data, f)

        self.artifacts.append(str(path.relative_to(self._run_dir)))
        logger.debug(f"Logged artifact: {name}")

    def log_data_version(self, name: str, version: DataVersion):
        """Log a data version used in this experiment."""
        self.data_versions[name] = version

    def add_tag(self, tag: str):
        """Add a tag to this experiment."""
        if tag not in self.tags:
            self.tags.append(tag)

    def set_notes(self, notes: str):
        """Set experiment notes."""
        self.notes = notes


class ExperimentTracker:
    """
    Tracks experiments for reproducible research.

    Features:
    - Automatic code/data/config versioning
    - Experiment comparison
    - Artifact storage
    - Git integration
    """

    def __init__(self, experiments_dir: Optional[Path] = None):
        self.experiments_dir = experiments_dir or EXPERIMENTS_DIR
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.data_versioner = DataVersioner()

    def run(self, name: str) -> ExperimentRun:
        """Start a new experiment run."""
        return ExperimentRun(self, name)

    def list_experiments(
        self,
        name_filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[ExperimentManifest]:
        """List all experiments."""
        experiments = []

        for exp_dir in sorted(self.experiments_dir.iterdir(), reverse=True):
            if not exp_dir.is_dir():
                continue

            manifest_path = exp_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            try:
                manifest = ExperimentManifest.from_dict(
                    json.loads(manifest_path.read_text())
                )
                if name_filter is None or name_filter in manifest.name:
                    experiments.append(manifest)

                if len(experiments) >= limit:
                    break
            except Exception as e:
                logger.warning(f"Could not load experiment {exp_dir}: {e}")

        return experiments

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentManifest]:
        """Get a specific experiment by ID."""
        exp_dir = self.experiments_dir / experiment_id
        manifest_path = exp_dir / "manifest.json"

        if manifest_path.exists():
            return ExperimentManifest.from_dict(
                json.loads(manifest_path.read_text())
            )
        return None

    def compare_experiments(
        self,
        experiment_ids: List[str],
    ) -> pd.DataFrame:
        """Compare multiple experiments."""
        import pandas as pd

        rows = []
        for exp_id in experiment_ids:
            manifest = self.get_experiment(exp_id)
            if manifest:
                row = {
                    'experiment_id': exp_id,
                    'name': manifest.name,
                    'created_at': manifest.created_at,
                    **manifest.metrics,
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def load_artifact(
        self,
        experiment_id: str,
        artifact_name: str,
    ) -> Any:
        """Load an artifact from an experiment."""
        import pandas as pd
        import pickle

        exp_dir = self.experiments_dir / experiment_id / "artifacts"

        # Try different extensions
        for ext in ['.csv', '.json', '.pkl']:
            path = exp_dir / f"{artifact_name}{ext}"
            if path.exists():
                if ext == '.csv':
                    return pd.read_csv(path, index_col=0)
                elif ext == '.json':
                    return json.loads(path.read_text())
                else:
                    with open(path, 'rb') as f:
                        return pickle.load(f)

        raise FileNotFoundError(f"Artifact {artifact_name} not found")


# =============================================================================
# Phase 7: Global Seed Management (Codex #5)
# =============================================================================

# Track if seeds have been set
_seeds_set = False
_seed_value: Optional[int] = None


def set_global_seeds(seed: int = 42) -> None:
    """
    Ensure deterministic behavior across all random sources.

    Sets seeds for:
    - Python random module
    - NumPy random
    - PyTorch (if available)
    - TensorFlow/Keras (if available)
    - Python hash seed (via environment variable)

    Args:
        seed: Integer seed value for reproducibility (default: 42)

    Note:
        Call this at the START of any backtest or training script,
        BEFORE importing any ML libraries that initialize random state.
    """
    global _seeds_set, _seed_value

    import random

    # Python built-in random
    random.seed(seed)
    logger.debug(f"Set Python random seed: {seed}")

    # Python hash seed (for dict ordering in Python 3.7+)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.debug(f"Set PYTHONHASHSEED: {seed}")

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
        logger.debug(f"Set NumPy random seed: {seed}")
    except ImportError:
        pass

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For full determinism (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.debug(f"Set PyTorch seeds: {seed}")
    except ImportError:
        pass

    # TensorFlow/Keras
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.debug(f"Set TensorFlow seed: {seed}")
    except ImportError:
        pass

    _seeds_set = True
    _seed_value = seed
    logger.info(f"Global reproducibility seeds set to: {seed}")


def get_seed() -> Optional[int]:
    """Get the current global seed value (if set)."""
    return _seed_value


def seeds_are_set() -> bool:
    """Check if global seeds have been set."""
    return _seeds_set


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Get information about the current reproducibility state.

    Useful for logging and debugging reproducibility issues.

    Returns:
        Dict with seed status, Python version, platform, and package versions
    """
    info = {
        "seeds_set": _seeds_set,
        "seed_value": _seed_value,
        "python_version": sys.version,
        "platform": platform.platform(),
        "python_hash_seed": os.environ.get('PYTHONHASHSEED', 'NOT_SET'),
        "timestamp": datetime.now().isoformat(),
        "packages": {},
    }

    # Get versions of key packages
    packages_to_check = [
        "numpy",
        "pandas",
        "torch",
        "tensorflow",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "hmmlearn",
    ]

    for pkg in packages_to_check:
        try:
            mod = __import__(pkg)
            info["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info["packages"][pkg] = "not_installed"

    return info


class ReproducibleContext:
    """
    Context manager for reproducible code blocks.

    Usage:
        with ReproducibleContext(seed=42):
            # Code here will be reproducible
            model.fit(X, y)
    """

    def __init__(self, seed: int = 42, verbose: bool = False):
        self.seed = seed
        self.verbose = verbose
        self._previous_seed = _seed_value

    def __enter__(self):
        set_global_seeds(self.seed)
        if self.verbose:
            info = get_reproducibility_info()
            logger.info(f"Reproducible context started with seed={self.seed}")
        return self

    def __exit__(self, *args):
        # Optionally restore previous seed state
        if self._previous_seed is not None:
            set_global_seeds(self._previous_seed)


def create_reproducibility_report(experiment_id: str) -> str:
    """Generate a reproducibility report for an experiment."""
    tracker = ExperimentTracker()
    manifest = tracker.get_experiment(experiment_id)

    if not manifest:
        return f"Experiment {experiment_id} not found"

    lines = [
        "=" * 60,
        "REPRODUCIBILITY REPORT",
        "=" * 60,
        "",
        f"Experiment: {manifest.name}",
        f"ID: {manifest.experiment_id}",
        f"Created: {manifest.created_at}",
        "",
        "--- Environment ---",
        f"Python: {manifest.python_version.split()[0]}",
        f"Platform: {manifest.platform}",
        "",
        "--- Git State ---",
        f"Branch: {manifest.git_branch}",
        f"Commit: {manifest.git_commit}",
        f"Dirty: {'Yes - UNCOMMITTED CHANGES' if manifest.git_dirty else 'No - Clean'}",
        "",
        "--- Checksums ---",
        f"Code: {manifest.code_checksum[:16]}...",
        f"Config: {manifest.config_checksum[:16]}...",
        "",
        "--- Data Versions ---",
    ]

    for name, version in manifest.data_versions.items():
        lines.append(f"  {name}:")
        lines.append(f"    Version: {version.version_id}")
        lines.append(f"    Checksum: {version.checksum[:16]}...")
        lines.append(f"    Rows: {version.row_count:,}")

    lines.extend([
        "",
        "--- Parameters ---",
    ])
    for key, value in manifest.parameters.items():
        lines.append(f"  {key}: {value}")

    lines.extend([
        "",
        "--- Metrics ---",
    ])
    for key, value in manifest.metrics.items():
        lines.append(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    lines.extend([
        "",
        "--- Artifacts ---",
    ])
    for artifact in manifest.artifacts:
        lines.append(f"  - {artifact}")

    lines.extend([
        "",
        "=" * 60,
        "To reproduce this experiment:",
        f"  1. git checkout {manifest.git_commit}",
        "  2. Verify data checksums match",
        "  3. Run with same parameters",
        "=" * 60,
    ])

    return "\n".join(lines)
