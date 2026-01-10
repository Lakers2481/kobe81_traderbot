"""
Evidence Pack System for Reproducibility.

This module provides a comprehensive evidence pack system for ensuring
full reproducibility of backtests, walk-forward tests, and live trades.

An EvidencePack captures everything needed to reproduce a result:
- Git state (commit, branch, dirty status)
- Environment (Python version, package versions)
- Configuration (frozen params, settings)
- Data (dataset ID, universe hash, date range)
- Results (metrics, trade list)
- Artifacts (file paths with hashes)

Blueprint Alignment:
    This implements the "Evidence Pack" requirement from Section 2.4
    of the production-grade trading system blueprint.

Usage:
    from research.evidence import EvidencePack, EvidencePackBuilder

    # Create evidence pack for a backtest
    builder = EvidencePackBuilder(pack_type="backtest")
    builder.capture_git_state()
    builder.capture_environment()
    builder.set_config(config_dict)
    builder.set_frozen_params(frozen_params_path)
    builder.set_dataset(dataset_id, universe_hash, start, end)
    builder.set_metrics(metrics_dict)
    builder.add_artifact("trade_list", "path/to/trades.csv")

    pack = builder.build()
    pack.save(output_dir)

    # Generate reproduction script
    script = pack.generate_reproduce_script()
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.structured_log import jlog


@dataclass
class EvidencePack:
    """
    Complete evidence pack for reproducibility.

    Contains all information needed to reproduce a backtest,
    walk-forward test, or live trade result.
    """
    # Pack identification
    pack_id: str
    created_at: datetime
    pack_type: str  # "backtest", "walk_forward", "live_trade", "experiment"

    # Git/Environment
    git_commit: str
    git_branch: str
    git_dirty: bool
    python_version: str
    package_versions: Dict[str, str]

    # Configuration
    config_snapshot: Dict[str, Any]
    frozen_params: Dict[str, Any]
    frozen_params_path: Optional[str] = None

    # Data
    dataset_id: str = ""
    universe_sha256: str = ""
    universe_path: Optional[str] = None
    date_range: Tuple[str, str] = ("", "")
    symbol_count: int = 0

    # Results
    metrics: Dict[str, float] = field(default_factory=dict)
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    # Artifacts (path -> sha256)
    artifacts: Dict[str, str] = field(default_factory=dict)

    # Verification
    pack_hash: str = ""

    def __post_init__(self):
        """Generate pack hash after initialization if not set."""
        if not self.pack_hash:
            self.pack_hash = self._compute_pack_hash()

    def _compute_pack_hash(self) -> str:
        """Compute deterministic hash of pack contents."""
        # Create deterministic string representation
        hash_input = json.dumps({
            "pack_id": self.pack_id,
            "pack_type": self.pack_type,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "config_snapshot": self.config_snapshot,
            "frozen_params": self.frozen_params,
            "dataset_id": self.dataset_id,
            "universe_sha256": self.universe_sha256,
            "date_range": list(self.date_range),
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }, sort_keys=True)

        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def generate_reproduce_script(self) -> str:
        """
        Generate a bash script that can reproduce this result.

        Returns:
            Shell script content as string
        """
        lines = [
            "#!/bin/bash",
            "# Auto-generated reproduction script",
            f"# Evidence Pack: {self.pack_id}",
            f"# Created: {self.created_at.isoformat()}",
            f"# Pack Hash: {self.pack_hash}",
            "",
            "set -e  # Exit on error",
            "",
            "# Verify git state",
            f'EXPECTED_COMMIT="{self.git_commit}"',
            'CURRENT_COMMIT=$(git rev-parse HEAD)',
            'if [ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]; then',
            '    echo "WARNING: Git commit mismatch!"',
            '    echo "  Expected: $EXPECTED_COMMIT"',
            '    echo "  Current:  $CURRENT_COMMIT"',
            '    read -p "Continue anyway? (y/n) " -n 1 -r',
            '    echo',
            '    if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi',
            'fi',
            "",
        ]

        if self.git_dirty:
            lines.extend([
                "# WARNING: Original run had uncommitted changes",
                'echo "WARNING: Original run had uncommitted changes"',
                "",
            ])

        # Add reproduction command based on pack type
        if self.pack_type == "backtest":
            cmd = self._generate_backtest_command()
        elif self.pack_type == "walk_forward":
            cmd = self._generate_walkforward_command()
        else:
            cmd = "# Custom reproduction command needed"

        lines.extend([
            "# Run reproduction",
            cmd,
            "",
            "# Verify results",
            f'echo "Expected metrics:"',
        ])

        for key, value in self.metrics.items():
            lines.append(f'echo "  {key}: {value}"')

        return "\n".join(lines)

    def _generate_backtest_command(self) -> str:
        """Generate backtest reproduction command."""
        cmd_parts = ["python scripts/backtest_dual_strategy.py"]

        if self.universe_path:
            cmd_parts.append(f"--universe {self.universe_path}")
        if self.date_range[0]:
            cmd_parts.append(f"--start {self.date_range[0]}")
        if self.date_range[1]:
            cmd_parts.append(f"--end {self.date_range[1]}")
        if self.frozen_params_path:
            cmd_parts.append(f"--params {self.frozen_params_path}")

        return " \\\n    ".join(cmd_parts)

    def _generate_walkforward_command(self) -> str:
        """Generate walk-forward reproduction command."""
        cmd_parts = ["python scripts/run_wf_polygon.py"]

        if self.universe_path:
            cmd_parts.append(f"--universe {self.universe_path}")
        if self.date_range[0]:
            cmd_parts.append(f"--start {self.date_range[0]}")
        if self.date_range[1]:
            cmd_parts.append(f"--end {self.date_range[1]}")

        return " \\\n    ".join(cmd_parts)

    def save(self, output_dir: Path) -> Path:
        """
        Save evidence pack to disk.

        Args:
            output_dir: Directory to save the pack

        Returns:
            Path to the saved pack file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with pack_id and timestamp
        timestamp = self.created_at.strftime("%Y%m%d_%H%M%S")
        filename = f"evidence_pack_{self.pack_type}_{timestamp}_{self.pack_hash}.json"
        filepath = output_dir / filename

        # Serialize to JSON
        pack_dict = {
            "pack_id": self.pack_id,
            "created_at": self.created_at.isoformat(),
            "pack_type": self.pack_type,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "git_dirty": self.git_dirty,
            "python_version": self.python_version,
            "package_versions": self.package_versions,
            "config_snapshot": self.config_snapshot,
            "frozen_params": self.frozen_params,
            "frozen_params_path": self.frozen_params_path,
            "dataset_id": self.dataset_id,
            "universe_sha256": self.universe_sha256,
            "universe_path": self.universe_path,
            "date_range": list(self.date_range),
            "symbol_count": self.symbol_count,
            "metrics": self.metrics,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "artifacts": self.artifacts,
            "pack_hash": self.pack_hash,
        }

        filepath.write_text(json.dumps(pack_dict, indent=2), encoding="utf-8")

        # Save reproduction script
        script_path = output_dir / f"reproduce_{self.pack_hash}.sh"
        script_path.write_text(self.generate_reproduce_script(), encoding="utf-8")

        jlog("evidence_pack_saved", pack_id=self.pack_id, path=str(filepath))

        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "EvidencePack":
        """
        Load evidence pack from disk.

        Args:
            filepath: Path to the pack JSON file

        Returns:
            EvidencePack instance
        """
        filepath = Path(filepath)
        data = json.loads(filepath.read_text(encoding="utf-8"))

        return cls(
            pack_id=data["pack_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            pack_type=data["pack_type"],
            git_commit=data["git_commit"],
            git_branch=data["git_branch"],
            git_dirty=data["git_dirty"],
            python_version=data["python_version"],
            package_versions=data["package_versions"],
            config_snapshot=data["config_snapshot"],
            frozen_params=data["frozen_params"],
            frozen_params_path=data.get("frozen_params_path"),
            dataset_id=data.get("dataset_id", ""),
            universe_sha256=data.get("universe_sha256", ""),
            universe_path=data.get("universe_path"),
            date_range=tuple(data.get("date_range", ["", ""])),
            symbol_count=data.get("symbol_count", 0),
            metrics=data.get("metrics", {}),
            total_trades=data.get("total_trades", 0),
            win_rate=data.get("win_rate", 0.0),
            profit_factor=data.get("profit_factor", 0.0),
            sharpe_ratio=data.get("sharpe_ratio", 0.0),
            max_drawdown=data.get("max_drawdown", 0.0),
            artifacts=data.get("artifacts", {}),
            pack_hash=data.get("pack_hash", ""),
        )

    def verify_hash(self) -> bool:
        """
        Verify that the pack hasn't been tampered with.

        Returns:
            True if hash matches, False otherwise
        """
        computed = self._compute_pack_hash()
        return computed == self.pack_hash

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pack_id": self.pack_id,
            "created_at": self.created_at.isoformat(),
            "pack_type": self.pack_type,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "git_dirty": self.git_dirty,
            "python_version": self.python_version,
            "config_snapshot": self.config_snapshot,
            "frozen_params": self.frozen_params,
            "dataset_id": self.dataset_id,
            "universe_sha256": self.universe_sha256,
            "date_range": list(self.date_range),
            "metrics": self.metrics,
            "total_trades": self.total_trades,
            "pack_hash": self.pack_hash,
        }


class EvidencePackBuilder:
    """
    Builder for creating EvidencePack instances.

    Provides a fluent interface for assembling evidence packs
    with proper validation at each step.
    """

    def __init__(self, pack_type: str = "backtest"):
        """
        Initialize the builder.

        Args:
            pack_type: Type of evidence pack (backtest, walk_forward, live_trade, experiment)
        """
        self.pack_type = pack_type
        self.created_at = datetime.utcnow()

        # Generate unique pack ID with high entropy
        timestamp = self.created_at.strftime("%Y%m%d_%H%M%S")
        # Use multiple entropy sources for uniqueness
        import uuid
        entropy = f"{timestamp}{os.getpid()}{id(self)}{uuid.uuid4().hex}"
        random_suffix = hashlib.sha256(entropy.encode()).hexdigest()[:8]
        self.pack_id = f"{pack_type}_{timestamp}_{random_suffix}"

        # Initialize all fields
        self.git_commit = ""
        self.git_branch = ""
        self.git_dirty = False
        self.python_version = ""
        self.package_versions: Dict[str, str] = {}
        self.config_snapshot: Dict[str, Any] = {}
        self.frozen_params: Dict[str, Any] = {}
        self.frozen_params_path: Optional[str] = None
        self.dataset_id = ""
        self.universe_sha256 = ""
        self.universe_path: Optional[str] = None
        self.date_range: Tuple[str, str] = ("", "")
        self.symbol_count = 0
        self.metrics: Dict[str, float] = {}
        self.total_trades = 0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.artifacts: Dict[str, str] = {}

    def capture_git_state(self) -> "EvidencePackBuilder":
        """
        Capture current git state.

        Returns:
            Self for chaining
        """
        try:
            # Get current commit
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            self.git_commit = result.stdout.strip()

            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            self.git_branch = result.stdout.strip()

            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True
            )
            self.git_dirty = bool(result.stdout.strip())

        except subprocess.CalledProcessError as e:
            jlog("evidence_pack_git_error", error=str(e), level="WARNING")
            self.git_commit = "unknown"
            self.git_branch = "unknown"
            self.git_dirty = True
        except FileNotFoundError:
            # Git not installed
            self.git_commit = "git_not_available"
            self.git_branch = "unknown"
            self.git_dirty = True

        return self

    def capture_environment(self) -> "EvidencePackBuilder":
        """
        Capture Python environment details.

        Returns:
            Self for chaining
        """
        import sys
        self.python_version = sys.version

        # Capture key package versions
        key_packages = [
            "pandas", "numpy", "scipy", "scikit-learn",
            "alpaca-py", "polygon-api-client",
            "pydantic", "pandera", "pytest"
        ]

        for pkg in key_packages:
            try:
                import importlib.metadata
                self.package_versions[pkg] = importlib.metadata.version(pkg)
            except Exception:
                self.package_versions[pkg] = "unknown"

        return self

    def set_config(self, config: Dict[str, Any]) -> "EvidencePackBuilder":
        """
        Set configuration snapshot.

        Args:
            config: Configuration dictionary

        Returns:
            Self for chaining
        """
        self.config_snapshot = config
        return self

    def set_frozen_params(
        self,
        params: Dict[str, Any],
        path: Optional[str] = None
    ) -> "EvidencePackBuilder":
        """
        Set frozen strategy parameters.

        Args:
            params: Frozen parameters dictionary
            path: Optional path to frozen params file

        Returns:
            Self for chaining
        """
        self.frozen_params = params
        self.frozen_params_path = path
        return self

    def load_frozen_params_from_file(self, filepath: Path) -> "EvidencePackBuilder":
        """
        Load frozen params from file.

        Args:
            filepath: Path to frozen params JSON file

        Returns:
            Self for chaining
        """
        filepath = Path(filepath)
        if filepath.exists():
            self.frozen_params = json.loads(filepath.read_text(encoding="utf-8"))
            self.frozen_params_path = str(filepath)
        return self

    def set_dataset(
        self,
        dataset_id: str,
        universe_sha256: str,
        start: str,
        end: str,
        universe_path: Optional[str] = None,
        symbol_count: int = 0
    ) -> "EvidencePackBuilder":
        """
        Set dataset information.

        Args:
            dataset_id: Unique dataset identifier
            universe_sha256: Hash of universe file
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            universe_path: Optional path to universe file
            symbol_count: Number of symbols in universe

        Returns:
            Self for chaining
        """
        self.dataset_id = dataset_id
        self.universe_sha256 = universe_sha256
        self.date_range = (start, end)
        self.universe_path = universe_path
        self.symbol_count = symbol_count
        return self

    def compute_universe_hash(self, universe_path: Path) -> "EvidencePackBuilder":
        """
        Compute hash of universe file.

        Args:
            universe_path: Path to universe CSV

        Returns:
            Self for chaining
        """
        universe_path = Path(universe_path)
        if universe_path.exists():
            content = universe_path.read_bytes()
            self.universe_sha256 = hashlib.sha256(content).hexdigest()
            self.universe_path = str(universe_path)

            # Count symbols
            try:
                import pandas as pd
                df = pd.read_csv(universe_path)
                self.symbol_count = len(df)
            except Exception:
                pass

        return self

    def set_metrics(self, metrics: Dict[str, float]) -> "EvidencePackBuilder":
        """
        Set result metrics.

        Args:
            metrics: Dictionary of metric name -> value

        Returns:
            Self for chaining
        """
        self.metrics = metrics

        # Extract common metrics if present
        self.win_rate = metrics.get("win_rate", 0.0)
        self.profit_factor = metrics.get("profit_factor", 0.0)
        self.sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
        self.max_drawdown = metrics.get("max_drawdown", 0.0)
        self.total_trades = int(metrics.get("total_trades", 0))

        return self

    def add_artifact(self, name: str, filepath: Path) -> "EvidencePackBuilder":
        """
        Add an artifact file with its hash.

        Args:
            name: Artifact name (e.g., "trade_list", "equity_curve")
            filepath: Path to the artifact file

        Returns:
            Self for chaining
        """
        filepath = Path(filepath)
        if filepath.exists():
            content = filepath.read_bytes()
            file_hash = hashlib.sha256(content).hexdigest()
            self.artifacts[name] = f"{filepath}:{file_hash}"
        return self

    def build(self) -> EvidencePack:
        """
        Build the final EvidencePack.

        Returns:
            Complete EvidencePack instance
        """
        return EvidencePack(
            pack_id=self.pack_id,
            created_at=self.created_at,
            pack_type=self.pack_type,
            git_commit=self.git_commit,
            git_branch=self.git_branch,
            git_dirty=self.git_dirty,
            python_version=self.python_version,
            package_versions=self.package_versions,
            config_snapshot=self.config_snapshot,
            frozen_params=self.frozen_params,
            frozen_params_path=self.frozen_params_path,
            dataset_id=self.dataset_id,
            universe_sha256=self.universe_sha256,
            universe_path=self.universe_path,
            date_range=self.date_range,
            symbol_count=self.symbol_count,
            metrics=self.metrics,
            total_trades=self.total_trades,
            win_rate=self.win_rate,
            profit_factor=self.profit_factor,
            sharpe_ratio=self.sharpe_ratio,
            max_drawdown=self.max_drawdown,
            artifacts=self.artifacts,
        )


def create_backtest_evidence_pack(
    universe_path: Path,
    start_date: str,
    end_date: str,
    frozen_params_path: Optional[Path] = None,
    metrics: Optional[Dict[str, float]] = None,
    trade_list_path: Optional[Path] = None,
    equity_curve_path: Optional[Path] = None,
) -> EvidencePack:
    """
    Convenience function to create a backtest evidence pack.

    Args:
        universe_path: Path to universe CSV
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        frozen_params_path: Optional path to frozen params
        metrics: Optional metrics dictionary
        trade_list_path: Optional path to trade list CSV
        equity_curve_path: Optional path to equity curve CSV

    Returns:
        Complete EvidencePack
    """
    builder = EvidencePackBuilder(pack_type="backtest")

    # Capture environment
    builder.capture_git_state()
    builder.capture_environment()

    # Set data info
    builder.compute_universe_hash(universe_path)
    builder.date_range = (start_date, end_date)

    # Load frozen params if provided
    if frozen_params_path:
        builder.load_frozen_params_from_file(frozen_params_path)

    # Set metrics
    if metrics:
        builder.set_metrics(metrics)

    # Add artifacts
    if trade_list_path:
        builder.add_artifact("trade_list", trade_list_path)
    if equity_curve_path:
        builder.add_artifact("equity_curve", equity_curve_path)

    return builder.build()


def create_walkforward_evidence_pack(
    universe_path: Path,
    start_date: str,
    end_date: str,
    train_days: int,
    test_days: int,
    metrics: Optional[Dict[str, float]] = None,
    output_dir: Optional[Path] = None,
) -> EvidencePack:
    """
    Convenience function to create a walk-forward evidence pack.

    Args:
        universe_path: Path to universe CSV
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        train_days: Training window size
        test_days: Test window size
        metrics: Optional metrics dictionary
        output_dir: Optional output directory for artifacts

    Returns:
        Complete EvidencePack
    """
    builder = EvidencePackBuilder(pack_type="walk_forward")

    # Capture environment
    builder.capture_git_state()
    builder.capture_environment()

    # Set data info
    builder.compute_universe_hash(universe_path)
    builder.date_range = (start_date, end_date)

    # Add walk-forward specific config
    builder.config_snapshot["train_days"] = train_days
    builder.config_snapshot["test_days"] = test_days

    # Set metrics
    if metrics:
        builder.set_metrics(metrics)

    # Add artifacts if output dir provided
    if output_dir:
        output_dir = Path(output_dir)
        summary_path = output_dir / "wf_summary.csv"
        if summary_path.exists():
            builder.add_artifact("wf_summary", summary_path)

    return builder.build()
