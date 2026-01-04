"""
Base Pipeline class for Kobe autonomous system.

All pipelines inherit from this base class which provides:
- Common initialization
- Logging
- State management
- Result tracking
- Error handling

Author: Kobe Trading System
Version: 1.0.0
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class PipelineResult:
    """Result from a pipeline execution."""
    success: bool
    pipeline_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "pipeline_name": self.pipeline_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
            "errors": self.errors,
            "warnings": self.warnings,
            "artifacts": self.artifacts,
        }


class Pipeline(ABC):
    """Base class for all pipelines."""

    def __init__(
        self,
        universe_cap: int = 150,
        config: Optional[Dict] = None,
    ):
        """
        Initialize pipeline.

        Args:
            universe_cap: Number of stocks to process
            config: Optional configuration override
        """
        self.universe_cap = universe_cap
        self.config = config or {}

        # Setup paths
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.state_dir = self.project_root / "state"
        self.reports_dir = self.project_root / "reports"
        self.logs_dir = self.project_root / "logs"

        # Ensure directories exist
        for d in [self.state_dir, self.reports_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Track execution
        self._start_time: Optional[datetime] = None
        self._errors: List[str] = []
        self._warnings: List[str] = []
        self._artifacts: List[str] = []
        self._metrics: Dict[str, Any] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Pipeline name."""
        pass

    @abstractmethod
    def execute(self) -> bool:
        """
        Execute the pipeline.

        Returns:
            True if successful, False otherwise
        """
        pass

    def run(self) -> PipelineResult:
        """
        Run the pipeline with full lifecycle management.

        Returns:
            PipelineResult with execution details
        """
        self._start_time = datetime.utcnow()
        self._errors = []
        self._warnings = []
        self._artifacts = []
        self._metrics = {}

        self.logger.info(f"Starting {self.name} pipeline")

        try:
            success = self.execute()
        except Exception as e:
            self.logger.error(f"Pipeline failed with exception: {e}")
            self._errors.append(str(e))
            success = False

        end_time = datetime.utcnow()
        duration = (end_time - self._start_time).total_seconds()

        result = PipelineResult(
            success=success,
            pipeline_name=self.name,
            start_time=self._start_time,
            end_time=end_time,
            duration_seconds=duration,
            metrics=self._metrics,
            errors=self._errors,
            warnings=self._warnings,
            artifacts=self._artifacts,
        )

        # Log result
        self.logger.info(
            f"Pipeline {self.name} completed: success={success}, "
            f"duration={duration:.1f}s, errors={len(self._errors)}"
        )

        # Save result
        self._save_result(result)

        return result

    def _save_result(self, result: PipelineResult):
        """Save pipeline result to state."""
        results_file = self.state_dir / "pipeline_results.jsonl"
        with open(results_file, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def add_error(self, error: str):
        """Add an error message."""
        self._errors.append(error)
        self.logger.error(error)

    def add_warning(self, warning: str):
        """Add a warning message."""
        self._warnings.append(warning)
        self.logger.warning(warning)

    def add_artifact(self, path: str):
        """Add an artifact path."""
        self._artifacts.append(path)
        self.logger.info(f"Artifact created: {path}")

    def set_metric(self, name: str, value: Any):
        """Set a metric value."""
        self._metrics[name] = value

    def load_universe(self) -> List[str]:
        """Load stock universe."""
        universe_file = self.data_dir / "universe" / "optionable_liquid_900.csv"
        if not universe_file.exists():
            self.add_error(f"Universe file not found: {universe_file}")
            return []

        try:
            import pandas as pd
            df = pd.read_csv(universe_file)
            symbols = df['symbol'].tolist()[:self.universe_cap]
            return symbols
        except Exception as e:
            self.add_error(f"Failed to load universe: {e}")
            return []


def run_pipeline(name: str, **kwargs) -> PipelineResult:
    """
    Run a pipeline by name.

    Args:
        name: Pipeline name (e.g., "universe", "backtest")
        **kwargs: Arguments passed to pipeline constructor

    Returns:
        PipelineResult
    """
    from pipelines import PIPELINE_REGISTRY

    if name not in PIPELINE_REGISTRY:
        raise ValueError(f"Unknown pipeline: {name}")

    pipeline_cls = PIPELINE_REGISTRY[name]
    pipeline = pipeline_cls(**kwargs)
    return pipeline.run()
