"""
Pattern Library - Storage and retrieval of discovered patterns.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from .clustering import PatternCluster

logger = logging.getLogger(__name__)


class PatternLibrary:
    """
    Persistent storage for discovered patterns.

    Stores patterns in JSON format with metadata for versioning
    and retrieval by various criteria.
    """

    def __init__(self, library_path: str = "state/pattern_library"):
        """
        Initialize pattern library.

        Args:
            library_path: Path to library directory
        """
        self.library_path = Path(library_path)
        self.library_path.mkdir(parents=True, exist_ok=True)
        self._patterns: Dict[str, PatternCluster] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all patterns from disk."""
        patterns_file = self.library_path / "patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file) as f:
                    data = json.load(f)
                for p in data.get('patterns', []):
                    cluster = PatternCluster.from_dict(p)
                    self._patterns[cluster.cluster_id] = cluster
                logger.info(f"Loaded {len(self._patterns)} patterns from library")
            except Exception as e:
                logger.warning(f"Error loading pattern library: {e}")

    def _save_all(self) -> None:
        """Save all patterns to disk."""
        patterns_file = self.library_path / "patterns.json"
        data = {
            'version': '1.0',
            'updated_at': datetime.utcnow().isoformat(),
            'patterns': [p.to_dict() for p in self._patterns.values()],
        }
        with open(patterns_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved {len(self._patterns)} patterns to library")

    def add(self, pattern: PatternCluster) -> None:
        """Add or update a pattern in the library."""
        self._patterns[pattern.cluster_id] = pattern
        self._save_all()

    def add_many(self, patterns: List[PatternCluster]) -> None:
        """Add multiple patterns to the library."""
        for p in patterns:
            self._patterns[p.cluster_id] = p
        self._save_all()

    def get(self, cluster_id: str) -> Optional[PatternCluster]:
        """Get a pattern by ID."""
        return self._patterns.get(cluster_id)

    def get_all(self) -> List[PatternCluster]:
        """Get all patterns."""
        return list(self._patterns.values())

    def get_high_edge(
        self,
        min_win_rate: float = 0.55,
        min_samples: int = 30,
    ) -> List[PatternCluster]:
        """Get high-edge patterns meeting criteria."""
        return [
            p for p in self._patterns.values()
            if p.win_rate >= min_win_rate and p.n_samples >= min_samples
        ]

    def remove(self, cluster_id: str) -> bool:
        """Remove a pattern from the library."""
        if cluster_id in self._patterns:
            del self._patterns[cluster_id]
            self._save_all()
            return True
        return False

    def clear(self) -> None:
        """Clear all patterns from the library."""
        self._patterns.clear()
        self._save_all()

    def count(self) -> int:
        """Get number of patterns in library."""
        return len(self._patterns)

    def get_by_regime(self, regime: str) -> List[PatternCluster]:
        """Get patterns that perform well in a specific regime."""
        return [
            p for p in self._patterns.values()
            if p.regime_distribution.get(regime, 0) >= 0.3
        ]
