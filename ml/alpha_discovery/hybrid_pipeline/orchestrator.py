"""
Hybrid Pattern Pipeline - Orchestrates ML discovery to deployment.

Workflow:
1. Pattern Miner discovers clusters
2. Pattern Narrator explains them
3. Human reviews via dashboard
4. Approved patterns promoted to SemanticMemory
5. RL Agent optimizes timing
"""
from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from ..pattern_miner import PatternCluster, PatternClusteringEngine, PatternLibrary
from ..pattern_narrator import PatternNarrative, PatternPlaybook, PatternNarrator
from ..feature_discovery import FeatureImportanceReport, FeatureImportanceAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """Result of a pattern discovery run."""
    run_id: str
    started_at: datetime
    completed_at: datetime = field(default_factory=datetime.utcnow)

    # Pattern Mining results
    patterns_discovered: int = 0
    patterns_validated: int = 0
    high_edge_patterns: List[PatternCluster] = field(default_factory=list)

    # Narrative results
    narratives_generated: int = 0
    playbook: Optional[PatternPlaybook] = None

    # Feature Discovery results
    feature_report: Optional[FeatureImportanceReport] = None

    # RL Agent results
    rl_metrics: Optional[Dict[str, float]] = None

    # Promotion status
    patterns_promoted: int = 0
    patterns_pending_review: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'run_id': self.run_id,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat(),
            'patterns_discovered': self.patterns_discovered,
            'patterns_validated': self.patterns_validated,
            'high_edge_patterns': [p.to_dict() for p in self.high_edge_patterns],
            'narratives_generated': self.narratives_generated,
            'playbook': self.playbook.to_dict() if self.playbook else None,
            'feature_report': self.feature_report.to_dict() if self.feature_report else None,
            'rl_metrics': self.rl_metrics,
            'patterns_promoted': self.patterns_promoted,
            'patterns_pending_review': self.patterns_pending_review,
        }

    def save(self, path: str) -> None:
        """Save result to JSON file."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / 'discovery_result.json', 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class HybridPatternPipeline:
    """
    Orchestrates the complete pattern discovery workflow.

    Integrates:
    - Pattern Miner (ML clustering)
    - Pattern Narrator (LLM explanations)
    - Feature Discovery (importance analysis)
    - Human Review (approval workflow)
    - Pattern Library (storage)
    """

    def __init__(
        self,
        pattern_miner: Optional[PatternClusteringEngine] = None,
        narrator: Optional[PatternNarrator] = None,
        feature_analyzer: Optional[FeatureImportanceAnalyzer] = None,
        library: Optional[PatternLibrary] = None,
    ):
        """
        Initialize pipeline.

        Args:
            pattern_miner: Clustering engine (default: creates new)
            narrator: LLM narrator (default: creates new)
            feature_analyzer: Feature importance analyzer (default: creates new)
            library: Pattern library (default: creates new)
        """
        self.pattern_miner = pattern_miner
        self.narrator = narrator
        self.feature_analyzer = feature_analyzer
        self.library = library or PatternLibrary()

        # State tracking
        self.pending_review: Dict[str, Dict] = {}  # pattern_id -> {cluster, narrative}
        self.approved_patterns: List[str] = []
        self.rejected_patterns: List[str] = []

    def _ensure_components(self) -> None:
        """Lazy-initialize components."""
        if self.pattern_miner is None:
            self.pattern_miner = PatternClusteringEngine()
        if self.narrator is None:
            self.narrator = PatternNarrator()
        if self.feature_analyzer is None:
            self.feature_analyzer = FeatureImportanceAnalyzer()

    def run_discovery(
        self,
        trades_df: pd.DataFrame,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
        include_narratives: bool = True,
        include_features: bool = True,
        include_rl: bool = False,
        min_win_rate: float = 0.55,
        min_samples: int = 30,
    ) -> DiscoveryResult:
        """
        Run the complete discovery pipeline.

        Args:
            trades_df: Trade data with features
            price_data: Optional price data for context
            include_narratives: Generate LLM narratives
            include_features: Run feature importance analysis
            include_rl: Train RL agent (slow)
            min_win_rate: Minimum win rate for high-edge patterns
            min_samples: Minimum samples for high-edge patterns

        Returns:
            DiscoveryResult with all findings
        """
        self._ensure_components()

        run_id = f"discovery_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.utcnow()

        logger.info(f"Starting discovery run: {run_id}")

        result = DiscoveryResult(
            run_id=run_id,
            started_at=started_at,
        )

        # Step 1: Pattern Mining
        logger.info("Step 1: Running pattern mining...")
        all_clusters = self.pattern_miner.fit(trades_df)
        result.patterns_discovered = len(all_clusters)

        high_edge = self.pattern_miner.get_high_edge_clusters(
            min_win_rate=min_win_rate,
            min_samples=min_samples,
        )
        result.patterns_validated = len(high_edge)
        result.high_edge_patterns = high_edge

        logger.info(f"Found {len(all_clusters)} clusters, {len(high_edge)} high-edge")

        # Step 2: Generate Narratives
        if include_narratives and high_edge:
            logger.info("Step 2: Generating narratives...")
            playbook = self.narrator.build_playbook(high_edge)
            result.playbook = playbook
            result.narratives_generated = len(playbook.patterns)

            # Add to pending review
            for i, cluster in enumerate(high_edge):
                narrative = playbook.patterns[i] if i < len(playbook.patterns) else None
                self.pending_review[cluster.cluster_id] = {
                    'cluster': cluster,
                    'narrative': narrative,
                    'submitted_at': datetime.utcnow().isoformat(),
                }

            result.patterns_pending_review = len(self.pending_review)

        # Step 3: Feature Analysis
        if include_features:
            logger.info("Step 3: Running feature analysis...")
            feature_report = self.feature_analyzer.analyze(trades_df)
            result.feature_report = feature_report
            logger.info(f"Top features: {feature_report.top_10_features[:3]}")

        # Step 4: RL Agent (optional)
        if include_rl and price_data:
            logger.info("Step 4: Training RL agent...")
            from ..rl_agent import RLTradingAgent, RLAgentConfig

            config = RLAgentConfig(total_timesteps=10000)  # Short for demo
            agent = RLTradingAgent(config)
            rl_result = agent.train(price_data)
            result.rl_metrics = rl_result.get('metrics', {})

        result.completed_at = datetime.utcnow()
        logger.info(f"Discovery complete: {run_id}")

        return result

    def get_pending_reviews(self) -> List[Dict]:
        """Get patterns awaiting human review."""
        reviews = []
        for pattern_id, data in self.pending_review.items():
            cluster = data['cluster']
            narrative = data['narrative']
            reviews.append({
                'pattern_id': pattern_id,
                'win_rate': cluster.win_rate,
                'profit_factor': cluster.profit_factor,
                'n_samples': cluster.n_samples,
                'confidence': cluster.confidence,
                'title': narrative.title if narrative else f'Pattern {pattern_id}',
                'summary': narrative.executive_summary if narrative else '',
                'submitted_at': data['submitted_at'],
            })
        return reviews

    def approve_pattern(
        self,
        pattern_id: str,
        reviewer_notes: str = '',
    ) -> bool:
        """
        Approve pattern and promote to library.

        Args:
            pattern_id: Pattern cluster ID
            reviewer_notes: Optional reviewer notes

        Returns:
            True if approved successfully
        """
        if pattern_id not in self.pending_review:
            logger.warning(f"Pattern {pattern_id} not in pending review")
            return False

        data = self.pending_review.pop(pattern_id)
        cluster = data['cluster']

        # Add to library
        self.library.add(cluster)
        self.approved_patterns.append(pattern_id)

        logger.info(f"Approved pattern: {pattern_id}")
        return True

    def reject_pattern(
        self,
        pattern_id: str,
        reason: str = '',
    ) -> bool:
        """
        Reject pattern from promotion.

        Args:
            pattern_id: Pattern cluster ID
            reason: Rejection reason

        Returns:
            True if rejected successfully
        """
        if pattern_id not in self.pending_review:
            logger.warning(f"Pattern {pattern_id} not in pending review")
            return False

        self.pending_review.pop(pattern_id)
        self.rejected_patterns.append(pattern_id)

        logger.info(f"Rejected pattern: {pattern_id} - {reason}")
        return True

    def auto_approve_high_confidence(
        self,
        confidence_threshold: float = 0.75,
    ) -> int:
        """
        Auto-approve patterns above confidence threshold.

        Args:
            confidence_threshold: Minimum confidence to auto-approve

        Returns:
            Number of patterns auto-approved
        """
        approved = 0
        to_approve = []

        for pattern_id, data in self.pending_review.items():
            if data['cluster'].confidence >= confidence_threshold:
                to_approve.append(pattern_id)

        for pattern_id in to_approve:
            if self.approve_pattern(pattern_id, 'Auto-approved: high confidence'):
                approved += 1

        logger.info(f"Auto-approved {approved} patterns")
        return approved

    def get_library_stats(self) -> Dict[str, Any]:
        """Get statistics on pattern library."""
        patterns = self.library.get_all()
        if not patterns:
            return {'total': 0}

        return {
            'total': len(patterns),
            'avg_win_rate': sum(p.win_rate for p in patterns) / len(patterns),
            'avg_profit_factor': sum(p.profit_factor for p in patterns) / len(patterns),
            'total_samples': sum(p.n_samples for p in patterns),
            'pending_review': len(self.pending_review),
            'approved': len(self.approved_patterns),
            'rejected': len(self.rejected_patterns),
        }
