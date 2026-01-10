"""
Clone detector for evolution guardrails.

Detects near-duplicate strategies to prevent redundant promotions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json


@dataclass
class CloneCheckResult:
    """Result of clone detection check."""
    is_clone: bool
    similarity_score: float
    matched_fingerprint: Optional[str]
    matched_candidate_id: Optional[str]
    reason: str


class CloneDetector:
    """
    Detects clones and near-duplicates in strategy candidates.

    Uses multiple approaches:
    1. Exact fingerprint matching (params hash)
    2. Parameter similarity scoring
    3. Rule/logic similarity (if applicable)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        min_param_diff_ratio: float = 0.05,
    ):
        """
        Initialize clone detector.

        Args:
            similarity_threshold: Similarity score above which candidates are clones
            min_param_diff_ratio: Minimum parameter difference ratio to not be clone
        """
        self.similarity_threshold = similarity_threshold
        self.min_param_diff_ratio = min_param_diff_ratio

    def compute_params_fingerprint(self, params: Dict[str, Any]) -> str:
        """Compute fingerprint hash from parameters."""
        params_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(params_str.encode()).hexdigest()

    def compute_params_similarity(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any],
    ) -> float:
        """
        Compute similarity score between two parameter sets.

        Returns value between 0 (completely different) and 1 (identical).
        """
        if not params1 or not params2:
            return 0.0 if params1 != params2 else 1.0

        all_keys = set(params1.keys()) | set(params2.keys())
        if not all_keys:
            return 1.0

        matching_score = 0.0
        total_weight = 0.0

        for key in all_keys:
            weight = 1.0  # Could weight by parameter importance
            total_weight += weight

            if key not in params1 or key not in params2:
                # Missing key - no match
                continue

            v1 = params1[key]
            v2 = params2[key]

            # Compute match for this parameter
            param_sim = self._compute_value_similarity(v1, v2)
            matching_score += weight * param_sim

        return matching_score / total_weight if total_weight > 0 else 0.0

    def _compute_value_similarity(self, v1: Any, v2: Any) -> float:
        """Compute similarity between two values."""
        if v1 == v2:
            return 1.0

        if type(v1) != type(v2):
            return 0.0

        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            # Numeric values - check relative difference
            if v1 == 0 and v2 == 0:
                return 1.0
            max_val = max(abs(v1), abs(v2))
            if max_val == 0:
                return 1.0
            diff_ratio = abs(v1 - v2) / max_val
            return max(0.0, 1.0 - diff_ratio)

        if isinstance(v1, str) and isinstance(v2, str):
            # String values - exact match or nothing
            return 1.0 if v1 == v2 else 0.0

        if isinstance(v1, bool) and isinstance(v2, bool):
            return 1.0 if v1 == v2 else 0.0

        if isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
            if len(v1) != len(v2):
                return 0.0
            if len(v1) == 0:
                return 1.0
            return sum(
                self._compute_value_similarity(a, b)
                for a, b in zip(v1, v2)
            ) / len(v1)

        if isinstance(v1, dict) and isinstance(v2, dict):
            return self.compute_params_similarity(v1, v2)

        return 0.0

    def check_clone(
        self,
        strategy_name: str,
        params: Dict[str, Any],
        existing_fingerprints: Dict[str, Tuple[str, Dict[str, Any]]],
    ) -> CloneCheckResult:
        """
        Check if a candidate is a clone of existing candidates.

        Args:
            strategy_name: Name of the strategy
            params: Parameters of the candidate
            existing_fingerprints: Dict of fingerprint -> (candidate_id, params)

        Returns:
            CloneCheckResult
        """
        # Compute fingerprint
        fingerprint = self.compute_params_fingerprint(params)

        # Check exact match first
        for fp, (cand_id, existing_params) in existing_fingerprints.items():
            if fp == fingerprint:
                return CloneCheckResult(
                    is_clone=True,
                    similarity_score=1.0,
                    matched_fingerprint=fp,
                    matched_candidate_id=cand_id,
                    reason="Exact parameter match (identical fingerprint)",
                )

        # Check near-duplicates
        best_match: Optional[Tuple[str, str, float]] = None

        for fp, (cand_id, existing_params) in existing_fingerprints.items():
            similarity = self.compute_params_similarity(params, existing_params)

            if similarity >= self.similarity_threshold:
                if best_match is None or similarity > best_match[2]:
                    best_match = (fp, cand_id, similarity)

        if best_match:
            fp, cand_id, score = best_match
            return CloneCheckResult(
                is_clone=True,
                similarity_score=score,
                matched_fingerprint=fp,
                matched_candidate_id=cand_id,
                reason=f"Near-duplicate ({score:.1%} similar, threshold {self.similarity_threshold:.1%})",
            )

        # Not a clone
        return CloneCheckResult(
            is_clone=False,
            similarity_score=0.0,
            matched_fingerprint=None,
            matched_candidate_id=None,
            reason="No clone detected",
        )

    def check_clone_batch(
        self,
        candidates: List[Tuple[str, Dict[str, Any]]],
    ) -> Dict[int, List[int]]:
        """
        Check for clones within a batch of candidates.

        Args:
            candidates: List of (strategy_name, params) tuples

        Returns:
            Dict mapping candidate index to list of clone indices
        """
        clones: Dict[int, List[int]] = {}

        for i, (name1, params1) in enumerate(candidates):
            clones[i] = []
            for j, (name2, params2) in enumerate(candidates):
                if i >= j:
                    continue  # Only check pairs once

                if name1 != name2:
                    continue  # Different strategies can't be clones

                similarity = self.compute_params_similarity(params1, params2)
                if similarity >= self.similarity_threshold:
                    clones[i].append(j)
                    if j not in clones:
                        clones[j] = []
                    clones[j].append(i)

        return clones

    def filter_clones(
        self,
        candidates: List[Tuple[str, Dict[str, Any], float]],
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Filter out clones from a list of candidates, keeping best performers.

        Args:
            candidates: List of (strategy_name, params, score) tuples

        Returns:
            Filtered list with clones removed (keeping highest score)
        """
        if len(candidates) <= 1:
            return candidates

        # Sort by score descending
        sorted_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)

        # Keep track of fingerprints we've seen
        seen_fingerprints: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        filtered = []

        for name, params, score in sorted_candidates:
            result = self.check_clone(name, params, seen_fingerprints)

            if not result.is_clone:
                # Not a clone - keep it
                fingerprint = self.compute_params_fingerprint(params)
                seen_fingerprints[fingerprint] = (f"idx_{len(filtered)}", params)
                filtered.append((name, params, score))

        return filtered

    def compute_diversity_score(
        self,
        candidates: List[Dict[str, Any]],
    ) -> float:
        """
        Compute diversity score for a set of candidates.

        Higher score = more diverse parameter combinations.
        Range: 0 (all identical) to 1 (all maximally different).
        """
        if len(candidates) <= 1:
            return 1.0

        total_pairs = 0
        total_difference = 0.0

        for i, params1 in enumerate(candidates):
            for j, params2 in enumerate(candidates):
                if i >= j:
                    continue
                total_pairs += 1
                similarity = self.compute_params_similarity(params1, params2)
                total_difference += (1 - similarity)

        if total_pairs == 0:
            return 1.0

        return total_difference / total_pairs
