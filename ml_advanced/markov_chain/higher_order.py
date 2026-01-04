"""
Higher-Order Markov Chains

Extends basic Markov chains to consider 2-3 prior states for predictions.
This captures multi-day patterns that first-order chains miss.

Key Insight:
- 1st order: P(Up | Down) - what happens after 1 down day
- 2nd order: P(Up | Down, Down) - what happens after 2 consecutive down days
- 3rd order: P(Up | Down, Down, Down) - what happens after 3 consecutive down days

Trading Applications:
- Bounce patterns: Down → Down → likely Up (oversold bounce)
- Momentum exhaustion: Up → Up → Up → likely Down (overextended)
- Trend continuation: Up → Up → Up → Up (strong trend)

The state space grows exponentially (3^order states), so we use:
- Composite state encoding for efficient storage
- Sparse matrix representations
- Minimum sample thresholds for reliability

Usage:
    hom = HigherOrderMarkov(order=2, n_states=3)
    hom.fit(states)

    # Predict after seeing Down, Down
    probs = hom.predict(0, 0)  # Returns [P(Down), P(Flat), P(Up)]

Created: 2026-01-04
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HigherOrderConfig:
    """Configuration for higher-order Markov chain."""

    order: int = 2  # 2nd order by default
    n_states: int = 3  # Ternary states
    smoothing: float = 0.1  # Lower smoothing for sparse matrices
    min_samples: int = 10  # Minimum samples per composite state


class HigherOrderMarkov:
    """
    Higher-order Markov chain for multi-day pattern recognition.

    A 2nd order chain considers the last 2 states:
        P(S_t+1 | S_t, S_t-1)

    Instead of 3x3 matrix, we have 9x3 matrix:
        Composite state (S_t-1, S_t) → next state S_t+1

    For trading, this captures patterns like:
    - (Down, Down) → Up: Oversold bounce (mean reversion)
    - (Up, Up) → Up: Momentum continuation
    - (Up, Down) → Down: Trend reversal confirmation

    Example:
        hom = HigherOrderMarkov(order=2, n_states=3)
        hom.fit(states)

        # What happens after 2 consecutive down days?
        probs = hom.predict(DOWN, DOWN)
        # probs might be [0.35, 0.25, 0.40] showing higher UP probability
    """

    def __init__(
        self,
        order: int = 2,
        n_states: int = 3,
        smoothing: float = 0.1,
        min_samples: int = 10,
    ):
        """
        Initialize higher-order Markov chain.

        Args:
            order: Order of the chain (2 or 3 recommended)
            n_states: Number of base states (e.g., 3 for Down/Flat/Up)
            smoothing: Laplace smoothing factor
            min_samples: Minimum samples for reliable prediction
        """
        self.config = HigherOrderConfig(
            order=order,
            n_states=n_states,
            smoothing=smoothing,
            min_samples=min_samples,
        )

        # Number of composite states = n_states^order
        self.n_composite = n_states ** order

        # Transition counts: composite_state → next_state → count
        self.counts: Dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(n_states, dtype=float)
        )

        # Total counts per composite state
        self.composite_counts: Dict[int, int] = defaultdict(int)

        # Tracking
        self._fitted = False
        self._total_transitions = 0

        logger.debug(f"HigherOrderMarkov initialized: order={order}, n_states={n_states}")

    def encode_composite_state(self, *states: int) -> int:
        """
        Encode multi-state sequence as single composite index.

        For order=2, n_states=3:
            (0, 0) → 0
            (0, 1) → 1
            (0, 2) → 2
            (1, 0) → 3
            ...
            (2, 2) → 8

        Args:
            *states: State sequence (length = order)

        Returns:
            Composite state index
        """
        if len(states) != self.config.order:
            raise ValueError(f"Expected {self.config.order} states, got {len(states)}")

        n = self.config.n_states
        composite = 0

        for i, s in enumerate(states):
            if not (0 <= s < n):
                raise ValueError(f"Invalid state {s}, must be 0-{n-1}")
            composite += s * (n ** i)

        return composite

    def decode_composite_state(self, composite: int) -> Tuple[int, ...]:
        """
        Decode composite index back to state sequence.

        Args:
            composite: Composite state index

        Returns:
            Tuple of states (length = order)
        """
        n = self.config.n_states
        order = self.config.order
        states = []

        for _ in range(order):
            states.append(composite % n)
            composite //= n

        return tuple(states)

    def fit(self, states: Union[np.ndarray, List[int]]) -> "HigherOrderMarkov":
        """
        Build higher-order transition matrix from state sequence.

        Args:
            states: Sequence of state indices

        Returns:
            Self for method chaining
        """
        states = np.asarray(states)
        order = self.config.order
        n = self.config.n_states

        if len(states) < order + 1:
            logger.warning(f"Need at least {order + 1} states, got {len(states)}")
            return self

        # Reset counts
        self.counts = defaultdict(lambda: np.zeros(n, dtype=float))
        self.composite_counts = defaultdict(int)
        self._total_transitions = 0

        # Count transitions
        for i in range(len(states) - order):
            # Get the prior `order` states
            prior_states = tuple(states[i:i + order])
            next_state = int(states[i + order])

            if all(0 <= s < n for s in prior_states) and 0 <= next_state < n:
                composite = self.encode_composite_state(*prior_states)
                self.counts[composite][next_state] += 1
                self.composite_counts[composite] += 1
                self._total_transitions += 1

        self._fitted = True
        logger.debug(f"Fitted with {self._total_transitions} transitions, "
                    f"{len(self.composite_counts)} unique composite states")

        return self

    def predict(self, *recent_states: int) -> np.ndarray:
        """
        Predict next state distribution given recent states.

        Args:
            *recent_states: Last `order` states (most recent last)

        Returns:
            Probability distribution over next states
        """
        order = self.config.order
        n = self.config.n_states
        smoothing = self.config.smoothing

        if len(recent_states) != order:
            raise ValueError(f"Expected {order} states, got {len(recent_states)}")

        composite = self.encode_composite_state(*recent_states)

        if composite in self.counts:
            counts = self.counts[composite].copy()
        else:
            # No data for this composite state - return uniform
            logger.debug(f"No data for composite state {recent_states}")
            return np.ones(n) / n

        # Add smoothing
        counts += smoothing

        # Normalize
        probs = counts / counts.sum()

        return probs

    def get_probability(self, *recent_states: int, next_state: int) -> float:
        """
        Get specific transition probability.

        Args:
            *recent_states: Last `order` states
            next_state: Target next state

        Returns:
            P(next_state | recent_states)
        """
        probs = self.predict(*recent_states)
        return float(probs[next_state])

    def most_likely_next(self, *recent_states: int) -> int:
        """
        Get most probable next state.

        Args:
            *recent_states: Last `order` states

        Returns:
            Most likely next state index
        """
        probs = self.predict(*recent_states)
        return int(np.argmax(probs))

    def is_reliable(self, *recent_states: int) -> bool:
        """
        Check if prediction for this state sequence is reliable.

        Args:
            *recent_states: State sequence

        Returns:
            True if enough samples for reliable prediction
        """
        composite = self.encode_composite_state(*recent_states)
        return self.composite_counts.get(composite, 0) >= self.config.min_samples

    def get_bounce_probability(
        self,
        down_state: int = 0,
        up_state: int = 2,
        consecutive_downs: int = 2,
    ) -> float:
        """
        Get probability of bouncing UP after consecutive DOWN days.

        This is a key mean-reversion signal:
        - After N down days, what's the probability of an up day?

        Args:
            down_state: Index of DOWN state
            up_state: Index of UP state
            consecutive_downs: Number of consecutive down days

        Returns:
            P(UP | consecutive DOWNs)
        """
        order = self.config.order

        if consecutive_downs > order:
            logger.warning(f"consecutive_downs ({consecutive_downs}) > order ({order})")
            consecutive_downs = order

        # Build state sequence of consecutive downs
        states = tuple([down_state] * order)

        return self.get_probability(*states, next_state=up_state)

    def get_momentum_continuation(
        self,
        up_state: int = 2,
        consecutive_ups: int = 2,
    ) -> float:
        """
        Get probability of continuing UP after consecutive UP days.

        Measures trend persistence:
        - After N up days, what's the probability of another up day?

        Args:
            up_state: Index of UP state
            consecutive_ups: Number of consecutive up days

        Returns:
            P(UP | consecutive UPs)
        """
        order = self.config.order

        if consecutive_ups > order:
            consecutive_ups = order

        states = tuple([up_state] * order)
        return self.get_probability(*states, next_state=up_state)

    def get_pattern_stats(
        self,
        state_names: Optional[Dict[int, str]] = None,
    ) -> pd.DataFrame:
        """
        Get statistics for all observed patterns.

        Returns DataFrame with:
        - pattern: State sequence (e.g., "DOWN→DOWN")
        - samples: Number of observations
        - prob_up: P(UP | pattern)
        - prob_down: P(DOWN | pattern)
        - most_likely: Most likely next state

        Args:
            state_names: Optional mapping of state index to name

        Returns:
            DataFrame of pattern statistics
        """
        if state_names is None:
            state_names = {0: "DOWN", 1: "FLAT", 2: "UP"}

        results = []

        for composite, count in self.composite_counts.items():
            if count < self.config.min_samples:
                continue

            states = self.decode_composite_state(composite)
            probs = self.predict(*states)

            pattern = "→".join(state_names.get(s, f"S{s}") for s in states)

            results.append({
                "pattern": pattern,
                "samples": count,
                "prob_up": probs[2] if len(probs) > 2 else probs[-1],
                "prob_down": probs[0],
                "most_likely": state_names.get(np.argmax(probs), f"S{np.argmax(probs)}"),
                "confidence": np.max(probs),
            })

        df = pd.DataFrame(results)

        if not df.empty:
            df = df.sort_values("samples", ascending=False)

        return df

    def find_high_probability_patterns(
        self,
        target_state: int = 2,
        min_probability: float = 0.5,
        min_samples: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Find patterns with high probability of reaching target state.

        For trading: Find sequences that often lead to UP days.

        Args:
            target_state: Target state (default 2 = UP)
            min_probability: Minimum probability threshold
            min_samples: Minimum samples for reliability

        Returns:
            List of high-probability patterns with stats
        """
        patterns = []

        for composite, count in self.composite_counts.items():
            if count < min_samples:
                continue

            states = self.decode_composite_state(composite)
            probs = self.predict(*states)

            if probs[target_state] >= min_probability:
                patterns.append({
                    "states": states,
                    "probability": float(probs[target_state]),
                    "samples": count,
                    "all_probs": probs.tolist(),
                })

        # Sort by probability
        patterns.sort(key=lambda x: x["probability"], reverse=True)

        return patterns

    @property
    def order(self) -> int:
        """Order of the chain."""
        return self.config.order

    @property
    def n_states(self) -> int:
        """Number of base states."""
        return self.config.n_states

    @property
    def is_fitted(self) -> bool:
        """Whether chain has been fitted."""
        return self._fitted

    @property
    def total_transitions(self) -> int:
        """Total observed transitions."""
        return self._total_transitions

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "order": self.config.order,
            "n_states": self.config.n_states,
            "smoothing": self.config.smoothing,
            "min_samples": self.config.min_samples,
            "counts": {k: v.tolist() for k, v in self.counts.items()},
            "composite_counts": dict(self.composite_counts),
            "total_transitions": self._total_transitions,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "HigherOrderMarkov":
        """Deserialize from dictionary."""
        hom = cls(
            order=data["order"],
            n_states=data["n_states"],
            smoothing=data.get("smoothing", 0.1),
            min_samples=data.get("min_samples", 10),
        )

        for k, v in data.get("counts", {}).items():
            hom.counts[int(k)] = np.array(v)

        hom.composite_counts = defaultdict(int, {
            int(k): v for k, v in data.get("composite_counts", {}).items()
        })

        hom._total_transitions = data.get("total_transitions", 0)
        hom._fitted = data.get("fitted", False)

        return hom


def find_consecutive_patterns(
    returns: pd.Series,
    n_states: int = 3,
    max_order: int = 5,
) -> pd.DataFrame:
    """
    Analyze what happens after consecutive up/down days.

    Returns a summary of bounce and continuation probabilities
    for 1, 2, 3, 4, 5 consecutive days.

    Args:
        returns: Daily returns series
        n_states: Number of states
        max_order: Maximum consecutive days to analyze

    Returns:
        DataFrame with bounce/continuation probabilities
    """
    from .state_classifier import StateClassifier

    # Classify returns
    classifier = StateClassifier(n_states=n_states, method="threshold")
    classifier.fit(returns)
    states = classifier.classify(returns)

    results = []

    for order in range(1, max_order + 1):
        hom = HigherOrderMarkov(order=order, n_states=n_states)
        hom.fit(states)

        # Bounce probability (DOWN...DOWN → UP)
        down_seq = tuple([0] * order)
        if hom.composite_counts.get(hom.encode_composite_state(*down_seq), 0) >= 10:
            bounce_prob = hom.get_probability(*down_seq, next_state=2)
            bounce_samples = hom.composite_counts[hom.encode_composite_state(*down_seq)]
        else:
            bounce_prob = np.nan
            bounce_samples = 0

        # Continuation probability (UP...UP → UP)
        up_seq = tuple([2] * order)
        if hom.composite_counts.get(hom.encode_composite_state(*up_seq), 0) >= 10:
            cont_prob = hom.get_probability(*up_seq, next_state=2)
            cont_samples = hom.composite_counts[hom.encode_composite_state(*up_seq)]
        else:
            cont_prob = np.nan
            cont_samples = 0

        results.append({
            "consecutive_days": order,
            "bounce_prob": bounce_prob,
            "bounce_samples": bounce_samples,
            "continuation_prob": cont_prob,
            "continuation_samples": cont_samples,
        })

    return pd.DataFrame(results)
