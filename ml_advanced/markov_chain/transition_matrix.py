"""
Transition Matrix Builder for Markov Chains

Builds and maintains the core transition probability matrix P where:
    P[i,j] = P(next_state = j | current_state = i)
    Each row sums to 1.

Features:
- Laplace smoothing for rare transitions
- Incremental updates for online learning
- Validation and normalization utilities
- Persistence (save/load)

Usage:
    tm = TransitionMatrix(n_states=3)
    tm.fit(states)

    # Get specific probability
    p_up_given_down = tm.get_probability(from_state=0, to_state=2)

    # Predict next state
    probs = tm.predict_next(current_state=1)  # [P(DOWN), P(FLAT), P(UP)]

    # Incremental update
    tm.update(from_state=0, to_state=2)  # Observed DOWN -> UP

Created: 2026-01-04
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TransitionMatrixConfig:
    """Configuration for transition matrix."""

    n_states: int = 3
    smoothing: float = 1.0  # Laplace smoothing (pseudocounts)
    min_samples: int = 30  # Minimum samples for reliable matrix


class TransitionMatrix:
    """
    Build and maintain transition probability matrix for Markov chain.

    The transition matrix P is the heart of the Markov chain:
    - P[i,j] = P(S_{t+1} = j | S_t = i)
    - Each row i represents the probability distribution of next states
      given that we're currently in state i
    - Rows sum to 1.0

    Example for 3 states (DOWN=0, FLAT=1, UP=2):
        P = [[0.4, 0.3, 0.3],   # From DOWN: 40% stay DOWN, 30% FLAT, 30% UP
             [0.3, 0.4, 0.3],   # From FLAT: 30% DOWN, 40% stay FLAT, 30% UP
             [0.2, 0.3, 0.5]]   # From UP: 20% DOWN, 30% FLAT, 50% stay UP

    The matrix tells us that stocks in UP state tend to stay UP (50%),
    while stocks in DOWN state have equal chance of staying or reversing.
    """

    def __init__(
        self,
        n_states: int = 3,
        smoothing: float = 1.0,
        min_samples: int = 30,
    ):
        """
        Initialize transition matrix.

        Args:
            n_states: Number of discrete states
            smoothing: Laplace smoothing factor (add to counts)
                - 0: No smoothing (may have zero probabilities)
                - 1: Standard Laplace smoothing
                - >1: Stronger prior toward uniform
            min_samples: Minimum samples to consider matrix reliable
        """
        self.config = TransitionMatrixConfig(
            n_states=n_states,
            smoothing=smoothing,
            min_samples=min_samples,
        )

        # Raw transition counts
        self.counts = np.zeros((n_states, n_states), dtype=float)

        # Normalized probability matrix
        self._matrix: Optional[np.ndarray] = None

        # Tracking
        self._total_transitions = 0
        self._fitted = False

        logger.debug(f"TransitionMatrix initialized: {n_states} states, smoothing={smoothing}")

    def fit(self, states: Union[np.ndarray, List[int]]) -> "TransitionMatrix":
        """
        Build transition matrix from sequence of states.

        Args:
            states: Sequence of state indices [s_0, s_1, s_2, ...]

        Returns:
            Self for method chaining
        """
        states = np.asarray(states)

        if len(states) < 2:
            logger.warning("Need at least 2 states to compute transitions")
            return self

        # Reset counts
        self.counts.fill(0)

        # Count transitions
        for i in range(len(states) - 1):
            from_state = int(states[i])
            to_state = int(states[i + 1])

            if 0 <= from_state < self.config.n_states and 0 <= to_state < self.config.n_states:
                self.counts[from_state, to_state] += 1

        self._total_transitions = int(self.counts.sum())

        # Compute probability matrix
        self._compute_matrix()
        self._fitted = True

        logger.debug(f"Fitted with {self._total_transitions} transitions")
        return self

    def _compute_matrix(self) -> None:
        """Compute normalized probability matrix with smoothing."""
        n = self.config.n_states
        s = self.config.smoothing

        # Add smoothing (Laplace/additive smoothing)
        smoothed = self.counts + s

        # Normalize rows to sum to 1
        row_sums = smoothed.sum(axis=1, keepdims=True)

        # Handle zero rows (states never observed as source)
        # Set to uniform distribution for unobserved source states
        zero_rows = (row_sums == 0).flatten()
        row_sums = np.where(row_sums == 0, 1, row_sums)

        self._matrix = smoothed / row_sums

        # Set zero rows to uniform distribution
        if np.any(zero_rows):
            self._matrix[zero_rows] = 1.0 / n

        # Verify normalization
        assert np.allclose(self._matrix.sum(axis=1), 1.0), "Rows must sum to 1"

    def update(self, from_state: int, to_state: int, count: int = 1) -> None:
        """
        Incrementally update matrix with new transition(s).

        For online learning: update as new data arrives without
        reprocessing entire history.

        Args:
            from_state: Source state index
            to_state: Destination state index
            count: Number of times this transition occurred
        """
        if not (0 <= from_state < self.config.n_states):
            raise ValueError(f"Invalid from_state: {from_state}")
        if not (0 <= to_state < self.config.n_states):
            raise ValueError(f"Invalid to_state: {to_state}")

        self.counts[from_state, to_state] += count
        self._total_transitions += count

        # Recompute matrix
        self._compute_matrix()
        self._fitted = True

    def batch_update(self, transitions: List[Tuple[int, int]]) -> None:
        """
        Update with multiple transitions at once.

        Args:
            transitions: List of (from_state, to_state) tuples
        """
        for from_state, to_state in transitions:
            if 0 <= from_state < self.config.n_states and 0 <= to_state < self.config.n_states:
                self.counts[from_state, to_state] += 1

        self._total_transitions += len(transitions)
        self._compute_matrix()
        self._fitted = True

    @property
    def matrix(self) -> np.ndarray:
        """
        Get the transition probability matrix.

        Returns:
            n_states x n_states probability matrix
        """
        if self._matrix is None:
            # Return uniform if not fitted
            n = self.config.n_states
            return np.ones((n, n)) / n
        return self._matrix.copy()

    def get_probability(self, from_state: int, to_state: int) -> float:
        """
        Get specific transition probability P(to_state | from_state).

        Args:
            from_state: Current state index
            to_state: Next state index

        Returns:
            Probability (0 to 1)
        """
        if self._matrix is None:
            return 1.0 / self.config.n_states  # Uniform prior

        return float(self._matrix[from_state, to_state])

    def predict_next(self, current_state: int) -> np.ndarray:
        """
        Get probability distribution for next state.

        Args:
            current_state: Current state index

        Returns:
            Array of probabilities for each possible next state
        """
        if self._matrix is None:
            return np.ones(self.config.n_states) / self.config.n_states

        return self._matrix[current_state].copy()

    def most_likely_next(self, current_state: int) -> int:
        """
        Get most probable next state.

        Args:
            current_state: Current state index

        Returns:
            Most likely next state index
        """
        probs = self.predict_next(current_state)
        return int(np.argmax(probs))

    def sample_next(self, current_state: int) -> int:
        """
        Randomly sample next state according to transition probabilities.

        Args:
            current_state: Current state index

        Returns:
            Sampled next state index
        """
        probs = self.predict_next(current_state)
        return int(np.random.choice(self.config.n_states, p=probs))

    def simulate_path(
        self,
        start_state: int,
        n_steps: int,
    ) -> np.ndarray:
        """
        Simulate a path through the Markov chain.

        Args:
            start_state: Initial state
            n_steps: Number of steps to simulate

        Returns:
            Array of states [s_0, s_1, ..., s_n]
        """
        path = [start_state]
        current = start_state

        for _ in range(n_steps):
            current = self.sample_next(current)
            path.append(current)

        return np.array(path)

    @property
    def is_reliable(self) -> bool:
        """Whether matrix has enough samples to be reliable."""
        return self._total_transitions >= self.config.min_samples

    @property
    def n_states(self) -> int:
        """Number of states."""
        return self.config.n_states

    @property
    def total_transitions(self) -> int:
        """Total number of observed transitions."""
        return self._total_transitions

    @property
    def is_fitted(self) -> bool:
        """Whether matrix has been fitted."""
        return self._fitted

    def get_row_entropy(self) -> np.ndarray:
        """
        Compute entropy of each row (state uncertainty).

        High entropy = uncertain about next state
        Low entropy = predictable transitions

        Returns:
            Array of entropies for each state
        """
        if self._matrix is None:
            return np.zeros(self.config.n_states)

        # H = -sum(p * log(p))
        with np.errstate(divide='ignore', invalid='ignore'):
            log_probs = np.log2(self._matrix)
            log_probs = np.where(np.isfinite(log_probs), log_probs, 0)

        return -np.sum(self._matrix * log_probs, axis=1)

    def get_persistence(self) -> np.ndarray:
        """
        Get self-transition probabilities (diagonal).

        High persistence = states tend to persist (trending)
        Low persistence = states tend to switch (mean-reverting)

        Returns:
            Array of P(stay in state i | in state i)
        """
        if self._matrix is None:
            return np.ones(self.config.n_states) / self.config.n_states

        return np.diag(self._matrix)

    def is_ergodic(self) -> bool:
        """
        Check if the chain is ergodic (irreducible and aperiodic).

        Ergodic chains have a unique stationary distribution.
        """
        if self._matrix is None:
            return False

        # Simple check: all entries positive (sufficient for ergodicity)
        return np.all(self._matrix > 0)

    def expected_return_time(self, state: int) -> float:
        """
        Expected number of steps to return to a state.

        For ergodic chains: E[return to i] = 1 / pi[i]
        where pi is stationary distribution.

        Args:
            state: State index

        Returns:
            Expected return time (may be inf for transient states)
        """
        from .stationary_dist import StationaryDistribution

        sd = StationaryDistribution()
        pi = sd.compute(self.matrix)

        if pi[state] > 0:
            return 1.0 / pi[state]
        return float('inf')

    def to_dict(self) -> Dict:
        """Serialize matrix to dictionary."""
        return {
            "n_states": self.config.n_states,
            "smoothing": self.config.smoothing,
            "min_samples": self.config.min_samples,
            "counts": self.counts.tolist(),
            "matrix": self._matrix.tolist() if self._matrix is not None else None,
            "total_transitions": self._total_transitions,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TransitionMatrix":
        """Deserialize matrix from dictionary."""
        tm = cls(
            n_states=data["n_states"],
            smoothing=data["smoothing"],
            min_samples=data.get("min_samples", 30),
        )
        tm.counts = np.array(data["counts"])
        if data.get("matrix") is not None:
            tm._matrix = np.array(data["matrix"])
        tm._total_transitions = data.get("total_transitions", 0)
        tm._fitted = data.get("fitted", False)
        return tm

    def save(self, filepath: Union[str, Path]) -> None:
        """Save matrix to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved transition matrix to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TransitionMatrix":
        """Load matrix from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self._fitted else "not fitted"
        reliable = "reliable" if self.is_reliable else "unreliable"
        return (
            f"TransitionMatrix(n_states={self.config.n_states}, "
            f"transitions={self._total_transitions}, {status}, {reliable})"
        )

    def pretty_print(self, state_names: Optional[Dict[int, str]] = None) -> str:
        """
        Create human-readable representation of matrix.

        Args:
            state_names: Optional mapping of state index to name

        Returns:
            Formatted string representation
        """
        if state_names is None:
            state_names = {i: f"S{i}" for i in range(self.config.n_states)}
        else:
            # Ensure keys are integers
            state_names = {int(k): v for k, v in state_names.items()}

        lines = ["Transition Matrix:"]

        # Header
        header = "        " + "  ".join(f"{state_names[i]:>6}" for i in range(self.config.n_states))
        lines.append(header)

        # Rows
        mat = self.matrix
        for i in range(self.config.n_states):
            row = f"{state_names[i]:>6}: " + "  ".join(f"{mat[i,j]:6.3f}" for j in range(self.config.n_states))
            lines.append(row)

        return "\n".join(lines)


def build_transition_matrix(
    returns: pd.Series,
    n_states: int = 3,
    method: str = "threshold",
    smoothing: float = 1.0,
) -> Tuple[TransitionMatrix, np.ndarray]:
    """
    Convenience function to build transition matrix from returns.

    Args:
        returns: Daily returns series
        n_states: Number of discrete states
        method: State classification method
        smoothing: Laplace smoothing factor

    Returns:
        Tuple of (TransitionMatrix, state_array)
    """
    from .state_classifier import StateClassifier

    # Classify returns
    classifier = StateClassifier(n_states=n_states, method=method)
    classifier.fit(returns)
    states = classifier.classify(returns)

    # Build matrix
    tm = TransitionMatrix(n_states=n_states, smoothing=smoothing)
    tm.fit(states)

    return tm, states
