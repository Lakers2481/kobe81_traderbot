"""
State Classifier for Markov Chains

Discretizes continuous daily returns into discrete states for Markov chain analysis.

Supports multiple classification methods:
- threshold: Fixed percentage thresholds (default: +/-0.5%)
- percentile: Dynamic thresholds based on return distribution
- volatility_adjusted: Thresholds scaled by rolling volatility

States:
- 3-state (ternary): DOWN (0), FLAT (1), UP (2)
- 2-state (binary): DOWN (0), UP (1)
- 5-state (quintile): For finer-grained analysis

Usage:
    classifier = StateClassifier(n_states=3, method="threshold")
    classifier.fit(historical_returns)
    states = classifier.classify(new_returns)

    # Get state name
    print(classifier.state_name(2))  # "UP"

Created: 2026-01-04
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StateNames:
    """Human-readable state names by number of states."""

    BINARY = {0: "DOWN", 1: "UP"}
    TERNARY = {0: "DOWN", 1: "FLAT", 2: "UP"}
    QUINTILE = {0: "STRONG_DOWN", 1: "DOWN", 2: "FLAT", 3: "UP", 4: "STRONG_UP"}

    @classmethod
    def get_names(cls, n_states: int) -> Dict[int, str]:
        """Get state name mapping for given number of states."""
        if n_states == 2:
            return cls.BINARY
        elif n_states == 3:
            return cls.TERNARY
        elif n_states == 5:
            return cls.QUINTILE
        else:
            return {i: f"STATE_{i}" for i in range(n_states)}


@dataclass
class ClassifierParams:
    """Parameters for state classification."""

    # Number of states (2, 3, or 5 recommended)
    n_states: int = 3

    # Classification method
    method: str = "threshold"  # "threshold", "percentile", "volatility_adjusted"

    # Fixed thresholds for "threshold" method (percentage)
    threshold_up: float = 0.5  # Return > 0.5% = UP
    threshold_down: float = -0.5  # Return < -0.5% = DOWN

    # Volatility window for "volatility_adjusted" method
    vol_window: int = 20

    # Volatility multiplier for thresholds
    vol_multiplier: float = 0.5  # threshold = vol_multiplier * rolling_vol

    # Percentile boundaries for "percentile" method (auto-computed)
    percentile_boundaries: List[float] = field(default_factory=list)


class StateClassifier:
    """
    Classify price movements into discrete Markov states.

    This is the foundation of the Markov chain system - it converts
    continuous daily returns into discrete states that can be used
    to build transition matrices.

    Example:
        >>> classifier = StateClassifier(n_states=3, method="threshold")
        >>> returns = pd.Series([0.01, -0.02, 0.005, -0.001, 0.03])
        >>> classifier.fit(returns)
        >>> states = classifier.classify(returns)
        >>> print(states)
        [2, 0, 1, 1, 2]  # UP, DOWN, FLAT, FLAT, UP
    """

    def __init__(
        self,
        n_states: int = 3,
        method: str = "threshold",
        threshold_up: float = 0.5,
        threshold_down: float = -0.5,
        vol_window: int = 20,
        vol_multiplier: float = 0.5,
    ):
        """
        Initialize state classifier.

        Args:
            n_states: Number of discrete states (2, 3, or 5)
            method: Classification method
                - "threshold": Fixed percentage thresholds
                - "percentile": Dynamic thresholds from data
                - "volatility_adjusted": Thresholds scaled by volatility
            threshold_up: Upper threshold for "threshold" method (percentage)
            threshold_down: Lower threshold for "threshold" method (percentage)
            vol_window: Rolling window for volatility calculation
            vol_multiplier: Multiplier for volatility-adjusted thresholds
        """
        self.params = ClassifierParams(
            n_states=n_states,
            method=method,
            threshold_up=threshold_up / 100,  # Convert to decimal
            threshold_down=threshold_down / 100,
            vol_window=vol_window,
            vol_multiplier=vol_multiplier,
        )

        self._fitted = False
        self._boundaries: Optional[np.ndarray] = None
        self._rolling_vol: Optional[pd.Series] = None
        self._state_names = StateNames.get_names(n_states)

        logger.debug(f"StateClassifier initialized: {n_states} states, method={method}")

    def fit(self, returns: Union[pd.Series, np.ndarray]) -> "StateClassifier":
        """
        Fit classifier to historical returns.

        For "percentile" method, this computes the percentile boundaries.
        For "volatility_adjusted" method, this computes rolling volatility.
        For "threshold" method, this is a no-op but required for API consistency.

        Args:
            returns: Historical daily returns (as decimals, e.g., 0.01 = 1%)

        Returns:
            Self for method chaining
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        returns = returns.dropna()

        if len(returns) < self.params.vol_window:
            logger.warning(f"Insufficient data for fit: {len(returns)} < {self.params.vol_window}")

        if self.params.method == "percentile":
            # Compute percentile boundaries
            n = self.params.n_states
            percentiles = np.linspace(0, 100, n + 1)
            self._boundaries = np.percentile(returns, percentiles)
            self.params.percentile_boundaries = self._boundaries.tolist()
            logger.debug(f"Percentile boundaries: {self._boundaries}")

        elif self.params.method == "volatility_adjusted":
            # Compute rolling volatility
            self._rolling_vol = returns.rolling(
                window=self.params.vol_window,
                min_periods=5
            ).std()
            logger.debug(f"Rolling volatility computed, mean={self._rolling_vol.mean():.4f}")

        self._fitted = True
        return self

    def classify(
        self,
        returns: Union[pd.Series, np.ndarray],
        rolling_vol: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        Classify returns into discrete states.

        Args:
            returns: Daily returns to classify (decimals)
            rolling_vol: Optional rolling volatility for volatility_adjusted method

        Returns:
            Array of state indices (0, 1, 2, ...)
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        method = self.params.method
        n_states = self.params.n_states

        if method == "threshold":
            return self._classify_threshold(returns)
        elif method == "percentile":
            return self._classify_percentile(returns)
        elif method == "volatility_adjusted":
            vol = rolling_vol if rolling_vol is not None else self._rolling_vol
            return self._classify_volatility_adjusted(returns, vol)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _classify_threshold(self, returns: pd.Series) -> np.ndarray:
        """Classify using fixed thresholds."""
        n = self.params.n_states
        up_thresh = self.params.threshold_up
        down_thresh = self.params.threshold_down

        if n == 2:
            # Binary: 0=DOWN, 1=UP
            return (returns > 0).astype(int).values

        elif n == 3:
            # Ternary: 0=DOWN, 1=FLAT, 2=UP
            states = np.ones(len(returns), dtype=int)  # Default FLAT
            states[returns > up_thresh] = 2  # UP
            states[returns < down_thresh] = 0  # DOWN
            return states

        elif n == 5:
            # Quintile with fixed thresholds
            strong_up = up_thresh * 2
            strong_down = down_thresh * 2
            states = np.full(len(returns), 2, dtype=int)  # Default FLAT
            states[returns > strong_up] = 4  # STRONG_UP
            states[(returns > up_thresh) & (returns <= strong_up)] = 3  # UP
            states[(returns < down_thresh) & (returns >= strong_down)] = 1  # DOWN
            states[returns < strong_down] = 0  # STRONG_DOWN
            return states

        else:
            # Generic: equal-width bins
            min_ret = returns.min()
            max_ret = returns.max()
            bins = np.linspace(min_ret, max_ret, n + 1)
            return np.clip(np.digitize(returns, bins[1:-1]), 0, n - 1)

    def _classify_percentile(self, returns: pd.Series) -> np.ndarray:
        """Classify using percentile boundaries."""
        if self._boundaries is None:
            raise RuntimeError("Must call fit() before classify() for percentile method")

        # np.digitize returns 1-indexed, so subtract 1 and clip
        states = np.digitize(returns, self._boundaries[1:-1])
        return np.clip(states, 0, self.params.n_states - 1)

    def _classify_volatility_adjusted(
        self,
        returns: pd.Series,
        rolling_vol: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """Classify using volatility-adjusted thresholds."""
        n = self.params.n_states
        mult = self.params.vol_multiplier

        if rolling_vol is None:
            # Compute on-the-fly
            rolling_vol = returns.rolling(
                window=self.params.vol_window,
                min_periods=5
            ).std()

        # Align indices
        returns = returns.reindex(rolling_vol.index)

        # Dynamic thresholds
        up_thresh = mult * rolling_vol
        down_thresh = -mult * rolling_vol

        if n == 2:
            return (returns > 0).astype(int).values

        elif n == 3:
            states = np.ones(len(returns), dtype=int)  # FLAT
            states[returns > up_thresh] = 2  # UP
            states[returns < down_thresh] = 0  # DOWN
            return states

        else:
            # For n>3, use multiple of volatility
            states = np.full(len(returns), n // 2, dtype=int)  # Middle state
            for i in range(1, (n + 1) // 2):
                upper = mult * i * rolling_vol
                lower = -mult * i * rolling_vol
                states[returns > upper] = n // 2 + i
                states[returns < lower] = n // 2 - i
            return np.clip(states, 0, n - 1)

    def state_name(self, state_idx: int) -> str:
        """
        Get human-readable name for a state index.

        Args:
            state_idx: State index (0, 1, 2, ...)

        Returns:
            State name (e.g., "UP", "DOWN", "FLAT")
        """
        return self._state_names.get(state_idx, f"STATE_{state_idx}")

    def state_index(self, name: str) -> int:
        """
        Get state index from name.

        Args:
            name: State name (e.g., "UP", "DOWN")

        Returns:
            State index
        """
        for idx, n in self._state_names.items():
            if n.upper() == name.upper():
                return idx
        raise ValueError(f"Unknown state name: {name}")

    @property
    def n_states(self) -> int:
        """Number of states."""
        return self.params.n_states

    @property
    def is_fitted(self) -> bool:
        """Whether classifier has been fitted."""
        return self._fitted

    def get_state_counts(self, states: np.ndarray) -> Dict[str, int]:
        """
        Count occurrences of each state.

        Args:
            states: Array of state indices

        Returns:
            Dict mapping state name to count
        """
        unique, counts = np.unique(states, return_counts=True)
        return {self.state_name(int(s)): int(c) for s, c in zip(unique, counts)}

    def get_state_frequencies(self, states: np.ndarray) -> Dict[str, float]:
        """
        Get frequency of each state.

        Args:
            states: Array of state indices

        Returns:
            Dict mapping state name to frequency (0-1)
        """
        counts = self.get_state_counts(states)
        total = len(states)
        return {name: count / total for name, count in counts.items()}

    def to_dict(self) -> Dict:
        """Serialize classifier parameters."""
        return {
            "n_states": self.params.n_states,
            "method": self.params.method,
            "threshold_up": self.params.threshold_up * 100,
            "threshold_down": self.params.threshold_down * 100,
            "vol_window": self.params.vol_window,
            "vol_multiplier": self.params.vol_multiplier,
            "percentile_boundaries": self.params.percentile_boundaries,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StateClassifier":
        """Deserialize classifier from dict."""
        classifier = cls(
            n_states=data["n_states"],
            method=data["method"],
            threshold_up=data["threshold_up"],
            threshold_down=data["threshold_down"],
            vol_window=data.get("vol_window", 20),
            vol_multiplier=data.get("vol_multiplier", 0.5),
        )
        if data.get("percentile_boundaries"):
            classifier._boundaries = np.array(data["percentile_boundaries"])
        classifier._fitted = data.get("fitted", False)
        return classifier


def compute_returns(
    prices: Union[pd.Series, pd.DataFrame],
    price_col: str = "close",
) -> pd.Series:
    """
    Compute daily returns from prices.

    Args:
        prices: Price series or DataFrame with price column
        price_col: Column name if DataFrame

    Returns:
        Daily returns as decimals
    """
    if isinstance(prices, pd.DataFrame):
        prices = prices[price_col]

    return prices.pct_change()


# Convenience function
def classify_returns(
    returns: pd.Series,
    n_states: int = 3,
    method: str = "threshold",
) -> Tuple[np.ndarray, StateClassifier]:
    """
    Quick utility to classify returns.

    Args:
        returns: Daily returns
        n_states: Number of states
        method: Classification method

    Returns:
        Tuple of (state_array, classifier)
    """
    classifier = StateClassifier(n_states=n_states, method=method)
    classifier.fit(returns)
    states = classifier.classify(returns)
    return states, classifier
