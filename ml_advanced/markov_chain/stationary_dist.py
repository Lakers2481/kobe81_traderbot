"""
Stationary Distribution Calculator for Markov Chains

Computes the equilibrium (stationary) distribution π where πP = π.

The stationary distribution tells us the long-run proportion of time
the system spends in each state. For trading:
- π(UP) = long-run probability stock is in UP state
- Stocks with higher π(UP) trend upward more often
- Use for asset ranking and mean-reversion signals

Mathematical Background:
- π is the left eigenvector of P corresponding to eigenvalue 1
- For ergodic chains, π is unique and all entries are positive
- π[i] = 1/E[return time to i]

Usage:
    sd = StationaryDistribution()
    pi = sd.compute(transition_matrix)

    # Rank assets by UP probability
    rankings = sd.rank_assets(symbols, {sym: P for sym, P in matrices.items()})

    # Mean-reversion signal
    deviation = sd.deviation_score(current_state=0, transition_matrix=P)

Created: 2026-01-04
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import linalg

logger = logging.getLogger(__name__)


class StationaryDistribution:
    """
    Compute and analyze stationary distributions for Markov chains.

    The stationary distribution π satisfies:
        πP = π    (left eigenvector with eigenvalue 1)
        sum(π) = 1 (probability distribution)

    For a 3-state chain (DOWN, FLAT, UP):
        π = [0.30, 0.35, 0.35]

    This means in the long run:
        - 30% of days the stock is in DOWN state
        - 35% of days the stock is in FLAT state
        - 35% of days the stock is in UP state

    Trading Applications:
    1. Asset Ranking: Stocks with higher π(UP) trend up more often
    2. Mean Reversion: If current state is below equilibrium → buy signal
    3. Regime Detection: Compare actual frequencies to stationary distribution
    """

    def __init__(self, method: str = "eigen"):
        """
        Initialize stationary distribution calculator.

        Args:
            method: Computation method
                - "eigen": Eigenvalue decomposition (default, fast)
                - "power": Power iteration (more stable for ill-conditioned)
                - "linear": Solve linear system (robust)
        """
        self.method = method
        self._last_matrix: Optional[np.ndarray] = None
        self._last_distribution: Optional[np.ndarray] = None

    def compute(
        self,
        transition_matrix: np.ndarray,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute stationary distribution.

        Args:
            transition_matrix: Row-stochastic transition matrix (rows sum to 1)
            method: Override default computation method

        Returns:
            Stationary distribution π (sums to 1)
        """
        P = np.asarray(transition_matrix)
        n = P.shape[0]

        # Validate input
        if P.shape[0] != P.shape[1]:
            raise ValueError("Transition matrix must be square")

        if not np.allclose(P.sum(axis=1), 1.0, rtol=1e-5):
            logger.warning("Rows don't sum to 1, normalizing...")
            P = P / P.sum(axis=1, keepdims=True)

        method = method or self.method

        if method == "eigen":
            pi = self._compute_eigen(P)
        elif method == "power":
            pi = self._compute_power(P)
        elif method == "linear":
            pi = self._compute_linear(P)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Cache result
        self._last_matrix = P.copy()
        self._last_distribution = pi.copy()

        return pi

    def _compute_eigen(self, P: np.ndarray) -> np.ndarray:
        """
        Compute via eigenvalue decomposition.

        The stationary distribution is the left eigenvector of P
        corresponding to eigenvalue 1.

        Left eigenvector: πP = λπ where λ = 1
        Equivalent to: P^T π^T = π^T
        """
        # Compute left eigenvectors (transpose and find right eigenvectors)
        eigenvalues, eigenvectors = linalg.eig(P.T)

        # Find eigenvector for eigenvalue ≈ 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))

        # Extract and normalize
        pi = np.real(eigenvectors[:, idx])

        # Ensure all positive (handle numerical issues)
        pi = np.abs(pi)
        pi = pi / pi.sum()

        return pi

    def _compute_power(
        self,
        P: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-10,
    ) -> np.ndarray:
        """
        Compute via power iteration.

        Start with uniform distribution and repeatedly multiply by P
        until convergence.
        """
        n = P.shape[0]
        pi = np.ones(n) / n  # Start uniform

        for i in range(max_iter):
            pi_new = pi @ P  # Row vector × matrix

            # Check convergence
            if np.max(np.abs(pi_new - pi)) < tol:
                logger.debug(f"Power iteration converged in {i+1} iterations")
                break

            pi = pi_new
        else:
            logger.warning(f"Power iteration did not converge in {max_iter} iterations")

        return pi / pi.sum()

    def _compute_linear(self, P: np.ndarray) -> np.ndarray:
        """
        Compute by solving linear system.

        Solve: π(P - I) = 0 with constraint sum(π) = 1

        Reformulated as augmented system:
        [P^T - I; 1...1] @ π = [0...0; 1]
        """
        n = P.shape[0]

        # Build augmented system
        A = P.T - np.eye(n)
        A = np.vstack([A, np.ones(n)])

        b = np.zeros(n + 1)
        b[-1] = 1.0

        # Solve least squares (overdetermined system)
        pi, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # Ensure positive and normalized
        pi = np.maximum(pi, 0)
        pi = pi / pi.sum()

        return pi

    def deviation_score(
        self,
        current_state: int,
        transition_matrix: np.ndarray,
        current_frequencies: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate deviation from equilibrium.

        Measures how far current state is from long-run equilibrium.
        Positive score = state is underrepresented = potential buy signal.
        Negative score = state is overrepresented = potential sell signal.

        Args:
            current_state: Current state index (0, 1, 2, ...)
            transition_matrix: Transition probability matrix
            current_frequencies: Optional recent state frequencies

        Returns:
            Deviation score (-1 to +1)
                >0: Current state below equilibrium (mean-reversion buy)
                <0: Current state above equilibrium (mean-reversion sell)
        """
        pi = self.compute(transition_matrix)

        if current_frequencies is not None:
            # Compare actual recent frequencies to stationary
            deviation = pi[current_state] - current_frequencies[current_state]
        else:
            # Use simple indicator: 1/n - pi[current]
            # Positive if state has low equilibrium probability
            n = len(pi)
            uniform = 1.0 / n
            deviation = uniform - pi[current_state]

        return float(deviation)

    def mean_reversion_signal(
        self,
        current_state: int,
        transition_matrix: np.ndarray,
        up_state: int = 2,
        threshold: float = 0.05,
    ) -> str:
        """
        Generate mean-reversion signal based on stationary distribution.

        Logic:
        - If in DOWN state and π(DOWN) < threshold → expect reversion UP → BUY
        - If in UP state and π(UP) < threshold → expect reversion DOWN → SELL
        - Otherwise → HOLD

        Args:
            current_state: Current state index
            transition_matrix: Transition matrix
            up_state: Index of UP state (default 2 for ternary)
            threshold: Deviation threshold for signals

        Returns:
            "BUY", "SELL", or "HOLD"
        """
        pi = self.compute(transition_matrix)
        n = len(pi)
        down_state = 0

        # Deviation from uniform
        uniform = 1.0 / n

        if current_state == down_state:
            # In DOWN state
            if pi[down_state] < uniform - threshold:
                # DOWN is underrepresented in long-run → expect bounce
                return "BUY"

        elif current_state == up_state:
            # In UP state
            if pi[up_state] < uniform - threshold:
                # UP is underrepresented → expect pullback
                return "SELL"

        return "HOLD"

    def rank_assets(
        self,
        symbols: List[str],
        matrices: Dict[str, np.ndarray],
        up_state: int = 2,
    ) -> pd.DataFrame:
        """
        Rank assets by their stationary UP probability.

        Stocks with higher π(UP) spend more time trending up.
        Use for pre-filtering universe before detailed analysis.

        Args:
            symbols: List of stock symbols
            matrices: Dict mapping symbol to transition matrix
            up_state: Index of UP state (default 2)

        Returns:
            DataFrame with columns:
                - symbol
                - pi_up: P(UP) in stationary distribution
                - pi_down: P(DOWN) in stationary distribution
                - pi_flat: P(FLAT) in stationary distribution
                - trend_score: pi_up - pi_down
                - rank: 1 = highest pi_up
        """
        results = []

        for sym in symbols:
            if sym not in matrices:
                continue

            try:
                pi = self.compute(matrices[sym])
                n = len(pi)

                result = {
                    "symbol": sym,
                    "pi_up": pi[up_state] if up_state < n else 0.0,
                    "pi_down": pi[0],
                }

                # FLAT state if 3 states
                if n >= 3:
                    result["pi_flat"] = pi[1]

                result["trend_score"] = result["pi_up"] - result["pi_down"]

                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to compute stationary for {sym}: {e}")

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values("pi_up", ascending=False)
        df["rank"] = range(1, len(df) + 1)

        return df

    def mixing_time(
        self,
        transition_matrix: np.ndarray,
        epsilon: float = 0.01,
    ) -> int:
        """
        Estimate mixing time (steps to reach near-equilibrium).

        The mixing time tells us how quickly the chain "forgets"
        its initial state and approaches stationary distribution.

        Shorter mixing time = faster mean-reversion.

        Args:
            transition_matrix: Transition matrix
            epsilon: Distance from stationary to consider "mixed"

        Returns:
            Estimated number of steps to mix
        """
        P = np.asarray(transition_matrix)
        pi = self.compute(P)
        n = P.shape[0]

        # Start from worst-case initial distribution
        max_steps = 1000

        for k in range(1, max_steps + 1):
            # Compute P^k
            P_k = np.linalg.matrix_power(P, k)

            # Check if all rows are close to stationary
            max_dist = 0.0
            for i in range(n):
                dist = np.sum(np.abs(P_k[i] - pi))
                max_dist = max(max_dist, dist)

            if max_dist < epsilon:
                return k

        logger.warning(f"Mixing time exceeds {max_steps} steps")
        return max_steps

    def second_eigenvalue(self, transition_matrix: np.ndarray) -> float:
        """
        Get second largest eigenvalue magnitude.

        The second eigenvalue determines convergence rate:
        - |λ2| close to 0 = fast mixing
        - |λ2| close to 1 = slow mixing, high persistence

        For trading:
        - High |λ2| = trending behavior
        - Low |λ2| = mean-reverting behavior

        Returns:
            Magnitude of second largest eigenvalue
        """
        eigenvalues = linalg.eigvals(transition_matrix)
        mags = np.abs(eigenvalues)
        mags_sorted = np.sort(mags)[::-1]  # Descending

        return float(mags_sorted[1]) if len(mags_sorted) > 1 else 0.0

    def expected_hitting_time(
        self,
        transition_matrix: np.ndarray,
        target_state: int,
    ) -> np.ndarray:
        """
        Compute expected hitting times to target state.

        E[T_j | X_0 = i] = expected steps to reach state j starting from i

        For trading: "How many days until stock enters UP state?"

        Args:
            transition_matrix: Transition matrix
            target_state: Target state index

        Returns:
            Array of expected hitting times from each state
        """
        P = np.asarray(transition_matrix)
        n = P.shape[0]
        j = target_state

        # Remove target state from system
        mask = np.ones(n, dtype=bool)
        mask[j] = False

        Q = P[mask][:, mask]  # Transitions among non-target states
        r = P[mask, j]  # Transitions to target state

        # Solve: h = 1 + Q @ h
        # Rearranged: (I - Q) @ h = 1
        I = np.eye(n - 1)

        try:
            h_partial = linalg.solve(I - Q, np.ones(n - 1))
        except linalg.LinAlgError:
            logger.warning("Singular matrix in hitting time calculation")
            h_partial = np.full(n - 1, float('inf'))

        # Insert 0 for target state
        h = np.zeros(n)
        h[mask] = h_partial
        h[j] = 0.0  # Already at target

        return h

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "method": self.method,
            "last_distribution": (
                self._last_distribution.tolist()
                if self._last_distribution is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StationaryDistribution":
        """Deserialize from dictionary."""
        sd = cls(method=data.get("method", "eigen"))
        if data.get("last_distribution"):
            sd._last_distribution = np.array(data["last_distribution"])
        return sd


def compute_stationary_distribution(
    transition_matrix: np.ndarray,
) -> np.ndarray:
    """
    Convenience function to compute stationary distribution.

    Args:
        transition_matrix: Row-stochastic matrix

    Returns:
        Stationary distribution π
    """
    sd = StationaryDistribution()
    return sd.compute(transition_matrix)


def rank_by_trend_probability(
    returns_dict: Dict[str, pd.Series],
    n_states: int = 3,
    up_state: int = 2,
) -> pd.DataFrame:
    """
    Rank assets by their long-run UP probability.

    Builds transition matrices and computes stationary distributions
    for each asset, then ranks by π(UP).

    Args:
        returns_dict: Dict mapping symbol to returns series
        n_states: Number of states for classification
        up_state: Index of UP state

    Returns:
        Ranked DataFrame with pi_up, trend_score, rank
    """
    from .transition_matrix import build_transition_matrix

    matrices = {}

    for symbol, returns in returns_dict.items():
        try:
            tm, _ = build_transition_matrix(returns, n_states=n_states)
            if tm.is_reliable:
                matrices[symbol] = tm.matrix
        except Exception as e:
            logger.warning(f"Failed to build matrix for {symbol}: {e}")

    sd = StationaryDistribution()
    return sd.rank_assets(list(matrices.keys()), matrices, up_state=up_state)
