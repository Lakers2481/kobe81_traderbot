"""
Markov Asset Scorer for Universe Ranking

Scores and ranks stocks by Markov chain metrics for pre-filtering.
Use this to prioritize which stocks to analyze in detail.

Scoring Factors:
1. π(Up) - Long-run probability of being in UP state
2. P(current → Up) - Transition probability to UP given current state
3. Stationary deviation - Distance from equilibrium (mean-reversion)
4. Regime persistence - Self-transition probability

Use Cases:
- Pre-filter 800 stocks to top 100 by π(Up) before detailed scan
- Rank watchlist by today's UP probability
- Identify mean-reversion candidates (high deviation)

Usage:
    scorer = MarkovAssetScorer()
    rankings = scorer.score_universe(symbols, returns_dict)

    # Get top 100 by trend probability
    top_100 = scorer.filter_top_n(rankings, n=100)

Created: 2026-01-04
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .stationary_dist import StationaryDistribution
from .transition_matrix import build_transition_matrix

logger = logging.getLogger(__name__)


@dataclass
class ScorerConfig:
    """Configuration for asset scorer."""

    # State classification
    n_states: int = 3
    classification_method: str = "threshold"

    # Lookback for building chains
    lookback_days: int = 252  # 1 year

    # Minimum data requirements
    min_days: int = 100
    min_transitions: int = 50

    # Scoring weights
    weight_pi_up: float = 0.35  # Weight for stationary UP probability
    weight_p_up_today: float = 0.30  # Weight for today's UP transition prob
    weight_deviation: float = 0.20  # Weight for mean-reversion score
    weight_persistence: float = 0.15  # Weight for trend persistence

    # Parallel processing
    max_workers: int = 10


class AssetScore:
    """Score for a single asset."""

    def __init__(
        self,
        symbol: str,
        pi_up: float,
        p_up_today: float,
        deviation: float,
        persistence: float,
        composite_score: float,
        current_state: int,
        samples: int,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.symbol = symbol
        self.pi_up = pi_up
        self.p_up_today = p_up_today
        self.deviation = deviation
        self.persistence = persistence
        self.composite_score = composite_score
        self.current_state = current_state
        self.samples = samples
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "pi_up": self.pi_up,
            "p_up_today": self.p_up_today,
            "deviation": self.deviation,
            "persistence": self.persistence,
            "composite_score": self.composite_score,
            "current_state": self.current_state,
            "samples": self.samples,
            **self.details,
        }


class MarkovAssetScorer:
    """
    Score and rank assets using Markov chain metrics.

    This is the core ranking system for pre-filtering the universe.
    Instead of scanning all 800 stocks with full strategy logic,
    first score them by Markov metrics and focus on the top N.

    Scoring Components:

    1. π(Up) - Stationary Distribution
       - Long-run probability of being in UP state
       - Higher = stock trends up more often
       - Most important factor for momentum

    2. P(Up | current) - Transition Probability
       - Probability of going UP from current state
       - Higher when in DOWN state (bounce expected)
       - Context-dependent signal

    3. Deviation - Mean-Reversion Score
       - How far current state is from equilibrium
       - Positive = underrepresented = buy signal
       - Key for mean-reversion strategies

    4. Persistence - Trend Strength
       - Self-transition probability
       - High = trending behavior
       - Low = mean-reverting behavior

    Example:
        scorer = MarkovAssetScorer()

        # Score entire universe
        rankings = scorer.score_universe(
            symbols=["AAPL", "MSFT", ...],
            returns_dict=returns_data,
        )

        # Get top 100 momentum candidates
        top_momentum = scorer.filter_top_n(rankings, n=100, sort_by="pi_up")

        # Get top 50 mean-reversion candidates
        top_mr = scorer.filter_top_n(rankings, n=50, sort_by="deviation")
    """

    def __init__(
        self,
        n_states: int = 3,
        lookback_days: int = 252,
        classification_method: str = "threshold",
        weight_pi_up: float = 0.35,
        weight_p_up_today: float = 0.30,
        weight_deviation: float = 0.20,
        weight_persistence: float = 0.15,
        max_workers: int = 10,
    ):
        """
        Initialize asset scorer.

        Args:
            n_states: Number of states for classification
            lookback_days: Days of history for building chains
            classification_method: State classification method
            weight_pi_up: Weight for stationary UP probability
            weight_p_up_today: Weight for transition probability
            weight_deviation: Weight for mean-reversion score
            weight_persistence: Weight for trend persistence
            max_workers: Parallel processing threads
        """
        self.config = ScorerConfig(
            n_states=n_states,
            lookback_days=lookback_days,
            classification_method=classification_method,
            weight_pi_up=weight_pi_up,
            weight_p_up_today=weight_p_up_today,
            weight_deviation=weight_deviation,
            weight_persistence=weight_persistence,
            max_workers=max_workers,
        )

        self.stationary_dist = StationaryDistribution()

        logger.debug(f"MarkovAssetScorer initialized: n_states={n_states}, lookback={lookback_days}")

    def score_symbol(
        self,
        symbol: str,
        returns: pd.Series,
    ) -> Optional[AssetScore]:
        """
        Score a single symbol.

        Args:
            symbol: Stock symbol
            returns: Daily returns series

        Returns:
            AssetScore or None if insufficient data
        """
        cfg = self.config

        # Validate data
        returns = returns.dropna()
        if len(returns) < cfg.min_days:
            logger.debug(f"{symbol}: insufficient data ({len(returns)} < {cfg.min_days})")
            return None

        # Use most recent data
        returns = returns.tail(cfg.lookback_days)

        try:
            # Build transition matrix
            tm, states = build_transition_matrix(
                returns,
                n_states=cfg.n_states,
                method=cfg.classification_method,
            )

            if tm.total_transitions < cfg.min_transitions:
                logger.debug(f"{symbol}: insufficient transitions ({tm.total_transitions})")
                return None

            # Current state
            current_state = int(states[-1])

            # 1. Stationary UP probability
            pi = self.stationary_dist.compute(tm.matrix)
            up_state = cfg.n_states - 1
            pi_up = float(pi[up_state])

            # 2. Transition probability to UP from current
            p_up_today = float(tm.get_probability(current_state, up_state))

            # 3. Deviation from equilibrium
            # Positive if current state is underrepresented
            recent_freq = np.zeros(cfg.n_states)
            unique, counts = np.unique(states[-20:], return_counts=True)
            for s, c in zip(unique, counts):
                recent_freq[s] = c / 20
            deviation = float(pi[up_state] - recent_freq[up_state])

            # 4. Persistence (self-transition of UP state)
            persistence = float(tm.get_probability(up_state, up_state))

            # Composite score
            composite = (
                cfg.weight_pi_up * pi_up +
                cfg.weight_p_up_today * p_up_today +
                cfg.weight_deviation * (deviation + 0.5) +  # Shift to positive
                cfg.weight_persistence * persistence
            )

            return AssetScore(
                symbol=symbol,
                pi_up=pi_up,
                p_up_today=p_up_today,
                deviation=deviation,
                persistence=persistence,
                composite_score=composite,
                current_state=current_state,
                samples=tm.total_transitions,
                details={
                    "pi_down": float(pi[0]),
                    "p_down_today": float(tm.get_probability(current_state, 0)),
                },
            )

        except Exception as e:
            logger.warning(f"Failed to score {symbol}: {e}")
            return None

    def score_universe(
        self,
        symbols: List[str],
        returns_dict: Dict[str, pd.Series],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        """
        Score entire universe in parallel.

        Args:
            symbols: List of stock symbols
            returns_dict: Dict mapping symbol to returns series
            progress_callback: Optional callback(completed, total)

        Returns:
            DataFrame with scores for all symbols, sorted by composite_score
        """
        scores: List[AssetScore] = []
        completed = 0
        total = len(symbols)

        def score_one(sym: str) -> Optional[AssetScore]:
            if sym not in returns_dict:
                return None
            return self.score_symbol(sym, returns_dict[sym])

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(score_one, sym): sym for sym in symbols}

            for future in as_completed(futures):
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

                try:
                    score = future.result()
                    if score is not None:
                        scores.append(score)
                except Exception as e:
                    sym = futures[future]
                    logger.warning(f"Error scoring {sym}: {e}")

        if not scores:
            logger.warning("No symbols scored successfully")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([s.to_dict() for s in scores])

        # Sort by composite score descending
        df = df.sort_values("composite_score", ascending=False)
        df["rank"] = range(1, len(df) + 1)

        logger.info(f"Scored {len(df)} symbols out of {total}")

        return df

    def filter_top_n(
        self,
        scored_df: pd.DataFrame,
        n: int = 100,
        sort_by: str = "composite_score",
        ascending: bool = False,
        min_score: Optional[float] = None,
    ) -> List[str]:
        """
        Get top N symbols by score.

        Args:
            scored_df: DataFrame from score_universe()
            n: Number of symbols to return
            sort_by: Column to sort by
            ascending: Sort ascending (default False = highest first)
            min_score: Optional minimum score threshold

        Returns:
            List of top N symbols
        """
        if scored_df.empty:
            return []

        df = scored_df.copy()

        # Apply minimum score filter
        if min_score is not None and sort_by in df.columns:
            df = df[df[sort_by] >= min_score]

        # Sort
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)

        # Get top N
        return df.head(n)["symbol"].tolist()

    def get_momentum_candidates(
        self,
        scored_df: pd.DataFrame,
        n: int = 50,
        min_pi_up: float = 0.35,
    ) -> List[str]:
        """
        Get top momentum candidates (high π(Up)).

        Stocks that spend more time in UP state.

        Args:
            scored_df: Scored DataFrame
            n: Number of candidates
            min_pi_up: Minimum stationary UP probability

        Returns:
            List of momentum candidate symbols
        """
        if scored_df.empty:
            return []

        df = scored_df[scored_df["pi_up"] >= min_pi_up].copy()
        df = df.sort_values("pi_up", ascending=False)

        return df.head(n)["symbol"].tolist()

    def get_mean_reversion_candidates(
        self,
        scored_df: pd.DataFrame,
        n: int = 50,
        min_deviation: float = 0.05,
    ) -> List[str]:
        """
        Get top mean-reversion candidates (high positive deviation).

        Stocks currently below their equilibrium.

        Args:
            scored_df: Scored DataFrame
            n: Number of candidates
            min_deviation: Minimum positive deviation

        Returns:
            List of mean-reversion candidate symbols
        """
        if scored_df.empty:
            return []

        df = scored_df[scored_df["deviation"] >= min_deviation].copy()
        df = df.sort_values("deviation", ascending=False)

        return df.head(n)["symbol"].tolist()

    def get_bounce_candidates(
        self,
        scored_df: pd.DataFrame,
        n: int = 30,
        min_p_up: float = 0.5,
    ) -> List[str]:
        """
        Get bounce candidates (currently in DOWN, high P(Up)).

        Stocks likely to bounce based on current state.

        Args:
            scored_df: Scored DataFrame
            n: Number of candidates
            min_p_up: Minimum transition probability to UP

        Returns:
            List of bounce candidate symbols
        """
        if scored_df.empty:
            return []

        # Filter to stocks in DOWN state (0) with high P(Up)
        df = scored_df[
            (scored_df["current_state"] == 0) &
            (scored_df["p_up_today"] >= min_p_up)
        ].copy()

        df = df.sort_values("p_up_today", ascending=False)

        return df.head(n)["symbol"].tolist()

    def get_trending_candidates(
        self,
        scored_df: pd.DataFrame,
        n: int = 30,
        min_persistence: float = 0.45,
    ) -> List[str]:
        """
        Get trending candidates (high UP persistence).

        Stocks with strong trend continuation.

        Args:
            scored_df: Scored DataFrame
            n: Number of candidates
            min_persistence: Minimum UP self-transition probability

        Returns:
            List of trending candidate symbols
        """
        if scored_df.empty:
            return []

        # Filter to stocks in UP state with high persistence
        df = scored_df[
            (scored_df["current_state"] == self.config.n_states - 1) &
            (scored_df["persistence"] >= min_persistence)
        ].copy()

        df = df.sort_values("persistence", ascending=False)

        return df.head(n)["symbol"].tolist()

    def create_analysis_report(
        self,
        scored_df: pd.DataFrame,
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """
        Create comprehensive analysis report.

        Args:
            scored_df: Scored DataFrame
            top_n: Number of top stocks per category

        Returns:
            Analysis report dictionary
        """
        if scored_df.empty:
            return {"error": "No scored data"}

        report = {
            "summary": {
                "total_scored": len(scored_df),
                "mean_pi_up": scored_df["pi_up"].mean(),
                "mean_composite": scored_df["composite_score"].mean(),
            },
            "top_overall": self.filter_top_n(scored_df, n=top_n),
            "top_momentum": self.get_momentum_candidates(scored_df, n=top_n),
            "top_mean_reversion": self.get_mean_reversion_candidates(scored_df, n=top_n),
            "top_bounce": self.get_bounce_candidates(scored_df, n=top_n),
            "top_trending": self.get_trending_candidates(scored_df, n=top_n),
            "state_distribution": {
                "down": int((scored_df["current_state"] == 0).sum()),
                "flat": int((scored_df["current_state"] == 1).sum()),
                "up": int((scored_df["current_state"] == 2).sum()),
            },
            "score_percentiles": {
                "p10": scored_df["composite_score"].quantile(0.10),
                "p25": scored_df["composite_score"].quantile(0.25),
                "p50": scored_df["composite_score"].quantile(0.50),
                "p75": scored_df["composite_score"].quantile(0.75),
                "p90": scored_df["composite_score"].quantile(0.90),
            },
        }

        return report


def quick_rank(
    symbols: List[str],
    returns_dict: Dict[str, pd.Series],
    n: int = 100,
) -> List[str]:
    """
    Quick utility to rank symbols by Markov score.

    Args:
        symbols: List of symbols
        returns_dict: Dict of returns data
        n: Number of top symbols to return

    Returns:
        Top N symbols by composite score
    """
    scorer = MarkovAssetScorer()
    scored = scorer.score_universe(symbols, returns_dict)
    return scorer.filter_top_n(scored, n=n)
