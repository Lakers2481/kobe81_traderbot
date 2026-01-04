"""
Markov Chain Module for Next-Day Price Direction Prediction

This module implements Simple Markov Chains for trading, complementing
the existing HMM regime detector. While HMM uses hidden states,
Markov Chains provide observable state transitions for direction prediction.

Key Components:
- StateClassifier: Discretize returns into Up/Down/Flat states
- TransitionMatrix: Build and update P(next|current) matrices
- StationaryDistribution: Compute equilibrium π for asset ranking
- HigherOrderMarkov: 2nd/3rd order chains for multi-day patterns
- MarkovPredictor: Generate trading signals from chains
- MarkovAssetScorer: Rank universe by π(Up) probability

Usage:
    from ml_advanced.markov_chain import (
        StateClassifier,
        TransitionMatrix,
        StationaryDistribution,
        MarkovPredictor,
        MarkovAssetScorer,
    )

    # Classify returns into states
    classifier = StateClassifier(n_states=3, method="threshold")
    states = classifier.fit(returns).classify(returns)

    # Build transition matrix
    tm = TransitionMatrix(n_states=3)
    tm.fit(states)

    # Get prediction
    prob_up = tm.get_probability(current_state=0, to_state=2)  # P(Up|Down)

    # Compute stationary distribution for ranking
    sd = StationaryDistribution()
    pi = sd.compute(tm.matrix)
    pi_up = pi[2]  # Long-run probability of Up state

Created: 2026-01-04
Author: Kobe Trading Robot
"""

from .state_classifier import StateClassifier, StateNames
from .transition_matrix import TransitionMatrix
from .stationary_dist import StationaryDistribution

from .higher_order import HigherOrderMarkov
from .predictor import MarkovPredictor, MarkovPrediction
from .scorer import MarkovAssetScorer, AssetScore

__all__ = [
    # Core components
    "StateClassifier",
    "StateNames",
    "TransitionMatrix",
    "StationaryDistribution",
    # Higher-order
    "HigherOrderMarkov",
    # Prediction
    "MarkovPredictor",
    "MarkovPrediction",
    # Scoring
    "MarkovAssetScorer",
    "AssetScore",
]

__version__ = "1.0.0"
