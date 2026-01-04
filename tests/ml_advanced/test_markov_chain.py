"""
Unit Tests for Markov Chain Module

Tests all components:
- StateClassifier
- TransitionMatrix
- StationaryDistribution
- HigherOrderMarkov
- MarkovPredictor
- MarkovAssetScorer

Run: pytest tests/ml_advanced/test_markov_chain.py -v

Created: 2026-01-04
"""

import numpy as np
import pandas as pd
import pytest

# Import all components
from ml_advanced.markov_chain import (
    StateClassifier,
    StateNames,
    TransitionMatrix,
    StationaryDistribution,
    HigherOrderMarkov,
    MarkovPredictor,
    MarkovAssetScorer,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_returns():
    """Generate sample daily returns."""
    np.random.seed(42)
    # Mix of up, down, and flat days
    returns = np.random.normal(0.0005, 0.015, 500)  # Mean 0.05%, std 1.5%
    return pd.Series(returns, index=pd.date_range("2023-01-01", periods=500))


@pytest.fixture
def trending_returns():
    """Generate trending (upward biased) returns."""
    np.random.seed(42)
    returns = np.random.normal(0.002, 0.012, 300)  # Mean 0.2%, std 1.2%
    return pd.Series(returns, index=pd.date_range("2023-01-01", periods=300))


@pytest.fixture
def mean_reverting_returns():
    """Generate mean-reverting returns (negative autocorrelation)."""
    np.random.seed(42)
    base = np.random.normal(0, 0.015, 300)
    # Add negative autocorrelation
    returns = np.zeros(300)
    returns[0] = base[0]
    for i in range(1, 300):
        returns[i] = -0.3 * returns[i-1] + base[i]
    return pd.Series(returns, index=pd.date_range("2023-01-01", periods=300))


# =============================================================================
# StateClassifier Tests
# =============================================================================

class TestStateClassifier:
    """Tests for StateClassifier."""

    def test_init_default(self):
        """Test default initialization."""
        classifier = StateClassifier()
        assert classifier.n_states == 3
        assert classifier.params.method == "threshold"

    def test_init_custom(self):
        """Test custom initialization."""
        classifier = StateClassifier(n_states=5, method="percentile")
        assert classifier.n_states == 5
        assert classifier.params.method == "percentile"

    def test_classify_threshold_ternary(self, sample_returns):
        """Test ternary threshold classification."""
        classifier = StateClassifier(n_states=3, method="threshold")
        classifier.fit(sample_returns)
        states = classifier.classify(sample_returns)

        assert len(states) == len(sample_returns)
        assert set(states).issubset({0, 1, 2})

        # Check that large positive returns are UP (2)
        large_up = sample_returns > 0.005
        assert np.all(states[large_up.values] == 2)

        # Check that large negative returns are DOWN (0)
        large_down = sample_returns < -0.005
        assert np.all(states[large_down.values] == 0)

    def test_classify_binary(self, sample_returns):
        """Test binary classification."""
        classifier = StateClassifier(n_states=2, method="threshold")
        classifier.fit(sample_returns)
        states = classifier.classify(sample_returns)

        assert set(states).issubset({0, 1})

    def test_classify_percentile(self, sample_returns):
        """Test percentile classification."""
        classifier = StateClassifier(n_states=3, method="percentile")
        classifier.fit(sample_returns)
        states = classifier.classify(sample_returns)

        # Each state should have roughly 1/3 of data
        counts = np.bincount(states)
        assert len(counts) == 3
        for count in counts:
            assert 0.2 * len(sample_returns) < count < 0.5 * len(sample_returns)

    def test_state_names(self):
        """Test state name mapping."""
        classifier = StateClassifier(n_states=3)
        assert classifier.state_name(0) == "DOWN"
        assert classifier.state_name(1) == "FLAT"
        assert classifier.state_name(2) == "UP"

    def test_state_counts(self, sample_returns):
        """Test state count functionality."""
        classifier = StateClassifier(n_states=3)
        classifier.fit(sample_returns)
        states = classifier.classify(sample_returns)

        counts = classifier.get_state_counts(states)
        assert sum(counts.values()) == len(states)

    def test_serialization(self, sample_returns):
        """Test to_dict/from_dict."""
        classifier = StateClassifier(n_states=3, method="threshold")
        classifier.fit(sample_returns)

        data = classifier.to_dict()
        restored = StateClassifier.from_dict(data)

        assert restored.n_states == classifier.n_states
        assert restored.params.method == classifier.params.method


# =============================================================================
# TransitionMatrix Tests
# =============================================================================

class TestTransitionMatrix:
    """Tests for TransitionMatrix."""

    def test_init(self):
        """Test initialization."""
        tm = TransitionMatrix(n_states=3)
        assert tm.n_states == 3
        assert not tm.is_fitted

    def test_fit_simple(self):
        """Test fitting on simple sequence."""
        states = [0, 1, 2, 0, 1, 2, 0, 1, 2]  # Repeating pattern
        tm = TransitionMatrix(n_states=3, smoothing=0)
        tm.fit(states)

        assert tm.is_fitted
        assert tm.total_transitions == 8

        # Pattern 0→1, 1→2, 2→0 should have probability 1.0
        assert tm.get_probability(0, 1) > 0.9
        assert tm.get_probability(1, 2) > 0.9
        assert tm.get_probability(2, 0) > 0.9

    def test_row_sum_to_one(self, sample_returns):
        """Test that rows sum to 1."""
        classifier = StateClassifier(n_states=3)
        classifier.fit(sample_returns)
        states = classifier.classify(sample_returns)

        tm = TransitionMatrix(n_states=3)
        tm.fit(states)

        matrix = tm.matrix
        row_sums = matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_predict_next(self, sample_returns):
        """Test next state prediction."""
        classifier = StateClassifier(n_states=3)
        classifier.fit(sample_returns)
        states = classifier.classify(sample_returns)

        tm = TransitionMatrix(n_states=3)
        tm.fit(states)

        probs = tm.predict_next(0)
        assert len(probs) == 3
        assert np.isclose(probs.sum(), 1.0)

    def test_incremental_update(self):
        """Test incremental updates."""
        tm = TransitionMatrix(n_states=3, smoothing=0.1)

        # Start with empty
        assert tm.total_transitions == 0

        # Update
        tm.update(0, 1)
        tm.update(1, 2)
        tm.update(2, 0)

        assert tm.total_transitions == 3
        assert tm.is_fitted

    def test_smoothing_effect(self):
        """Test Laplace smoothing."""
        states = [0, 0, 0, 0, 0]  # Only 0→0 transitions

        # Without smoothing
        tm_no_smooth = TransitionMatrix(n_states=3, smoothing=0)
        tm_no_smooth.fit(states)
        assert tm_no_smooth.get_probability(0, 0) == 1.0
        assert tm_no_smooth.get_probability(0, 1) == 0.0

        # With smoothing
        tm_smooth = TransitionMatrix(n_states=3, smoothing=1.0)
        tm_smooth.fit(states)
        assert tm_smooth.get_probability(0, 0) > 0
        assert tm_smooth.get_probability(0, 1) > 0  # Non-zero due to smoothing

    def test_serialization(self, sample_returns):
        """Test to_dict/from_dict."""
        classifier = StateClassifier(n_states=3)
        classifier.fit(sample_returns)
        states = classifier.classify(sample_returns)

        tm = TransitionMatrix(n_states=3)
        tm.fit(states)

        data = tm.to_dict()
        restored = TransitionMatrix.from_dict(data)

        assert restored.n_states == tm.n_states
        assert restored.total_transitions == tm.total_transitions
        assert np.allclose(restored.matrix, tm.matrix)


# =============================================================================
# StationaryDistribution Tests
# =============================================================================

class TestStationaryDistribution:
    """Tests for StationaryDistribution."""

    def test_compute_uniform(self):
        """Test stationary distribution of doubly stochastic matrix."""
        # Doubly stochastic matrix has uniform stationary distribution
        P = np.array([
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5],
        ])

        sd = StationaryDistribution()
        pi = sd.compute(P)

        assert len(pi) == 3
        assert np.isclose(pi.sum(), 1.0)
        # Should be approximately uniform
        assert np.allclose(pi, [1/3, 1/3, 1/3], atol=0.01)

    def test_compute_biased(self):
        """Test stationary distribution of biased matrix."""
        # Matrix that favors state 2 (UP)
        P = np.array([
            [0.3, 0.3, 0.4],  # From DOWN, 40% go to UP
            [0.2, 0.3, 0.5],  # From FLAT, 50% go to UP
            [0.1, 0.2, 0.7],  # From UP, 70% stay UP
        ])

        sd = StationaryDistribution()
        pi = sd.compute(P)

        # UP state should have highest probability
        assert pi[2] > pi[0]
        assert pi[2] > pi[1]

    def test_mean_reversion_signal(self):
        """Test mean reversion signal generation."""
        P = np.array([
            [0.3, 0.3, 0.4],
            [0.3, 0.4, 0.3],
            [0.4, 0.3, 0.3],
        ])

        sd = StationaryDistribution()
        signal = sd.mean_reversion_signal(current_state=0, transition_matrix=P)

        # Should return a valid signal
        assert signal in ["BUY", "SELL", "HOLD"]

    def test_methods_consistent(self):
        """Test that different methods give same result."""
        np.random.seed(42)
        P = np.random.dirichlet([1, 1, 1], size=3)

        sd_eigen = StationaryDistribution(method="eigen")
        sd_power = StationaryDistribution(method="power")
        sd_linear = StationaryDistribution(method="linear")

        pi_eigen = sd_eigen.compute(P)
        pi_power = sd_power.compute(P)
        pi_linear = sd_linear.compute(P)

        assert np.allclose(pi_eigen, pi_power, atol=1e-5)
        assert np.allclose(pi_eigen, pi_linear, atol=1e-5)


# =============================================================================
# HigherOrderMarkov Tests
# =============================================================================

class TestHigherOrderMarkov:
    """Tests for HigherOrderMarkov."""

    def test_init(self):
        """Test initialization."""
        hom = HigherOrderMarkov(order=2, n_states=3)
        assert hom.order == 2
        assert hom.n_states == 3
        assert hom.n_composite == 9  # 3^2

    def test_encode_decode(self):
        """Test composite state encoding/decoding."""
        hom = HigherOrderMarkov(order=2, n_states=3)

        # Test round-trip
        for s1 in range(3):
            for s2 in range(3):
                encoded = hom.encode_composite_state(s1, s2)
                decoded = hom.decode_composite_state(encoded)
                assert decoded == (s1, s2)

    def test_fit_and_predict(self, sample_returns):
        """Test fitting and prediction."""
        classifier = StateClassifier(n_states=3)
        classifier.fit(sample_returns)
        states = classifier.classify(sample_returns)

        hom = HigherOrderMarkov(order=2, n_states=3)
        hom.fit(states)

        assert hom.is_fitted
        assert hom.total_transitions > 0

        # Predict from (DOWN, DOWN)
        probs = hom.predict(0, 0)
        assert len(probs) == 3
        assert np.isclose(probs.sum(), 1.0)

    def test_bounce_probability(self, sample_returns):
        """Test bounce probability calculation."""
        classifier = StateClassifier(n_states=3)
        classifier.fit(sample_returns)
        states = classifier.classify(sample_returns)

        hom = HigherOrderMarkov(order=2, n_states=3)
        hom.fit(states)

        bounce_prob = hom.get_bounce_probability()
        assert 0 <= bounce_prob <= 1

    def test_pattern_stats(self, sample_returns):
        """Test pattern statistics."""
        classifier = StateClassifier(n_states=3)
        classifier.fit(sample_returns)
        states = classifier.classify(sample_returns)

        hom = HigherOrderMarkov(order=2, n_states=3, min_samples=5)
        hom.fit(states)

        stats = hom.get_pattern_stats()
        assert isinstance(stats, pd.DataFrame)
        if not stats.empty:
            assert "pattern" in stats.columns
            assert "samples" in stats.columns


# =============================================================================
# MarkovPredictor Tests
# =============================================================================

class TestMarkovPredictor:
    """Tests for MarkovPredictor."""

    def test_init(self):
        """Test initialization."""
        predictor = MarkovPredictor(buy_threshold=0.6)
        assert predictor.config.buy_threshold == 0.6
        assert not predictor.is_fitted

    def test_fit(self, sample_returns):
        """Test fitting."""
        predictor = MarkovPredictor()
        predictor.fit(sample_returns)

        assert predictor.is_fitted

    def test_predict(self, sample_returns):
        """Test prediction."""
        predictor = MarkovPredictor()
        predictor.fit(sample_returns)

        prediction = predictor.predict(returns=sample_returns)

        assert prediction.signal in ["BUY", "SELL", "HOLD"]
        assert 0 <= prediction.prob_up <= 1
        assert 0 <= prediction.prob_down <= 1
        assert 0 <= prediction.confidence <= 1

    def test_predict_trending(self, trending_returns):
        """Test prediction on trending data."""
        predictor = MarkovPredictor(buy_threshold=0.5)
        predictor.fit(trending_returns)

        prediction = predictor.predict(returns=trending_returns)

        # Trending data should show higher UP probability
        assert prediction.prob_up > 0.35

    def test_score_signal(self, sample_returns):
        """Test signal scoring."""
        predictor = MarkovPredictor()
        predictor.fit(sample_returns)

        # Score a long signal
        long_signal = {"side": "long", "symbol": "TEST"}
        score = predictor.score_signal(long_signal, returns=sample_returns)

        assert 0 <= score <= 1

    def test_bounce_probability(self, sample_returns):
        """Test bounce probability."""
        predictor = MarkovPredictor()
        predictor.fit(sample_returns)

        prob, consecutive = predictor.get_bounce_probability(sample_returns)
        assert 0 <= prob <= 1
        assert consecutive >= 0

    def test_serialization(self, sample_returns):
        """Test to_dict/from_dict."""
        predictor = MarkovPredictor()
        predictor.fit(sample_returns)

        data = predictor.to_dict()
        restored = MarkovPredictor.from_dict(data)

        assert restored.config.buy_threshold == predictor.config.buy_threshold
        assert restored.is_fitted == predictor.is_fitted


# =============================================================================
# MarkovAssetScorer Tests
# =============================================================================

class TestMarkovAssetScorer:
    """Tests for MarkovAssetScorer."""

    def test_init(self):
        """Test initialization."""
        scorer = MarkovAssetScorer()
        assert scorer.config.n_states == 3
        assert scorer.config.lookback_days == 252

    def test_score_symbol(self, sample_returns):
        """Test scoring single symbol."""
        scorer = MarkovAssetScorer()
        score = scorer.score_symbol("TEST", sample_returns)

        assert score is not None
        assert score.symbol == "TEST"
        assert 0 <= score.pi_up <= 1
        assert 0 <= score.composite_score <= 2

    def test_score_universe(self, sample_returns, trending_returns):
        """Test scoring universe."""
        scorer = MarkovAssetScorer()
        returns_dict = {
            "AAPL": sample_returns,
            "MSFT": trending_returns,
        }

        rankings = scorer.score_universe(
            symbols=["AAPL", "MSFT"],
            returns_dict=returns_dict,
        )

        assert isinstance(rankings, pd.DataFrame)
        assert len(rankings) == 2
        assert "rank" in rankings.columns

    def test_filter_top_n(self, sample_returns, trending_returns):
        """Test filtering top N."""
        scorer = MarkovAssetScorer()
        returns_dict = {
            "AAPL": sample_returns,
            "MSFT": trending_returns,
        }

        rankings = scorer.score_universe(
            symbols=["AAPL", "MSFT"],
            returns_dict=returns_dict,
        )

        top = scorer.filter_top_n(rankings, n=1)
        assert len(top) == 1

    def test_analysis_report(self, sample_returns, trending_returns):
        """Test analysis report generation."""
        scorer = MarkovAssetScorer()
        returns_dict = {
            "AAPL": sample_returns,
            "MSFT": trending_returns,
        }

        rankings = scorer.score_universe(
            symbols=["AAPL", "MSFT"],
            returns_dict=returns_dict,
        )

        report = scorer.create_analysis_report(rankings)
        assert "summary" in report
        assert "top_overall" in report


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self, sample_returns):
        """Test full prediction pipeline."""
        # 1. Classify returns
        classifier = StateClassifier(n_states=3, method="threshold")
        classifier.fit(sample_returns)
        states = classifier.classify(sample_returns)

        # 2. Build transition matrix
        tm = TransitionMatrix(n_states=3)
        tm.fit(states)

        # 3. Compute stationary distribution
        sd = StationaryDistribution()
        pi = sd.compute(tm.matrix)

        # 4. Higher-order analysis
        hom = HigherOrderMarkov(order=2, n_states=3)
        hom.fit(states)

        # 5. Generate prediction
        predictor = MarkovPredictor()
        predictor.fit(sample_returns)
        prediction = predictor.predict(returns=sample_returns)

        # Verify all components work together
        assert len(states) == len(sample_returns)
        assert np.isclose(pi.sum(), 1.0)
        assert hom.is_fitted
        assert prediction.signal in ["BUY", "SELL", "HOLD"]

    def test_multi_symbol_workflow(self, sample_returns, trending_returns):
        """Test multi-symbol scoring workflow."""
        returns_dict = {
            "AAPL": sample_returns,
            "MSFT": trending_returns,
            "GOOGL": sample_returns * 0.8,  # Scaled version
        }

        # Score universe
        scorer = MarkovAssetScorer()
        rankings = scorer.score_universe(
            symbols=list(returns_dict.keys()),
            returns_dict=returns_dict,
        )

        # Get different candidate types
        momentum = scorer.get_momentum_candidates(rankings)
        mr = scorer.get_mean_reversion_candidates(rankings)
        bounce = scorer.get_bounce_candidates(rankings)

        # All should be lists
        assert isinstance(momentum, list)
        assert isinstance(mr, list)
        assert isinstance(bounce, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
