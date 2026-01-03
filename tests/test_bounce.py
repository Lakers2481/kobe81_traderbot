#!/usr/bin/env python3
"""
Tests for Bounce Analysis Module

Tests data loading, streak calculation, event generation,
BounceScore computation, and strategy integration.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestStreakAnalyzer:
    """Tests for streak calculation."""

    def test_calculate_streaks_basic(self):
        """Test basic streak calculation."""
        from bounce.streak_analyzer import calculate_streaks_vectorized

        # Create test data: 3 down days, 1 up, 2 down
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=7),
            'close': [100, 99, 98, 97, 100, 99, 98],  # down, down, down, up, down, down
            'open': [100] * 7,
            'high': [101] * 7,
            'low': [96] * 7,
            'volume': [1000] * 7,
        })

        result = calculate_streaks_vectorized(df)

        assert 'down' in result.columns
        assert 'streak_len' in result.columns

        # Check streak lengths
        expected_streaks = [0, 1, 2, 3, 0, 1, 2]  # first day has no prior
        assert list(result['streak_len'].values) == expected_streaks

    def test_calculate_streaks_no_down_days(self):
        """Test with no down days."""
        from bounce.streak_analyzer import calculate_streaks_vectorized

        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [100, 101, 102, 103, 104],  # all up
            'open': [100] * 5,
            'high': [105] * 5,
            'low': [99] * 5,
            'volume': [1000] * 5,
        })

        result = calculate_streaks_vectorized(df)
        assert result['streak_len'].max() == 0

    def test_calculate_streaks_all_down(self):
        """Test with all down days."""
        from bounce.streak_analyzer import calculate_streaks_vectorized

        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [100, 99, 98, 97, 96],  # all down
            'open': [100] * 5,
            'high': [101] * 5,
            'low': [95] * 5,
            'volume': [1000] * 5,
        })

        result = calculate_streaks_vectorized(df)
        expected_streaks = [0, 1, 2, 3, 4]
        assert list(result['streak_len'].values) == expected_streaks


class TestBounceScore:
    """Tests for BounceScore calculation."""

    def test_calculate_bounce_score_perfect(self):
        """Test perfect score scenario."""
        from bounce.bounce_score import calculate_bounce_score

        # Perfect: 100% recovery, 1 day, 10% return, 100 events, 0% drawdown
        score = calculate_bounce_score(
            recovery_rate=1.0,
            avg_days=1.0,
            avg_return=0.10,
            events=100,
            avg_drawdown=0.0
        )

        # Should be close to 100
        assert score >= 95
        assert score <= 100

    def test_calculate_bounce_score_poor(self):
        """Test poor score scenario."""
        from bounce.bounce_score import calculate_bounce_score

        # Poor: 20% recovery, 7 days, -5% return, 5 events, -20% drawdown
        score = calculate_bounce_score(
            recovery_rate=0.20,
            avg_days=7.0,
            avg_return=-0.05,
            events=5,
            avg_drawdown=-0.20
        )

        # Should be low
        assert score < 30

    def test_calculate_bounce_score_bounds(self):
        """Test score is always 0-100."""
        from bounce.bounce_score import calculate_bounce_score

        # Extreme values
        score1 = calculate_bounce_score(2.0, 0.0, 1.0, 1000, 0.5)  # beyond max
        score2 = calculate_bounce_score(-0.5, 20.0, -1.0, 0, -1.0)  # beyond min

        assert 0 <= score1 <= 100
        assert 0 <= score2 <= 100


class TestBounceGates:
    """Tests for bounce gate logic."""

    def test_apply_bounce_gates_pass(self):
        """Test passing all gates."""
        from bounce.bounce_score import apply_bounce_gates

        passed, reason = apply_bounce_gates(
            events=50,
            recovery_rate=0.80,
            avg_days=2.0
        )

        assert passed is True
        assert reason is None

    def test_apply_bounce_gates_low_sample(self):
        """Test failing sample size gate."""
        from bounce.bounce_score import apply_bounce_gates

        passed, reason = apply_bounce_gates(
            events=10,  # < 20
            recovery_rate=0.80,
            avg_days=2.0
        )

        assert passed is False
        assert reason.startswith("LOW_SAMPLE")

    def test_apply_bounce_gates_low_recovery(self):
        """Test failing recovery rate gate."""
        from bounce.bounce_score import apply_bounce_gates

        passed, reason = apply_bounce_gates(
            events=50,
            recovery_rate=0.50,  # < 0.75
            avg_days=2.0
        )

        assert passed is False
        assert reason.startswith("LOW_RECOVERY")

    def test_apply_bounce_gates_slow_recovery(self):
        """Test failing recovery speed gate."""
        from bounce.bounce_score import apply_bounce_gates

        passed, reason = apply_bounce_gates(
            events=50,
            recovery_rate=0.80,
            avg_days=5.0  # > 3.2
        )

        assert passed is False
        assert reason.startswith("SLOW_RECOVERY")


class TestValidation:
    """Tests for lookahead bias prevention."""

    def test_verify_no_lookahead_clean_data(self):
        """Test validation passes with clean data."""
        from bounce.validation import verify_no_lookahead

        # Create clean event data with all required columns
        events_df = pd.DataFrame({
            'ticker': ['AAPL'] * 5,
            'event_date': pd.date_range('2024-01-01', periods=5),
            'streak_n': [1, 2, 3, 1, 2],
            'event_close': [150.0] * 5,
            'recovered_7d_close': [True, False, True, True, False],
            'forward_7d_best_close_return_pct': [5.0, -2.0, 3.0, 4.0, -1.0],
            'max_drawdown_7d_pct': [-1.0, -3.0, -2.0, -1.5, -4.0],
            'edge_case_flag': [None] * 5,
        })

        result = verify_no_lookahead(events_df)

        assert result['passed'] is True
        assert len(result['violations']) == 0


class TestStrategyIntegration:
    """Tests for strategy integration."""

    def test_get_bounce_profile_structure(self):
        """Test bounce profile returns correct structure."""
        from bounce.bounce_score import get_bounce_profile_for_signal

        # Create minimal per-stock data
        per_stock = pd.DataFrame({
            'ticker': ['AAPL'] * 7,
            'streak_n': [1, 2, 3, 4, 5, 6, 7],
            'events': [100, 50, 25, 12, 6, 3, 1],
            'recovery_7d_close_rate': [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            'avg_days_to_recover_7d': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
            'avg_best_7d_return': [0.05, 0.03, 0.01, -0.01, -0.03, -0.05, -0.10],
            'avg_max_drawdown_7d_pct': [-0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08],
            'sample_quality_flag': ['GOOD'] * 7,
        })

        profile = get_bounce_profile_for_signal(
            ticker='AAPL',
            current_streak=3,
            per_stock_5y=per_stock,
            per_stock_10y=per_stock,
        )

        # Check required keys
        assert 'bounce_window_used' in profile
        assert 'events' in profile
        assert 'recovery_rate' in profile
        assert 'bounce_score' in profile
        assert 'gate_passed' in profile


class TestDataLoader:
    """Tests for data loading (integration tests - may need API)."""

    def test_normalize_columns(self):
        """Test column normalization."""
        from bounce.data_loader import _normalize_columns

        df = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [95],
            'Close': [102],
            'Volume': [1000],
        })

        result = _normalize_columns(df)

        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns


class TestEventTable:
    """Tests for event table generation."""

    def test_compute_overall_summary_structure(self):
        """Test overall summary has correct structure."""
        from bounce.event_table import compute_overall_summary

        # Create minimal events data with all required columns
        events_df = pd.DataFrame({
            'ticker': ['AAPL'] * 10 + ['MSFT'] * 10,
            'event_date': pd.date_range('2024-01-01', periods=20),
            'streak_n': [1, 2, 3, 1, 2, 1, 1, 2, 3, 4] * 2,
            'event_close': [150.0] * 20,
            'recovered_7d_close': [True, False, True, True, False] * 4,
            'recovered_7d_high': [True, False, True, True, False] * 4,
            'days_to_recover_close': [1.0, np.nan, 2.0, 1.0, np.nan] * 4,
            'days_to_recover_high': [1.0, np.nan, 2.0, 1.0, np.nan] * 4,
            'forward_7d_best_close_return_pct': [5.0, -2.0, 3.0, 4.0, -1.0] * 4,
            'forward_7d_max_high_return_pct': [6.0, -1.0, 4.0, 5.0, 0.0] * 4,
            'max_drawdown_7d_pct': [-1.0, -3.0, -2.0, -1.5, -4.0] * 4,
            'edge_case_flag': [None] * 20,
        })

        result = compute_overall_summary(events_df)

        assert 'streak_n' in result.columns
        assert 'events' in result.columns
        assert 'recovery_7d_close_rate' in result.columns


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
