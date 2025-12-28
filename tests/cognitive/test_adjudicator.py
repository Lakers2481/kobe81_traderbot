"""
Unit tests for cognitive/adjudicator.py

Tests the final judgment layer for trading signal adjudication.
"""
import pytest


class TestVerdictDataclass:
    """Tests for the Verdict dataclass."""

    def test_verdict_creation(self):
        from cognitive.adjudicator import Verdict

        verdict = Verdict(
            approved=True,
            score=0.85,
            reasons=["good_setup"],
        )

        assert verdict.approved is True
        assert verdict.score == 0.85
        assert "good_setup" in verdict.reasons

    def test_verdict_rejected(self):
        from cognitive.adjudicator import Verdict

        verdict = Verdict(
            approved=False,
            score=0.3,
            reasons=["low_price", "wide_stop"],
        )

        assert verdict.approved is False
        assert verdict.score == 0.3
        assert len(verdict.reasons) == 2


class TestHeuristicFunction:
    """Tests for the heuristic adjudication function."""

    def test_good_signal_approved(self):
        from cognitive.adjudicator import heuristic

        signal = {
            'symbol': 'AAPL',
            'entry_price': 150.0,
            'stop_loss': 147.0,  # 2% stop, well within 5% threshold
            'take_profit': 160.0,
        }

        verdict = heuristic(signal)

        assert verdict.approved is True
        assert verdict.score >= 0.6
        assert len(verdict.reasons) == 0

    def test_penny_stock_penalized(self):
        from cognitive.adjudicator import heuristic

        signal = {
            'symbol': 'PENNY',
            'entry_price': 2.50,  # Below $5 threshold
            'stop_loss': 2.40,
            'take_profit': 3.00,
        }

        verdict = heuristic(signal)

        assert 'low_price' in verdict.reasons
        assert verdict.score < 1.0

    def test_wide_stop_penalized(self):
        from cognitive.adjudicator import heuristic

        signal = {
            'symbol': 'RISKY',
            'entry_price': 100.0,
            'stop_loss': 90.0,  # 10% stop, exceeds 5% threshold
            'take_profit': 120.0,
        }

        verdict = heuristic(signal)

        assert 'wide_stop' in verdict.reasons
        assert verdict.score < 1.0

    def test_multiple_penalties(self):
        from cognitive.adjudicator import heuristic

        signal = {
            'symbol': 'BAD',
            'entry_price': 3.00,  # Penny stock
            'stop_loss': 2.50,   # Wide stop (>5%)
            'take_profit': 4.00,
        }

        verdict = heuristic(signal)

        assert 'low_price' in verdict.reasons
        assert 'wide_stop' in verdict.reasons
        assert verdict.approved is False

    def test_score_clamped_to_zero(self):
        from cognitive.adjudicator import heuristic

        # Signal with many penalties should not go below 0
        signal = {
            'entry_price': 1.0,  # Very low price
            'stop_loss': 0.5,   # Very wide stop
        }

        verdict = heuristic(signal)

        assert verdict.score >= 0.0

    def test_handles_missing_prices(self):
        from cognitive.adjudicator import heuristic

        signal = {
            'symbol': 'TEST',
            # Missing entry_price and stop_loss
        }

        # Should not raise an exception
        verdict = heuristic(signal)

        assert isinstance(verdict.score, float)

    def test_handles_invalid_prices(self):
        from cognitive.adjudicator import heuristic
        import pytest

        signal = {
            'entry_price': 'invalid',
            'stop_loss': 'also_invalid',
        }

        # Invalid price strings should raise an exception
        with pytest.raises(ValueError):
            heuristic(signal)


class TestAdjudicateFunction:
    """Tests for the main adjudicate function."""

    def test_adjudicate_returns_verdict(self):
        from cognitive.adjudicator import adjudicate, Verdict

        signal = {
            'symbol': 'AAPL',
            'entry_price': 150.0,
            'stop_loss': 147.0,
        }

        verdict = adjudicate(signal)

        assert isinstance(verdict, Verdict)

    def test_adjudicate_good_signal(self):
        from cognitive.adjudicator import adjudicate

        signal = {
            'symbol': 'MSFT',
            'entry_price': 380.0,
            'stop_loss': 375.0,
            'take_profit': 395.0,
        }

        verdict = adjudicate(signal)

        assert verdict.approved is True

    def test_adjudicate_bad_signal(self):
        from cognitive.adjudicator import adjudicate

        signal = {
            'symbol': 'PENNY',
            'entry_price': 2.0,
            'stop_loss': 1.5,
        }

        verdict = adjudicate(signal)

        # With low price and wide stop, should be rejected
        assert verdict.approved is False

    def test_adjudicate_empty_signal(self):
        from cognitive.adjudicator import adjudicate, Verdict

        signal = {}

        # Should not raise an exception
        verdict = adjudicate(signal)

        assert isinstance(verdict, Verdict)


class TestScoreThresholds:
    """Tests for approval score thresholds."""

    def test_score_exactly_at_threshold(self):
        from cognitive.adjudicator import heuristic

        # A signal with exactly one minor penalty
        signal = {
            'entry_price': 100.0,
            'stop_loss': 93.0,  # Slightly over 5% = one penalty of 0.2
        }

        verdict = heuristic(signal)

        # Score should be 1.0 - 0.2 = 0.8, which is >= 0.6
        assert verdict.score == 0.8
        assert verdict.approved is True

    def test_score_just_below_threshold(self):
        from cognitive.adjudicator import heuristic

        # A signal with penalties that bring score below 0.6
        signal = {
            'entry_price': 4.5,  # Low price penalty: -0.3
            'stop_loss': 4.0,   # Wide stop penalty: -0.2
        }

        verdict = heuristic(signal)

        # Score: 1.0 - 0.3 - 0.2 = 0.5, which is < 0.6 (allow for floating point)
        assert abs(verdict.score - 0.5) < 0.01
        assert verdict.approved is False


class TestEdgeCases:
    """Tests for edge cases in adjudication."""

    def test_zero_entry_price(self):
        from cognitive.adjudicator import heuristic

        signal = {
            'entry_price': 0.0,
            'stop_loss': 0.0,
        }

        # Should handle gracefully
        verdict = heuristic(signal)
        assert isinstance(verdict.score, float)

    def test_negative_prices(self):
        from cognitive.adjudicator import heuristic

        signal = {
            'entry_price': -10.0,
            'stop_loss': -15.0,
        }

        # Should handle gracefully (though unusual)
        verdict = heuristic(signal)
        assert isinstance(verdict.score, float)

    def test_none_values(self):
        from cognitive.adjudicator import heuristic
        import pytest

        signal = {
            'entry_price': None,
            'stop_loss': None,
        }

        # None values should raise a TypeError
        with pytest.raises(TypeError):
            heuristic(signal)


class TestIntegration:
    """Integration tests for adjudicator."""

    def test_adjudicator_with_full_signal(self):
        from cognitive.adjudicator import adjudicate, Verdict

        signal = {
            'symbol': 'GOOGL',
            'side': 'BUY',
            'strategy': 'ibs_rsi',
            'entry_price': 140.0,
            'stop_loss': 137.0,
            'take_profit': 150.0,
            'timestamp': '2025-01-01T10:00:00',
            'confidence': 0.75,
        }

        verdict = adjudicate(signal)

        assert isinstance(verdict, Verdict)
        assert verdict.score >= 0.0
        assert verdict.score <= 1.0
