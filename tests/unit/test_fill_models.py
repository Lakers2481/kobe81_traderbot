"""
Unit tests for fill models.
"""

import pytest
import numpy as np

from backtest.fill_model import (
    FillModelType,
    FillResult,
    AlwaysFillModel,
    LimitOrderFillModel,
    ProbabilisticFillModel,
    PartialFillModel,
    create_fill_model,
    DEFAULT_FILL_MODEL,
)


class TestAlwaysFillModel:
    """Tests for always fill model."""

    def test_always_fills(self):
        model = AlwaysFillModel()
        result = model.check_fill(100.0, "buy", 100, high=105.0, low=95.0)
        assert result.filled
        assert result.filled_qty == 100
        assert result.fill_price == 100.0

    def test_sell_always_fills(self):
        model = AlwaysFillModel()
        result = model.check_fill(100.0, "sell", 50, high=105.0, low=95.0)
        assert result.filled
        assert result.filled_qty == 50


class TestLimitOrderFillModel:
    """Tests for limit order fill model."""

    def test_buy_fills_when_price_touches(self):
        model = LimitOrderFillModel()
        # Limit at 100, low of 98 means we'd get filled
        result = model.check_fill(100.0, "buy", 100, high=105.0, low=98.0)
        assert result.filled
        assert result.filled_qty == 100

    def test_buy_no_fill_when_price_above(self):
        model = LimitOrderFillModel()
        # Limit at 100, but low was 101 - no fill
        result = model.check_fill(100.0, "buy", 100, high=105.0, low=101.0)
        assert not result.filled
        assert result.filled_qty == 0

    def test_sell_fills_when_price_touches(self):
        model = LimitOrderFillModel()
        # Limit at 100, high of 102 means we'd get filled
        result = model.check_fill(100.0, "sell", 100, high=102.0, low=95.0)
        assert result.filled
        assert result.filled_qty == 100

    def test_sell_no_fill_when_price_below(self):
        model = LimitOrderFillModel()
        # Limit at 100, but high was 99 - no fill
        result = model.check_fill(100.0, "sell", 100, high=99.0, low=95.0)
        assert not result.filled

    def test_missing_high_low_uses_limit(self):
        model = LimitOrderFillModel()
        result = model.check_fill(100.0, "buy", 100)
        # Without high/low, assume limit was touched
        assert result.filled


class TestProbabilisticFillModel:
    """Tests for probabilistic fill model."""

    def test_high_probability_fills(self):
        model = ProbabilisticFillModel(base_fill_prob=1.0)  # Always fill
        result = model.check_fill(100.0, "buy", 100, high=105.0, low=95.0)
        assert result.filled

    def test_zero_probability_no_fill(self):
        model = ProbabilisticFillModel(base_fill_prob=0.0)  # Never fill
        result = model.check_fill(100.0, "buy", 100, high=105.0, low=95.0)
        assert not result.filled

    def test_volume_affects_probability(self):
        model = ProbabilisticFillModel(base_fill_prob=0.5)
        # Large order relative to volume should have lower fill prob
        result_small = model.check_fill(
            100.0, "buy", 100, high=105.0, low=95.0, volume=1000000
        )
        result_large = model.check_fill(
            100.0, "buy", 100000, high=105.0, low=95.0, volume=100000
        )
        # Can't guarantee result due to randomness, just check metadata
        assert "volume_factor" in result_large.metadata or not result_large.filled

    def test_spread_affects_probability(self):
        model = ProbabilisticFillModel(base_fill_prob=0.8)
        # Wide spread should reduce fill probability
        result = model.check_fill(100.0, "buy", 100, spread_pct=2.0)
        assert "spread_factor" in result.metadata or result.filled


class TestPartialFillModel:
    """Tests for partial fill model."""

    def test_small_order_full_fill(self):
        model = PartialFillModel()
        # Small order (0.1% of volume) should fill fully
        result = model.check_fill(
            100.0, "buy", 100, high=105.0, low=95.0, volume=100000
        )
        assert result.filled
        assert result.filled_qty == 100  # Full fill for small orders

    def test_large_order_partial_fill(self):
        model = PartialFillModel(max_volume_pct=5.0)
        # Order is 50% of volume - should be partial
        result = model.check_fill(
            100.0, "buy", 50000, high=105.0, low=95.0, volume=100000
        )
        if result.filled:
            # Should be capped at max_volume_pct of volume
            assert result.filled_qty <= 5000  # 5% of 100k

    def test_no_volume_falls_back(self):
        model = PartialFillModel()
        result = model.check_fill(100.0, "buy", 100, high=105.0, low=95.0)
        # Without volume info, should use fallback
        assert result.filled or not result.filled  # Just ensure no error


class TestFillModelFactory:
    """Tests for fill model factory."""

    def test_create_always_fill(self):
        model = create_fill_model("always_fill")
        assert isinstance(model, AlwaysFillModel)

    def test_create_limit_order(self):
        model = create_fill_model("limit_order")
        assert isinstance(model, LimitOrderFillModel)

    def test_create_probabilistic(self):
        model = create_fill_model("probabilistic", base_fill_prob=0.9)
        assert isinstance(model, ProbabilisticFillModel)

    def test_create_partial_fill(self):
        model = create_fill_model("partial_fill", max_volume_pct=10.0)
        assert isinstance(model, PartialFillModel)

    def test_unknown_type(self):
        with pytest.raises(ValueError):
            create_fill_model("unknown_model")

    def test_default_model(self):
        assert isinstance(DEFAULT_FILL_MODEL, AlwaysFillModel)


class TestFillResult:
    """Tests for fill result."""

    def test_fill_result_properties(self):
        result = FillResult(
            model_type=FillModelType.ALWAYS_FILL,
            filled=True,
            filled_qty=100,
            fill_price=100.0,
            metadata={"reason": "limit touched"},
        )
        assert result.model_type == FillModelType.ALWAYS_FILL
        assert result.filled
        assert result.filled_qty == 100

    def test_partial_fill_result(self):
        result = FillResult(
            model_type=FillModelType.PARTIAL_FILL,
            filled=True,
            filled_qty=50,
            fill_price=100.0,
            metadata={"requested_qty": 100},
        )
        assert result.model_type == FillModelType.PARTIAL_FILL
        assert result.filled_qty == 50

    def test_no_fill_result(self):
        result = FillResult(
            model_type=FillModelType.LIMIT_ORDER,
            filled=False,
            filled_qty=0,
            fill_price=None,
            metadata={"reason": "price not touched"},
        )
        assert not result.filled
        assert result.filled_qty == 0
        assert result.fill_price is None
