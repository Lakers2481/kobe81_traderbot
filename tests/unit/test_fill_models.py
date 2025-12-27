"""
Unit tests for fill models.
"""

import pytest

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
        result = model.calculate_fill(
            order_side="buy",
            limit_price=100.0,
            qty=100,
            bar_open=99.0,
            bar_high=105.0,
            bar_low=95.0,
            bar_close=100.0,
        )
        assert result.would_fill
        assert result.fill_qty == 100
        assert result.fill_price == 100.0

    def test_sell_always_fills(self):
        model = AlwaysFillModel()
        result = model.calculate_fill(
            order_side="sell",
            limit_price=100.0,
            qty=50,
            bar_open=99.0,
            bar_high=105.0,
            bar_low=95.0,
            bar_close=100.0,
        )
        assert result.would_fill
        assert result.fill_qty == 50


class TestLimitOrderFillModel:
    """Tests for limit order fill model."""

    def test_buy_fills_when_price_touches(self):
        model = LimitOrderFillModel()
        # Limit at 100, low of 98 means we'd get filled
        result = model.calculate_fill(
            order_side="buy",
            limit_price=100.0,
            qty=100,
            bar_open=99.0,
            bar_high=105.0,
            bar_low=98.0,
            bar_close=102.0,
        )
        assert result.would_fill
        assert result.fill_qty == 100

    def test_buy_no_fill_when_price_above(self):
        model = LimitOrderFillModel()
        # Limit at 100, but low was 101 - no fill
        result = model.calculate_fill(
            order_side="buy",
            limit_price=100.0,
            qty=100,
            bar_open=102.0,
            bar_high=105.0,
            bar_low=101.0,
            bar_close=103.0,
        )
        assert not result.would_fill
        assert result.fill_qty == 0

    def test_sell_fills_when_price_touches(self):
        model = LimitOrderFillModel()
        # Limit at 100, high of 102 means we'd get filled
        result = model.calculate_fill(
            order_side="sell",
            limit_price=100.0,
            qty=100,
            bar_open=99.0,
            bar_high=102.0,
            bar_low=95.0,
            bar_close=98.0,
        )
        assert result.would_fill
        assert result.fill_qty == 100

    def test_sell_no_fill_when_price_below(self):
        model = LimitOrderFillModel()
        # Limit at 100, but high was 99 - no fill
        result = model.calculate_fill(
            order_side="sell",
            limit_price=100.0,
            qty=100,
            bar_open=97.0,
            bar_high=99.0,
            bar_low=95.0,
            bar_close=96.0,
        )
        assert not result.would_fill

    def test_gap_through_limit(self):
        model = LimitOrderFillModel(use_open_for_gap=True)
        # Buy limit at 100, but bar opens at 98 (gapped below limit)
        result = model.calculate_fill(
            order_side="buy",
            limit_price=100.0,
            qty=100,
            bar_open=98.0,
            bar_high=102.0,
            bar_low=97.0,
            bar_close=101.0,
        )
        assert result.would_fill
        assert result.fill_price == 98.0  # Filled at open, not limit


class TestProbabilisticFillModel:
    """Tests for probabilistic fill model."""

    def test_high_probability_fills(self):
        model = ProbabilisticFillModel(base_probability=1.0)
        result = model.calculate_fill(
            order_side="buy",
            limit_price=110.0,  # Above bar_high of 105
            qty=100,
            bar_open=99.0,
            bar_high=105.0,
            bar_low=95.0,
            bar_close=102.0,
        )
        # With limit above range, should have 100% probability
        assert result.fill_probability == 1.0

    def test_zero_probability_no_fill(self):
        model = ProbabilisticFillModel(base_probability=0.0)
        result = model.calculate_fill(
            order_side="buy",
            limit_price=96.0,  # Just inside range
            qty=100,
            bar_open=99.0,
            bar_high=105.0,
            bar_low=95.0,
            bar_close=102.0,
        )
        # With base_probability=0, probability should be 0
        assert result.fill_probability == 0.0

    def test_limit_below_range_no_fill(self):
        model = ProbabilisticFillModel(base_probability=0.95)
        result = model.calculate_fill(
            order_side="buy",
            limit_price=90.0,  # Below bar low of 95
            qty=100,
            bar_open=99.0,
            bar_high=105.0,
            bar_low=95.0,
            bar_close=102.0,
        )
        assert result.fill_probability == 0.0

    def test_limit_above_range_guaranteed_fill(self):
        model = ProbabilisticFillModel(base_probability=0.5)
        result = model.calculate_fill(
            order_side="buy",
            limit_price=110.0,  # Above bar high of 105
            qty=100,
            bar_open=99.0,
            bar_high=105.0,
            bar_low=95.0,
            bar_close=102.0,
        )
        assert result.fill_probability == 1.0


class TestPartialFillModel:
    """Tests for partial fill model."""

    def test_small_order_full_fill(self):
        model = PartialFillModel(max_participation_rate=0.10)
        # Small order relative to volume should fill fully
        result = model.calculate_fill(
            order_side="buy",
            limit_price=100.0,
            qty=100,
            bar_open=99.0,
            bar_high=105.0,
            bar_low=95.0,
            bar_close=102.0,
            bar_volume=100000,  # 0.1% of volume
        )
        assert result.would_fill
        assert result.fill_qty == 100  # Full fill for small orders

    def test_large_order_partial_fill(self):
        # Use min_fill_ratio=0.0 to disable the minimum guarantee
        model = PartialFillModel(max_participation_rate=0.05, min_fill_ratio=0.0)
        # Order is 50% of volume - should be capped at 5% participation
        result = model.calculate_fill(
            order_side="buy",
            limit_price=100.0,
            qty=50000,
            bar_open=99.0,
            bar_high=105.0,
            bar_low=95.0,
            bar_close=102.0,
            bar_volume=100000,
        )
        if result.would_fill:
            # Should be capped at max_participation_rate of volume
            assert result.fill_qty <= 5000  # 5% of 100k

    def test_no_volume_falls_back(self):
        model = PartialFillModel()
        result = model.calculate_fill(
            order_side="buy",
            limit_price=100.0,
            qty=100,
            bar_open=99.0,
            bar_high=105.0,
            bar_low=95.0,
            bar_close=102.0,
            bar_volume=None,  # No volume data
        )
        # Without volume info, should assume full fill
        assert result.would_fill
        assert result.fill_qty == 100


class TestFillModelFactory:
    """Tests for fill model factory."""

    def test_create_always_fill(self):
        model = create_fill_model("always_fill")
        assert isinstance(model, AlwaysFillModel)

    def test_create_limit_order(self):
        model = create_fill_model("limit_order")
        assert isinstance(model, LimitOrderFillModel)

    def test_create_probabilistic(self):
        model = create_fill_model("probabilistic", base_probability=0.9)
        assert isinstance(model, ProbabilisticFillModel)

    def test_create_partial_fill(self):
        model = create_fill_model("partial_fill", max_participation_rate=0.10)
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
            would_fill=True,
            fill_qty=100,
            fill_price=100.0,
            fill_probability=1.0,
            metadata={"reason": "limit touched"},
        )
        assert result.model_type == FillModelType.ALWAYS_FILL
        assert result.would_fill
        assert result.fill_qty == 100

    def test_partial_fill_result(self):
        result = FillResult(
            model_type=FillModelType.PARTIAL_FILL,
            would_fill=True,
            fill_qty=50,
            fill_price=100.0,
            fill_probability=0.5,
            metadata={"requested_qty": 100},
        )
        assert result.model_type == FillModelType.PARTIAL_FILL
        assert result.fill_qty == 50

    def test_no_fill_result(self):
        result = FillResult(
            model_type=FillModelType.LIMIT_ORDER,
            would_fill=False,
            fill_qty=0,
            fill_price=0.0,
            fill_probability=0.0,
            metadata={"reason": "price not touched"},
        )
        assert not result.would_fill
        assert result.fill_qty == 0
