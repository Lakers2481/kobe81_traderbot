"""
Unit tests for slippage models.
"""

import pytest

from backtest.slippage import (
    SlippageType,
    SlippageResult,
    ZeroSlippage,
    FixedBpsSlippage,
    ATRFractionSlippage,
    SpreadPercentileSlippage,
    VolumeImpactSlippage,
    create_slippage_model,
    DEFAULT_SLIPPAGE_MODEL,
)


class TestZeroSlippage:
    """Tests for zero slippage model."""

    def test_no_slippage(self):
        model = ZeroSlippage()
        result = model.calculate(100.0, "buy", 100)
        assert result.slippage_amount == 0.0
        assert result.adjusted_price == 100.0
        assert result.slippage_bps == 0.0


class TestFixedBpsSlippage:
    """Tests for fixed basis points slippage."""

    def test_default_5bps(self):
        model = FixedBpsSlippage()
        result = model.calculate(100.0, "buy", 100)
        assert result.slippage_bps == 5.0
        assert result.slippage_amount == pytest.approx(0.05, rel=0.01)

    def test_custom_bps(self):
        model = FixedBpsSlippage(bps=10.0)
        result = model.calculate(100.0, "buy", 100)
        assert result.slippage_bps == 10.0
        assert result.slippage_amount == pytest.approx(0.10, rel=0.01)

    def test_buy_adds_slippage(self):
        model = FixedBpsSlippage(bps=10.0)
        result = model.calculate(100.0, "buy", 100)
        assert result.adjusted_price > 100.0

    def test_sell_subtracts_slippage(self):
        model = FixedBpsSlippage(bps=10.0)
        result = model.calculate(100.0, "sell", 100)
        assert result.adjusted_price < 100.0


class TestATRFractionSlippage:
    """Tests for ATR-based slippage."""

    def test_with_atr(self):
        model = ATRFractionSlippage(atr_fraction=0.10)
        result = model.calculate(100.0, "buy", 100, atr=2.0)
        # Slippage = 2.0 * 0.10 = 0.20
        assert result.slippage_amount == pytest.approx(0.20, rel=0.01)

    def test_fallback_without_atr(self):
        model = ATRFractionSlippage()
        result = model.calculate(100.0, "buy", 100, atr=None)
        # Falls back to 5 bps
        assert result.slippage_bps == 5.0
        assert result.metadata.get("fallback") is True

    def test_zero_atr(self):
        model = ATRFractionSlippage()
        result = model.calculate(100.0, "buy", 100, atr=0.0)
        # Falls back to 5 bps
        assert result.slippage_bps == 5.0


class TestSpreadPercentileSlippage:
    """Tests for spread-based slippage."""

    def test_with_spread(self):
        model = SpreadPercentileSlippage(spread_fraction=0.50)
        result = model.calculate(100.0, "buy", 100, spread=0.20)
        # Slippage = 0.20 * 0.50 = 0.10
        assert result.slippage_amount == pytest.approx(0.10, rel=0.01)

    def test_with_spread_pct(self):
        model = SpreadPercentileSlippage(spread_fraction=0.50)
        result = model.calculate(100.0, "buy", 100, spread_pct=0.20)  # 0.2%
        # Spread = 100 * 0.002 = 0.20, Slippage = 0.20 * 0.50 = 0.10
        assert result.slippage_amount == pytest.approx(0.10, rel=0.01)

    def test_estimated_spread(self):
        model = SpreadPercentileSlippage()
        # High-priced stock should have tighter estimated spread
        result = model.calculate(200.0, "buy", 100)
        # Spread estimated at 2 bps = 0.04, slippage = 0.04 * 0.50 = 0.02
        assert result.slippage_amount < 0.10


class TestVolumeImpactSlippage:
    """Tests for volume impact slippage."""

    def test_small_order(self):
        model = VolumeImpactSlippage()
        result = model.calculate(100.0, "buy", 100, daily_volume=1000000, volatility=0.02)
        # Small participation = small impact
        assert result.slippage_bps >= 2.0  # Min floor
        assert result.slippage_bps < 10.0

    def test_large_order(self):
        model = VolumeImpactSlippage()
        result = model.calculate(100.0, "buy", 10000, daily_volume=100000, volatility=0.03)
        # 10% participation = higher impact (compare to small order baseline)
        # With square-root impact model, ~9.5 bps is expected for 10% participation
        assert result.slippage_bps > 5.0  # Higher than small order baseline

    def test_max_cap(self):
        model = VolumeImpactSlippage(max_bps=50.0)
        result = model.calculate(100.0, "buy", 50000, daily_volume=100000, volatility=0.05)
        # Should be capped at 50 bps
        assert result.slippage_bps <= 50.0

    def test_fallback_no_volume(self):
        model = VolumeImpactSlippage()
        result = model.calculate(100.0, "buy", 100, daily_volume=None)
        # Falls back to 10 bps
        assert result.slippage_bps == 10.0
        assert result.metadata.get("fallback") is True


class TestSlippageFactory:
    """Tests for slippage model factory."""

    def test_create_zero(self):
        model = create_slippage_model("zero")
        assert isinstance(model, ZeroSlippage)

    def test_create_fixed_bps(self):
        model = create_slippage_model("fixed_bps", bps=15.0)
        assert isinstance(model, FixedBpsSlippage)
        result = model.calculate(100.0, "buy", 100)
        assert result.slippage_bps == 15.0

    def test_create_atr_fraction(self):
        model = create_slippage_model("atr_fraction", atr_fraction=0.15)
        assert isinstance(model, ATRFractionSlippage)

    def test_create_spread_percentile(self):
        model = create_slippage_model("spread_percentile")
        assert isinstance(model, SpreadPercentileSlippage)

    def test_create_volume_impact(self):
        model = create_slippage_model("volume_impact", min_bps=5.0, max_bps=75.0)
        assert isinstance(model, VolumeImpactSlippage)

    def test_unknown_type(self):
        with pytest.raises(ValueError):
            create_slippage_model("unknown_model")

    def test_default_model(self):
        assert isinstance(DEFAULT_SLIPPAGE_MODEL, FixedBpsSlippage)


class TestSlippageResult:
    """Tests for slippage result."""

    def test_slippage_pct(self):
        result = SlippageResult(
            model_type=SlippageType.FIXED_BPS,
            base_price=100.0,
            slippage_amount=0.05,
            adjusted_price=100.05,
            slippage_bps=5.0,
            metadata={},
        )
        assert result.slippage_pct == pytest.approx(0.05, rel=0.01)
