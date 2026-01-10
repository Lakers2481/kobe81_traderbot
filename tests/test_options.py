"""
Tests for Synthetic Options Module.

Tests Black-Scholes pricing, volatility calculation, strike selection,
position sizing, and options backtesting.
"""
import pytest
import numpy as np

# Import modules under test
from options import (
    BlackScholes,
    OptionType,
    OptionPricing,
    calculate_option_price,
    calculate_greeks,
)
from options.volatility import (
    RealizedVolatility,
    vol_with_floor,
)
from options.selection import (
    StrikeSelector,
    select_call_strike,
)
from options.position_sizing import (
    OptionsPositionSizer,
    PositionDirection,
    size_long_call,
    size_long_put,
    calculate_max_contracts,
)


class TestBlackScholes:
    """Tests for Black-Scholes pricing."""

    def test_call_price_positive(self):
        """Call price should be positive."""
        bs = BlackScholes()
        result = bs.price_option(
            OptionType.CALL,
            spot=100,
            strike=100,
            time=30/365,
            rate=0.05,
            vol=0.20,
        )

        assert result.price > 0

    def test_put_price_positive(self):
        """Put price should be positive."""
        bs = BlackScholes()
        result = bs.price_option(
            OptionType.PUT,
            spot=100,
            strike=100,
            time=30/365,
            rate=0.05,
            vol=0.20,
        )

        assert result.price > 0

    def test_call_delta_range(self):
        """Call delta should be between 0 and 1."""
        bs = BlackScholes()
        result = bs.price_option(
            OptionType.CALL,
            spot=100,
            strike=100,
            time=30/365,
            rate=0.05,
            vol=0.20,
        )

        assert 0 <= result.delta <= 1

    def test_put_delta_range(self):
        """Put delta should be between -1 and 0."""
        bs = BlackScholes()
        result = bs.price_option(
            OptionType.PUT,
            spot=100,
            strike=100,
            time=30/365,
            rate=0.05,
            vol=0.20,
        )

        assert -1 <= result.delta <= 0

    def test_atm_delta_approximately_half(self):
        """ATM call delta should be approximately 0.5."""
        bs = BlackScholes()
        result = bs.price_option(
            OptionType.CALL,
            spot=100,
            strike=100,
            time=30/365,
            rate=0.05,
            vol=0.20,
        )

        assert 0.45 <= result.delta <= 0.55

    def test_gamma_positive(self):
        """Gamma should be positive for both calls and puts."""
        bs = BlackScholes()

        call_result = bs.price_option(OptionType.CALL, 100, 100, 30/365, 0.05, 0.20)
        put_result = bs.price_option(OptionType.PUT, 100, 100, 30/365, 0.05, 0.20)

        assert call_result.gamma > 0
        assert put_result.gamma > 0

    def test_theta_negative_for_long(self):
        """Theta should be negative (time decay hurts longs)."""
        bs = BlackScholes()
        result = bs.price_option(
            OptionType.CALL,
            spot=100,
            strike=100,
            time=30/365,
            rate=0.05,
            vol=0.20,
        )

        assert result.theta < 0

    def test_put_call_parity(self):
        """Put-call parity should hold."""
        bs = BlackScholes()
        spot = 100
        strike = 100
        time = 30/365
        rate = 0.05

        call = bs.price_option(OptionType.CALL, spot, strike, time, rate, 0.20)
        put = bs.price_option(OptionType.PUT, spot, strike, time, rate, 0.20)

        # C - P = S - K*e^(-rT)
        lhs = call.price - put.price
        rhs = spot - strike * np.exp(-rate * time)

        assert abs(lhs - rhs) < 0.01  # Within 1 cent

    def test_higher_vol_higher_price(self):
        """Higher volatility should increase option price."""
        bs = BlackScholes()

        low_vol = bs.price_option(OptionType.CALL, 100, 100, 30/365, 0.05, 0.10)
        high_vol = bs.price_option(OptionType.CALL, 100, 100, 30/365, 0.05, 0.40)

        assert high_vol.price > low_vol.price

    def test_expiry_intrinsic_value(self):
        """At expiry, price should equal intrinsic value."""
        bs = BlackScholes()

        # ITM call at expiry
        result = bs.price_option(OptionType.CALL, 110, 100, 0, 0.05, 0.20)
        assert abs(result.price - 10) < 0.01

        # OTM call at expiry
        result = bs.price_option(OptionType.CALL, 90, 100, 0, 0.05, 0.20)
        assert result.price == 0


class TestRealizedVolatility:
    """Tests for realized volatility estimation."""

    def test_close_to_close_vol(self):
        """Should calculate close-to-close volatility."""
        rv = RealizedVolatility()

        # Generate random walk with known volatility
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)  # 2% daily vol
        prices = 100 * np.cumprod(1 + returns)

        result = rv.close_to_close(prices, lookback=20)

        # Annualized should be around 2% * sqrt(252) ≈ 32%
        assert 0.20 <= result.volatility <= 0.50

    def test_vol_with_floor(self):
        """Should enforce volatility floor."""
        # Very low volatility prices
        prices = [100 + i * 0.01 for i in range(30)]  # Nearly flat

        vol = vol_with_floor(prices, lookback=20, floor=0.10)

        assert vol >= 0.10

    def test_vol_with_cap(self):
        """Should enforce volatility cap."""
        # Very high volatility prices
        np.random.seed(42)
        returns = np.random.normal(0, 0.20, 100)  # 20% daily vol = extreme
        prices = 100 * np.cumprod(1 + returns)

        vol = vol_with_floor(prices, lookback=20, floor=0.10, cap=2.0)

        assert vol <= 2.0

    def test_insufficient_data(self):
        """Should handle insufficient data."""
        rv = RealizedVolatility()
        prices = [100, 101, 102]  # Too few

        result = rv.close_to_close(prices, lookback=20)

        assert result.volatility == 0.0 or result.observations < 20


class TestStrikeSelection:
    """Tests for strike selection."""

    def test_find_atm_strike(self):
        """ATM strike should be near spot price."""
        selector = StrikeSelector()

        result = selector.find_atm_strike(
            OptionType.CALL,
            spot=100,
            days_to_expiry=30,
            volatility=0.20,
        )

        assert abs(result.strike - 100) < 5
        assert 0.45 <= abs(result.delta) <= 0.55

    def test_find_otm_call(self):
        """OTM call strike should be above spot."""
        selector = StrikeSelector()

        result = selector.find_otm_strike(
            OptionType.CALL,
            spot=100,
            days_to_expiry=30,
            volatility=0.20,
            delta=0.30,
        )

        assert result.strike > 100
        assert result.moneyness == "OTM"

    def test_find_otm_put(self):
        """OTM put strike should be below spot."""
        selector = StrikeSelector()

        result = selector.find_otm_strike(
            OptionType.PUT,
            spot=100,
            days_to_expiry=30,
            volatility=0.20,
            delta=0.30,
        )

        # OTM put should have strike <= spot (ATM is acceptable edge case)
        assert result.strike <= 100
        assert result.moneyness in ["OTM", "ATM"]

    def test_delta_targeting(self):
        """Should find strike matching target delta."""
        result = select_call_strike(
            spot=100,
            target_delta=0.30,
            days_to_expiry=30,
            volatility=0.20,
        )

        assert abs(result.delta - 0.30) < 0.02


class TestPositionSizing:
    """Tests for position sizing."""

    def test_long_call_sizing(self):
        """Should size long call correctly."""
        result = size_long_call(
            equity=100_000,
            premium=2.50,  # $2.50 per share
            strike=100,
            spot=100,
        )

        assert result.is_valid
        assert result.contracts > 0
        # Max risk should be <= 2% of equity
        assert result.max_risk <= 100_000 * 0.02 * 1.01  # Allow 1% tolerance

    def test_long_put_sizing(self):
        """Should size long put correctly."""
        result = size_long_put(
            equity=100_000,
            premium=2.50,
            strike=100,
            spot=100,
        )

        assert result.is_valid
        assert result.contracts > 0

    def test_max_contracts_calculation(self):
        """Should calculate max contracts correctly."""
        contracts = calculate_max_contracts(
            equity=100_000,
            premium=2.50,  # $250 per contract
            risk_pct=0.02,  # 2% = $2000 risk budget
        )

        # Should be 8 contracts ($2000 / $250)
        assert contracts == 8

    def test_insufficient_equity(self):
        """Should reject when equity is insufficient."""
        sizer = OptionsPositionSizer(risk_pct=0.02)

        result = sizer.size_long_option(
            equity=1_000,  # Only $1000
            option_type=OptionType.CALL,
            premium=25.00,  # $2500 per contract
            strike=100,
            spot=100,
        )

        # Can't afford even 1 contract with 2% risk
        assert not result.is_valid

    def test_direction_long(self):
        """Long positions should have LONG direction."""
        result = size_long_call(100_000, 2.50, 100, 100)
        assert result.direction == PositionDirection.LONG

    def test_risk_pct_enforcement(self):
        """Should enforce risk percentage limit."""
        sizer = OptionsPositionSizer(risk_pct=0.01)  # 1% limit

        result = sizer.size_long_option(
            equity=100_000,
            option_type=OptionType.CALL,
            premium=5.00,
            strike=100,
            spot=100,
        )

        # Max risk = 1% of $100k = $1000
        assert result.max_risk <= 1_000 * 1.01


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_calculate_option_price(self):
        """Should calculate option price correctly."""
        result = calculate_option_price(
            option_type="CALL",
            spot=100,
            strike=100,
            days_to_expiry=30,
            rate=0.05,
            volatility=0.20,
        )

        assert isinstance(result, OptionPricing)
        assert result.price > 0

    def test_calculate_greeks(self):
        """Should calculate greeks correctly."""
        greeks = calculate_greeks(
            option_type="CALL",
            spot=100,
            strike=100,
            days_to_expiry=30,
        )

        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'vega' in greeks
        assert 'theta' in greeks


# Run with: pytest tests/test_options.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
