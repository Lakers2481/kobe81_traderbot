"""
Black-Scholes Options Pricing Model.

Provides options pricing and Greeks calculation:
- European option pricing (calls and puts)
- Greeks: Delta, Gamma, Vega, Theta, Rho
- Implied volatility calculation via Newton-Raphson
- Protective put recommendations
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OptionType(Enum):
    """Option type."""
    CALL = "CALL"
    PUT = "PUT"


@dataclass
class OptionPricing:
    """Option pricing result with Greeks."""
    # Inputs
    option_type: OptionType
    spot_price: float
    strike_price: float
    time_to_expiry: float  # Years
    risk_free_rate: float
    volatility: float
    dividend_yield: float = 0.0

    # Outputs
    price: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0

    # Additional info
    intrinsic_value: float = 0.0
    time_value: float = 0.0
    probability_itm: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "option_type": self.option_type.value,
            "spot_price": round(self.spot_price, 2),
            "strike_price": round(self.strike_price, 2),
            "time_to_expiry_years": round(self.time_to_expiry, 4),
            "time_to_expiry_days": round(self.time_to_expiry * 365, 0),
            "risk_free_rate": round(self.risk_free_rate, 4),
            "volatility": round(self.volatility, 4),
            "dividend_yield": round(self.dividend_yield, 4),
            "price": round(self.price, 4),
            "greeks": {
                "delta": round(self.delta, 4),
                "gamma": round(self.gamma, 6),
                "vega": round(self.vega, 4),
                "theta": round(self.theta, 4),
                "rho": round(self.rho, 4),
            },
            "intrinsic_value": round(self.intrinsic_value, 4),
            "time_value": round(self.time_value, 4),
            "probability_itm": round(self.probability_itm, 4),
        }


class BlackScholes:
    """
    Black-Scholes Option Pricing Model.

    Calculates European option prices and Greeks.
    """

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Standard normal probability density function."""
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def calculate_d1_d2(
        self,
        spot: float,
        strike: float,
        time: float,
        rate: float,
        vol: float,
        div: float = 0.0
    ) -> tuple:
        """
        Calculate d1 and d2 parameters.

        Args:
            spot: Current stock price
            strike: Option strike price
            time: Time to expiry in years
            rate: Risk-free interest rate
            vol: Volatility (annualized)
            div: Dividend yield

        Returns:
            Tuple of (d1, d2)
        """
        if time <= 0 or vol <= 0:
            return 0.0, 0.0

        d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol ** 2) * time) / (vol * math.sqrt(time))
        d2 = d1 - vol * math.sqrt(time)

        return d1, d2

    def price_option(
        self,
        option_type: OptionType,
        spot: float,
        strike: float,
        time: float,
        rate: float,
        vol: float,
        div: float = 0.0
    ) -> OptionPricing:
        """
        Calculate option price and Greeks.

        Args:
            option_type: CALL or PUT
            spot: Current stock price
            strike: Option strike price
            time: Time to expiry in years
            rate: Risk-free interest rate (decimal, e.g., 0.05 for 5%)
            vol: Volatility (annualized, decimal, e.g., 0.20 for 20%)
            div: Dividend yield (decimal)

        Returns:
            OptionPricing with price and Greeks
        """
        result = OptionPricing(
            option_type=option_type,
            spot_price=spot,
            strike_price=strike,
            time_to_expiry=time,
            risk_free_rate=rate,
            volatility=vol,
            dividend_yield=div,
        )

        # Handle edge cases
        if time <= 0:
            # At expiry
            if option_type == OptionType.CALL:
                result.price = max(0, spot - strike)
                result.delta = 1.0 if spot > strike else 0.0
            else:
                result.price = max(0, strike - spot)
                result.delta = -1.0 if spot < strike else 0.0
            result.intrinsic_value = result.price
            return result

        if vol <= 0:
            # Zero volatility - deterministic
            pv_strike = strike * math.exp(-rate * time)
            if option_type == OptionType.CALL:
                result.price = max(0, spot * math.exp(-div * time) - pv_strike)
            else:
                result.price = max(0, pv_strike - spot * math.exp(-div * time))
            return result

        # Calculate d1 and d2
        d1, d2 = self.calculate_d1_d2(spot, strike, time, rate, vol, div)

        # Calculate price
        exp_div = math.exp(-div * time)
        exp_rate = math.exp(-rate * time)

        if option_type == OptionType.CALL:
            result.price = spot * exp_div * self._norm_cdf(d1) - strike * exp_rate * self._norm_cdf(d2)
            result.delta = exp_div * self._norm_cdf(d1)
            result.probability_itm = self._norm_cdf(d2)
            result.intrinsic_value = max(0, spot - strike)
        else:
            result.price = strike * exp_rate * self._norm_cdf(-d2) - spot * exp_div * self._norm_cdf(-d1)
            result.delta = -exp_div * self._norm_cdf(-d1)
            result.probability_itm = self._norm_cdf(-d2)
            result.intrinsic_value = max(0, strike - spot)

        result.time_value = result.price - result.intrinsic_value

        # Calculate Greeks
        sqrt_t = math.sqrt(time)
        pdf_d1 = self._norm_pdf(d1)

        # Gamma (same for calls and puts)
        result.gamma = exp_div * pdf_d1 / (spot * vol * sqrt_t)

        # Vega (same for calls and puts) - per 1% change in vol
        result.vega = spot * exp_div * pdf_d1 * sqrt_t / 100

        # Theta (per day)
        if option_type == OptionType.CALL:
            theta = (
                -spot * exp_div * pdf_d1 * vol / (2 * sqrt_t)
                - rate * strike * exp_rate * self._norm_cdf(d2)
                + div * spot * exp_div * self._norm_cdf(d1)
            )
        else:
            theta = (
                -spot * exp_div * pdf_d1 * vol / (2 * sqrt_t)
                + rate * strike * exp_rate * self._norm_cdf(-d2)
                - div * spot * exp_div * self._norm_cdf(-d1)
            )
        result.theta = theta / 365  # Convert to per-day

        # Rho (per 1% change in rate)
        if option_type == OptionType.CALL:
            result.rho = strike * time * exp_rate * self._norm_cdf(d2) / 100
        else:
            result.rho = -strike * time * exp_rate * self._norm_cdf(-d2) / 100

        return result

    def calculate_implied_volatility(
        self,
        option_type: OptionType,
        market_price: float,
        spot: float,
        strike: float,
        time: float,
        rate: float,
        div: float = 0.0,
        max_iterations: int = 100,
        tolerance: float = 1e-5
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method.

        Args:
            option_type: CALL or PUT
            market_price: Observed market price
            spot: Current stock price
            strike: Option strike price
            time: Time to expiry in years
            rate: Risk-free interest rate
            div: Dividend yield
            max_iterations: Max Newton-Raphson iterations
            tolerance: Convergence tolerance

        Returns:
            Implied volatility or None if not found
        """
        if market_price <= 0 or time <= 0:
            return None

        # Initial guess
        vol = 0.20  # 20% starting point

        for _ in range(max_iterations):
            result = self.price_option(option_type, spot, strike, time, rate, vol, div)
            price = result.price

            # Calculate vega (sensitivity to vol change)
            d1, _ = self.calculate_d1_d2(spot, strike, time, rate, vol, div)
            vega = spot * math.exp(-div * time) * self._norm_pdf(d1) * math.sqrt(time)

            if vega < 1e-10:
                break

            # Newton-Raphson update
            diff = market_price - price
            if abs(diff) < tolerance:
                return vol

            vol = vol + diff / vega

            # Keep vol in reasonable bounds
            vol = max(0.01, min(5.0, vol))

        return vol if abs(market_price - self.price_option(option_type, spot, strike, time, rate, vol, div).price) < tolerance * 10 else None


# Convenience functions
_bs = BlackScholes()


def calculate_option_price(
    option_type: str,
    spot: float,
    strike: float,
    days_to_expiry: float,
    rate: float = 0.05,
    volatility: float = 0.20,
    dividend_yield: float = 0.0
) -> OptionPricing:
    """
    Calculate option price and Greeks.

    Args:
        option_type: "CALL" or "PUT"
        spot: Current stock price
        strike: Strike price
        days_to_expiry: Days until expiration
        rate: Risk-free rate (default 5%)
        volatility: Implied volatility (default 20%)
        dividend_yield: Dividend yield (default 0%)

    Returns:
        OptionPricing with price and Greeks
    """
    opt_type = OptionType.CALL if option_type.upper() == "CALL" else OptionType.PUT
    time = days_to_expiry / 365.0
    return _bs.price_option(opt_type, spot, strike, time, rate, volatility, dividend_yield)


def calculate_greeks(
    option_type: str,
    spot: float,
    strike: float,
    days_to_expiry: float,
    rate: float = 0.05,
    volatility: float = 0.20
) -> dict:
    """
    Calculate option Greeks.

    Returns dict with delta, gamma, vega, theta, rho.
    """
    result = calculate_option_price(option_type, spot, strike, days_to_expiry, rate, volatility)
    return {
        "delta": result.delta,
        "gamma": result.gamma,
        "vega": result.vega,
        "theta": result.theta,
        "rho": result.rho,
    }


def calculate_iv(
    option_type: str,
    market_price: float,
    spot: float,
    strike: float,
    days_to_expiry: float,
    rate: float = 0.05
) -> Optional[float]:
    """
    Calculate implied volatility from market price.

    Returns IV as decimal (0.25 = 25%) or None if not found.
    """
    opt_type = OptionType.CALL if option_type.upper() == "CALL" else OptionType.PUT
    time = days_to_expiry / 365.0
    return _bs.calculate_implied_volatility(opt_type, market_price, spot, strike, time, rate)


# Convenience wrapper functions for backward compatibility
def bs_call_price(spot: float, strike: float, time: float, rate: float, volatility: float) -> float:
    """Calculate Black-Scholes call price."""
    result = _bs.price_option(OptionType.CALL, spot, strike, time, rate, volatility)
    return result.price


def bs_put_price(spot: float, strike: float, time: float, rate: float, volatility: float) -> float:
    """Calculate Black-Scholes put price."""
    result = _bs.price_option(OptionType.PUT, spot, strike, time, rate, volatility)
    return result.price


def bs_delta(option_type: str, spot: float, strike: float, time: float, rate: float, volatility: float) -> float:
    """Calculate option delta."""
    opt_type = OptionType.CALL if option_type.upper() == "CALL" else OptionType.PUT
    result = _bs.price_option(opt_type, spot, strike, time, rate, volatility)
    return result.delta


def bs_gamma(spot: float, strike: float, time: float, rate: float, volatility: float) -> float:
    """Calculate option gamma (same for calls and puts)."""
    result = _bs.price_option(OptionType.CALL, spot, strike, time, rate, volatility)
    return result.gamma


def bs_vega(spot: float, strike: float, time: float, rate: float, volatility: float) -> float:
    """Calculate option vega (same for calls and puts)."""
    result = _bs.price_option(OptionType.CALL, spot, strike, time, rate, volatility)
    return result.vega


def bs_theta(option_type: str, spot: float, strike: float, time: float, rate: float, volatility: float) -> float:
    """Calculate option theta."""
    opt_type = OptionType.CALL if option_type.upper() == "CALL" else OptionType.PUT
    result = _bs.price_option(opt_type, spot, strike, time, rate, volatility)
    return result.theta


def implied_volatility(
    option_type: str,
    market_price: float,
    spot: float,
    strike: float,
    time: float,
    rate: float
) -> Optional[float]:
    """Calculate implied volatility from market price."""
    opt_type = OptionType.CALL if option_type.upper() == "CALL" else OptionType.PUT
    return _bs.calculate_implied_volatility(opt_type, market_price, spot, strike, time, rate)
