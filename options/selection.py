"""
Option Strike Selection Module.

Provides delta-targeted strike selection for synthetic options:
- Binary search to find strike matching target delta
- ATM (at-the-money) strike selection
- OTM/ITM strike selection by delta
- Expiration selection based on DTE range
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .black_scholes import BlackScholes, OptionType, OptionPricing


@dataclass
class StrikeSelection:
    """Result of strike selection."""
    strike: float
    delta: float
    target_delta: float
    option_type: OptionType
    price: float
    days_to_expiry: int
    moneyness: str  # 'ITM', 'ATM', 'OTM'

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "strike": self.strike,
            "delta": round(self.delta, 4),
            "target_delta": self.target_delta,
            "option_type": self.option_type.value,
            "price": round(self.price, 4),
            "days_to_expiry": self.days_to_expiry,
            "moneyness": self.moneyness,
        }


class StrikeSelector:
    """
    Selects option strikes based on delta or moneyness targets.

    For synthetic options without real strike chains, we use
    binary search to find strikes matching target deltas.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        min_delta: float = 0.05,
        max_delta: float = 0.95,
    ):
        """
        Initialize strike selector.

        Args:
            risk_free_rate: Risk-free rate (default 5%)
            min_delta: Minimum delta for OTM options (default 5 delta)
            max_delta: Maximum delta for ITM options (default 95 delta)
        """
        self.risk_free_rate = risk_free_rate
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.bs = BlackScholes()

    def _get_moneyness(
        self,
        spot: float,
        strike: float,
        option_type: OptionType,
    ) -> str:
        """Determine moneyness (ITM/ATM/OTM)."""
        pct_diff = abs(spot - strike) / spot

        if pct_diff < 0.02:  # Within 2% is ATM
            return "ATM"

        if option_type == OptionType.CALL:
            return "ITM" if strike < spot else "OTM"
        else:  # PUT
            return "ITM" if strike > spot else "OTM"

    def find_strike_by_delta(
        self,
        option_type: OptionType,
        spot: float,
        target_delta: float,
        days_to_expiry: int,
        volatility: float,
        tolerance: float = 0.005,
        max_iterations: int = 50,
    ) -> StrikeSelection:
        """
        Find strike price that gives target delta using binary search.

        Args:
            option_type: CALL or PUT
            spot: Current stock price
            target_delta: Desired delta (absolute value, e.g., 0.30 for 30-delta)
            days_to_expiry: Days until expiration
            volatility: Implied/realized volatility
            tolerance: Acceptable delta difference (default 0.5%)
            max_iterations: Max binary search iterations

        Returns:
            StrikeSelection with strike matching target delta
        """
        time = days_to_expiry / 365.0

        # Adjust target delta sign for puts (puts have negative delta)
        if option_type == OptionType.PUT:
            signed_target = -abs(target_delta)
        else:
            signed_target = abs(target_delta)

        # Binary search bounds
        # For calls: higher strike = lower delta
        # For puts: higher strike = higher (less negative) delta
        if option_type == OptionType.CALL:
            # 95-delta call is deep ITM (low strike), 5-delta is far OTM (high strike)
            low_strike = spot * 0.5
            high_strike = spot * 1.5
        else:
            # 95-delta put (abs) is deep ITM (high strike), 5-delta is far OTM (low strike)
            low_strike = spot * 0.5
            high_strike = spot * 1.5

        best_strike = spot
        best_delta = 0.5 if option_type == OptionType.CALL else -0.5
        best_price = 0.0

        for _ in range(max_iterations):
            mid_strike = (low_strike + high_strike) / 2

            result = self.bs.price_option(
                option_type, spot, mid_strike, time,
                self.risk_free_rate, volatility
            )

            current_delta = result.delta
            delta_diff = current_delta - signed_target

            # Update best if closer
            if abs(current_delta - signed_target) < abs(best_delta - signed_target):
                best_strike = mid_strike
                best_delta = current_delta
                best_price = result.price

            # Check convergence
            if abs(delta_diff) < tolerance:
                break

            # Adjust bounds
            if option_type == OptionType.CALL:
                # For calls: want higher delta -> lower strike
                if current_delta < signed_target:
                    high_strike = mid_strike
                else:
                    low_strike = mid_strike
            else:
                # For puts: delta goes from ~0 (low strike, OTM) to ~-1 (high strike, ITM)
                # If current delta is too negative (ITM), need lower strike
                if current_delta < signed_target:
                    high_strike = mid_strike
                else:
                    low_strike = mid_strike

        moneyness = self._get_moneyness(spot, best_strike, option_type)

        return StrikeSelection(
            strike=round(best_strike, 2),
            delta=best_delta,
            target_delta=target_delta,
            option_type=option_type,
            price=best_price,
            days_to_expiry=days_to_expiry,
            moneyness=moneyness,
        )

    def find_atm_strike(
        self,
        option_type: OptionType,
        spot: float,
        days_to_expiry: int,
        volatility: float,
    ) -> StrikeSelection:
        """
        Find ATM strike (delta ~0.50 for calls, ~-0.50 for puts).

        Args:
            option_type: CALL or PUT
            spot: Current stock price
            days_to_expiry: Days until expiration
            volatility: Implied/realized volatility

        Returns:
            StrikeSelection for ATM option
        """
        return self.find_strike_by_delta(
            option_type, spot, 0.50, days_to_expiry, volatility
        )

    def find_otm_strike(
        self,
        option_type: OptionType,
        spot: float,
        days_to_expiry: int,
        volatility: float,
        delta: float = 0.30,
    ) -> StrikeSelection:
        """
        Find OTM strike with specified delta.

        Common OTM deltas:
        - 0.30 (30-delta): Moderate OTM
        - 0.20 (20-delta): Further OTM
        - 0.10 (10-delta): Deep OTM

        Args:
            option_type: CALL or PUT
            spot: Current stock price
            days_to_expiry: Days until expiration
            volatility: Implied/realized volatility
            delta: Target delta (absolute value, default 0.30)

        Returns:
            StrikeSelection for OTM option
        """
        return self.find_strike_by_delta(
            option_type, spot, delta, days_to_expiry, volatility
        )

    def find_itm_strike(
        self,
        option_type: OptionType,
        spot: float,
        days_to_expiry: int,
        volatility: float,
        delta: float = 0.70,
    ) -> StrikeSelection:
        """
        Find ITM strike with specified delta.

        Common ITM deltas:
        - 0.70 (70-delta): Moderate ITM
        - 0.80 (80-delta): Further ITM
        - 0.90 (90-delta): Deep ITM

        Args:
            option_type: CALL or PUT
            spot: Current stock price
            days_to_expiry: Days until expiration
            volatility: Implied/realized volatility
            delta: Target delta (absolute value, default 0.70)

        Returns:
            StrikeSelection for ITM option
        """
        return self.find_strike_by_delta(
            option_type, spot, delta, days_to_expiry, volatility
        )

    def build_strike_ladder(
        self,
        option_type: OptionType,
        spot: float,
        days_to_expiry: int,
        volatility: float,
        deltas: List[float] = None,
    ) -> List[StrikeSelection]:
        """
        Build a ladder of strikes at different deltas.

        Useful for visualizing the strike chain or comparing premium levels.

        Args:
            option_type: CALL or PUT
            spot: Current stock price
            days_to_expiry: Days until expiration
            volatility: Implied/realized volatility
            deltas: List of target deltas (default: 10, 20, 30, 40, 50, 60, 70, 80, 90)

        Returns:
            List of StrikeSelection objects
        """
        if deltas is None:
            deltas = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

        ladder = []
        for delta in deltas:
            selection = self.find_strike_by_delta(
                option_type, spot, delta, days_to_expiry, volatility
            )
            ladder.append(selection)

        return ladder

    def select_expiration(
        self,
        target_dte: int,
        available_expirations: List[int] = None,
        min_dte: int = 7,
        max_dte: int = 60,
    ) -> int:
        """
        Select appropriate expiration DTE.

        For synthetic options, we generate our own expirations.
        For swing trading, common DTEs are 14-45 days.

        Args:
            target_dte: Desired days to expiry
            available_expirations: List of available DTEs (if using real chain)
            min_dte: Minimum acceptable DTE (default 7)
            max_dte: Maximum acceptable DTE (default 60)

        Returns:
            Selected DTE (clamped to min/max)
        """
        if available_expirations:
            # Find closest available expiration
            valid = [d for d in available_expirations if min_dte <= d <= max_dte]
            if not valid:
                valid = available_expirations

            return min(valid, key=lambda x: abs(x - target_dte))

        # Synthetic: clamp to range
        return max(min_dte, min(max_dte, target_dte))


# Convenience functions
_selector = StrikeSelector()


def select_call_strike(
    spot: float,
    target_delta: float,
    days_to_expiry: int,
    volatility: float,
) -> StrikeSelection:
    """
    Select call strike by delta target.

    Args:
        spot: Current stock price
        target_delta: Target delta (e.g., 0.30 for 30-delta)
        days_to_expiry: Days until expiration
        volatility: Realized/implied volatility

    Returns:
        StrikeSelection with matching strike
    """
    return _selector.find_strike_by_delta(
        OptionType.CALL, spot, target_delta, days_to_expiry, volatility
    )


def select_put_strike(
    spot: float,
    target_delta: float,
    days_to_expiry: int,
    volatility: float,
) -> StrikeSelection:
    """
    Select put strike by delta target.

    Args:
        spot: Current stock price
        target_delta: Target delta (e.g., 0.30 for 30-delta)
        days_to_expiry: Days until expiration
        volatility: Realized/implied volatility

    Returns:
        StrikeSelection with matching strike
    """
    return _selector.find_strike_by_delta(
        OptionType.PUT, spot, target_delta, days_to_expiry, volatility
    )


def select_atm(
    option_type: str,
    spot: float,
    days_to_expiry: int,
    volatility: float,
) -> StrikeSelection:
    """
    Select ATM strike (50-delta).

    Args:
        option_type: "CALL" or "PUT"
        spot: Current stock price
        days_to_expiry: Days until expiration
        volatility: Realized/implied volatility

    Returns:
        StrikeSelection for ATM option
    """
    opt_type = OptionType.CALL if option_type.upper() == "CALL" else OptionType.PUT
    return _selector.find_atm_strike(opt_type, spot, days_to_expiry, volatility)
