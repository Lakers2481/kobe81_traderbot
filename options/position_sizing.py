"""
Options Position Sizing Module.

Enforces 2% equity risk per trade for single-leg options:
- Long options: Max loss = premium paid (limited)
- Short options: Max loss = strike (put) or unlimited (call) - require collateral

Position sizing formulas:
- Long call/put: contracts = (equity * risk_pct) / (premium * 100)
- Short put (cash-secured): contracts = min(equity / (strike * 100), risk_budget / (premium * 100))
- Short call (covered): requires underlying shares

Constraints:
- Single-leg only (no spreads)
- 2% max risk per trade
- Collateral checks for shorts
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .black_scholes import BlackScholes, OptionType, OptionPricing


class PositionDirection(Enum):
    """Option position direction."""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    contracts: int
    direction: PositionDirection
    option_type: OptionType
    premium_per_contract: float
    total_premium: float
    max_risk: float
    max_profit: float
    risk_pct_of_equity: float
    collateral_required: float
    margin_required: float
    buying_power_required: float
    is_valid: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "contracts": self.contracts,
            "direction": self.direction.value,
            "option_type": self.option_type.value,
            "premium_per_contract": round(self.premium_per_contract, 2),
            "total_premium": round(self.total_premium, 2),
            "max_risk": round(self.max_risk, 2),
            "max_profit": round(self.max_profit, 2),
            "risk_pct_of_equity": round(self.risk_pct_of_equity, 4),
            "collateral_required": round(self.collateral_required, 2),
            "margin_required": round(self.margin_required, 2),
            "buying_power_required": round(self.buying_power_required, 2),
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason,
        }


class OptionsPositionSizer:
    """
    Position sizing for single-leg options with 2% risk enforcement.

    Swing trading constraints:
    - Max 2% equity at risk per trade
    - Single-leg options only (calls or puts)
    - Supports long and short positions
    - Collateral requirements for short positions
    """

    # Standard option contract size
    SHARES_PER_CONTRACT = 100

    def __init__(
        self,
        risk_pct: float = 0.02,  # 2% default
        max_contracts: int = 100,  # Max contracts per position
        min_contracts: int = 1,  # Min contracts
        margin_requirement: float = 0.20,  # 20% margin for naked shorts
    ):
        """
        Initialize position sizer.

        Args:
            risk_pct: Maximum risk per trade as decimal (default 2%)
            max_contracts: Maximum contracts per position
            min_contracts: Minimum contracts (default 1)
            margin_requirement: Margin requirement for short options (default 20%)
        """
        self.risk_pct = risk_pct
        self.max_contracts = max_contracts
        self.min_contracts = min_contracts
        self.margin_requirement = margin_requirement
        self.bs = BlackScholes()

    def size_long_option(
        self,
        equity: float,
        option_type: OptionType,
        premium: float,
        strike: float,
        spot: float,
    ) -> PositionSizeResult:
        """
        Size a long option position.

        Long options have limited risk (premium paid).
        Max risk = premium * contracts * 100

        Args:
            equity: Account equity
            option_type: CALL or PUT
            premium: Option premium per share
            strike: Strike price
            spot: Current stock price

        Returns:
            PositionSizeResult with contracts and risk metrics
        """
        if premium <= 0:
            return PositionSizeResult(
                contracts=0,
                direction=PositionDirection.LONG,
                option_type=option_type,
                premium_per_contract=0,
                total_premium=0,
                max_risk=0,
                max_profit=0,
                risk_pct_of_equity=0,
                collateral_required=0,
                margin_required=0,
                buying_power_required=0,
                is_valid=False,
                rejection_reason="Premium must be positive",
            )

        # Risk budget
        risk_budget = equity * self.risk_pct

        # Premium per contract (100 shares)
        premium_per_contract = premium * self.SHARES_PER_CONTRACT

        # Calculate max contracts based on risk budget
        max_by_risk = math.floor(risk_budget / premium_per_contract)

        # Apply position limits
        contracts = min(max_by_risk, self.max_contracts)
        contracts = max(contracts, 0)

        if contracts < self.min_contracts:
            return PositionSizeResult(
                contracts=0,
                direction=PositionDirection.LONG,
                option_type=option_type,
                premium_per_contract=premium_per_contract,
                total_premium=0,
                max_risk=risk_budget,
                max_profit=0,
                risk_pct_of_equity=self.risk_pct,
                collateral_required=0,
                margin_required=0,
                buying_power_required=premium_per_contract,
                is_valid=False,
                rejection_reason=f"Risk budget ${risk_budget:.2f} insufficient for min {self.min_contracts} contract(s)",
            )

        total_premium = contracts * premium_per_contract
        max_risk = total_premium  # Max loss is premium paid

        # Max profit
        if option_type == OptionType.CALL:
            # Unlimited for calls (use 10x as proxy)
            max_profit = contracts * spot * self.SHARES_PER_CONTRACT
        else:
            # Puts max out at strike - premium
            max_profit = contracts * (strike - premium) * self.SHARES_PER_CONTRACT

        actual_risk_pct = max_risk / equity

        return PositionSizeResult(
            contracts=contracts,
            direction=PositionDirection.LONG,
            option_type=option_type,
            premium_per_contract=premium_per_contract,
            total_premium=total_premium,
            max_risk=max_risk,
            max_profit=max_profit,
            risk_pct_of_equity=actual_risk_pct,
            collateral_required=0,
            margin_required=0,
            buying_power_required=total_premium,
            is_valid=True,
        )

    def size_short_put(
        self,
        equity: float,
        premium: float,
        strike: float,
        spot: float,
        cash_secured: bool = True,
    ) -> PositionSizeResult:
        """
        Size a short put position (cash-secured or margin).

        Short puts have risk = (strike - premium) per share if exercised.
        Cash-secured requires full collateral (strike * 100).
        Margin requires partial collateral.

        Args:
            equity: Account equity
            premium: Option premium received per share
            strike: Strike price
            spot: Current stock price
            cash_secured: If True, require full cash collateral

        Returns:
            PositionSizeResult with contracts and risk metrics
        """
        if premium <= 0 or strike <= 0:
            return PositionSizeResult(
                contracts=0,
                direction=PositionDirection.SHORT,
                option_type=OptionType.PUT,
                premium_per_contract=0,
                total_premium=0,
                max_risk=0,
                max_profit=0,
                risk_pct_of_equity=0,
                collateral_required=0,
                margin_required=0,
                buying_power_required=0,
                is_valid=False,
                rejection_reason="Premium and strike must be positive",
            )

        premium_per_contract = premium * self.SHARES_PER_CONTRACT
        risk_budget = equity * self.risk_pct

        # Max loss per contract (if stock goes to zero)
        max_loss_per_contract = (strike - premium) * self.SHARES_PER_CONTRACT

        # More realistic max loss: if stock drops 50%
        realistic_loss_per_contract = max(0, (strike - spot * 0.5 - premium)) * self.SHARES_PER_CONTRACT

        # Use realistic loss for sizing (conservative)
        sizing_loss = max(realistic_loss_per_contract, max_loss_per_contract * 0.5)

        # Calculate contracts by risk
        if sizing_loss > 0:
            max_by_risk = math.floor(risk_budget / sizing_loss)
        else:
            max_by_risk = self.max_contracts

        # Collateral requirements
        if cash_secured:
            collateral_per_contract = strike * self.SHARES_PER_CONTRACT
        else:
            # Margin: greater of 20% of stock or (premium + 10% OTM amount)
            otm_amount = max(0, spot - strike)
            margin_per_contract = max(
                spot * self.margin_requirement * self.SHARES_PER_CONTRACT,
                (premium + otm_amount * 0.1) * self.SHARES_PER_CONTRACT
            )
            collateral_per_contract = margin_per_contract

        # Calculate contracts by collateral
        max_by_collateral = math.floor(equity / collateral_per_contract)

        # Take minimum
        contracts = min(max_by_risk, max_by_collateral, self.max_contracts)
        contracts = max(contracts, 0)

        if contracts < self.min_contracts:
            return PositionSizeResult(
                contracts=0,
                direction=PositionDirection.SHORT,
                option_type=OptionType.PUT,
                premium_per_contract=premium_per_contract,
                total_premium=0,
                max_risk=max_loss_per_contract,
                max_profit=premium_per_contract,
                risk_pct_of_equity=self.risk_pct,
                collateral_required=collateral_per_contract,
                margin_required=collateral_per_contract if not cash_secured else 0,
                buying_power_required=collateral_per_contract,
                is_valid=False,
                rejection_reason=f"Insufficient equity for {self.min_contracts} contract(s). Need ${collateral_per_contract:.2f}",
            )

        total_premium = contracts * premium_per_contract
        max_risk = contracts * max_loss_per_contract
        max_profit = total_premium
        total_collateral = contracts * collateral_per_contract

        return PositionSizeResult(
            contracts=contracts,
            direction=PositionDirection.SHORT,
            option_type=OptionType.PUT,
            premium_per_contract=premium_per_contract,
            total_premium=total_premium,
            max_risk=max_risk,
            max_profit=max_profit,
            risk_pct_of_equity=max_risk / equity,
            collateral_required=total_collateral,
            margin_required=total_collateral if not cash_secured else 0,
            buying_power_required=total_collateral - total_premium,
            is_valid=True,
        )

    def size_short_call(
        self,
        equity: float,
        premium: float,
        strike: float,
        spot: float,
        shares_owned: int = 0,
    ) -> PositionSizeResult:
        """
        Size a short call position (covered or naked).

        Covered calls: Risk is opportunity cost (stock above strike)
        Naked calls: Unlimited risk (require significant margin)

        Args:
            equity: Account equity
            premium: Option premium received per share
            strike: Strike price
            spot: Current stock price
            shares_owned: Shares of underlying owned (for covered calls)

        Returns:
            PositionSizeResult with contracts and risk metrics
        """
        if premium <= 0 or strike <= 0:
            return PositionSizeResult(
                contracts=0,
                direction=PositionDirection.SHORT,
                option_type=OptionType.CALL,
                premium_per_contract=0,
                total_premium=0,
                max_risk=0,
                max_profit=0,
                risk_pct_of_equity=0,
                collateral_required=0,
                margin_required=0,
                buying_power_required=0,
                is_valid=False,
                rejection_reason="Premium and strike must be positive",
            )

        premium_per_contract = premium * self.SHARES_PER_CONTRACT
        risk_budget = equity * self.risk_pct

        # Max covered contracts
        max_covered = shares_owned // self.SHARES_PER_CONTRACT

        if max_covered > 0:
            # Covered call: risk is limited to (stock going to zero - premium)
            # But opportunity cost if stock rises above strike
            # For sizing, use opportunity cost if stock rises 50%
            opportunity_cost = max(0, spot * 1.5 - strike - premium) * self.SHARES_PER_CONTRACT

            if opportunity_cost > 0:
                max_by_risk = math.floor(risk_budget / opportunity_cost)
            else:
                max_by_risk = max_covered

            contracts = min(max_by_risk, max_covered, self.max_contracts)

            if contracts < self.min_contracts:
                return PositionSizeResult(
                    contracts=0,
                    direction=PositionDirection.SHORT,
                    option_type=OptionType.CALL,
                    premium_per_contract=premium_per_contract,
                    total_premium=0,
                    max_risk=0,
                    max_profit=premium_per_contract,
                    risk_pct_of_equity=0,
                    collateral_required=0,
                    margin_required=0,
                    buying_power_required=0,
                    is_valid=False,
                    rejection_reason=f"Not enough shares for covered call. Have {shares_owned}, need {self.SHARES_PER_CONTRACT}",
                )

            total_premium = contracts * premium_per_contract

            return PositionSizeResult(
                contracts=contracts,
                direction=PositionDirection.SHORT,
                option_type=OptionType.CALL,
                premium_per_contract=premium_per_contract,
                total_premium=total_premium,
                max_risk=0,  # Covered calls have no additional cash risk
                max_profit=total_premium + contracts * max(0, strike - spot) * self.SHARES_PER_CONTRACT,
                risk_pct_of_equity=0,
                collateral_required=0,
                margin_required=0,
                buying_power_required=0,
                is_valid=True,
            )

        else:
            # Naked call - VERY RISKY (reject by default for swing trading)
            return PositionSizeResult(
                contracts=0,
                direction=PositionDirection.SHORT,
                option_type=OptionType.CALL,
                premium_per_contract=premium_per_contract,
                total_premium=0,
                max_risk=float('inf'),
                max_profit=premium_per_contract,
                risk_pct_of_equity=0,
                collateral_required=0,
                margin_required=0,
                buying_power_required=0,
                is_valid=False,
                rejection_reason="Naked short calls not allowed (unlimited risk). Need shares for covered call.",
            )

    def size_option(
        self,
        equity: float,
        direction: PositionDirection,
        option_type: OptionType,
        premium: float,
        strike: float,
        spot: float,
        shares_owned: int = 0,
        cash_secured: bool = True,
    ) -> PositionSizeResult:
        """
        Universal position sizing entry point.

        Args:
            equity: Account equity
            direction: LONG or SHORT
            option_type: CALL or PUT
            premium: Option premium per share
            strike: Strike price
            spot: Current stock price
            shares_owned: Shares owned (for covered calls)
            cash_secured: Cash-secured puts (vs margin)

        Returns:
            PositionSizeResult with sizing details
        """
        if direction == PositionDirection.LONG:
            return self.size_long_option(equity, option_type, premium, strike, spot)

        elif direction == PositionDirection.SHORT:
            if option_type == OptionType.PUT:
                return self.size_short_put(equity, premium, strike, spot, cash_secured)
            else:
                return self.size_short_call(equity, premium, strike, spot, shares_owned)

        else:
            return PositionSizeResult(
                contracts=0,
                direction=direction,
                option_type=option_type,
                premium_per_contract=0,
                total_premium=0,
                max_risk=0,
                max_profit=0,
                risk_pct_of_equity=0,
                collateral_required=0,
                margin_required=0,
                buying_power_required=0,
                is_valid=False,
                rejection_reason=f"Unknown direction: {direction}",
            )


# Convenience functions
_sizer = OptionsPositionSizer(risk_pct=0.02)


def size_long_call(
    equity: float,
    premium: float,
    strike: float,
    spot: float,
) -> PositionSizeResult:
    """Size a long call with 2% risk limit."""
    return _sizer.size_long_option(equity, OptionType.CALL, premium, strike, spot)


def size_long_put(
    equity: float,
    premium: float,
    strike: float,
    spot: float,
) -> PositionSizeResult:
    """Size a long put with 2% risk limit."""
    return _sizer.size_long_option(equity, OptionType.PUT, premium, strike, spot)


def size_cash_secured_put(
    equity: float,
    premium: float,
    strike: float,
    spot: float,
) -> PositionSizeResult:
    """Size a cash-secured short put with 2% risk limit."""
    return _sizer.size_short_put(equity, premium, strike, spot, cash_secured=True)


def size_covered_call(
    equity: float,
    premium: float,
    strike: float,
    spot: float,
    shares_owned: int,
) -> PositionSizeResult:
    """Size a covered call with 2% risk limit."""
    return _sizer.size_short_call(equity, premium, strike, spot, shares_owned)


def calculate_max_contracts(
    equity: float,
    premium: float,
    risk_pct: float = 0.02,
) -> int:
    """
    Quick calculation of max contracts for a long option.

    Args:
        equity: Account equity
        premium: Option premium per share
        risk_pct: Risk per trade (default 2%)

    Returns:
        Maximum contracts allowed
    """
    if premium <= 0:
        return 0

    risk_budget = equity * risk_pct
    premium_per_contract = premium * 100
    return int(risk_budget / premium_per_contract)
