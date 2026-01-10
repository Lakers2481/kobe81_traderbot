"""
Options Spread Strategies for Live Trading.

Implements multi-leg options strategies:
- Vertical Spreads (Bull/Bear Call/Put)
- Credit Spreads
- Debit Spreads
- Iron Condors
- Strangles/Straddles

Uses chain_fetcher for contract selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from options.chain_fetcher import (
    ChainFetcher,
    OptionContract,
    OptionType,
)

logger = logging.getLogger(__name__)


class SpreadType(Enum):
    """Types of options spreads."""
    # Vertical Spreads
    BULL_CALL_SPREAD = auto()      # Buy lower strike call, sell higher strike call
    BEAR_CALL_SPREAD = auto()      # Sell lower strike call, buy higher strike call
    BULL_PUT_SPREAD = auto()       # Sell higher strike put, buy lower strike put
    BEAR_PUT_SPREAD = auto()       # Buy higher strike put, sell lower strike put

    # Iron Condor
    IRON_CONDOR = auto()           # Bull put spread + Bear call spread

    # Straddle/Strangle
    LONG_STRADDLE = auto()         # Buy ATM call + ATM put
    SHORT_STRADDLE = auto()        # Sell ATM call + ATM put
    LONG_STRANGLE = auto()         # Buy OTM call + OTM put
    SHORT_STRANGLE = auto()        # Sell OTM call + OTM put

    # Calendar Spreads
    CALENDAR_CALL = auto()         # Sell near-term, buy far-term (same strike)
    CALENDAR_PUT = auto()

    # Butterfly
    LONG_BUTTERFLY_CALL = auto()   # Buy 1 lower, sell 2 middle, buy 1 upper
    LONG_BUTTERFLY_PUT = auto()


@dataclass
class SpreadLeg:
    """Single leg of an options spread."""
    contract: OptionContract
    quantity: int              # Positive = long, negative = short
    side: str                  # "buy" or "sell"

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def cost(self) -> float:
        """Cost to enter this leg (negative for credit)."""
        if self.contract.mid is None:
            return 0.0
        # Long = pay mid, Short = receive mid
        return self.contract.mid * abs(self.quantity) * 100 * (1 if self.is_long else -1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_symbol": self.contract.contract_symbol,
            "option_type": self.contract.option_type.value,
            "strike": self.contract.strike,
            "expiration": self.contract.expiration.isoformat(),
            "quantity": self.quantity,
            "side": self.side,
            "cost": round(self.cost, 2),
        }


@dataclass
class OptionsSpread:
    """Complete options spread with multiple legs."""
    spread_type: SpreadType
    symbol: str
    legs: List[SpreadLeg] = field(default_factory=list)
    expiration: Optional[date] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_cost(self) -> float:
        """Total cost to enter spread (negative = credit)."""
        return sum(leg.cost for leg in self.legs)

    @property
    def is_credit(self) -> bool:
        """True if spread receives net premium."""
        return self.total_cost < 0

    @property
    def is_debit(self) -> bool:
        """True if spread pays net premium."""
        return self.total_cost > 0

    @property
    def max_profit(self) -> Optional[float]:
        """Maximum profit potential."""
        if self.spread_type == SpreadType.BULL_CALL_SPREAD:
            # Max profit = (high strike - low strike) * 100 - net debit
            strikes = sorted([leg.contract.strike for leg in self.legs])
            if len(strikes) >= 2:
                spread_width = (strikes[1] - strikes[0]) * 100
                return spread_width - self.total_cost
        elif self.spread_type == SpreadType.BEAR_PUT_SPREAD:
            strikes = sorted([leg.contract.strike for leg in self.legs])
            if len(strikes) >= 2:
                spread_width = (strikes[1] - strikes[0]) * 100
                return spread_width - self.total_cost
        elif self.spread_type in (SpreadType.BULL_PUT_SPREAD, SpreadType.BEAR_CALL_SPREAD):
            # Credit spreads: max profit = premium received
            return abs(self.total_cost) if self.is_credit else 0
        elif self.spread_type == SpreadType.IRON_CONDOR:
            # Max profit = net credit received
            return abs(self.total_cost) if self.is_credit else 0
        return None

    @property
    def max_loss(self) -> Optional[float]:
        """Maximum loss potential."""
        if self.spread_type in (SpreadType.BULL_CALL_SPREAD, SpreadType.BEAR_PUT_SPREAD):
            # Debit spreads: max loss = net debit paid
            return self.total_cost if self.is_debit else 0
        elif self.spread_type in (SpreadType.BULL_PUT_SPREAD, SpreadType.BEAR_CALL_SPREAD):
            # Credit spreads: max loss = spread width - premium
            strikes = sorted([leg.contract.strike for leg in self.legs])
            if len(strikes) >= 2:
                spread_width = (strikes[1] - strikes[0]) * 100
                return spread_width - abs(self.total_cost)
        elif self.spread_type == SpreadType.IRON_CONDOR:
            # Max loss = wider wing width - net credit
            if len(self.legs) >= 4:
                put_strikes = sorted([l.contract.strike for l in self.legs
                                     if l.contract.option_type == OptionType.PUT])
                call_strikes = sorted([l.contract.strike for l in self.legs
                                      if l.contract.option_type == OptionType.CALL])
                if len(put_strikes) >= 2 and len(call_strikes) >= 2:
                    put_width = (put_strikes[1] - put_strikes[0]) * 100
                    call_width = (call_strikes[1] - call_strikes[0]) * 100
                    max_width = max(put_width, call_width)
                    return max_width - abs(self.total_cost)
        return None

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Risk/reward ratio (lower is better)."""
        if self.max_profit and self.max_loss:
            return self.max_loss / self.max_profit if self.max_profit > 0 else None
        return None

    @property
    def breakeven_prices(self) -> List[float]:
        """Breakeven price(s) for the spread."""
        breakevens = []
        strikes = sorted(set(leg.contract.strike for leg in self.legs))

        if self.spread_type == SpreadType.BULL_CALL_SPREAD:
            # BE = lower strike + net debit per share
            if strikes:
                be = strikes[0] + self.total_cost / 100
                breakevens.append(be)
        elif self.spread_type == SpreadType.BEAR_PUT_SPREAD:
            # BE = higher strike - net debit per share
            if strikes:
                be = strikes[-1] - self.total_cost / 100
                breakevens.append(be)
        elif self.spread_type == SpreadType.BULL_PUT_SPREAD:
            # BE = higher strike - net credit per share
            if len(strikes) >= 2:
                be = strikes[1] - abs(self.total_cost) / 100
                breakevens.append(be)
        elif self.spread_type == SpreadType.BEAR_CALL_SPREAD:
            # BE = lower strike + net credit per share
            if strikes:
                be = strikes[0] + abs(self.total_cost) / 100
                breakevens.append(be)
        elif self.spread_type == SpreadType.IRON_CONDOR:
            # Two breakevens: put side and call side
            put_strikes = sorted([l.contract.strike for l in self.legs
                                 if l.contract.option_type == OptionType.PUT])
            call_strikes = sorted([l.contract.strike for l in self.legs
                                  if l.contract.option_type == OptionType.CALL])
            if len(put_strikes) >= 2 and len(call_strikes) >= 2:
                credit_per_share = abs(self.total_cost) / 100
                # Lower BE = higher put strike - credit
                breakevens.append(put_strikes[1] - credit_per_share)
                # Upper BE = lower call strike + credit
                breakevens.append(call_strikes[0] + credit_per_share)

        return breakevens

    @property
    def net_delta(self) -> float:
        """Net delta of the spread."""
        total_delta = 0.0
        for leg in self.legs:
            if leg.contract.delta is not None:
                total_delta += leg.contract.delta * leg.quantity
        return total_delta

    @property
    def net_theta(self) -> float:
        """Net theta of the spread (daily decay)."""
        total_theta = 0.0
        for leg in self.legs:
            if leg.contract.theta is not None:
                total_theta += leg.contract.theta * leg.quantity
        return total_theta

    @property
    def net_vega(self) -> float:
        """Net vega of the spread."""
        total_vega = 0.0
        for leg in self.legs:
            if leg.contract.vega is not None:
                total_vega += leg.contract.vega * leg.quantity
        return total_vega

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spread_type": self.spread_type.name,
            "symbol": self.symbol,
            "expiration": self.expiration.isoformat() if self.expiration else None,
            "legs": [leg.to_dict() for leg in self.legs],
            "total_cost": round(self.total_cost, 2),
            "is_credit": self.is_credit,
            "max_profit": round(self.max_profit, 2) if self.max_profit else None,
            "max_loss": round(self.max_loss, 2) if self.max_loss else None,
            "risk_reward_ratio": round(self.risk_reward_ratio, 2) if self.risk_reward_ratio else None,
            "breakeven_prices": [round(be, 2) for be in self.breakeven_prices],
            "net_delta": round(self.net_delta, 4),
            "net_theta": round(self.net_theta, 4),
            "net_vega": round(self.net_vega, 4),
        }


class SpreadBuilder:
    """
    Builds options spreads from chain data.

    Uses ChainFetcher to get live chain data and constructs
    appropriate spreads based on market conditions.
    """

    def __init__(self, chain_fetcher: Optional[ChainFetcher] = None):
        """
        Initialize spread builder.

        Args:
            chain_fetcher: ChainFetcher instance (creates default if None)
        """
        self.chain_fetcher = chain_fetcher or ChainFetcher()

    def build_bull_call_spread(
        self,
        symbol: str,
        target_dte: int = 30,
        long_delta: float = 0.50,
        short_delta: float = 0.30,
        quantity: int = 1,
    ) -> Optional[OptionsSpread]:
        """
        Build a bull call spread (debit spread).

        Buy lower strike call, sell higher strike call.

        Args:
            symbol: Underlying symbol
            target_dte: Target days to expiration
            long_delta: Delta for long leg (typically ATM ~0.50)
            short_delta: Delta for short leg (typically OTM ~0.30)
            quantity: Number of spreads

        Returns:
            OptionsSpread or None if construction fails
        """
        chain = self.chain_fetcher.fetch_chain(symbol)
        if not chain:
            logger.error(f"Could not fetch chain for {symbol}")
            return None

        expiration = chain.get_expiration(target_dte)
        if not expiration:
            logger.error(f"No suitable expiration for {symbol}")
            return None

        # Get long call (higher delta = lower strike)
        long_call = chain.get_contract_by_delta(expiration, OptionType.CALL, long_delta)
        if not long_call:
            logger.error(f"Could not find long call for {symbol}")
            return None

        # Get short call (lower delta = higher strike)
        short_call = chain.get_contract_by_delta(expiration, OptionType.CALL, short_delta)
        if not short_call:
            logger.error(f"Could not find short call for {symbol}")
            return None

        # Validate strike order
        if long_call.strike >= short_call.strike:
            logger.error("Long call strike must be lower than short call strike")
            return None

        spread = OptionsSpread(
            spread_type=SpreadType.BULL_CALL_SPREAD,
            symbol=symbol,
            expiration=expiration,
            legs=[
                SpreadLeg(contract=long_call, quantity=quantity, side="buy"),
                SpreadLeg(contract=short_call, quantity=-quantity, side="sell"),
            ],
        )

        logger.info(f"Built bull call spread for {symbol}: {long_call.strike}/{short_call.strike}")
        return spread

    def build_bear_put_spread(
        self,
        symbol: str,
        target_dte: int = 30,
        long_delta: float = 0.50,
        short_delta: float = 0.30,
        quantity: int = 1,
    ) -> Optional[OptionsSpread]:
        """
        Build a bear put spread (debit spread).

        Buy higher strike put, sell lower strike put.

        Args:
            symbol: Underlying symbol
            target_dte: Target days to expiration
            long_delta: Delta for long leg (typically ATM ~0.50)
            short_delta: Delta for short leg (typically OTM ~0.30)
            quantity: Number of spreads

        Returns:
            OptionsSpread or None if construction fails
        """
        chain = self.chain_fetcher.fetch_chain(symbol)
        if not chain:
            return None

        expiration = chain.get_expiration(target_dte)
        if not expiration:
            return None

        # Get long put (higher delta = higher strike, remembering put deltas are negative)
        long_put = chain.get_contract_by_delta(expiration, OptionType.PUT, long_delta)
        if not long_put:
            return None

        # Get short put (lower delta = lower strike)
        short_put = chain.get_contract_by_delta(expiration, OptionType.PUT, short_delta)
        if not short_put:
            return None

        # Validate strike order
        if long_put.strike <= short_put.strike:
            logger.error("Long put strike must be higher than short put strike")
            return None

        spread = OptionsSpread(
            spread_type=SpreadType.BEAR_PUT_SPREAD,
            symbol=symbol,
            expiration=expiration,
            legs=[
                SpreadLeg(contract=long_put, quantity=quantity, side="buy"),
                SpreadLeg(contract=short_put, quantity=-quantity, side="sell"),
            ],
        )

        logger.info(f"Built bear put spread for {symbol}: {short_put.strike}/{long_put.strike}")
        return spread

    def build_bull_put_spread(
        self,
        symbol: str,
        target_dte: int = 30,
        short_delta: float = 0.30,
        spread_width: float = 5.0,
        quantity: int = 1,
    ) -> Optional[OptionsSpread]:
        """
        Build a bull put spread (credit spread).

        Sell higher strike put, buy lower strike put.
        Profits if stock stays above short put strike.

        Args:
            symbol: Underlying symbol
            target_dte: Target days to expiration
            short_delta: Delta for short leg (typically ~0.30 OTM)
            spread_width: Dollar width between strikes
            quantity: Number of spreads

        Returns:
            OptionsSpread or None if construction fails
        """
        chain = self.chain_fetcher.fetch_chain(symbol)
        if not chain:
            return None

        expiration = chain.get_expiration(target_dte)
        if not expiration:
            return None

        # Get short put (the one we sell)
        short_put = chain.get_contract_by_delta(expiration, OptionType.PUT, short_delta)
        if not short_put:
            return None

        # Find long put at lower strike
        target_long_strike = short_put.strike - spread_width
        contracts = chain.get_contracts_by_expiration(expiration, OptionType.PUT)
        long_put = min(
            [c for c in contracts if c.strike <= target_long_strike],
            key=lambda c: abs(c.strike - target_long_strike),
            default=None,
        )

        if not long_put:
            return None

        spread = OptionsSpread(
            spread_type=SpreadType.BULL_PUT_SPREAD,
            symbol=symbol,
            expiration=expiration,
            legs=[
                SpreadLeg(contract=short_put, quantity=-quantity, side="sell"),
                SpreadLeg(contract=long_put, quantity=quantity, side="buy"),
            ],
        )

        logger.info(f"Built bull put spread for {symbol}: {long_put.strike}/{short_put.strike}")
        return spread

    def build_bear_call_spread(
        self,
        symbol: str,
        target_dte: int = 30,
        short_delta: float = 0.30,
        spread_width: float = 5.0,
        quantity: int = 1,
    ) -> Optional[OptionsSpread]:
        """
        Build a bear call spread (credit spread).

        Sell lower strike call, buy higher strike call.
        Profits if stock stays below short call strike.

        Args:
            symbol: Underlying symbol
            target_dte: Target days to expiration
            short_delta: Delta for short leg (typically ~0.30 OTM)
            spread_width: Dollar width between strikes
            quantity: Number of spreads

        Returns:
            OptionsSpread or None if construction fails
        """
        chain = self.chain_fetcher.fetch_chain(symbol)
        if not chain:
            return None

        expiration = chain.get_expiration(target_dte)
        if not expiration:
            return None

        # Get short call (the one we sell)
        short_call = chain.get_contract_by_delta(expiration, OptionType.CALL, short_delta)
        if not short_call:
            return None

        # Find long call at higher strike
        target_long_strike = short_call.strike + spread_width
        contracts = chain.get_contracts_by_expiration(expiration, OptionType.CALL)
        long_call = min(
            [c for c in contracts if c.strike >= target_long_strike],
            key=lambda c: abs(c.strike - target_long_strike),
            default=None,
        )

        if not long_call:
            return None

        spread = OptionsSpread(
            spread_type=SpreadType.BEAR_CALL_SPREAD,
            symbol=symbol,
            expiration=expiration,
            legs=[
                SpreadLeg(contract=short_call, quantity=-quantity, side="sell"),
                SpreadLeg(contract=long_call, quantity=quantity, side="buy"),
            ],
        )

        logger.info(f"Built bear call spread for {symbol}: {short_call.strike}/{long_call.strike}")
        return spread

    def build_iron_condor(
        self,
        symbol: str,
        target_dte: int = 30,
        put_short_delta: float = 0.15,
        call_short_delta: float = 0.15,
        wing_width: float = 5.0,
        quantity: int = 1,
    ) -> Optional[OptionsSpread]:
        """
        Build an iron condor (credit strategy).

        Combines bull put spread + bear call spread.
        Profits if stock stays between short strikes.

        Args:
            symbol: Underlying symbol
            target_dte: Target days to expiration
            put_short_delta: Delta for short put (~0.15 = 15% ITM probability)
            call_short_delta: Delta for short call (~0.15)
            wing_width: Dollar width of each wing
            quantity: Number of spreads

        Returns:
            OptionsSpread or None if construction fails
        """
        chain = self.chain_fetcher.fetch_chain(symbol)
        if not chain:
            return None

        expiration = chain.get_expiration(target_dte)
        if not expiration:
            return None

        # Get short put and short call
        short_put = chain.get_contract_by_delta(expiration, OptionType.PUT, put_short_delta)
        short_call = chain.get_contract_by_delta(expiration, OptionType.CALL, call_short_delta)

        if not short_put or not short_call:
            return None

        # Find wing contracts
        put_contracts = chain.get_contracts_by_expiration(expiration, OptionType.PUT)
        call_contracts = chain.get_contracts_by_expiration(expiration, OptionType.CALL)

        target_long_put = short_put.strike - wing_width
        long_put = min(
            [c for c in put_contracts if c.strike <= target_long_put],
            key=lambda c: abs(c.strike - target_long_put),
            default=None,
        )

        target_long_call = short_call.strike + wing_width
        long_call = min(
            [c for c in call_contracts if c.strike >= target_long_call],
            key=lambda c: abs(c.strike - target_long_call),
            default=None,
        )

        if not long_put or not long_call:
            return None

        spread = OptionsSpread(
            spread_type=SpreadType.IRON_CONDOR,
            symbol=symbol,
            expiration=expiration,
            legs=[
                # Put spread (bull put)
                SpreadLeg(contract=long_put, quantity=quantity, side="buy"),
                SpreadLeg(contract=short_put, quantity=-quantity, side="sell"),
                # Call spread (bear call)
                SpreadLeg(contract=short_call, quantity=-quantity, side="sell"),
                SpreadLeg(contract=long_call, quantity=quantity, side="buy"),
            ],
        )

        logger.info(
            f"Built iron condor for {symbol}: "
            f"{long_put.strike}/{short_put.strike}/{short_call.strike}/{long_call.strike}"
        )
        return spread

    def build_long_straddle(
        self,
        symbol: str,
        target_dte: int = 30,
        quantity: int = 1,
    ) -> Optional[OptionsSpread]:
        """
        Build a long straddle (volatility play).

        Buy ATM call + ATM put.
        Profits on large moves in either direction.

        Args:
            symbol: Underlying symbol
            target_dte: Target days to expiration
            quantity: Number of straddles

        Returns:
            OptionsSpread or None if construction fails
        """
        chain = self.chain_fetcher.fetch_chain(symbol)
        if not chain:
            return None

        expiration = chain.get_expiration(target_dte)
        if not expiration:
            return None

        atm_strike = chain.get_atm_strike(expiration)
        if atm_strike is None:
            return None

        # Find ATM call and put
        calls = chain.get_contracts_by_expiration(expiration, OptionType.CALL)
        puts = chain.get_contracts_by_expiration(expiration, OptionType.PUT)

        atm_call = min(calls, key=lambda c: abs(c.strike - atm_strike), default=None)
        atm_put = min(puts, key=lambda c: abs(c.strike - atm_strike), default=None)

        if not atm_call or not atm_put:
            return None

        spread = OptionsSpread(
            spread_type=SpreadType.LONG_STRADDLE,
            symbol=symbol,
            expiration=expiration,
            legs=[
                SpreadLeg(contract=atm_call, quantity=quantity, side="buy"),
                SpreadLeg(contract=atm_put, quantity=quantity, side="buy"),
            ],
        )

        logger.info(f"Built long straddle for {symbol} @ {atm_strike}")
        return spread

    def build_long_strangle(
        self,
        symbol: str,
        target_dte: int = 30,
        call_delta: float = 0.25,
        put_delta: float = 0.25,
        quantity: int = 1,
    ) -> Optional[OptionsSpread]:
        """
        Build a long strangle (cheaper volatility play).

        Buy OTM call + OTM put.
        Similar to straddle but requires larger move.

        Args:
            symbol: Underlying symbol
            target_dte: Target days to expiration
            call_delta: Delta for OTM call
            put_delta: Delta for OTM put
            quantity: Number of strangles

        Returns:
            OptionsSpread or None if construction fails
        """
        chain = self.chain_fetcher.fetch_chain(symbol)
        if not chain:
            return None

        expiration = chain.get_expiration(target_dte)
        if not expiration:
            return None

        otm_call = chain.get_contract_by_delta(expiration, OptionType.CALL, call_delta)
        otm_put = chain.get_contract_by_delta(expiration, OptionType.PUT, put_delta)

        if not otm_call or not otm_put:
            return None

        spread = OptionsSpread(
            spread_type=SpreadType.LONG_STRANGLE,
            symbol=symbol,
            expiration=expiration,
            legs=[
                SpreadLeg(contract=otm_call, quantity=quantity, side="buy"),
                SpreadLeg(contract=otm_put, quantity=quantity, side="buy"),
            ],
        )

        logger.info(f"Built long strangle for {symbol}: {otm_put.strike}/{otm_call.strike}")
        return spread

    def suggest_spread(
        self,
        symbol: str,
        bias: str = "neutral",  # "bullish", "bearish", "neutral", "volatile"
        target_dte: int = 30,
        max_risk: float = 500.0,
        prefer_credit: bool = True,
    ) -> Optional[OptionsSpread]:
        """
        Suggest an appropriate spread based on market bias.

        Args:
            symbol: Underlying symbol
            bias: Market outlook ("bullish", "bearish", "neutral", "volatile")
            target_dte: Target days to expiration
            max_risk: Maximum risk per spread
            prefer_credit: Prefer credit spreads when possible

        Returns:
            Suggested OptionsSpread or None
        """
        if bias == "bullish":
            if prefer_credit:
                spread = self.build_bull_put_spread(symbol, target_dte)
            else:
                spread = self.build_bull_call_spread(symbol, target_dte)

        elif bias == "bearish":
            if prefer_credit:
                spread = self.build_bear_call_spread(symbol, target_dte)
            else:
                spread = self.build_bear_put_spread(symbol, target_dte)

        elif bias == "neutral":
            spread = self.build_iron_condor(symbol, target_dte)

        elif bias == "volatile":
            spread = self.build_long_straddle(symbol, target_dte)

        else:
            logger.warning(f"Unknown bias: {bias}")
            return None

        if spread and spread.max_loss and spread.max_loss > max_risk:
            logger.warning(f"Spread risk ${spread.max_loss:.2f} exceeds max ${max_risk:.2f}")
            # Could scale down quantity here

        return spread


# Default instances
def get_spread_builder() -> SpreadBuilder:
    """Get default spread builder instance."""
    return SpreadBuilder()
