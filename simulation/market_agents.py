"""
Market Agents - Trader Personas for Hive Mind Simulation

Different agent types with distinct trading behaviors:
- ValueInvestorAgent: Buys undervalued, sells overvalued, low frequency
- MomentumFollowerAgent: Trend following, medium frequency
- HFTScalperAgent: High frequency, small profits, liquidity provider
- RiskAversePensionFundAgent: Conservative, rebalancing driven
- RetailTraderAgent: Emotional, FOMO/panic driven
- MarketMakerAgent: Two-sided quotes, inventory management

Author: Kobe Trading System
Date: 2026-01-07
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class Order:
    """Order submitted by an agent."""
    agent_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    order_id: str = field(default_factory=lambda: f"ORD_{random.randint(100000, 999999)}")


@dataclass
class AgentState:
    """Current state of an agent."""
    cash: float = 100000.0
    positions: Dict[str, int] = field(default_factory=dict)
    pnl: float = 0.0
    trades_today: int = 0
    last_trade_time: Optional[datetime] = None


class MarketAgent(ABC):
    """
    Abstract base class for market simulation agents.

    Each agent type has:
    - Distinct trading behavior and frequency
    - Different response to price/volume signals
    - Unique risk tolerance and position sizing
    """

    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 100000.0,
        risk_tolerance: float = 0.5,
    ):
        self.agent_id = agent_id
        self.state = AgentState(cash=initial_cash)
        self.risk_tolerance = risk_tolerance
        self._random = random.Random(hash(agent_id))

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return agent type identifier."""
        pass

    @abstractmethod
    def generate_orders(
        self,
        symbol: str,
        current_price: float,
        price_history: List[float],
        volume_history: List[int],
        order_book_state: Dict[str, Any],
    ) -> List[Order]:
        """
        Generate orders based on market state.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            price_history: Recent price history (oldest first)
            volume_history: Recent volume history
            order_book_state: Current order book state (bids, asks, spread)

        Returns:
            List of orders to submit
        """
        pass

    def get_position(self, symbol: str) -> int:
        """Get current position in symbol."""
        return self.state.positions.get(symbol, 0)

    def update_position(self, symbol: str, quantity: int, price: float) -> None:
        """Update position after fill."""
        current = self.state.positions.get(symbol, 0)
        self.state.positions[symbol] = current + quantity
        self.state.cash -= quantity * price
        self.state.trades_today += 1
        self.state.last_trade_time = datetime.now()

    def _calculate_position_size(self, price: float, max_pct: float = 0.1) -> int:
        """Calculate position size based on risk tolerance."""
        max_value = self.state.cash * max_pct * self.risk_tolerance
        return int(max_value / price)


class ValueInvestorAgent(MarketAgent):
    """
    Value investor that buys undervalued and sells overvalued assets.

    Characteristics:
    - Low trading frequency (holds for long periods)
    - Uses moving averages to estimate fair value
    - Contrarian: buys dips, sells rips
    - Large position sizes when conviction is high
    """

    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 500000.0,
        value_threshold: float = 0.05,  # 5% deviation triggers action
        ma_period: int = 50,
    ):
        super().__init__(agent_id, initial_cash, risk_tolerance=0.3)
        self.value_threshold = value_threshold
        self.ma_period = ma_period

    @property
    def agent_type(self) -> str:
        return "VALUE_INVESTOR"

    def generate_orders(
        self,
        symbol: str,
        current_price: float,
        price_history: List[float],
        volume_history: List[int],
        order_book_state: Dict[str, Any],
    ) -> List[Order]:
        orders = []

        if len(price_history) < self.ma_period:
            return orders

        # Calculate fair value (simple MA)
        fair_value = np.mean(price_history[-self.ma_period:])
        deviation = (current_price - fair_value) / fair_value

        position = self.get_position(symbol)

        # Buy if significantly undervalued
        if deviation < -self.value_threshold and position >= 0:
            size = self._calculate_position_size(current_price, max_pct=0.15)
            if size > 0 and self._random.random() < 0.3:  # 30% chance to act
                orders.append(Order(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=size,
                    limit_price=current_price * 0.998,  # Slightly below market
                ))

        # Sell if significantly overvalued
        elif deviation > self.value_threshold and position > 0:
            size = min(position, self._calculate_position_size(current_price, max_pct=0.10))
            if size > 0 and self._random.random() < 0.3:
                orders.append(Order(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=size,
                    limit_price=current_price * 1.002,
                ))

        return orders


class MomentumFollowerAgent(MarketAgent):
    """
    Momentum trader that follows trends.

    Characteristics:
    - Medium trading frequency
    - Buys on breakouts, sells on breakdowns
    - Uses price momentum indicators
    - Stops out quickly on reversals
    """

    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 200000.0,
        lookback: int = 20,
        momentum_threshold: float = 0.02,
    ):
        super().__init__(agent_id, initial_cash, risk_tolerance=0.6)
        self.lookback = lookback
        self.momentum_threshold = momentum_threshold

    @property
    def agent_type(self) -> str:
        return "MOMENTUM_FOLLOWER"

    def generate_orders(
        self,
        symbol: str,
        current_price: float,
        price_history: List[float],
        volume_history: List[int],
        order_book_state: Dict[str, Any],
    ) -> List[Order]:
        orders = []

        if len(price_history) < self.lookback:
            return orders

        # Calculate momentum
        past_price = price_history[-self.lookback]
        momentum = (current_price - past_price) / past_price

        position = self.get_position(symbol)

        # Strong upward momentum - buy
        if momentum > self.momentum_threshold:
            if position <= 0 and self._random.random() < 0.5:
                size = self._calculate_position_size(current_price, max_pct=0.08)
                if size > 0:
                    orders.append(Order(
                        agent_id=self.agent_id,
                        symbol=symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=size,
                    ))

        # Strong downward momentum - sell
        elif momentum < -self.momentum_threshold:
            if position > 0 and self._random.random() < 0.5:
                orders.append(Order(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=position,
                ))

        return orders


class HFTScalperAgent(MarketAgent):
    """
    High-frequency scalper that captures small price movements.

    Characteristics:
    - Very high frequency (trades constantly)
    - Small position sizes
    - Tight profit targets, quick stops
    - Provides liquidity
    """

    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 1000000.0,
        spread_threshold: float = 0.001,  # 10 bps minimum spread
        tick_size: float = 0.01,
    ):
        super().__init__(agent_id, initial_cash, risk_tolerance=0.2)
        self.spread_threshold = spread_threshold
        self.tick_size = tick_size

    @property
    def agent_type(self) -> str:
        return "HFT_SCALPER"

    def generate_orders(
        self,
        symbol: str,
        current_price: float,
        price_history: List[float],
        volume_history: List[int],
        order_book_state: Dict[str, Any],
    ) -> List[Order]:
        orders = []

        # Calculate spread opportunity
        best_bid = order_book_state.get('best_bid', current_price * 0.999)
        best_ask = order_book_state.get('best_ask', current_price * 1.001)
        spread = (best_ask - best_bid) / current_price

        position = self.get_position(symbol)

        # Always try to provide liquidity if spread is wide enough
        if spread > self.spread_threshold:
            size = self._calculate_position_size(current_price, max_pct=0.01)

            # Place bid slightly above best bid
            if self._random.random() < 0.7:  # 70% chance to quote
                orders.append(Order(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=size,
                    limit_price=best_bid + self.tick_size,
                ))

                # Place ask slightly below best ask
                orders.append(Order(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=size,
                    limit_price=best_ask - self.tick_size,
                ))

        # Inventory management - flatten position periodically
        if abs(position) > 100 and self._random.random() < 0.3:
            side = OrderSide.SELL if position > 0 else OrderSide.BUY
            orders.append(Order(
                agent_id=self.agent_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=abs(position) // 2,
            ))

        return orders


class RiskAversePensionFundAgent(MarketAgent):
    """
    Conservative pension fund that rebalances periodically.

    Characteristics:
    - Very low frequency (monthly rebalancing)
    - Large position sizes
    - Target allocation driven
    - Avoids volatile periods
    """

    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 10000000.0,
        target_allocation: float = 0.6,  # 60% equity target
        rebalance_threshold: float = 0.05,  # 5% drift triggers rebalance
    ):
        super().__init__(agent_id, initial_cash, risk_tolerance=0.1)
        self.target_allocation = target_allocation
        self.rebalance_threshold = rebalance_threshold

    @property
    def agent_type(self) -> str:
        return "PENSION_FUND"

    def generate_orders(
        self,
        symbol: str,
        current_price: float,
        price_history: List[float],
        volume_history: List[int],
        order_book_state: Dict[str, Any],
    ) -> List[Order]:
        orders = []

        position = self.get_position(symbol)
        position_value = position * current_price
        total_value = self.state.cash + position_value
        current_allocation = position_value / total_value if total_value > 0 else 0

        allocation_drift = current_allocation - self.target_allocation

        # Check volatility - don't trade in high volatility
        if len(price_history) >= 20:
            volatility = np.std(price_history[-20:]) / np.mean(price_history[-20:])
            if volatility > 0.03:  # 3% volatility threshold
                return orders  # Stay on sidelines

        # Rebalance if drift exceeds threshold
        if abs(allocation_drift) > self.rebalance_threshold:
            if self._random.random() < 0.1:  # Low probability to act (monthly-ish)
                target_value = total_value * self.target_allocation
                target_shares = int(target_value / current_price)
                delta = target_shares - position

                if delta > 0:
                    orders.append(Order(
                        agent_id=self.agent_id,
                        symbol=symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.LIMIT,
                        quantity=delta,
                        limit_price=current_price * 0.995,  # Patient buyer
                    ))
                elif delta < 0:
                    orders.append(Order(
                        agent_id=self.agent_id,
                        symbol=symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.LIMIT,
                        quantity=abs(delta),
                        limit_price=current_price * 1.005,  # Patient seller
                    ))

        return orders


class RetailTraderAgent(MarketAgent):
    """
    Retail trader driven by emotion and FOMO.

    Characteristics:
    - Erratic trading frequency
    - Buys tops, sells bottoms (poor timing)
    - Influenced by recent price action (recency bias)
    - Panic sells on drops, FOMO buys on rips
    """

    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 25000.0,
        fomo_threshold: float = 0.03,  # 3% move triggers FOMO
        panic_threshold: float = -0.02,  # 2% drop triggers panic
    ):
        super().__init__(agent_id, initial_cash, risk_tolerance=0.8)
        self.fomo_threshold = fomo_threshold
        self.panic_threshold = panic_threshold

    @property
    def agent_type(self) -> str:
        return "RETAIL_TRADER"

    def generate_orders(
        self,
        symbol: str,
        current_price: float,
        price_history: List[float],
        volume_history: List[int],
        order_book_state: Dict[str, Any],
    ) -> List[Order]:
        orders = []

        if len(price_history) < 5:
            return orders

        # Recent move (recency bias)
        recent_return = (current_price - price_history[-5]) / price_history[-5]

        position = self.get_position(symbol)

        # FOMO buying - chase the rally
        if recent_return > self.fomo_threshold:
            if position == 0 and self._random.random() < 0.6:  # Impulsive
                size = self._calculate_position_size(current_price, max_pct=0.20)
                orders.append(Order(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,  # Market order - needs it NOW
                    quantity=size,
                ))

        # Panic selling - bail on drops
        elif recent_return < self.panic_threshold:
            if position > 0 and self._random.random() < 0.7:  # Fearful
                orders.append(Order(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,  # Market order - get out NOW
                    quantity=position,
                ))

        return orders


class MarketMakerAgent(MarketAgent):
    """
    Market maker providing two-sided quotes.

    Characteristics:
    - Continuous quoting (high frequency)
    - Earns spread
    - Manages inventory risk
    - Widens spreads in volatile conditions
    """

    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 5000000.0,
        base_spread: float = 0.001,  # 10 bps base spread
        inventory_limit: int = 1000,
    ):
        super().__init__(agent_id, initial_cash, risk_tolerance=0.4)
        self.base_spread = base_spread
        self.inventory_limit = inventory_limit

    @property
    def agent_type(self) -> str:
        return "MARKET_MAKER"

    def generate_orders(
        self,
        symbol: str,
        current_price: float,
        price_history: List[float],
        volume_history: List[int],
        order_book_state: Dict[str, Any],
    ) -> List[Order]:
        orders = []

        position = self.get_position(symbol)

        # Calculate dynamic spread based on volatility
        if len(price_history) >= 10:
            volatility = np.std(price_history[-10:]) / np.mean(price_history[-10:])
            spread = self.base_spread * (1 + volatility * 10)  # Widen in vol
        else:
            spread = self.base_spread

        # Skew quotes based on inventory
        inventory_ratio = position / self.inventory_limit if self.inventory_limit > 0 else 0
        bid_skew = -inventory_ratio * spread  # Lower bid when long
        ask_skew = inventory_ratio * spread   # Higher ask when long

        half_spread = spread / 2
        size = self._calculate_position_size(current_price, max_pct=0.02)

        # Only quote if within inventory limits
        if abs(position) < self.inventory_limit:
            # Bid
            bid_price = current_price * (1 - half_spread + bid_skew)
            orders.append(Order(
                agent_id=self.agent_id,
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=size,
                limit_price=round(bid_price, 2),
            ))

            # Ask
            ask_price = current_price * (1 + half_spread + ask_skew)
            orders.append(Order(
                agent_id=self.agent_id,
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=size,
                limit_price=round(ask_price, 2),
            ))

        return orders


def create_agent_population(
    seed: int = 42,
    n_value: int = 5,
    n_momentum: int = 10,
    n_hft: int = 3,
    n_pension: int = 2,
    n_retail: int = 50,
    n_mm: int = 3,
) -> List[MarketAgent]:
    """
    Create a population of diverse market agents.

    Default mix approximates real market participant distribution.
    """
    random.seed(seed)
    agents = []

    for i in range(n_value):
        agents.append(ValueInvestorAgent(
            f"VALUE_{i}",
            initial_cash=random.uniform(200000, 1000000),
            value_threshold=random.uniform(0.03, 0.08),
        ))

    for i in range(n_momentum):
        agents.append(MomentumFollowerAgent(
            f"MOMENTUM_{i}",
            initial_cash=random.uniform(100000, 500000),
            momentum_threshold=random.uniform(0.01, 0.04),
        ))

    for i in range(n_hft):
        agents.append(HFTScalperAgent(
            f"HFT_{i}",
            initial_cash=random.uniform(500000, 2000000),
        ))

    for i in range(n_pension):
        agents.append(RiskAversePensionFundAgent(
            f"PENSION_{i}",
            initial_cash=random.uniform(5000000, 20000000),
        ))

    for i in range(n_retail):
        agents.append(RetailTraderAgent(
            f"RETAIL_{i}",
            initial_cash=random.uniform(10000, 100000),
        ))

    for i in range(n_mm):
        agents.append(MarketMakerAgent(
            f"MM_{i}",
            initial_cash=random.uniform(2000000, 10000000),
        ))

    logger.info(f"Created agent population: {len(agents)} agents")
    return agents
