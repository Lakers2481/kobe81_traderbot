"""
Market Simulator - Hive Mind Order Book and Price Discovery

Simulates realistic market dynamics with multi-agent interaction:
- Limit order book with price-time priority
- Market impact and slippage modeling
- Emergent price discovery from agent orders
- Scenario injection (flash crash, mania, etc.)

Author: Kobe Trading System
Date: 2026-01-07
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable
from heapq import heappush, heappop

import numpy as np
import pandas as pd

from simulation.market_agents import (
    MarketAgent,
    Order,
    OrderSide,
    OrderType,
    create_agent_population,
)

logger = logging.getLogger(__name__)


@dataclass
class Fill:
    """Record of an executed trade."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    agent_id: str


@dataclass
class SimulatedOrder:
    """Order in the order book."""
    order: Order
    remaining_qty: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def price(self) -> float:
        return self.order.limit_price or 0

    @property
    def side(self) -> OrderSide:
        return self.order.side


class OrderBook:
    """
    Limit order book with price-time priority matching.

    Maintains bid/ask levels and matches incoming orders.
    """

    def __init__(self, symbol: str, tick_size: float = 0.01):
        self.symbol = symbol
        self.tick_size = tick_size
        self.bids: List[Tuple[float, datetime, SimulatedOrder]] = []  # Max heap (negative price)
        self.asks: List[Tuple[float, datetime, SimulatedOrder]] = []  # Min heap
        self.fills: List[Fill] = []
        self._last_price: float = 0

    def add_order(self, order: Order, current_price: float) -> List[Fill]:
        """
        Add order to book, match if possible.

        Returns list of fills generated.
        """
        fills = []

        if order.order_type == OrderType.MARKET:
            fills = self._execute_market_order(order, current_price)
        else:
            fills = self._execute_limit_order(order, current_price)

        return fills

    def _execute_market_order(self, order: Order, current_price: float) -> List[Fill]:
        """Execute market order against book."""
        fills = []
        remaining = order.quantity

        if order.side == OrderSide.BUY:
            # Match against asks
            while remaining > 0 and self.asks:
                _, _, best_ask = heappop(self.asks)
                fill_qty = min(remaining, best_ask.remaining_qty)
                fill_price = best_ask.price if best_ask.price > 0 else current_price

                fills.append(Fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=fill_qty,
                    price=fill_price,
                    timestamp=datetime.now(),
                    agent_id=order.agent_id,
                ))

                # Update counterparty
                fills.append(Fill(
                    order_id=best_ask.order.order_id,
                    symbol=order.symbol,
                    side=best_ask.side,
                    quantity=fill_qty,
                    price=fill_price,
                    timestamp=datetime.now(),
                    agent_id=best_ask.order.agent_id,
                ))

                best_ask.remaining_qty -= fill_qty
                remaining -= fill_qty
                self._last_price = fill_price

                # Re-add if partially filled
                if best_ask.remaining_qty > 0:
                    heappush(self.asks, (best_ask.price, best_ask.timestamp, best_ask))

        else:  # SELL
            # Match against bids
            while remaining > 0 and self.bids:
                neg_price, ts, best_bid = heappop(self.bids)
                fill_qty = min(remaining, best_bid.remaining_qty)
                fill_price = -neg_price if neg_price < 0 else current_price

                fills.append(Fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=fill_qty,
                    price=fill_price,
                    timestamp=datetime.now(),
                    agent_id=order.agent_id,
                ))

                fills.append(Fill(
                    order_id=best_bid.order.order_id,
                    symbol=order.symbol,
                    side=best_bid.side,
                    quantity=fill_qty,
                    price=fill_price,
                    timestamp=datetime.now(),
                    agent_id=best_bid.order.agent_id,
                ))

                best_bid.remaining_qty -= fill_qty
                remaining -= fill_qty
                self._last_price = fill_price

                if best_bid.remaining_qty > 0:
                    heappush(self.bids, (-fill_price, ts, best_bid))

        self.fills.extend(fills)
        return fills

    def _execute_limit_order(self, order: Order, current_price: float) -> List[Fill]:
        """Execute limit order - match or add to book."""
        fills = []
        remaining = order.quantity
        limit = order.limit_price or current_price

        if order.side == OrderSide.BUY:
            # Try to match against asks at or below limit
            while remaining > 0 and self.asks:
                ask_price, ts, best_ask = self.asks[0]
                if ask_price > limit:
                    break  # No more matchable orders

                heappop(self.asks)
                fill_qty = min(remaining, best_ask.remaining_qty)

                fills.append(Fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=fill_qty,
                    price=ask_price,
                    timestamp=datetime.now(),
                    agent_id=order.agent_id,
                ))

                fills.append(Fill(
                    order_id=best_ask.order.order_id,
                    symbol=order.symbol,
                    side=best_ask.side,
                    quantity=fill_qty,
                    price=ask_price,
                    timestamp=datetime.now(),
                    agent_id=best_ask.order.agent_id,
                ))

                best_ask.remaining_qty -= fill_qty
                remaining -= fill_qty
                self._last_price = ask_price

                if best_ask.remaining_qty > 0:
                    heappush(self.asks, (ask_price, ts, best_ask))

            # Add remainder to book
            if remaining > 0:
                sim_order = SimulatedOrder(order=order, remaining_qty=remaining)
                heappush(self.bids, (-limit, datetime.now(), sim_order))

        else:  # SELL
            while remaining > 0 and self.bids:
                neg_price, ts, best_bid = self.bids[0]
                bid_price = -neg_price
                if bid_price < limit:
                    break

                heappop(self.bids)
                fill_qty = min(remaining, best_bid.remaining_qty)

                fills.append(Fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=fill_qty,
                    price=bid_price,
                    timestamp=datetime.now(),
                    agent_id=order.agent_id,
                ))

                fills.append(Fill(
                    order_id=best_bid.order.order_id,
                    symbol=order.symbol,
                    side=best_bid.side,
                    quantity=fill_qty,
                    price=bid_price,
                    timestamp=datetime.now(),
                    agent_id=best_bid.order.agent_id,
                ))

                best_bid.remaining_qty -= fill_qty
                remaining -= fill_qty
                self._last_price = bid_price

                if best_bid.remaining_qty > 0:
                    heappush(self.bids, (-bid_price, ts, best_bid))

            if remaining > 0:
                sim_order = SimulatedOrder(order=order, remaining_qty=remaining)
                heappush(self.asks, (limit, datetime.now(), sim_order))

        self.fills.extend(fills)
        return fills

    def get_state(self) -> Dict[str, Any]:
        """Get current order book state."""
        best_bid = -self.bids[0][0] if self.bids else 0
        best_ask = self.asks[0][0] if self.asks else float('inf')

        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': best_ask - best_bid if best_bid > 0 and best_ask < float('inf') else 0,
            'bid_depth': len(self.bids),
            'ask_depth': len(self.asks),
            'last_price': self._last_price,
        }

    def cancel_stale_orders(self, max_age_seconds: int = 300) -> int:
        """Cancel orders older than max_age."""
        cutoff = datetime.now() - timedelta(seconds=max_age_seconds)
        original_bids = len(self.bids)
        original_asks = len(self.asks)

        self.bids = [b for b in self.bids if b[1] > cutoff]
        self.asks = [a for a in self.asks if a[1] > cutoff]

        cancelled = (original_bids - len(self.bids)) + (original_asks - len(self.asks))
        return cancelled


@dataclass
class SimulationResult:
    """Result of a market simulation run."""
    symbol: str
    duration_bars: int
    price_history: List[float]
    volume_history: List[int]
    fills: List[Fill]
    agent_pnls: Dict[str, float]
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    final_price: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert price history to DataFrame."""
        return pd.DataFrame({
            'close': self.price_history,
            'volume': self.volume_history,
        })


class ScenarioType(Enum):
    """Types of market scenarios to inject."""
    NORMAL = "normal"
    FLASH_CRASH = "flash_crash"
    MANIA = "mania"
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"


class MarketSimulator:
    """
    Multi-agent market simulator with emergent price discovery.

    Creates realistic synthetic market data by simulating interactions
    between diverse trader types. Can inject scenarios to stress test
    trading strategies.
    """

    def __init__(
        self,
        symbol: str = "SIM",
        initial_price: float = 100.0,
        tick_size: float = 0.01,
        seed: int = 42,
    ):
        self.symbol = symbol
        self.initial_price = initial_price
        self.tick_size = tick_size
        self.seed = seed

        self.order_book = OrderBook(symbol, tick_size)
        self.agents: List[MarketAgent] = []
        self.price_history: List[float] = [initial_price]
        self.volume_history: List[int] = [0]

        self._random = random.Random(seed)
        np.random.seed(seed)

    def add_agents(self, agents: List[MarketAgent]) -> None:
        """Add agents to simulation."""
        self.agents.extend(agents)
        logger.info(f"Added {len(agents)} agents, total: {len(self.agents)}")

    def create_default_population(self) -> None:
        """Create default agent population."""
        agents = create_agent_population(seed=self.seed)
        self.add_agents(agents)

    def run_simulation(
        self,
        n_bars: int = 1000,
        scenario: ScenarioType = ScenarioType.NORMAL,
        scenario_start: int = 500,
        scenario_duration: int = 100,
    ) -> SimulationResult:
        """
        Run market simulation for n_bars.

        Args:
            n_bars: Number of price bars to simulate
            scenario: Type of scenario to inject
            scenario_start: Bar number where scenario begins
            scenario_duration: How long scenario lasts

        Returns:
            SimulationResult with price history and metrics
        """
        if not self.agents:
            self.create_default_population()

        logger.info(f"Starting simulation: {n_bars} bars, scenario={scenario.value}")

        for bar in range(n_bars):
            current_price = self.price_history[-1]

            # Check for scenario injection
            scenario_active = scenario_start <= bar < scenario_start + scenario_duration
            price_adjustment = self._get_scenario_adjustment(
                scenario, bar - scenario_start
            ) if scenario_active and scenario != ScenarioType.NORMAL else 0

            # Apply scenario adjustment to price
            if price_adjustment != 0:
                current_price *= (1 + price_adjustment)

            # Get order book state
            book_state = self.order_book.get_state()

            # Collect orders from all agents
            all_orders = []
            for agent in self.agents:
                orders = agent.generate_orders(
                    symbol=self.symbol,
                    current_price=current_price,
                    price_history=self.price_history,
                    volume_history=self.volume_history,
                    order_book_state=book_state,
                )
                all_orders.extend(orders)

            # Shuffle orders (no unfair advantage)
            self._random.shuffle(all_orders)

            # Execute orders
            bar_volume = 0
            for order in all_orders:
                fills = self.order_book.add_order(order, current_price)
                for fill in fills:
                    bar_volume += fill.quantity
                    # Update agent positions
                    agent = self._find_agent(fill.agent_id)
                    if agent:
                        qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
                        agent.update_position(self.symbol, qty, fill.price)

            # Determine bar close price from last fill or random walk
            if self.order_book.fills:
                recent_fills = [f for f in self.order_book.fills[-10:]
                               if f.symbol == self.symbol]
                if recent_fills:
                    current_price = recent_fills[-1].price

            # Add some random noise to price
            noise = self._random.gauss(0, 0.001)  # 0.1% noise
            current_price = max(current_price * (1 + noise), self.tick_size)

            self.price_history.append(round(current_price, 2))
            self.volume_history.append(bar_volume)

            # Clean up stale orders periodically
            if bar % 100 == 0:
                self.order_book.cancel_stale_orders()

        # Calculate results
        return self._calculate_results()

    def _get_scenario_adjustment(self, scenario: ScenarioType, bar_in_scenario: int) -> float:
        """Get price adjustment for scenario."""
        if scenario == ScenarioType.FLASH_CRASH:
            if bar_in_scenario < 20:
                return -0.002 * (1 + bar_in_scenario * 0.5)  # Accelerating drop
            elif bar_in_scenario < 50:
                return 0.001  # Slow recovery
            return 0

        elif scenario == ScenarioType.MANIA:
            if bar_in_scenario < 50:
                return 0.002 * (1 + bar_in_scenario * 0.1)  # Accelerating rally
            return -0.003  # Sharp correction

        elif scenario == ScenarioType.HIGH_VOLATILITY:
            return self._random.gauss(0, 0.01)  # 1% std dev moves

        elif scenario == ScenarioType.LOW_VOLATILITY:
            return self._random.gauss(0, 0.001)  # 0.1% std dev moves

        elif scenario == ScenarioType.TRENDING_UP:
            return 0.001  # Steady uptrend

        elif scenario == ScenarioType.TRENDING_DOWN:
            return -0.001  # Steady downtrend

        elif scenario == ScenarioType.MEAN_REVERTING:
            deviation = (self.price_history[-1] - self.initial_price) / self.initial_price
            return -deviation * 0.1  # Pull back toward mean

        return 0

    def _find_agent(self, agent_id: str) -> Optional[MarketAgent]:
        """Find agent by ID."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def _calculate_results(self) -> SimulationResult:
        """Calculate simulation results and metrics."""
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]

        # Sharpe ratio (annualized)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

        # Agent P&Ls
        agent_pnls = {}
        for agent in self.agents:
            position = agent.get_position(self.symbol)
            position_value = position * self.price_history[-1]
            total_value = agent.state.cash + position_value
            pnl = total_value - 100000  # Assuming 100k initial
            agent_pnls[agent.agent_id] = pnl

        return SimulationResult(
            symbol=self.symbol,
            duration_bars=len(self.price_history),
            price_history=self.price_history,
            volume_history=self.volume_history,
            fills=self.order_book.fills,
            agent_pnls=agent_pnls,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            volatility=volatility,
            final_price=self.price_history[-1],
        )

    def test_strategy(
        self,
        strategy_func: Callable[[pd.DataFrame], pd.DataFrame],
        n_simulations: int = 100,
        n_bars: int = 500,
        scenarios: Optional[List[ScenarioType]] = None,
    ) -> Dict[str, Any]:
        """
        Test a trading strategy against multiple simulations.

        Args:
            strategy_func: Function that takes OHLCV DataFrame and returns signals
            n_simulations: Number of simulation runs
            n_bars: Bars per simulation
            scenarios: List of scenarios to test (defaults to all)

        Returns:
            Dict with aggregated performance metrics
        """
        if scenarios is None:
            scenarios = list(ScenarioType)

        results = []

        for scenario in scenarios:
            for i in range(n_simulations // len(scenarios)):
                # Reset simulator
                self.order_book = OrderBook(self.symbol, self.tick_size)
                self.price_history = [self.initial_price]
                self.volume_history = [0]

                # Vary seed for each run
                self._random = random.Random(self.seed + i + hash(scenario))

                # Run simulation
                sim_result = self.run_simulation(
                    n_bars=n_bars,
                    scenario=scenario,
                    scenario_start=n_bars // 2,
                    scenario_duration=n_bars // 10,
                )

                # Create OHLCV-like DataFrame
                df = pd.DataFrame({
                    'open': sim_result.price_history,
                    'high': [p * 1.001 for p in sim_result.price_history],
                    'low': [p * 0.999 for p in sim_result.price_history],
                    'close': sim_result.price_history,
                    'volume': sim_result.volume_history,
                })

                # Get strategy signals
                try:
                    signals = strategy_func(df)
                    # Calculate strategy P&L (simplified)
                    if 'signal' in signals.columns:
                        strategy_returns = signals['signal'].shift(1) * (
                            df['close'].pct_change()
                        )
                        strategy_sharpe = (
                            strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                            if strategy_returns.std() > 0 else 0
                        )
                    else:
                        strategy_sharpe = 0
                except Exception as e:
                    logger.warning(f"Strategy error: {e}")
                    strategy_sharpe = 0

                results.append({
                    'scenario': scenario.value,
                    'market_sharpe': sim_result.sharpe_ratio,
                    'strategy_sharpe': strategy_sharpe,
                    'volatility': sim_result.volatility,
                    'max_drawdown': sim_result.max_drawdown,
                })

        # Aggregate results
        results_df = pd.DataFrame(results)
        return {
            'n_simulations': len(results),
            'mean_strategy_sharpe': results_df['strategy_sharpe'].mean(),
            'std_strategy_sharpe': results_df['strategy_sharpe'].std(),
            'min_strategy_sharpe': results_df['strategy_sharpe'].min(),
            'max_strategy_sharpe': results_df['strategy_sharpe'].max(),
            'by_scenario': results_df.groupby('scenario')['strategy_sharpe'].agg(['mean', 'std']).to_dict(),
            'worst_case_drawdown': results_df['max_drawdown'].max(),
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create simulator
    sim = MarketSimulator(symbol="SIM", initial_price=100.0, seed=42)

    # Run basic simulation
    print("Running basic simulation...")
    result = sim.run_simulation(n_bars=500, scenario=ScenarioType.NORMAL)

    print(f"\nSimulation Results:")
    print(f"  Final Price: ${result.final_price:.2f}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown:.1%}")
    print(f"  Volatility: {result.volatility:.1%}")
    print(f"  Total Fills: {len(result.fills)}")

    # Run flash crash scenario
    print("\nRunning flash crash scenario...")
    sim2 = MarketSimulator(symbol="SIM", initial_price=100.0, seed=123)
    crash_result = sim2.run_simulation(
        n_bars=500,
        scenario=ScenarioType.FLASH_CRASH,
        scenario_start=200,
        scenario_duration=100,
    )

    print(f"\nFlash Crash Results:")
    print(f"  Final Price: ${crash_result.final_price:.2f}")
    print(f"  Max Drawdown: {crash_result.max_drawdown:.1%}")
