"""
MEDALLION ORCHESTRATOR
======================

The brain of the Renaissance-inspired system.
Coordinates all subsystems to generate trading decisions.

"The system is like a large factory with many machines.
Each machine does one thing well. Together they make money."
- Paraphrased from Renaissance approach

This orchestrator:
1. Queries regime engine for market state
2. Generates signals from multiple strategies
3. Filters signals through risk engine
4. Optimizes portfolio for diversification
5. Executes with smart order routing
6. Learns from outcomes
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
from enum import Enum
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode determines aggression level."""
    CONSERVATIVE = "conservative"  # 2x leverage, 10 positions
    MODERATE = "moderate"          # 5x leverage, 20 positions
    AGGRESSIVE = "aggressive"      # 10x leverage, 50 positions
    MEDALLION = "medallion"        # 12.5x leverage, 100+ positions


@dataclass
class MedallionConfig:
    """
    Configuration for Medallion-style trading.

    Renaissance used 12.5-20x leverage with 3,500+ positions.
    We scale down but maintain the RATIOS.
    """
    # Capital
    initial_capital: float = 100_000.0

    # Leverage (Renaissance: 12.5-20x)
    max_leverage: float = 5.0  # Conservative start
    target_leverage: float = 3.0

    # Diversification (Renaissance: 3,500+ positions)
    min_positions: int = 10
    max_positions: int = 50
    target_positions: int = 20

    # Position sizing
    max_position_pct: float = 0.10  # 10% max per position
    kelly_fraction: float = 0.25   # Quarter-Kelly for safety

    # Risk limits
    max_daily_loss_pct: float = 0.02  # 2% daily stop
    max_drawdown_pct: float = 0.15    # 15% max drawdown
    max_sector_exposure: float = 0.30  # 30% max sector
    max_correlation: float = 0.70      # Reject highly correlated positions

    # Signal thresholds
    min_edge: float = 0.005  # 0.5% minimum expected edge
    min_confidence: float = 0.52  # Barely above coin flip (Renaissance: 50.75%)

    # Regime filters
    use_regime_filter: bool = True
    regime_leverage_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'BULLISH': 1.2,   # Increase leverage in bull
        'NEUTRAL': 1.0,   # Normal leverage
        'BEARISH': 0.5,   # Reduce leverage in bear
    })

    # Trading frequency
    rebalance_frequency: str = 'daily'  # daily, hourly, or continuous
    holding_period_days: int = 2  # Renaissance: 1-2 days average

    # Mode
    mode: TradingMode = TradingMode.MODERATE


@dataclass
class PortfolioState:
    """Current state of the portfolio."""
    timestamp: datetime
    cash: float
    positions: Dict[str, 'Position']
    total_equity: float
    leverage: float
    exposure_long: float
    exposure_short: float
    regime: str
    regime_confidence: float
    daily_pnl: float
    total_pnl: float
    drawdown: float

    @property
    def gross_exposure(self) -> float:
        return self.exposure_long + abs(self.exposure_short)

    @property
    def net_exposure(self) -> float:
        return self.exposure_long + self.exposure_short


@dataclass
class Position:
    """A position in the portfolio."""
    symbol: str
    side: str  # 'long' or 'short'
    shares: int
    entry_price: float
    current_price: float
    entry_date: datetime
    strategy: str
    stop_loss: float
    take_profit: float

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.side == 'long':
            return (self.current_price - self.entry_price) * self.shares
        else:
            return (self.entry_price - self.current_price) * self.shares

    @property
    def pnl_pct(self) -> float:
        return self.unrealized_pnl / (self.entry_price * self.shares)


@dataclass
class TradeSignal:
    """A trading signal from any strategy."""
    timestamp: datetime
    symbol: str
    side: str  # 'long' or 'short'
    strategy: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    expected_edge: float
    holding_period: int
    regime_aligned: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class MedallionOrchestrator:
    """
    Master orchestrator for Medallion-style trading.

    Coordinates:
    - Regime detection (HMM + Markov)
    - Signal generation (multi-strategy ensemble)
    - Risk management (Kelly + VaR + diversification)
    - Portfolio optimization (mean-variance + constraints)
    - Execution (smart routing)
    - Learning (feedback loops)
    """

    def __init__(self, config: Optional[MedallionConfig] = None):
        self.config = config or MedallionConfig()
        self.state: Optional[PortfolioState] = None
        self.signals_history: List[TradeSignal] = []
        self.trades_history: List[Dict] = []

        # Lazy load components
        self._regime_engine = None
        self._signal_ensemble = None
        self._risk_engine = None
        self._portfolio_optimizer = None
        self._execution_engine = None

        logger.info(f"MedallionOrchestrator initialized: mode={self.config.mode.value}")

    @property
    def regime_engine(self):
        """Lazy load regime engine."""
        if self._regime_engine is None:
            from .regime_engine import RegimeEngine
            self._regime_engine = RegimeEngine()
        return self._regime_engine

    @property
    def signal_ensemble(self):
        """Lazy load signal ensemble."""
        if self._signal_ensemble is None:
            from .signal_ensemble import SignalEnsemble
            self._signal_ensemble = SignalEnsemble()
        return self._signal_ensemble

    @property
    def risk_engine(self):
        """Lazy load risk engine."""
        if self._risk_engine is None:
            from .risk_engine import MedallionRiskEngine
            self._risk_engine = MedallionRiskEngine(self.config)
        return self._risk_engine

    @property
    def portfolio_optimizer(self):
        """Lazy load portfolio optimizer."""
        if self._portfolio_optimizer is None:
            from .portfolio_optimizer import PortfolioOptimizer
            self._portfolio_optimizer = PortfolioOptimizer(self.config)
        return self._portfolio_optimizer

    def run_cycle(
        self,
        market_data: pd.DataFrame,
        current_positions: Optional[Dict[str, Position]] = None,
        current_cash: float = None,
    ) -> Dict[str, Any]:
        """
        Run one trading cycle.

        This is the main entry point called by the scheduler.

        Steps:
        1. Detect current regime (HMM + Markov)
        2. Generate signals from all strategies
        3. Filter signals through risk engine
        4. Optimize portfolio allocation
        5. Generate execution orders
        6. Return decisions

        Args:
            market_data: DataFrame with OHLCV for universe
            current_positions: Current portfolio positions
            current_cash: Available cash

        Returns:
            Dict with orders, analysis, and diagnostics
        """
        cycle_start = datetime.now()

        # Initialize state if needed
        if current_cash is None:
            current_cash = self.config.initial_capital
        if current_positions is None:
            current_positions = {}

        # Step 1: Detect regime
        regime_state = self.regime_engine.detect(market_data)
        logger.info(f"Regime: {regime_state['regime']} (confidence: {regime_state['confidence']:.2%})")

        # Step 2: Generate signals from ensemble
        raw_signals = self.signal_ensemble.generate(
            market_data,
            regime=regime_state['regime'],
            regime_confidence=regime_state['confidence']
        )
        logger.info(f"Raw signals generated: {len(raw_signals)}")

        # Step 3: Filter through risk engine
        filtered_signals = self.risk_engine.filter_signals(
            signals=raw_signals,
            current_positions=current_positions,
            regime=regime_state['regime'],
            cash=current_cash,
        )
        logger.info(f"Signals after risk filter: {len(filtered_signals)}")

        # Step 4: Optimize portfolio
        target_portfolio = self.portfolio_optimizer.optimize(
            signals=filtered_signals,
            current_positions=current_positions,
            regime=regime_state['regime'],
            cash=current_cash,
            market_data=market_data,
        )

        # Step 5: Generate orders
        orders = self._generate_orders(
            target_portfolio=target_portfolio,
            current_positions=current_positions,
            market_data=market_data,
        )

        # Calculate expected metrics
        expected_return = self._calculate_expected_return(orders, target_portfolio)

        cycle_time = (datetime.now() - cycle_start).total_seconds()

        return {
            'timestamp': cycle_start,
            'regime': regime_state,
            'signals_raw': len(raw_signals),
            'signals_filtered': len(filtered_signals),
            'target_positions': len(target_portfolio),
            'orders': orders,
            'expected_daily_return': expected_return,
            'expected_annual_return': expected_return * 252,
            'leverage': self._calculate_leverage(target_portfolio, current_cash),
            'cycle_time_seconds': cycle_time,
            'diagnostics': {
                'regime_engine': regime_state,
                'top_signals': filtered_signals[:5] if filtered_signals else [],
            }
        }

    def _generate_orders(
        self,
        target_portfolio: Dict[str, Dict],
        current_positions: Dict[str, Position],
        market_data: pd.DataFrame,
    ) -> List[Dict]:
        """Generate orders to move from current to target portfolio."""
        orders = []

        # Close positions not in target
        for symbol, position in current_positions.items():
            if symbol not in target_portfolio:
                orders.append({
                    'symbol': symbol,
                    'action': 'close',
                    'side': 'sell' if position.side == 'long' else 'buy',
                    'shares': position.shares,
                    'reason': 'not_in_target',
                })

        # Open/adjust positions in target
        for symbol, target in target_portfolio.items():
            current = current_positions.get(symbol)

            if current is None:
                # New position
                orders.append({
                    'symbol': symbol,
                    'action': 'open',
                    'side': target['side'],
                    'shares': target['shares'],
                    'entry_price': target['entry_price'],
                    'stop_loss': target['stop_loss'],
                    'take_profit': target['take_profit'],
                    'strategy': target['strategy'],
                    'confidence': target['confidence'],
                })
            elif current.shares != target['shares']:
                # Adjust position
                diff = target['shares'] - current.shares
                orders.append({
                    'symbol': symbol,
                    'action': 'adjust',
                    'side': 'buy' if diff > 0 else 'sell',
                    'shares': abs(diff),
                    'reason': 'rebalance',
                })

        return orders

    def _calculate_expected_return(
        self,
        orders: List[Dict],
        target_portfolio: Dict[str, Dict],
    ) -> float:
        """Calculate expected daily return based on signals."""
        if not target_portfolio:
            return 0.0

        total_edge = 0.0
        total_weight = 0.0

        for symbol, target in target_portfolio.items():
            edge = target.get('expected_edge', 0.005)
            weight = target.get('weight', 1.0 / len(target_portfolio))
            total_edge += edge * weight
            total_weight += weight

        if total_weight > 0:
            return total_edge / total_weight * self.config.target_leverage
        return 0.0

    def _calculate_leverage(
        self,
        target_portfolio: Dict[str, Dict],
        cash: float,
    ) -> float:
        """Calculate portfolio leverage."""
        if cash <= 0:
            return 0.0

        total_exposure = sum(
            t.get('shares', 0) * t.get('entry_price', 0)
            for t in target_portfolio.values()
        )

        return total_exposure / cash


def calculate_path_to_66_percent() -> Dict[str, Any]:
    """
    Calculate how to achieve 66% annual returns like Renaissance.

    Renaissance Formula:
    - Base edge: ~0.5% per trade (50.75% win rate)
    - Trades: 150,000-300,000 per day
    - Leverage: 12.5-20x
    - Diversification: 3,500+ positions

    Kobe Formula (scaled down):
    - Base edge: ~1% per trade (61% win rate - BETTER than Renaissance!)
    - Trades: 20-50 per day (scaled to our universe)
    - Leverage: 5-10x (conservative)
    - Diversification: 20-50 positions
    """

    scenarios = []

    # Current Kobe (no leverage, 2 positions)
    scenarios.append({
        'name': 'Current Kobe',
        'win_rate': 0.61,
        'avg_win': 0.02,  # 2% average win
        'avg_loss': 0.01,  # 1% average loss (2:1 R:R)
        'trades_per_day': 2,
        'leverage': 1.0,
        'positions': 2,
        'annual_return': None,  # Calculate
    })

    # Conservative Medallion
    scenarios.append({
        'name': 'Conservative Medallion',
        'win_rate': 0.55,
        'avg_win': 0.015,
        'avg_loss': 0.01,
        'trades_per_day': 10,
        'leverage': 3.0,
        'positions': 20,
        'annual_return': None,
    })

    # Moderate Medallion
    scenarios.append({
        'name': 'Moderate Medallion',
        'win_rate': 0.55,
        'avg_win': 0.015,
        'avg_loss': 0.01,
        'trades_per_day': 20,
        'leverage': 5.0,
        'positions': 30,
        'annual_return': None,
    })

    # Aggressive Medallion
    scenarios.append({
        'name': 'Aggressive Medallion',
        'win_rate': 0.52,
        'avg_win': 0.012,
        'avg_loss': 0.01,
        'trades_per_day': 50,
        'leverage': 10.0,
        'positions': 50,
        'annual_return': None,
    })

    # Full Medallion (theoretical)
    scenarios.append({
        'name': 'Full Medallion (Theoretical)',
        'win_rate': 0.5075,
        'avg_win': 0.01,
        'avg_loss': 0.01,
        'trades_per_day': 1000,
        'leverage': 12.5,
        'positions': 100,
        'annual_return': None,
    })

    # Calculate returns
    for scenario in scenarios:
        # Expected value per trade
        ev_per_trade = (
            scenario['win_rate'] * scenario['avg_win'] -
            (1 - scenario['win_rate']) * scenario['avg_loss']
        )

        # Daily return (before leverage)
        daily_return = ev_per_trade * scenario['trades_per_day'] / scenario['positions']

        # Apply leverage
        daily_return_leveraged = daily_return * scenario['leverage']

        # Annual return (252 trading days)
        annual_return = (1 + daily_return_leveraged) ** 252 - 1

        scenario['ev_per_trade'] = ev_per_trade
        scenario['daily_return'] = daily_return
        scenario['daily_return_leveraged'] = daily_return_leveraged
        scenario['annual_return'] = annual_return
        scenario['sharpe_estimate'] = annual_return / (daily_return_leveraged * np.sqrt(252) * 2)

    return {
        'scenarios': scenarios,
        'recommendation': """
To achieve 66% annual returns:

1. INCREASE POSITIONS: 2 -> 20-50 positions
   - Reduces idiosyncratic risk
   - Enables safe use of leverage

2. USE LEVERAGE: 1x -> 5-10x
   - ONLY after diversification
   - With strict risk limits

3. INCREASE TURNOVER: 2 -> 20-50 trades/day
   - More opportunities = more edge capture
   - Shorter holding periods

4. MAINTAIN EDGE: Keep 55%+ win rate
   - Your 61% is BETTER than Renaissance's 50.75%
   - Don't sacrifice accuracy for volume

MATH:
- 55% win rate, 1.5:1 R:R = 0.325% edge per trade
- 20 trades/day, 20 positions = 0.325% daily return
- 5x leverage = 1.625% daily return
- 252 days = (1.01625)^252 - 1 = 56.7% annual return

With 10x leverage = 98.5% annual return (exceeds 66%)
""",
    }


# Quick test
if __name__ == '__main__':
    results = calculate_path_to_66_percent()

    print("\n" + "="*60)
    print("PATH TO 66% ANNUAL RETURNS (Renaissance-Style)")
    print("="*60)

    for scenario in results['scenarios']:
        print(f"\n{scenario['name']}:")
        print(f"  Win Rate: {scenario['win_rate']:.2%}")
        print(f"  Trades/Day: {scenario['trades_per_day']}")
        print(f"  Leverage: {scenario['leverage']}x")
        print(f"  Positions: {scenario['positions']}")
        print(f"  EV/Trade: {scenario['ev_per_trade']:.4%}")
        print(f"  Daily Return: {scenario['daily_return_leveraged']:.4%}")
        print(f"  ANNUAL RETURN: {scenario['annual_return']:.1%}")

    print("\n" + results['recommendation'])
