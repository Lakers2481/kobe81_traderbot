"""
Synthetic Options Backtester.

Daily repricing of options positions using Black-Scholes with:
- Transaction costs (bid-ask spread + commissions)
- Greek tracking (Delta, Gamma, Theta, Vega)
- P&L decomposition (Delta P&L, Theta decay, Vega P&L)
- Early exit and expiration handling

Designed for swing trading single-leg options with daily decision-making.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .black_scholes import BlackScholes, OptionType, OptionPricing
from .volatility import RealizedVolatility, VolatilityMethod
from .position_sizing import OptionsPositionSizer, PositionDirection, PositionSizeResult

logger = logging.getLogger(__name__)


class TradeAction(Enum):
    """Trade action types."""
    BUY_TO_OPEN = "BUY_TO_OPEN"
    SELL_TO_OPEN = "SELL_TO_OPEN"
    BUY_TO_CLOSE = "BUY_TO_CLOSE"
    SELL_TO_CLOSE = "SELL_TO_CLOSE"
    EXPIRE_WORTHLESS = "EXPIRE_WORTHLESS"
    EXERCISE = "EXERCISE"
    ASSIGNMENT = "ASSIGNMENT"


@dataclass
class OptionsTradeRecord:
    """Record of a single options trade."""
    timestamp: datetime
    symbol: str
    action: TradeAction
    option_type: OptionType
    direction: PositionDirection
    contracts: int
    strike: float
    expiry_date: datetime
    spot_price: float
    premium: float
    volatility: float
    transaction_cost: float
    greeks: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "symbol": self.symbol,
            "action": self.action.value,
            "option_type": self.option_type.value,
            "direction": self.direction.value,
            "contracts": self.contracts,
            "strike": self.strike,
            "expiry_date": self.expiry_date.isoformat() if isinstance(self.expiry_date, datetime) else str(self.expiry_date),
            "spot_price": round(self.spot_price, 2),
            "premium": round(self.premium, 4),
            "volatility": round(self.volatility, 4),
            "transaction_cost": round(self.transaction_cost, 2),
            "greeks": {k: round(v, 4) for k, v in self.greeks.items()},
            "reason": self.reason,
        }


@dataclass
class OptionsPosition:
    """Active options position."""
    symbol: str
    option_type: OptionType
    direction: PositionDirection
    contracts: int
    strike: float
    expiry_date: datetime
    entry_date: datetime
    entry_premium: float
    entry_spot: float
    entry_vol: float
    current_premium: float = 0.0
    current_spot: float = 0.0
    current_vol: float = 0.0
    days_held: int = 0
    unrealized_pnl: float = 0.0
    greeks: Dict[str, float] = field(default_factory=dict)

    @property
    def dte(self) -> int:
        """Days to expiry."""
        if isinstance(self.expiry_date, datetime):
            return max(0, (self.expiry_date - datetime.now()).days)
        return 0

    @property
    def notional_value(self) -> float:
        """Notional value of position."""
        return self.contracts * self.strike * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "option_type": self.option_type.value,
            "direction": self.direction.value,
            "contracts": self.contracts,
            "strike": self.strike,
            "expiry_date": self.expiry_date.isoformat() if isinstance(self.expiry_date, datetime) else str(self.expiry_date),
            "entry_date": self.entry_date.isoformat() if isinstance(self.entry_date, datetime) else str(self.entry_date),
            "entry_premium": round(self.entry_premium, 4),
            "entry_spot": round(self.entry_spot, 2),
            "current_premium": round(self.current_premium, 4),
            "current_spot": round(self.current_spot, 2),
            "days_held": self.days_held,
            "dte": self.dte,
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "greeks": {k: round(v, 4) for k, v in self.greeks.items()},
        }


@dataclass
class OptionsBacktestResult:
    """Results from options backtest."""
    # Summary metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_premium_collected: float = 0.0
    total_premium_paid: float = 0.0
    total_transaction_costs: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_days_held: float = 0.0
    avg_trade_pnl: float = 0.0

    # P&L decomposition
    delta_pnl: float = 0.0
    theta_pnl: float = 0.0
    vega_pnl: float = 0.0
    gamma_pnl: float = 0.0

    # Time series
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    trade_list: List[Dict] = field(default_factory=list)
    daily_pnl: List[Tuple[datetime, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "summary": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": round(self.win_rate, 4),
                "total_pnl": round(self.total_pnl, 2),
                "total_premium_collected": round(self.total_premium_collected, 2),
                "total_premium_paid": round(self.total_premium_paid, 2),
                "total_transaction_costs": round(self.total_transaction_costs, 2),
                "max_drawdown": round(self.max_drawdown, 4),
                "sharpe_ratio": round(self.sharpe_ratio, 4),
                "avg_days_held": round(self.avg_days_held, 1),
                "avg_trade_pnl": round(self.avg_trade_pnl, 2),
            },
            "pnl_decomposition": {
                "delta_pnl": round(self.delta_pnl, 2),
                "theta_pnl": round(self.theta_pnl, 2),
                "vega_pnl": round(self.vega_pnl, 2),
                "gamma_pnl": round(self.gamma_pnl, 2),
            },
            "equity_curve_length": len(self.equity_curve),
            "trade_count": len(self.trade_list),
        }


class SyntheticOptionsBacktester:
    """
    Backtester for synthetic options using Black-Scholes pricing.

    Features:
    - Daily repricing with realized volatility
    - Transaction costs (spread + commission)
    - Greek tracking and P&L decomposition
    - Position sizing with risk limits
    - Early exit and expiration handling
    """

    def __init__(
        self,
        initial_equity: float = 100_000,
        risk_pct: float = 0.02,
        commission_per_contract: float = 0.65,
        spread_pct: float = 0.02,  # 2% bid-ask spread
        risk_free_rate: float = 0.05,
        vol_lookback: int = 20,
        vol_method: str = 'yang_zhang',
        default_dte: int = 30,
        min_dte_to_hold: int = 5,
    ):
        """
        Initialize backtester.

        Args:
            initial_equity: Starting equity
            risk_pct: Max risk per trade (default 2%)
            commission_per_contract: Commission per contract
            spread_pct: Bid-ask spread as percentage
            risk_free_rate: Risk-free rate
            vol_lookback: Lookback for realized volatility
            vol_method: Volatility method ('close_to_close', 'yang_zhang', etc.)
            default_dte: Default days to expiry for new positions
            min_dte_to_hold: Close positions when DTE falls below this
        """
        self.initial_equity = initial_equity
        self.risk_pct = risk_pct
        self.commission_per_contract = commission_per_contract
        self.spread_pct = spread_pct
        self.risk_free_rate = risk_free_rate
        self.vol_lookback = vol_lookback
        self.vol_method = vol_method
        self.default_dte = default_dte
        self.min_dte_to_hold = min_dte_to_hold

        self.bs = BlackScholes()
        self.vol_calc = RealizedVolatility()
        self.sizer = OptionsPositionSizer(risk_pct=risk_pct)

        # State
        self.equity = initial_equity
        self.positions: Dict[str, OptionsPosition] = {}
        self.trade_history: List[OptionsTradeRecord] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        self.daily_pnl_history: List[Tuple[datetime, float]] = []

    def _get_volatility(self, df: pd.DataFrame, symbol: str) -> float:
        """Calculate realized volatility for a symbol."""
        sym_df = df[df['symbol'] == symbol].copy() if 'symbol' in df.columns else df.copy()

        if len(sym_df) < self.vol_lookback + 1:
            return 0.25  # Default 25% if insufficient data

        method_map = {
            'close_to_close': VolatilityMethod.CLOSE_TO_CLOSE,
            'yang_zhang': VolatilityMethod.YANG_ZHANG,
            'parkinson': VolatilityMethod.PARKINSON,
            'garman_klass': VolatilityMethod.GARMAN_KLASS,
        }

        vol_method = method_map.get(self.vol_method, VolatilityMethod.CLOSE_TO_CLOSE)

        result = self.vol_calc.calculate(
            sym_df.tail(self.vol_lookback + 5),
            method=vol_method,
            lookback=self.vol_lookback,
        )

        # Floor and cap volatility
        return max(0.10, min(1.50, result.volatility))

    def _calculate_transaction_cost(self, contracts: int, premium: float) -> float:
        """Calculate total transaction cost."""
        commission = self.commission_per_contract * contracts
        spread_cost = premium * self.spread_pct * contracts * 100
        return commission + spread_cost

    def _price_option(
        self,
        option_type: OptionType,
        spot: float,
        strike: float,
        dte: int,
        vol: float,
    ) -> OptionPricing:
        """Price an option using Black-Scholes."""
        time = max(dte, 1) / 365.0
        return self.bs.price_option(
            option_type, spot, strike, time,
            self.risk_free_rate, vol
        )

    def open_position(
        self,
        timestamp: datetime,
        symbol: str,
        option_type: OptionType,
        direction: PositionDirection,
        spot: float,
        strike: float,
        expiry_date: datetime,
        vol: float,
        reason: str = "",
    ) -> Optional[OptionsTradeRecord]:
        """
        Open a new options position.

        Args:
            timestamp: Trade timestamp
            symbol: Underlying symbol
            option_type: CALL or PUT
            direction: LONG or SHORT
            spot: Current stock price
            strike: Option strike price
            expiry_date: Expiration date
            vol: Volatility for pricing
            reason: Trade reason

        Returns:
            OptionsTradeRecord if successful, None otherwise
        """
        # Check if position already exists
        position_key = f"{symbol}_{option_type.value}_{strike}_{expiry_date.strftime('%Y%m%d')}"
        if position_key in self.positions:
            logger.warning(f"Position already exists: {position_key}")
            return None

        # Calculate DTE
        dte = (expiry_date - timestamp).days
        if dte < self.min_dte_to_hold:
            logger.warning(f"DTE {dte} below minimum {self.min_dte_to_hold}")
            return None

        # Price option
        pricing = self._price_option(option_type, spot, strike, dte, vol)

        # Size position
        if direction == PositionDirection.LONG:
            size_result = self.sizer.size_long_option(
                self.equity, option_type, pricing.price, strike, spot
            )
        else:
            if option_type == OptionType.PUT:
                size_result = self.sizer.size_short_put(
                    self.equity, pricing.price, strike, spot
                )
            else:
                # Short calls require underlying shares - reject for now
                logger.warning("Short calls not supported without underlying")
                return None

        if not size_result.is_valid:
            logger.warning(f"Position sizing failed: {size_result.rejection_reason}")
            return None

        contracts = size_result.contracts
        premium = pricing.price

        # Calculate transaction cost
        tx_cost = self._calculate_transaction_cost(contracts, premium)

        # Create trade record
        action = TradeAction.BUY_TO_OPEN if direction == PositionDirection.LONG else TradeAction.SELL_TO_OPEN

        trade = OptionsTradeRecord(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            option_type=option_type,
            direction=direction,
            contracts=contracts,
            strike=strike,
            expiry_date=expiry_date,
            spot_price=spot,
            premium=premium,
            volatility=vol,
            transaction_cost=tx_cost,
            greeks={
                'delta': pricing.delta,
                'gamma': pricing.gamma,
                'theta': pricing.theta,
                'vega': pricing.vega,
            },
            reason=reason,
        )

        # Update equity
        if direction == PositionDirection.LONG:
            # Pay premium
            cost = contracts * premium * 100 + tx_cost
            self.equity -= cost
        else:
            # Collect premium (minus collateral for shorts)
            credit = contracts * premium * 100 - tx_cost
            # For cash-secured puts, set aside collateral
            if option_type == OptionType.PUT:
                collateral = contracts * strike * 100
                self.equity -= (collateral - credit)
            else:
                self.equity += credit

        # Create position
        position = OptionsPosition(
            symbol=symbol,
            option_type=option_type,
            direction=direction,
            contracts=contracts,
            strike=strike,
            expiry_date=expiry_date,
            entry_date=timestamp,
            entry_premium=premium,
            entry_spot=spot,
            entry_vol=vol,
            current_premium=premium,
            current_spot=spot,
            current_vol=vol,
            greeks={
                'delta': pricing.delta,
                'gamma': pricing.gamma,
                'theta': pricing.theta,
                'vega': pricing.vega,
            },
        )

        self.positions[position_key] = position
        self.trade_history.append(trade)

        return trade

    def close_position(
        self,
        timestamp: datetime,
        position_key: str,
        spot: float,
        vol: float,
        reason: str = "",
    ) -> Optional[OptionsTradeRecord]:
        """
        Close an existing position.

        Args:
            timestamp: Trade timestamp
            position_key: Position identifier
            spot: Current stock price
            vol: Current volatility
            reason: Close reason

        Returns:
            OptionsTradeRecord if successful
        """
        if position_key not in self.positions:
            logger.warning(f"Position not found: {position_key}")
            return None

        position = self.positions[position_key]

        # Calculate DTE
        dte = (position.expiry_date - timestamp).days

        # Price option
        pricing = self._price_option(position.option_type, spot, position.strike, dte, vol)
        premium = pricing.price

        # Calculate transaction cost
        tx_cost = self._calculate_transaction_cost(position.contracts, premium)

        # Determine action
        if position.direction == PositionDirection.LONG:
            action = TradeAction.SELL_TO_CLOSE
        else:
            action = TradeAction.BUY_TO_CLOSE

        trade = OptionsTradeRecord(
            timestamp=timestamp,
            symbol=position.symbol,
            action=action,
            option_type=position.option_type,
            direction=position.direction,
            contracts=position.contracts,
            strike=position.strike,
            expiry_date=position.expiry_date,
            spot_price=spot,
            premium=premium,
            volatility=vol,
            transaction_cost=tx_cost,
            greeks={
                'delta': pricing.delta,
                'gamma': pricing.gamma,
                'theta': pricing.theta,
                'vega': pricing.vega,
            },
            reason=reason,
        )

        # Calculate P&L
        if position.direction == PositionDirection.LONG:
            # Long: profit if premium increased
            pnl = (premium - position.entry_premium) * position.contracts * 100 - tx_cost
            self.equity += position.contracts * premium * 100 - tx_cost
        else:
            # Short: profit if premium decreased
            pnl = (position.entry_premium - premium) * position.contracts * 100 - tx_cost
            # Return collateral for short puts
            if position.option_type == OptionType.PUT:
                collateral = position.contracts * position.strike * 100
                self.equity += collateral
            self.equity -= position.contracts * premium * 100 + tx_cost

        # Remove position
        del self.positions[position_key]
        self.trade_history.append(trade)

        return trade

    def update_positions(
        self,
        timestamp: datetime,
        df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Daily mark-to-market of all positions.

        Args:
            timestamp: Current date
            df: DataFrame with OHLCV data

        Returns:
            Dict of position unrealized P&L
        """
        pnl_by_position = {}
        total_theta = 0.0
        total_delta = 0.0

        positions_to_close = []

        for position_key, position in self.positions.items():
            # Get current spot price
            sym_df = df[df['symbol'] == position.symbol] if 'symbol' in df.columns else df
            if sym_df.empty:
                continue

            spot = sym_df['close'].iloc[-1]
            vol = self._get_volatility(df, position.symbol)

            # Calculate DTE
            dte = (position.expiry_date - timestamp).days

            # Check for expiration or near-expiration
            if dte <= 0:
                positions_to_close.append((position_key, spot, vol, "expired"))
                continue
            elif dte <= self.min_dte_to_hold:
                positions_to_close.append((position_key, spot, vol, "DTE limit reached"))
                continue

            # Reprice option
            pricing = self._price_option(position.option_type, spot, position.strike, dte, vol)

            # Calculate unrealized P&L
            if position.direction == PositionDirection.LONG:
                unrealized = (pricing.price - position.entry_premium) * position.contracts * 100
            else:
                unrealized = (position.entry_premium - pricing.price) * position.contracts * 100

            # Update position
            position.current_premium = pricing.price
            position.current_spot = spot
            position.current_vol = vol
            position.days_held = (timestamp - position.entry_date).days
            position.unrealized_pnl = unrealized
            position.greeks = {
                'delta': pricing.delta * position.contracts * 100,
                'gamma': pricing.gamma * position.contracts * 100,
                'theta': pricing.theta * position.contracts * 100,
                'vega': pricing.vega * position.contracts * 100,
            }

            pnl_by_position[position_key] = unrealized
            total_theta += position.greeks['theta']
            total_delta += position.greeks['delta']

        # Close expired or near-expiration positions
        for position_key, spot, vol, reason in positions_to_close:
            self.close_position(timestamp, position_key, spot, vol, reason)

        return pnl_by_position

    def run_backtest(
        self,
        df: pd.DataFrame,
        signals_df: pd.DataFrame,
    ) -> OptionsBacktestResult:
        """
        Run options backtest.

        Args:
            df: OHLCV DataFrame with columns: timestamp, symbol, open, high, low, close, volume
            signals_df: Signals DataFrame with columns:
                - timestamp: signal date
                - symbol: underlying symbol
                - side: 'long' or 'short'
                - option_type: 'CALL' or 'PUT'
                - delta: target delta (e.g., 0.30)
                - dte: days to expiry (optional, default uses self.default_dte)
                - reason: signal reason

        Returns:
            OptionsBacktestResult with performance metrics
        """
        # Reset state
        self.equity = self.initial_equity
        self.positions = {}
        self.trade_history = []
        self.equity_history = []
        self.daily_pnl_history = []

        # Get unique dates
        df = df.sort_values('timestamp').reset_index(drop=True)
        dates = df['timestamp'].dt.date.unique() if hasattr(df['timestamp'].iloc[0], 'date') else pd.to_datetime(df['timestamp']).dt.date.unique()

        prev_equity = self.initial_equity
        trade_pnls = []

        for date in dates:
            timestamp = pd.Timestamp(date)

            # Get data up to this date
            daily_df = df[pd.to_datetime(df['timestamp']).dt.date <= date]

            # Update existing positions (mark-to-market)
            self.update_positions(timestamp, daily_df)

            # Process signals for this date
            if signals_df is not None and len(signals_df) > 0:
                day_signals = signals_df[
                    pd.to_datetime(signals_df['timestamp']).dt.date == date
                ] if 'timestamp' in signals_df.columns else pd.DataFrame()

                for _, signal in day_signals.iterrows():
                    symbol = signal['symbol']

                    # Get current price
                    sym_df = daily_df[daily_df['symbol'] == symbol] if 'symbol' in daily_df.columns else daily_df
                    if sym_df.empty:
                        continue

                    spot = sym_df['close'].iloc[-1]
                    vol = self._get_volatility(daily_df, symbol)

                    # Parse signal
                    direction = PositionDirection.LONG if signal.get('side', 'long').lower() == 'long' else PositionDirection.SHORT
                    option_type = OptionType.CALL if signal.get('option_type', 'CALL').upper() == 'CALL' else OptionType.PUT
                    dte = int(signal.get('dte', self.default_dte))

                    expiry_date = timestamp + timedelta(days=dte)

                    # Calculate strike from delta target
                    target_delta = signal.get('delta', 0.50)
                    from .selection import StrikeSelector
                    selector = StrikeSelector(risk_free_rate=self.risk_free_rate)
                    strike_result = selector.find_strike_by_delta(
                        option_type, spot, target_delta, dte, vol
                    )
                    strike = strike_result.strike

                    # Open position
                    self.open_position(
                        timestamp=timestamp,
                        symbol=symbol,
                        option_type=option_type,
                        direction=direction,
                        spot=spot,
                        strike=strike,
                        expiry_date=expiry_date,
                        vol=vol,
                        reason=signal.get('reason', ''),
                    )

            # Calculate total equity (cash + positions value)
            total_equity = self.equity
            for position in self.positions.values():
                if position.direction == PositionDirection.LONG:
                    total_equity += position.current_premium * position.contracts * 100
                # Short positions: collateral already deducted

            # Record equity
            self.equity_history.append((timestamp, total_equity))

            # Record daily P&L
            daily_pnl = total_equity - prev_equity
            self.daily_pnl_history.append((timestamp, daily_pnl))
            prev_equity = total_equity

        # Calculate results
        result = self._calculate_results()

        return result

    def _calculate_results(self) -> OptionsBacktestResult:
        """Calculate backtest performance metrics."""
        result = OptionsBacktestResult()

        # Count trades (entries only)
        entry_trades = [t for t in self.trade_history if t.action in [TradeAction.BUY_TO_OPEN, TradeAction.SELL_TO_OPEN]]
        exit_trades = [t for t in self.trade_history if t.action in [TradeAction.BUY_TO_CLOSE, TradeAction.SELL_TO_CLOSE]]

        result.total_trades = len(entry_trades)

        # Calculate P&L per trade
        trade_pnls = []
        days_held_list = []

        for entry in entry_trades:
            # Find matching exit
            position_key = f"{entry.symbol}_{entry.option_type.value}_{entry.strike}_{entry.expiry_date.strftime('%Y%m%d')}"

            exit_trade = None
            for exit_t in exit_trades:
                if (exit_t.symbol == entry.symbol and
                    exit_t.option_type == entry.option_type and
                    exit_t.strike == entry.strike):
                    exit_trade = exit_t
                    break

            if exit_trade:
                if entry.direction == PositionDirection.LONG:
                    pnl = (exit_trade.premium - entry.premium) * entry.contracts * 100
                else:
                    pnl = (entry.premium - exit_trade.premium) * entry.contracts * 100

                pnl -= (entry.transaction_cost + exit_trade.transaction_cost)
                trade_pnls.append(pnl)

                days = (exit_trade.timestamp - entry.timestamp).days
                days_held_list.append(days)

        # Win/loss stats
        winning = [p for p in trade_pnls if p > 0]
        losing = [p for p in trade_pnls if p <= 0]

        result.winning_trades = len(winning)
        result.losing_trades = len(losing)
        result.win_rate = len(winning) / len(trade_pnls) if trade_pnls else 0.0
        result.total_pnl = sum(trade_pnls)
        result.avg_trade_pnl = np.mean(trade_pnls) if trade_pnls else 0.0
        result.avg_days_held = np.mean(days_held_list) if days_held_list else 0.0

        # Transaction costs
        result.total_transaction_costs = sum(t.transaction_cost for t in self.trade_history)

        # Premium stats
        result.total_premium_paid = sum(
            t.premium * t.contracts * 100
            for t in self.trade_history
            if t.action == TradeAction.BUY_TO_OPEN
        )
        result.total_premium_collected = sum(
            t.premium * t.contracts * 100
            for t in self.trade_history
            if t.action == TradeAction.SELL_TO_OPEN
        )

        # Equity curve metrics
        if self.equity_history:
            equities = [e[1] for e in self.equity_history]
            peak = equities[0]
            max_dd = 0
            for eq in equities:
                peak = max(peak, eq)
                dd = (peak - eq) / peak
                max_dd = max(max_dd, dd)
            result.max_drawdown = max_dd

            # Sharpe ratio
            if self.daily_pnl_history:
                daily_returns = [p[1] / self.initial_equity for p in self.daily_pnl_history]
                if np.std(daily_returns) > 0:
                    result.sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

        # Convert to serializable format
        result.equity_curve = self.equity_history
        result.trade_list = [t.to_dict() for t in self.trade_history]
        result.daily_pnl = self.daily_pnl_history

        return result


# Convenience function
def run_options_backtest(
    ohlcv_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    initial_equity: float = 100_000,
    risk_pct: float = 0.02,
    **kwargs,
) -> OptionsBacktestResult:
    """
    Run a synthetic options backtest.

    Args:
        ohlcv_df: OHLCV DataFrame
        signals_df: Signals DataFrame
        initial_equity: Starting capital
        risk_pct: Risk per trade
        **kwargs: Additional backtester parameters

    Returns:
        OptionsBacktestResult
    """
    backtester = SyntheticOptionsBacktester(
        initial_equity=initial_equity,
        risk_pct=risk_pct,
        **kwargs,
    )

    return backtester.run_backtest(ohlcv_df, signals_df)
