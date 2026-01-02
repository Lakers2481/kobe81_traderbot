"""
Kobe Trading System - Options Module.

Provides complete options infrastructure for backtesting and live trading:
- Black-Scholes pricing model with Greeks
- Realized volatility estimation (no IV needed)
- Delta-targeted strike selection via binary search
- Position sizing with 2% risk enforcement
- Options backtester with daily repricing

Live Trading Components:
- Chain fetcher for real options data (Polygon.io)
- Multi-leg spread strategies (verticals, iron condors, straddles)
- Options order router with broker integration
"""

from .black_scholes import (
    BlackScholes,
    OptionType,
    OptionPricing,
    calculate_option_price,
    calculate_greeks,
    calculate_iv,
)

from .volatility import (
    RealizedVolatility,
    VolatilityMethod,
    VolatilityResult,
    realized_vol,
    realized_vol_ohlc,
    vol_with_floor,
)

from .selection import (
    StrikeSelector,
    StrikeSelection,
    select_call_strike,
    select_put_strike,
    select_atm,
)

from .position_sizing import (
    OptionsPositionSizer,
    PositionDirection,
    PositionSizeResult,
    size_long_call,
    size_long_put,
    size_cash_secured_put,
    size_covered_call,
    calculate_max_contracts,
)

from .backtest import (
    SyntheticOptionsBacktester,
    OptionsBacktestResult,
    OptionsTradeRecord,
    OptionsPosition,
    TradeAction,
    run_options_backtest,
)

# Live trading components
from .chain_fetcher import (
    ChainFetcher,
    OptionsChain,
    OptionContract,
    OptionType as LiveOptionType,
    get_chain_fetcher,
)

from .spreads import (
    SpreadBuilder,
    OptionsSpread,
    SpreadLeg,
    SpreadType,
    get_spread_builder,
)

from .order_router import (
    OptionsOrderRouter,
    OptionsOrder,
    OptionsOrderLeg,
    OptionsOrderResult,
    OptionsOrderType,
    OptionsOrderSide,
    OptionsOrderStatus,
    get_options_router,
    quick_buy_call,
    quick_buy_put,
)

__all__ = [
    # Black-Scholes
    'BlackScholes',
    'OptionType',
    'OptionPricing',
    'calculate_option_price',
    'calculate_greeks',
    'calculate_iv',
    # Volatility
    'RealizedVolatility',
    'VolatilityMethod',
    'VolatilityResult',
    'realized_vol',
    'realized_vol_ohlc',
    'vol_with_floor',
    # Strike Selection
    'StrikeSelector',
    'StrikeSelection',
    'select_call_strike',
    'select_put_strike',
    'select_atm',
    # Position Sizing
    'OptionsPositionSizer',
    'PositionDirection',
    'PositionSizeResult',
    'size_long_call',
    'size_long_put',
    'size_cash_secured_put',
    'size_covered_call',
    'calculate_max_contracts',
    # Backtester
    'SyntheticOptionsBacktester',
    'OptionsBacktestResult',
    'OptionsTradeRecord',
    'OptionsPosition',
    'TradeAction',
    'run_options_backtest',
    # Live Chain Fetching
    'ChainFetcher',
    'OptionsChain',
    'OptionContract',
    'LiveOptionType',
    'get_chain_fetcher',
    # Spreads
    'SpreadBuilder',
    'OptionsSpread',
    'SpreadLeg',
    'SpreadType',
    'get_spread_builder',
    # Order Router
    'OptionsOrderRouter',
    'OptionsOrder',
    'OptionsOrderLeg',
    'OptionsOrderResult',
    'OptionsOrderType',
    'OptionsOrderSide',
    'OptionsOrderStatus',
    'get_options_router',
    'quick_buy_call',
    'quick_buy_put',
]
