"""
Kobe Trading System - Synthetic Options Module.

Provides complete synthetic options infrastructure for backtesting:
- Black-Scholes pricing model with Greeks
- Realized volatility estimation (no IV needed)
- Delta-targeted strike selection via binary search
- Position sizing with 2% risk enforcement
- Options backtester with daily repricing

All pricing is synthetic (no real options chains required).
Uses realized volatility when IV is unavailable (free data sources).
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
]
