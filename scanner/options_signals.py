"""
Options Signal Generator for Scanner Integration.

Converts equity signals to options signals:
- For each equity signal, generates BOTH call AND put options
- Uses Black-Scholes pricing with realized volatility
- Delta-targeted strike selection (default 30-delta)

Output includes:
- CALL options (bullish plays)
- PUT options (bearish plays / hedges)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

# Options infrastructure imports
try:
    from options.selection import StrikeSelector, StrikeSelection
    from options.black_scholes import OptionType
    from options.volatility import RealizedVolatility, VolatilityMethod
    OPTIONS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Options modules not available: {e}")
    OPTIONS_AVAILABLE = False


@dataclass
class OptionsSignal:
    """Options trading signal."""
    # Underlying info
    symbol: str
    underlying_price: float

    # Option details
    option_type: str  # 'CALL' or 'PUT'
    strike: float
    expiration: str  # YYYY-MM-DD
    days_to_expiry: int

    # Pricing
    option_price: float
    delta: float
    implied_volatility: float

    # Trade details
    contract_symbol: str  # OCC symbol
    action: str  # 'BUY_TO_OPEN'
    quantity: int
    max_risk: float  # Maximum loss (premium paid)

    # Scoring
    conf_score: float
    strategy: str
    reason: str

    # Timestamps
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame."""
        return {
            'symbol': self.symbol,
            'underlying_price': self.underlying_price,
            'option_type': self.option_type,
            'strike': self.strike,
            'expiration': self.expiration,
            'days_to_expiry': self.days_to_expiry,
            'option_price': self.option_price,
            'delta': self.delta,
            'implied_volatility': self.implied_volatility,
            'contract_symbol': self.contract_symbol,
            'action': self.action,
            'quantity': self.quantity,
            'max_risk': self.max_risk,
            'conf_score': self.conf_score,
            'strategy': self.strategy,
            'reason': self.reason,
            'timestamp': self.timestamp,
            'asset_class': 'OPTIONS',
            'side': 'long',  # Buying options
            'entry_price': self.option_price,
            'stop_loss': 0.0,  # Options: max loss is premium
            'take_profit': self.option_price * 2.0,  # Target 100% gain
        }


class OptionsSignalGenerator:
    """
    Generate options signals from equity signals.

    For each equity signal, generates BOTH:
    - CALL option (for bullish plays)
    - PUT option (for bearish plays / hedges)
    """

    DEFAULT_TARGET_DELTA = 0.30  # 30-delta options
    DEFAULT_DTE = 21  # 3 weeks
    MIN_DTE = 7
    MAX_DTE = 45

    def __init__(
        self,
        target_delta: float = DEFAULT_TARGET_DELTA,
        target_dte: int = DEFAULT_DTE,
        max_premium_pct: float = 0.05,  # Max 5% of underlying price
    ):
        """
        Initialize options signal generator.

        Args:
            target_delta: Target delta for strike selection (default 0.30)
            target_dte: Target days to expiry (default 21)
            max_premium_pct: Maximum premium as % of underlying price
        """
        self.target_delta = target_delta
        self.target_dte = target_dte
        self.max_premium_pct = max_premium_pct

        if OPTIONS_AVAILABLE:
            self.strike_selector = StrikeSelector()
            self.vol_estimator = RealizedVolatility()
        else:
            self.strike_selector = None
            self.vol_estimator = None

    def generate_from_equity_signals(
        self,
        equity_signals: pd.DataFrame,
        price_data: pd.DataFrame,
        max_signals: int = 6,  # 3 calls + 3 puts
    ) -> pd.DataFrame:
        """
        Generate options signals from equity signals.

        For each equity signal, generates BOTH a CALL and a PUT option.

        Args:
            equity_signals: DataFrame of equity signals
            price_data: Historical OHLCV data for volatility calculation
            max_signals: Maximum total options signals (calls + puts)

        Returns:
            DataFrame of options signals (calls and puts)
        """
        if not OPTIONS_AVAILABLE:
            logger.warning("Options modules not available - returning empty DataFrame")
            return pd.DataFrame()

        if equity_signals.empty:
            return pd.DataFrame()

        options_signals = []
        signals_per_type = max_signals // 2  # Split between calls and puts

        for _, row in equity_signals.head(signals_per_type).iterrows():
            try:
                # Generate CALL option
                call_signal = self._generate_option(row, price_data, OptionType.CALL)
                if call_signal:
                    options_signals.append(call_signal.to_dict())

                # Generate PUT option
                put_signal = self._generate_option(row, price_data, OptionType.PUT)
                if put_signal:
                    options_signals.append(put_signal.to_dict())

            except Exception as e:
                logger.debug(f"Failed to generate options for {row.get('symbol', '?')}: {e}")
                continue

        if not options_signals:
            return pd.DataFrame()

        df = pd.DataFrame(options_signals)
        return df.head(max_signals)

    def _generate_option(
        self,
        equity_signal: pd.Series,
        price_data: pd.DataFrame,
        option_type: "OptionType",
    ) -> Optional[OptionsSignal]:
        """Generate a single option signal (call or put)."""
        symbol = str(equity_signal.get('symbol', ''))
        entry_price = float(equity_signal.get('entry_price', 0))
        parent_conf_score = float(equity_signal.get('conf_score', 0.5))
        strategy = str(equity_signal.get('strategy', 'unknown'))
        timestamp = equity_signal.get('timestamp', datetime.now().isoformat())

        if not symbol or entry_price <= 0:
            return None

        # Calculate realized volatility from price data
        vol = self._calculate_volatility(symbol, price_data)
        if vol <= 0:
            vol = 0.30  # Default 30% vol if calculation fails

        # Calculate expiration date
        expiration_date = self._get_expiration_date()
        dte = (expiration_date - date.today()).days

        # Select strike using delta targeting
        strike_result = self.strike_selector.find_strike_by_delta(
            option_type=option_type,
            spot=entry_price,
            target_delta=self.target_delta,
            days_to_expiry=dte,
            volatility=vol,
        )

        # Calculate option price
        option_price = strike_result.price

        # Validate premium isn't too expensive
        if option_price > entry_price * self.max_premium_pct:
            logger.debug(f"{symbol}: {option_type.value} premium too high ({option_price:.2f})")
            return None

        # Generate OCC contract symbol
        contract_symbol = self._generate_occ_symbol(
            symbol, expiration_date, option_type, strike_result.strike
        )

        type_str = option_type.value.upper()

        # === ADJUSTED CONF_SCORE FOR OPTIONS ===
        # Options have additional risks vs equity: theta decay, wider spreads, leverage
        # Adjustment ensures parent equity always ranks above its derivative options
        #
        # Base adjustment by option type:
        #   CALL: 0.92 multiplier (8% haircut - directional bet aligned with bullish)
        #   PUT:  0.88 multiplier (12% haircut - opposite direction or hedge)
        #
        # Delta adjustment (higher delta = more like stock = less reduction):
        #   Add back up to 0.03 for high-delta options (delta >= 0.50)
        #
        # DTE adjustment (shorter DTE = more theta risk = more reduction):
        #   Subtract up to 0.02 for short-dated options (DTE < 14 days)
        #
        base_multiplier = 0.92 if type_str == 'CALL' else 0.88
        actual_delta = abs(strike_result.delta)

        # Delta bonus: high-delta options are more like stock
        delta_bonus = min(0.03, max(0.0, (actual_delta - 0.30) * 0.10))

        # DTE penalty: short-dated options have more theta risk
        dte_penalty = 0.02 if dte < 14 else (0.01 if dte < 21 else 0.0)

        # Calculate adjusted conf_score
        adjusted_conf = parent_conf_score * base_multiplier + delta_bonus - dte_penalty
        adjusted_conf = max(0.10, min(0.99, adjusted_conf))  # Clamp to valid range

        # Build options signal
        return OptionsSignal(
            symbol=symbol,
            underlying_price=entry_price,
            option_type=type_str,
            strike=strike_result.strike,
            expiration=expiration_date.isoformat(),
            days_to_expiry=dte,
            option_price=round(option_price, 2),
            delta=round(abs(strike_result.delta), 4),
            implied_volatility=round(vol, 4),
            contract_symbol=contract_symbol,
            action='BUY_TO_OPEN',
            quantity=1,
            max_risk=round(option_price * 100, 2),  # Per contract
            conf_score=round(adjusted_conf, 4),  # Adjusted, not parent's raw score
            strategy=f"{strategy}_{type_str}",
            reason=f"BUY {type_str} @ {self.target_delta*100:.0f}Δ, ${strike_result.strike:.2f} strike",
            timestamp=str(timestamp),
        )

    def _calculate_volatility(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        lookback: int = 20,
    ) -> float:
        """Calculate realized volatility for a symbol."""
        try:
            sym_data = price_data[price_data['symbol'] == symbol].copy()
            if len(sym_data) < lookback + 1:
                return 0.30  # Default

            sym_data = sym_data.sort_values('timestamp')
            result = self.vol_estimator.calculate(
                sym_data.tail(lookback + 1),
                method=VolatilityMethod.YANG_ZHANG,
                lookback=lookback,
            )
            return max(0.10, min(2.0, result.volatility))  # Clamp 10%-200%
        except Exception:
            return 0.30

    def _get_expiration_date(self) -> date:
        """Get target expiration date (next Friday >= target DTE)."""
        today = date.today()
        target = today + timedelta(days=self.target_dte)

        # Find next Friday
        days_until_friday = (4 - target.weekday()) % 7
        if days_until_friday == 0 and target.weekday() != 4:
            days_until_friday = 7

        expiration = target + timedelta(days=days_until_friday)

        # Ensure minimum DTE
        if (expiration - today).days < self.MIN_DTE:
            expiration += timedelta(days=7)

        return expiration

    def _generate_occ_symbol(
        self,
        symbol: str,
        expiration: date,
        option_type: "OptionType",
        strike: float,
    ) -> str:
        """
        Generate OCC option symbol.

        Format: ROOT + YYMMDD + C/P + STRIKE (8 digits, 3 decimal places implied)
        Example: AAPL230120C00150000 = AAPL Jan 20, 2023 $150 Call
        """
        root = symbol.upper()[:6].ljust(6)
        date_str = expiration.strftime('%y%m%d')
        type_char = 'C' if option_type == OptionType.CALL else 'P'
        strike_int = int(strike * 1000)
        strike_str = f"{strike_int:08d}"

        return f"{root}{date_str}{type_char}{strike_str}"


# Convenience function for scanner integration
def generate_options_signals(
    equity_signals: pd.DataFrame,
    price_data: pd.DataFrame,
    max_signals: int = 6,
    target_delta: float = 0.30,
    target_dte: int = 21,
) -> pd.DataFrame:
    """
    Generate options signals (calls AND puts) from equity signals.

    This is the main entry point for scanner integration.

    Args:
        equity_signals: DataFrame of equity signals
        price_data: Historical OHLCV data
        max_signals: Maximum options signals to generate (calls + puts)
        target_delta: Target delta for strikes (default 0.30)
        target_dte: Target days to expiration (default 21)

    Returns:
        DataFrame with both CALL and PUT options for each equity signal
    """
    if not OPTIONS_AVAILABLE:
        return pd.DataFrame()

    generator = OptionsSignalGenerator(
        target_delta=target_delta,
        target_dte=target_dte,
    )
    return generator.generate_from_equity_signals(
        equity_signals, price_data, max_signals
    )
