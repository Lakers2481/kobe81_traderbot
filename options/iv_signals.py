"""
Implied Volatility Signals and Flow Intelligence.

Provides IV-based trading signals and options flow analysis:
- IV percentile ranking (mean reversion signals)
- IV term structure (event risk detection)
- IV skew (crash protection pricing)
- Put/Call ratio (sentiment indicator)
- GEX (Gamma Exposure) for dealer positioning

Sources:
- CBOE Put/Call Ratio methodology
- SqueezeMetrics GEX formula
- Academic: Ni, Pearson, Poteshman (2008) "Informed Trading in Options Markets"
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Dict, Union
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.structured_log import jlog

# Import Black-Scholes from existing module
try:
    from options.black_scholes import (
        bs_call_price, bs_put_price,
        bs_delta, bs_gamma, bs_vega, bs_theta,
        implied_volatility
    )
    BS_AVAILABLE = True
except ImportError:
    BS_AVAILABLE = False
    jlog("black_scholes_not_available", level="WARNING")


@dataclass
class IVSignalConfig:
    """Configuration for IV-based signals."""
    # IV percentile
    iv_lookback_days: int = 252  # 1 year for percentile calculation
    iv_low_threshold: float = 20.0  # Below 20th percentile = cheap
    iv_high_threshold: float = 80.0  # Above 80th percentile = expensive

    # Term structure
    front_month_dte_max: int = 35  # Front month = < 35 DTE
    back_month_dte_min: int = 60  # Back month = > 60 DTE

    # Put/Call ratio
    pcr_oversold_threshold: float = 1.15  # High put buying = oversold
    pcr_overbought_threshold: float = 0.70  # High call buying = overbought

    # GEX
    gex_positive_threshold: float = 1e9  # $1B positive GEX = strong support
    gex_negative_threshold: float = -1e9  # $1B negative GEX = amplified moves


class IVPercentileCalculator:
    """
    Calculate IV percentile rankings for mean reversion signals.

    IV tends to mean-revert:
    - Low IV percentile (<20) = Buy volatility (IV likely to expand)
    - High IV percentile (>80) = Sell volatility (IV likely to contract)
    """

    def __init__(self, config: Optional[IVSignalConfig] = None):
        self.config = config or IVSignalConfig()

    def calculate_percentile(
        self,
        current_iv: float,
        historical_iv: Union[List[float], np.ndarray, pd.Series],
        lookback: Optional[int] = None
    ) -> float:
        """
        Calculate current IV's percentile rank vs history.

        Args:
            current_iv: Current implied volatility (annualized)
            historical_iv: Historical IV values
            lookback: Optional lookback period (defaults to config)

        Returns:
            Percentile rank (0-100)
        """
        lookback = lookback or self.config.iv_lookback_days

        if isinstance(historical_iv, pd.Series):
            hist = historical_iv.dropna().values
        else:
            hist = np.array(historical_iv)

        # Use most recent lookback period
        hist = hist[-lookback:] if len(hist) > lookback else hist

        if len(hist) == 0:
            return 50.0  # Default to middle

        percentile = (hist < current_iv).sum() / len(hist) * 100
        return float(percentile)

    def get_signal(self, iv_percentile: float) -> str:
        """
        Get trading signal from IV percentile.

        Returns:
            "BUY_VOL" if cheap, "SELL_VOL" if expensive, "NEUTRAL" otherwise
        """
        if iv_percentile < self.config.iv_low_threshold:
            return "BUY_VOL"  # IV likely to expand
        elif iv_percentile > self.config.iv_high_threshold:
            return "SELL_VOL"  # IV likely to contract
        return "NEUTRAL"

    def calculate_iv_rank(
        self,
        current_iv: float,
        historical_iv: Union[List[float], np.ndarray, pd.Series],
        lookback: Optional[int] = None
    ) -> float:
        """
        Calculate IV Rank (IVR) - linear version of percentile.

        Formula: IVR = (current_IV - min_IV) / (max_IV - min_IV) * 100

        Args:
            current_iv: Current implied volatility
            historical_iv: Historical IV values
            lookback: Optional lookback period

        Returns:
            IV Rank (0-100)
        """
        lookback = lookback or self.config.iv_lookback_days

        if isinstance(historical_iv, pd.Series):
            hist = historical_iv.dropna().values
        else:
            hist = np.array(historical_iv)

        hist = hist[-lookback:] if len(hist) > lookback else hist

        if len(hist) == 0:
            return 50.0

        min_iv = hist.min()
        max_iv = hist.max()

        if max_iv == min_iv:
            return 50.0

        iv_rank = (current_iv - min_iv) / (max_iv - min_iv) * 100
        return float(np.clip(iv_rank, 0, 100))


class TermStructureAnalyzer:
    """
    Analyze options term structure for event risk detection.

    Term structure signals:
    - Contango (front IV < back IV): Normal market, no near-term event
    - Backwardation (front IV > back IV): Near-term event risk priced in
    - Steep contango (>1.3 ratio): Sell front month, buy back month
    - Steep backwardation (<0.7 ratio): Buy front month, sell back month
    """

    def __init__(self, config: Optional[IVSignalConfig] = None):
        self.config = config or IVSignalConfig()

    def calculate_ratio(self, front_iv: float, back_iv: float) -> float:
        """
        Calculate term structure ratio.

        Args:
            front_iv: Front month IV (e.g., 30 DTE)
            back_iv: Back month IV (e.g., 90 DTE)

        Returns:
            Ratio (front / back)
        """
        if back_iv <= 0:
            return 1.0
        return front_iv / back_iv

    def get_signal(self, term_ratio: float) -> str:
        """
        Get trading signal from term structure.

        Returns:
            "SELL_FRONT" if expensive, "BUY_FRONT" if cheap, "NEUTRAL" otherwise
        """
        if term_ratio > 1.3:
            return "SELL_FRONT"  # Front month overpriced (event priced in)
        elif term_ratio < 0.7:
            return "BUY_FRONT"  # Front month underpriced
        return "NEUTRAL"

    def analyze_chain(
        self,
        options_chain: pd.DataFrame,
        spot_price: float
    ) -> Dict[str, float]:
        """
        Analyze term structure from full options chain.

        Args:
            options_chain: DataFrame with columns: strike, dte, iv, type
            spot_price: Current underlying price

        Returns:
            Dict with term structure metrics
        """
        if options_chain.empty:
            return {'term_ratio': 1.0, 'front_iv': 0.0, 'back_iv': 0.0}

        # Get ATM options (closest to spot)
        options_chain = options_chain.copy()
        options_chain['atm_dist'] = abs(options_chain['strike'] - spot_price)

        # Front month ATM
        front = options_chain[
            options_chain['dte'] <= self.config.front_month_dte_max
        ]
        if not front.empty:
            front_atm = front.loc[front['atm_dist'].idxmin()]
            front_iv = front_atm['iv']
        else:
            front_iv = 0.0

        # Back month ATM
        back = options_chain[
            options_chain['dte'] >= self.config.back_month_dte_min
        ]
        if not back.empty:
            back_atm = back.loc[back['atm_dist'].idxmin()]
            back_iv = back_atm['iv']
        else:
            back_iv = 0.0

        term_ratio = self.calculate_ratio(front_iv, back_iv)

        return {
            'term_ratio': term_ratio,
            'front_iv': front_iv,
            'back_iv': back_iv,
            'signal': self.get_signal(term_ratio)
        }


class SkewAnalyzer:
    """
    Analyze volatility skew for crash protection pricing.

    Skew measures the difference between OTM put IV and ATM IV:
    - High skew = crash protection expensive (fear)
    - Low skew = crash protection cheap (complacency)

    Mean reversion trading:
    - Extreme high skew (>90th percentile) = Sell OTM puts
    - Extreme low skew (<10th percentile) = Buy OTM puts
    """

    def __init__(self):
        pass

    def calculate_25d_skew(
        self,
        otm_put_iv: float,
        atm_iv: float
    ) -> float:
        """
        Calculate 25-delta skew.

        Args:
            otm_put_iv: IV of 25-delta put
            atm_iv: IV of ATM option

        Returns:
            Skew value (positive = puts expensive)
        """
        return otm_put_iv - atm_iv

    def calculate_skew_percentile(
        self,
        current_skew: float,
        historical_skew: Union[List[float], np.ndarray, pd.Series],
        lookback: int = 252
    ) -> float:
        """
        Calculate skew percentile vs history.

        Args:
            current_skew: Current skew value
            historical_skew: Historical skew values
            lookback: Lookback period

        Returns:
            Percentile (0-100)
        """
        if isinstance(historical_skew, pd.Series):
            hist = historical_skew.dropna().values
        else:
            hist = np.array(historical_skew)

        hist = hist[-lookback:] if len(hist) > lookback else hist

        if len(hist) == 0:
            return 50.0

        percentile = (hist < current_skew).sum() / len(hist) * 100
        return float(percentile)

    def get_signal(self, skew_percentile: float) -> str:
        """
        Get trading signal from skew percentile.

        Returns:
            "SELL_OTM_PUTS" if overpriced, "BUY_OTM_PUTS" if cheap, "NEUTRAL"
        """
        if skew_percentile > 90:
            return "SELL_OTM_PUTS"  # Crash protection overpriced
        elif skew_percentile < 10:
            return "BUY_OTM_PUTS"  # Crash protection cheap
        return "NEUTRAL"


class PutCallRatioAnalyzer:
    """
    Analyze Put/Call ratio as sentiment indicator.

    PCR interpretation (contrarian):
    - High PCR (>1.15) = Excessive fear, bullish signal
    - Low PCR (<0.70) = Excessive greed, bearish signal
    - Normal PCR (0.8-1.0) = Neutral

    Source: CBOE Equity Put/Call Ratio
    """

    def __init__(self, config: Optional[IVSignalConfig] = None):
        self.config = config or IVSignalConfig()

    def calculate_pcr(
        self,
        put_volume: int,
        call_volume: int
    ) -> float:
        """
        Calculate Put/Call ratio.

        Args:
            put_volume: Total put volume
            call_volume: Total call volume

        Returns:
            PCR ratio
        """
        if call_volume <= 0:
            return 1.0
        return put_volume / call_volume

    def calculate_weighted_pcr(
        self,
        options_chain: pd.DataFrame
    ) -> float:
        """
        Calculate volume-weighted PCR from options chain.

        Args:
            options_chain: DataFrame with columns: type, volume

        Returns:
            PCR ratio
        """
        puts = options_chain[options_chain['type'] == 'put']['volume'].sum()
        calls = options_chain[options_chain['type'] == 'call']['volume'].sum()
        return self.calculate_pcr(puts, calls)

    def get_signal(self, pcr: float) -> str:
        """
        Get sentiment signal from PCR.

        Contrarian interpretation.

        Returns:
            "BULLISH" if oversold, "BEARISH" if overbought, "NEUTRAL"
        """
        if pcr > self.config.pcr_oversold_threshold:
            return "BULLISH"  # Too much put buying = fear = contrarian bullish
        elif pcr < self.config.pcr_overbought_threshold:
            return "BEARISH"  # Too much call buying = greed = contrarian bearish
        return "NEUTRAL"


class GEXCalculator:
    """
    Gamma Exposure (GEX) Calculator.

    GEX measures net dealer gamma positioning:
    - Positive GEX = Dealers long gamma = Market stabilizing
    - Negative GEX = Dealers short gamma = Market destabilizing

    Dealer hedging creates mean reversion at high GEX levels
    and momentum at low/negative GEX levels.

    Source: SqueezeMetrics methodology
    """

    def __init__(self, config: Optional[IVSignalConfig] = None):
        self.config = config or IVSignalConfig()

    def calculate_option_gamma(
        self,
        spot: float,
        strike: float,
        dte: float,
        iv: float,
        risk_free: float = 0.05
    ) -> float:
        """
        Calculate Black-Scholes gamma for an option.

        Args:
            spot: Current underlying price
            strike: Option strike price
            dte: Days to expiration
            iv: Implied volatility (annualized, decimal)
            risk_free: Risk-free rate (annualized, decimal)

        Returns:
            Gamma value
        """
        if dte <= 0 or iv <= 0:
            return 0.0

        T = dte / 365.0
        sigma = iv

        try:
            d1 = (math.log(spot / strike) + (risk_free + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            gamma = math.exp(-d1 ** 2 / 2) / (spot * sigma * math.sqrt(2 * math.pi * T))
            return gamma
        except (ValueError, ZeroDivisionError):
            return 0.0

    def calculate_gex(
        self,
        options_chain: pd.DataFrame,
        spot_price: float
    ) -> float:
        """
        Calculate net Gamma Exposure from options chain.

        Formula: GEX = Σ(OI × gamma × 100 × spot) for all strikes
        - Calls: Dealers are short, so negative GEX contribution
        - Puts: Dealers are long, so positive GEX contribution

        Args:
            options_chain: DataFrame with columns: strike, dte, iv, type, oi
            spot_price: Current underlying price

        Returns:
            Net GEX in dollars
        """
        if options_chain.empty:
            return 0.0

        total_gex = 0.0

        for _, opt in options_chain.iterrows():
            try:
                strike = opt['strike']
                dte = opt['dte']
                iv = opt['iv']
                oi = opt.get('oi', opt.get('open_interest', 0))
                opt_type = opt['type'].lower()

                gamma = self.calculate_option_gamma(spot_price, strike, dte, iv)

                # GEX per contract = gamma × 100 (shares per contract) × spot
                gex_per_contract = gamma * 100 * spot_price

                # Dealer positioning (opposite of market)
                # Market buys calls → Dealer sells calls → Dealer short gamma
                # Market buys puts → Dealer buys puts → Dealer long gamma
                if opt_type == 'call':
                    total_gex -= oi * gex_per_contract  # Negative (dealers short)
                else:  # put
                    total_gex += oi * gex_per_contract  # Positive (dealers long)

            except Exception:
                continue

        return total_gex

    def get_signal(self, gex: float) -> str:
        """
        Get market structure signal from GEX.

        Args:
            gex: Net gamma exposure in dollars

        Returns:
            "MEAN_REVERSION" if high positive, "MOMENTUM" if negative, "NEUTRAL"
        """
        if gex > self.config.gex_positive_threshold:
            return "MEAN_REVERSION"  # Dealers hedge = stabilizing = fade moves
        elif gex < self.config.gex_negative_threshold:
            return "MOMENTUM"  # Dealers amplify = destabilizing = follow moves
        return "NEUTRAL"

    def find_gex_levels(
        self,
        options_chain: pd.DataFrame,
        spot_price: float,
        num_levels: int = 5
    ) -> Dict[str, List[float]]:
        """
        Find key GEX levels (support/resistance).

        Args:
            options_chain: DataFrame with options data
            spot_price: Current underlying price
            num_levels: Number of levels to return

        Returns:
            Dict with 'support' and 'resistance' levels
        """
        if options_chain.empty:
            return {'support': [], 'resistance': []}

        # Calculate GEX at each strike
        strikes = options_chain['strike'].unique()
        gex_by_strike = {}

        for strike in strikes:
            strike_opts = options_chain[options_chain['strike'] == strike]
            gex = 0.0

            for _, opt in strike_opts.iterrows():
                gamma = self.calculate_option_gamma(
                    spot_price, strike, opt['dte'], opt['iv']
                )
                oi = opt.get('oi', opt.get('open_interest', 0))
                gex_per = gamma * 100 * spot_price

                if opt['type'].lower() == 'call':
                    gex -= oi * gex_per
                else:
                    gex += oi * gex_per

            gex_by_strike[strike] = gex

        # Sort by absolute GEX
        sorted_strikes = sorted(gex_by_strike.items(), key=lambda x: abs(x[1]), reverse=True)
        top_strikes = sorted_strikes[:num_levels * 2]

        # Separate into support and resistance
        support = [s for s, g in top_strikes if s < spot_price and g > 0][:num_levels]
        resistance = [s for s, g in top_strikes if s > spot_price and g > 0][:num_levels]

        return {
            'support': sorted(support),
            'resistance': sorted(resistance)
        }


class MaxPainCalculator:
    """
    Calculate Max Pain strike.

    Max Pain = strike where option sellers keep most premium
    Theory: Stock tends to pin near max pain at expiration

    Used for:
    - Expiration week directional bias
    - Strike selection for credit spreads
    """

    def calculate_max_pain(
        self,
        options_chain: pd.DataFrame,
        expiration_date: Optional[str] = None
    ) -> float:
        """
        Calculate max pain strike.

        Args:
            options_chain: DataFrame with columns: strike, type, oi
            expiration_date: Optional filter for specific expiration

        Returns:
            Max pain strike price
        """
        if options_chain.empty:
            return 0.0

        if expiration_date:
            options_chain = options_chain[
                options_chain['expiration'] == expiration_date
            ]

        strikes = sorted(options_chain['strike'].unique())

        if not strikes:
            return 0.0

        min_pain = float('inf')
        max_pain_strike = strikes[len(strikes) // 2]  # Default to middle

        for test_strike in strikes:
            # Calculate total pain at this strike
            call_pain = 0
            put_pain = 0

            for _, opt in options_chain.iterrows():
                oi = opt.get('oi', opt.get('open_interest', 0))
                strike = opt['strike']

                if opt['type'].lower() == 'call':
                    # Calls are worthless below strike, ITM above
                    if test_strike > strike:
                        call_pain += (test_strike - strike) * oi * 100
                else:  # put
                    # Puts are worthless above strike, ITM below
                    if test_strike < strike:
                        put_pain += (strike - test_strike) * oi * 100

            total_pain = call_pain + put_pain

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike

        return max_pain_strike


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_iv_signal(
    current_iv: float,
    historical_iv: Union[List[float], np.ndarray, pd.Series],
    lookback: int = 252
) -> Dict[str, Union[str, float]]:
    """
    Get IV-based trading signal.

    Args:
        current_iv: Current implied volatility
        historical_iv: Historical IV values
        lookback: Lookback period for percentile

    Returns:
        Dict with signal and metrics
    """
    calc = IVPercentileCalculator()
    percentile = calc.calculate_percentile(current_iv, historical_iv, lookback)
    iv_rank = calc.calculate_iv_rank(current_iv, historical_iv, lookback)
    signal = calc.get_signal(percentile)

    return {
        'iv_percentile': percentile,
        'iv_rank': iv_rank,
        'signal': signal,
        'current_iv': current_iv
    }


def get_pcr_signal(put_volume: int, call_volume: int) -> Dict[str, Union[str, float]]:
    """
    Get Put/Call ratio sentiment signal.

    Args:
        put_volume: Total put volume
        call_volume: Total call volume

    Returns:
        Dict with signal and PCR value
    """
    analyzer = PutCallRatioAnalyzer()
    pcr = analyzer.calculate_pcr(put_volume, call_volume)
    signal = analyzer.get_signal(pcr)

    return {
        'pcr': pcr,
        'signal': signal
    }


def calculate_gex(
    options_chain: pd.DataFrame,
    spot_price: float
) -> Dict[str, Union[str, float, List[float]]]:
    """
    Calculate GEX and related metrics.

    Args:
        options_chain: DataFrame with options data
        spot_price: Current underlying price

    Returns:
        Dict with GEX value, signal, and key levels
    """
    calc = GEXCalculator()
    gex = calc.calculate_gex(options_chain, spot_price)
    signal = calc.get_signal(gex)
    levels = calc.find_gex_levels(options_chain, spot_price)

    return {
        'gex': gex,
        'gex_billions': gex / 1e9,
        'signal': signal,
        'support_levels': levels['support'],
        'resistance_levels': levels['resistance']
    }
