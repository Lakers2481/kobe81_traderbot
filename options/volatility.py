"""
Realized Volatility Estimator.

Provides historical/realized volatility calculation for synthetic options pricing.
When no implied volatility is available (free data), we use realized vol as proxy.

Methods:
- Standard close-to-close volatility (default)
- Parkinson (high-low) volatility
- Garman-Klass (OHLC) volatility
- Yang-Zhang volatility (handles overnight gaps)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import pandas as pd


class VolatilityMethod(Enum):
    """Volatility estimation method."""
    CLOSE_TO_CLOSE = "close_to_close"
    PARKINSON = "parkinson"
    GARMAN_KLASS = "garman_klass"
    YANG_ZHANG = "yang_zhang"


@dataclass
class VolatilityResult:
    """Volatility calculation result."""
    method: VolatilityMethod
    volatility: float  # Annualized (decimal, e.g., 0.25 = 25%)
    lookback_days: int
    observations: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    def as_percent(self) -> float:
        """Return volatility as percentage."""
        return self.volatility * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "method": self.method.value,
            "volatility": round(self.volatility, 4),
            "volatility_pct": round(self.as_percent(), 2),
            "lookback_days": self.lookback_days,
            "observations": self.observations,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }


class RealizedVolatility:
    """
    Historical/Realized volatility estimator.

    For synthetic options pricing when IV is not available (free data sources).
    """

    TRADING_DAYS_PER_YEAR = 252

    def __init__(self, annualization_factor: int = 252):
        """
        Initialize volatility estimator.

        Args:
            annualization_factor: Trading days per year (default 252)
        """
        self.annualization_factor = annualization_factor

    def close_to_close(
        self,
        prices: Union[pd.Series, np.ndarray, List[float]],
        lookback: int = 20,
    ) -> VolatilityResult:
        """
        Standard close-to-close volatility (most common method).

        Args:
            prices: Close prices
            lookback: Number of periods (default 20 trading days)

        Returns:
            VolatilityResult with annualized volatility
        """
        if isinstance(prices, list):
            prices = np.array(prices)
        elif isinstance(prices, pd.Series):
            prices = prices.values

        if len(prices) < lookback + 1:
            lookback = len(prices) - 1

        if lookback < 2:
            return VolatilityResult(
                method=VolatilityMethod.CLOSE_TO_CLOSE,
                volatility=0.0,
                lookback_days=lookback,
                observations=len(prices),
            )

        # Use last 'lookback' prices
        recent_prices = prices[-(lookback + 1):]

        # Log returns
        log_returns = np.diff(np.log(recent_prices))

        # Standard deviation of returns
        std = np.std(log_returns, ddof=1)

        # Annualize
        annualized_vol = std * math.sqrt(self.annualization_factor)

        return VolatilityResult(
            method=VolatilityMethod.CLOSE_TO_CLOSE,
            volatility=float(annualized_vol),
            lookback_days=lookback,
            observations=len(log_returns),
        )

    def parkinson(
        self,
        highs: Union[pd.Series, np.ndarray, List[float]],
        lows: Union[pd.Series, np.ndarray, List[float]],
        lookback: int = 20,
    ) -> VolatilityResult:
        """
        Parkinson volatility (uses high-low range).

        More efficient than close-to-close (uses intraday info).

        Args:
            highs: High prices
            lows: Low prices
            lookback: Number of periods

        Returns:
            VolatilityResult with annualized volatility
        """
        if isinstance(highs, list):
            highs = np.array(highs)
        if isinstance(lows, list):
            lows = np.array(lows)
        if isinstance(highs, pd.Series):
            highs = highs.values
        if isinstance(lows, pd.Series):
            lows = lows.values

        n = min(len(highs), len(lows), lookback)
        if n < 2:
            return VolatilityResult(
                method=VolatilityMethod.PARKINSON,
                volatility=0.0,
                lookback_days=lookback,
                observations=n,
            )

        # Use last n bars
        h = highs[-n:]
        l = lows[-n:]

        # Parkinson formula
        log_hl = np.log(h / l) ** 2
        variance = log_hl.mean() / (4 * math.log(2))

        # Annualize
        annualized_vol = math.sqrt(variance * self.annualization_factor)

        return VolatilityResult(
            method=VolatilityMethod.PARKINSON,
            volatility=float(annualized_vol),
            lookback_days=n,
            observations=n,
        )

    def garman_klass(
        self,
        opens: Union[pd.Series, np.ndarray, List[float]],
        highs: Union[pd.Series, np.ndarray, List[float]],
        lows: Union[pd.Series, np.ndarray, List[float]],
        closes: Union[pd.Series, np.ndarray, List[float]],
        lookback: int = 20,
    ) -> VolatilityResult:
        """
        Garman-Klass volatility (uses OHLC data).

        More efficient than Parkinson, handles opening gaps.

        Args:
            opens, highs, lows, closes: OHLC prices
            lookback: Number of periods

        Returns:
            VolatilityResult with annualized volatility
        """
        # Convert to numpy arrays
        for arr_name in ['opens', 'highs', 'lows', 'closes']:
            arr = locals()[arr_name]
            if isinstance(arr, list):
                locals()[arr_name] = np.array(arr)
            elif isinstance(arr, pd.Series):
                locals()[arr_name] = arr.values

        opens = np.array(opens) if isinstance(opens, list) else (opens.values if isinstance(opens, pd.Series) else opens)
        highs = np.array(highs) if isinstance(highs, list) else (highs.values if isinstance(highs, pd.Series) else highs)
        lows = np.array(lows) if isinstance(lows, list) else (lows.values if isinstance(lows, pd.Series) else lows)
        closes = np.array(closes) if isinstance(closes, list) else (closes.values if isinstance(closes, pd.Series) else closes)

        n = min(len(opens), len(highs), len(lows), len(closes), lookback)
        if n < 2:
            return VolatilityResult(
                method=VolatilityMethod.GARMAN_KLASS,
                volatility=0.0,
                lookback_days=lookback,
                observations=n,
            )

        # Use last n bars
        o = opens[-n:]
        h = highs[-n:]
        l = lows[-n:]
        c = closes[-n:]

        # Garman-Klass formula
        log_hl_sq = np.log(h / l) ** 2
        log_co_sq = np.log(c / o) ** 2

        variance = 0.5 * log_hl_sq.mean() - (2 * math.log(2) - 1) * log_co_sq.mean()

        # Handle negative variance (data issues)
        if variance < 0:
            variance = log_hl_sq.mean() / (4 * math.log(2))

        # Annualize
        annualized_vol = math.sqrt(variance * self.annualization_factor)

        return VolatilityResult(
            method=VolatilityMethod.GARMAN_KLASS,
            volatility=float(annualized_vol),
            lookback_days=n,
            observations=n,
        )

    def yang_zhang(
        self,
        opens: Union[pd.Series, np.ndarray, List[float]],
        highs: Union[pd.Series, np.ndarray, List[float]],
        lows: Union[pd.Series, np.ndarray, List[float]],
        closes: Union[pd.Series, np.ndarray, List[float]],
        lookback: int = 20,
    ) -> VolatilityResult:
        """
        Yang-Zhang volatility (best for overnight gaps).

        Combines overnight, open-close, and Rogers-Satchell volatility.
        Most accurate for assets with significant overnight moves.

        Args:
            opens, highs, lows, closes: OHLC prices
            lookback: Number of periods

        Returns:
            VolatilityResult with annualized volatility
        """
        # Convert to numpy arrays
        opens = np.array(opens) if isinstance(opens, list) else (opens.values if isinstance(opens, pd.Series) else opens)
        highs = np.array(highs) if isinstance(highs, list) else (highs.values if isinstance(highs, pd.Series) else highs)
        lows = np.array(lows) if isinstance(lows, list) else (lows.values if isinstance(lows, pd.Series) else lows)
        closes = np.array(closes) if isinstance(closes, list) else (closes.values if isinstance(closes, pd.Series) else closes)

        n = min(len(opens), len(highs), len(lows), len(closes), lookback)
        if n < 3:
            return VolatilityResult(
                method=VolatilityMethod.YANG_ZHANG,
                volatility=0.0,
                lookback_days=lookback,
                observations=n,
            )

        # Use last n+1 bars (need prev close)
        o = opens[-(n):]
        h = highs[-(n):]
        l = lows[-(n):]
        c = closes[-(n):]
        c_prev = closes[-(n+1):-1]

        # Overnight returns (open vs prev close)
        log_oc = np.log(o / c_prev)
        overnight_var = np.var(log_oc, ddof=1)

        # Open-to-close returns
        log_co = np.log(c / o)
        open_close_var = np.var(log_co, ddof=1)

        # Rogers-Satchell component
        log_ho = np.log(h / o)
        log_hc = np.log(h / c)
        log_lo = np.log(l / o)
        log_lc = np.log(l / c)
        rs_var = (log_ho * log_hc + log_lo * log_lc).mean()

        # Yang-Zhang combination
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        variance = overnight_var + k * open_close_var + (1 - k) * rs_var

        # Handle negative variance
        if variance < 0:
            variance = abs(overnight_var + open_close_var + rs_var) / 3

        # Annualize
        annualized_vol = math.sqrt(variance * self.annualization_factor)

        return VolatilityResult(
            method=VolatilityMethod.YANG_ZHANG,
            volatility=float(annualized_vol),
            lookback_days=n,
            observations=n,
        )

    def calculate(
        self,
        df: pd.DataFrame,
        method: VolatilityMethod = VolatilityMethod.CLOSE_TO_CLOSE,
        lookback: int = 20,
        close_col: str = 'close',
        open_col: str = 'open',
        high_col: str = 'high',
        low_col: str = 'low',
    ) -> VolatilityResult:
        """
        Calculate volatility from a DataFrame.

        Args:
            df: DataFrame with OHLC data
            method: Volatility calculation method
            lookback: Number of periods
            close_col, open_col, high_col, low_col: Column names

        Returns:
            VolatilityResult with annualized volatility
        """
        if method == VolatilityMethod.CLOSE_TO_CLOSE:
            return self.close_to_close(df[close_col], lookback)

        elif method == VolatilityMethod.PARKINSON:
            return self.parkinson(df[high_col], df[low_col], lookback)

        elif method == VolatilityMethod.GARMAN_KLASS:
            return self.garman_klass(
                df[open_col], df[high_col], df[low_col], df[close_col], lookback
            )

        elif method == VolatilityMethod.YANG_ZHANG:
            return self.yang_zhang(
                df[open_col], df[high_col], df[low_col], df[close_col], lookback
            )

        else:
            raise ValueError(f"Unknown method: {method}")

    def rolling(
        self,
        df: pd.DataFrame,
        method: VolatilityMethod = VolatilityMethod.CLOSE_TO_CLOSE,
        lookback: int = 20,
        close_col: str = 'close',
    ) -> pd.Series:
        """
        Calculate rolling volatility (close-to-close only for speed).

        Args:
            df: DataFrame with close prices
            method: Currently only CLOSE_TO_CLOSE supported for rolling
            lookback: Rolling window size
            close_col: Close price column name

        Returns:
            Series of annualized volatilities
        """
        if method != VolatilityMethod.CLOSE_TO_CLOSE:
            raise ValueError("Rolling volatility only supports CLOSE_TO_CLOSE method")

        log_returns = np.log(df[close_col] / df[close_col].shift(1))
        rolling_std = log_returns.rolling(window=lookback).std()
        annualized = rolling_std * math.sqrt(self.annualization_factor)

        return annualized


# Convenience functions
_rv = RealizedVolatility()


def realized_vol(
    closes: Union[pd.Series, List[float]],
    lookback: int = 20,
) -> float:
    """
    Calculate realized volatility (annualized).

    Args:
        closes: Close prices
        lookback: Number of trading days (default 20)

    Returns:
        Annualized volatility as decimal (0.25 = 25%)
    """
    result = _rv.close_to_close(closes, lookback)
    return result.volatility


def realized_vol_ohlc(
    df: pd.DataFrame,
    method: str = 'yang_zhang',
    lookback: int = 20,
) -> float:
    """
    Calculate realized volatility using OHLC data.

    Args:
        df: DataFrame with open, high, low, close columns
        method: 'close_to_close', 'parkinson', 'garman_klass', 'yang_zhang'
        lookback: Number of trading days

    Returns:
        Annualized volatility as decimal
    """
    method_map = {
        'close_to_close': VolatilityMethod.CLOSE_TO_CLOSE,
        'parkinson': VolatilityMethod.PARKINSON,
        'garman_klass': VolatilityMethod.GARMAN_KLASS,
        'yang_zhang': VolatilityMethod.YANG_ZHANG,
    }

    vol_method = method_map.get(method.lower(), VolatilityMethod.CLOSE_TO_CLOSE)
    result = _rv.calculate(df, vol_method, lookback)
    return result.volatility


def vol_with_floor(
    closes: Union[pd.Series, List[float]],
    lookback: int = 20,
    floor: float = 0.10,
    cap: float = 2.0,
) -> float:
    """
    Calculate realized volatility with floor and cap.

    For synthetic options, we need reasonable vol bounds:
    - Floor prevents unrealistic low premiums
    - Cap prevents unrealistic high premiums

    Args:
        closes: Close prices
        lookback: Number of trading days
        floor: Minimum volatility (default 10%)
        cap: Maximum volatility (default 200%)

    Returns:
        Bounded annualized volatility
    """
    vol = realized_vol(closes, lookback)
    return max(floor, min(cap, vol))
