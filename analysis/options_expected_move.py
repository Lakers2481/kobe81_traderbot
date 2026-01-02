"""
Options Expected Move Calculator
================================

Calculates the expected price range for a stock based on volatility.
Since we don't have live options data, we use realized volatility as a proxy.

Expected Move Formula:
    Expected Move = Price * Volatility * sqrt(Days/252)

For weekly expected move (5 trading days):
    Weekly EM = Price * Vol * sqrt(5/252) = Price * Vol * 0.141

Usage:
    from analysis.options_expected_move import ExpectedMoveCalculator

    calc = ExpectedMoveCalculator()
    em = calc.calculate_weekly_expected_move('AAPL', 175.50)
    print(f"Expected move: +/- {em.weekly_expected_move_pct:.1%}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ExpectedMove:
    """Options-implied expected move for the week."""
    symbol: str
    current_price: float
    weekly_expected_move_pct: float  # e.g., 0.045 (4.5%)
    weekly_expected_move_dollars: float  # e.g., $7.85
    upper_bound: float  # e.g., $183.35
    lower_bound: float  # e.g., $167.65
    week_open_price: float  # Monday's open price
    move_from_week_open_pct: float  # How much moved already this week
    remaining_room_up_pct: float  # Room to move up within expected range
    remaining_room_down_pct: float  # Room to move down within expected range
    remaining_room_direction: str  # "UP" | "DOWN" | "BOTH" | "EXHAUSTED"
    volatility_20d: float  # Annualized 20-day volatility
    volatility_percentile: float  # Where current vol sits vs historical (0-100)
    calculation_method: str  # "realized_vol" (no live IV available)
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExpectedMoveCalculator:
    """
    Calculate expected price ranges based on realized volatility.

    This provides options-style expected move without needing options data.
    """

    # Trading days in a year
    TRADING_DAYS_PER_YEAR = 252
    TRADING_DAYS_PER_WEEK = 5

    def __init__(self):
        """Initialize calculator."""
        self._data_provider = None
        self._vol_calculator = None

    @property
    def data_provider(self):
        """Lazy load data provider function."""
        if self._data_provider is None:
            try:
                from data.providers.polygon_eod import fetch_daily_bars_polygon
                self._data_provider = fetch_daily_bars_polygon
            except Exception as e:
                logger.warning(f"Could not load Polygon provider: {e}")
        return self._data_provider

    @property
    def vol_calculator(self):
        """Lazy load volatility calculator."""
        if self._vol_calculator is None:
            try:
                from options.volatility import RealizedVolatility
                self._vol_calculator = RealizedVolatility()
            except Exception as e:
                logger.warning(f"Could not load volatility calculator: {e}")
        return self._vol_calculator

    def _get_historical_data(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """Fetch historical data for volatility calculation."""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            if self.data_provider:
                # data_provider is fetch_daily_bars_polygon function
                df = self.data_provider(symbol, start_date, end_date)
                if df is not None and len(df) > 0:
                    return df

            # Fallback to cached data
            cache_path = ROOT / 'data' / 'cache' / f'{symbol}_daily.csv'
            if cache_path.exists():
                df = pd.read_csv(cache_path, parse_dates=['date'])
                df = df.set_index('date').sort_index()
                return df

        except Exception as e:
            logger.warning(f"Could not get data for {symbol}: {e}")

        return None

    def _get_week_open_price(self, df: pd.DataFrame) -> float:
        """Get Monday's open price for the current week."""
        if df is None or len(df) == 0:
            return 0.0

        # Find the most recent Monday or first day of this week
        today = datetime.now().date()
        days_since_monday = today.weekday()  # Monday = 0
        monday = today - timedelta(days=days_since_monday)

        # Look for Monday or the first trading day after
        for i in range(5):  # Check up to 5 days
            check_date = monday + timedelta(days=i)
            if check_date in df.index.date if hasattr(df.index, 'date') else []:
                return float(df.loc[check_date.strftime('%Y-%m-%d')]['open'])

        # Fallback: use the open from 5 trading days ago
        if len(df) >= 5:
            return float(df['open'].iloc[-5])

        return float(df['open'].iloc[0])

    def calculate_weekly_expected_move(
        self,
        symbol: str,
        current_price: Optional[float] = None,
    ) -> ExpectedMove:
        """
        Calculate the weekly expected move for a stock.

        Args:
            symbol: Stock symbol
            current_price: Current price (if None, uses latest close)

        Returns:
            ExpectedMove with range and room analysis
        """
        df = self._get_historical_data(symbol, 365)

        if df is None or len(df) < 30:
            return ExpectedMove(
                symbol=symbol,
                current_price=current_price or 0,
                weekly_expected_move_pct=0,
                weekly_expected_move_dollars=0,
                upper_bound=0,
                lower_bound=0,
                week_open_price=0,
                move_from_week_open_pct=0,
                remaining_room_up_pct=0,
                remaining_room_down_pct=0,
                remaining_room_direction="UNKNOWN",
                volatility_20d=0,
                volatility_percentile=0,
                calculation_method="insufficient_data",
                interpretation="Insufficient data for expected move calculation",
            )

        # Get current price if not provided
        if current_price is None:
            current_price = float(df['close'].iloc[-1])

        # Calculate 20-day realized volatility
        if self.vol_calculator:
            vol_result = self.vol_calculator.close_to_close(df['close'], lookback=20)
            vol_20d = vol_result.volatility
        else:
            # Manual calculation
            log_returns = np.log(df['close'] / df['close'].shift(1)).tail(20).dropna()
            vol_20d = float(np.std(log_returns) * np.sqrt(252))

        # Calculate weekly expected move
        # EM = Price * Vol * sqrt(DTE/252)
        weekly_factor = np.sqrt(self.TRADING_DAYS_PER_WEEK / self.TRADING_DAYS_PER_YEAR)
        weekly_em_pct = vol_20d * weekly_factor
        weekly_em_dollars = current_price * weekly_em_pct

        # Calculate bounds
        upper_bound = current_price + weekly_em_dollars
        lower_bound = current_price - weekly_em_dollars

        # Get week open price
        week_open = self._get_week_open_price(df)
        if week_open == 0:
            week_open = current_price  # Fallback

        # Calculate move from week open
        move_from_open_pct = (current_price - week_open) / week_open if week_open > 0 else 0

        # Calculate remaining room
        # Expected range from week open
        week_upper = week_open * (1 + weekly_em_pct)
        week_lower = week_open * (1 - weekly_em_pct)

        remaining_up = (week_upper - current_price) / current_price if current_price > 0 else 0
        remaining_down = (current_price - week_lower) / current_price if current_price > 0 else 0

        # Determine direction with most room
        if remaining_up > weekly_em_pct * 0.5 and remaining_down > weekly_em_pct * 0.5:
            direction = "BOTH"
        elif remaining_up > weekly_em_pct * 0.25:
            direction = "UP"
        elif remaining_down > weekly_em_pct * 0.25:
            direction = "DOWN"
        else:
            direction = "EXHAUSTED"

        # Calculate volatility percentile
        # How does current 20d vol compare to historical?
        rolling_vol = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        if len(rolling_vol) > 0:
            vol_percentile = (rolling_vol < vol_20d).sum() / len(rolling_vol) * 100
        else:
            vol_percentile = 50.0

        # Build interpretation
        interpretations = []

        if weekly_em_pct > 0.08:
            interpretations.append(f"High volatility ({vol_20d:.0%} annualized)")
        elif weekly_em_pct < 0.03:
            interpretations.append(f"Low volatility ({vol_20d:.0%} annualized)")
        else:
            interpretations.append(f"Normal volatility ({vol_20d:.0%} annualized)")

        if direction == "EXHAUSTED":
            interpretations.append(f"Already moved {abs(move_from_open_pct):.1%} from week open, near edge of expected range")
        elif direction == "UP":
            interpretations.append(f"More room to move up ({remaining_up:.1%})")
        elif direction == "DOWN":
            interpretations.append(f"More room to move down ({remaining_down:.1%})")
        else:
            interpretations.append(f"Room to move in either direction (+{remaining_up:.1%} / -{remaining_down:.1%})")

        interpretation = ". ".join(interpretations)

        return ExpectedMove(
            symbol=symbol,
            current_price=round(current_price, 2),
            weekly_expected_move_pct=round(weekly_em_pct, 4),
            weekly_expected_move_dollars=round(weekly_em_dollars, 2),
            upper_bound=round(upper_bound, 2),
            lower_bound=round(lower_bound, 2),
            week_open_price=round(week_open, 2),
            move_from_week_open_pct=round(move_from_open_pct, 4),
            remaining_room_up_pct=round(max(0, remaining_up), 4),
            remaining_room_down_pct=round(max(0, remaining_down), 4),
            remaining_room_direction=direction,
            volatility_20d=round(vol_20d, 4),
            volatility_percentile=round(vol_percentile, 1),
            calculation_method="realized_vol",
            interpretation=interpretation,
        )

    def get_multiple_timeframe_moves(
        self,
        symbol: str,
        current_price: Optional[float] = None,
    ) -> Dict[str, ExpectedMove]:
        """
        Calculate expected moves for multiple timeframes.

        Returns daily, weekly, and monthly expected moves.
        """
        df = self._get_historical_data(symbol, 365)

        if df is None or len(df) < 30:
            return {}

        if current_price is None:
            current_price = float(df['close'].iloc[-1])

        # Calculate volatility
        log_returns = np.log(df['close'] / df['close'].shift(1)).tail(20).dropna()
        vol_20d = float(np.std(log_returns) * np.sqrt(252))

        results = {}

        # Daily (1 trading day)
        daily_factor = np.sqrt(1 / 252)
        daily_em = current_price * vol_20d * daily_factor
        results['daily'] = {
            'expected_move_pct': round(vol_20d * daily_factor, 4),
            'expected_move_dollars': round(daily_em, 2),
            'upper_bound': round(current_price + daily_em, 2),
            'lower_bound': round(current_price - daily_em, 2),
        }

        # Weekly (5 trading days)
        weekly = self.calculate_weekly_expected_move(symbol, current_price)
        results['weekly'] = weekly.to_dict()

        # Monthly (21 trading days)
        monthly_factor = np.sqrt(21 / 252)
        monthly_em = current_price * vol_20d * monthly_factor
        results['monthly'] = {
            'expected_move_pct': round(vol_20d * monthly_factor, 4),
            'expected_move_dollars': round(monthly_em, 2),
            'upper_bound': round(current_price + monthly_em, 2),
            'lower_bound': round(current_price - monthly_em, 2),
        }

        return results


# Convenience function
def get_expected_move_calculator() -> ExpectedMoveCalculator:
    """Factory function to get calculator instance."""
    return ExpectedMoveCalculator()
