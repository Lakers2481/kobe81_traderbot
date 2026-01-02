"""
Historical Pattern Analyzer - Evidence-Based Trading Patterns
=============================================================

Analyzes historical price patterns to provide evidence for trade thesis:

1. Consecutive Day Patterns:
   - How many days has the stock been down/up?
   - What is the historical reversal rate for this streak length?
   - What is the average reversal magnitude?

2. Support/Resistance Analysis:
   - Key price levels based on pivots, volume, and fractals
   - Historical touches and bounces/breaks

3. Volume Profile:
   - Average volume comparison
   - Volume trends during pullbacks

4. Sector Relative Strength:
   - Performance vs sector ETF
   - Beta-adjusted relative strength

Usage:
    from analysis.historical_patterns import HistoricalPatternAnalyzer

    analyzer = HistoricalPatternAnalyzer()
    pattern = analyzer.analyze_consecutive_days(df, 'AAPL')
    print(f"{pattern.current_streak} days down, {pattern.historical_reversal_rate:.0%} reversal rate")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ConsecutiveDayPattern:
    """Historical pattern for consecutive down/up days."""
    symbol: str
    pattern_type: str  # "consecutive_down" | "consecutive_up" | "none"
    current_streak: int  # e.g., 6 days down
    historical_reversal_rate: float  # e.g., 0.72 (72% reversed on day N+1)
    sample_size: int  # How many times this pattern occurred historically
    avg_reversal_magnitude: float  # Average % move on reversal day
    median_reversal_magnitude: float  # Median % move on reversal day
    max_reversal_magnitude: float  # Best case reversal
    min_reversal_magnitude: float  # Worst case (continuation)
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"
    evidence: str  # Human-readable explanation
    lookback_years: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SupportResistanceLevel:
    """A significant support or resistance price level."""
    price: float
    level_type: str  # "support" | "resistance"
    strength: int  # Number of touches
    first_touch: str  # Date of first touch
    last_touch: str  # Date of last touch
    bounces: int  # Times price bounced off this level
    breaks: int  # Times price broke through
    distance_pct: float  # Distance from current price (%)
    justification: str  # Why this level matters

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VolumeProfile:
    """Volume analysis for a stock."""
    symbol: str
    avg_volume_20d: float
    avg_volume_50d: float
    relative_volume: float  # Today vs 20d avg
    volume_trend: str  # "increasing" | "decreasing" | "stable"
    buying_pressure: float  # 0-1 ratio of up-volume days
    high_volume_days_pct: float  # % of days with >1.5x avg volume
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SectorRelativeStrength:
    """Relative strength vs sector ETF."""
    symbol: str
    sector_etf: str
    period_days: int
    symbol_return: float
    sector_return: float
    relative_strength: float  # Symbol return - Sector return
    beta_vs_sector: float
    outperforming: bool
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HistoricalPatternAnalyzer:
    """
    Analyzes historical patterns for evidence-based trading.

    This is a core component of the Pre-Game Blueprint, providing
    the "WHY" behind trade setups with statistical backing.
    """

    # Sector ETF mapping for common stocks
    SECTOR_ETFS = {
        'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'AMD', 'CRM', 'ADBE', 'ORCL', 'CSCO', 'ACN', 'INTC', 'IBM', 'QCOM'],
        'XLY': ['AMZN', 'TSLA', 'MCD', 'HD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG', 'ORLY', 'MAR'],
        'XLF': ['BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'C', 'AXP', 'SCHW', 'BLK'],
        'XLV': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'MDT'],
        'XLE': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'WMB'],
        'XLI': ['GE', 'CAT', 'HON', 'UNP', 'RTX', 'BA', 'DE', 'LMT', 'UPS', 'ADP', 'MMM'],
        'XLP': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'MDLZ', 'CL', 'EL'],
        'XLU': ['NEE', 'SO', 'DUK', 'SRE', 'AEP', 'D', 'XEL', 'EXC', 'ED', 'WEC'],
        'XLRE': ['PLD', 'AMT', 'EQIX', 'PSA', 'O', 'WELL', 'SPG', 'AVB', 'DLR'],
        'XLC': ['META', 'GOOGL', 'GOOG', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'CHTR', 'EA', 'ATVI'],
    }

    def __init__(self, lookback_years: int = 5):
        """Initialize the analyzer."""
        self.lookback_years = lookback_years
        self._data_provider = None

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

    def _get_historical_data(
        self,
        symbol: str,
        years: int = 5,
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data for analysis."""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')

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

    def analyze_consecutive_days(
        self,
        df: Optional[pd.DataFrame] = None,
        symbol: str = "",
    ) -> ConsecutiveDayPattern:
        """
        Analyze consecutive up/down day patterns.

        Args:
            df: OHLCV DataFrame (if None, will fetch for symbol)
            symbol: Stock symbol

        Returns:
            ConsecutiveDayPattern with historical statistics
        """
        if df is None:
            df = self._get_historical_data(symbol, self.lookback_years)

        if df is None or len(df) < 30:
            return ConsecutiveDayPattern(
                symbol=symbol,
                pattern_type="none",
                current_streak=0,
                historical_reversal_rate=0.0,
                sample_size=0,
                avg_reversal_magnitude=0.0,
                median_reversal_magnitude=0.0,
                max_reversal_magnitude=0.0,
                min_reversal_magnitude=0.0,
                confidence="LOW",
                evidence="Insufficient data for analysis",
                lookback_years=self.lookback_years,
            )

        # Calculate daily returns
        df = df.copy()
        df['return'] = df['close'].pct_change()
        df['is_up'] = df['return'] > 0
        df['is_down'] = df['return'] < 0

        # Calculate current streak
        current_streak = 0
        pattern_type = "none"

        if len(df) > 0:
            # Check last few days
            recent = df.tail(10).copy()

            # Count consecutive down days from most recent
            down_streak = 0
            for i in range(len(recent) - 1, -1, -1):
                if recent.iloc[i]['is_down']:
                    down_streak += 1
                else:
                    break

            # Count consecutive up days from most recent
            up_streak = 0
            for i in range(len(recent) - 1, -1, -1):
                if recent.iloc[i]['is_up']:
                    up_streak += 1
                else:
                    break

            if down_streak >= 2:
                current_streak = down_streak
                pattern_type = "consecutive_down"
            elif up_streak >= 2:
                current_streak = up_streak
                pattern_type = "consecutive_up"

        # Calculate historical reversal rates for this streak length
        historical_reversals = []

        if pattern_type == "consecutive_down" and current_streak >= 2:
            # Find all historical instances of N consecutive down days
            consecutive_count = 0
            for i in range(1, len(df)):
                if df.iloc[i]['is_down']:
                    consecutive_count += 1
                else:
                    if consecutive_count >= current_streak:
                        # This is day N+1 after a streak matching current
                        next_return = df.iloc[i]['return']
                        if not np.isnan(next_return):
                            historical_reversals.append(next_return)
                    consecutive_count = 0

        elif pattern_type == "consecutive_up" and current_streak >= 2:
            # Find all historical instances of N consecutive up days
            consecutive_count = 0
            for i in range(1, len(df)):
                if df.iloc[i]['is_up']:
                    consecutive_count += 1
                else:
                    if consecutive_count >= current_streak:
                        next_return = df.iloc[i]['return']
                        if not np.isnan(next_return):
                            historical_reversals.append(next_return)
                    consecutive_count = 0

        # Calculate statistics
        if historical_reversals:
            reversals_array = np.array(historical_reversals)

            if pattern_type == "consecutive_down":
                # For down streaks, reversal = positive return
                reversal_rate = np.mean(reversals_array > 0)
                avg_magnitude = np.mean(reversals_array)
                median_magnitude = np.median(reversals_array)
                max_magnitude = np.max(reversals_array)
                min_magnitude = np.min(reversals_array)
            else:
                # For up streaks, reversal = negative return
                reversal_rate = np.mean(reversals_array < 0)
                avg_magnitude = np.mean(reversals_array)
                median_magnitude = np.median(reversals_array)
                max_magnitude = np.max(reversals_array)
                min_magnitude = np.min(reversals_array)

            sample_size = len(historical_reversals)

            # Determine confidence based on sample size
            if sample_size >= 30:
                confidence = "HIGH"
            elif sample_size >= 15:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            # Build evidence string
            direction = "down" if pattern_type == "consecutive_down" else "up"
            reversal_word = "bounced" if pattern_type == "consecutive_down" else "pulled back"

            evidence = (
                f"Based on {sample_size} instances since {datetime.now().year - self.lookback_years}, "
                f"when {symbol} was {direction} {current_streak}+ consecutive days, "
                f"day {current_streak + 1} {reversal_word} {reversal_rate:.0%} of the time "
                f"with an average move of {avg_magnitude:+.1%}."
            )

        else:
            reversal_rate = 0.0
            avg_magnitude = 0.0
            median_magnitude = 0.0
            max_magnitude = 0.0
            min_magnitude = 0.0
            sample_size = 0
            confidence = "LOW"
            evidence = f"No historical instances of {current_streak}+ consecutive days found."

        return ConsecutiveDayPattern(
            symbol=symbol,
            pattern_type=pattern_type,
            current_streak=current_streak,
            historical_reversal_rate=reversal_rate,
            sample_size=sample_size,
            avg_reversal_magnitude=avg_magnitude,
            median_reversal_magnitude=median_magnitude,
            max_reversal_magnitude=max_magnitude,
            min_reversal_magnitude=min_magnitude,
            confidence=confidence,
            evidence=evidence,
            lookback_years=self.lookback_years,
        )

    def analyze_support_resistance(
        self,
        df: Optional[pd.DataFrame] = None,
        symbol: str = "",
        num_levels: int = 5,
    ) -> List[SupportResistanceLevel]:
        """
        Identify key support and resistance levels.

        Uses pivot points, volume-weighted levels, and fractal analysis.
        """
        if df is None:
            df = self._get_historical_data(symbol, 1)  # 1 year for S/R

        if df is None or len(df) < 30:
            return []

        df = df.copy()
        current_price = float(df['close'].iloc[-1])

        levels = []

        # Method 1: Pivot Points (recent highs/lows)
        # Find local highs (resistance)
        df['is_pivot_high'] = (
            (df['high'] > df['high'].shift(1)) &
            (df['high'] > df['high'].shift(-1)) &
            (df['high'] > df['high'].shift(2)) &
            (df['high'] > df['high'].shift(-2))
        )

        # Find local lows (support)
        df['is_pivot_low'] = (
            (df['low'] < df['low'].shift(1)) &
            (df['low'] < df['low'].shift(-1)) &
            (df['low'] < df['low'].shift(2)) &
            (df['low'] < df['low'].shift(-2))
        )

        # Cluster pivot highs
        pivot_highs = df[df['is_pivot_high']]['high'].tolist()
        pivot_lows = df[df['is_pivot_low']]['low'].tolist()

        # Cluster nearby levels (within 1%)
        def cluster_levels(prices: List[float], tolerance: float = 0.01) -> List[Tuple[float, int]]:
            """Group nearby price levels and count touches."""
            if not prices:
                return []

            prices = sorted(prices)
            clusters = []
            current_cluster = [prices[0]]

            for price in prices[1:]:
                if abs(price - current_cluster[-1]) / current_cluster[-1] < tolerance:
                    current_cluster.append(price)
                else:
                    clusters.append((np.mean(current_cluster), len(current_cluster)))
                    current_cluster = [price]

            if current_cluster:
                clusters.append((np.mean(current_cluster), len(current_cluster)))

            return clusters

        # Get resistance levels
        resistance_clusters = cluster_levels(pivot_highs)
        for price, touches in sorted(resistance_clusters, key=lambda x: x[1], reverse=True)[:num_levels]:
            if price > current_price:  # Only above current price
                distance_pct = (price - current_price) / current_price * 100
                levels.append(SupportResistanceLevel(
                    price=round(price, 2),
                    level_type="resistance",
                    strength=touches,
                    first_touch="",
                    last_touch="",
                    bounces=touches - 1,
                    breaks=0,
                    distance_pct=round(distance_pct, 2),
                    justification=f"Pivot high touched {touches} times",
                ))

        # Get support levels
        support_clusters = cluster_levels(pivot_lows)
        for price, touches in sorted(support_clusters, key=lambda x: x[1], reverse=True)[:num_levels]:
            if price < current_price:  # Only below current price
                distance_pct = (current_price - price) / current_price * 100
                levels.append(SupportResistanceLevel(
                    price=round(price, 2),
                    level_type="support",
                    strength=touches,
                    first_touch="",
                    last_touch="",
                    bounces=touches - 1,
                    breaks=0,
                    distance_pct=round(distance_pct, 2),
                    justification=f"Pivot low touched {touches} times",
                ))

        # Method 2: Round numbers (psychological levels)
        # Find nearest round numbers
        round_levels = []
        for mult in [1, 5, 10, 25, 50, 100]:
            rounded = round(current_price / mult) * mult
            if rounded != current_price:
                round_levels.append(rounded)
            rounded_up = ((current_price // mult) + 1) * mult
            round_levels.append(rounded_up)
            rounded_down = (current_price // mult) * mult
            if rounded_down > 0:
                round_levels.append(rounded_down)

        # Add significant round numbers
        for price in set(round_levels):
            distance_pct = abs(price - current_price) / current_price * 100
            if 0.5 < distance_pct < 10:  # Within reasonable range
                level_type = "resistance" if price > current_price else "support"
                levels.append(SupportResistanceLevel(
                    price=round(price, 2),
                    level_type=level_type,
                    strength=1,
                    first_touch="",
                    last_touch="",
                    bounces=0,
                    breaks=0,
                    distance_pct=round(distance_pct, 2),
                    justification="Psychological round number",
                ))

        # Sort by strength and distance
        levels.sort(key=lambda x: (-x.strength, x.distance_pct))

        return levels[:num_levels * 2]  # Return top N support + N resistance

    def analyze_volume_profile(
        self,
        df: Optional[pd.DataFrame] = None,
        symbol: str = "",
    ) -> VolumeProfile:
        """Analyze volume patterns and trends."""
        if df is None:
            df = self._get_historical_data(symbol, 1)

        if df is None or len(df) < 30:
            return VolumeProfile(
                symbol=symbol,
                avg_volume_20d=0,
                avg_volume_50d=0,
                relative_volume=0,
                volume_trend="unknown",
                buying_pressure=0.5,
                high_volume_days_pct=0,
                interpretation="Insufficient data",
            )

        df = df.copy()

        # Calculate volume metrics
        avg_20d = df['volume'].tail(20).mean()
        avg_50d = df['volume'].tail(50).mean()
        latest_volume = df['volume'].iloc[-1] if 'volume' in df.columns else 0
        relative_volume = latest_volume / avg_20d if avg_20d > 0 else 1.0

        # Volume trend (is 20d avg higher than 50d avg?)
        if avg_20d > avg_50d * 1.1:
            volume_trend = "increasing"
        elif avg_20d < avg_50d * 0.9:
            volume_trend = "decreasing"
        else:
            volume_trend = "stable"

        # Buying pressure (up days with higher volume)
        df['up_day'] = df['close'] > df['close'].shift(1)
        up_volume = df[df['up_day']]['volume'].sum()
        total_volume = df['volume'].sum()
        buying_pressure = up_volume / total_volume if total_volume > 0 else 0.5

        # High volume days percentage
        high_vol_threshold = avg_50d * 1.5
        high_vol_days = (df['volume'] > high_vol_threshold).sum()
        high_volume_days_pct = high_vol_days / len(df)

        # Interpretation
        if relative_volume > 1.5 and buying_pressure > 0.55:
            interpretation = "High volume with buying pressure - bullish"
        elif relative_volume > 1.5 and buying_pressure < 0.45:
            interpretation = "High volume with selling pressure - bearish"
        elif relative_volume < 0.7:
            interpretation = "Low volume - weak conviction"
        else:
            interpretation = "Normal volume patterns"

        return VolumeProfile(
            symbol=symbol,
            avg_volume_20d=round(avg_20d, 0),
            avg_volume_50d=round(avg_50d, 0),
            relative_volume=round(relative_volume, 2),
            volume_trend=volume_trend,
            buying_pressure=round(buying_pressure, 2),
            high_volume_days_pct=round(high_volume_days_pct, 2),
            interpretation=interpretation,
        )

    def get_sector_etf(self, symbol: str) -> str:
        """Get the sector ETF for a symbol."""
        for etf, symbols in self.SECTOR_ETFS.items():
            if symbol in symbols:
                return etf
        return 'SPY'  # Default to SPY

    def get_sector_relative_strength(
        self,
        symbol: str,
        sector_etf: Optional[str] = None,
        period_days: int = 20,
    ) -> SectorRelativeStrength:
        """Calculate relative strength vs sector ETF."""
        if sector_etf is None:
            sector_etf = self.get_sector_etf(symbol)

        symbol_df = self._get_historical_data(symbol, 1)
        sector_df = self._get_historical_data(sector_etf, 1)

        if symbol_df is None or sector_df is None:
            return SectorRelativeStrength(
                symbol=symbol,
                sector_etf=sector_etf,
                period_days=period_days,
                symbol_return=0.0,
                sector_return=0.0,
                relative_strength=0.0,
                beta_vs_sector=1.0,
                outperforming=False,
                interpretation="Insufficient data",
            )

        # Calculate returns over period
        symbol_df = symbol_df.tail(period_days + 1)
        sector_df = sector_df.tail(period_days + 1)

        symbol_return = (symbol_df['close'].iloc[-1] / symbol_df['close'].iloc[0]) - 1
        sector_return = (sector_df['close'].iloc[-1] / sector_df['close'].iloc[0]) - 1

        relative_strength = symbol_return - sector_return

        # Calculate beta
        if len(symbol_df) > 10 and len(sector_df) > 10:
            symbol_returns = symbol_df['close'].pct_change().dropna()
            sector_returns = sector_df['close'].pct_change().dropna()

            # Align dates
            common_dates = symbol_returns.index.intersection(sector_returns.index)
            if len(common_dates) > 5:
                sym_ret = symbol_returns.loc[common_dates].values
                sec_ret = sector_returns.loc[common_dates].values

                covariance = np.cov(sym_ret, sec_ret)[0, 1]
                variance = np.var(sec_ret)
                beta = covariance / variance if variance > 0 else 1.0
            else:
                beta = 1.0
        else:
            beta = 1.0

        outperforming = relative_strength > 0

        # Interpretation
        if relative_strength > 0.05:
            interpretation = f"Significantly outperforming {sector_etf} by {relative_strength:.1%}"
        elif relative_strength > 0:
            interpretation = f"Slightly outperforming {sector_etf}"
        elif relative_strength > -0.05:
            interpretation = f"Slightly underperforming {sector_etf}"
        else:
            interpretation = f"Significantly underperforming {sector_etf} by {abs(relative_strength):.1%}"

        return SectorRelativeStrength(
            symbol=symbol,
            sector_etf=sector_etf,
            period_days=period_days,
            symbol_return=round(symbol_return, 4),
            sector_return=round(sector_return, 4),
            relative_strength=round(relative_strength, 4),
            beta_vs_sector=round(beta, 2),
            outperforming=outperforming,
            interpretation=interpretation,
        )

    def get_full_analysis(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive historical analysis for a symbol.

        Returns all analysis types in a single call.
        """
        if df is None:
            df = self._get_historical_data(symbol, self.lookback_years)

        return {
            'symbol': symbol,
            'consecutive_pattern': self.analyze_consecutive_days(df, symbol).to_dict(),
            'support_resistance': [sr.to_dict() for sr in self.analyze_support_resistance(df, symbol)],
            'volume_profile': self.analyze_volume_profile(df, symbol).to_dict(),
            'sector_relative_strength': self.get_sector_relative_strength(symbol).to_dict(),
            'analysis_timestamp': datetime.now().isoformat(),
        }


# Convenience function
def get_historical_pattern_analyzer(lookback_years: int = 5) -> HistoricalPatternAnalyzer:
    """Factory function to get analyzer instance."""
    return HistoricalPatternAnalyzer(lookback_years)
