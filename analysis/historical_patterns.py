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
class HistoricalInstance:
    """A single historical instance of a pattern occurrence."""
    end_date: str  # Date when streak ended (YYYY-MM-DD)
    streak_length: int  # How many consecutive days
    day1_return: float  # Return on day after streak ended
    bounce_days: int  # How many consecutive up days after
    total_bounce: float  # Total return over bounce period
    start_price: float  # Price at start of streak
    end_price: float  # Price at end of streak
    drop_pct: float  # Total drop during streak

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntryTimingRecommendation:
    """Entry timing recommendation based on historical pattern analysis."""
    symbol: str
    optimal_entry_day: int  # e.g., 5 = enter on day 5 of streak
    current_streak: int  # Current streak length
    should_enter_now: bool  # True if current streak >= optimal entry day
    timing_recommendation: str  # e.g., "ENTER_NOW", "WAIT", "TOO_EARLY"
    justification: str  # Human-readable explanation
    avg_days_to_bounce: float  # Average days from streak end to bounce
    bounce_day_distribution: Dict[int, float] = field(default_factory=dict)  # {day: win_rate}
    confidence: str = "MEDIUM"  # Confidence in timing

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    confidence: str  # "HIGH" | "MEDIUM-HIGH" | "MEDIUM" | "LOW-MEDIUM" | "LOW"
    evidence: str  # Human-readable explanation
    lookback_years: int = 5
    # NEW: Detailed metrics
    day1_bounce_avg: float = 0.0
    day1_bounce_min: float = 0.0
    day1_bounce_max: float = 0.0
    avg_bounce_days: float = 0.0
    total_bounce_avg: float = 0.0
    historical_instances: List[HistoricalInstance] = field(default_factory=list)
    # NEW: Entry timing fields
    optimal_entry_day: int = 5  # Default: enter on day 5
    entry_timing: Optional[EntryTimingRecommendation] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert HistoricalInstance objects to dicts
        d['historical_instances'] = [inst if isinstance(inst, dict) else asdict(inst)
                                      for inst in self.historical_instances]
        return d


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
        historical_instances = []  # NEW: Track all instances with dates

        if pattern_type == "consecutive_down" and current_streak >= 2:
            # Find all historical instances of N consecutive down days
            consecutive_count = 0
            streak_start_idx = 0
            for i in range(1, len(df)):
                if df.iloc[i]['is_down']:
                    if consecutive_count == 0:
                        streak_start_idx = i
                    consecutive_count += 1
                else:
                    if consecutive_count >= current_streak:
                        # This is day N+1 after a streak matching current
                        next_return = df.iloc[i]['return']
                        if not np.isnan(next_return):
                            historical_reversals.append(next_return)

                            # Calculate bounce days and total bounce
                            bounce_days = 0
                            total_bounce = 0.0
                            for j in range(i, min(i + 10, len(df))):
                                if df.iloc[j]['return'] > 0:
                                    bounce_days += 1
                                    total_bounce += df.iloc[j]['return']
                                else:
                                    break

                            # Get prices for drop calculation
                            start_price = df.iloc[streak_start_idx - 1]['close'] if streak_start_idx > 0 else df.iloc[streak_start_idx]['open']
                            end_price = df.iloc[i - 1]['close']
                            drop_pct = (end_price - start_price) / start_price

                            # Get date - handle both index types
                            try:
                                if hasattr(df.index, 'strftime'):
                                    end_date = df.index[i - 1].strftime('%Y-%m-%d')
                                elif 'timestamp' in df.columns:
                                    ts = df.iloc[i - 1]['timestamp']
                                    if hasattr(ts, 'strftime'):
                                        end_date = ts.strftime('%Y-%m-%d')
                                    else:
                                        end_date = str(ts)[:10]
                                else:
                                    end_date = str(df.index[i - 1])[:10]
                            except Exception:
                                end_date = "unknown"

                            historical_instances.append(HistoricalInstance(
                                end_date=end_date,
                                streak_length=consecutive_count,
                                day1_return=next_return,
                                bounce_days=bounce_days,
                                total_bounce=total_bounce,
                                start_price=round(start_price, 2),
                                end_price=round(end_price, 2),
                                drop_pct=round(drop_pct, 4),
                            ))
                    consecutive_count = 0

        elif pattern_type == "consecutive_up" and current_streak >= 2:
            # Find all historical instances of N consecutive up days
            consecutive_count = 0
            streak_start_idx = 0
            for i in range(1, len(df)):
                if df.iloc[i]['is_up']:
                    if consecutive_count == 0:
                        streak_start_idx = i
                    consecutive_count += 1
                else:
                    if consecutive_count >= current_streak:
                        next_return = df.iloc[i]['return']
                        if not np.isnan(next_return):
                            historical_reversals.append(next_return)
                            # Similar tracking for up streaks (pullback instead of bounce)
                            historical_instances.append(HistoricalInstance(
                                end_date="",
                                streak_length=consecutive_count,
                                day1_return=next_return,
                                bounce_days=0,
                                total_bounce=0.0,
                                start_price=0.0,
                                end_price=0.0,
                                drop_pct=0.0,
                            ))
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

            # NEW: Calculate detailed bounce metrics
            if historical_instances:
                day1_bounces = [inst.day1_return for inst in historical_instances]
                bounce_days_list = [inst.bounce_days for inst in historical_instances]
                total_bounces = [inst.total_bounce for inst in historical_instances]

                day1_bounce_avg = np.mean(day1_bounces)
                day1_bounce_min = np.min(day1_bounces)
                day1_bounce_max = np.max(day1_bounces)
                avg_bounce_days = np.mean(bounce_days_list)
                total_bounce_avg = np.mean(total_bounces)
            else:
                day1_bounce_avg = avg_magnitude
                day1_bounce_min = min_magnitude
                day1_bounce_max = max_magnitude
                avg_bounce_days = 1.0
                total_bounce_avg = avg_magnitude

            # NEW: Improved confidence calculation - weights BOTH sample size AND reversal rate
            if sample_size >= 20 and reversal_rate >= 0.95:
                confidence = "HIGH"
            elif sample_size >= 15 and reversal_rate >= 0.90:
                confidence = "MEDIUM-HIGH"
            elif sample_size >= 10 and reversal_rate >= 0.80:
                confidence = "MEDIUM"
            elif sample_size >= 5 and reversal_rate >= 0.70:
                confidence = "LOW-MEDIUM"
            else:
                confidence = "LOW"

            # Build evidence string with more detail
            direction = "down" if pattern_type == "consecutive_down" else "up"
            reversal_word = "bounced" if pattern_type == "consecutive_down" else "pulled back"

            evidence = (
                f"Based on {sample_size} instances since {datetime.now().year - self.lookback_years}, "
                f"when {symbol} was {direction} {current_streak}+ consecutive days, "
                f"day {current_streak + 1} {reversal_word} {reversal_rate:.0%} of the time "
                f"with an average move of {avg_magnitude:+.1%}. "
                f"Avg hold: {avg_bounce_days:.1f} days, total bounce: {total_bounce_avg:+.1%}."
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
            day1_bounce_avg = 0.0
            day1_bounce_min = 0.0
            day1_bounce_max = 0.0
            avg_bounce_days = 0.0
            total_bounce_avg = 0.0

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
            # NEW: Detailed metrics
            day1_bounce_avg=round(day1_bounce_avg, 4),
            day1_bounce_min=round(day1_bounce_min, 4),
            day1_bounce_max=round(day1_bounce_max, 4),
            avg_bounce_days=round(avg_bounce_days, 1),
            total_bounce_avg=round(total_bounce_avg, 4),
            historical_instances=historical_instances,
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

    def get_entry_timing_recommendation(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        pattern: Optional[ConsecutiveDayPattern] = None,
    ) -> EntryTimingRecommendation:
        """
        Get entry timing recommendation based on BACKTESTED historical data.

        Philosophy: Analyze each stock's actual historical bounce patterns to find
        the optimal streak length for entry. Don't use hardcoded values - let the
        data for THIS SPECIFIC STOCK determine when to enter.

        The optimal entry day is the streak length where:
        1. Win rate is highest (bounce probability)
        2. Sample size is sufficient for statistical confidence
        3. Risk/reward is favorable

        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame (optional)
            pattern: Pre-computed pattern (optional, will compute if not provided)

        Returns:
            EntryTimingRecommendation with data-driven optimal entry day
        """
        if pattern is None:
            pattern = self.analyze_consecutive_days(df, symbol)

        # Default values for no/insufficient data
        if pattern.sample_size < 5:
            return EntryTimingRecommendation(
                symbol=symbol,
                optimal_entry_day=5,  # Conservative default when no data
                current_streak=pattern.current_streak,
                should_enter_now=False,
                timing_recommendation="INSUFFICIENT_DATA",
                justification="Not enough historical samples to determine optimal entry",
                avg_days_to_bounce=1.0,
                bounce_day_distribution={},
                confidence="LOW",
            )

        # ============================================================
        # BACKTEST-DRIVEN OPTIMAL ENTRY CALCULATION
        # ============================================================
        # Group historical instances by streak length and calculate metrics
        streak_stats: Dict[int, Dict[str, Any]] = {}

        for inst in pattern.historical_instances:
            streak_len = inst.streak_length
            if streak_len not in streak_stats:
                streak_stats[streak_len] = {
                    'instances': [],
                    'wins': 0,
                    'total': 0,
                    'day1_returns': [],
                    'total_returns': [],
                    'bounce_days': [],
                }

            stats = streak_stats[streak_len]
            stats['instances'].append(inst)
            stats['total'] += 1

            # Win = positive return on day 1 OR positive bounce within 5 days
            is_win = inst.day1_return > 0 or inst.bounce_days > 0
            if is_win:
                stats['wins'] += 1
                stats['day1_returns'].append(inst.day1_return)
                stats['total_returns'].append(inst.total_bounce)
                stats['bounce_days'].append(inst.bounce_days if inst.bounce_days > 0 else 1)

        # Calculate win rate and expected return for each streak length
        win_rate_by_streak: Dict[int, float] = {}
        expected_return_by_streak: Dict[int, float] = {}
        sample_size_by_streak: Dict[int, int] = {}

        for streak_len, stats in streak_stats.items():
            win_rate = stats['wins'] / stats['total'] if stats['total'] > 0 else 0
            win_rate_by_streak[streak_len] = win_rate
            sample_size_by_streak[streak_len] = stats['total']

            # Expected return = win_rate * avg_win_return - (1-win_rate) * avg_loss
            if stats['day1_returns']:
                expected_return_by_streak[streak_len] = np.mean(stats['day1_returns'])
            else:
                expected_return_by_streak[streak_len] = 0

        # ============================================================
        # FIND OPTIMAL ENTRY DAY FROM BACKTEST DATA
        # ============================================================
        # Score each streak length: balance win rate, sample size, and expected return
        optimal_entry_day = None
        best_score = -float('inf')
        min_sample_for_confidence = 5  # Need at least 5 samples

        for streak_len in sorted(streak_stats.keys()):
            win_rate = win_rate_by_streak.get(streak_len, 0)
            sample_size = sample_size_by_streak.get(streak_len, 0)
            expected_ret = expected_return_by_streak.get(streak_len, 0)

            # Skip if insufficient samples
            if sample_size < min_sample_for_confidence:
                continue

            # Score formula: prioritize high win rate, then expected return
            # Penalty for very long streaks (may be missing the move)
            score = (
                win_rate * 100  # Win rate (max 100)
                + expected_ret * 500  # Expected return boost (e.g., 2% = 10 points)
                + np.log1p(sample_size) * 5  # Sample size confidence bonus
                - max(0, streak_len - 7) * 3  # Penalty for waiting too long
            )

            logger.debug(
                f"{symbol} streak={streak_len}: WR={win_rate:.0%}, "
                f"N={sample_size}, E[R]={expected_ret:+.1%}, score={score:.1f}"
            )

            if score > best_score:
                best_score = score
                optimal_entry_day = streak_len

        # Fallback if no good streak found
        if optimal_entry_day is None:
            # Use the streak length with highest sample size as fallback
            if sample_size_by_streak:
                optimal_entry_day = max(sample_size_by_streak, key=sample_size_by_streak.get)
            else:
                optimal_entry_day = 5  # Ultimate fallback

        logger.info(
            f"BACKTEST RESULT: {symbol} optimal_entry_day={optimal_entry_day} "
            f"(WR={win_rate_by_streak.get(optimal_entry_day, 0):.0%}, "
            f"N={sample_size_by_streak.get(optimal_entry_day, 0)})"
        )

        # Calculate average days to bounce from all historical instances
        all_bounce_times = []
        for inst in pattern.historical_instances:
            if inst.day1_return > 0:
                all_bounce_times.append(1)
            elif inst.bounce_days > 0:
                all_bounce_times.append(inst.bounce_days)
        avg_days_to_bounce = np.mean(all_bounce_times) if all_bounce_times else 1.0

        # Get stats for optimal entry day
        opt_win_rate = win_rate_by_streak.get(optimal_entry_day, 0)
        opt_sample_size = sample_size_by_streak.get(optimal_entry_day, 0)
        opt_expected_ret = expected_return_by_streak.get(optimal_entry_day, 0)

        # Determine recommendation based on current streak vs BACKTESTED optimal
        current_streak = pattern.current_streak

        if current_streak >= optimal_entry_day:
            should_enter_now = True
            timing_recommendation = "ENTER_NOW"
            if current_streak == optimal_entry_day:
                justification = (
                    f"Day {current_streak} = OPTIMAL ENTRY (backtested). "
                    f"Backtest shows {opt_win_rate:.0%} win rate at this streak length "
                    f"({opt_sample_size} samples, avg return +{opt_expected_ret:.1%}). "
                    f"This is the sweet spot for {symbol}."
                )
            else:
                # Get win rate for current streak if available
                curr_win_rate = win_rate_by_streak.get(current_streak, opt_win_rate)
                justification = (
                    f"Day {current_streak} exceeds optimal (day {optimal_entry_day}). "
                    f"Backtest shows {curr_win_rate:.0%} win rate at day {current_streak}. "
                    f"Still valid but may have missed some upside. "
                    f"Optimal was day {optimal_entry_day} with {opt_win_rate:.0%} WR."
                )
        elif current_streak == optimal_entry_day - 1:
            should_enter_now = False
            timing_recommendation = "ALMOST_READY"
            justification = (
                f"Day {current_streak}. Backtest optimal = day {optimal_entry_day}. "
                f"One more down day triggers entry. "
                f"Expected: {opt_win_rate:.0%} WR, +{opt_expected_ret:.1%} avg return."
            )
        elif current_streak >= optimal_entry_day - 2:
            should_enter_now = False
            timing_recommendation = "WAIT"
            days_to_wait = optimal_entry_day - current_streak
            justification = (
                f"Day {current_streak}. Backtest optimal = day {optimal_entry_day}. "
                f"Wait {days_to_wait} more down days. "
                f"Early entry at day {current_streak} has lower historical edge."
            )
        else:
            should_enter_now = False
            timing_recommendation = "TOO_EARLY"
            days_to_wait = optimal_entry_day - current_streak
            justification = (
                f"Day {current_streak}. Backtest optimal = day {optimal_entry_day}. "
                f"Need {days_to_wait} more down days. "
                f"Pattern not mature enough - premature entry reduces edge significantly."
            )

        # Confidence based on BACKTESTED sample size and win rate at optimal day
        if opt_sample_size >= 20 and opt_win_rate >= 0.90:
            confidence = "HIGH"
        elif opt_sample_size >= 15 and opt_win_rate >= 0.80:
            confidence = "MEDIUM-HIGH"
        elif opt_sample_size >= 10 and opt_win_rate >= 0.70:
            confidence = "MEDIUM"
        elif opt_sample_size >= 5:
            confidence = "LOW-MEDIUM"
        else:
            confidence = "LOW"

        return EntryTimingRecommendation(
            symbol=symbol,
            optimal_entry_day=optimal_entry_day,
            current_streak=current_streak,
            should_enter_now=should_enter_now,
            timing_recommendation=timing_recommendation,
            justification=justification,
            avg_days_to_bounce=round(avg_days_to_bounce, 1),
            bounce_day_distribution=win_rate_by_streak,
            confidence=confidence,
        )

    def get_pattern_significance(
        self,
        pattern: ConsecutiveDayPattern,
        alpha: float = 0.05,
        n_trials: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance of a pattern.

        Uses professional quant methods:
        - Binomial test with Bonferroni correction
        - Wilson confidence intervals
        - Sample size assessment

        Args:
            pattern: ConsecutiveDayPattern to test
            alpha: Significance level (default 0.05)
            n_trials: Number of patterns tested (for Bonferroni correction)

        Returns:
            Dict with statistical test results
        """
        try:
            from analytics.statistical_testing import (
                compute_binomial_pvalue,
                wilson_confidence_interval,
            )
        except ImportError:
            logger.warning("Statistical testing module not available")
            return {
                'available': False,
                'reason': 'Module not found'
            }

        if pattern.sample_size == 0:
            return {
                'available': False,
                'reason': 'No historical instances'
            }

        # Count wins (instances with positive day1 bounce)
        wins = sum(1 for inst in pattern.historical_instances if inst.day1_return > 0)
        total = pattern.sample_size

        # Binomial test with Bonferroni correction
        binomial_result = compute_binomial_pvalue(
            wins=wins,
            total=total,
            null_prob=0.5,
            alpha=alpha,
            n_trials=n_trials,
            alternative="greater"
        )

        # Wilson confidence interval
        ci = wilson_confidence_interval(wins, total, 0.95)

        return {
            'available': True,
            'sample_size': total,
            'wins': wins,
            'win_rate': wins / total if total > 0 else 0.0,
            'p_value': binomial_result.p_value,
            'alpha': binomial_result.alpha,
            'alpha_adjusted': binomial_result.alpha_adjusted,
            'is_significant': binomial_result.is_significant,
            'confidence_interval_lower': ci.lower_bound,
            'confidence_interval_upper': ci.upper_bound,
            'n_trials': n_trials,
        }

    def test_multiple_patterns(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        streak_range: Optional[range] = None,
        alpha: float = 0.05
    ) -> Dict[int, Dict[str, Any]]:
        """
        Test multiple consecutive-down-day patterns and return statistics.

        This is the core of pattern optimization: test MULTIPLE patterns
        (e.g., 2-10 consecutive down days) instead of hardcoding one value.

        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame (optional)
            streak_range: Range of streak lengths to test (default: range(2, 11))
            alpha: Significance level (default 0.05)

        Returns:
            Dict mapping streak_length -> pattern statistics with significance tests

        Example:
            >>> analyzer = HistoricalPatternAnalyzer()
            >>> results = analyzer.test_multiple_patterns('AAPL', df, range(2, 11))
            >>> for streak_len, stats in sorted(results.items()):
            ...     print(f"{streak_len}-day: WR={stats['win_rate']:.1%}, p={stats['p_value']:.4f}")
        """
        if df is None:
            df = self._get_historical_data(symbol, self.lookback_years)

        if streak_range is None:
            streak_range = range(2, 11)  # Test 2-10 days

        # Get full pattern analysis
        pattern = self.analyze_consecutive_days(df, symbol)

        # Number of patterns being tested (for Bonferroni correction)
        n_trials = len(streak_range)

        # Group instances by streak length
        results = {}
        for streak_len in streak_range:
            # Filter instances matching this streak length
            matching_instances = [
                inst for inst in pattern.historical_instances
                if inst.streak_length == streak_len
            ]

            if len(matching_instances) == 0:
                results[streak_len] = {
                    'streak_length': streak_len,
                    'sample_size': 0,
                    'win_rate': 0.0,
                    'avg_return': 0.0,
                    'median_return': 0.0,
                    'p_value': 1.0,
                    'is_significant': False,
                    'reason': 'No instances found'
                }
                continue

            # Calculate statistics
            total = len(matching_instances)
            wins = sum(1 for inst in matching_instances if inst.day1_return > 0)
            returns = [inst.day1_return for inst in matching_instances]

            # Binomial test
            try:
                from analytics.statistical_testing import (
                    compute_binomial_pvalue,
                    wilson_confidence_interval,
                )

                binomial_result = compute_binomial_pvalue(
                    wins=wins,
                    total=total,
                    null_prob=0.5,
                    alpha=alpha,
                    n_trials=n_trials,
                    alternative="greater"
                )

                ci = wilson_confidence_interval(wins, total, 0.95)

                results[streak_len] = {
                    'streak_length': streak_len,
                    'sample_size': total,
                    'wins': wins,
                    'win_rate': wins / total,
                    'avg_return': np.mean(returns),
                    'median_return': np.median(returns),
                    'p_value': binomial_result.p_value,
                    'alpha_adjusted': binomial_result.alpha_adjusted,
                    'is_significant': binomial_result.is_significant,
                    'confidence_interval_lower': ci.lower_bound,
                    'confidence_interval_upper': ci.upper_bound,
                }
            except ImportError:
                # Fallback if statistical testing module not available
                results[streak_len] = {
                    'streak_length': streak_len,
                    'sample_size': total,
                    'wins': wins,
                    'win_rate': wins / total,
                    'avg_return': np.mean(returns),
                    'median_return': np.median(returns),
                    'reason': 'Statistical module not available'
                }

        return results

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

        # Get consecutive pattern first
        pattern = self.analyze_consecutive_days(df, symbol)

        # Get entry timing based on pattern
        entry_timing = self.get_entry_timing_recommendation(symbol, df, pattern)

        return {
            'symbol': symbol,
            'consecutive_pattern': pattern.to_dict(),
            'entry_timing': entry_timing.to_dict(),
            'support_resistance': [sr.to_dict() for sr in self.analyze_support_resistance(df, symbol)],
            'volume_profile': self.analyze_volume_profile(df, symbol).to_dict(),
            'sector_relative_strength': self.get_sector_relative_strength(symbol).to_dict(),
            'analysis_timestamp': datetime.now().isoformat(),
        }


# Convenience function
def get_historical_pattern_analyzer(lookback_years: int = 5) -> HistoricalPatternAnalyzer:
    """Factory function to get analyzer instance."""
    return HistoricalPatternAnalyzer(lookback_years)


def qualifies_for_auto_pass(pattern: ConsecutiveDayPattern) -> bool:
    """
    Check if a pattern qualifies for automatic quality gate bypass.

    CRITERIA (ALL must be met):
    - 20+ historical samples (statistically significant at p < 0.05)
    - 90%+ historical win rate
    - Current streak matches pattern requirements (5+ days)

    Returns:
        True if pattern should auto-pass quality gate
    """
    if pattern.sample_size < 20:
        return False
    if pattern.historical_reversal_rate < 0.90:
        return False
    if pattern.current_streak < 5:  # Need meaningful streak
        return False
    return True


def enrich_signal_with_historical_pattern(
    signal: Dict[str, Any],
    analyzer: Optional[HistoricalPatternAnalyzer] = None,
) -> Dict[str, Any]:
    """
    Enrich a signal dict with historical pattern data for quality gate.

    This should be called BEFORE passing signals to the quality gate
    so that signals with strong historical patterns can auto-pass.

    Args:
        signal: Signal dict with 'symbol' key
        analyzer: Optional analyzer instance (creates one if not provided)

    Returns:
        Enriched signal dict with 'historical_pattern' and 'entry_timing' keys added
    """
    if analyzer is None:
        analyzer = get_historical_pattern_analyzer(lookback_years=5)

    symbol = signal.get('symbol', '')
    if not symbol:
        return signal

    try:
        pattern = analyzer.analyze_consecutive_days(symbol=symbol)
        entry_timing = analyzer.get_entry_timing_recommendation(symbol, pattern=pattern)

        # Add pattern to signal for quality gate evaluation
        signal['historical_pattern'] = {
            'pattern_type': pattern.pattern_type,
            'current_streak': pattern.current_streak,
            'historical_reversal_rate': pattern.historical_reversal_rate,
            'sample_size': pattern.sample_size,
            'avg_reversal_magnitude': pattern.avg_reversal_magnitude,
            'confidence': pattern.confidence,
            'evidence': pattern.evidence,
            'day1_bounce_avg': pattern.day1_bounce_avg,
            'total_bounce_avg': pattern.total_bounce_avg,
            'avg_bounce_days': pattern.avg_bounce_days,
            'qualifies_for_auto_pass': qualifies_for_auto_pass(pattern),
        }

        # Add entry timing recommendation
        signal['entry_timing'] = {
            'optimal_entry_day': entry_timing.optimal_entry_day,
            'current_streak': entry_timing.current_streak,
            'should_enter_now': entry_timing.should_enter_now,
            'timing_recommendation': entry_timing.timing_recommendation,
            'justification': entry_timing.justification,
            'avg_days_to_bounce': entry_timing.avg_days_to_bounce,
            'confidence': entry_timing.confidence,
        }

        if qualifies_for_auto_pass(pattern):
            logger.info(
                f"PATTERN AUTO-PASS ELIGIBLE: {symbol} - "
                f"{pattern.sample_size} samples, {pattern.historical_reversal_rate:.0%} win rate, "
                f"{pattern.current_streak} days down"
            )

        if entry_timing.should_enter_now:
            logger.info(
                f"ENTRY TIMING: {symbol} - {entry_timing.timing_recommendation} "
                f"(Day {entry_timing.current_streak} of streak, optimal={entry_timing.optimal_entry_day})"
            )

    except Exception as e:
        logger.warning(f"Could not analyze pattern for {symbol}: {e}")
        signal['historical_pattern'] = {}
        signal['entry_timing'] = {}

    return signal


def get_pattern_grade(pattern: ConsecutiveDayPattern) -> str:
    """
    Get letter grade for a pattern based on sample size and win rate.

    Grading:
    - A+: 20+ samples, 90%+ win rate (AUTO-PASS)
    - A:  15+ samples, 85%+ win rate
    - B:  10+ samples, 75%+ win rate
    - C:  5+ samples, 65%+ win rate
    - D:  < 5 samples or < 65% win rate

    Returns:
        Grade string: "A+", "A", "B", "C", or "D"
    """
    if pattern.sample_size >= 20 and pattern.historical_reversal_rate >= 0.90:
        return "A+"
    elif pattern.sample_size >= 15 and pattern.historical_reversal_rate >= 0.85:
        return "A"
    elif pattern.sample_size >= 10 and pattern.historical_reversal_rate >= 0.75:
        return "B"
    elif pattern.sample_size >= 5 and pattern.historical_reversal_rate >= 0.65:
        return "C"
    else:
        return "D"
