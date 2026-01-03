"""
Smart Money Concepts (SMC) / ICT Pattern Detection.

Enhanced ICT pattern detection using the smartmoneyconcepts library
combined with custom implementations for institutional trading patterns.

Features:
- Order Blocks (OB): Supply/demand zones from institutional orders
- Fair Value Gaps (FVG): Imbalances indicating future price targets
- Liquidity Sweeps: Stop hunts and liquidity grabs
- Market Structure: Break of Structure (BOS), Change of Character (CHoCH)
- Kill Zones: Optimal trading sessions (London, NY)
- Silver Bullet: 10:00-11:00 AM reversal setup

Based on Inner Circle Trader (ICT) methodology by Michael J. Huddleston.

Usage:
    from strategies.ict.smart_money import SmartMoneyDetector

    detector = SmartMoneyDetector()
    signals = detector.detect_all(df)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.structured_log import jlog

# Check for smartmoneyconcepts availability AND required functions
# The library API has changed - verify functions exist before using
SMC_AVAILABLE = False
SMC_OB_AVAILABLE = False
SMC_FVG_AVAILABLE = False
SMC_LIQ_AVAILABLE = False
SMC_BOS_AVAILABLE = False

try:
    import io
    import sys as _sys
    # Suppress library's print statement (has Unicode that fails on Windows)
    _old_stdout = _sys.stdout
    _sys.stdout = io.StringIO()
    try:
        import smartmoneyconcepts as smc
        SMC_AVAILABLE = True
        # Check which functions actually exist
        SMC_OB_AVAILABLE = hasattr(smc, 'ob') and callable(getattr(smc, 'ob', None))
        SMC_FVG_AVAILABLE = hasattr(smc, 'fvg') and callable(getattr(smc, 'fvg', None))
        SMC_LIQ_AVAILABLE = hasattr(smc, 'liquidity') and callable(getattr(smc, 'liquidity', None))
        SMC_BOS_AVAILABLE = hasattr(smc, 'bos_choch') and callable(getattr(smc, 'bos_choch', None))
    finally:
        _sys.stdout = _old_stdout
except ImportError:
    pass  # Silent - will use custom implementations


class MarketStructure(Enum):
    """Market structure states."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"


class KillZone(Enum):
    """ICT Kill Zones - optimal trading sessions."""
    ASIAN = "asian"  # 19:00 - 00:00 ET
    LONDON = "london"  # 02:00 - 05:00 ET
    NY_OPEN = "ny_open"  # 09:30 - 11:00 ET
    NY_LUNCH = "ny_lunch"  # 12:00 - 13:30 ET
    NY_PM = "ny_pm"  # 14:00 - 16:00 ET


@dataclass
class SMCConfig:
    """Configuration for Smart Money Concepts detection."""

    # Order Block settings
    ob_swing_length: int = 10  # Bars for swing detection
    ob_close_mitigation: bool = False  # OB invalidated on close through

    # Fair Value Gap settings
    fvg_min_size_pct: float = 0.001  # Minimum gap size (0.1% of price)

    # Liquidity settings
    liq_swing_length: int = 20  # Lookback for liquidity levels

    # Break of Structure settings
    bos_swing_length: int = 10

    # Kill Zone times (ET)
    kz_london_start: time = field(default_factory=lambda: time(2, 0))
    kz_london_end: time = field(default_factory=lambda: time(5, 0))
    kz_ny_start: time = field(default_factory=lambda: time(9, 30))
    kz_ny_end: time = field(default_factory=lambda: time(11, 0))
    kz_silver_bullet_start: time = field(default_factory=lambda: time(10, 0))
    kz_silver_bullet_end: time = field(default_factory=lambda: time(11, 0))

    # Confluence requirements
    require_fvg_confluence: bool = True  # Signal must be near FVG
    require_ob_confluence: bool = True  # Signal must be at order block
    require_kill_zone: bool = False  # Signal must be in kill zone


class OrderBlockDetector:
    """
    Detect Order Blocks - institutional supply/demand zones.

    Order Blocks are the last opposite-colored candle before an impulsive move.
    They represent zones where institutions accumulated positions.

    Bullish OB: Last bearish candle before strong bullish move
    Bearish OB: Last bullish candle before strong bearish move
    """

    def __init__(self, swing_length: int = 10, close_mitigation: bool = False):
        self.swing_length = swing_length
        self.close_mitigation = close_mitigation

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Order Blocks in price data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with OB columns added
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        if SMC_OB_AVAILABLE:
            try:
                # Use library implementation
                ob_data = smc.ob(
                    df,
                    swing_length=self.swing_length,
                    close_mitigation=self.close_mitigation
                )
                df['ob_bullish'] = ob_data['OB'] == 1
                df['ob_bearish'] = ob_data['OB'] == -1
                df['ob_top'] = ob_data.get('Top', np.nan)
                df['ob_bottom'] = ob_data.get('Bottom', np.nan)
                return df
            except Exception:
                pass  # Silent fallback to custom implementation

        # Use custom implementation
        return self._detect_custom(df)

    def _detect_custom(self, df: pd.DataFrame) -> pd.DataFrame:
        """Custom Order Block detection."""
        n = len(df)
        df['ob_bullish'] = False
        df['ob_bearish'] = False
        df['ob_top'] = np.nan
        df['ob_bottom'] = np.nan

        # Find swing highs and lows
        for i in range(self.swing_length, n - self.swing_length):
            # Check for swing low (potential bullish OB trigger)
            is_swing_low = all(
                df['low'].iloc[i] <= df['low'].iloc[i-j] and
                df['low'].iloc[i] <= df['low'].iloc[i+j]
                for j in range(1, min(self.swing_length, n - i))
            )

            if is_swing_low:
                # Find last bearish candle before the swing low
                for j in range(1, min(self.swing_length, i)):
                    if df['close'].iloc[i-j] < df['open'].iloc[i-j]:  # Bearish
                        # Check for impulsive move after
                        if i + 3 < n:
                            move_up = (df['close'].iloc[i+3] - df['low'].iloc[i]) / df['low'].iloc[i]
                            if move_up > 0.01:  # 1% move
                                df.loc[df.index[i-j], 'ob_bullish'] = True
                                df.loc[df.index[i-j], 'ob_top'] = df['high'].iloc[i-j]
                                df.loc[df.index[i-j], 'ob_bottom'] = df['low'].iloc[i-j]
                        break

            # Check for swing high (potential bearish OB trigger)
            is_swing_high = all(
                df['high'].iloc[i] >= df['high'].iloc[i-j] and
                df['high'].iloc[i] >= df['high'].iloc[i+j]
                for j in range(1, min(self.swing_length, n - i))
            )

            if is_swing_high:
                # Find last bullish candle before the swing high
                for j in range(1, min(self.swing_length, i)):
                    if df['close'].iloc[i-j] > df['open'].iloc[i-j]:  # Bullish
                        # Check for impulsive move after
                        if i + 3 < n:
                            move_down = (df['high'].iloc[i] - df['close'].iloc[i+3]) / df['high'].iloc[i]
                            if move_down > 0.01:  # 1% move
                                df.loc[df.index[i-j], 'ob_bearish'] = True
                                df.loc[df.index[i-j], 'ob_top'] = df['high'].iloc[i-j]
                                df.loc[df.index[i-j], 'ob_bottom'] = df['low'].iloc[i-j]
                        break

        return df


class FairValueGapDetector:
    """
    Detect Fair Value Gaps (FVG) - price imbalances.

    FVG occurs when price moves so fast that it leaves a gap between
    candle wicks. These gaps often act as magnets for price.

    Bullish FVG: Gap between candle 1 high and candle 3 low
    Bearish FVG: Gap between candle 1 low and candle 3 high
    """

    def __init__(self, min_size_pct: float = 0.001):
        self.min_size_pct = min_size_pct

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with FVG columns added
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        if SMC_FVG_AVAILABLE:
            try:
                fvg_data = smc.fvg(df)
                df['fvg_bullish'] = fvg_data['FVG'] == 1
                df['fvg_bearish'] = fvg_data['FVG'] == -1
                df['fvg_top'] = fvg_data.get('Top', np.nan)
                df['fvg_bottom'] = fvg_data.get('Bottom', np.nan)
                return df
            except Exception:
                pass  # Silent fallback to custom implementation

        # Use custom implementation
        return self._detect_custom(df)

    def _detect_custom(self, df: pd.DataFrame) -> pd.DataFrame:
        """Custom FVG detection."""
        n = len(df)
        df['fvg_bullish'] = False
        df['fvg_bearish'] = False
        df['fvg_top'] = np.nan
        df['fvg_bottom'] = np.nan

        for i in range(2, n):
            # Bullish FVG: candle 1 high < candle 3 low
            gap_bull = df['low'].iloc[i] - df['high'].iloc[i-2]
            if gap_bull > df['close'].iloc[i] * self.min_size_pct:
                df.loc[df.index[i-1], 'fvg_bullish'] = True
                df.loc[df.index[i-1], 'fvg_top'] = df['low'].iloc[i]
                df.loc[df.index[i-1], 'fvg_bottom'] = df['high'].iloc[i-2]

            # Bearish FVG: candle 1 low > candle 3 high
            gap_bear = df['low'].iloc[i-2] - df['high'].iloc[i]
            if gap_bear > df['close'].iloc[i] * self.min_size_pct:
                df.loc[df.index[i-1], 'fvg_bearish'] = True
                df.loc[df.index[i-1], 'fvg_top'] = df['low'].iloc[i-2]
                df.loc[df.index[i-1], 'fvg_bottom'] = df['high'].iloc[i]

        return df


class LiquiditySweepDetector:
    """
    Detect Liquidity Sweeps - stop hunts and liquidity grabs.

    Liquidity sweeps occur when price takes out a swing high/low
    only to reverse. This represents institutional stop hunting.
    """

    def __init__(self, swing_length: int = 20):
        self.swing_length = swing_length

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Liquidity Sweeps.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with liquidity sweep columns
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        if SMC_LIQ_AVAILABLE:
            try:
                liq_data = smc.liquidity(df, swing_length=self.swing_length)
                df['liq_sweep_high'] = liq_data.get('Liquidity', 0) == 1
                df['liq_sweep_low'] = liq_data.get('Liquidity', 0) == -1
                df['liq_level'] = liq_data.get('Level', np.nan)
                return df
            except Exception:
                pass  # Silent fallback to custom implementation

        # Use custom implementation
        return self._detect_custom(df)

    def _detect_custom(self, df: pd.DataFrame) -> pd.DataFrame:
        """Custom liquidity sweep detection."""
        n = len(df)
        df['liq_sweep_high'] = False
        df['liq_sweep_low'] = False
        df['liq_level'] = np.nan

        # Find swing highs and lows
        swing_highs = df['high'].rolling(self.swing_length, center=True).max()
        swing_lows = df['low'].rolling(self.swing_length, center=True).min()

        for i in range(self.swing_length, n - 1):
            # Sweep high: price exceeds swing high but closes below
            if df['high'].iloc[i] > swing_highs.iloc[i-1]:
                if df['close'].iloc[i] < swing_highs.iloc[i-1]:
                    df.loc[df.index[i], 'liq_sweep_high'] = True
                    df.loc[df.index[i], 'liq_level'] = swing_highs.iloc[i-1]

            # Sweep low: price exceeds swing low but closes above
            if df['low'].iloc[i] < swing_lows.iloc[i-1]:
                if df['close'].iloc[i] > swing_lows.iloc[i-1]:
                    df.loc[df.index[i], 'liq_sweep_low'] = True
                    df.loc[df.index[i], 'liq_level'] = swing_lows.iloc[i-1]

        return df


class BreakOfStructureDetector:
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH).

    BOS: Price breaks a swing high/low in the direction of trend
    CHoCH: Price breaks a swing high/low against the trend (reversal)
    """

    def __init__(self, swing_length: int = 10):
        self.swing_length = swing_length

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect BOS and CHoCH.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with structure columns
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        if SMC_BOS_AVAILABLE:
            try:
                bos_data = smc.bos_choch(df, swing_length=self.swing_length)
                df['bos_bullish'] = bos_data.get('BOS', 0) == 1
                df['bos_bearish'] = bos_data.get('BOS', 0) == -1
                df['choch_bullish'] = bos_data.get('CHOCH', 0) == 1
                df['choch_bearish'] = bos_data.get('CHOCH', 0) == -1
                return df
            except Exception:
                pass  # Silent fallback to custom implementation

        # Use custom implementation
        return self._detect_custom(df)

    def _detect_custom(self, df: pd.DataFrame) -> pd.DataFrame:
        """Custom BOS/CHoCH detection."""
        n = len(df)
        df['bos_bullish'] = False
        df['bos_bearish'] = False
        df['choch_bullish'] = False
        df['choch_bearish'] = False

        # Track market structure
        structure = 'ranging'
        last_swing_high = df['high'].iloc[0]
        last_swing_low = df['low'].iloc[0]

        for i in range(self.swing_length, n):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            close = df['close'].iloc[i]

            # Update swing levels
            period_high = df['high'].iloc[max(0, i-self.swing_length):i].max()
            period_low = df['low'].iloc[max(0, i-self.swing_length):i].min()

            # Bullish BOS: break above swing high in uptrend
            if close > last_swing_high:
                if structure == 'bullish':
                    df.loc[df.index[i], 'bos_bullish'] = True
                else:
                    df.loc[df.index[i], 'choch_bullish'] = True
                    structure = 'bullish'
                last_swing_high = current_high

            # Bearish BOS: break below swing low in downtrend
            if close < last_swing_low:
                if structure == 'bearish':
                    df.loc[df.index[i], 'bos_bearish'] = True
                else:
                    df.loc[df.index[i], 'choch_bearish'] = True
                    structure = 'bearish'
                last_swing_low = current_low

        return df


class KillZoneDetector:
    """
    Detect ICT Kill Zones - optimal trading sessions.

    Kill Zones are specific time windows with higher probability setups:
    - London: 02:00 - 05:00 ET
    - NY Open: 09:30 - 11:00 ET
    - Silver Bullet: 10:00 - 11:00 ET (specific reversal setup)
    """

    def __init__(self, config: Optional[SMCConfig] = None):
        self.config = config or SMCConfig()

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Kill Zone indicators.

        Args:
            df: DataFrame with datetime index or column

        Returns:
            DataFrame with kill zone columns
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Get datetime
        if isinstance(df.index, pd.DatetimeIndex):
            times = df.index.time
        elif 'timestamp' in df.columns:
            times = pd.to_datetime(df['timestamp']).dt.time
        elif 'date' in df.columns:
            # For daily data, kill zones don't apply
            df['in_london'] = False
            df['in_ny_open'] = False
            df['in_silver_bullet'] = False
            return df
        else:
            df['in_london'] = False
            df['in_ny_open'] = False
            df['in_silver_bullet'] = False
            return df

        # Detect kill zones
        df['in_london'] = [
            self.config.kz_london_start <= t <= self.config.kz_london_end
            for t in times
        ]

        df['in_ny_open'] = [
            self.config.kz_ny_start <= t <= self.config.kz_ny_end
            for t in times
        ]

        df['in_silver_bullet'] = [
            self.config.kz_silver_bullet_start <= t <= self.config.kz_silver_bullet_end
            for t in times
        ]

        return df


class SmartMoneyDetector:
    """
    Unified Smart Money Concepts detector.

    Combines all ICT pattern detection into a single interface.
    """

    def __init__(self, config: Optional[SMCConfig] = None):
        self.config = config or SMCConfig()

        # Initialize detectors
        self.ob_detector = OrderBlockDetector(
            swing_length=self.config.ob_swing_length,
            close_mitigation=self.config.ob_close_mitigation
        )
        self.fvg_detector = FairValueGapDetector(
            min_size_pct=self.config.fvg_min_size_pct
        )
        self.liq_detector = LiquiditySweepDetector(
            swing_length=self.config.liq_swing_length
        )
        self.bos_detector = BreakOfStructureDetector(
            swing_length=self.config.bos_swing_length
        )
        self.kz_detector = KillZoneDetector(config=self.config)

    def detect_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all Smart Money patterns.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all SMC columns added
        """
        df = df.copy()

        # Detect each pattern type
        df = self.ob_detector.detect(df)
        df = self.fvg_detector.detect(df)
        df = self.liq_detector.detect(df)
        df = self.bos_detector.detect(df)
        df = self.kz_detector.detect(df)

        # Add confluence signals
        df = self._add_confluence_signals(df)

        # Only log if high-probability signals detected (reduce noise)
        high_prob_count = df['smc_high_prob_long'].sum() + df['smc_high_prob_short'].sum()
        if high_prob_count > 0:
            jlog("smc_signals_found", level="INFO",
                 high_prob_long=int(df['smc_high_prob_long'].sum()),
                 high_prob_short=int(df['smc_high_prob_short'].sum()))

        return df

    def _add_confluence_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add confluence-based signals."""

        # Bullish confluence: Liquidity sweep low + bullish OB nearby + FVG
        df['smc_bullish_confluence'] = (
            df['liq_sweep_low'] &
            df['ob_bullish'].rolling(5).max().fillna(0).astype(bool)
        )

        # Bearish confluence: Liquidity sweep high + bearish OB nearby + FVG
        df['smc_bearish_confluence'] = (
            df['liq_sweep_high'] &
            df['ob_bearish'].rolling(5).max().fillna(0).astype(bool)
        )

        # High probability setups (all conditions met)
        df['smc_high_prob_long'] = (
            df['smc_bullish_confluence'] &
            df['choch_bullish'].rolling(10).max().fillna(0).astype(bool)
        )

        df['smc_high_prob_short'] = (
            df['smc_bearish_confluence'] &
            df['choch_bearish'].rolling(10).max().fillna(0).astype(bool)
        )

        return df

    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get trading signals from SMC patterns.

        Returns simplified signal DataFrame for integration with
        existing strategy scanners.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with signal columns
        """
        df = self.detect_all(df)

        # Create signal column
        df['smc_signal'] = 0
        df.loc[df['smc_high_prob_long'], 'smc_signal'] = 1
        df.loc[df['smc_high_prob_short'], 'smc_signal'] = -1

        return df[['smc_signal', 'smc_bullish_confluence', 'smc_bearish_confluence',
                   'ob_bullish', 'ob_bearish', 'fvg_bullish', 'fvg_bearish',
                   'liq_sweep_high', 'liq_sweep_low', 'choch_bullish', 'choch_bearish']]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def detect_smart_money_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect all Smart Money patterns.

    Convenience function for quick pattern detection.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with all SMC columns
    """
    detector = SmartMoneyDetector()
    return detector.detect_all(df)


def get_smc_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get SMC trading signals.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with signal columns
    """
    detector = SmartMoneyDetector()
    return detector.get_signals(df)
