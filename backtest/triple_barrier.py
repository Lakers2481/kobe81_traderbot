"""
Triple Barrier Method for Trade Labeling.

Implements Marcos Lopez de Prado's Triple Barrier Method from
"Advances in Financial Machine Learning" (2018).

The triple barrier method labels trades based on three possible outcomes:
1. Upper barrier: Profit target hit (label = +1)
2. Lower barrier: Stop loss hit (label = -1)
3. Vertical barrier: Time limit reached (label = 0 or based on return sign)

Advantages over fixed labels:
- Path-dependent: considers price path, not just final price
- Adaptive barriers: can use volatility-based stops
- Handles asymmetric risk/reward
- Better for ML training (more balanced labels)

Usage:
    from backtest.triple_barrier import TripleBarrierLabeler

    labeler = TripleBarrierLabeler()
    labels = labeler.label_trades(df, events)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict, Union, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.structured_log import jlog


@dataclass
class TripleBarrierConfig:
    """Configuration for triple barrier labeling."""

    # Barrier widths (as multiples of volatility)
    pt_sl_ratio: float = 1.0  # Profit take / stop loss ratio (1.0 = symmetric)
    upper_barrier_mult: float = 2.0  # Upper barrier = volatility * mult
    lower_barrier_mult: float = 2.0  # Lower barrier = volatility * mult

    # Vertical barrier (maximum holding period)
    max_holding_period: int = 7  # Days (or bars)

    # Volatility estimation
    volatility_lookback: int = 20  # Days for volatility calculation
    volatility_method: str = "atr"  # "atr", "std", "parkinson"

    # Labeling
    label_on_vertical: str = "sign"  # "sign", "zero", "none"
    min_return_threshold: float = 0.001  # Minimum return to assign non-zero label

    # Meta-labeling support
    enable_meta_labels: bool = False  # If True, return size labels (0, 1) instead of direction


class TripleBarrierLabeler:
    """
    Triple Barrier Method labeler.

    Creates trade labels based on which barrier is touched first:
    - Upper barrier (+1): Price rises by target amount
    - Lower barrier (-1): Price falls by stop amount
    - Vertical barrier (0 or sign): Time expires

    This replaces fixed TP/SL with adaptive, volatility-scaled barriers.
    """

    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        self.config = config or TripleBarrierConfig()

    def estimate_volatility(
        self,
        df: pd.DataFrame,
        method: Optional[str] = None,
        lookback: Optional[int] = None
    ) -> pd.Series:
        """
        Estimate daily volatility for barrier sizing.

        Args:
            df: DataFrame with OHLCV columns
            method: Volatility method (atr, std, parkinson)
            lookback: Lookback period

        Returns:
            Series of volatility estimates
        """
        method = method or self.config.volatility_method
        lookback = lookback or self.config.volatility_lookback

        df = df.copy()
        df.columns = df.columns.str.lower()

        if method == "atr":
            # Average True Range
            high = df['high']
            low = df['low']
            close = df['close']

            tr = pd.DataFrame({
                'hl': high - low,
                'hc': (high - close.shift(1)).abs(),
                'lc': (low - close.shift(1)).abs()
            }).max(axis=1)

            volatility = tr.rolling(lookback).mean()

        elif method == "std":
            # Standard deviation of returns
            returns = df['close'].pct_change()
            volatility = returns.rolling(lookback).std() * df['close']

        elif method == "parkinson":
            # Parkinson volatility (uses high-low range)
            high = df['high']
            low = df['low']
            hl_ratio = np.log(high / low)
            volatility = np.sqrt(hl_ratio ** 2 / (4 * np.log(2)))
            volatility = volatility.rolling(lookback).mean() * df['close']

        else:
            raise ValueError(f"Unknown volatility method: {method}")

        return volatility

    def get_barriers(
        self,
        entry_price: float,
        entry_idx: int,
        volatility: float,
        side: int = 1
    ) -> Tuple[float, float, int]:
        """
        Calculate barrier levels for a trade.

        Args:
            entry_price: Entry price
            entry_idx: Entry bar index
            volatility: Current volatility estimate
            side: Trade direction (1 = long, -1 = short)

        Returns:
            (upper_barrier, lower_barrier, vertical_barrier_idx)
        """
        # Upper barrier (profit target)
        upper_barrier = entry_price + side * volatility * self.config.upper_barrier_mult

        # Lower barrier (stop loss)
        lower_barrier = entry_price - side * volatility * self.config.lower_barrier_mult

        # Vertical barrier (time limit)
        vertical_barrier = entry_idx + self.config.max_holding_period

        return upper_barrier, lower_barrier, vertical_barrier

    def apply_triple_barrier(
        self,
        prices: pd.Series,
        entry_idx: int,
        entry_price: float,
        upper_barrier: float,
        lower_barrier: float,
        vertical_barrier: int,
        side: int = 1
    ) -> Dict[str, Union[int, float, str]]:
        """
        Apply triple barrier to a single trade.

        Args:
            prices: Price series (close prices)
            entry_idx: Entry bar index
            entry_price: Entry price
            upper_barrier: Upper barrier price
            lower_barrier: Lower barrier price
            vertical_barrier: Vertical barrier index
            side: Trade direction (1 = long, -1 = short)

        Returns:
            Dict with exit info: label, exit_idx, exit_price, barrier_type
        """
        # Limit to barrier window
        max_idx = min(vertical_barrier, len(prices) - 1)

        if entry_idx >= len(prices):
            return {
                'label': 0,
                'exit_idx': entry_idx,
                'exit_price': entry_price,
                'barrier_type': 'error',
                'return': 0.0
            }

        # Scan forward from entry
        for i in range(entry_idx + 1, max_idx + 1):
            current_price = prices.iloc[i]

            # Check barriers based on side
            if side == 1:  # Long position
                if current_price >= upper_barrier:
                    ret = (current_price - entry_price) / entry_price
                    return {
                        'label': 1,
                        'exit_idx': i,
                        'exit_price': current_price,
                        'barrier_type': 'upper',
                        'return': ret
                    }
                elif current_price <= lower_barrier:
                    ret = (current_price - entry_price) / entry_price
                    return {
                        'label': -1,
                        'exit_idx': i,
                        'exit_price': current_price,
                        'barrier_type': 'lower',
                        'return': ret
                    }
            else:  # Short position
                if current_price <= upper_barrier:
                    ret = (entry_price - current_price) / entry_price
                    return {
                        'label': 1,
                        'exit_idx': i,
                        'exit_price': current_price,
                        'barrier_type': 'upper',
                        'return': ret
                    }
                elif current_price >= lower_barrier:
                    ret = (entry_price - current_price) / entry_price
                    return {
                        'label': -1,
                        'exit_idx': i,
                        'exit_price': current_price,
                        'barrier_type': 'lower',
                        'return': ret
                    }

        # Vertical barrier reached
        exit_price = prices.iloc[max_idx]
        if side == 1:
            ret = (exit_price - entry_price) / entry_price
        else:
            ret = (entry_price - exit_price) / entry_price

        # Label on vertical barrier
        if self.config.label_on_vertical == "sign":
            if ret > self.config.min_return_threshold:
                label = 1
            elif ret < -self.config.min_return_threshold:
                label = -1
            else:
                label = 0
        elif self.config.label_on_vertical == "zero":
            label = 0
        else:
            label = 0

        return {
            'label': label,
            'exit_idx': max_idx,
            'exit_price': exit_price,
            'barrier_type': 'vertical',
            'return': ret
        }

    def label_trades(
        self,
        df: pd.DataFrame,
        events: pd.DataFrame,
        side_column: str = 'side'
    ) -> pd.DataFrame:
        """
        Label all trades using triple barrier method.

        Args:
            df: DataFrame with OHLCV data (must have DatetimeIndex or 'date' column)
            events: DataFrame with trade events (entry_idx, entry_price, side)
            side_column: Column name for trade direction

        Returns:
            DataFrame with labels added to events
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Ensure close prices available
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")

        # Estimate volatility
        volatility = self.estimate_volatility(df)

        # Process each event
        results = []
        for idx, event in events.iterrows():
            entry_idx = event.get('entry_idx', idx)
            if isinstance(entry_idx, pd.Timestamp):
                entry_idx = df.index.get_loc(entry_idx)

            entry_price = event.get('entry_price', df['close'].iloc[entry_idx])
            side = event.get(side_column, 1)
            if isinstance(side, str):
                side = 1 if side.lower() in ['long', 'buy'] else -1

            # Get current volatility
            vol = volatility.iloc[entry_idx] if entry_idx < len(volatility) else volatility.iloc[-1]
            if pd.isna(vol):
                vol = df['close'].pct_change().std() * df['close'].iloc[entry_idx] * np.sqrt(20)

            # Calculate barriers
            upper, lower, vertical = self.get_barriers(entry_price, entry_idx, vol, side)

            # Apply barriers
            result = self.apply_triple_barrier(
                df['close'],
                entry_idx,
                entry_price,
                upper,
                lower,
                vertical,
                side
            )

            result['event_idx'] = idx
            result['entry_idx'] = entry_idx
            result['entry_price'] = entry_price
            result['upper_barrier'] = upper
            result['lower_barrier'] = lower
            result['vertical_barrier'] = vertical
            result['volatility'] = vol

            results.append(result)

        labels_df = pd.DataFrame(results)

        # Meta-labeling: convert direction labels to bet size labels
        if self.config.enable_meta_labels:
            # 1 = take the bet, 0 = skip the bet
            labels_df['meta_label'] = (labels_df['label'] == 1).astype(int)

        jlog("triple_barrier_labeled", level="DEBUG",
             n_trades=len(labels_df),
             upper_hits=(labels_df['barrier_type'] == 'upper').sum(),
             lower_hits=(labels_df['barrier_type'] == 'lower').sum(),
             vertical_hits=(labels_df['barrier_type'] == 'vertical').sum())

        return labels_df

    def label_from_signals(
        self,
        df: pd.DataFrame,
        signal_column: str = 'signal'
    ) -> pd.DataFrame:
        """
        Label trades directly from a signal column.

        Args:
            df: DataFrame with OHLCV and signal column
            signal_column: Column containing signals (1=buy, -1=sell, 0=hold)

        Returns:
            DataFrame with labels for each signal
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Extract signal events
        signals = df[df[signal_column] != 0].copy()

        if signals.empty:
            return pd.DataFrame()

        # Create events DataFrame
        events = pd.DataFrame({
            'entry_idx': range(len(signals)),
            'entry_price': signals['close'].values,
            'side': signals[signal_column].values
        }, index=signals.index)

        # Reset index to numeric for barrier calculation
        df_reset = df.reset_index(drop=True)

        # Map signals to numeric index
        events['entry_idx'] = [df_reset.index[df.index.get_loc(idx)] for idx in events.index]

        return self.label_trades(df_reset, events)


class MetaLabeler:
    """
    Meta-Labeling for bet sizing.

    Lopez de Prado's meta-labeling approach:
    1. Primary model generates directional signals (buy/sell)
    2. Secondary model (meta-labeler) decides bet size (0 to 1)

    This separates:
    - Signal quality (is this a good time to trade?)
    - Direction (if trading, which direction?)

    Benefits:
    - Better probability calibration
    - More conservative sizing on low-confidence signals
    - Works with any primary model
    """

    def __init__(self, barrier_config: Optional[TripleBarrierConfig] = None):
        config = barrier_config or TripleBarrierConfig()
        config.enable_meta_labels = True
        self.labeler = TripleBarrierLabeler(config)

    def create_meta_labels(
        self,
        df: pd.DataFrame,
        primary_signals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create meta-labels for primary model signals.

        Args:
            df: DataFrame with OHLCV data
            primary_signals: DataFrame with primary model signals

        Returns:
            DataFrame with meta-labels (0 = skip, 1 = take)
        """
        # Label using triple barrier
        labels = self.labeler.label_trades(df, primary_signals)

        # Meta-label is 1 if trade was profitable (label == 1)
        labels['meta_label'] = (labels['label'] == 1).astype(int)

        return labels

    def calculate_sample_weights(
        self,
        labels: pd.DataFrame,
        decay: float = 0.5
    ) -> pd.Series:
        """
        Calculate sample weights for overlapping labels.

        When trade holding periods overlap, samples are not independent.
        Weight samples by inverse of overlap count.

        Args:
            labels: DataFrame with entry_idx, exit_idx
            decay: Decay factor for distant overlaps

        Returns:
            Series of sample weights
        """
        n = len(labels)
        weights = np.ones(n)

        for i, row_i in labels.iterrows():
            entry_i = row_i['entry_idx']
            exit_i = row_i['exit_idx']

            for j, row_j in labels.iterrows():
                if i >= j:
                    continue

                entry_j = row_j['entry_idx']
                exit_j = row_j['exit_idx']

                # Check for overlap
                overlap_start = max(entry_i, entry_j)
                overlap_end = min(exit_i, exit_j)

                if overlap_start < overlap_end:
                    # Reduce weight for overlapping samples
                    overlap_length = overlap_end - overlap_start
                    trade_length = max(exit_i - entry_i, exit_j - entry_j, 1)
                    overlap_ratio = overlap_length / trade_length

                    weights[i] *= (1 - overlap_ratio * decay)
                    weights[j] *= (1 - overlap_ratio * decay)

        # Normalize weights
        weights = weights / weights.sum() * n

        return pd.Series(weights, index=labels.index)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def label_with_triple_barrier(
    df: pd.DataFrame,
    signal_column: str = 'signal',
    pt_sl_ratio: float = 1.0,
    max_holding_period: int = 7,
    volatility_lookback: int = 20
) -> pd.DataFrame:
    """
    Label trades using triple barrier method.

    Convenience function with common defaults.

    Args:
        df: DataFrame with OHLCV and signal column
        signal_column: Column containing signals
        pt_sl_ratio: Profit target / stop loss ratio
        max_holding_period: Maximum holding period in bars
        volatility_lookback: Lookback for volatility estimation

    Returns:
        DataFrame with labels
    """
    config = TripleBarrierConfig(
        pt_sl_ratio=pt_sl_ratio,
        max_holding_period=max_holding_period,
        volatility_lookback=volatility_lookback
    )

    labeler = TripleBarrierLabeler(config)
    return labeler.label_from_signals(df, signal_column)


def add_triple_barrier_labels(
    df: pd.DataFrame,
    upper_mult: float = 2.0,
    lower_mult: float = 2.0,
    max_bars: int = 7
) -> pd.DataFrame:
    """
    Add triple barrier labels to every bar in DataFrame.

    Labels each bar with forward-looking outcome.
    Useful for supervised ML training.

    Args:
        df: DataFrame with OHLCV data
        upper_mult: Upper barrier as multiple of volatility
        lower_mult: Lower barrier as multiple of volatility
        max_bars: Maximum forward look period

    Returns:
        DataFrame with 'tb_label' column added
    """
    config = TripleBarrierConfig(
        upper_barrier_mult=upper_mult,
        lower_barrier_mult=lower_mult,
        max_holding_period=max_bars
    )

    labeler = TripleBarrierLabeler(config)

    # Create events for every bar
    df = df.copy()
    df.columns = df.columns.str.lower()

    events = pd.DataFrame({
        'entry_idx': range(len(df)),
        'entry_price': df['close'].values,
        'side': 1  # Assume long for labeling
    })

    # Get labels
    labels = labeler.label_trades(df, events)

    # Merge back
    df['tb_label'] = labels['label'].values
    df['tb_barrier'] = labels['barrier_type'].values
    df['tb_return'] = labels['return'].values

    return df
