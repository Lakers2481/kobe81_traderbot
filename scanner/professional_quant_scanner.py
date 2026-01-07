#!/usr/bin/env python3
"""
Professional Quant Scanner - The CORRECT Way to Scan for Trades

This scanner implements professional quant methodology:
1. Stock-specific historical pattern analysis (not aggregate stats)
2. Actual EV calculation for each signal
3. True oversold condition verification
4. Multi-asset support (stocks, crypto, options)
5. Ranking by Expected Value

A professional quant NEVER trades negative EV setups, regardless of "score".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QuantSignal:
    """Professional quant signal with full analysis."""
    symbol: str
    asset_class: str  # EQUITY, CRYPTO, OPTIONS
    strategy: str

    # Trade parameters
    entry_price: float
    stop_loss: float
    take_profit: float

    # Calculated metrics
    risk: float
    reward: float
    rr_ratio: float

    # Stock-specific historical stats (THE KEY DIFFERENTIATOR)
    pattern_win_rate: float  # Actual WR from THIS stock's history
    pattern_sample_size: int  # Number of historical instances
    pattern_avg_return: float  # Average next-day return

    # Current conditions
    ibs: float  # Internal Bar Strength (0-1)
    rsi: float  # RSI(14)
    is_oversold: bool  # IBS < 0.2 OR RSI < 30

    # Expected Value (THE DECISION METRIC)
    expected_value: float  # EV per $1 risked

    # Validation
    passes_quant_gate: bool
    rejection_reason: str = ""

    # Live data
    live_price: float = 0.0
    gap_pct: float = 0.0

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'asset_class': self.asset_class,
            'strategy': self.strategy,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'rr_ratio': round(self.rr_ratio, 2),
            'pattern_win_rate': round(self.pattern_win_rate, 4),
            'pattern_sample_size': self.pattern_sample_size,
            'ibs': round(self.ibs, 2),
            'rsi': round(self.rsi, 1),
            'is_oversold': self.is_oversold,
            'expected_value': round(self.expected_value, 4),
            'passes_quant_gate': self.passes_quant_gate,
            'rejection_reason': self.rejection_reason,
            'live_price': self.live_price,
            'gap_pct': round(self.gap_pct, 4),
        }


class ProfessionalQuantScanner:
    """
    Professional quant scanner implementing institutional-grade methodology.

    Key principles:
    1. Stock-specific analysis (not aggregate strategy stats)
    2. EV-based decision making
    3. Verify actual oversold conditions
    4. Multi-asset support
    5. Capital preservation first
    """

    # Quant gate thresholds
    MIN_SAMPLE_SIZE = 20  # Need enough data for statistical significance
    MIN_WIN_RATE = 0.45  # Below this, pattern is unreliable
    MIN_EV = 0.01  # Must be positive (we want > 0, but allow small positive)
    MAX_GAP_PCT = 0.03  # 3% maximum gap

    # Oversold thresholds
    IBS_OVERSOLD = 0.20  # IBS below this = oversold
    RSI_OVERSOLD = 30.0  # RSI below this = oversold

    def __init__(self):
        self.signals: List[QuantSignal] = []

    def calculate_stock_specific_stats(
        self,
        df: pd.DataFrame,
        pattern_type: str = 'turtle_soup'
    ) -> Tuple[float, int, float]:
        """
        Calculate historical win rate for THIS SPECIFIC stock's pattern.

        This is the KEY difference from naive scanning.
        We don't use aggregate strategy stats (61% WR).
        We calculate the ACTUAL win rate for this stock.

        Returns:
            (win_rate, sample_size, avg_return)
        """
        if len(df) < 50:
            return 0.5, 0, 0.0

        df = df.copy()
        df['next_day_return'] = df['close'].shift(-1) / df['close'] - 1

        if pattern_type == 'turtle_soup':
            # Find 20-day low sweeps
            df['low_20'] = df['low'].rolling(20).min().shift(1)
            df['swept_low'] = df['low'] < df['low_20']
            pattern_instances = df[df['swept_low'] == True]

        elif pattern_type == 'ibs_rsi':
            # Find IBS < 0.08 AND RSI < 5
            df['ibs'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(2).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
            rs = gain / (loss + 1e-10)
            df['rsi_2'] = 100 - (100 / (1 + rs))

            pattern_instances = df[(df['ibs'] < 0.08) & (df['rsi_2'] < 5)]
        else:
            return 0.5, 0, 0.0

        if len(pattern_instances) < 5:
            return 0.5, len(pattern_instances), 0.0

        # Calculate stats
        wins = (pattern_instances['next_day_return'] > 0).sum()
        total = len(pattern_instances)
        win_rate = wins / total if total > 0 else 0.5
        avg_return = pattern_instances['next_day_return'].mean() * 100

        return win_rate, total, avg_return

    def calculate_expected_value(self, win_rate: float, rr_ratio: float) -> float:
        """
        Calculate Expected Value per $1 risked.

        EV = (WR × Reward) - ((1-WR) × Risk)

        With R:R = reward/risk, and risk normalized to 1:
        EV = (WR × RR) - ((1-WR) × 1)
        """
        return (win_rate * rr_ratio) - ((1 - win_rate) * 1.0)

    def is_actually_oversold(self, df: pd.DataFrame) -> Tuple[bool, float, float]:
        """
        Check if stock is ACTUALLY oversold, not just made a technical low.

        Professional quants verify the exhaustion conditions, not just price levels.

        Returns:
            (is_oversold, ibs, rsi)
        """
        if len(df) < 14:
            return False, 0.5, 50.0

        # IBS (Internal Bar Strength)
        last = df.iloc[-1]
        ibs = (last['close'] - last['low']) / (last['high'] - last['low']) if last['high'] != last['low'] else 0.5

        # RSI(14)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        is_oversold = (ibs < self.IBS_OVERSOLD) or (rsi < self.RSI_OVERSOLD)

        return is_oversold, ibs, rsi

    def analyze_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        entry: float,
        stop: float,
        target: float,
        strategy: str,
        asset_class: str = 'EQUITY',
        live_price: float = None,
    ) -> Optional[QuantSignal]:
        """
        Full quant analysis of a potential signal.

        This is where we apply professional methodology:
        1. Calculate stock-specific historical stats
        2. Verify oversold conditions
        3. Calculate expected value
        4. Apply quant gate
        """
        # Calculate trade parameters
        risk = entry - stop
        reward = target - entry
        rr_ratio = reward / risk if risk > 0 else 0

        # Get stock-specific historical stats
        pattern_type = 'turtle_soup' if 'Turtle' in strategy else 'ibs_rsi'
        win_rate, sample_size, avg_return = self.calculate_stock_specific_stats(df, pattern_type)

        # Check oversold conditions
        is_oversold, ibs, rsi = self.is_actually_oversold(df)

        # Calculate Expected Value
        ev = self.calculate_expected_value(win_rate, rr_ratio)

        # Calculate gap if live price available
        gap_pct = 0.0
        if live_price and live_price > 0 and entry > 0:
            gap_pct = (live_price / entry) - 1

        # Apply quant gate
        passes_gate = True
        rejection_reason = ""

        if sample_size < self.MIN_SAMPLE_SIZE:
            passes_gate = False
            rejection_reason = f"Insufficient samples ({sample_size} < {self.MIN_SAMPLE_SIZE})"
        elif win_rate < self.MIN_WIN_RATE:
            passes_gate = False
            rejection_reason = f"Low win rate ({win_rate*100:.1f}% < {self.MIN_WIN_RATE*100}%)"
        elif ev < self.MIN_EV:
            passes_gate = False
            rejection_reason = f"Negative EV ({ev:.4f} < {self.MIN_EV})"
        elif not is_oversold:
            passes_gate = False
            rejection_reason = f"Not oversold (IBS={ibs:.2f}, RSI={rsi:.1f})"
        elif abs(gap_pct) > self.MAX_GAP_PCT:
            passes_gate = False
            rejection_reason = f"Gap too large ({gap_pct*100:.1f}% > {self.MAX_GAP_PCT*100}%)"

        return QuantSignal(
            symbol=symbol,
            asset_class=asset_class,
            strategy=strategy,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            risk=risk,
            reward=reward,
            rr_ratio=rr_ratio,
            pattern_win_rate=win_rate,
            pattern_sample_size=sample_size,
            pattern_avg_return=avg_return,
            ibs=ibs,
            rsi=rsi,
            is_oversold=is_oversold,
            expected_value=ev,
            passes_quant_gate=passes_gate,
            rejection_reason=rejection_reason,
            live_price=live_price or entry,
            gap_pct=gap_pct,
        )

    def rank_signals(self, signals: List[QuantSignal]) -> List[QuantSignal]:
        """
        Rank signals by Expected Value (the professional way).

        NOT by arbitrary "score" - by actual expected profit per dollar risked.
        """
        # Filter to only passing signals
        passing = [s for s in signals if s.passes_quant_gate]

        # Sort by EV descending
        passing.sort(key=lambda x: x.expected_value, reverse=True)

        return passing


def run_professional_scan(
    symbols: List[str],
    fetch_eod_func,
    fetch_intraday_func,
    scanner_func,
    end_date: str = None,
    verbose: bool = True,
) -> List[QuantSignal]:
    """
    Run professional quant scan on a universe of stocks.

    Args:
        symbols: List of stock symbols
        fetch_eod_func: Function to fetch EOD data
        fetch_intraday_func: Function to fetch intraday data
        scanner_func: DualStrategyScanner to generate raw signals
        end_date: End date for data
        verbose: Print progress

    Returns:
        List of QuantSignal objects, ranked by EV
    """
    quant = ProfessionalQuantScanner()
    all_signals = []

    if verbose:
        print(f"Scanning {len(symbols)} symbols with professional quant methodology...")
        print()

    for i, symbol in enumerate(symbols):
        if verbose and (i + 1) % 100 == 0:
            passing = len([s for s in all_signals if s.passes_quant_gate])
            print(f"  Progress: {i+1}/{len(symbols)} | Signals: {len(all_signals)} | Passing: {passing}")

        try:
            # Fetch EOD data
            # FIX (2026-01-07): Use dynamic start date (1 year ago), not hardcoded 2024
            from datetime import datetime, timedelta
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            df = fetch_eod_func(symbol, start=start_date, end=end_date, cache_dir=None)
            if df is None or len(df) < 50:
                continue

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Generate raw signal from scanner
            sig = scanner_func.generate_signals(df)
            if sig.empty:
                continue

            r = sig.iloc[0]
            entry = float(r.get('entry_price', 0))
            stop = float(r.get('stop_loss', 0))
            target = float(r.get('take_profit', 0)) if pd.notna(r.get('take_profit')) else entry + (entry - stop)
            strategy = r.get('strategy', 'Unknown')

            if entry <= 0 or stop <= 0:
                continue

            # Get live price for gap check
            try:
                bars = fetch_intraday_func(symbol, timeframe='15Min', limit=3)
                live_price = bars[-1].close if bars else entry
            except Exception:
                live_price = entry

            # Full quant analysis
            quant_signal = quant.analyze_signal(
                symbol=symbol,
                df=df,
                entry=entry,
                stop=stop,
                target=target,
                strategy=strategy,
                asset_class='EQUITY',
                live_price=live_price,
            )

            if quant_signal:
                all_signals.append(quant_signal)

        except Exception as e:
            logger.debug(f"Error scanning {symbol}: {e}")
            continue

    # Rank by EV
    ranked = quant.rank_signals(all_signals)

    if verbose:
        passing = len(ranked)
        total = len(all_signals)
        print()
        print(f"Scan complete: {total} signals, {passing} pass quant gate")

    return ranked, all_signals
