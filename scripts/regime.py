#!/usr/bin/env python3
"""
Market Regime Detection Script for Kobe Trading System

Detects bull/bear/chop market regimes using:
- VIX levels and trends
- SPY trend analysis (moving averages, momentum)
- Market breadth indicators

Usage:
    python regime.py --dotenv /path/to/.env
    python regime.py --history --days 60
    python regime.py --chart-data
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.env_loader import load_env


@dataclass
class RegimeResult:
    """Container for regime detection results."""
    regime: str  # 'bull', 'bear', 'chop'
    confidence: float  # 0.0 to 1.0
    vix_level: float
    vix_regime: str
    spy_trend: str
    breadth_signal: str
    timestamp: str
    signals: Dict[str, float]


def fetch_polygon_bars(symbol: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Fetch daily bars from Polygon API."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        results = data.get('results', [])
        if not results:
            return pd.DataFrame()
        rows = []
        for r in results:
            rows.append({
                'timestamp': pd.to_datetime(r.get('t'), unit='ms'),
                'open': float(r.get('o', 0)),
                'high': float(r.get('h', 0)),
                'low': float(r.get('l', 0)),
                'close': float(r.get('c', 0)),
                'volume': float(r.get('v', 0)),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=window, min_periods=1).mean()


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ADX (Average Directional Index) for trend strength."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr = calculate_atr(high, low, close, period)
    plus_di = 100 * calculate_ema(plus_dm, period) / atr
    minus_di = 100 * calculate_ema(minus_dm, period) / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = calculate_ema(dx, period)
    return adx


class RegimeDetector:
    """Market regime detection engine."""

    # VIX thresholds
    VIX_LOW = 15.0      # Complacent/bullish
    VIX_MEDIUM = 20.0   # Normal
    VIX_HIGH = 25.0     # Elevated fear
    VIX_EXTREME = 30.0  # Extreme fear/crisis

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.spy_data: Optional[pd.DataFrame] = None
        self.vix_data: Optional[pd.DataFrame] = None

    def fetch_data(self, lookback_days: int = 200) -> bool:
        """Fetch SPY and VIX data."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_days + 50)).strftime('%Y-%m-%d')

        print(f"Fetching SPY data from {start_date} to {end_date}...")
        self.spy_data = fetch_polygon_bars('SPY', start_date, end_date, self.api_key)

        # VIX is available as VIXY (VIX ETF) or we estimate from SPY volatility
        print("Fetching VIX proxy data...")
        self.vix_data = fetch_polygon_bars('VIXY', start_date, end_date, self.api_key)

        if self.spy_data.empty:
            print("Warning: Could not fetch SPY data")
            return False

        return True

    def calculate_implied_vix(self) -> pd.Series:
        """
        Calculate implied VIX from SPY realized volatility.
        VIX approximation: 21-day realized vol annualized * scaling factor.
        """
        if self.spy_data is None or self.spy_data.empty:
            return pd.Series([20.0])

        returns = self.spy_data['close'].pct_change()
        realized_vol = returns.rolling(window=21).std() * np.sqrt(252) * 100
        # VIX is typically higher than realized vol
        implied_vix = realized_vol * 1.15
        return implied_vix.fillna(20.0)

    def get_vix_level(self) -> Tuple[float, str]:
        """Get current VIX level and regime classification."""
        if self.vix_data is not None and not self.vix_data.empty:
            # Use VIXY as proxy, scale appropriately
            vixy_price = self.vix_data['close'].iloc[-1]
            # Rough approximation: VIXY tracks VIX short-term futures
            vix_approx = vixy_price * 1.5  # Scaling factor
        else:
            # Fall back to implied VIX
            implied = self.calculate_implied_vix()
            vix_approx = implied.iloc[-1] if len(implied) > 0 else 20.0

        if vix_approx < self.VIX_LOW:
            regime = 'low_vol'
        elif vix_approx < self.VIX_MEDIUM:
            regime = 'normal'
        elif vix_approx < self.VIX_HIGH:
            regime = 'elevated'
        elif vix_approx < self.VIX_EXTREME:
            regime = 'high'
        else:
            regime = 'extreme'

        return float(vix_approx), regime

    def analyze_spy_trend(self) -> Tuple[str, Dict[str, float]]:
        """
        Analyze SPY trend using multiple indicators.
        Returns trend direction and signal scores.
        """
        if self.spy_data is None or len(self.spy_data) < 50:
            return 'unknown', {}

        close = self.spy_data['close']
        high = self.spy_data['high']
        low = self.spy_data['low']

        signals = {}

        # Moving average signals
        sma_20 = calculate_sma(close, 20).iloc[-1]
        sma_50 = calculate_sma(close, 50).iloc[-1]
        sma_200 = calculate_sma(close, 200).iloc[-1] if len(close) >= 200 else sma_50
        current_price = close.iloc[-1]

        # Price vs MAs
        signals['price_vs_sma20'] = 1.0 if current_price > sma_20 else -1.0
        signals['price_vs_sma50'] = 1.0 if current_price > sma_50 else -1.0
        signals['price_vs_sma200'] = 1.0 if current_price > sma_200 else -1.0

        # MA alignment (golden/death cross proximity)
        signals['sma20_vs_sma50'] = 1.0 if sma_20 > sma_50 else -1.0
        signals['sma50_vs_sma200'] = 1.0 if sma_50 > sma_200 else -1.0

        # Momentum (ROC)
        roc_20 = (current_price / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0
        signals['momentum_20d'] = np.clip(roc_20 / 5, -1, 1)  # Normalize

        # RSI
        rsi = calculate_rsi(close, 14).iloc[-1]
        if rsi > 70:
            signals['rsi'] = 0.5  # Overbought but still bullish
        elif rsi > 50:
            signals['rsi'] = 1.0
        elif rsi > 30:
            signals['rsi'] = -0.5
        else:
            signals['rsi'] = -1.0

        # ADX for trend strength
        adx = calculate_adx(high, low, close, 14).iloc[-1]
        signals['trend_strength'] = min(adx / 25, 1.5) if not np.isnan(adx) else 0.5

        # Higher highs / lower lows
        recent_high = high.iloc[-20:].max()
        recent_low = low.iloc[-20:].min()
        prev_high = high.iloc[-40:-20].max() if len(high) >= 40 else recent_high
        prev_low = low.iloc[-40:-20].min() if len(low) >= 40 else recent_low

        if recent_high > prev_high and recent_low > prev_low:
            signals['structure'] = 1.0  # Higher highs, higher lows
        elif recent_high < prev_high and recent_low < prev_low:
            signals['structure'] = -1.0  # Lower highs, lower lows
        else:
            signals['structure'] = 0.0  # Mixed/choppy

        # Aggregate trend score
        weights = {
            'price_vs_sma20': 0.10,
            'price_vs_sma50': 0.15,
            'price_vs_sma200': 0.20,
            'sma20_vs_sma50': 0.10,
            'sma50_vs_sma200': 0.15,
            'momentum_20d': 0.10,
            'rsi': 0.05,
            'structure': 0.15,
        }

        trend_score = sum(signals.get(k, 0) * v for k, v in weights.items())

        # Adjust by trend strength
        trend_score *= signals.get('trend_strength', 1.0)

        if trend_score > 0.3:
            trend = 'bullish'
        elif trend_score < -0.3:
            trend = 'bearish'
        else:
            trend = 'neutral'

        signals['aggregate_score'] = float(trend_score)

        return trend, signals

    def analyze_breadth(self) -> Tuple[str, float]:
        """
        Analyze market breadth using SPY internals approximation.
        Without access to advance/decline data, we use SPY volatility patterns.
        """
        if self.spy_data is None or len(self.spy_data) < 20:
            return 'neutral', 0.5

        close = self.spy_data['close']
        high = self.spy_data['high']
        low = self.spy_data['low']

        # Use intraday range vs average as breadth proxy
        daily_range = (high - low) / close
        avg_range = daily_range.rolling(20).mean()

        # Consistent small ranges = healthy trend
        # Expanding ranges = volatility/uncertainty
        recent_range = daily_range.iloc[-5:].mean()
        avg_range_val = avg_range.iloc[-1]

        range_ratio = recent_range / avg_range_val if avg_range_val > 0 else 1.0

        # Up vs down volume approximation (close-to-close direction)
        returns = close.pct_change().iloc[-20:]
        up_days = (returns > 0).sum()
        down_days = (returns < 0).sum()

        up_ratio = up_days / (up_days + down_days) if (up_days + down_days) > 0 else 0.5

        # Combine signals
        breadth_score = up_ratio

        if range_ratio > 1.5:
            # High volatility reduces breadth confidence
            breadth_score *= 0.7

        if breadth_score > 0.6:
            return 'positive', float(breadth_score)
        elif breadth_score < 0.4:
            return 'negative', float(breadth_score)
        else:
            return 'neutral', float(breadth_score)

    def detect_regime(self) -> RegimeResult:
        """Main regime detection combining all signals."""
        vix_level, vix_regime = self.get_vix_level()
        spy_trend, trend_signals = self.analyze_spy_trend()
        breadth_signal, breadth_score = self.analyze_breadth()

        # Regime scoring
        bull_score = 0.0
        bear_score = 0.0
        chop_score = 0.0

        # VIX contribution
        if vix_regime in ('low_vol', 'normal'):
            bull_score += 0.2
        elif vix_regime in ('elevated', 'high'):
            bear_score += 0.15
            chop_score += 0.1
        else:  # extreme
            bear_score += 0.25

        # Trend contribution
        agg_score = trend_signals.get('aggregate_score', 0)
        if spy_trend == 'bullish':
            bull_score += 0.35 * (1 + min(agg_score, 1))
        elif spy_trend == 'bearish':
            bear_score += 0.35 * (1 + min(abs(agg_score), 1))
        else:
            chop_score += 0.3

        # Breadth contribution
        if breadth_signal == 'positive':
            bull_score += 0.15 * breadth_score
        elif breadth_signal == 'negative':
            bear_score += 0.15 * (1 - breadth_score)
        else:
            chop_score += 0.1

        # Trend strength affects chop score
        trend_strength = trend_signals.get('trend_strength', 0.5)
        if trend_strength < 0.6:
            chop_score += 0.2

        # Determine regime
        scores = {'bull': bull_score, 'bear': bear_score, 'chop': chop_score}
        regime = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[regime] / total_score if total_score > 0 else 0.33

        return RegimeResult(
            regime=regime,
            confidence=round(confidence, 3),
            vix_level=round(vix_level, 2),
            vix_regime=vix_regime,
            spy_trend=spy_trend,
            breadth_signal=breadth_signal,
            timestamp=datetime.now().isoformat(),
            signals={k: round(v, 4) for k, v in trend_signals.items()}
        )

    def get_regime_history(self, days: int = 60) -> pd.DataFrame:
        """Calculate historical regime data for charting."""
        if self.spy_data is None or len(self.spy_data) < days:
            return pd.DataFrame()

        history = []
        close = self.spy_data['close']
        high = self.spy_data['high']
        low = self.spy_data['low']

        for i in range(days, 0, -1):
            idx = len(self.spy_data) - i
            if idx < 50:
                continue

            # Calculate signals at this point
            subset_close = close.iloc[:idx+1]
            high.iloc[:idx+1]
            low.iloc[:idx+1]

            sma_20 = calculate_sma(subset_close, 20).iloc[-1]
            sma_50 = calculate_sma(subset_close, 50).iloc[-1]
            current = subset_close.iloc[-1]

            # Simplified regime score
            score = 0
            if current > sma_20:
                score += 1
            if current > sma_50:
                score += 1
            if sma_20 > sma_50:
                score += 1

            if score >= 2:
                regime = 'bull'
            elif score <= 0:
                regime = 'bear'
            else:
                regime = 'chop'

            history.append({
                'date': self.spy_data['timestamp'].iloc[idx],
                'close': current,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'regime': regime,
                'score': score
            })

        return pd.DataFrame(history)


def print_regime_report(result: RegimeResult) -> None:
    """Print formatted regime report."""
    print("\n" + "=" * 60)
    print("         MARKET REGIME DETECTION REPORT")
    print("=" * 60)
    print(f"  Timestamp: {result.timestamp}")
    print("-" * 60)

    # Main regime with visual indicator
    regime_icons = {'bull': '[BULL]', 'bear': '[BEAR]', 'chop': '[CHOP]'}
    regime_colors = {'bull': 'Bullish', 'bear': 'Bearish', 'chop': 'Choppy/Range-bound'}

    print(f"\n  CURRENT REGIME: {regime_icons.get(result.regime, '?')} {regime_colors.get(result.regime, 'Unknown')}")
    print(f"  CONFIDENCE:     {result.confidence:.1%}")

    print("\n" + "-" * 60)
    print("  COMPONENT SIGNALS:")
    print("-" * 60)

    print(f"    VIX Level:      {result.vix_level:.2f} ({result.vix_regime})")
    print(f"    SPY Trend:      {result.spy_trend.upper()}")
    print(f"    Market Breadth: {result.breadth_signal.upper()}")

    print("\n  DETAILED SIGNALS:")
    for key, value in result.signals.items():
        bar_len = int(abs(value) * 10)
        bar = '+' * bar_len if value > 0 else '-' * bar_len
        print(f"    {key:20s}: {value:+.3f} |{bar}")

    print("\n" + "-" * 60)
    print("  TRADING IMPLICATIONS:")
    print("-" * 60)

    if result.regime == 'bull':
        print("    - Favor long positions")
        print("    - Buy dips to moving averages")
        print("    - Reduce hedges, allow winners to run")
    elif result.regime == 'bear':
        print("    - Reduce position sizes")
        print("    - Consider defensive sectors/hedges")
        print("    - Sell rallies, tighter stops")
    else:  # chop
        print("    - Reduce position sizes")
        print("    - Mean-reversion strategies may work")
        print("    - Avoid trend-following, use ranges")

    print("\n" + "=" * 60)


def print_history(df: pd.DataFrame) -> None:
    """Print regime history table."""
    if df.empty:
        print("No history data available.")
        return

    print("\n" + "=" * 70)
    print("                    REGIME HISTORY")
    print("=" * 70)
    print(f"{'Date':12s} | {'Close':>10s} | {'SMA20':>10s} | {'SMA50':>10s} | {'Regime':8s}")
    print("-" * 70)

    for _, row in df.tail(30).iterrows():
        date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
        print(f"{date_str:12s} | {row['close']:10.2f} | {row['sma_20']:10.2f} | {row['sma_50']:10.2f} | {row['regime'].upper():8s}")

    print("-" * 70)

    # Summary
    regime_counts = df['regime'].value_counts()
    print("\nRegime Distribution (last 60 days):")
    for regime, count in regime_counts.items():
        pct = count / len(df) * 100
        print(f"  {regime.upper():8s}: {count:3d} days ({pct:.1f}%)")
    print()


def print_chart_data(df: pd.DataFrame) -> None:
    """Output chart-ready JSON data."""
    if df.empty:
        print(json.dumps({"error": "No data available"}))
        return

    chart_data = {
        'dates': [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in df['date'].tolist()],
        'close': df['close'].tolist(),
        'sma_20': df['sma_20'].tolist(),
        'sma_50': df['sma_50'].tolist(),
        'regime': df['regime'].tolist(),
        'metadata': {
            'generated': datetime.now().isoformat(),
            'days': len(df),
        }
    }
    print(json.dumps(chart_data, indent=2))


def main():
    ap = argparse.ArgumentParser(description='Market Regime Detection for Kobe Trading System')
    ap.add_argument('--dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env',
                    help='Path to .env file')
    ap.add_argument('--history', action='store_true',
                    help='Show regime history')
    ap.add_argument('--days', type=int, default=60,
                    help='Number of days for history (default: 60)')
    ap.add_argument('--chart-data', action='store_true',
                    help='Output chart-ready JSON data')
    ap.add_argument('--json', action='store_true',
                    help='Output current regime as JSON')
    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    api_key = os.getenv('POLYGON_API_KEY', '')
    if not api_key:
        print("Error: POLYGON_API_KEY not found in environment")
        sys.exit(1)

    # Initialize detector
    detector = RegimeDetector(api_key)

    # Fetch data
    lookback = max(args.days + 50, 250)
    if not detector.fetch_data(lookback_days=lookback):
        print("Error: Could not fetch market data")
        sys.exit(1)

    # Handle different output modes
    if args.chart_data:
        history = detector.get_regime_history(args.days)
        print_chart_data(history)
    elif args.history:
        history = detector.get_regime_history(args.days)
        print_history(history)
    else:
        # Current regime detection
        result = detector.detect_regime()

        if args.json:
            print(json.dumps(asdict(result), indent=2))
        else:
            print_regime_report(result)


if __name__ == '__main__':
    main()
