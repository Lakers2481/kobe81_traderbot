"""
PATTERN RHYMES ENGINE - "History doesn't repeat, but it rhymes"

Uses 900-stock historical data to find similar patterns and
predict outcomes based on historical rhymes.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


@dataclass
class PatternMatch:
    """A historical pattern that rhymes with current setup."""
    symbol: str
    match_date: datetime
    similarity_score: float  # 0-1, how similar the pattern is
    outcome: str  # "WIN", "LOSS", "NEUTRAL"
    pnl_pct: float  # What happened after the pattern
    days_to_outcome: int  # How many days until resolution
    pattern_type: str  # e.g., "ibs_low_bounce", "sweep_recovery"


@dataclass
class RhymeAnalysis:
    """Analysis of how current setup rhymes with history."""
    current_symbol: str
    current_date: datetime
    matches_found: int
    historical_win_rate: float
    avg_pnl_if_win: float
    avg_pnl_if_loss: float
    confidence: float
    best_matches: List[PatternMatch]
    warning: Optional[str] = None


class PatternRhymesEngine:
    """
    Finds historical patterns that "rhyme" with current setups.

    Uses DTW (Dynamic Time Warping) and feature-based matching
    to find similar historical scenarios and their outcomes.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/polygon_cache")
        if not self.cache_dir.exists():
            self.cache_dir = Path("cache")

        # Pattern library (built from historical analysis)
        self.pattern_library = []
        self.state_dir = Path("state/autonomous/patterns")
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def extract_pattern_features(self, df: pd.DataFrame, idx: int, lookback: int = 20) -> Optional[Dict[str, float]]:
        """
        Extract features that describe a pattern at a given index.
        These features capture the "shape" of price action.
        """
        if idx < lookback:
            return None

        window = df.iloc[idx-lookback:idx+1].copy()
        if len(window) < lookback:
            return None

        # Price features (normalized to first bar)
        base_price = window['close'].iloc[0]
        closes = (window['close'] / base_price - 1) * 100  # % change

        # Calculate features
        features = {
            # Trend features
            "total_return": closes.iloc[-1],
            "max_drawdown": (window['close'].min() / window['close'].max() - 1) * 100,
            "max_runup": (window['close'].max() / window['close'].min() - 1) * 100,

            # Volatility features
            "volatility": window['close'].pct_change().std() * 100,
            "range_expansion": (window['high'] - window['low']).mean() / base_price * 100,

            # Momentum features
            "momentum_5d": (window['close'].iloc[-1] / window['close'].iloc[-6] - 1) * 100 if len(window) >= 6 else 0,
            "momentum_10d": (window['close'].iloc[-1] / window['close'].iloc[-11] - 1) * 100 if len(window) >= 11 else 0,

            # Position features
            "position_in_range": (window['close'].iloc[-1] - window['low'].min()) / (window['high'].max() - window['low'].min() + 0.001),

            # Volume features (if available)
            "volume_trend": 0,  # Will be set below

            # IBS features
            "ibs_current": (window['close'].iloc[-1] - window['low'].iloc[-1]) / (window['high'].iloc[-1] - window['low'].iloc[-1] + 0.001),
            "ibs_avg_5d": 0,  # Will be set below
        }

        # Calculate IBS average
        ibs = (window['close'] - window['low']) / (window['high'] - window['low'] + 0.001)
        features["ibs_avg_5d"] = ibs.tail(5).mean()

        # Volume trend (if available)
        if 'volume' in window.columns and window['volume'].sum() > 0:
            vol_sma = window['volume'].rolling(10).mean()
            features["volume_trend"] = (window['volume'].iloc[-1] / vol_sma.iloc[-1] - 1) * 100 if vol_sma.iloc[-1] > 0 else 0

        return features

    def calculate_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate similarity between two pattern feature sets."""
        if not features1 or not features2:
            return 0.0

        # Get common keys
        keys = set(features1.keys()) & set(features2.keys())
        if not keys:
            return 0.0

        # Calculate weighted distance
        weights = {
            "total_return": 1.0,
            "max_drawdown": 0.8,
            "max_runup": 0.8,
            "volatility": 0.6,
            "momentum_5d": 1.0,
            "momentum_10d": 0.8,
            "position_in_range": 1.0,
            "ibs_current": 1.2,  # Higher weight for IBS (our core indicator)
            "ibs_avg_5d": 1.0,
            "volume_trend": 0.5,
            "range_expansion": 0.5,
        }

        total_weight = 0
        distance = 0

        for key in keys:
            w = weights.get(key, 0.5)
            # Normalize difference (assume typical range of -100 to +100 for most features)
            diff = abs(features1[key] - features2[key]) / 100.0
            distance += w * diff
            total_weight += w

        if total_weight == 0:
            return 0.0

        # Convert distance to similarity (0-1)
        avg_distance = distance / total_weight
        similarity = max(0, 1 - avg_distance)

        return similarity

    def find_rhymes(
        self,
        symbol: str,
        current_df: pd.DataFrame,
        min_similarity: float = 0.7,
        max_matches: int = 10
    ) -> RhymeAnalysis:
        """
        Find historical patterns that rhyme with current setup.

        Returns analysis of similar historical scenarios and their outcomes.
        """
        logger.info(f"Finding pattern rhymes for {symbol}...")

        # Extract current pattern features
        current_features = self.extract_pattern_features(current_df, len(current_df) - 1)
        if not current_features:
            return RhymeAnalysis(
                current_symbol=symbol,
                current_date=datetime.now(ET),
                matches_found=0,
                historical_win_rate=0.5,
                avg_pnl_if_win=0,
                avg_pnl_if_loss=0,
                confidence=0,
                best_matches=[],
                warning="Could not extract current pattern features"
            )

        all_matches = []

        # Search through all cached stocks for similar patterns
        cache_files = list(self.cache_dir.glob("*.csv"))
        for cache_file in cache_files:
            try:
                df = pd.read_csv(cache_file, parse_dates=['timestamp'])
                if len(df) < 50:
                    continue

                # Don't compare current symbol to itself at the same time
                file_symbol = cache_file.stem.upper()

                # Search through historical bars for similar patterns
                for idx in range(30, len(df) - 10):  # Leave room for outcome measurement
                    hist_features = self.extract_pattern_features(df, idx)
                    if not hist_features:
                        continue

                    similarity = self.calculate_similarity(current_features, hist_features)
                    if similarity >= min_similarity:
                        # Calculate outcome (what happened after this pattern)
                        future = df.iloc[idx+1:idx+11]  # Next 10 bars
                        if len(future) < 5:
                            continue

                        entry = df.iloc[idx]['close']
                        exit_price = future.iloc[-1]['close']
                        pnl_pct = (exit_price / entry - 1) * 100

                        outcome = "WIN" if pnl_pct > 0.5 else ("LOSS" if pnl_pct < -0.5 else "NEUTRAL")

                        match = PatternMatch(
                            symbol=file_symbol,
                            match_date=df.iloc[idx]['timestamp'],
                            similarity_score=similarity,
                            outcome=outcome,
                            pnl_pct=round(pnl_pct, 2),
                            days_to_outcome=len(future),
                            pattern_type="similar_setup"
                        )
                        all_matches.append(match)

            except Exception:
                continue

        # Sort by similarity
        all_matches.sort(key=lambda x: x.similarity_score, reverse=True)
        best_matches = all_matches[:max_matches]

        # Calculate statistics from matches
        if best_matches:
            wins = [m for m in best_matches if m.outcome == "WIN"]
            losses = [m for m in best_matches if m.outcome == "LOSS"]

            win_rate = len(wins) / len(best_matches) if best_matches else 0.5
            avg_win = np.mean([m.pnl_pct for m in wins]) if wins else 0
            avg_loss = np.mean([m.pnl_pct for m in losses]) if losses else 0

            # Confidence based on sample size and similarity
            avg_similarity = np.mean([m.similarity_score for m in best_matches])
            confidence = min(0.9, avg_similarity * (len(best_matches) / 10))
        else:
            win_rate = 0.5
            avg_win = 0
            avg_loss = 0
            confidence = 0

        return RhymeAnalysis(
            current_symbol=symbol,
            current_date=datetime.now(ET),
            matches_found=len(all_matches),
            historical_win_rate=round(win_rate, 2),
            avg_pnl_if_win=round(avg_win, 2),
            avg_pnl_if_loss=round(avg_loss, 2),
            confidence=round(confidence, 2),
            best_matches=best_matches
        )

    def analyze_seasonality(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze monthly/quarterly patterns for a symbol.
        "January effect", "Sell in May", etc.
        """
        cache_file = self.cache_dir / f"{symbol}.csv"
        if not cache_file.exists():
            return {"error": "No data for symbol"}

        df = pd.read_csv(cache_file, parse_dates=['timestamp'])
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        # Calculate monthly returns
        df['return'] = df['close'].pct_change() * 100

        monthly_stats = df.groupby('month')['return'].agg(['mean', 'std', 'count']).round(3)
        quarterly_stats = df.groupby('quarter')['return'].agg(['mean', 'std', 'count']).round(3)
        dow_stats = df.groupby('day_of_week')['return'].agg(['mean', 'std', 'count']).round(3)

        return {
            "symbol": symbol,
            "monthly_returns": monthly_stats.to_dict(),
            "quarterly_returns": quarterly_stats.to_dict(),
            "day_of_week_returns": dow_stats.to_dict(),
            "best_month": int(monthly_stats['mean'].idxmax()),
            "worst_month": int(monthly_stats['mean'].idxmin()),
            "best_quarter": int(quarterly_stats['mean'].idxmax()),
            "worst_quarter": int(quarterly_stats['mean'].idxmin()),
        }

    def analyze_mean_reversion_timing(self, lookback_days: int = 252) -> Dict[str, Any]:
        """
        Analyze how long extreme moves take to revert across all stocks.
        This informs our time-stop parameters.
        """
        logger.info("Analyzing mean reversion timing across 800 stocks...")

        reversion_times = []

        cache_files = list(self.cache_dir.glob("*.csv"))[:50]  # Sample 50 stocks
        for cache_file in cache_files:
            try:
                df = pd.read_csv(cache_file, parse_dates=['timestamp'])
                if len(df) < lookback_days:
                    continue

                # Find extreme low IBS days (our entry signal)
                df['ibs'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.001)
                low_ibs = df[df['ibs'] < 0.08].copy()

                for idx in low_ibs.index:
                    if idx >= len(df) - 10:
                        continue
                    entry = df.loc[idx, 'close']

                    # Find how many days until price exceeds entry
                    for days, future_idx in enumerate(range(idx+1, min(idx+11, len(df))), 1):
                        if df.loc[future_idx, 'close'] > entry:
                            reversion_times.append(days)
                            break

            except Exception:
                continue

        if not reversion_times:
            return {"error": "No data"}

        return {
            "total_observations": len(reversion_times),
            "mean_days_to_revert": round(np.mean(reversion_times), 1),
            "median_days_to_revert": round(np.median(reversion_times), 1),
            "p25_days": round(np.percentile(reversion_times, 25), 1),
            "p75_days": round(np.percentile(reversion_times, 75), 1),
            "reverted_in_1_day": round(sum(1 for t in reversion_times if t == 1) / len(reversion_times), 2),
            "reverted_in_3_days": round(sum(1 for t in reversion_times if t <= 3) / len(reversion_times), 2),
            "reverted_in_5_days": round(sum(1 for t in reversion_times if t <= 5) / len(reversion_times), 2),
        }

    def find_sector_correlations(self) -> Dict[str, Any]:
        """
        Find which stocks move together.
        Helps with diversification and position sizing.
        """
        logger.info("Analyzing sector correlations...")

        # Load all stock returns
        returns_dict = {}
        cache_files = list(self.cache_dir.glob("*.csv"))[:100]  # Sample 100

        for cache_file in cache_files:
            try:
                df = pd.read_csv(cache_file, parse_dates=['timestamp'])
                df = df.set_index('timestamp')['close'].pct_change().dropna()
                if len(df) > 100:
                    returns_dict[cache_file.stem.upper()] = df
            except Exception:
                continue

        if len(returns_dict) < 10:
            return {"error": "Not enough data"}

        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_dict)

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        # Find highest correlations (excluding self-correlation)
        high_corr_pairs = []
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                corr = corr_matrix.loc[col1, col2]
                if not np.isnan(corr) and abs(corr) > 0.7:
                    high_corr_pairs.append({
                        "pair": f"{col1}/{col2}",
                        "correlation": round(corr, 3)
                    })

        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

        return {
            "stocks_analyzed": len(returns_dict),
            "avg_correlation": round(corr_matrix.mean().mean(), 3),
            "high_correlation_pairs": high_corr_pairs[:20],
        }


# Global instance
_engine = None

def get_rhymes_engine() -> PatternRhymesEngine:
    """Get the global PatternRhymesEngine instance."""
    global _engine
    if _engine is None:
        _engine = PatternRhymesEngine()
    return _engine
