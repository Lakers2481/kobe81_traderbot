"""
Strategy Integration Module for Bounce Analysis

Integrates bounce profiles into the existing strategy engine for:
- Signal filtering via bounce gates
- Confidence adjustment via BounceScore
- Position sizing based on bounce profile
- Exit timing based on historical recovery
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bounce.bounce_score import (
    get_bounce_profile_for_signal,
    adjust_signal_for_bounce,
)


class BounceIntegration:
    """
    Integrates bounce analysis into the trading strategy.

    Usage:
        integration = BounceIntegration(
            per_stock_5y=pd.read_csv("reports/bounce/week_down_then_bounce_per_stock_5y.csv"),
            per_stock_10y=pd.read_csv("reports/bounce/week_down_then_bounce_per_stock_10y.csv"),
        )

        # Filter and rank signals
        ranked_signals = integration.process_signals(signals, current_streaks)
    """

    def __init__(
        self,
        per_stock_5y: Optional[pd.DataFrame] = None,
        per_stock_10y: Optional[pd.DataFrame] = None,
        min_events: int = 20,
        min_bounce_score: float = 50.0,
        require_gate_pass: bool = True,
    ):
        """
        Initialize bounce integration.

        Args:
            per_stock_5y: 5Y per-stock summary DataFrame
            per_stock_10y: 10Y per-stock summary DataFrame
            min_events: Minimum events required for profile
            min_bounce_score: Minimum BounceScore to accept signal
            require_gate_pass: Whether to require gate pass
        """
        self.per_stock_5y = per_stock_5y
        self.per_stock_10y = per_stock_10y
        self.min_events = min_events
        self.min_bounce_score = min_bounce_score
        self.require_gate_pass = require_gate_pass

        # Cache for profiles
        self._profile_cache: Dict[Tuple[str, int], Dict] = {}

    @classmethod
    def from_files(
        cls,
        reports_dir: Path = None,
        prefer_5y: bool = True,
    ) -> "BounceIntegration":
        """
        Load bounce data from standard report files.

        Args:
            reports_dir: Reports directory (default: reports/bounce)
            prefer_5y: Whether to prefer 5Y data

        Returns:
            BounceIntegration instance
        """
        if reports_dir is None:
            reports_dir = PROJECT_ROOT / "reports" / "bounce"

        reports_dir = Path(reports_dir)

        per_stock_5y = None
        per_stock_10y = None

        # Load 5Y
        path_5y = reports_dir / "week_down_then_bounce_per_stock_5y.csv"
        if path_5y.exists():
            per_stock_5y = pd.read_csv(path_5y)

        # Load 10Y
        path_10y = reports_dir / "week_down_then_bounce_per_stock_10y.csv"
        if path_10y.exists():
            per_stock_10y = pd.read_csv(path_10y)

        return cls(
            per_stock_5y=per_stock_5y,
            per_stock_10y=per_stock_10y,
        )

    def get_profile(
        self,
        ticker: str,
        current_streak: int,
    ) -> Dict:
        """
        Get bounce profile for a ticker/streak combination.

        Uses caching for performance.

        Args:
            ticker: Stock ticker
            current_streak: Current streak level (1-7)

        Returns:
            Bounce profile dict
        """
        cache_key = (ticker, current_streak)

        if cache_key in self._profile_cache:
            return self._profile_cache[cache_key]

        profile = get_bounce_profile_for_signal(
            ticker=ticker,
            current_streak=current_streak,
            per_stock_5y=self.per_stock_5y,
            per_stock_10y=self.per_stock_10y,
            min_events=self.min_events,
        )

        self._profile_cache[cache_key] = profile
        return profile

    def filter_signal(
        self,
        signal: Dict,
        current_streak: int = None,
    ) -> Tuple[bool, Dict, str]:
        """
        Filter a single signal through bounce gates.

        Args:
            signal: Signal dict (must have 'ticker' or 'symbol')
            current_streak: Override streak level (otherwise uses signal['streak'])

        Returns:
            (passed: bool, enriched_signal: Dict, reason: str)
        """
        ticker = signal.get('ticker') or signal.get('symbol')
        if ticker is None:
            return False, signal, "NO_TICKER"

        streak = current_streak or signal.get('streak') or signal.get('current_streak', 1)

        # Get bounce profile
        profile = self.get_profile(ticker, streak)

        # Check gate
        if self.require_gate_pass and not profile.get("gate_passed", False):
            return False, signal, profile.get("reject_reason", "GATE_FAILED")

        # Check minimum score
        if profile.get("bounce_score", 0) < self.min_bounce_score:
            return False, signal, f"LOW_SCORE: {profile.get('bounce_score', 0):.0f} < {self.min_bounce_score}"

        # Enrich signal
        enriched = adjust_signal_for_bounce(signal, profile)

        return True, enriched, None

    def process_signals(
        self,
        signals: List[Dict],
        current_streaks: Optional[Dict[str, int]] = None,
        max_signals: int = None,
    ) -> List[Dict]:
        """
        Process multiple signals through bounce filtering and ranking.

        Args:
            signals: List of signal dicts
            current_streaks: Optional dict of ticker -> current_streak
            max_signals: Maximum signals to return

        Returns:
            List of enriched signals, sorted by BounceScore
        """
        if current_streaks is None:
            current_streaks = {}

        passed_signals = []

        for signal in signals:
            ticker = signal.get('ticker') or signal.get('symbol')
            streak = current_streaks.get(ticker) or signal.get('streak') or signal.get('current_streak', 1)

            passed, enriched, reason = self.filter_signal(signal, streak)

            if passed:
                passed_signals.append(enriched)

        # Sort by bounce score
        passed_signals.sort(key=lambda x: x.get("bounce_score", 0), reverse=True)

        # Limit if requested
        if max_signals is not None:
            passed_signals = passed_signals[:max_signals]

        return passed_signals

    def get_sizing_factor(
        self,
        ticker: str,
        current_streak: int,
    ) -> float:
        """
        Get position sizing factor based on bounce profile.

        Returns:
            Sizing factor (0.5 to 1.0)
            - BounceScore >= 80: 1.0 (full size)
            - BounceScore >= 60: 0.75 (3/4 size)
            - BounceScore < 60: 0.5 (half size)
        """
        profile = self.get_profile(ticker, current_streak)
        bounce_score = profile.get("bounce_score", 0)

        if bounce_score >= 80:
            return 1.0
        elif bounce_score >= 60:
            return 0.75
        else:
            return 0.5

    def get_exit_parameters(
        self,
        ticker: str,
        current_streak: int,
    ) -> Dict:
        """
        Get suggested exit parameters based on bounce profile.

        Returns:
            Dict with:
            - suggested_target_pct: Based on p95 return * 0.8
            - suggested_stop_pct: Based on median drawdown * 1.2
            - suggested_time_stop: Based on median days + 2
        """
        profile = self.get_profile(ticker, current_streak)

        avg_return = profile.get("avg_return")
        avg_drawdown = profile.get("avg_drawdown")
        avg_days = profile.get("avg_days")

        # Convert if percentage
        if avg_return is not None and abs(avg_return) > 1:
            avg_return = avg_return / 100
        if avg_drawdown is not None and abs(avg_drawdown) > 1:
            avg_drawdown = avg_drawdown / 100

        return {
            "suggested_target_pct": (avg_return * 0.8) if avg_return else None,
            "suggested_stop_pct": (abs(avg_drawdown) * 1.2) if avg_drawdown else None,
            "suggested_time_stop": int(avg_days + 2) if avg_days else 7,
            "bounce_window_used": profile.get("bounce_window_used", "NONE"),
        }

    def generate_signal_report(
        self,
        signals: List[Dict],
        current_streaks: Optional[Dict[str, int]] = None,
    ) -> str:
        """
        Generate markdown report for processed signals.

        Args:
            signals: List of signal dicts
            current_streaks: Optional dict of ticker -> current_streak

        Returns:
            Markdown report string
        """
        if current_streaks is None:
            current_streaks = {}

        lines = []
        lines.append("# Bounce-Filtered Signals Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Input Signals:** {len(signals)}")
        lines.append("")

        # Process signals
        processed = []
        rejected = []

        for signal in signals:
            ticker = signal.get('ticker') or signal.get('symbol')
            streak = current_streaks.get(ticker) or signal.get('streak') or signal.get('current_streak', 1)

            passed, enriched, reason = self.filter_signal(signal, streak)

            if passed:
                processed.append(enriched)
            else:
                rejected.append({
                    "ticker": ticker,
                    "streak": streak,
                    "reason": reason,
                })

        lines.append(f"**Passed:** {len(processed)}")
        lines.append(f"**Rejected:** {len(rejected)}")
        lines.append("")

        # Passed signals table
        if processed:
            # Sort by bounce score
            processed.sort(key=lambda x: x.get("bounce_score", 0), reverse=True)

            lines.append("## Approved Signals")
            lines.append("")
            lines.append("| Rank | Ticker | Streak | BounceScore | Window | Recovery | Avg Days | Gate |")
            lines.append("|------|--------|--------|-------------|--------|----------|----------|------|")

            for i, sig in enumerate(processed, 1):
                ticker = sig.get('ticker') or sig.get('symbol')
                streak = sig.get('bounce_streak', sig.get('streak', '-'))
                score = sig.get('bounce_score', 0)
                window = sig.get('bounce_window_used', '-')
                recovery = sig.get('bounce_recovery_rate')
                avg_days = sig.get('bounce_avg_days')

                recovery_str = f"{recovery:.0%}" if recovery else "-"
                days_str = f"{avg_days:.1f}" if avg_days else "-"
                gate_str = "✓" if sig.get('bounce_gate_passed') else "✗"

                lines.append(f"| {i} | {ticker} | {streak} | {score:.0f} | {window} | {recovery_str} | {days_str} | {gate_str} |")

            lines.append("")

        # Rejected signals
        if rejected:
            lines.append("## Rejected Signals")
            lines.append("")
            lines.append("| Ticker | Streak | Rejection Reason |")
            lines.append("|--------|--------|------------------|")

            for rej in rejected:
                lines.append(f"| {rej['ticker']} | {rej['streak']} | {rej['reason']} |")

            lines.append("")

        return "\n".join(lines)


def integrate_with_scanner(
    scanner,
    bounce_integration: BounceIntegration,
    signals_df: pd.DataFrame,
    current_streaks: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Integrate bounce filtering with DualStrategyScanner output.

    Args:
        scanner: DualStrategyScanner instance
        bounce_integration: BounceIntegration instance
        signals_df: DataFrame of signals from scanner
        current_streaks: DataFrame with ticker, current_streak columns

    Returns:
        Filtered and enriched signals DataFrame
    """
    if signals_df is None or len(signals_df) == 0:
        return pd.DataFrame()

    # Build streak dict
    streak_dict = {}
    if current_streaks is not None and len(current_streaks) > 0:
        for _, row in current_streaks.iterrows():
            ticker = row.get('ticker') or row.get('symbol')
            streak = row.get('current_streak', 1)
            streak_dict[ticker] = streak

    # Convert DataFrame to list of dicts
    signals = signals_df.to_dict('records')

    # Process through bounce integration
    processed = bounce_integration.process_signals(signals, streak_dict)

    if not processed:
        return pd.DataFrame()

    return pd.DataFrame(processed)


def create_bounce_watchlist(
    current_streaks_df: pd.DataFrame,
    bounce_integration: BounceIntegration,
    min_streak: int = 3,
    max_signals: int = 20,
) -> pd.DataFrame:
    """
    Create bounce watchlist from current streak data.

    Args:
        current_streaks_df: DataFrame with ticker, current_streak columns
        bounce_integration: BounceIntegration instance
        min_streak: Minimum streak level to consider
        max_signals: Maximum signals to return

    Returns:
        Watchlist DataFrame sorted by BounceScore
    """
    if current_streaks_df is None or len(current_streaks_df) == 0:
        return pd.DataFrame()

    # Filter for minimum streak
    candidates = current_streaks_df[current_streaks_df['current_streak'] >= min_streak].copy()

    if len(candidates) == 0:
        return pd.DataFrame()

    # Build signals list
    signals = []
    for _, row in candidates.iterrows():
        ticker = row.get('ticker') or row.get('symbol')
        streak = row.get('current_streak', min_streak)

        signals.append({
            'ticker': ticker,
            'streak': streak,
            'last_close': row.get('last_close'),
            'last_date': row.get('last_date'),
            'pct_off_high_20d': row.get('pct_off_high_20d'),
        })

    # Build streak dict
    streak_dict = {s['ticker']: s['streak'] for s in signals}

    # Process through bounce integration
    processed = bounce_integration.process_signals(
        signals=signals,
        current_streaks=streak_dict,
        max_signals=max_signals,
    )

    if not processed:
        return pd.DataFrame()

    return pd.DataFrame(processed)
