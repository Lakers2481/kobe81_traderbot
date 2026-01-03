"""
Bounce Analysis Database Module

Comprehensive bounce analysis for consecutive down-day streaks across 900 tickers.
Uses REAL data from Polygon (primary) with yfinance/Stooq fallback.

Primary Recovery Window: 7 trading days
Price Basis: Split-adjusted (NOT dividend-adjusted)
"""

from bounce.data_loader import (
    load_ticker_data,
    load_universe_data,
    validate_against_stooq,
)
from bounce.streak_analyzer import (
    calculate_streaks_vectorized,
    calculate_forward_metrics,
    build_events_table,
    analyze_ticker,
)
from bounce.event_table import (
    compute_overall_summary,
    compute_per_stock_summary,
    derive_5y_from_10y,
)
from bounce.bounce_score import (
    calculate_bounce_score,
    get_bounce_profile_for_signal,
    apply_bounce_gates,
)
from bounce.validation import (
    verify_no_lookahead,
    validate_data_quality,
)
from bounce.profile_generator import (
    generate_ticker_profile,
    generate_all_profiles,
    generate_summary_report,
)
from bounce.strategy_integration import (
    BounceIntegration,
    integrate_with_scanner,
    create_bounce_watchlist,
)

__version__ = "1.0.0"


def quick_bounce_check(ticker: str, streak: int = None) -> dict:
    """
    Quick bounce profile lookup for a ticker.

    Args:
        ticker: Stock symbol (e.g., 'PLTR')
        streak: Specific streak level to check (1-7), or None for all

    Returns:
        dict with bounce profile, score, and gate results

    Usage:
        >>> from bounce import quick_bounce_check
        >>> result = quick_bounce_check('PLTR', streak=5)
        >>> print(result['bounce_score'], result['gate_passed'])
    """
    import pandas as pd
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent

    # Try to load pre-computed per-stock data
    per_stock_5y_path = PROJECT_ROOT / "reports" / "bounce" / "week_down_then_bounce_per_stock_5y.csv"
    per_stock_10y_path = PROJECT_ROOT / "reports" / "bounce" / "week_down_then_bounce_per_stock_10y.csv"

    result = {
        'ticker': ticker,
        'streak': streak,
        'data_available': False,
        'profiles': [],
        'best_streak': None,
        'bounce_score': None,
        'gate_passed': None,
        'gate_reason': None,
    }

    try:
        if per_stock_5y_path.exists():
            df_5y = pd.read_csv(per_stock_5y_path)
            df_10y = pd.read_csv(per_stock_10y_path) if per_stock_10y_path.exists() else df_5y

            ticker_data = df_5y[df_5y['ticker'] == ticker.upper()]
            if len(ticker_data) == 0:
                ticker_data = df_10y[df_10y['ticker'] == ticker.upper()]

            if len(ticker_data) > 0:
                result['data_available'] = True

                if streak is not None:
                    row = ticker_data[ticker_data['streak_n'] == streak]
                    if len(row) > 0:
                        row = row.iloc[0]
                        score = calculate_bounce_score(
                            row['recovery_7d_close_rate'],
                            row['avg_days_to_recover_7d'],
                            row['avg_best_7d_return'],
                            row['events'],
                            row['avg_max_drawdown_7d_pct']
                        )
                        passed, reason = apply_bounce_gates(
                            row['events'],
                            row['recovery_7d_close_rate'],
                            row['avg_days_to_recover_7d']
                        )
                        result['bounce_score'] = round(score, 2)
                        result['gate_passed'] = passed
                        result['gate_reason'] = reason
                        result['profiles'] = [{
                            'streak': streak,
                            'events': int(row['events']),
                            'recovery_rate': round(row['recovery_7d_close_rate'] * 100, 1),
                            'avg_days': round(row['avg_days_to_recover_7d'], 2),
                            'avg_return': round(row['avg_best_7d_return'] * 100, 1),
                        }]
                else:
                    # Return all streak levels
                    profiles = []
                    best_score = 0
                    best_streak = None

                    for _, row in ticker_data.iterrows():
                        score = calculate_bounce_score(
                            row['recovery_7d_close_rate'],
                            row['avg_days_to_recover_7d'],
                            row['avg_best_7d_return'],
                            row['events'],
                            row['avg_max_drawdown_7d_pct']
                        )
                        passed, reason = apply_bounce_gates(
                            row['events'],
                            row['recovery_7d_close_rate'],
                            row['avg_days_to_recover_7d']
                        )
                        profiles.append({
                            'streak': int(row['streak_n']),
                            'events': int(row['events']),
                            'recovery_rate': round(row['recovery_7d_close_rate'] * 100, 1),
                            'avg_days': round(row['avg_days_to_recover_7d'], 2),
                            'score': round(score, 2),
                            'gate_passed': passed,
                        })
                        if score > best_score:
                            best_score = score
                            best_streak = int(row['streak_n'])

                    result['profiles'] = profiles
                    result['best_streak'] = best_streak
                    result['bounce_score'] = round(best_score, 2)

    except Exception as e:
        result['error'] = str(e)

    return result


def run_bounce_analysis(tickers: list = None, min_streak: int = 3, top_n: int = 10) -> dict:
    """
    Run bounce analysis on tickers with current streaks.

    Args:
        tickers: List of tickers to analyze (None = scan universe)
        min_streak: Minimum streak level to consider
        top_n: Number of top candidates to return

    Returns:
        dict with watchlist and analysis results

    Usage:
        >>> from bounce import run_bounce_analysis
        >>> results = run_bounce_analysis(min_streak=3, top_n=5)
        >>> for ticker in results['watchlist']:
        ...     print(f"{ticker['ticker']}: streak={ticker['streak']}, score={ticker['bounce_score']}")
    """
    from bounce.strategy_integration import create_bounce_watchlist
    return create_bounce_watchlist(
        prefer_window=5,
        fallback_window=10,
        min_streak=min_streak,
        top_n=top_n,
    )


__all__ = [
    # Data loading
    "load_ticker_data",
    "load_universe_data",
    "validate_against_stooq",
    # Streak analysis
    "calculate_streaks_vectorized",
    "calculate_forward_metrics",
    "build_events_table",
    "analyze_ticker",
    # Event tables
    "compute_overall_summary",
    "compute_per_stock_summary",
    "derive_5y_from_10y",
    # Bounce scoring
    "calculate_bounce_score",
    "get_bounce_profile_for_signal",
    "apply_bounce_gates",
    # Validation
    "verify_no_lookahead",
    "validate_data_quality",
    # Profile generation
    "generate_ticker_profile",
    "generate_all_profiles",
    "generate_summary_report",
    # Strategy integration
    "BounceIntegration",
    "integrate_with_scanner",
    "create_bounce_watchlist",
]
