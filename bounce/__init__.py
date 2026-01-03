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
