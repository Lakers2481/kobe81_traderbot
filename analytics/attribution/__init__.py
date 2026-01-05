"""
P&L Attribution Engine - Know WHY You Made/Lost Money

Not just "we made $500" but "we made $800 from momentum, lost $200
to mean-reversion timing, lost $100 to slippage."

This is how institutional traders understand their performance.

Components:
- DailyPnL: Daily P&L breakdown by source
- FactorAttribution: Factor-based P&L decomposition
- StrategyAttribution: Per-strategy contribution
- SectorAttribution: Sector contribution analysis
- AttributionReport: Human-readable reports

Solo Trader Features:
- Plain English daily summary via Telegram
- Weekly attribution report (PDF-ready)
- Identify your REAL edge (not what you think it is)
- Spot strategy drift before it costs you

Author: Kobe Trading System
Created: 2026-01-04
"""

from .daily_pnl import (
    DailyPnLTracker,
    DailyPnL,
    get_daily_pnl_tracker,
)
from .factor_attribution import (
    FactorAttributor,
    FactorPnL,
)
from .strategy_attribution import (
    StrategyAttributor,
    StrategyPnL,
)
from .attribution_report import (
    AttributionReporter,
    AttributionReport,
    generate_daily_attribution,
    generate_weekly_attribution,
)

__all__ = [
    "DailyPnLTracker",
    "DailyPnL",
    "get_daily_pnl_tracker",
    "FactorAttributor",
    "FactorPnL",
    "StrategyAttributor",
    "StrategyPnL",
    "AttributionReporter",
    "AttributionReport",
    "generate_daily_attribution",
    "generate_weekly_attribution",
]
