"""
Time, Day, and Season Awareness for Kobe.

Kobe always knows:
- What time it is (and what that means for trading)
- What day it is (weekday, weekend, holiday)
- What season it is (earnings, FOMC, opex, etc.)
- What the market is doing right now
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

# Eastern Time for US markets
ET = ZoneInfo("America/New_York")


class MarketPhase(Enum):
    """Current phase of the trading day."""
    PRE_MARKET_EARLY = "pre_market_early"      # 4:00 AM - 7:00 AM
    PRE_MARKET_ACTIVE = "pre_market_active"    # 7:00 AM - 9:30 AM
    MARKET_OPENING = "market_opening"          # 9:30 AM - 10:00 AM (volatile)
    MARKET_MORNING = "market_morning"          # 10:00 AM - 11:30 AM (best setups)
    MARKET_LUNCH = "market_lunch"              # 11:30 AM - 14:00 PM (choppy)
    MARKET_AFTERNOON = "market_afternoon"      # 14:00 PM - 15:30 PM (power hour)
    MARKET_CLOSE = "market_close"              # 15:30 PM - 16:00 PM (closing)
    AFTER_HOURS = "after_hours"                # 16:00 PM - 20:00 PM
    NIGHT = "night"                            # 20:00 PM - 4:00 AM
    WEEKEND = "weekend"
    HOLIDAY = "holiday"


class Season(Enum):
    """Market seasons and special periods."""
    NORMAL = "normal"
    EARNINGS_SEASON = "earnings_season"        # Jan/Apr/Jul/Oct
    FOMC_WEEK = "fomc_week"
    TRIPLE_WITCHING = "triple_witching"        # 3rd Friday of Mar/Jun/Sep/Dec
    TAX_LOSS_SEASON = "tax_loss_season"        # December
    JANUARY_EFFECT = "january_effect"          # January
    SELL_IN_MAY = "sell_in_may"               # May-October
    SANTA_RALLY = "santa_rally"               # Last week of December
    SUMMER_DOLDRUMS = "summer_doldrums"       # July-August


class WorkMode(Enum):
    """What Kobe should be doing right now."""
    ACTIVE_TRADING = "active_trading"          # Scanning, trading
    MONITORING = "monitoring"                  # Watching positions
    RESEARCH = "research"                      # Backtesting, strategy discovery
    LEARNING = "learning"                      # Analyzing past trades
    OPTIMIZATION = "optimization"              # Parameter tuning
    MAINTENANCE = "maintenance"                # Data updates, cleanup
    DEEP_RESEARCH = "deep_research"           # Weekend: extensive experiments


@dataclass
class MarketContext:
    """Complete context of what's happening right now."""
    timestamp: datetime
    phase: MarketPhase
    season: Season
    work_mode: WorkMode

    # Time details
    is_market_open: bool
    is_weekend: bool
    is_holiday: bool
    day_of_week: str

    # Market state
    minutes_to_open: Optional[int] = None
    minutes_to_close: Optional[int] = None

    # Special events
    is_fomc_day: bool = False
    is_earnings_heavy: bool = False
    is_opex_day: bool = False
    is_early_close: bool = False

    # Recommendations
    trading_allowed: bool = True
    recommended_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "phase": self.phase.value,
            "season": self.season.value,
            "work_mode": self.work_mode.value,
            "is_market_open": self.is_market_open,
            "is_weekend": self.is_weekend,
            "is_holiday": self.is_holiday,
            "day_of_week": self.day_of_week,
            "minutes_to_open": self.minutes_to_open,
            "minutes_to_close": self.minutes_to_close,
            "is_fomc_day": self.is_fomc_day,
            "is_earnings_heavy": self.is_earnings_heavy,
            "is_opex_day": self.is_opex_day,
            "trading_allowed": self.trading_allowed,
            "recommended_actions": self.recommended_actions,
        }


class TimeAwareness:
    """Knows what time it is and what that means."""

    # Market hours (Eastern Time)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    EARLY_CLOSE = time(13, 0)

    # Phase boundaries
    PHASE_TIMES = {
        time(4, 0): MarketPhase.PRE_MARKET_EARLY,
        time(7, 0): MarketPhase.PRE_MARKET_ACTIVE,
        time(9, 30): MarketPhase.MARKET_OPENING,
        time(10, 0): MarketPhase.MARKET_MORNING,
        time(11, 30): MarketPhase.MARKET_LUNCH,
        time(14, 0): MarketPhase.MARKET_AFTERNOON,
        time(15, 30): MarketPhase.MARKET_CLOSE,
        time(16, 0): MarketPhase.AFTER_HOURS,
        time(20, 0): MarketPhase.NIGHT,
    }

    def __init__(self):
        self._early_close_dates: set = set()
        self._load_early_close_dates()

    def _load_early_close_dates(self):
        """Load early close dates (day before Thanksgiving, Christmas Eve, etc.)"""
        # These dates have 1 PM close
        # For 2025-2026, common early closes:
        self._early_close_dates = {
            # 2025
            datetime(2025, 7, 3).date(),   # Day before July 4th
            datetime(2025, 11, 28).date(), # Day after Thanksgiving (Friday)
            datetime(2025, 12, 24).date(), # Christmas Eve
            # 2026
            datetime(2026, 7, 3).date(),
            datetime(2026, 11, 27).date(),
            datetime(2026, 12, 24).date(),
        }

    def now(self) -> datetime:
        """Current time in Eastern."""
        return datetime.now(ET)

    def get_phase(self, dt: Optional[datetime] = None) -> MarketPhase:
        """Get current market phase."""
        if dt is None:
            dt = self.now()

        # Check weekend first
        if dt.weekday() >= 5:
            return MarketPhase.WEEKEND

        current_time = dt.time()

        # Find the appropriate phase
        phase = MarketPhase.NIGHT  # Default
        for phase_time, phase_value in sorted(self.PHASE_TIMES.items()):
            if current_time >= phase_time:
                phase = phase_value

        return phase

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Is the market currently open?"""
        if dt is None:
            dt = self.now()

        if dt.weekday() >= 5:
            return False

        close_time = self.EARLY_CLOSE if dt.date() in self._early_close_dates else self.MARKET_CLOSE

        return self.MARKET_OPEN <= dt.time() < close_time

    def minutes_to_open(self, dt: Optional[datetime] = None) -> Optional[int]:
        """Minutes until market opens (None if already open or weekend)."""
        if dt is None:
            dt = self.now()

        if self.is_market_open(dt):
            return None

        if dt.weekday() >= 5:
            # Weekend - calculate to Monday
            days_until_monday = 7 - dt.weekday()
            next_open = dt.replace(
                hour=9, minute=30, second=0, microsecond=0
            ) + timedelta(days=days_until_monday)
        elif dt.time() < self.MARKET_OPEN:
            # Before open today
            next_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        else:
            # After close today
            next_open = (dt + timedelta(days=1)).replace(
                hour=9, minute=30, second=0, microsecond=0
            )

        return int((next_open - dt).total_seconds() / 60)

    def minutes_to_close(self, dt: Optional[datetime] = None) -> Optional[int]:
        """Minutes until market closes (None if closed)."""
        if dt is None:
            dt = self.now()

        if not self.is_market_open(dt):
            return None

        close_time = self.EARLY_CLOSE if dt.date() in self._early_close_dates else self.MARKET_CLOSE
        close_dt = dt.replace(
            hour=close_time.hour, minute=close_time.minute, second=0, microsecond=0
        )

        return int((close_dt - dt).total_seconds() / 60)


class MarketCalendarAwareness:
    """Knows about holidays, special days, and market calendar."""

    # Major US market holidays for 2025-2026
    HOLIDAYS = {
        # 2025
        datetime(2025, 1, 1).date(): "New Year's Day",
        datetime(2025, 1, 20).date(): "MLK Day",
        datetime(2025, 2, 17).date(): "Presidents Day",
        datetime(2025, 4, 18).date(): "Good Friday",
        datetime(2025, 5, 26).date(): "Memorial Day",
        datetime(2025, 6, 19).date(): "Juneteenth",
        datetime(2025, 7, 4).date(): "Independence Day",
        datetime(2025, 9, 1).date(): "Labor Day",
        datetime(2025, 11, 27).date(): "Thanksgiving",
        datetime(2025, 12, 25).date(): "Christmas",
        # 2026
        datetime(2026, 1, 1).date(): "New Year's Day",
        datetime(2026, 1, 19).date(): "MLK Day",
        datetime(2026, 2, 16).date(): "Presidents Day",
        datetime(2026, 4, 3).date(): "Good Friday",
        datetime(2026, 5, 25).date(): "Memorial Day",
        datetime(2026, 6, 19).date(): "Juneteenth",
        datetime(2026, 7, 3).date(): "Independence Day (Observed)",
        datetime(2026, 9, 7).date(): "Labor Day",
        datetime(2026, 11, 26).date(): "Thanksgiving",
        datetime(2026, 12, 25).date(): "Christmas",
    }

    # FOMC meeting dates (2025-2026)
    FOMC_DATES = {
        # 2025
        datetime(2025, 1, 29).date(),
        datetime(2025, 3, 19).date(),
        datetime(2025, 5, 7).date(),
        datetime(2025, 6, 18).date(),
        datetime(2025, 7, 30).date(),
        datetime(2025, 9, 17).date(),
        datetime(2025, 11, 5).date(),
        datetime(2025, 12, 17).date(),
        # 2026
        datetime(2026, 1, 28).date(),
        datetime(2026, 3, 18).date(),
        datetime(2026, 5, 6).date(),
        datetime(2026, 6, 17).date(),
        datetime(2026, 7, 29).date(),
        datetime(2026, 9, 16).date(),
        datetime(2026, 11, 4).date(),
        datetime(2026, 12, 16).date(),
    }

    def is_holiday(self, dt: Optional[datetime] = None) -> bool:
        """Is today a market holiday?"""
        if dt is None:
            dt = datetime.now(ET)
        return dt.date() in self.HOLIDAYS

    def is_fomc_day(self, dt: Optional[datetime] = None) -> bool:
        """Is today an FOMC decision day?"""
        if dt is None:
            dt = datetime.now(ET)
        return dt.date() in self.FOMC_DATES

    def is_fomc_week(self, dt: Optional[datetime] = None) -> bool:
        """Is this an FOMC week?"""
        if dt is None:
            dt = datetime.now(ET)

        # Check if any FOMC date falls in this week
        week_start = dt - timedelta(days=dt.weekday())
        week_end = week_start + timedelta(days=6)

        for fomc_date in self.FOMC_DATES:
            if week_start.date() <= fomc_date <= week_end.date():
                return True
        return False

    def is_opex_day(self, dt: Optional[datetime] = None) -> bool:
        """Is today options expiration (3rd Friday)?"""
        if dt is None:
            dt = datetime.now(ET)

        if dt.weekday() != 4:  # Not Friday
            return False

        # Check if 3rd Friday (day 15-21)
        return 15 <= dt.day <= 21

    def is_triple_witching(self, dt: Optional[datetime] = None) -> bool:
        """Is today triple witching (3rd Friday of Mar/Jun/Sep/Dec)?"""
        if dt is None:
            dt = datetime.now(ET)

        if not self.is_opex_day(dt):
            return False

        return dt.month in (3, 6, 9, 12)

    def get_holiday_name(self, dt: Optional[datetime] = None) -> Optional[str]:
        """Get holiday name if today is a holiday."""
        if dt is None:
            dt = datetime.now(ET)
        return self.HOLIDAYS.get(dt.date())


class SeasonalAwareness:
    """Knows about market seasonality and special periods."""

    # Earnings season months (peak earnings weeks)
    EARNINGS_MONTHS = {1, 4, 7, 10}  # Jan, Apr, Jul, Oct

    def get_season(self, dt: Optional[datetime] = None) -> Season:
        """Get current market season."""
        if dt is None:
            dt = datetime.now(ET)

        month = dt.month
        day = dt.day

        # Check specific seasons
        if month == 12 and day >= 24:
            return Season.SANTA_RALLY

        if month == 12 or (month == 11 and day >= 15):
            return Season.TAX_LOSS_SEASON

        if month == 1:
            return Season.JANUARY_EFFECT

        if 5 <= month <= 10:
            if month in (7, 8):
                return Season.SUMMER_DOLDRUMS
            return Season.SELL_IN_MAY

        if month in self.EARNINGS_MONTHS:
            # First 3 weeks of earnings months
            if day <= 21:
                return Season.EARNINGS_SEASON

        return Season.NORMAL

    def is_earnings_season(self, dt: Optional[datetime] = None) -> bool:
        """Are we in earnings season?"""
        return self.get_season(dt) == Season.EARNINGS_SEASON

    def get_seasonal_bias(self, dt: Optional[datetime] = None) -> Dict[str, Any]:
        """Get seasonal bias and statistics."""
        if dt is None:
            dt = datetime.now(ET)

        season = self.get_season(dt)

        # Historical biases (simplified)
        biases = {
            Season.NORMAL: {"bias": "neutral", "strength": 0.0},
            Season.EARNINGS_SEASON: {"bias": "volatile", "strength": 0.3},
            Season.FOMC_WEEK: {"bias": "cautious", "strength": 0.5},
            Season.TRIPLE_WITCHING: {"bias": "volatile", "strength": 0.7},
            Season.TAX_LOSS_SEASON: {"bias": "bearish_small_caps", "strength": 0.4},
            Season.JANUARY_EFFECT: {"bias": "bullish_small_caps", "strength": 0.3},
            Season.SELL_IN_MAY: {"bias": "cautious", "strength": 0.2},
            Season.SANTA_RALLY: {"bias": "bullish", "strength": 0.4},
            Season.SUMMER_DOLDRUMS: {"bias": "low_volume", "strength": 0.3},
        }

        return {
            "season": season.value,
            **biases.get(season, {"bias": "neutral", "strength": 0.0})
        }


class ContextBuilder:
    """Builds complete market context."""

    def __init__(self):
        self.time_awareness = TimeAwareness()
        self.calendar = MarketCalendarAwareness()
        self.seasonal = SeasonalAwareness()

    def get_context(self, dt: Optional[datetime] = None) -> MarketContext:
        """Get complete market context."""
        if dt is None:
            dt = self.time_awareness.now()

        phase = self.time_awareness.get_phase(dt)
        is_holiday = self.calendar.is_holiday(dt)
        is_weekend = dt.weekday() >= 5

        if is_holiday:
            phase = MarketPhase.HOLIDAY

        season = self.seasonal.get_season(dt)
        if self.calendar.is_fomc_week(dt):
            season = Season.FOMC_WEEK

        # Determine work mode
        work_mode = self._determine_work_mode(phase, is_weekend, is_holiday)

        # Build recommended actions
        actions = self._get_recommended_actions(phase, work_mode, season, dt)

        # Trading allowed?
        trading_allowed = self._is_trading_allowed(phase, is_holiday)

        return MarketContext(
            timestamp=dt,
            phase=phase,
            season=season,
            work_mode=work_mode,
            is_market_open=self.time_awareness.is_market_open(dt),
            is_weekend=is_weekend,
            is_holiday=is_holiday,
            day_of_week=dt.strftime("%A"),
            minutes_to_open=self.time_awareness.minutes_to_open(dt),
            minutes_to_close=self.time_awareness.minutes_to_close(dt),
            is_fomc_day=self.calendar.is_fomc_day(dt),
            is_earnings_heavy=self.seasonal.is_earnings_season(dt),
            is_opex_day=self.calendar.is_opex_day(dt),
            is_early_close=dt.date() in self.time_awareness._early_close_dates,
            trading_allowed=trading_allowed,
            recommended_actions=actions,
        )

    def _determine_work_mode(
        self, phase: MarketPhase, is_weekend: bool, is_holiday: bool
    ) -> WorkMode:
        """Determine what Kobe should be doing."""
        if is_weekend:
            return WorkMode.DEEP_RESEARCH

        if is_holiday:
            return WorkMode.DEEP_RESEARCH

        mode_map = {
            MarketPhase.PRE_MARKET_EARLY: WorkMode.RESEARCH,
            MarketPhase.PRE_MARKET_ACTIVE: WorkMode.MONITORING,
            MarketPhase.MARKET_OPENING: WorkMode.MONITORING,  # Don't trade opening
            MarketPhase.MARKET_MORNING: WorkMode.ACTIVE_TRADING,
            MarketPhase.MARKET_LUNCH: WorkMode.RESEARCH,  # Choppy, do research
            MarketPhase.MARKET_AFTERNOON: WorkMode.ACTIVE_TRADING,
            MarketPhase.MARKET_CLOSE: WorkMode.MONITORING,
            MarketPhase.AFTER_HOURS: WorkMode.LEARNING,
            MarketPhase.NIGHT: WorkMode.OPTIMIZATION,
        }

        return mode_map.get(phase, WorkMode.MAINTENANCE)

    def _is_trading_allowed(self, phase: MarketPhase, is_holiday: bool) -> bool:
        """Is trading allowed right now?"""
        if is_holiday:
            return False

        # Only trade during morning and afternoon sessions
        return phase in (MarketPhase.MARKET_MORNING, MarketPhase.MARKET_AFTERNOON)

    def _get_recommended_actions(
        self, phase: MarketPhase, work_mode: WorkMode, season: Season, dt: datetime
    ) -> List[str]:
        """Get recommended actions for current context."""
        actions = []

        if work_mode == WorkMode.ACTIVE_TRADING:
            actions.append("Run scanner for signals")
            actions.append("Check existing positions")
            actions.append("Monitor kill zones")

        elif work_mode == WorkMode.MONITORING:
            actions.append("Watch positions P&L")
            actions.append("Prepare watchlist")
            if phase == MarketPhase.MARKET_OPENING:
                actions.append("Observe opening range (NO trades)")

        elif work_mode == WorkMode.RESEARCH:
            actions.append("Run strategy backtests")
            actions.append("Explore new features")
            actions.append("Test parameter variations")

        elif work_mode == WorkMode.LEARNING:
            actions.append("Analyze today's trades")
            actions.append("Update episodic memory")
            actions.append("Review what worked/failed")

        elif work_mode == WorkMode.OPTIMIZATION:
            actions.append("Run walk-forward optimization")
            actions.append("Test strategy combinations")
            actions.append("Feature importance analysis")

        elif work_mode == WorkMode.DEEP_RESEARCH:
            actions.append("Extended backtesting (full history)")
            actions.append("Strategy discovery experiments")
            actions.append("Cross-asset correlation studies")
            actions.append("ML model retraining")
            actions.append("Knowledge consolidation")

        elif work_mode == WorkMode.MAINTENANCE:
            actions.append("Data quality checks")
            actions.append("Clean up old logs")
            actions.append("Verify system health")

        # Season-specific actions
        if season == Season.EARNINGS_SEASON:
            actions.append("Check earnings calendar")
            actions.append("Avoid holding through earnings")

        if season == Season.FOMC_WEEK:
            actions.append("Reduce position sizes")
            actions.append("Avoid new positions before FOMC")

        return actions


# Convenience function
def get_context(dt: Optional[datetime] = None) -> MarketContext:
    """Get current market context."""
    return ContextBuilder().get_context(dt)


if __name__ == "__main__":
    # Demo
    ctx = get_context()
    print(f"Current Context:")
    print(f"  Time: {ctx.timestamp}")
    print(f"  Phase: {ctx.phase.value}")
    print(f"  Season: {ctx.season.value}")
    print(f"  Work Mode: {ctx.work_mode.value}")
    print(f"  Market Open: {ctx.is_market_open}")
    print(f"  Trading Allowed: {ctx.trading_allowed}")
    print(f"  Recommended Actions:")
    for action in ctx.recommended_actions:
        print(f"    - {action}")
