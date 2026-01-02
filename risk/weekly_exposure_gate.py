"""
Weekly Exposure Gate - Professional Portfolio Allocation System

This module implements institutional-grade position and exposure management:
- 40% max weekly exposure (HARD cap)
- 20% max daily exposure (HARD cap)
- 10% max per position (HARD cap)
- 2 positions per day (HARD cap)
- 10 positions per week (SOFT cap - warning only)

Budget is freed when positions close, available at next scan.
Week resets Friday after close ONLY if all positions are closed.

Quant Interview Ready: Implements defense-in-depth risk management.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytz

from core.structured_log import jlog

# Constants
ET = pytz.timezone('America/New_York')
STATE_FILE = Path('state/weekly_budget.json')


@dataclass
class PositionEntry:
    """Record of a position entered this week."""
    symbol: str
    entry_date: str  # ISO format
    entry_time: str  # ISO format
    notional: float
    pct_of_account: float
    status: str = 'open'  # open, closed
    exit_date: Optional[str] = None
    exit_reason: Optional[str] = None  # target, stop, time_stop, manual


@dataclass
class WeeklyBudgetState:
    """Persistent state for weekly budget tracking."""
    week_start: str  # ISO date
    week_end: str  # ISO date (Friday)
    account_equity_at_week_start: float
    max_weekly_pct: float = 0.40
    max_daily_pct: float = 0.20
    max_per_position_pct: float = 0.10
    min_per_position_pct: float = 0.05
    max_positions_per_day: int = 2
    max_positions_per_week: int = 10  # Soft cap

    # Current state
    current_exposure_pct: float = 0.0
    positions_opened: List[Dict] = field(default_factory=list)
    positions_closed: List[Dict] = field(default_factory=list)
    daily_entries: Dict[str, int] = field(default_factory=dict)

    # Budget recovery
    budget_freed_pending: float = 0.0
    budget_freed_available_at: Optional[str] = None
    unused_daily_budget_carried: float = 0.0

    # Metadata
    can_reset: bool = True
    reset_blocked_reason: Optional[str] = None
    last_updated: str = field(default_factory=lambda: datetime.now(ET).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'WeeklyBudgetState':
        return cls(**data)


class WeeklyExposureGate:
    """
    Professional weekly budget management system.

    Implements institutional-grade exposure management:
    - Rolling 40% weekly cap on CONCURRENT exposure
    - 20% daily cap (max new entries per day)
    - Budget freed when positions close
    - State persists across restarts
    """

    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self.state: WeeklyBudgetState = self._load_or_init_state()
        self._maybe_reset_week()

    def _load_or_init_state(self) -> WeeklyBudgetState:
        """Load state from disk or initialize new week."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                jlog('weekly_budget_loaded', file=str(self.state_file))
                return WeeklyBudgetState.from_dict(data)
            except Exception as e:
                jlog('weekly_budget_load_error', error=str(e), level='WARN')

        # Initialize new state
        return self._create_new_week_state()

    def _create_new_week_state(self, account_equity: float = 105000.0) -> WeeklyBudgetState:
        """Create fresh state for new trading week."""
        now = datetime.now(ET)

        # Find Monday of this week
        days_since_monday = now.weekday()
        monday = now.date() - timedelta(days=days_since_monday)
        friday = monday + timedelta(days=4)

        state = WeeklyBudgetState(
            week_start=monday.isoformat(),
            week_end=friday.isoformat(),
            account_equity_at_week_start=account_equity,
        )

        jlog('weekly_budget_initialized',
             week_start=state.week_start,
             week_end=state.week_end,
             account_equity=account_equity)

        return state

    def _maybe_reset_week(self) -> None:
        """Check if we should reset for new week (Friday after close, if flat)."""
        now = datetime.now(ET)
        today = now.date()
        week_end = date.fromisoformat(self.state.week_end)

        # Check if we're past Friday close (4 PM ET)
        is_past_week = today > week_end
        is_friday_after_close = (
            today == week_end and
            now.hour >= 16
        )

        if is_past_week or is_friday_after_close:
            # Check if we can reset (all positions closed)
            open_positions = [p for p in self.state.positions_opened
                           if p.get('status') == 'open']

            if len(open_positions) == 0:
                # Safe to reset
                jlog('weekly_budget_reset', reason='all_positions_closed')
                self.state = self._create_new_week_state(
                    account_equity=self.state.account_equity_at_week_start
                )
                self._save_state()
            else:
                # Can't reset - positions still open
                self.state.can_reset = False
                self.state.reset_blocked_reason = f"open_positions: {[p['symbol'] for p in open_positions]}"
                jlog('weekly_budget_reset_blocked',
                     reason=self.state.reset_blocked_reason,
                     level='WARN')

    def _save_state(self) -> None:
        """Persist state to disk."""
        self.state.last_updated = datetime.now(ET).isoformat()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
        jlog('weekly_budget_saved', file=str(self.state_file))

    def update_account_equity(self, equity: float) -> None:
        """Update account equity (call at start of each session)."""
        self.state.account_equity_at_week_start = equity
        self._recalculate_exposure()
        self._save_state()

    def _recalculate_exposure(self) -> None:
        """Recalculate current exposure from open positions."""
        open_notional = sum(
            p.get('notional', 0)
            for p in self.state.positions_opened
            if p.get('status') == 'open'
        )
        equity = self.state.account_equity_at_week_start
        self.state.current_exposure_pct = open_notional / equity if equity > 0 else 0

    def _get_today_key(self) -> str:
        """Get today's date as string key."""
        return datetime.now(ET).date().isoformat()

    def _get_entries_today(self) -> int:
        """Get number of entries made today."""
        today = self._get_today_key()
        return self.state.daily_entries.get(today, 0)

    def _process_freed_budget(self) -> None:
        """Process any pending freed budget that's now available."""
        if self.state.budget_freed_pending > 0 and self.state.budget_freed_available_at:
            available_time = datetime.fromisoformat(self.state.budget_freed_available_at)
            now = datetime.now(ET)

            if now >= available_time:
                # Budget is now available
                jlog('weekly_budget_freed',
                     amount=self.state.budget_freed_pending,
                     available_at=self.state.budget_freed_available_at)
                self.state.budget_freed_pending = 0
                self.state.budget_freed_available_at = None
                self._recalculate_exposure()
                self._save_state()

    def check_can_enter(
        self,
        symbol: str,
        notional: float,
        account_equity: float
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be entered.

        Returns: (can_enter, reason)

        Checks (in order):
        1. Daily entry count (max 2)
        2. Weekly exposure cap (40%)
        3. Daily exposure cap (20%)
        4. Per-position cap (10%)
        5. Position count warning (10/week soft cap)
        """
        self._process_freed_budget()

        # Update equity if different
        if abs(account_equity - self.state.account_equity_at_week_start) > 100:
            self.update_account_equity(account_equity)

        pct_of_account = notional / account_equity

        # Check 1: Daily entry count (HARD)
        entries_today = self._get_entries_today()
        if entries_today >= self.state.max_positions_per_day:
            return False, f"daily_entries_exhausted: {entries_today}/{self.state.max_positions_per_day}"

        # Check 2: Weekly exposure (HARD)
        new_weekly_pct = self.state.current_exposure_pct + pct_of_account
        if new_weekly_pct > self.state.max_weekly_pct:
            return False, f"weekly_exposure_exceeded: {new_weekly_pct:.1%} > {self.state.max_weekly_pct:.0%}"

        # Check 3: Daily exposure (HARD)
        today_notional = sum(
            p.get('notional', 0)
            for p in self.state.positions_opened
            if p.get('entry_date') == self._get_today_key() and p.get('status') == 'open'
        )
        today_pct = today_notional / account_equity
        new_daily_pct = today_pct + pct_of_account
        if new_daily_pct > self.state.max_daily_pct:
            return False, f"daily_exposure_exceeded: {new_daily_pct:.1%} > {self.state.max_daily_pct:.0%}"

        # Check 4: Per-position cap (HARD)
        if pct_of_account > self.state.max_per_position_pct:
            return False, f"position_too_large: {pct_of_account:.1%} > {self.state.max_per_position_pct:.0%}"

        # Check 5: Position count (SOFT - warning only)
        open_count = len([p for p in self.state.positions_opened if p.get('status') == 'open'])
        if open_count >= self.state.max_positions_per_week:
            jlog('weekly_position_soft_cap',
                 count=open_count,
                 max=self.state.max_positions_per_week,
                 level='WARN')
            # Still allow, but log warning
        elif open_count >= 8:  # Warning at 8
            jlog('weekly_position_warning',
                 count=open_count,
                 remaining=self.state.max_positions_per_week - open_count,
                 level='INFO')

        # Check 6: Duplicate symbol
        open_symbols = [p['symbol'] for p in self.state.positions_opened if p.get('status') == 'open']
        if symbol in open_symbols:
            return False, f"duplicate_symbol: already have open position in {symbol}"

        return True, "approved"

    def record_entry(
        self,
        symbol: str,
        notional: float,
        account_equity: float,
        entry_time: Optional[datetime] = None
    ) -> None:
        """Record a new position entry."""
        if entry_time is None:
            entry_time = datetime.now(ET)

        pct_of_account = notional / account_equity

        entry = {
            'symbol': symbol,
            'entry_date': entry_time.date().isoformat(),
            'entry_time': entry_time.isoformat(),
            'notional': notional,
            'pct_of_account': pct_of_account,
            'status': 'open',
        }

        self.state.positions_opened.append(entry)

        # Update daily count
        today = entry_time.date().isoformat()
        self.state.daily_entries[today] = self.state.daily_entries.get(today, 0) + 1

        # Recalculate exposure
        self._recalculate_exposure()

        jlog('weekly_budget_entry',
             symbol=symbol,
             notional=notional,
             pct=pct_of_account,
             new_exposure=self.state.current_exposure_pct,
             entries_today=self.state.daily_entries[today])

        self._save_state()

    def record_exit(
        self,
        symbol: str,
        exit_reason: str = 'unknown',
        exit_time: Optional[datetime] = None
    ) -> None:
        """
        Record a position exit and schedule budget recovery.

        Budget will be available at next scan time (10:30 or 15:55 ET).
        """
        if exit_time is None:
            exit_time = datetime.now(ET)

        # Find the position
        for pos in self.state.positions_opened:
            if pos.get('symbol') == symbol and pos.get('status') == 'open':
                pos['status'] = 'closed'
                pos['exit_date'] = exit_time.isoformat()
                pos['exit_reason'] = exit_reason

                # Schedule budget recovery for next scan
                freed_notional = pos.get('notional', 0)
                next_scan = self._get_next_scan_time(exit_time)

                self.state.budget_freed_pending += freed_notional
                self.state.budget_freed_available_at = next_scan.isoformat()

                # Move to closed list
                self.state.positions_closed.append(pos)

                jlog('weekly_budget_exit',
                     symbol=symbol,
                     reason=exit_reason,
                     freed=freed_notional,
                     available_at=next_scan.isoformat())

                break

        self._recalculate_exposure()
        self._save_state()

    def _get_next_scan_time(self, from_time: datetime) -> datetime:
        """Get next scan time (10:30 or 15:55 ET)."""
        scan_times = [(10, 30), (15, 55)]

        for hour, minute in scan_times:
            scan = from_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if scan > from_time:
                return scan

        # Next day 10:30
        next_day = from_time.date() + timedelta(days=1)
        return datetime.combine(next_day, datetime.min.time().replace(hour=10, minute=30), tzinfo=ET)

    def get_available_daily_budget(self, account_equity: float) -> float:
        """Get remaining daily budget in dollars."""
        self._process_freed_budget()

        today_notional = sum(
            p.get('notional', 0)
            for p in self.state.positions_opened
            if p.get('entry_date') == self._get_today_key() and p.get('status') == 'open'
        )

        max_daily = account_equity * self.state.max_daily_pct
        return max(0, max_daily - today_notional)

    def get_available_weekly_budget(self, account_equity: float) -> float:
        """Get remaining weekly budget in dollars."""
        self._process_freed_budget()

        open_notional = sum(
            p.get('notional', 0)
            for p in self.state.positions_opened
            if p.get('status') == 'open'
        )

        max_weekly = account_equity * self.state.max_weekly_pct
        return max(0, max_weekly - open_notional)

    def get_current_exposure_pct(self) -> float:
        """Get current exposure as percentage of account."""
        self._recalculate_exposure()
        return self.state.current_exposure_pct

    def get_status(self) -> Dict:
        """Get comprehensive status for display."""
        self._process_freed_budget()
        self._recalculate_exposure()

        open_positions = [p for p in self.state.positions_opened if p.get('status') == 'open']
        entries_today = self._get_entries_today()

        return {
            'week': f"{self.state.week_start} to {self.state.week_end}",
            'exposure': {
                'current_pct': f"{self.state.current_exposure_pct:.1%}",
                'max_weekly_pct': f"{self.state.max_weekly_pct:.0%}",
                'max_daily_pct': f"{self.state.max_daily_pct:.0%}",
            },
            'positions': {
                'open': len(open_positions),
                'open_symbols': [p['symbol'] for p in open_positions],
                'total_this_week': len(self.state.positions_opened),
                'closed_this_week': len(self.state.positions_closed),
            },
            'daily': {
                'entries_today': entries_today,
                'max_per_day': self.state.max_positions_per_day,
                'remaining': self.state.max_positions_per_day - entries_today,
            },
            'budget': {
                'freed_pending': self.state.budget_freed_pending,
                'freed_available_at': self.state.budget_freed_available_at,
                'unused_carried': self.state.unused_daily_budget_carried,
            },
            'can_reset': self.state.can_reset,
            'reset_blocked_reason': self.state.reset_blocked_reason,
            'last_updated': self.state.last_updated,
        }


# Singleton instance
_gate_instance: Optional[WeeklyExposureGate] = None

def get_weekly_exposure_gate() -> WeeklyExposureGate:
    """Get singleton instance of WeeklyExposureGate."""
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = WeeklyExposureGate()
    return _gate_instance
