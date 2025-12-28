from __future__ import annotations

"""
Prohibited List (optional helper).

Allows simple checks to prohibit trading for a symbol/date due to:
- Earnings proximity (configurable) — wrapper around core.earnings_filter
- News events / sentiment (stub)
- Volatility spikes (stub)
- Auto-expire entries

This is designed as a facade the production code can call if desired.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple


@dataclass
class ProhibitedReason:
    code: str
    detail: str


def check_earnings(symbol: str, on_date: date) -> List[ProhibitedReason]:
    try:
        # If core earnings filter would exclude the date, return a reason
        from core.earnings_filter import is_near_earnings
        if is_near_earnings(symbol, on_date):
            return [ProhibitedReason(code="earnings", detail="near_earnings_window")]
    except Exception:
        pass
    return []


def check_news(symbol: str, on_date: date) -> List[ProhibitedReason]:
    # Placeholder for future: high-impact news ban list
    return []


def check_volatility(symbol: str, on_date: date) -> List[ProhibitedReason]:
    # Placeholder for realized/implied vol spike checks
    return []


def prohibited_reasons(symbol: str, on_date: date) -> List[ProhibitedReason]:
    out: List[ProhibitedReason] = []
    out.extend(check_earnings(symbol, on_date))
    out.extend(check_news(symbol, on_date))
    out.extend(check_volatility(symbol, on_date))
    return out


def is_prohibited(symbol: str, on_date: date) -> Tuple[bool, List[ProhibitedReason]]:
    reasons = prohibited_reasons(symbol, on_date)
    return (len(reasons) > 0), reasons

