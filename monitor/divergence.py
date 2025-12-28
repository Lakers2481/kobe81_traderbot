from __future__ import annotations

"""
Divergence Guard

Compares shadow decisions (Top-3/TOTD) with actual submissions logged by OMS.
Raises alerts on mismatches or unexplained skips.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DivergenceResult:
    date: str
    totd_symbol: Optional[str]
    submitted_symbol: Optional[str]
    divergent: bool
    reason: str


def _load_shadow_totd(day: str) -> Optional[str]:
    p = ROOT / 'logs' / 'shadow' / f'trade_of_day_{day}.csv'
    if not p.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(p)
        if df.empty:
            return None
        return str(df.iloc[0].get('symbol') or '')
    except Exception:
        return None


def _load_submitted_symbol(day: str) -> Optional[str]:
    """Return the first submitted symbol in logs/events.jsonl for the day."""
    ev = ROOT / 'logs' / 'events.jsonl'
    if not ev.exists():
        return None
    sub = None
    try:
        with ev.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                ts = str(j.get('ts') or j.get('timestamp') or '')
                if not ts:
                    continue
                if ts[:10] != day:
                    continue
                if j.get('event') == 'order_submit' or 'order_submit' in str(j.get('msg','')):
                    sub = str(j.get('symbol') or j.get('sym') or '')
                    if sub:
                        return sub
    except Exception:
        return None
    return sub


def compare(day: str) -> DivergenceResult:
    totd = _load_shadow_totd(day)
    sub = _load_submitted_symbol(day)
    if not totd and not sub:
        return DivergenceResult(day, None, None, False, 'No TOTD and no submissions')
    if totd and not sub:
        return DivergenceResult(day, totd, None, True, 'Shadow had TOTD but nothing was submitted')
    if sub and not totd:
        return DivergenceResult(day, None, sub, True, 'Submission without a shadow TOTD')
    div = (str(totd).upper() != str(sub).upper())
    return DivergenceResult(day, totd, sub, div, 'Symbol mismatch' if div else 'OK')


def _send_telegram(msg: str) -> None:
    try:
        from core.alerts import send_telegram
        send_telegram(msg)
    except Exception:
        pass


def cli() -> None:
    import argparse
    from zoneinfo import ZoneInfo
    ap = argparse.ArgumentParser(description='Divergence guard compare shadow vs live submissions')
    ap.add_argument('--date', type=str, default=None)
    ap.add_argument('--telegram', action='store_true')
    args = ap.parse_args()
    day = args.date or datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    res = compare(day)
    print(json.dumps(res.__dict__, indent=2))
    if args.telegram and res.divergent:
        try:
            from core.clock.tz_utils import fmt_ct, now_et
            now = now_et(); stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
        except Exception:
            stamp = None
        note = f" [{stamp}]" if stamp else ''
        _send_telegram(f"<b>Divergence</b> {res.date}: shadow TOTD={res.totd_symbol}, submitted={res.submitted_symbol} ({res.reason}){note}")


if __name__ == '__main__':
    cli()
