from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


JOURNAL_PATH = Path("state/journal.jsonl")


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _season_for_month(m: int) -> str:
    # Meteorological seasons (Northern Hemisphere)
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    return "fall"


def awareness_tags(ts: Optional[datetime] = None) -> Dict[str, Any]:
    ts = ts or datetime.utcnow()
    return {
        "utc_ts": ts.isoformat(),
        "dow": ts.strftime("%A"),
        "dom": ts.day,
        "month": ts.month,
        "quarter": (ts.month - 1) // 3 + 1,
        "week_of_year": int(ts.strftime("%U")),
        "season": _season_for_month(ts.month),
    }


def append_journal(event: str, payload: Optional[Dict[str, Any]] = None, tags: Optional[Dict[str, Any]] = None) -> None:
    record: Dict[str, Any] = {"event": event, **awareness_tags()}
    if payload:
        record["payload"] = payload
    if tags:
        record["tags"] = tags
    _ensure_dir(JOURNAL_PATH)
    with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

