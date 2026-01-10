from __future__ import annotations

"""
Compliance audit trail writer.

Writes JSONL records to logs/compliance.jsonl and also emits structured logs
for aggregation. This is separate from the hash-chain audit, which captures
order/audit events at a lower level.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any


LOG_PATH = Path("logs/compliance.jsonl")


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def write_event(kind: str, payload: Dict[str, Any]) -> None:
    _ensure_dir(LOG_PATH)
    rec = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "kind": kind,
        **payload,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        import json
        f.write(json.dumps(rec) + "\n")
    try:
        from core.structured_log import jlog
        jlog(f"compliance_{kind}", **payload)
    except Exception:
        pass

