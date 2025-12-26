from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "events.jsonl"


def jlog(event: str, level: str = "INFO", **fields: Any) -> None:
    rec: Dict[str, Any] = {
        "ts": datetime.utcnow().isoformat(),
        "level": level,
        "event": event,
        **fields,
    }
    line = json.dumps(rec, default=str)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    # Also echo concise line to console
    print(f"[{rec['level']}] {rec['event']} | {fields}")

