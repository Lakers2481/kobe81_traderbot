from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def load_env(path: str | Path) -> Dict[str, str]:
    """
    Minimal .env loader: reads KEY=VALUE pairs and sets os.environ.
    - Ignores blank lines and lines starting with '#'
    - Trims surrounding quotes from values
    Returns a dict of keys loaded.
    """
    p = Path(path)
    loaded: Dict[str, str] = {}
    if not p.exists():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ[key] = val
        loaded[key] = val

    # Backward-compatible alias mapping
    # Map ALPACA_API_KEY -> ALPACA_API_KEY_ID, ALPACA_SECRET_KEY -> ALPACA_API_SECRET_KEY if missing
    if 'ALPACA_API_KEY_ID' not in os.environ and 'ALPACA_API_KEY' in os.environ:
        os.environ['ALPACA_API_KEY_ID'] = os.environ['ALPACA_API_KEY']
        loaded['ALPACA_API_KEY_ID'] = os.environ['ALPACA_API_KEY']
    if 'ALPACA_API_SECRET_KEY' not in os.environ:
        if 'ALPACA_SECRET_KEY' in os.environ:
            os.environ['ALPACA_API_SECRET_KEY'] = os.environ['ALPACA_SECRET_KEY']
            loaded['ALPACA_API_SECRET_KEY'] = os.environ['ALPACA_SECRET_KEY']
    return loaded
