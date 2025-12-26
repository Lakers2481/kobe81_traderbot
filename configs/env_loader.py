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
    return loaded

