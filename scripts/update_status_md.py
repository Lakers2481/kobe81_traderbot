#!/usr/bin/env python3
from __future__ import annotations

"""
Update docs/STATUS.md timestamp only.

The STATUS.md file is a comprehensive alignment document for all AI and human
collaborators. This script only updates the timestamp line, preserving all
other content.

Usage:
    python scripts/update_status_md.py
"""

import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    status = ROOT / 'docs' / 'STATUS.md'

    if not status.exists():
        # Create minimal STATUS.md if it doesn't exist
        content = f"""# Kobe81 Traderbot - STATUS

> **Last Updated:** {now}
> **Verified By:** Initial Bootstrap

---

## CRITICAL: Strategy Alignment

### Active Strategies (ONLY THESE TWO) - v2.2 QUANT INTERVIEW READY

| Strategy | Type | Entry Condition | Win Rate | Profit Factor |
|----------|------|-----------------|----------|---------------|
| **IBS+RSI v2.2** | Mean Reversion | IBS < 0.08 AND RSI(2) < 5 AND Close > SMA(200) | 59.9% | 1.46 |
| **Turtle Soup v2.2** | Mean Reversion | Sweep > 0.3 ATR below 20-day low (3+ bars aged), revert inside | 61.0% | 1.37 |

### Deprecated Strategies (DO NOT USE)

| Strategy | Status | Notes |
|----------|--------|-------|
| ~~Donchian Breakout~~ | **REMOVED** | Deleted from codebase |

---

For complete replication steps and evidence, see docs/STATUS.md.
"""
        status.write_text(content, encoding='utf-8')
        print(f'Created: {status}')
    else:
        # Update timestamp only
        content = status.read_text(encoding='utf-8')

        # Update the "Last Updated" line
        content = re.sub(
            r'\*\*Last Updated:\*\* \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC',
            f'**Last Updated:** {now}',
            content
        )

        status.write_text(content, encoding='utf-8')
        print(f'Updated timestamp: {status}')

    # Archive to history
    hist = ROOT / 'docs' / 'history'
    hist.mkdir(parents=True, exist_ok=True)
    archive = hist / f'status_{datetime.utcnow().strftime("%Y%m%d_%H%M")}.md'
    archive.write_text(status.read_text(encoding='utf-8'), encoding='utf-8')
    print(f'Archived: {archive}')


if __name__ == '__main__':
    main()
