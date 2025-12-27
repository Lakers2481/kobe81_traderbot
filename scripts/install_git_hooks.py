#!/usr/bin/env python3
from __future__ import annotations

"""
Install a post-commit git hook that updates docs/STATUS.md from the journal/state.

Usage:
  python scripts/install_git_hooks.py
"""

import os
from pathlib import Path


HOOK = """#!/bin/sh
# Kobe post-commit hook: update STATUS.md snapshot
PY=python
ROOT_DIR=$(git rev-parse --show-toplevel)
cd "$ROOT_DIR" || exit 0
$PY scripts/update_status_md.py >/dev/null 2>&1 || true
exit 0
"""


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    hooks = repo / '.git' / 'hooks'
    if not hooks.exists():
        print('Not a git repository (no .git/hooks). Skipping.')
        return
    pc = hooks / 'post-commit'
    pc.write_text(HOOK, encoding='utf-8')
    try:
        os.chmod(pc, 0o755)
    except Exception:
        pass
    print('Installed git post-commit hook:', pc)


if __name__ == '__main__':
    main()

