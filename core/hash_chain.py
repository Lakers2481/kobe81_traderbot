from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Any


CHAIN_FILE = Path("state/hash_chain.jsonl")


def _hash_payload(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def append_block(record: Dict[str, Any]) -> str:
    CHAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    prev_hash = None
    if CHAIN_FILE.exists():
        last = None
        for line in CHAIN_FILE.read_text(encoding='utf-8').splitlines():
            if line.strip():
                last = json.loads(line)
        if last:
            prev_hash = last.get('this_hash')
    block = {
        "prev_hash": prev_hash,
        "payload": record,
    }
    block["this_hash"] = _hash_payload(block)
    with CHAIN_FILE.open('a', encoding='utf-8') as f:
        f.write(json.dumps(block) + "\n")
    return block["this_hash"]


def verify_chain() -> bool:
    if not CHAIN_FILE.exists():
        return True
    prev = None
    for i, line in enumerate(CHAIN_FILE.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        block = json.loads(line)
        if block.get('prev_hash') != prev:
            return False
        recomputed = _hash_payload({k: block[k] for k in block if k != 'this_hash'})
        if recomputed != block.get('this_hash'):
            return False
        prev = block.get('this_hash')
    return True

