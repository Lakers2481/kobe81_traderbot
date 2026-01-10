"""
State file helpers for testing.

Provides utilities for:
- Creating test state directories
- Creating/modifying state files (positions, orders, hash chain)
- Verifying hash chain integrity
- Intentionally corrupting files for recovery tests
"""

import json
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


def create_test_state_dir(tmp_path: Path) -> Path:
    """
    Create a complete test state directory structure.

    Args:
        tmp_path: pytest tmp_path fixture

    Returns:
        Path to state directory
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (state_dir / "watchlist").mkdir(exist_ok=True)
    (state_dir / "cognitive").mkdir(exist_ok=True)
    (state_dir / "autonomous").mkdir(exist_ok=True)

    # Create initial empty files
    (state_dir / "positions.json").write_text("[]")
    (state_dir / "order_state.json").write_text("{}")
    (state_dir / "hash_chain.jsonl").write_text("")

    return state_dir


def create_positions_file(
    state_dir: Path,
    positions: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """
    Create positions.json file.

    Args:
        state_dir: State directory path
        positions: List of position dictionaries

    Returns:
        Path to positions file
    """
    if positions is None:
        positions = []

    positions_file = state_dir / "positions.json"
    positions_file.write_text(json.dumps(positions, indent=2))
    return positions_file


def create_order_state_file(
    state_dir: Path,
    orders: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Path:
    """
    Create order_state.json file.

    Args:
        state_dir: State directory path
        orders: Dict of order_id -> order details

    Returns:
        Path to order state file
    """
    if orders is None:
        orders = {}

    order_file = state_dir / "order_state.json"
    order_file.write_text(json.dumps(orders, indent=2))
    return order_file


def create_hash_chain_file(
    state_dir: Path,
    entries: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """
    Create hash_chain.jsonl file with valid hashes.

    Args:
        state_dir: State directory path
        entries: List of chain entries (will compute hashes)

    Returns:
        Path to hash chain file
    """
    hash_file = state_dir / "hash_chain.jsonl"

    if entries is None or len(entries) == 0:
        hash_file.write_text("")
        return hash_file

    lines = []
    prev_hash = "0" * 64  # Genesis block

    for entry in entries:
        # Compute hash
        entry_copy = entry.copy()
        entry_copy["prev_hash"] = prev_hash
        entry_str = json.dumps(entry_copy, sort_keys=True)
        curr_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry_copy["hash"] = curr_hash

        lines.append(json.dumps(entry_copy))
        prev_hash = curr_hash

    hash_file.write_text("\n".join(lines) + "\n")
    return hash_file


def create_idempotency_db(
    state_dir: Path,
    entries: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """
    Create idempotency SQLite database.

    Args:
        state_dir: State directory path
        entries: List of idempotency entries

    Returns:
        Path to database file
    """
    db_path = state_dir / "idempotency.db"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create table matching actual schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS idempotency_keys (
            key TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            created_at TEXT NOT NULL,
            order_id TEXT,
            status TEXT DEFAULT 'pending'
        )
    """)

    if entries:
        for entry in entries:
            cursor.execute(
                """
                INSERT OR REPLACE INTO idempotency_keys
                (key, symbol, side, created_at, order_id, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.get("key"),
                    entry.get("symbol", "TEST"),
                    entry.get("side", "buy"),
                    entry.get("created_at", datetime.now().isoformat()),
                    entry.get("order_id"),
                    entry.get("status", "pending"),
                )
            )

    conn.commit()
    conn.close()

    return db_path


def verify_hash_chain_integrity(hash_file: Path) -> Dict[str, Any]:
    """
    Verify hash chain file integrity.

    Args:
        hash_file: Path to hash_chain.jsonl

    Returns:
        Dict with 'valid' bool and 'errors' list
    """
    result = {"valid": True, "errors": [], "entries": 0}

    if not hash_file.exists():
        result["errors"].append("Hash chain file does not exist")
        result["valid"] = False
        return result

    content = hash_file.read_text().strip()
    if not content:
        return result  # Empty chain is valid

    lines = content.split("\n")
    prev_hash = "0" * 64

    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
            result["entries"] += 1

            # Verify prev_hash links
            if entry.get("prev_hash") != prev_hash:
                result["errors"].append(
                    f"Entry {i}: prev_hash mismatch (expected {prev_hash[:8]}..., got {entry.get('prev_hash', 'missing')[:8]}...)"
                )
                result["valid"] = False

            # Verify hash computation
            stored_hash = entry.pop("hash", None)
            entry_str = json.dumps(entry, sort_keys=True)
            computed_hash = hashlib.sha256(entry_str.encode()).hexdigest()

            if stored_hash != computed_hash:
                result["errors"].append(
                    f"Entry {i}: hash mismatch (stored {stored_hash[:8]}..., computed {computed_hash[:8]}...)"
                )
                result["valid"] = False

            prev_hash = stored_hash or computed_hash

        except json.JSONDecodeError as e:
            result["errors"].append(f"Entry {i}: Invalid JSON - {e}")
            result["valid"] = False

    return result


def corrupt_positions_file(state_dir: Path) -> None:
    """
    Intentionally corrupt positions.json for recovery testing.

    Args:
        state_dir: State directory path
    """
    positions_file = state_dir / "positions.json"
    positions_file.write_text("{ invalid json [")


def corrupt_json_file(file_path: Path, corruption_type: str = "syntax") -> None:
    """
    Corrupt a JSON file in various ways.

    Args:
        file_path: Path to file
        corruption_type: Type of corruption (syntax, truncate, binary)
    """
    if corruption_type == "syntax":
        file_path.write_text("{ broken: json [}")
    elif corruption_type == "truncate":
        content = file_path.read_text()
        file_path.write_text(content[:len(content)//2])
    elif corruption_type == "binary":
        file_path.write_bytes(b"\x00\xff\x00\xff" * 100)


def delete_state_file(state_dir: Path, filename: str) -> bool:
    """
    Delete a specific state file.

    Args:
        state_dir: State directory path
        filename: Name of file to delete

    Returns:
        True if file existed and was deleted
    """
    file_path = state_dir / filename
    if file_path.exists():
        file_path.unlink()
        return True
    return False


def create_kill_switch(state_dir: Path, reason: str = "Test kill switch") -> Path:
    """
    Create KILL_SWITCH file.

    Args:
        state_dir: State directory path
        reason: Reason for kill switch

    Returns:
        Path to kill switch file
    """
    kill_file = state_dir / "KILL_SWITCH"
    kill_file.write_text(json.dumps({
        "created_at": datetime.now().isoformat(),
        "reason": reason,
    }))
    return kill_file


def remove_kill_switch(state_dir: Path) -> bool:
    """
    Remove KILL_SWITCH file.

    Args:
        state_dir: State directory path

    Returns:
        True if file existed and was removed
    """
    kill_file = state_dir / "KILL_SWITCH"
    if kill_file.exists():
        kill_file.unlink()
        return True
    return False


def create_watchlist_file(
    state_dir: Path,
    watchlist: Optional[List[Dict[str, Any]]] = None,
    filename: str = "today_validated.json",
) -> Path:
    """
    Create watchlist file.

    Args:
        state_dir: State directory path
        watchlist: List of watchlist entries
        filename: Name of watchlist file

    Returns:
        Path to watchlist file
    """
    watchlist_dir = state_dir / "watchlist"
    watchlist_dir.mkdir(exist_ok=True)

    if watchlist is None:
        watchlist = []

    watchlist_file = watchlist_dir / filename
    watchlist_file.write_text(json.dumps({
        "created_at": datetime.now().isoformat(),
        "stocks": watchlist,
    }, indent=2))

    return watchlist_file
