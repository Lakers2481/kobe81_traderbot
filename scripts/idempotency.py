#!/usr/bin/env python3
"""
Idempotency Store Management for Kobe Trading System

Lists stored order IDs, checks for duplicates, and clears old entries.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from oms.idempotency_store import IdempotencyStore


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_DOTENV = "C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env"
STATE_DIR = ROOT / "state"
DB_PATH = STATE_DIR / "idempotency.sqlite"
RETENTION_DAYS = 7  # Default retention period


# -----------------------------------------------------------------------------
# Store Operations
# -----------------------------------------------------------------------------
def get_db_path() -> Path:
    """Get the path to the idempotency database."""
    return DB_PATH


def db_exists() -> bool:
    """Check if the database exists."""
    return DB_PATH.exists()


def get_all_entries() -> List[Dict[str, Any]]:
    """Get all entries from the idempotency store."""
    if not db_exists():
        return []

    entries = []
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            cur = con.execute(
                "SELECT decision_id, idempotency_key, created_at FROM idempotency ORDER BY created_at DESC"
            )
            for row in cur.fetchall():
                entries.append({
                    "decision_id": row["decision_id"],
                    "idempotency_key": row["idempotency_key"],
                    "created_at": row["created_at"]
                })
    except Exception:
        pass

    return entries


def get_entry_count() -> int:
    """Get total count of entries."""
    if not db_exists():
        return 0

    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.execute("SELECT COUNT(*) FROM idempotency")
            return cur.fetchone()[0]
    except Exception:
        return 0


def get_entries_by_age() -> Dict[str, int]:
    """Get count of entries by age bucket."""
    if not db_exists():
        return {}

    now = datetime.utcnow()
    buckets = {
        "last_hour": 0,
        "last_24h": 0,
        "last_7d": 0,
        "older": 0
    }

    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.execute("SELECT created_at FROM idempotency")
            for row in cur.fetchall():
                try:
                    created = datetime.fromisoformat(row[0].replace("Z", ""))
                    age = now - created

                    if age < timedelta(hours=1):
                        buckets["last_hour"] += 1
                    elif age < timedelta(hours=24):
                        buckets["last_24h"] += 1
                    elif age < timedelta(days=7):
                        buckets["last_7d"] += 1
                    else:
                        buckets["older"] += 1
                except Exception:
                    buckets["older"] += 1
    except Exception:
        pass

    return buckets


def find_duplicates() -> List[Dict[str, Any]]:
    """Find duplicate idempotency keys."""
    if not db_exists():
        return []

    duplicates = []
    try:
        with sqlite3.connect(DB_PATH) as con:
            # Find idempotency_keys that appear more than once
            cur = con.execute("""
                SELECT idempotency_key, COUNT(*) as count
                FROM idempotency
                GROUP BY idempotency_key
                HAVING count > 1
            """)
            for row in cur.fetchall():
                duplicates.append({
                    "idempotency_key": row[0],
                    "count": row[1]
                })
    except Exception:
        pass

    return duplicates


def check_specific_id(decision_id: str) -> Dict[str, Any]:
    """Check if a specific decision ID exists."""
    store = IdempotencyStore(DB_PATH)
    exists = store.exists(decision_id)
    key = store.get(decision_id)

    return {
        "decision_id": decision_id,
        "exists": exists,
        "idempotency_key": key
    }


def clear_old_entries(days: int = RETENTION_DAYS, dry_run: bool = True) -> Dict[str, Any]:
    """Clear entries older than specified days."""
    if not db_exists():
        return {
            "success": True,
            "deleted": 0,
            "message": "No database to clear"
        }

    cutoff = datetime.utcnow() - timedelta(days=days)
    cutoff_str = cutoff.isoformat()

    try:
        with sqlite3.connect(DB_PATH) as con:
            # Count entries to delete
            cur = con.execute(
                "SELECT COUNT(*) FROM idempotency WHERE created_at < ?",
                (cutoff_str,)
            )
            count = cur.fetchone()[0]

            if dry_run:
                return {
                    "success": True,
                    "dry_run": True,
                    "would_delete": count,
                    "cutoff_date": cutoff_str,
                    "message": f"Would delete {count} entries older than {days} days"
                }
            else:
                con.execute(
                    "DELETE FROM idempotency WHERE created_at < ?",
                    (cutoff_str,)
                )
                return {
                    "success": True,
                    "dry_run": False,
                    "deleted": count,
                    "cutoff_date": cutoff_str,
                    "message": f"Deleted {count} entries older than {days} days"
                }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_db_stats() -> Dict[str, Any]:
    """Get database statistics."""
    if not db_exists():
        return {
            "exists": False,
            "path": str(DB_PATH)
        }

    try:
        stat = DB_PATH.stat()
        return {
            "exists": True,
            "path": str(DB_PATH),
            "size_bytes": stat.st_size,
            "size_kb": f"{stat.st_size / 1024:.2f} KB",
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    except Exception as e:
        return {
            "exists": True,
            "path": str(DB_PATH),
            "error": str(e)
        }


# -----------------------------------------------------------------------------
# Check Operations
# -----------------------------------------------------------------------------
def list_entries(limit: int = 50) -> Dict[str, Any]:
    """List stored order IDs."""
    entries = get_all_entries()
    total = len(entries)

    return {
        "action": "list",
        "status": "PASS",
        "details": {
            "database": get_db_stats(),
            "total_entries": total,
            "showing": min(limit, total),
            "entries": entries[:limit],
            "age_distribution": get_entries_by_age()
        }
    }


def check_duplicates() -> Dict[str, Any]:
    """Check for duplicate entries."""
    duplicates = find_duplicates()

    return {
        "action": "check_duplicates",
        "status": "FAIL" if duplicates else "PASS",
        "details": {
            "duplicates_found": len(duplicates),
            "duplicates": duplicates,
            "total_entries": get_entry_count()
        }
    }


def check_integrity() -> Dict[str, Any]:
    """Check database integrity."""
    if not db_exists():
        return {
            "action": "check_integrity",
            "status": "PASS",
            "details": {
                "message": "No database (system not yet initialized)",
                "path": str(DB_PATH)
            }
        }

    try:
        with sqlite3.connect(DB_PATH) as con:
            # Run integrity check
            cur = con.execute("PRAGMA integrity_check")
            result = cur.fetchone()[0]

            # Check table exists
            cur = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='idempotency'"
            )
            table_exists = cur.fetchone() is not None

            return {
                "action": "check_integrity",
                "status": "PASS" if result == "ok" and table_exists else "FAIL",
                "details": {
                    "integrity_check": result,
                    "table_exists": table_exists,
                    "entry_count": get_entry_count(),
                    "database": get_db_stats()
                }
            }
    except Exception as e:
        return {
            "action": "check_integrity",
            "status": "FAIL",
            "details": {
                "error": str(e)
            }
        }


def run_all_checks() -> List[Dict[str, Any]]:
    """Run all idempotency checks."""
    return [
        check_integrity(),
        check_duplicates(),
        list_entries(limit=20)
    ]


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def print_results(results: List[Dict[str, Any]], verbose: bool = False) -> int:
    """Print results and return exit code."""
    print("=" * 70)
    print("KOBE IDEMPOTENCY STORE MANAGEMENT")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 70)

    all_pass = True

    for result in results:
        action = result.get("action", "unknown")
        status = result.get("status", "UNKNOWN")

        if status == "FAIL":
            all_pass = False
            status_str = "[FAIL]"
        elif status == "WARN":
            status_str = "[WARN]"
        else:
            status_str = "[PASS]"

        print(f"\n{status_str} {action}")
        print("-" * 50)

        details = result.get("details", {})
        for key, value in details.items():
            if key == "entries" and isinstance(value, list):
                print(f"  {key}: ({len(value)} shown)")
                if verbose or len(value) <= 10:
                    for entry in value[:20]:
                        if isinstance(entry, dict):
                            print(f"    - {entry.get('decision_id', 'N/A')}")
                            print(f"      key: {entry.get('idempotency_key', 'N/A')}")
                            print(f"      created: {entry.get('created_at', 'N/A')}")
                        else:
                            print(f"    - {entry}")
                elif value:
                    print("    (use --verbose to see all entries)")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            elif isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    if isinstance(item, dict):
                        parts = [f"{k}={v}" for k, v in item.items()]
                        print(f"    - {', '.join(parts)}")
                    else:
                        print(f"    - {item}")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    if all_pass:
        print("OVERALL: PASS - Idempotency store healthy")
        return 0
    else:
        print("OVERALL: FAIL - Issues detected in idempotency store")
        return 2


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Kobe Idempotency Store Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python idempotency.py --list              # List stored order IDs
  python idempotency.py --check             # Check for duplicates and integrity
  python idempotency.py --clear             # Clear entries >7 days (dry run)
  python idempotency.py --clear --confirm   # Actually delete old entries
  python idempotency.py --lookup DEC_123    # Check specific decision ID
        """
    )
    parser.add_argument("--dotenv", type=str, default=DEFAULT_DOTENV,
                        help="Path to .env file")
    parser.add_argument("--list", action="store_true",
                        help="List stored order IDs")
    parser.add_argument("--check", action="store_true",
                        help="Check for duplicates and integrity")
    parser.add_argument("--clear", action="store_true",
                        help="Clear old entries (>7 days by default)")
    parser.add_argument("--days", type=int, default=RETENTION_DAYS,
                        help=f"Retention period in days (default: {RETENTION_DAYS})")
    parser.add_argument("--confirm", action="store_true",
                        help="Actually delete entries (without this, --clear is dry run)")
    parser.add_argument("--lookup", type=str, metavar="DECISION_ID",
                        help="Look up specific decision ID")
    parser.add_argument("--limit", type=int, default=50,
                        help="Limit number of entries to show (default: 50)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
        if args.verbose:
            print(f"Loaded {len(loaded)} env vars from {dotenv}")

    # Default to --list if no flags
    if not args.list and not args.check and not args.clear and not args.lookup:
        args.list = True

    try:
        results = []

        if args.lookup:
            result = check_specific_id(args.lookup)
            print(f"\nLookup: {args.lookup}")
            print(f"  Exists: {result['exists']}")
            print(f"  Idempotency Key: {result['idempotency_key']}")
            sys.exit(0 if result['exists'] else 1)

        if args.clear:
            result = clear_old_entries(days=args.days, dry_run=not args.confirm)
            results.append({
                "action": "clear_old_entries",
                "status": "PASS" if result["success"] else "FAIL",
                "details": result
            })

        if args.check:
            results.extend([
                check_integrity(),
                check_duplicates()
            ])

        if args.list:
            results.append(list_entries(limit=args.limit))

        # If only clear was run, still run list to show state
        if args.clear and not args.list and not args.check:
            results.append(list_entries(limit=10))

        exit_code = print_results(results, verbose=args.verbose)
        sys.exit(exit_code)
    except Exception as e:
        print(f"[ERROR] Idempotency store operation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
