#!/usr/bin/env python3
"""
Audit Trail Verification for Kobe Trading System

Verifies hash chain integrity, checks for event log gaps, and validates trade records.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from core.hash_chain import verify_chain, CHAIN_FILE


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_DOTENV = "C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env"
STATE_DIR = ROOT / "state"
LOGS_DIR = ROOT / "logs"
RECONCILE_DIR = STATE_DIR / "reconcile"


# -----------------------------------------------------------------------------
# Hash Chain Verification
# -----------------------------------------------------------------------------
def verify_hash_chain_integrity() -> Dict[str, Any]:
    """Verify the hash chain integrity using core module."""
    if not CHAIN_FILE.exists():
        return {
            "check": "hash_chain_integrity",
            "status": "PASS",
            "details": {
                "chain_file": str(CHAIN_FILE),
                "exists": False,
                "message": "No hash chain file (new system or not yet initialized)"
            }
        }

    try:
        is_valid = verify_chain()
        block_count = count_chain_blocks()
        first_block, last_block = get_chain_endpoints()

        return {
            "check": "hash_chain_integrity",
            "status": "PASS" if is_valid else "FAIL",
            "details": {
                "chain_file": str(CHAIN_FILE),
                "valid": is_valid,
                "block_count": block_count,
                "first_block_hash": first_block[:16] + "..." if first_block else None,
                "last_block_hash": last_block[:16] + "..." if last_block else None
            }
        }
    except Exception as e:
        return {
            "check": "hash_chain_integrity",
            "status": "FAIL",
            "details": {
                "chain_file": str(CHAIN_FILE),
                "error": str(e)
            }
        }


def count_chain_blocks() -> int:
    """Count the number of blocks in the hash chain."""
    if not CHAIN_FILE.exists():
        return 0

    count = 0
    for line in CHAIN_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            count += 1
    return count


def get_chain_endpoints() -> Tuple[Optional[str], Optional[str]]:
    """Get the first and last block hashes from the chain."""
    if not CHAIN_FILE.exists():
        return None, None

    first_hash = None
    last_hash = None

    for line in CHAIN_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                block = json.loads(line)
                if first_hash is None:
                    first_hash = block.get("this_hash")
                last_hash = block.get("this_hash")
            except Exception:
                pass

    return first_hash, last_hash


# -----------------------------------------------------------------------------
# Event Log Analysis
# -----------------------------------------------------------------------------
def check_event_log_gaps() -> Dict[str, Any]:
    """Check for gaps in the event log / hash chain."""
    if not CHAIN_FILE.exists():
        return {
            "check": "event_log_gaps",
            "status": "PASS",
            "details": {
                "message": "No event log to check",
                "gaps_found": 0
            }
        }

    gaps = []
    prev_hash = None
    line_num = 0

    try:
        for line in CHAIN_FILE.read_text(encoding="utf-8").splitlines():
            line_num += 1
            if not line.strip():
                continue

            block = json.loads(line)
            block_prev = block.get("prev_hash")

            if block_prev != prev_hash:
                gaps.append({
                    "line": line_num,
                    "expected_prev": prev_hash[:16] + "..." if prev_hash else None,
                    "actual_prev": block_prev[:16] + "..." if block_prev else None
                })

            prev_hash = block.get("this_hash")

        return {
            "check": "event_log_gaps",
            "status": "FAIL" if gaps else "PASS",
            "details": {
                "total_blocks": line_num,
                "gaps_found": len(gaps),
                "gaps": gaps[:10]  # Limit to first 10
            }
        }
    except Exception as e:
        return {
            "check": "event_log_gaps",
            "status": "FAIL",
            "details": {
                "error": str(e)
            }
        }


def analyze_event_types() -> Dict[str, Any]:
    """Analyze the types of events in the hash chain."""
    if not CHAIN_FILE.exists():
        return {
            "check": "event_type_analysis",
            "status": "PASS",
            "details": {
                "message": "No events to analyze"
            }
        }

    event_counts: Dict[str, int] = {}
    total_events = 0

    try:
        for line in CHAIN_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue

            block = json.loads(line)
            payload = block.get("payload", {})
            event_type = payload.get("event_type", payload.get("type", "unknown"))
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            total_events += 1

        return {
            "check": "event_type_analysis",
            "status": "PASS",
            "details": {
                "total_events": total_events,
                "event_types": event_counts
            }
        }
    except Exception as e:
        return {
            "check": "event_type_analysis",
            "status": "WARN",
            "details": {
                "error": str(e)
            }
        }


# -----------------------------------------------------------------------------
# Trade Record Validation
# -----------------------------------------------------------------------------
def validate_trade_records() -> Dict[str, Any]:
    """Validate local trade records match broker records."""
    # Load local order history from hash chain
    local_orders = extract_orders_from_chain()

    # Load broker orders from reconcile snapshot
    broker_orders = load_broker_orders()

    if not broker_orders:
        return {
            "check": "trade_record_validation",
            "status": "WARN",
            "details": {
                "message": "No broker records available for comparison",
                "suggestion": "Run reconcile_alpaca.py to fetch broker state",
                "local_order_count": len(local_orders)
            }
        }

    # Compare records
    mismatches = []
    matched = 0

    broker_order_ids = {o.get("client_order_id"): o for o in broker_orders if o.get("client_order_id")}

    for local_order in local_orders:
        local_id = local_order.get("idempotency_key") or local_order.get("decision_id")
        if local_id in broker_order_ids:
            matched += 1
            broker_order = broker_order_ids[local_id]

            # Check for discrepancies
            if local_order.get("symbol") != broker_order.get("symbol"):
                mismatches.append({
                    "order_id": local_id,
                    "field": "symbol",
                    "local": local_order.get("symbol"),
                    "broker": broker_order.get("symbol")
                })

    return {
        "check": "trade_record_validation",
        "status": "FAIL" if mismatches else "PASS",
        "details": {
            "local_orders": len(local_orders),
            "broker_orders": len(broker_orders),
            "matched": matched,
            "mismatches": mismatches[:10]
        }
    }


def extract_orders_from_chain() -> List[Dict[str, Any]]:
    """Extract order records from hash chain."""
    orders = []

    if not CHAIN_FILE.exists():
        return orders

    try:
        for line in CHAIN_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue

            block = json.loads(line)
            payload = block.get("payload", {})

            # Look for order-related events
            if "order" in payload or "decision_id" in payload or "symbol" in payload:
                orders.append(payload)
    except Exception:
        pass

    return orders


def load_broker_orders() -> List[Dict[str, Any]]:
    """Load broker orders from reconcile snapshot."""
    orders_file = RECONCILE_DIR / "orders_all.json"

    if not orders_file.exists():
        return []

    try:
        return json.loads(orders_file.read_text(encoding="utf-8"))
    except Exception:
        return []


# -----------------------------------------------------------------------------
# Full Audit Report
# -----------------------------------------------------------------------------
def run_quick_audit() -> List[Dict[str, Any]]:
    """Run quick audit checks."""
    return [
        verify_hash_chain_integrity(),
        check_event_log_gaps()
    ]


def run_full_audit() -> List[Dict[str, Any]]:
    """Run comprehensive audit."""
    return [
        verify_hash_chain_integrity(),
        check_event_log_gaps(),
        analyze_event_types(),
        validate_trade_records(),
        check_state_files()
    ]


def check_state_files() -> Dict[str, Any]:
    """Check for required state files and their health."""
    required_files = [
        ("hash_chain.jsonl", CHAIN_FILE),
        ("idempotency.sqlite", STATE_DIR / "idempotency.sqlite"),
    ]

    optional_files = [
        ("positions.json", RECONCILE_DIR / "positions.json"),
        ("orders_all.json", RECONCILE_DIR / "orders_all.json"),
    ]

    file_status = []
    missing_required = []

    for name, path in required_files:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat() if exists else None

        file_status.append({
            "name": name,
            "exists": exists,
            "size_bytes": size,
            "last_modified": mtime,
            "required": True
        })

        if not exists:
            missing_required.append(name)

    for name, path in optional_files:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat() if exists else None

        file_status.append({
            "name": name,
            "exists": exists,
            "size_bytes": size,
            "last_modified": mtime,
            "required": False
        })

    # Missing required files is not a fail for new systems
    status = "PASS" if not missing_required else "WARN"

    return {
        "check": "state_files",
        "status": status,
        "details": {
            "state_dir": str(STATE_DIR),
            "files": file_status,
            "missing_required": missing_required
        }
    }


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def print_audit_report(results: List[Dict[str, Any]], verbose: bool = False) -> int:
    """Print audit report and return exit code."""
    print("=" * 70)
    print("KOBE AUDIT TRAIL VERIFICATION REPORT")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 70)

    all_pass = True
    has_warn = False

    for result in results:
        check_name = result.get("check", "unknown")
        status = result.get("status", "UNKNOWN")

        if status == "FAIL":
            all_pass = False
            status_str = "[FAIL]"
        elif status == "WARN":
            has_warn = True
            status_str = "[WARN]"
        else:
            status_str = "[PASS]"

        print(f"\n{status_str} {check_name}")
        print("-" * 50)

        details = result.get("details", {})
        for key, value in details.items():
            if isinstance(value, list):
                print(f"  {key}:")
                if value:
                    for item in value[:10]:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                print(f"      {k}: {v}")
                            print()
                        else:
                            print(f"    - {item}")
                else:
                    print("    (none)")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("-" * 70)

    pass_count = sum(1 for r in results if r.get("status") == "PASS")
    warn_count = sum(1 for r in results if r.get("status") == "WARN")
    fail_count = sum(1 for r in results if r.get("status") == "FAIL")

    print(f"  Checks passed:  {pass_count}")
    print(f"  Warnings:       {warn_count}")
    print(f"  Failures:       {fail_count}")
    print(f"  Total checks:   {len(results)}")

    print("\n" + "=" * 70)
    if all_pass and not has_warn:
        print("OVERALL RESULT: PASS - Audit trail verified successfully")
        return 0
    elif all_pass and has_warn:
        print("OVERALL RESULT: WARN - Passed with warnings (review recommended)")
        return 1
    else:
        print("OVERALL RESULT: FAIL - Audit trail integrity compromised")
        return 2


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Kobe Audit Trail Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audit.py --quick             # Quick hash chain verification
  python audit.py --full              # Full comprehensive audit
  python audit.py --full --verbose    # Detailed audit output
        """
    )
    parser.add_argument("--dotenv", type=str, default=DEFAULT_DOTENV,
                        help="Path to .env file")
    parser.add_argument("--quick", action="store_true",
                        help="Quick audit (hash chain only)")
    parser.add_argument("--full", action="store_true",
                        help="Full comprehensive audit")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
        if args.verbose:
            print(f"Loaded {len(loaded)} env vars from {dotenv}")

    # Default to --quick if no flags
    if not args.quick and not args.full:
        args.quick = True

    try:
        if args.full:
            results = run_full_audit()
        else:
            results = run_quick_audit()

        exit_code = print_audit_report(results, verbose=args.verbose)
        sys.exit(exit_code)
    except Exception as e:
        print(f"[ERROR] Audit failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
