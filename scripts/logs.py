#!/usr/bin/env python3
"""
logs.py - View recent events from the Kobe trading system logs.

Usage:
    python scripts/logs.py --tail 50
    python scripts/logs.py --level ERROR
    python scripts/logs.py --grep AAPL
    python scripts/logs.py --from 2024-01-01 --to 2024-01-31
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from datetime import datetime, date
from typing import Any, Dict, Generator, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env

LOG_DIR = ROOT / "logs"
EVENTS_FILE = LOG_DIR / "events.jsonl"


def parse_date(date_str: str) -> Optional[date]:
    """Parse date string in YYYY-MM-DD format."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a JSONL log line."""
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        # Return raw line as fallback
        return {"raw": line, "level": "UNKNOWN", "ts": None}


def tail_file(path: Path, n: int) -> List[str]:
    """Read last n lines from a file efficiently."""
    if not path.exists():
        return []

    lines: List[str] = []
    with open(path, "rb") as f:
        # Seek to end
        f.seek(0, 2)
        file_size = f.tell()

        if file_size == 0:
            return []

        # Read in chunks from end
        chunk_size = 8192
        remaining_size = file_size
        lines_found = 0

        while remaining_size > 0 and lines_found < n + 1:
            read_size = min(chunk_size, remaining_size)
            remaining_size -= read_size
            f.seek(remaining_size)
            chunk = f.read(read_size).decode("utf-8", errors="replace")
            chunk_lines = chunk.split("\n")
            lines = chunk_lines + lines
            lines_found = len([l for l in lines if l.strip()])

    # Return last n non-empty lines
    non_empty = [l for l in lines if l.strip()]
    return non_empty[-n:] if len(non_empty) > n else non_empty


def read_all_lines(path: Path) -> Generator[str, None, None]:
    """Read all lines from a file."""
    if not path.exists():
        return

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line


def filter_by_level(record: Dict[str, Any], levels: List[str]) -> bool:
    """Check if record matches any of the specified levels."""
    if not levels:
        return True
    record_level = record.get("level", "INFO").upper()
    return record_level in [l.upper() for l in levels]


def filter_by_date_range(
    record: Dict[str, Any],
    from_date: Optional[date],
    to_date: Optional[date]
) -> bool:
    """Check if record falls within date range."""
    if not from_date and not to_date:
        return True

    ts = record.get("ts")
    if not ts:
        return True  # Include records without timestamp

    try:
        if isinstance(ts, str):
            # Handle ISO format with or without microseconds
            ts_clean = ts.replace("Z", "+00:00")
            record_date = datetime.fromisoformat(ts_clean).date()
        else:
            record_date = ts
    except (ValueError, TypeError):
        return True  # Include records with unparseable timestamp

    if from_date and record_date < from_date:
        return False
    if to_date and record_date > to_date:
        return False

    return True


def filter_by_grep(record: Dict[str, Any], pattern: Optional[str]) -> bool:
    """Check if record matches grep pattern."""
    if not pattern:
        return True

    # Search in all string fields
    record_str = json.dumps(record, default=str)
    try:
        return bool(re.search(pattern, record_str, re.IGNORECASE))
    except re.error:
        # Treat as literal string if not valid regex
        return pattern.lower() in record_str.lower()


def format_record(record: Dict[str, Any], verbose: bool = False) -> str:
    """Format a log record for display."""
    if "raw" in record:
        return record["raw"]

    ts = record.get("ts", "?")
    level = record.get("level", "INFO")
    event = record.get("event", "?")

    # Color codes for levels
    level_colors = {
        "ERROR": "\033[91m",  # Red
        "WARN": "\033[93m",   # Yellow
        "WARNING": "\033[93m",
        "INFO": "\033[92m",   # Green
        "DEBUG": "\033[94m",  # Blue
    }
    reset = "\033[0m"

    # Check if terminal supports colors
    use_color = sys.stdout.isatty()

    if use_color:
        level_color = level_colors.get(level.upper(), "")
        level_str = f"{level_color}{level:7}{reset}"
    else:
        level_str = f"{level:7}"

    # Extract key fields
    extra_fields = []
    for key in ["symbol", "side", "qty", "order_id", "decision_id", "mode", "error", "reason"]:
        if key in record:
            extra_fields.append(f"{key}={record[key]}")

    extra_str = " | " + ", ".join(extra_fields) if extra_fields else ""

    if verbose:
        # Show full JSON
        return f"[{ts}] {level_str} {event}{extra_str}\n  {json.dumps(record, default=str)}"
    else:
        return f"[{ts}] {level_str} {event}{extra_str}"


def show_stats(records: List[Dict[str, Any]]) -> None:
    """Show summary statistics for log records."""
    if not records:
        print("No records to analyze")
        return

    level_counts: Dict[str, int] = {}
    event_counts: Dict[str, int] = {}
    symbols: set = set()

    for r in records:
        level = r.get("level", "UNKNOWN")
        level_counts[level] = level_counts.get(level, 0) + 1

        event = r.get("event", "unknown")
        event_counts[event] = event_counts.get(event, 0) + 1

        if "symbol" in r:
            symbols.add(r["symbol"])

    print("\n--- STATISTICS ---")
    print(f"Total records: {len(records)}")

    print("\nBy level:")
    for level, count in sorted(level_counts.items()):
        pct = count / len(records) * 100
        print(f"  {level:10} {count:6} ({pct:5.1f}%)")

    print(f"\nUnique events: {len(event_counts)}")
    print("Top 10 events:")
    for event, count in sorted(event_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {event:30} {count:6}")

    if symbols:
        print(f"\nUnique symbols: {len(symbols)}")
        if len(symbols) <= 20:
            print(f"  {', '.join(sorted(symbols))}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="View Kobe trading system logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/logs.py --tail 50
    python scripts/logs.py --level ERROR
    python scripts/logs.py --level ERROR --level WARN
    python scripts/logs.py --grep AAPL
    python scripts/logs.py --grep "order.*filled"
    python scripts/logs.py --from 2024-01-01 --to 2024-01-31
    python scripts/logs.py --tail 100 --level ERROR --grep TSLA
    python scripts/logs.py --stats
        """
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file"
    )
    ap.add_argument(
        "--tail",
        "-n",
        type=int,
        default=50,
        help="Number of recent lines to show (default: 50)"
    )
    ap.add_argument(
        "--level",
        "-l",
        action="append",
        dest="levels",
        choices=["ERROR", "WARN", "WARNING", "INFO", "DEBUG"],
        help="Filter by log level (can specify multiple)"
    )
    ap.add_argument(
        "--grep",
        "-g",
        type=str,
        help="Search pattern (regex or literal string)"
    )
    ap.add_argument(
        "--from",
        dest="from_date",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    ap.add_argument(
        "--to",
        dest="to_date",
        type=str,
        help="End date (YYYY-MM-DD)"
    )
    ap.add_argument(
        "--file",
        "-f",
        type=str,
        default=str(EVENTS_FILE),
        help=f"Log file to read (default: {EVENTS_FILE})"
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Read entire file (not just tail)"
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show full JSON for each record"
    )
    ap.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics summary"
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format"
    )
    ap.add_argument(
        "--follow",
        "-F",
        action="store_true",
        help="Follow log file (like tail -f)"
    )

    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    log_file = Path(args.file)
    from_date = parse_date(args.from_date) if args.from_date else None
    to_date = parse_date(args.to_date) if args.to_date else None

    print("=" * 60)
    print("KOBE TRADING SYSTEM - LOG VIEWER")
    print(f"Time: {datetime.utcnow().isoformat()}Z")
    print(f"File: {log_file}")
    print("=" * 60)

    if not log_file.exists():
        print(f"\nLog file does not exist: {log_file}")
        print("No events have been logged yet.")
        sys.exit(0)

    # Follow mode
    if args.follow:
        print("\nFollowing log file (Ctrl+C to stop)...")
        import time

        # First show tail
        lines = tail_file(log_file, args.tail)
        for line in lines:
            record = parse_log_line(line)
            if record and filter_by_level(record, args.levels or []) and \
               filter_by_grep(record, args.grep):
                print(format_record(record, args.verbose))

        # Then follow
        with open(log_file, "r", encoding="utf-8") as f:
            f.seek(0, 2)  # Go to end
            try:
                while True:
                    line = f.readline()
                    if line:
                        record = parse_log_line(line)
                        if record and filter_by_level(record, args.levels or []) and \
                           filter_by_grep(record, args.grep):
                            print(format_record(record, args.verbose))
                    else:
                        time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nStopped following.")
        return

    # Read lines
    if args.all or from_date or to_date:
        # Need to read entire file for date filtering
        lines = list(read_all_lines(log_file))
    else:
        lines = tail_file(log_file, args.tail)

    # Parse and filter
    records: List[Dict[str, Any]] = []
    for line in lines:
        record = parse_log_line(line)
        if record is None:
            continue
        if not filter_by_level(record, args.levels or []):
            continue
        if not filter_by_date_range(record, from_date, to_date):
            continue
        if not filter_by_grep(record, args.grep):
            continue
        records.append(record)

    # Apply tail limit after filtering
    if not args.all and len(records) > args.tail:
        records = records[-args.tail:]

    # Output
    if args.json:
        print(json.dumps(records, indent=2, default=str))
        return

    if args.stats:
        show_stats(records)
        return

    print(f"\nShowing {len(records)} records")
    if args.levels:
        print(f"Filtered by levels: {', '.join(args.levels)}")
    if args.grep:
        print(f"Grep pattern: {args.grep}")
    if from_date or to_date:
        print(f"Date range: {from_date or '*'} to {to_date or '*'}")
    print("-" * 60)

    for record in records:
        print(format_record(record, args.verbose))

    print("-" * 60)
    print(f"Total: {len(records)} records")


if __name__ == "__main__":
    main()
