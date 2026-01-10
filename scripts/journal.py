#!/usr/bin/env python3
"""
Kobe Trading Journal - Track notes, tags, and learnings for trades.

Usage:
    python journal.py --add --trade-id TRD_123 --note "Good entry timing" --tags "good_entry,momentum"
    python journal.py --review
    python journal.py --review --trade-id TRD_123
    python journal.py --export --output journal_export.md
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.env_loader import load_env

STATE_DIR = ROOT / "state"
JOURNAL_FILE = STATE_DIR / "journal.jsonl"

# Predefined tags for quick categorization
VALID_TAGS = [
    "good_entry", "bad_entry", "good_exit", "bad_exit",
    "momentum", "reversal", "gap_play", "breakout",
    "stopped_out", "target_hit", "partial_exit",
    "overtraded", "right_idea", "wrong_timing",
    "news_driven", "technical", "emotional",
]


def ensure_state_dir() -> None:
    """Ensure state directory exists."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def load_journal() -> List[Dict[str, Any]]:
    """Load all journal entries from JSONL file."""
    if not JOURNAL_FILE.exists():
        return []
    entries = []
    for line in JOURNAL_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def save_entry(entry: Dict[str, Any]) -> None:
    """Append a single entry to journal file."""
    ensure_state_dir()
    with JOURNAL_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def add_entry(trade_id: str, note: str, tags: List[str]) -> Dict[str, Any]:
    """Add a new journal entry for a trade."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "trade_id": trade_id,
        "note": note,
        "tags": tags,
    }
    save_entry(entry)
    return entry


def review_entries(trade_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Review journal entries, optionally filtered by trade_id."""
    entries = load_journal()
    if trade_id:
        entries = [e for e in entries if e.get("trade_id") == trade_id]
    # Sort by timestamp descending (most recent first)
    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return entries[:limit]


def export_to_markdown(output_path: Path) -> str:
    """Export entire journal to markdown format."""
    entries = load_journal()
    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    lines = [
        "# Kobe Trading Journal",
        "",
        f"*Exported: {datetime.utcnow().isoformat()}*",
        "",
        f"**Total Entries:** {len(entries)}",
        "",
        "---",
        "",
    ]

    # Group entries by trade_id
    by_trade: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        tid = e.get("trade_id", "unknown")
        if tid not in by_trade:
            by_trade[tid] = []
        by_trade[tid].append(e)

    for trade_id, trade_entries in by_trade.items():
        lines.append(f"## Trade: {trade_id}")
        lines.append("")
        for entry in trade_entries:
            ts = entry.get("timestamp", "N/A")
            note = entry.get("note", "")
            tags = entry.get("tags", [])
            tags_str = ", ".join(f"`{t}`" for t in tags) if tags else "none"
            lines.append(f"### {ts}")
            lines.append("")
            lines.append(f"**Tags:** {tags_str}")
            lines.append("")
            lines.append(note)
            lines.append("")
        lines.append("---")
        lines.append("")

    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    return content


def print_entry(entry: Dict[str, Any]) -> None:
    """Pretty-print a journal entry."""
    ts = entry.get("timestamp", "N/A")
    tid = entry.get("trade_id", "N/A")
    note = entry.get("note", "")
    tags = entry.get("tags", [])
    tags_str = ", ".join(tags) if tags else "none"
    print(f"[{ts}] Trade: {tid}")
    print(f"  Tags: {tags_str}")
    print(f"  Note: {note}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Kobe Trading Journal")
    ap.add_argument("--dotenv", type=str, default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
                    help="Path to .env file")

    # Actions
    ap.add_argument("--add", action="store_true", help="Add a new journal entry")
    ap.add_argument("--review", action="store_true", help="Review past entries")
    ap.add_argument("--export", action="store_true", help="Export journal to markdown")

    # Add options
    ap.add_argument("--trade-id", type=str, help="Trade ID to annotate or filter")
    ap.add_argument("--note", type=str, help="Note text for the entry")
    ap.add_argument("--tags", type=str, help="Comma-separated tags (e.g., 'good_entry,momentum')")

    # Review options
    ap.add_argument("--limit", type=int, default=50, help="Max entries to show in review")

    # Export options
    ap.add_argument("--output", type=str, default="journal_export.md",
                    help="Output file for markdown export")

    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    if args.add:
        if not args.trade_id:
            print("Error: --trade-id is required for --add")
            sys.exit(1)
        if not args.note:
            print("Error: --note is required for --add")
            sys.exit(1)

        tags = []
        if args.tags:
            tags = [t.strip() for t in args.tags.split(",") if t.strip()]
            # Warn about unknown tags
            unknown = [t for t in tags if t not in VALID_TAGS]
            if unknown:
                print(f"Warning: Unknown tags (still added): {unknown}")
                print(f"Valid tags: {VALID_TAGS}")

        entry = add_entry(args.trade_id, args.note, tags)
        print("Journal entry added:")
        print_entry(entry)

    elif args.review:
        entries = review_entries(trade_id=args.trade_id, limit=args.limit)
        if not entries:
            print("No journal entries found.")
            if args.trade_id:
                print(f"  (Filtered by trade_id: {args.trade_id})")
        else:
            print(f"Showing {len(entries)} journal entries:")
            print("=" * 60)
            for entry in entries:
                print_entry(entry)

    elif args.export:
        output_path = Path(args.output)
        export_to_markdown(output_path)
        print(f"Journal exported to: {output_path.absolute()}")

    else:
        print("Specify --add, --review, or --export")
        print("Use --help for usage information")
        print(f"\nValid tags for --tags: {', '.join(VALID_TAGS)}")


if __name__ == "__main__":
    main()
