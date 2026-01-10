#!/usr/bin/env python
"""
Generate Evidence Pack CLI Tool.

Creates comprehensive evidence packs for reproducibility of backtests,
walk-forward tests, and experiments.

Usage:
    # Generate evidence pack for a backtest
    python scripts/generate_evidence.py backtest \
        --universe data/universe/optionable_liquid_800.csv \
        --start 2023-01-01 --end 2024-12-31 \
        --params config/frozen_strategy_params_v2.6.json \
        --trades wf_outputs/trades.csv \
        --metrics wf_outputs/summary.json

    # Generate evidence pack for walk-forward
    python scripts/generate_evidence.py walk_forward \
        --universe data/universe/optionable_liquid_800.csv \
        --start 2015-01-01 --end 2025-12-31 \
        --train-days 252 --test-days 63 \
        --output wf_outputs/

    # Verify an existing evidence pack
    python scripts/generate_evidence.py verify \
        --pack reports/evidence_pack_backtest_20260110_123456_abc12345.json

    # Show environment details (useful for debugging)
    python scripts/generate_evidence.py env-info
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.evidence import (
    EvidencePack,
    EvidencePackBuilder,
    create_backtest_evidence_pack,
    create_walkforward_evidence_pack,
)


def cmd_backtest(args: argparse.Namespace) -> int:
    """Generate evidence pack for a backtest."""
    print(f"Generating backtest evidence pack...")
    print(f"  Universe: {args.universe}")
    print(f"  Date range: {args.start} to {args.end}")

    # Load metrics if provided
    metrics = None
    if args.metrics:
        metrics_path = Path(args.metrics)
        if metrics_path.exists():
            if metrics_path.suffix == ".json":
                metrics = json.loads(metrics_path.read_text())
            else:
                print(f"  Metrics file must be JSON: {args.metrics}")

    # Create evidence pack
    pack = create_backtest_evidence_pack(
        universe_path=Path(args.universe),
        start_date=args.start,
        end_date=args.end,
        frozen_params_path=Path(args.params) if args.params else None,
        metrics=metrics,
        trade_list_path=Path(args.trades) if args.trades else None,
        equity_curve_path=Path(args.equity) if args.equity else None,
    )

    # Save the pack
    output_dir = Path(args.output) if args.output else Path("reports")
    filepath = pack.save(output_dir)

    print(f"\n{'='*60}")
    print(f"EVIDENCE PACK GENERATED")
    print(f"{'='*60}")
    print(f"  Pack ID:    {pack.pack_id}")
    print(f"  Pack Hash:  {pack.pack_hash}")
    print(f"  Git Commit: {pack.git_commit[:8]}...")
    print(f"  Git Dirty:  {pack.git_dirty}")
    print(f"  Saved to:   {filepath}")
    print(f"{'='*60}")

    if metrics:
        print(f"\nMetrics captured:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    return 0


def cmd_walk_forward(args: argparse.Namespace) -> int:
    """Generate evidence pack for walk-forward test."""
    print(f"Generating walk-forward evidence pack...")
    print(f"  Universe: {args.universe}")
    print(f"  Date range: {args.start} to {args.end}")
    print(f"  Train/Test: {args.train_days}/{args.test_days} days")

    # Load metrics if summary exists
    metrics = None
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        summary_path = output_dir / "wf_summary.json"
        if summary_path.exists():
            metrics = json.loads(summary_path.read_text())

    # Create evidence pack
    pack = create_walkforward_evidence_pack(
        universe_path=Path(args.universe),
        start_date=args.start,
        end_date=args.end,
        train_days=args.train_days,
        test_days=args.test_days,
        metrics=metrics,
        output_dir=output_dir,
    )

    # Save the pack
    save_dir = output_dir if output_dir else Path("reports")
    filepath = pack.save(save_dir)

    print(f"\n{'='*60}")
    print(f"EVIDENCE PACK GENERATED")
    print(f"{'='*60}")
    print(f"  Pack ID:    {pack.pack_id}")
    print(f"  Pack Hash:  {pack.pack_hash}")
    print(f"  Git Commit: {pack.git_commit[:8]}...")
    print(f"  Git Dirty:  {pack.git_dirty}")
    print(f"  Saved to:   {filepath}")
    print(f"{'='*60}")

    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify an existing evidence pack."""
    pack_path = Path(args.pack)

    if not pack_path.exists():
        print(f"ERROR: Pack file not found: {pack_path}")
        return 1

    print(f"Loading evidence pack: {pack_path}")
    pack = EvidencePack.load(pack_path)

    print(f"\n{'='*60}")
    print(f"EVIDENCE PACK VERIFICATION")
    print(f"{'='*60}")
    print(f"  Pack ID:    {pack.pack_id}")
    print(f"  Pack Type:  {pack.pack_type}")
    print(f"  Created:    {pack.created_at.isoformat()}")
    print(f"  Git Commit: {pack.git_commit}")
    print(f"  Git Branch: {pack.git_branch}")
    print(f"  Git Dirty:  {pack.git_dirty}")

    # Verify hash
    hash_valid = pack.verify_hash()
    status = "VALID" if hash_valid else "INVALID - TAMPERED!"

    print(f"\n  Pack Hash:  {pack.pack_hash}")
    print(f"  Hash Status: {status}")

    if not hash_valid:
        print(f"\n  WARNING: Pack has been modified since creation!")
        return 1

    print(f"\n  Date Range: {pack.date_range[0]} to {pack.date_range[1]}")
    print(f"  Symbols:    {pack.symbol_count}")

    if pack.metrics:
        print(f"\nMetrics:")
        for key, value in pack.metrics.items():
            print(f"  {key}: {value}")

    if pack.artifacts:
        print(f"\nArtifacts:")
        for name, path_hash in pack.artifacts.items():
            print(f"  {name}: {path_hash}")

    print(f"{'='*60}")
    print(f"VERIFICATION: {'PASSED' if hash_valid else 'FAILED'}")
    print(f"{'='*60}")

    return 0 if hash_valid else 1


def cmd_env_info(args: argparse.Namespace) -> int:
    """Show current environment details."""
    builder = EvidencePackBuilder("env_info")
    builder.capture_git_state()
    builder.capture_environment()

    print(f"\n{'='*60}")
    print(f"ENVIRONMENT INFORMATION")
    print(f"{'='*60}")
    print(f"\nGit State:")
    print(f"  Commit: {builder.git_commit}")
    print(f"  Branch: {builder.git_branch}")
    print(f"  Dirty:  {builder.git_dirty}")

    print(f"\nPython: {builder.python_version.split()[0]}")

    print(f"\nPackage Versions:")
    for pkg, version in sorted(builder.package_versions.items()):
        print(f"  {pkg}: {version}")

    print(f"{'='*60}")

    return 0


def cmd_reproduce_script(args: argparse.Namespace) -> int:
    """Generate reproduction script from evidence pack."""
    pack_path = Path(args.pack)

    if not pack_path.exists():
        print(f"ERROR: Pack file not found: {pack_path}")
        return 1

    pack = EvidencePack.load(pack_path)
    script = pack.generate_reproduce_script()

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(script)
        print(f"Reproduction script saved to: {output_path}")
    else:
        print(script)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate and verify evidence packs for reproducibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate backtest evidence pack
  python scripts/generate_evidence.py backtest \\
      --universe data/universe/optionable_liquid_800.csv \\
      --start 2023-01-01 --end 2024-12-31

  # Verify an evidence pack
  python scripts/generate_evidence.py verify --pack reports/evidence_pack_*.json

  # Show environment info
  python scripts/generate_evidence.py env-info
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Generate backtest evidence pack")
    bt_parser.add_argument("--universe", required=True, help="Path to universe CSV")
    bt_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    bt_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    bt_parser.add_argument("--params", help="Path to frozen params JSON")
    bt_parser.add_argument("--trades", help="Path to trade list CSV")
    bt_parser.add_argument("--equity", help="Path to equity curve CSV")
    bt_parser.add_argument("--metrics", help="Path to metrics JSON")
    bt_parser.add_argument("--output", "-o", help="Output directory (default: reports/)")

    # Walk-forward command
    wf_parser = subparsers.add_parser("walk_forward", help="Generate walk-forward evidence pack")
    wf_parser.add_argument("--universe", required=True, help="Path to universe CSV")
    wf_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    wf_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    wf_parser.add_argument("--train-days", type=int, default=252, help="Training window (default: 252)")
    wf_parser.add_argument("--test-days", type=int, default=63, help="Test window (default: 63)")
    wf_parser.add_argument("--output", "-o", help="Output directory")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify an evidence pack")
    verify_parser.add_argument("--pack", required=True, help="Path to evidence pack JSON")

    # Env-info command
    subparsers.add_parser("env-info", help="Show current environment details")

    # Reproduce script command
    repr_parser = subparsers.add_parser("reproduce-script", help="Generate reproduction script")
    repr_parser.add_argument("--pack", required=True, help="Path to evidence pack JSON")
    repr_parser.add_argument("--output", "-o", help="Output script path (default: stdout)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handler
    handlers = {
        "backtest": cmd_backtest,
        "walk_forward": cmd_walk_forward,
        "verify": cmd_verify,
        "env-info": cmd_env_info,
        "reproduce-script": cmd_reproduce_script,
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
