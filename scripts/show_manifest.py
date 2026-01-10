#!/usr/bin/env python3
"""
show_manifest.py - Display Core vs Extensions breakdown for Kobe trading robot.

Shows exactly what's CORE (required for trading) vs what's an EXTENSION (optional).

Usage:
    python scripts/show_manifest.py              # Show summary
    python scripts/show_manifest.py --detailed   # Show all files
    python scripts/show_manifest.py --json       # Output as JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_manifest():
    """Load the core manifest."""
    manifest_path = ROOT / "config" / "core_manifest.json"
    with open(manifest_path) as f:
        return json.load(f)


def load_extensions_config():
    """Load extensions enabled config."""
    config_path = ROOT / "config" / "extensions_enabled.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def print_summary(manifest, extensions_config):
    """Print a summary of core vs extensions."""
    print("\n" + "=" * 70)
    print("          KOBE TRADING ROBOT - CORE vs EXTENSIONS")
    print("=" * 70)

    # Count core files
    core = manifest.get("core", {})
    total_core_files = 0
    for section_name, section_data in core.items():
        if isinstance(section_data, dict) and "files" in section_data:
            total_core_files += len(section_data["files"])

    print(f"\nCORE TRADING SYSTEM: {total_core_files} files (MUST WORK)")
    print("-" * 50)

    for section_name, section_data in core.items():
        if isinstance(section_data, dict) and "files" in section_data:
            files = section_data["files"]
            desc = section_data.get("description", "")
            print(f"  [{len(files):2d}] {section_name.upper()}: {desc}")

    # Extensions
    extensions = manifest.get("extensions", {})
    enabled_config = extensions_config.get("extensions", {})

    enabled_count = 0
    disabled_count = 0
    not_wired_count = 0

    for ext_name, ext_data in extensions.items():
        if not isinstance(ext_data, dict):
            continue

        # Check enabled status from config
        config_ext = enabled_config.get(ext_name, {})
        enabled = config_ext.get("enabled", ext_data.get("enabled", False))
        wired = ext_data.get("integration_point") != "NOT WIRED"

        if not wired:
            not_wired_count += 1
        elif enabled:
            enabled_count += 1
        else:
            disabled_count += 1

    print(f"\nEXTENSIONS: {enabled_count} enabled, {disabled_count} disabled, {not_wired_count} not wired")
    print("-" * 50)

    for ext_name, ext_data in extensions.items():
        if not isinstance(ext_data, dict):
            continue

        desc = ext_data.get("description", "")
        files = ext_data.get("files", [])
        integration = ext_data.get("integration_point", "Unknown")

        # Check enabled status
        config_ext = enabled_config.get(ext_name, {})
        enabled = config_ext.get("enabled", ext_data.get("enabled", False))

        if integration == "NOT WIRED":
            status = "[---]"
        elif enabled:
            status = "[ON ]"
        else:
            status = "[OFF]"

        print(f"  {status} {ext_name}: {desc}")
        print(f"         Integration: {integration}")

    # Not wired
    not_wired = manifest.get("not_wired", {})
    not_wired_files = not_wired.get("files", [])

    print(f"\nNOT WIRED (DEAD CODE?): {len(not_wired_files)} files/folders")
    print("-" * 50)
    for f in not_wired_files[:10]:  # Show first 10
        print(f"  [---] {f}")
    if len(not_wired_files) > 10:
        print(f"  ... and {len(not_wired_files) - 10} more")

    # Duplicates
    duplicates = manifest.get("duplicates_to_resolve", {})
    items = duplicates.get("items", [])

    if items:
        print(f"\nDUPLICATES TO RESOLVE: {len(items)} categories")
        print("-" * 50)
        for item in items:
            cat = item.get("category", "Unknown")
            files = item.get("files", [])
            rec = item.get("recommendation", "")
            print(f"  [{len(files)}] {cat}")
            print(f"      Recommendation: {rec}")

    print("\n" + "=" * 70)
    print("COMMANDS:")
    print("  python scripts/verify_core.py        # Verify all core files")
    print("  python scripts/runner.py --core-only # Run with only core")
    print("  python scripts/runner.py --extensions ml_markov,cognitive")
    print("=" * 70 + "\n")


def print_detailed(manifest, extensions_config):
    """Print detailed breakdown with all files."""
    print("\n" + "=" * 70)
    print("          KOBE TRADING ROBOT - DETAILED MANIFEST")
    print("=" * 70)

    core = manifest.get("core", {})

    print("\n>>> CORE FILES (Required for trading)")
    print("-" * 50)

    for section_name, section_data in core.items():
        if isinstance(section_data, dict) and "files" in section_data:
            files = section_data["files"]
            desc = section_data.get("description", "")
            print(f"\n{section_name.upper()}: {desc}")
            for f in files:
                exists = (ROOT / f).exists()
                status = "[OK]" if exists else "[!!]"
                print(f"    {status} {f}")

    extensions = manifest.get("extensions", {})
    enabled_config = extensions_config.get("extensions", {})

    print("\n\n>>> EXTENSIONS (Optional enhancements)")
    print("-" * 50)

    for ext_name, ext_data in extensions.items():
        if not isinstance(ext_data, dict):
            continue

        desc = ext_data.get("description", "")
        files = ext_data.get("files", [])
        integration = ext_data.get("integration_point", "Unknown")
        impact = ext_data.get("impact", "Unknown")

        config_ext = enabled_config.get(ext_name, {})
        enabled = config_ext.get("enabled", ext_data.get("enabled", False))

        status = "ENABLED" if enabled else "DISABLED"
        print(f"\n{ext_name.upper()} [{status}]: {desc}")
        print(f"    Impact: {impact}")
        print(f"    Integration: {integration}")
        print("    Files:")
        for f in files:
            exists = (ROOT / f).exists()
            file_status = "[OK]" if exists else "[!!]"
            print(f"      {file_status} {f}")

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Show Kobe core vs extensions manifest")
    parser.add_argument("--detailed", action="store_true", help="Show all files in each section")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    manifest = load_manifest()
    extensions_config = load_extensions_config()

    if args.json:
        output = {
            "manifest": manifest,
            "extensions_enabled": extensions_config,
        }
        print(json.dumps(output, indent=2))
    elif args.detailed:
        print_detailed(manifest, extensions_config)
    else:
        print_summary(manifest, extensions_config)


if __name__ == "__main__":
    main()
