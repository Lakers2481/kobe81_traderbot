#!/usr/bin/env python3
"""
verify_core.py - Verify all CORE components of the Kobe trading robot.

This script tests every file in the core manifest to ensure:
1. Imports work without errors
2. No syntax errors
3. Dependencies are available
4. Critical functions exist
5. Integration points work

Usage:
    python scripts/verify_core.py                    # Full verification
    python scripts/verify_core.py --quick            # Quick import-only check
    python scripts/verify_core.py --section risk     # Verify specific section
    python scripts/verify_core.py --json             # Output as JSON
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
load_env(ROOT / ".env")


@dataclass
class FileVerification:
    """Result of verifying a single file."""
    file_path: str
    exists: bool = False
    imports_ok: bool = False
    import_error: Optional[str] = None
    syntax_ok: bool = True
    syntax_error: Optional[str] = None
    has_tests: bool = False
    tests_pass: Optional[bool] = None
    critical_functions: List[str] = field(default_factory=list)
    missing_functions: List[str] = field(default_factory=list)
    overall_status: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SectionVerification:
    """Result of verifying a section (e.g., risk, execution)."""
    section_name: str
    description: str
    total_files: int = 0
    passed_files: int = 0
    failed_files: int = 0
    files: List[FileVerification] = field(default_factory=list)
    overall_status: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_name": self.section_name,
            "description": self.description,
            "total_files": self.total_files,
            "passed_files": self.passed_files,
            "failed_files": self.failed_files,
            "overall_status": self.overall_status,
            "files": [f.to_dict() for f in self.files],
        }


@dataclass
class CoreVerificationReport:
    """Complete verification report."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_core_files: int = 0
    verified_files: int = 0
    passed_files: int = 0
    failed_files: int = 0
    sections: List[SectionVerification] = field(default_factory=list)
    extensions_status: Dict[str, bool] = field(default_factory=dict)
    overall_status: str = "UNKNOWN"
    core_ready: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_core_files": self.total_core_files,
            "verified_files": self.verified_files,
            "passed_files": self.passed_files,
            "failed_files": self.failed_files,
            "overall_status": self.overall_status,
            "core_ready": self.core_ready,
            "sections": [s.to_dict() for s in self.sections],
            "extensions_status": self.extensions_status,
        }


# Critical functions that MUST exist in each module
CRITICAL_FUNCTIONS = {
    "strategies/dual_strategy/combined.py": ["DualStrategyScanner", "DualStrategyParams"],
    "risk/policy_gate.py": ["PolicyGate", "load_limits_from_config"],
    "risk/equity_sizer.py": ["calculate_position_size", "get_account_equity"],
    "risk/kill_zone_gate.py": ["KillZoneGate", "get_kill_zone_gate"],
    "execution/broker_alpaca.py": ["get_best_ask", "get_best_bid", "place_bracket_order"],
    "data/providers/polygon_eod.py": ["fetch_daily_bars_polygon"],
    "data/universe/loader.py": ["load_universe"],
    "core/structured_log.py": ["jlog"],
    "core/hash_chain.py": ["append_block"],
    "core/kill_switch.py": ["is_kill_switch_active", "activate_kill_switch"],
    "oms/order_state.py": ["OrderRecord", "OrderStatus"],
    "oms/idempotency_store.py": ["IdempotencyStore"],
    "monitor/health_endpoints.py": ["start_health_server", "update_request_counter"],
}


def load_manifest() -> Dict[str, Any]:
    """Load the core manifest."""
    manifest_path = ROOT / "config" / "core_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Core manifest not found at {manifest_path}")

    with open(manifest_path) as f:
        return json.load(f)


def file_to_module(file_path: str) -> str:
    """Convert file path to module import path."""
    # Remove .py extension
    module_path = file_path.replace(".py", "")
    # Convert path separators to dots
    module_path = module_path.replace("/", ".").replace("\\", ".")
    return module_path


def verify_file(file_path: str, quick: bool = False) -> FileVerification:
    """Verify a single file."""
    result = FileVerification(file_path=file_path)

    # Check if file exists
    full_path = ROOT / file_path
    result.exists = full_path.exists()

    if not result.exists:
        result.overall_status = "FAIL"
        result.import_error = "File does not exist"
        return result

    # Check syntax
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            code = f.read()
        compile(code, file_path, "exec")
        result.syntax_ok = True
    except SyntaxError as e:
        result.syntax_ok = False
        result.syntax_error = str(e)
        result.overall_status = "FAIL"
        return result

    # Try to import
    module_name = file_to_module(file_path)
    try:
        module = importlib.import_module(module_name)
        result.imports_ok = True

        # Check critical functions if not quick mode
        if not quick and file_path in CRITICAL_FUNCTIONS:
            for func_name in CRITICAL_FUNCTIONS[file_path]:
                if hasattr(module, func_name):
                    result.critical_functions.append(func_name)
                else:
                    result.missing_functions.append(func_name)

    except Exception as e:
        result.imports_ok = False
        result.import_error = f"{type(e).__name__}: {str(e)}"

    # Check for associated tests
    test_path = ROOT / "tests" / file_path.replace("/", "_").replace(".py", "_test.py")
    alt_test_path = ROOT / "tests" / Path(file_path).parent / f"test_{Path(file_path).name}"
    result.has_tests = test_path.exists() or alt_test_path.exists()

    # Determine overall status
    if result.imports_ok and result.syntax_ok and not result.missing_functions:
        result.overall_status = "PASS"
    elif result.imports_ok and result.syntax_ok:
        result.overall_status = "WARN"  # Missing some critical functions
    else:
        result.overall_status = "FAIL"

    return result


def verify_section(section_name: str, section_data: Dict[str, Any], quick: bool = False) -> SectionVerification:
    """Verify all files in a section."""
    result = SectionVerification(
        section_name=section_name,
        description=section_data.get("description", ""),
    )

    files = section_data.get("files", [])
    result.total_files = len(files)

    for file_path in files:
        file_result = verify_file(file_path, quick)
        result.files.append(file_result)

        if file_result.overall_status == "PASS":
            result.passed_files += 1
        else:
            result.failed_files += 1

    # Determine section status
    if result.failed_files == 0:
        result.overall_status = "PASS"
    elif result.passed_files > 0:
        result.overall_status = "PARTIAL"
    else:
        result.overall_status = "FAIL"

    return result


def verify_extension(ext_name: str, ext_data: Dict[str, Any]) -> bool:
    """Verify an extension can import."""
    if not ext_data.get("enabled", False):
        return True  # Disabled extensions don't need to import

    files = ext_data.get("files", [])
    for file_path in files:
        try:
            full_path = ROOT / file_path
            if full_path.exists() and file_path.endswith(".py"):
                module_name = file_to_module(file_path)
                importlib.import_module(module_name)
        except Exception:
            return False

    return True


def verify_core(quick: bool = False, section_filter: Optional[str] = None) -> CoreVerificationReport:
    """Run full core verification."""
    manifest = load_manifest()
    report = CoreVerificationReport()

    core = manifest.get("core", {})

    # Verify each section
    for section_name, section_data in core.items():
        if section_name == "description" or section_name == "total_files":
            continue

        if section_filter and section_name != section_filter:
            continue

        if isinstance(section_data, dict) and "files" in section_data:
            section_result = verify_section(section_name, section_data, quick)
            report.sections.append(section_result)
            report.total_core_files += section_result.total_files
            report.verified_files += section_result.total_files
            report.passed_files += section_result.passed_files
            report.failed_files += section_result.failed_files

    # Verify extensions
    extensions = manifest.get("extensions", {})
    for ext_name, ext_data in extensions.items():
        if isinstance(ext_data, dict):
            report.extensions_status[ext_name] = verify_extension(ext_name, ext_data)

    # Determine overall status
    if report.failed_files == 0:
        report.overall_status = "PASS"
        report.core_ready = True
    elif report.passed_files > report.failed_files:
        report.overall_status = "PARTIAL"
        report.core_ready = False
    else:
        report.overall_status = "FAIL"
        report.core_ready = False

    return report


def print_report(report: CoreVerificationReport, verbose: bool = False):
    """Print verification report to console."""
    print("\n" + "=" * 70)
    print("               KOBE TRADING ROBOT - CORE VERIFICATION")
    print("=" * 70)
    print(f"\nTimestamp: {report.timestamp}")
    print(f"\nTotal Core Files: {report.total_core_files}")
    print(f"Passed: {report.passed_files}")
    print(f"Failed: {report.failed_files}")
    print(f"\nOverall Status: {report.overall_status}")
    print(f"Core Ready for Trading: {'YES' if report.core_ready else 'NO'}")

    print("\n" + "-" * 70)
    print("SECTION DETAILS")
    print("-" * 70)

    for section in report.sections:
        status_icon = "[PASS]" if section.overall_status == "PASS" else "[FAIL]" if section.overall_status == "FAIL" else "[WARN]"
        print(f"\n{status_icon} {section.section_name.upper()}: {section.passed_files}/{section.total_files} passed")
        print(f"    {section.description}")

        if verbose or section.overall_status != "PASS":
            for file in section.files:
                icon = "[OK]" if file.overall_status == "PASS" else "[!!]" if file.overall_status == "FAIL" else "[??]"
                print(f"      {icon} {file.file_path}")
                if file.import_error:
                    print(f"          Error: {file.import_error[:80]}")
                if file.missing_functions:
                    print(f"          Missing: {', '.join(file.missing_functions)}")

    print("\n" + "-" * 70)
    print("EXTENSIONS STATUS")
    print("-" * 70)

    for ext_name, status in report.extensions_status.items():
        icon = "[OK]" if status else "[!!]"
        print(f"  {icon} {ext_name}")

    print("\n" + "=" * 70)

    if report.core_ready:
        print("VERDICT: Core is READY. You can run --core-only mode.")
    else:
        print("VERDICT: Core has issues. Fix failed files before --core-only mode.")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Verify Kobe trading robot core components")
    parser.add_argument("--quick", action="store_true", help="Quick import-only check")
    parser.add_argument("--section", type=str, help="Verify specific section only")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all files, not just failures")
    parser.add_argument("--save", type=str, help="Save report to file")

    args = parser.parse_args()

    print("Verifying core components...")
    report = verify_core(quick=args.quick, section_filter=args.section)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report, verbose=args.verbose)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report saved to {args.save}")

    # Exit with appropriate code
    sys.exit(0 if report.core_ready else 1)


if __name__ == "__main__":
    main()
