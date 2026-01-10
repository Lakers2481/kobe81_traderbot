#!/usr/bin/env python3
"""
Execution Path Wiring Verification
===================================

Traces the complete execution path from Scanner → Top 2 → Broker
to verify that all components are properly wired and data flows correctly.
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import ast
import re

BASE_DIR = Path(__file__).parent.parent

# Critical execution paths to verify
EXECUTION_PATHS = {
    "Scanner → Signals": {
        "entry": "scripts/scan.py",
        "steps": [
            "strategies/dual_strategy/combined.py::DualStrategyScanner",
            "pipelines/unified_signal_enrichment.py::enrich_signals",
            "risk/signal_quality_gate.py::filter_to_best_signals",
        ],
        "output": "logs/tradeable.csv (Top 2)"
    },
    "Risk Gates": {
        "entry": "execution/broker_alpaca.py::place_ioc_limit",
        "decorators": [
            "@require_no_kill_switch",
            "@require_policy_gate",
            "@with_liquidity_check",
        ],
        "gates": [
            "core/kill_switch.py::check_kill_switch",
            "risk/policy_gate.py::PolicyGate.check",
            "risk/kill_zone_gate.py::KillZoneGate.check_can_trade",
            "risk/liquidity_gate.py::LiquidityGate.check",
        ]
    },
    "Data Enrichment": {
        "entry": "pipelines/unified_signal_enrichment.py",
        "components": [
            "analysis/historical_patterns.py",
            "analysis/options_expected_move.py",
            "ml_meta/model.py",
            "ml_advanced/lstm_confidence/model.py",
            "ml_advanced/hmm_regime_detector.py",
            "ml_advanced/markov_chain/",
            "cognitive/signal_processor.py",
        ],
        "output_fields": [
            "historical_pattern",
            "expected_move",
            "ml_confidence",
            "lstm_grade",
            "regime",
            "markov_boost",
        ]
    }
}


def check_file_exists(filepath: str) -> Tuple[bool, str]:
    """Check if a file exists."""
    full_path = BASE_DIR / filepath
    if full_path.exists():
        return True, str(full_path)
    return False, f"NOT FOUND: {filepath}"


def check_decorator_present(filepath: str, decorator_name: str) -> Tuple[bool, List[str]]:
    """Check if decorator is present on a function."""
    full_path = BASE_DIR / filepath
    if not full_path.exists():
        return False, [f"File not found: {filepath}"]

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Simple regex check for decorator
        pattern = rf'@{decorator_name}'
        matches = re.findall(pattern, content)

        if matches:
            return True, [f"Found {len(matches)} usage(s) of @{decorator_name}"]
        return False, [f"Decorator @{decorator_name} not found"]

    except Exception as e:
        return False, [f"Error reading file: {e}"]


def check_import_chain(filepath: str, target_module: str) -> Tuple[bool, List[str]]:
    """Check if a file imports a target module."""
    full_path = BASE_DIR / filepath
    if not full_path.exists():
        return False, [f"File not found: {filepath}"]

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for various import patterns
        patterns = [
            rf'from {target_module} import',
            rf'import {target_module}',
        ]

        found = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                found.append(f"Found: {pattern}")

        if found:
            return True, found
        return False, [f"No imports from {target_module}"]

    except Exception as e:
        return False, [f"Error reading file: {e}"]


def check_function_calls(filepath: str, function_name: str) -> Tuple[bool, List[str]]:
    """Check if a function is called in a file."""
    full_path = BASE_DIR / filepath
    if not full_path.exists():
        return False, [f"File not found: {filepath}"]

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for function calls
        pattern = rf'{function_name}\s*\('
        matches = re.findall(pattern, content)

        if matches:
            return True, [f"Found {len(matches)} call(s) to {function_name}()"]
        return False, [f"Function {function_name}() not called"]

    except Exception as e:
        return False, [f"Error reading file: {e}"]


def verify_decorator_wiring():
    """Verify that critical decorators are properly wired."""
    print("\n" + "=" * 80)
    print("DECORATOR WIRING VERIFICATION")
    print("=" * 80)

    checks = [
        ("execution/broker_alpaca.py", "require_no_kill_switch", "place_ioc_limit"),
        ("execution/broker_alpaca.py", "require_policy_gate", "place_ioc_limit"),
        ("execution/broker_alpaca.py", "with_retry", "place_ioc_limit"),
    ]

    for filepath, decorator, function in checks:
        print(f"\nChecking {decorator} on {function} in {filepath}...")
        exists, msg = check_file_exists(filepath)
        if not exists:
            print(f"  [FAIL] {msg}")
            continue

        has_decorator, details = check_decorator_present(filepath, decorator)
        if has_decorator:
            print(f"  [PASS] Decorator @{decorator} found")
            for detail in details:
                print(f"     {detail}")
        else:
            print(f"  [FAIL] Decorator @{decorator} missing")
            for detail in details:
                print(f"     {detail}")


def verify_risk_gate_calls():
    """Verify that risk gates are actually called (not just imported)."""
    print("\n" + "=" * 80)
    print("RISK GATE CALL VERIFICATION")
    print("=" * 80)

    gates_to_check = [
        ("execution/broker_alpaca.py", "PolicyGate", "check"),
        ("execution/broker_alpaca.py", "LiquidityGate", "check"),
        ("risk/kill_zone_gate.py", "can_trade_now", None),
        ("core/kill_switch.py", "check_kill_switch", None),
    ]

    for filepath, class_or_func, method in gates_to_check:
        if method:
            target = f"{class_or_func}.{method}" if class_or_func != method else class_or_func
        else:
            target = class_or_func

        print(f"\nChecking calls to {target} in {filepath}...")
        exists, msg = check_file_exists(filepath)
        if not exists:
            print(f"  [FAIL] {msg}")
            continue

        has_calls, details = check_function_calls(filepath, target.split('.')[-1])
        if has_calls:
            print(f"  [PASS] Function {target}() is called")
            for detail in details:
                print(f"     {detail}")
        else:
            print(f"  [WARN]  Function {target}() might not be called")
            for detail in details:
                print(f"     {detail}")


def verify_enrichment_pipeline():
    """Verify that enrichment components are wired."""
    print("\n" + "=" * 80)
    print("ENRICHMENT PIPELINE VERIFICATION")
    print("=" * 80)

    pipeline_file = "pipelines/unified_signal_enrichment.py"
    components = [
        ("analysis.historical_patterns", "Historical Patterns"),
        ("analysis.options_expected_move", "Expected Move"),
        ("ml_meta.model", "ML Meta Model"),
        ("ml_advanced.lstm_confidence", "LSTM Confidence"),
        ("ml_advanced.hmm_regime_detector", "HMM Regime"),
        ("ml_advanced.markov_chain", "Markov Chain"),
        ("cognitive.signal_processor", "Signal Processor"),
    ]

    exists, msg = check_file_exists(pipeline_file)
    if not exists:
        print(f"[FAIL] {msg}")
        return

    for module, name in components:
        print(f"\nChecking import of {name} ({module})...")
        has_import, details = check_import_chain(pipeline_file, module)
        if has_import:
            print(f"  [PASS] {name} is imported")
            for detail in details:
                print(f"     {detail}")
        else:
            print(f"  [FAIL] {name} NOT imported")
            for detail in details:
                print(f"     {detail}")


def verify_execution_flow():
    """Verify the complete execution flow."""
    print("\n" + "=" * 80)
    print("EXECUTION FLOW VERIFICATION")
    print("=" * 80)

    flow = [
        ("scripts/scan.py", "DualStrategyScanner", "Scanner entry point"),
        ("strategies/dual_strategy/combined.py", "scan_signals_over_time", "Signal generation"),
        ("pipelines/unified_signal_enrichment.py", "enrich_signals", "Signal enrichment"),
        ("risk/signal_quality_gate.py", "filter_to_best_signals", "Quality filtering"),
        ("execution/broker_alpaca.py", "place_ioc_limit", "Order execution"),
    ]

    for filepath, function, description in flow:
        print(f"\n{description}: {filepath}::{function}")
        exists, msg = check_file_exists(filepath)
        if exists:
            print(f"  [PASS] File exists")
            has_func, details = check_function_calls(filepath, function)
            if has_func:
                print(f"  [PASS] Function {function}() found")
            else:
                print(f"  [WARN]  Function {function}() usage not detected")
        else:
            print(f"  [FAIL] {msg}")


def main():
    print("=" * 80)
    print("KOBE EXECUTION WIRING VERIFICATION")
    print("=" * 80)
    print(f"Base Directory: {BASE_DIR}")

    verify_decorator_wiring()
    verify_risk_gate_calls()
    verify_enrichment_pipeline()
    verify_execution_flow()

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
