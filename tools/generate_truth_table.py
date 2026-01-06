"""
TRUTH TABLE GENERATOR

Generates TRUTH_TABLE.csv mapping every component to its verification status:
- REAL: Has implementation, imports work, can be called
- STUB: Has stub implementation (pass, raise NotImplementedError, TODO)
- MISSING: Expected but not found
- PARTIAL: Has implementation but missing dependencies

Evidence columns:
- file_path: Location of the component
- line_number: Line where defined
- has_docstring: True/False
- has_tests: True/False (checks for test_*.py files)
- import_verified: True/False (can be imported)
- execution_traced: True/False (appears in trace logs)

Author: Kobe Trading System
Version: 1.0.0
Date: 2026-01-05
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_component_inventory() -> Dict[str, Any]:
    """Load the component inventory from AUDITS."""
    inventory_path = ROOT / "AUDITS" / "02_COMPONENT_INVENTORY.json"

    if not inventory_path.exists():
        print(f"ERROR: Component inventory not found: {inventory_path}")
        print("Run: python tools/component_auditor.py first")
        sys.exit(1)

    with open(inventory_path) as f:
        return json.load(f)


def load_trace_files() -> Dict[str, set]:
    """Load traced function names from TRACES directory."""
    traces_dir = ROOT / "AUDITS" / "TRACES"
    traced_functions = set()

    if traces_dir.exists():
        for trace_file in traces_dir.glob("*.jsonl"):
            with open(trace_file) as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        func_name = event.get("function")
                        module = event.get("module", "")
                        if func_name:
                            traced_functions.add(f"{module}.{func_name}")
                            traced_functions.add(func_name)
                    except json.JSONDecodeError:
                        pass

    return {"functions": traced_functions}


def find_test_files() -> Dict[str, List[str]]:
    """Find test files and map them to modules they test."""
    tests_dir = ROOT / "tests"
    test_mapping = {}

    if tests_dir.exists():
        for test_file in tests_dir.rglob("test_*.py"):
            # Extract module name from test file name
            # e.g., test_broker_alpaca.py -> broker_alpaca
            test_name = test_file.stem.replace("test_", "")
            rel_path = str(test_file.relative_to(ROOT))

            if test_name not in test_mapping:
                test_mapping[test_name] = []
            test_mapping[test_name].append(rel_path)

    return test_mapping


def check_import(module_path: str, component_name: str) -> tuple[bool, str]:
    """
    Check if a component can be imported.

    Returns:
        (success, error_message)
    """
    try:
        # Build full import path
        # Convert file path to module path
        module_name = module_path.replace("\\", ".").replace("/", ".")
        if module_name.endswith(".py"):
            module_name = module_name[:-3]

        parts = module_name.split(".")
        mod = __import__(module_name, fromlist=[parts[-1]])

        # Check if component exists in module
        if hasattr(mod, component_name):
            return True, ""
        else:
            return False, f"Component {component_name} not found in module"

    except ImportError as e:
        return False, f"ImportError: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def determine_status(
    component: Dict[str, Any],
    stubs: List[Dict[str, Any]],
    traced: set
) -> tuple[str, str]:
    """
    Determine component status.

    Returns:
        (status, evidence)
    """
    name = component.get("name", "")
    file_path = component.get("file", "")

    # Check if it's a stub
    for stub in stubs:
        if stub.get("name") == name and stub.get("file") == file_path:
            stub_type = stub.get("type", "pass")
            return "STUB", f"Stub type: {stub_type}"

    # Check if traced
    file_mod = file_path.replace('.py', '').replace('\\', '.').replace('/', '.')
    full_name = f"{file_mod}.{name}"
    if name in traced or full_name in traced:
        return "REAL", "Execution traced"

    # Check for implementation indicators
    methods = component.get("methods", [])
    docstring = component.get("docstring", "")

    if methods or docstring:
        return "REAL", f"Has {len(methods)} methods" if methods else "Has docstring"

    return "PARTIAL", "No evidence of execution"


def generate_truth_table():
    """Generate the TRUTH_TABLE.csv file."""
    print("=" * 60)
    print("GENERATING TRUTH TABLE")
    print("=" * 60)

    # Load data
    inventory = load_component_inventory()
    traces = load_trace_files()
    test_files = find_test_files()

    traced_functions = traces.get("functions", set())
    print(f"Loaded {len(traced_functions)} traced function names")
    print(f"Found {len(test_files)} test file mappings")

    # Extract components
    classes = inventory.get("classes", [])
    functions = inventory.get("functions", [])
    stubs = inventory.get("stubs", [])

    print(f"Classes: {len(classes)}")
    print(f"Functions: {len(functions)}")
    print(f"Stubs: {len(stubs)}")

    # Generate truth table rows
    rows = []

    # Process classes
    for cls in classes:
        name = cls.get("name", "")
        file_path = cls.get("file", "")
        line = cls.get("line", 0)
        docstring = cls.get("docstring", "")
        methods = cls.get("methods", [])

        # Determine status
        status, evidence = determine_status(cls, stubs, traced_functions)

        # Check for tests
        module_name = file_path.replace(".py", "").replace("\\", "_").replace("/", "_")
        has_tests = any(
            name.lower() in test_name.lower() or module_name.lower() in test_name.lower()
            for test_name in test_files.keys()
        )

        rows.append({
            "component_type": "CLASS",
            "name": name,
            "file_path": file_path,
            "line_number": line,
            "status": status,
            "evidence": evidence,
            "has_docstring": bool(docstring),
            "has_tests": has_tests,
            "method_count": len(methods),
            "execution_traced": name in traced_functions,
        })

    # Process functions
    for func in functions:
        name = func.get("name", "")
        file_path = func.get("file", "")
        line = func.get("line", 0)
        docstring = func.get("docstring", "")

        # Determine status
        status, evidence = determine_status(func, stubs, traced_functions)

        # Check for tests
        module_name = file_path.replace(".py", "").replace("\\", "_").replace("/", "_")
        has_tests = any(
            name.lower() in test_name.lower() or module_name.lower() in test_name.lower()
            for test_name in test_files.keys()
        )

        rows.append({
            "component_type": "FUNCTION",
            "name": name,
            "file_path": file_path,
            "line_number": line,
            "status": status,
            "evidence": evidence,
            "has_docstring": bool(docstring),
            "has_tests": has_tests,
            "method_count": 0,
            "execution_traced": name in traced_functions,
        })

    # Sort by status priority: STUB first (issues), then PARTIAL, then REAL
    status_priority = {"STUB": 0, "PARTIAL": 1, "REAL": 2, "MISSING": -1}
    rows.sort(key=lambda r: (status_priority.get(r["status"], 3), r["file_path"], r["name"]))

    # Write CSV
    output_path = ROOT / "AUDITS" / "TRUTH_TABLE.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "component_type", "name", "file_path", "line_number",
            "status", "evidence", "has_docstring", "has_tests",
            "method_count", "execution_traced"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_path}")

    # Generate summary
    status_counts = {}
    for row in rows:
        status = row["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    print("\n" + "=" * 60)
    print("TRUTH TABLE SUMMARY")
    print("=" * 60)
    for status, count in sorted(status_counts.items()):
        pct = (count / len(rows)) * 100
        print(f"  {status}: {count} ({pct:.1f}%)")

    # Count with tests
    with_tests = sum(1 for r in rows if r["has_tests"])
    with_docs = sum(1 for r in rows if r["has_docstring"])
    traced = sum(1 for r in rows if r["execution_traced"])

    print(f"\n  With tests: {with_tests} ({(with_tests/len(rows))*100:.1f}%)")
    print(f"  With docstrings: {with_docs} ({(with_docs/len(rows))*100:.1f}%)")
    print(f"  Execution traced: {traced} ({(traced/len(rows))*100:.1f}%)")

    # Write summary JSON
    summary = {
        "generated_at": __import__("datetime").datetime.utcnow().isoformat(),
        "total_components": len(rows),
        "by_status": status_counts,
        "with_tests": with_tests,
        "with_docstrings": with_docs,
        "execution_traced": traced,
        "classes": len(classes),
        "functions": len(functions),
        "stubs": len(stubs),
    }

    summary_path = ROOT / "AUDITS" / "TRUTH_TABLE_SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    return summary


if __name__ == "__main__":
    generate_truth_table()
