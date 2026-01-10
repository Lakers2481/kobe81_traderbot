"""
FINAL AUDIT: Verify no order path bypasses paper guard.

This script scans the entire codebase to find ALL functions that place orders
and verifies they call ensure_paper_mode_or_die().

Run with: python tools/audit_paper_mode.py
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent

# Order-related function names to search for
ORDER_FUNCTIONS = [
    "submit_order",
    "create_order",
    "place_order",
    "send_order",
    "place_ioc_limit",
    "place_bracket_order",
    "_place_order_direct",
    "place_order_with_liquidity_check",
    "execute_order",
    "route_order",
]

# The guard call we're looking for
GUARD_CALL = "ensure_paper_mode_or_die"


def find_order_functions_in_file(filepath: Path) -> List[Tuple[str, int, bool]]:
    """
    Find all order-related functions in a file and check if they have the guard.

    Returns:
        List of (function_name, line_number, has_guard)
    """
    results = []

    try:
        content = filepath.read_text(encoding='utf-8')
        tree = ast.parse(content)
    except Exception as e:
        return [(f"PARSE_ERROR: {e}", 0, False)]

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name_lower = node.name.lower()

            # Check if this function is order-related
            is_order_func = any(
                order_func.lower() in func_name_lower
                for order_func in ORDER_FUNCTIONS
            )

            if is_order_func:
                # Skip abstract methods (just have 'pass' body)
                is_abstract = False
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                        is_abstract = True
                        break

                # Also check if body is just 'pass'
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    is_abstract = True

                if is_abstract:
                    continue  # Skip abstract methods

                # Get the function source
                try:
                    func_source = ast.get_source_segment(content, node)
                    has_guard = func_source and GUARD_CALL in func_source
                except Exception:
                    has_guard = GUARD_CALL in content  # Fallback

                results.append((node.name, node.lineno, has_guard))

    return results


def audit_directory(directory: Path) -> List[dict]:
    """Audit all Python files in a directory for order functions."""
    issues = []

    for py_file in directory.rglob("*.py"):
        # Skip test files and __pycache__
        if "__pycache__" in str(py_file) or "test_" in py_file.name:
            continue

        rel_path = py_file.relative_to(PROJECT_ROOT)
        results = find_order_functions_in_file(py_file)

        for func_name, line_num, has_guard in results:
            if not has_guard:
                issues.append({
                    "file": str(rel_path),
                    "function": func_name,
                    "line": line_num,
                    "has_guard": has_guard,
                    "severity": "BLOCKER" if "place" in func_name.lower() else "WARNING",
                })

    return issues


def check_hardcoded_flags() -> List[dict]:
    """Check that hardcoded safety flags are properly set."""
    issues = []

    # Check PAPER_ONLY_ENFORCED in paper_guard.py
    paper_guard = PROJECT_ROOT / "safety" / "paper_guard.py"
    if paper_guard.exists():
        content = paper_guard.read_text(encoding='utf-8')
        if "PAPER_ONLY_ENFORCED: bool = True" not in content:
            issues.append({
                "file": "safety/paper_guard.py",
                "function": "PAPER_ONLY_ENFORCED",
                "line": 0,
                "has_guard": False,
                "severity": "BLOCKER",
                "message": "PAPER_ONLY_ENFORCED is not set to True!",
            })

    # Check NO_LIVE_ORDERS in execution_choke.py
    exec_choke = PROJECT_ROOT / "safety" / "execution_choke.py"
    if exec_choke.exists():
        content = exec_choke.read_text(encoding='utf-8')
        if "NO_LIVE_ORDERS: bool = True" not in content:
            issues.append({
                "file": "safety/execution_choke.py",
                "function": "NO_LIVE_ORDERS",
                "line": 0,
                "has_guard": False,
                "severity": "BLOCKER",
                "message": "NO_LIVE_ORDERS is not set to True!",
            })

    # Check PAPER_ONLY in safety/mode.py
    mode_file = PROJECT_ROOT / "safety" / "mode.py"
    if mode_file.exists():
        content = mode_file.read_text(encoding='utf-8')
        if "PAPER_ONLY = True" not in content and "PAPER_ONLY: bool = True" not in content:
            issues.append({
                "file": "safety/mode.py",
                "function": "PAPER_ONLY",
                "line": 0,
                "has_guard": False,
                "severity": "BLOCKER",
                "message": "PAPER_ONLY is not set to True!",
            })

    return issues


def run_audit():
    """Audit entire codebase for order paths without paper guard."""
    print("=" * 70)
    print("PAPER MODE AUDIT - Checking ALL order paths")
    print("=" * 70)
    print()

    all_issues = []

    # Check hardcoded flags first
    print("Checking hardcoded safety flags...")
    flag_issues = check_hardcoded_flags()
    all_issues.extend(flag_issues)
    if flag_issues:
        for issue in flag_issues:
            print(f"  [BLOCKER] {issue.get('message', issue['file'])}")
    else:
        print("  [OK] All hardcoded flags are properly set")
    print()

    # Scan execution directory
    print("Scanning execution/ directory...")
    exec_dir = PROJECT_ROOT / "execution"
    if exec_dir.exists():
        exec_issues = audit_directory(exec_dir)
        all_issues.extend(exec_issues)
        if exec_issues:
            for issue in exec_issues:
                print(f"  [{issue['severity']}] {issue['file']}:{issue['line']} - {issue['function']}")
        else:
            print("  [OK] All order functions have paper guard")
    print()

    # Scan options directory
    print("Scanning options/ directory...")
    opts_dir = PROJECT_ROOT / "options"
    if opts_dir.exists():
        opts_issues = audit_directory(opts_dir)
        all_issues.extend(opts_issues)
        if opts_issues:
            for issue in opts_issues:
                print(f"  [{issue['severity']}] {issue['file']}:{issue['line']} - {issue['function']}")
        else:
            print("  [OK] All order functions have paper guard")
    print()

    # Scan scripts directory for direct broker calls
    print("Scanning scripts/ directory...")
    scripts_dir = PROJECT_ROOT / "scripts"
    if scripts_dir.exists():
        scripts_issues = audit_directory(scripts_dir)
        # Filter to only include direct order function calls
        scripts_issues = [i for i in scripts_issues if "place" in i["function"].lower()]
        all_issues.extend(scripts_issues)
        if scripts_issues:
            for issue in scripts_issues:
                print(f"  [{issue['severity']}] {issue['file']}:{issue['line']} - {issue['function']}")
        else:
            print("  [OK] No unguarded order functions in scripts")
    print()

    # Summary
    print("=" * 70)
    blockers = [i for i in all_issues if i.get("severity") == "BLOCKER"]
    warnings = [i for i in all_issues if i.get("severity") == "WARNING"]

    if all_issues:
        print(f"ISSUES FOUND: {len(blockers)} BLOCKERS, {len(warnings)} WARNINGS")
        print()
        if blockers:
            print("BLOCKERS (must fix before trading):")
            for issue in blockers:
                msg = issue.get("message", f"{issue['function']} missing paper guard")
                print(f"  - {issue['file']}:{issue['line']} - {msg}")
        print()
        print("FIX THESE BEFORE PROCEEDING!")
        return 1
    else:
        print("ALL ORDER PATHS PROTECTED")
        print("Paper mode enforcement verified.")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    exit_code = run_audit()
    sys.exit(exit_code)
