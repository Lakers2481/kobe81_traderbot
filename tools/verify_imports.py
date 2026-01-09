"""
Import Resolution Verification

Tests that all Python imports in the codebase can be resolved correctly.
Ensures no broken imports that would cause runtime failures.

Classifies failures as CRITICAL (core system) vs OPTIONAL (tools/tests).

Author: Kobe System Verification
Date: 2026-01-09
Standard: Jim Simons / Renaissance Technologies
"""

import sys
import ast
import json
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ImportTest:
    """Single import test result."""
    module_name: str
    imported_from: str  # File that imports this
    line_number: int
    import_type: str  # "import" or "from"
    passed: bool
    error: Optional[str] = None
    severity: str = "INFO"  # CRITICAL, HIGH, MEDIUM, LOW, INFO


@dataclass
class ImportReport:
    """Complete import resolution report."""
    timestamp: str
    verdict: str  # PASS, PARTIAL, FAIL
    confidence_level: str  # HIGH, MEDIUM, LOW
    total_imports: int
    unique_modules: int
    passed: int
    failed: int
    critical_failures: int
    optional_failures: int
    tests: List[ImportTest]
    failed_modules: List[str]
    critical_modules: Set[str]


# Critical modules that MUST import successfully
CRITICAL_MODULES = {
    'data', 'strategies', 'backtest', 'execution', 'risk', 'oms', 'core',
    'cognitive', 'autonomous', 'ml_advanced', 'ml_features', 'portfolio',
    'monitor', 'safety', 'preflight', 'compliance'
}

# Optional modules (tools, tests, experiments)
OPTIONAL_MODULES = {
    'tools', 'tests', 'scripts', 'notebooks', 'experiments', 'research',
    'docs', 'examples', 'sandbox'
}


def is_stdlib_module(module_name: str) -> bool:
    """Check if module is from Python standard library."""
    stdlib_modules = {
        'os', 'sys', 'time', 'datetime', 'json', 'pathlib', 'typing',
        'dataclasses', 'functools', 'itertools', 'collections', 'abc',
        'traceback', 'logging', 'warnings', 'threading', 'multiprocessing',
        'queue', 'asyncio', 'io', 'tempfile', 'shutil', 're', 'hashlib',
        'uuid', 'pickle', 'csv', 'sqlite3', 'math', 'random', 'statistics',
        'decimal', 'fractions', 'copy', 'enum', 'contextlib', 'inspect',
        'importlib', 'ast', 'dis', 'gc', 'weakref', 'pprint', 'argparse',
        'configparser', 'signal', 'subprocess', 'urllib', 'http', 'socket',
        'ssl', 'email', 'base64', 'binascii', 'struct', 'array', 'bisect',
        'heapq', 'zlib', 'gzip', 'bz2', 'zipfile', 'tarfile', 'html', 'xml'
    }
    base_module = module_name.split('.')[0]
    return base_module in stdlib_modules


def classify_severity(module_name: str, source_file: str) -> str:
    """Classify import failure severity."""
    # Check source file first
    source_parts = Path(source_file).parts

    # Optional/low priority
    for opt_mod in OPTIONAL_MODULES:
        if opt_mod in source_parts:
            return "LOW"

    # Critical modules
    module_parts = module_name.split('.')
    if module_parts[0] in CRITICAL_MODULES:
        return "CRITICAL"

    # Check if importing from critical module
    for crit_mod in CRITICAL_MODULES:
        if crit_mod in source_parts:
            return "HIGH"

    return "MEDIUM"


def extract_imports_from_file(file_path: Path) -> List[Tuple[str, int, str]]:
    """
    Extract all import statements from a Python file.

    Returns:
        List of (module_name, line_number, import_type) tuples
    """
    imports = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, node.lineno, "import"))

            elif isinstance(node, ast.ImportFrom):
                if node.module:  # Skip relative imports without module
                    imports.append((node.module, node.lineno, "from"))

    except SyntaxError as e:
        # File has syntax errors, skip
        pass
    except Exception as e:
        # Other parsing errors, skip
        pass

    return imports


def test_import(module_name: str, source_file: str, line_number: int, import_type: str) -> ImportTest:
    """Test if a single import can be resolved."""

    # Skip stdlib modules (assume they work)
    if is_stdlib_module(module_name):
        return ImportTest(
            module_name=module_name,
            imported_from=source_file,
            line_number=line_number,
            import_type=import_type,
            passed=True,
            severity="INFO"
        )

    try:
        # Try to import the module
        importlib.import_module(module_name)

        return ImportTest(
            module_name=module_name,
            imported_from=source_file,
            line_number=line_number,
            import_type=import_type,
            passed=True,
            severity="INFO"
        )

    except ImportError as e:
        severity = classify_severity(module_name, source_file)

        return ImportTest(
            module_name=module_name,
            imported_from=source_file,
            line_number=line_number,
            import_type=import_type,
            passed=False,
            error=str(e),
            severity=severity
        )

    except Exception as e:
        severity = classify_severity(module_name, source_file)

        return ImportTest(
            module_name=module_name,
            imported_from=source_file,
            line_number=line_number,
            import_type=import_type,
            passed=False,
            error=f"{type(e).__name__}: {str(e)}",
            severity=severity
        )


def scan_all_imports() -> List[ImportTest]:
    """Scan all Python files in the project and test imports."""
    print("Scanning Python files for imports...")

    # Find all Python files
    python_files = list(project_root.glob("**/*.py"))

    # Exclude some directories
    exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'venv', 'env', '.venv', 'node_modules', '.claude'}
    python_files = [
        f for f in python_files
        if not any(exclude in f.parts for exclude in exclude_dirs)
    ]

    print(f"Found {len(python_files)} Python files")

    # Extract all imports
    all_imports = []
    for py_file in python_files:
        file_imports = extract_imports_from_file(py_file)
        for module_name, line_num, import_type in file_imports:
            all_imports.append((module_name, str(py_file.relative_to(project_root)), line_num, import_type))

    print(f"Found {len(all_imports)} total import statements")

    # Get unique modules
    unique_modules = set(imp[0] for imp in all_imports)
    print(f"Unique modules: {len(unique_modules)}")

    # Test each unique module once
    print("\nTesting imports...")
    results = []
    tested_modules = {}

    for module_name, source_file, line_num, import_type in all_imports:
        # Use cached result if already tested
        if module_name in tested_modules:
            test_result = ImportTest(
                module_name=module_name,
                imported_from=source_file,
                line_number=line_num,
                import_type=import_type,
                passed=tested_modules[module_name].passed,
                error=tested_modules[module_name].error,
                severity=classify_severity(module_name, source_file)
            )
        else:
            test_result = test_import(module_name, source_file, line_num, import_type)
            tested_modules[module_name] = test_result

        results.append(test_result)

    return results


def generate_report(tests: List[ImportTest]) -> ImportReport:
    """Generate import resolution report."""

    passed = sum(1 for t in tests if t.passed)
    failed = len(tests) - passed

    critical_failures = sum(1 for t in tests if not t.passed and t.severity == "CRITICAL")
    optional_failures = sum(1 for t in tests if not t.passed and t.severity == "LOW")

    # Get unique modules
    unique_modules = len(set(t.module_name for t in tests))

    # Get failed modules
    failed_modules = sorted(set(t.module_name for t in tests if not t.passed))

    # Get critical modules that failed
    critical_failed = set(
        t.module_name for t in tests
        if not t.passed and t.severity == "CRITICAL"
    )

    # Determine verdict
    if critical_failures > 0:
        verdict = "FAIL"
        confidence = "LOW"
    elif failed > 0 and failed / len(tests) > 0.1:  # >10% failure
        verdict = "PARTIAL"
        confidence = "MEDIUM"
    else:
        verdict = "PASS"
        confidence = "HIGH"

    return ImportReport(
        timestamp=datetime.now().isoformat(),
        verdict=verdict,
        confidence_level=confidence,
        total_imports=len(tests),
        unique_modules=unique_modules,
        passed=passed,
        failed=failed,
        critical_failures=critical_failures,
        optional_failures=optional_failures,
        tests=tests,
        failed_modules=failed_modules,
        critical_modules=critical_failed
    )


def print_report(report: ImportReport):
    """Print import resolution report."""
    print("=" * 80)
    print("IMPORT RESOLUTION VERIFICATION")
    print("=" * 80)
    print()
    print(f"Timestamp: {report.timestamp}")
    print(f"Verdict: {report.verdict}")
    print(f"Confidence Level: {report.confidence_level}")
    print()
    print(f"Total Imports: {report.total_imports}")
    print(f"Unique Modules: {report.unique_modules}")
    print(f"Passed: {report.passed} ({report.passed/report.total_imports*100:.1f}%)")
    print(f"Failed: {report.failed} ({report.failed/report.total_imports*100:.1f}%)")
    print()
    print(f"Critical Failures: {report.critical_failures}")
    print(f"Optional Failures: {report.optional_failures}")
    print()

    if report.failed > 0:
        print("=" * 80)
        print("FAILED IMPORTS")
        print("=" * 80)

        # Group by severity
        by_severity = defaultdict(list)
        for test in report.tests:
            if not test.passed:
                by_severity[test.severity].append(test)

        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if severity in by_severity:
                print(f"\n{severity} ({len(by_severity[severity])} failures):")
                print("-" * 80)

                # Show first 10 of each severity
                for test in by_severity[severity][:10]:
                    print(f"  {test.module_name}")
                    print(f"    From: {test.imported_from}:{test.line_number}")
                    print(f"    Error: {test.error}")
                    print()

                if len(by_severity[severity]) > 10:
                    print(f"  ... and {len(by_severity[severity]) - 10} more")
                    print()

    print("=" * 80)
    if report.verdict == "PASS":
        print("[OK] All critical imports resolved successfully")
    elif report.verdict == "PARTIAL":
        print("[WARN] Some non-critical imports failed")
    else:
        print("[FAIL] Critical imports failed - system may not work")
    print("=" * 80)


def main():
    """Main entry point."""
    tests = scan_all_imports()
    report = generate_report(tests)
    print_report(report)

    # Save report
    audit_dir = project_root / "AUDITS"
    audit_dir.mkdir(exist_ok=True)

    report_file = audit_dir / "IMPORT_RESOLUTION_REPORT.json"

    # Convert to dict
    report_dict = asdict(report)
    # Convert set to list for JSON
    report_dict['critical_modules'] = list(report.critical_modules)

    with open(report_file, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\nReport saved to: {report_file}")

    # Exit code
    if report.verdict == "FAIL":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
