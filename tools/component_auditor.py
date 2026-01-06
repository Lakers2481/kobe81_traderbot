#!/usr/bin/env python3
"""
COMPONENT AUDITOR - Full AST-Based Codebase Analysis

This script performs a comprehensive audit of the KOBE trading system by:
1. AST parsing all .py files to extract classes, functions, and constants
2. Building an import graph (what imports what)
3. Detecting stub patterns (pass, NotImplementedError, TODO)
4. Generating AUDITS/02_COMPONENT_INVENTORY.json with full inventory

Usage:
    python tools/component_auditor.py
    python tools/component_auditor.py --verbose
    python tools/component_auditor.py --json

Author: KOBE Trading System Auditor
Version: 1.0.0
Date: 2026-01-05
"""

from __future__ import annotations

import ast
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Directories to exclude from scanning
EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", ".tox", "build", "dist", "*.egg-info", ".claude",
    "wf_outputs", "data/polygon_cache", "data/lake/datasets"
}


@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    file: str
    line: int
    bases: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class FunctionInfo:
    """Information about a function or method."""
    name: str
    file: str
    line: int
    is_method: bool = False
    parent_class: Optional[str] = None
    args: List[str] = field(default_factory=list)
    returns: Optional[str] = None
    docstring: Optional[str] = None
    is_stub: bool = False
    stub_type: Optional[str] = None  # "pass", "not_implemented", "todo"


@dataclass
class ConstantInfo:
    """Information about a module-level constant."""
    name: str
    file: str
    line: int
    value_type: str
    value_repr: Optional[str] = None


@dataclass
class ImportInfo:
    """Information about an import."""
    module: str
    names: List[str] = field(default_factory=list)
    is_from_import: bool = False
    line: int = 0


@dataclass
class FileAnalysis:
    """Analysis results for a single file."""
    file_path: str
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    constants: List[ConstantInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class ASTAnalyzer(ast.NodeVisitor):
    """AST visitor that extracts classes, functions, constants, and imports."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.relative_path = str(Path(file_path).relative_to(ROOT))
        self.classes: List[ClassInfo] = []
        self.functions: List[FunctionInfo] = []
        self.constants: List[ConstantInfo] = []
        self.imports: List[ImportInfo] = []
        self._current_class: Optional[str] = None
        self._source_lines: List[str] = []

    def analyze(self, source: str) -> FileAnalysis:
        """Analyze the source code and return results."""
        self._source_lines = source.split('\n')
        try:
            tree = ast.parse(source)
            self.visit(tree)
            return FileAnalysis(
                file_path=self.relative_path,
                classes=self.classes,
                functions=self.functions,
                constants=self.constants,
                imports=self.imports
            )
        except SyntaxError as e:
            return FileAnalysis(
                file_path=self.relative_path,
                errors=[f"SyntaxError: {e}"]
            )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{self._get_attribute_name(base)}")

        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                methods.append(item.name)

        docstring = ast.get_docstring(node)

        self.classes.append(ClassInfo(
            name=node.name,
            file=self.relative_path,
            line=node.lineno,
            bases=bases,
            methods=methods,
            docstring=docstring[:200] if docstring else None
        ))

        # Visit methods within the class
        old_class = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an async function definition."""
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Common logic for function/async function visits."""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        returns = None
        if node.returns:
            returns = self._get_annotation_str(node.returns)

        docstring = ast.get_docstring(node)

        # Check if stub
        is_stub = False
        stub_type = None

        # Check for pass-only body
        if len(node.body) == 1:
            stmt = node.body[0]
            if isinstance(stmt, ast.Pass):
                is_stub = True
                stub_type = "pass"
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                # Just a docstring, check next statement
                pass
            elif isinstance(stmt, ast.Raise):
                if isinstance(stmt.exc, ast.Call):
                    if isinstance(stmt.exc.func, ast.Name):
                        if stmt.exc.func.id == "NotImplementedError":
                            is_stub = True
                            stub_type = "not_implemented"

        # Check for pass after docstring
        if len(node.body) == 2:
            first = node.body[0]
            second = node.body[1]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and
                isinstance(second, ast.Pass)):
                is_stub = True
                stub_type = "pass"
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and
                isinstance(second, ast.Raise)):
                if isinstance(second.exc, ast.Call) and isinstance(second.exc.func, ast.Name):
                    if second.exc.func.id == "NotImplementedError":
                        is_stub = True
                        stub_type = "not_implemented"

        # Check for TODO in docstring or function body
        if docstring and "TODO" in docstring.upper():
            is_stub = True
            stub_type = "todo"

        # Check function body for TODO comments
        start_line = node.lineno - 1
        end_line = node.end_lineno if node.end_lineno else start_line + 10
        for i in range(start_line, min(end_line, len(self._source_lines))):
            line = self._source_lines[i]
            if "# TODO" in line or "#TODO" in line:
                is_stub = True
                stub_type = "todo"
                break

        self.functions.append(FunctionInfo(
            name=node.name,
            file=self.relative_path,
            line=node.lineno,
            is_method=self._current_class is not None,
            parent_class=self._current_class,
            args=args,
            returns=returns,
            docstring=docstring[:200] if docstring else None,
            is_stub=is_stub,
            stub_type=stub_type
        ))

        # Don't recurse into nested functions

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit an assignment (potential constant)."""
        if self._current_class is not None:
            return  # Skip class attributes

        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                # Check if it looks like a constant (UPPER_CASE)
                if name.isupper() or (name[0].isupper() and '_' in name):
                    value_type = type(self._get_value_type(node.value)).__name__
                    value_repr = self._get_value_repr(node.value)

                    self.constants.append(ConstantInfo(
                        name=name,
                        file=self.relative_path,
                        line=node.lineno,
                        value_type=value_type,
                        value_repr=value_repr[:100] if value_repr else None
                    ))

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit an annotated assignment (potential typed constant)."""
        if self._current_class is not None:
            return

        if isinstance(node.target, ast.Name):
            name = node.target.id
            if name.isupper() or (name[0].isupper() and '_' in name):
                value_type = self._get_annotation_str(node.annotation)
                value_repr = self._get_value_repr(node.value) if node.value else None

                self.constants.append(ConstantInfo(
                    name=name,
                    file=self.relative_path,
                    line=node.lineno,
                    value_type=value_type,
                    value_repr=value_repr[:100] if value_repr else None
                ))

    def visit_Import(self, node: ast.Import) -> None:
        """Visit an import statement."""
        for alias in node.names:
            self.imports.append(ImportInfo(
                module=alias.name,
                names=[alias.asname if alias.asname else alias.name],
                is_from_import=False,
                line=node.lineno
            ))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit a from...import statement."""
        module = node.module or ""
        names = [alias.name for alias in node.names]

        self.imports.append(ImportInfo(
            module=module,
            names=names,
            is_from_import=True,
            line=node.lineno
        ))

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get the full name of an attribute access."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    def _get_annotation_str(self, node: ast.expr) -> str:
        """Get string representation of a type annotation."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Subscript):
            return f"{self._get_annotation_str(node.value)}[...]"
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        return "Any"

    def _get_value_type(self, node: ast.expr) -> Any:
        """Get the type of a value node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.List):
            return []
        elif isinstance(node, ast.Dict):
            return {}
        elif isinstance(node, ast.Set):
            return set()
        elif isinstance(node, ast.Tuple):
            return ()
        return None

    def _get_value_repr(self, node: ast.expr) -> Optional[str]:
        """Get string representation of a value."""
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.List):
            return f"[...] ({len(node.elts)} items)"
        elif isinstance(node, ast.Dict):
            return f"{{...}} ({len(node.keys)} items)"
        elif isinstance(node, ast.Set):
            return f"{{...}} ({len(node.elts)} items)"
        elif isinstance(node, ast.Tuple):
            return f"(...) ({len(node.elts)} items)"
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return f"{node.func.id}(...)"
            elif isinstance(node.func, ast.Attribute):
                return f"{self._get_attribute_name(node.func)}(...)"
        return "..."


def should_skip_path(path: Path) -> bool:
    """Check if a path should be skipped."""
    for part in path.parts:
        if part in EXCLUDE_DIRS or part.startswith('.'):
            return True
        for pattern in EXCLUDE_DIRS:
            if '*' in pattern and part.endswith(pattern.replace('*', '')):
                return True
    return False


def scan_python_files(root_path: Path) -> List[Path]:
    """Find all Python files in the project."""
    python_files = []

    for py_file in root_path.rglob("*.py"):
        if should_skip_path(py_file.relative_to(root_path)):
            continue
        python_files.append(py_file)

    return sorted(python_files)


def build_import_graph(analyses: List[FileAnalysis]) -> Dict[str, Any]:
    """Build forward and reverse import graphs."""
    # Forward graph: file -> [modules it imports]
    forward_graph: Dict[str, List[str]] = defaultdict(list)

    # Reverse graph: module -> [files that import it]
    reverse_graph: Dict[str, List[str]] = defaultdict(list)

    # Map module paths to files
    file_to_module: Dict[str, str] = {}
    for analysis in analyses:
        # Convert file path to module path
        module_path = analysis.file_path.replace("\\", "/").replace("/", ".").replace(".py", "")
        if module_path.endswith(".__init__"):
            module_path = module_path[:-9]  # Remove .__init__
        file_to_module[analysis.file_path] = module_path

    for analysis in analyses:
        for imp in analysis.imports:
            forward_graph[analysis.file_path].append(imp.module)
            reverse_graph[imp.module].append(analysis.file_path)

    return {
        "forward": dict(forward_graph),
        "reverse": dict(reverse_graph),
        "file_to_module": file_to_module
    }


def generate_inventory(analyses: List[FileAnalysis], import_graph: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the full component inventory."""

    # Aggregate all components
    all_classes = []
    all_functions = []
    all_constants = []
    all_stubs = []

    for analysis in analyses:
        for cls in analysis.classes:
            all_classes.append(asdict(cls))

        for func in analysis.functions:
            func_dict = asdict(func)
            all_functions.append(func_dict)
            if func.is_stub:
                all_stubs.append({
                    "name": func.name,
                    "file": func.file,
                    "line": func.line,
                    "stub_type": func.stub_type,
                    "parent_class": func.parent_class
                })

        for const in analysis.constants:
            all_constants.append(asdict(const))

    # Calculate statistics
    stats = {
        "total_files": len(analyses),
        "total_classes": len(all_classes),
        "total_functions": len(all_functions),
        "total_methods": sum(1 for f in all_functions if f["is_method"]),
        "total_standalone_functions": sum(1 for f in all_functions if not f["is_method"]),
        "total_constants": len(all_constants),
        "total_stubs": len(all_stubs),
        "stubs_by_type": {
            "pass": sum(1 for s in all_stubs if s["stub_type"] == "pass"),
            "not_implemented": sum(1 for s in all_stubs if s["stub_type"] == "not_implemented"),
            "todo": sum(1 for s in all_stubs if s["stub_type"] == "todo")
        },
        "files_with_errors": sum(1 for a in analyses if a.errors)
    }

    # Group classes by category
    classes_by_category: Dict[str, List[Dict]] = defaultdict(list)
    for cls in all_classes:
        category = cls["file"].split("\\")[0] if "\\" in cls["file"] else cls["file"].split("/")[0]
        classes_by_category[category].append(cls)

    return {
        "metadata": {
            "generated": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "root": str(ROOT)
        },
        "statistics": stats,
        "classes": all_classes,
        "functions": all_functions,
        "constants": all_constants,
        "stubs": all_stubs,
        "classes_by_category": dict(classes_by_category),
        "import_graph": import_graph,
        "files_with_errors": [
            {"file": a.file_path, "errors": a.errors}
            for a in analyses if a.errors
        ]
    }


def print_summary(inventory: Dict[str, Any], verbose: bool = False) -> None:
    """Print a summary of the inventory."""
    stats = inventory["statistics"]

    print("=" * 70)
    print("KOBE COMPONENT AUDITOR - FULL INVENTORY")
    print("=" * 70)
    print()

    print("SUMMARY STATISTICS")
    print("-" * 40)
    print(f"Total Python Files:       {stats['total_files']:>6}")
    print(f"Total Classes:            {stats['total_classes']:>6}")
    print(f"Total Functions/Methods:  {stats['total_functions']:>6}")
    print(f"  - Methods:              {stats['total_methods']:>6}")
    print(f"  - Standalone Functions: {stats['total_standalone_functions']:>6}")
    print(f"Total Constants:          {stats['total_constants']:>6}")
    print(f"Total Stubs:              {stats['total_stubs']:>6}")
    print()

    print("STUB BREAKDOWN")
    print("-" * 40)
    stubs = stats["stubs_by_type"]
    print(f"  pass only:              {stubs['pass']:>6}")
    print(f"  NotImplementedError:    {stubs['not_implemented']:>6}")
    print(f"  TODO comments:          {stubs['todo']:>6}")
    print()

    print("CLASSES BY CATEGORY")
    print("-" * 40)
    for category, classes in sorted(inventory["classes_by_category"].items()):
        print(f"  {category:30} {len(classes):>4} classes")
    print()

    if inventory["files_with_errors"]:
        print("FILES WITH PARSE ERRORS")
        print("-" * 40)
        for file_info in inventory["files_with_errors"]:
            print(f"  {file_info['file']}: {file_info['errors']}")
        print()

    if verbose:
        print("STUBS DETECTED")
        print("-" * 40)
        for stub in inventory["stubs"][:20]:
            parent = f" ({stub['parent_class']})" if stub.get("parent_class") else ""
            print(f"  [{stub['stub_type']:15}] {stub['name']}{parent}")
            print(f"    {stub['file']}:{stub['line']}")
        if len(inventory["stubs"]) > 20:
            print(f"  ... and {len(inventory['stubs']) - 20} more")
        print()

    print("=" * 70)
    print(f"CLASS COUNT:    {stats['total_classes']}")
    print(f"FUNCTION COUNT: {stats['total_functions']}")
    print(f"STUB COUNT:     {stats['total_stubs']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="KOBE Component Auditor")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--json", action="store_true", help="Output as JSON only")
    args = parser.parse_args()

    print("Scanning Python files...")
    python_files = scan_python_files(ROOT)
    print(f"Found {len(python_files)} Python files")

    print("Analyzing files...")
    analyses: List[FileAnalysis] = []

    for py_file in python_files:
        try:
            source = py_file.read_text(encoding="utf-8")
            analyzer = ASTAnalyzer(str(py_file))
            analysis = analyzer.analyze(source)
            analyses.append(analysis)
        except Exception as e:
            analyses.append(FileAnalysis(
                file_path=str(py_file.relative_to(ROOT)),
                errors=[str(e)]
            ))

    print("Building import graph...")
    import_graph = build_import_graph(analyses)

    print("Generating inventory...")
    inventory = generate_inventory(analyses, import_graph)

    # Save to AUDITS directory
    audits_dir = ROOT / "AUDITS"
    audits_dir.mkdir(exist_ok=True)

    output_path = audits_dir / "02_COMPONENT_INVENTORY.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    if args.json:
        print(json.dumps(inventory, indent=2))
    else:
        print_summary(inventory, args.verbose)
        print(f"\nInventory saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
