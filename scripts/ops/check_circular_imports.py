#!/usr/bin/env python3
"""Check for circular imports in the codebase"""
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List

def extract_imports(file_path: Path) -> Set[str]:
    """Extract module imports from a Python file."""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Match 'import module' and 'from module import ...'
        import_patterns = [
            r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_\.]*)',
            r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_\.]*)\s+import',
        ]

        for line in content.split('\n'):
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    module = match.group(1)
                    # Only track local imports (not stdlib or external packages)
                    if not module.startswith(('os', 'sys', 'datetime', 'typing', 'pathlib',
                                             'json', 'pandas', 'numpy', 'requests', 'yaml',
                                             'logging', 'argparse', 'dataclasses', 'enum',
                                             'subprocess', 'time', 'signal', 'atexit',
                                             'collections', 're', 'hashlib', 'uuid',
                                             'sklearn', 'scipy', 'matplotlib', 'seaborn',
                                             'pytest', 'flask', 'fastapi', 'pydantic')):
                        imports.add(module.split('.')[0])  # Get top-level module
    except Exception:
        pass

    return imports

def get_module_name(file_path: Path, root: Path) -> str:
    """Get module name from file path."""
    rel_path = file_path.relative_to(root)
    parts = list(rel_path.parts[:-1])  # Remove filename

    if file_path.name != '__init__.py':
        parts.append(file_path.stem)  # Add module name

    return '.'.join(parts) if parts else '__main__'

def find_circular_imports(root_dir: Path) -> List[List[str]]:
    """Find circular imports in the codebase."""
    # Build dependency graph
    dependencies: Dict[str, Set[str]] = defaultdict(set)
    module_files: Dict[str, Path] = {}

    # Scan all Python files
    exclude_patterns = ['.git', '__pycache__', 'wf_outputs', 'smoke_', 'showdown_', 'optimize_outputs']

    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        if any(pattern in root for pattern in exclude_patterns):
            continue

        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                module = get_module_name(file_path, root_dir)
                module_files[module] = file_path

                # Extract imports
                imports = extract_imports(file_path)
                dependencies[module] = imports

    # Find cycles using DFS
    def find_cycle(node: str, visited: Set[str], rec_stack: List[str]) -> List[str]:
        visited.add(node)
        rec_stack.append(node)

        for neighbor in dependencies.get(node, set()):
            if neighbor not in module_files:
                continue  # Skip external imports

            if neighbor not in visited:
                cycle = find_cycle(neighbor, visited, rec_stack)
                if cycle:
                    return cycle
            elif neighbor in rec_stack:
                # Found a cycle
                cycle_start = rec_stack.index(neighbor)
                return rec_stack[cycle_start:] + [neighbor]

        rec_stack.pop()
        return []

    # Check all modules for cycles
    cycles = []
    visited = set()

    for module in sorted(module_files.keys()):
        if module not in visited:
            cycle = find_cycle(module, visited, [])
            if cycle and cycle not in cycles:
                cycles.append(cycle)

    return cycles

if __name__ == '__main__':
    root = Path('.')
    cycles = find_circular_imports(root)

    if cycles:
        print(f"Found {len(cycles)} circular import(s):")
        for i, cycle in enumerate(cycles, 1):
            print(f"\n{i}. Cycle:")
            for j, module in enumerate(cycle):
                print(f"   {module}", end='')
                if j < len(cycle) - 1:
                    print(' ->')
                else:
                    print()
    else:
        print("No circular imports detected")
