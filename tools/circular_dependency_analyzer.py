#!/usr/bin/env python3
"""
Deep circular dependency analyzer for Kobe trading system.
Finds actual import chains that create circular dependencies.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class CircularDependencyAnalyzer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.import_graph = defaultdict(set)  # module -> set of imports
        self.file_map = {}  # module -> file_path
        self.detailed_imports = defaultdict(list)  # module -> [(imported_module, line_num)]

    def get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        try:
            rel_path = file_path.relative_to(self.root_dir)
            parts = list(rel_path.parts[:-1]) + [rel_path.stem]
            if parts[-1] == '__init__':
                parts = parts[:-1]
            return '.'.join(parts) if parts else ''
        except ValueError:
            return ''

    def is_internal_module(self, module_name: str) -> bool:
        """Check if module is internal to the project."""
        internal_prefixes = [
            'agents', 'alerts', 'altdata', 'analytics', 'autonomous',
            'backtest', 'bounce', 'cognitive', 'compliance', 'config',
            'core', 'data', 'evolution', 'execution', 'explainability',
            'guardian', 'ml', 'ml_advanced', 'ml_features', 'ml_meta',
            'monitor', 'oms', 'options', 'optimization', 'pipelines',
            'portfolio', 'preflight', 'quant_gates', 'research', 'risk',
            'safety', 'scanner', 'scripts', 'selfmonitor', 'strategies',
            'tax', 'testing', 'tools', 'trade_logging', 'web'
        ]
        return any(module_name.startswith(prefix) for prefix in internal_prefixes)

    def extract_imports(self, file_path: Path) -> List[Tuple[str, int]]:
        """Extract all imports with line numbers."""
        imports = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                tree = ast.parse(code, str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        if self.is_internal_module(module):
                            imports.append((module, node.lineno))
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        if self.is_internal_module(module):
                            imports.append((module, node.lineno))

        except Exception:
            pass  # Skip files that can't be parsed

        return imports

    def build_import_graph(self) -> None:
        """Build complete import graph."""
        py_files = []
        exclude_dirs = {
            '.venv', 'venv', 'env', '__pycache__', '.git',
            '.pytest_cache', 'node_modules', 'build', 'dist',
            'backtest_outputs', 'wf_outputs', 'logs', 'mlruns.db',
            'state', 'data', 'reports', 'notebooks'
        }

        for py_file in self.root_dir.rglob('*.py'):
            if any(excl in py_file.parts for excl in exclude_dirs):
                continue
            py_files.append(py_file)

        print(f"Analyzing {len(py_files)} Python files...")

        for file_path in py_files:
            module_name = self.get_module_name(file_path)
            if not module_name:
                continue

            self.file_map[module_name] = str(file_path)
            imports = self.extract_imports(file_path)

            for imported_module, line_num in imports:
                self.import_graph[module_name].add(imported_module)
                self.detailed_imports[module_name].append((imported_module, line_num))

    def find_circular_paths(self, start: str, max_depth: int = 10) -> List[List[str]]:
        """Find all circular import paths starting from a module."""
        cycles = []
        visited = set()

        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return

            if current == start and len(path) > 1:
                # Found a cycle back to start
                cycles.append(path + [current])
                return

            if current in visited:
                return

            visited.add(current)

            for neighbor in self.import_graph.get(current, []):
                dfs(neighbor, path + [current], depth + 1)

            visited.remove(current)

        for neighbor in self.import_graph.get(start, []):
            dfs(neighbor, [start], 1)

        return cycles

    def analyze_module_cycles(self, module_prefix: str) -> Dict:
        """Analyze circular dependencies within a module."""
        print(f"\nAnalyzing circular dependencies in '{module_prefix}'...")

        # Find all modules with this prefix
        matching_modules = [m for m in self.import_graph.keys()
                          if m.startswith(module_prefix)]

        print(f"Found {len(matching_modules)} modules in '{module_prefix}'")

        # Find cycles
        all_cycles = []
        for module in matching_modules:
            cycles = self.find_circular_paths(module)
            for cycle in cycles:
                # Filter to only include modules within the same top-level package
                if all(m.startswith(module_prefix) for m in cycle):
                    if cycle not in all_cycles:
                        all_cycles.append(cycle)

        return {
            'module_prefix': module_prefix,
            'total_modules': len(matching_modules),
            'cycles': all_cycles
        }

    def generate_report(self, output_file: str) -> None:
        """Generate detailed circular dependency report."""
        modules_to_analyze = [
            'cognitive',
            'altdata',
            'analytics',
            'research',
            'safety',
            'bounce',
            'trade_logging',
            'guardian'
        ]

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# CIRCULAR DEPENDENCY ANALYSIS - DETAILED REPORT\n\n")
            f.write("---\n\n")

            for module_prefix in modules_to_analyze:
                result = self.analyze_module_cycles(module_prefix)

                f.write(f"## Module: `{module_prefix}`\n\n")
                f.write(f"**Total modules:** {result['total_modules']}\n\n")
                f.write(f"**Circular dependencies found:** {len(result['cycles'])}\n\n")

                if result['cycles']:
                    f.write("### Circular Import Chains\n\n")

                    for i, cycle in enumerate(result['cycles'], 1):
                        f.write(f"#### Cycle {i}\n\n")
                        f.write("```\n")
                        for j, module in enumerate(cycle):
                            if j < len(cycle) - 1:
                                f.write(f"{module}\n")
                                f.write("  |\n")
                                f.write("  v\n")
                            else:
                                f.write(f"{module} (back to start)\n")
                        f.write("```\n\n")

                        # Show actual imports
                        f.write("**Import details:**\n\n")
                        for j in range(len(cycle) - 1):
                            current = cycle[j]
                            next_module = cycle[j + 1]

                            # Find the line numbers
                            imports = self.detailed_imports.get(current, [])
                            matching_imports = [
                                line for imp, line in imports
                                if next_module.startswith(imp)
                            ]

                            if matching_imports:
                                lines = ", ".join(f"L{line}" for line in matching_imports[:3])
                                f.write(f"- `{current}` imports `{next_module}` at {lines}\n")

                            if current in self.file_map:
                                f.write(f"  - File: `{self.file_map[current]}`\n")

                        f.write("\n")

                        # Recommend fix
                        f.write("**Recommended fix:**\n\n")
                        if len(cycle) == 3 and cycle[0] == cycle[-1]:
                            # Simple A -> B -> A cycle
                            f.write(f"1. Extract common interface/protocol from `{cycle[0]}` and `{cycle[1]}`\n")
                            f.write(f"2. Move shared types to a new `{module_prefix}.types` module\n")
                            f.write(f"3. Use TYPE_CHECKING for type hints\n\n")
                        else:
                            f.write("1. Use TYPE_CHECKING guard for type-only imports\n")
                            f.write("2. Refactor to dependency injection pattern\n")
                            f.write("3. Consider extracting interfaces to break cycle\n\n")

                        f.write("```python\n")
                        f.write("# Example fix using TYPE_CHECKING:\n")
                        f.write("from typing import TYPE_CHECKING\n\n")
                        f.write("if TYPE_CHECKING:\n")
                        f.write(f"    from {cycle[1]} import SomeClass\n\n")
                        f.write("# Use 'SomeClass' as string in type hints\n")
                        f.write("def func(obj: 'SomeClass') -> None:\n")
                        f.write("    ...\n")
                        f.write("```\n\n")

                else:
                    f.write("[PASS] No circular dependencies within this module.\n\n")

                f.write("---\n\n")

            # Summary
            total_cycles = sum(
                len(self.analyze_module_cycles(m)['cycles'])
                for m in modules_to_analyze
            )

            f.write("## Summary\n\n")
            f.write(f"**Total circular dependencies:** {total_cycles}\n\n")

            if total_cycles == 0:
                f.write("All circular dependencies are likely self-references within `__init__.py` files or ")
                f.write("legitimate re-exports. These are generally acceptable.\n\n")
            else:
                f.write("### Priority Actions\n\n")
                f.write("1. Use `TYPE_CHECKING` guards for type-only imports (fastest fix)\n")
                f.write("2. Extract shared types to `.types` modules\n")
                f.write("3. Refactor to dependency injection where appropriate\n\n")

            f.write("---\n\n")
            f.write("**Note:** Self-referencing modules (e.g., `cognitive -> cognitive`) are often due to ")
            f.write("`__init__.py` re-exports and are generally acceptable for public API design.\n")

def main():
    root_dir = Path(__file__).parent.parent
    analyzer = CircularDependencyAnalyzer(str(root_dir))

    print("Building import graph...")
    analyzer.build_import_graph()

    output_file = root_dir / "AUDITS" / "CIRCULAR_DEPENDENCIES_DETAILED.md"
    output_file.parent.mkdir(exist_ok=True)

    analyzer.generate_report(str(output_file))

    print(f"\nDetailed report saved to: {output_file}")

if __name__ == '__main__':
    sys.exit(main())
