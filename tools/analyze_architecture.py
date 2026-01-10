#!/usr/bin/env python3
"""
System Architecture Analyzer
Comprehensive file inventory, duplicate detection, and dependency mapping
"""
import os
import sys
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent

def get_all_python_files():
    """Get all Python files excluding cache/archive/outputs"""
    exclude_dirs = {
        '.git', '__pycache__', 'archive', 'simulation', 'mlruns',
        'backtest_outputs', 'wf_outputs', 'smoke_outputs', 'showdown_outputs',
        'optimize_outputs', 'smoke_wf_audit', 'smoke_turtle_soup',
        'showdown_2025_cap60', '.pytest_cache', '.ruff_cache'
    }

    files = []
    for root, dirs, filenames in os.walk(BASE_DIR):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.endswith('_outputs')]

        for filename in filenames:
            if filename.endswith('.py'):
                filepath = Path(root) / filename
                try:
                    stat = filepath.stat()
                    files.append({
                        'path': str(filepath.relative_to(BASE_DIR)),
                        'name': filename,
                        'size': stat.st_size,
                        'mtime': int(stat.st_mtime),
                        'module': get_module(filepath)
                    })
                except Exception as e:
                    print(f"Warning: Could not stat {filepath}: {e}", file=sys.stderr)

    return files

def get_module(filepath):
    """Get top-level module from filepath"""
    rel_path = filepath.relative_to(BASE_DIR)
    parts = rel_path.parts
    if len(parts) > 1:
        return parts[0]
    return 'root'

def categorize_files(files):
    """Categorize files by module"""
    categories = defaultdict(list)
    for f in files:
        categories[f['module']].append(f)
    return dict(categories)

def find_duplicates(files):
    """Find duplicate filenames"""
    by_name = defaultdict(list)
    for f in files:
        by_name[f['name']].append(f['path'])

    # Return only duplicates (excluding __init__.py)
    return {name: paths for name, paths in by_name.items()
            if len(paths) > 1 and not name.startswith('__')}

def analyze_imports(filepath):
    """Extract imports from a Python file"""
    imports = set()
    try:
        with open(BASE_DIR / filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('from ') and ' import ' in line:
                    # from module import ...
                    module = line.split('from ')[1].split(' import')[0].strip()
                    imports.add(module)
                elif line.startswith('import '):
                    # import module
                    module = line.split('import ')[1].split(' ')[0].split(',')[0].strip()
                    imports.add(module)
    except Exception as e:
        pass
    return list(imports)

def generate_report(files, categories, duplicates):
    """Generate comprehensive architecture report"""
    total_size = sum(f['size'] for f in files)

    report = []
    report.append("=" * 80)
    report.append("KOBE TRADING SYSTEM - ARCHITECTURE ANALYSIS REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")

    # Summary
    report.append("=== INVENTORY SUMMARY ===")
    report.append(f"Total Python Files: {len(files)}")
    report.append(f"Total Size: {total_size / 1024 / 1024:.2f} MB")
    report.append(f"Modules: {len(categories)}")
    report.append(f"Duplicate Filenames: {len(duplicates)}")
    report.append("")

    # Files per module
    report.append("=== FILES PER MODULE ===")
    sorted_modules = sorted(categories.items(), key=lambda x: -len(x[1]))
    for module, module_files in sorted_modules[:40]:
        count = len(module_files)
        size_mb = sum(f['size'] for f in module_files) / 1024 / 1024
        report.append(f"{module:30s} {count:4d} files  {size_mb:8.2f} MB")
    report.append("")

    # Duplicates
    if duplicates:
        report.append(f"=== DUPLICATE FILENAMES ({len(duplicates)} found) ===")
        report.append("Files with same name in different locations:")
        report.append("")
        for name, paths in sorted(duplicates.items()):
            report.append(f"{name}:")
            for p in sorted(paths):
                file_info = next(f for f in files if f['path'] == p)
                size_kb = file_info['size'] / 1024
                mtime = datetime.fromtimestamp(file_info['mtime']).strftime('%Y-%m-%d')
                report.append(f"  - {p:70s} {size_kb:8.1f} KB  {mtime}")
            report.append("")

    return "\n".join(report)

def main():
    print("Scanning Python files...")
    files = get_all_python_files()

    print(f"Found {len(files)} Python files")
    print("Categorizing...")
    categories = categorize_files(files)

    print("Detecting duplicates...")
    duplicates = find_duplicates(files)

    print("Generating report...")
    report = generate_report(files, categories, duplicates)

    # Write report
    output_file = BASE_DIR / 'AUDITS' / 'ARCHITECTURE_INVENTORY.txt'
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report)

    print(report)
    print()
    print(f"Report saved to: {output_file}")

    # Save JSON data for further analysis
    json_file = BASE_DIR / 'AUDITS' / 'file_inventory.json'
    with open(json_file, 'w') as f:
        json.dump({
            'files': files,
            'categories': {k: len(v) for k, v in categories.items()},
            'duplicates': duplicates,
            'generated': datetime.now().isoformat()
        }, f, indent=2)
    print(f"JSON data saved to: {json_file}")

if __name__ == '__main__':
    main()
