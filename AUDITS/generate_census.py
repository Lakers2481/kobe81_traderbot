#!/usr/bin/env python3
"""Generate repository census report and manifests."""

import os
from pathlib import Path
from collections import defaultdict
import datetime

def main():
    os.chdir(Path(__file__).parent.parent)

    # Collect detailed statistics
    ext_counts = defaultdict(int)
    dir_file_counts = defaultdict(int)
    top_level_dirs = set()
    total_lines_py = 0
    largest_files = []

    skip_dirs = {'.git', '__pycache__', 'node_modules', 'venv', '.venv', 'env', '.pytest_cache', '.mypy_cache'}

    for root, dirs, files in os.walk('.'):
        # Skip hidden dirs and common exclusions
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in skip_dirs]

        # Track top-level directories
        rel_root = os.path.relpath(root, '.')
        if rel_root != '.':
            top_dir = rel_root.split(os.sep)[0]
            top_level_dirs.add(top_dir)

        for f in files:
            rel_path = os.path.join(root, f)

            # Extension counting
            ext = os.path.splitext(f)[1].lower() or '(no ext)'
            ext_counts[ext] += 1

            # Directory counting (top-level)
            if rel_root != '.':
                top_dir = rel_root.split(os.sep)[0]
                dir_file_counts[top_dir] += 1
            else:
                dir_file_counts['(root)'] += 1

            # Count lines in Python files
            if ext == '.py':
                try:
                    with open(rel_path, 'r', encoding='utf-8', errors='ignore') as pf:
                        lines = len(pf.readlines())
                        total_lines_py += lines
                        largest_files.append((lines, rel_path))
                except Exception:
                    pass  # Skip unreadable files

    # Sort for output
    sorted_exts = sorted(ext_counts.items(), key=lambda x: -x[1])
    sorted_dirs = sorted(dir_file_counts.items(), key=lambda x: -x[1])
    largest_files.sort(reverse=True)

    total = sum(ext_counts.values())

    # Generate markdown report
    report_lines = [
        '# Repository Census Report',
        '',
        f'**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '**Repository:** kobe81_traderbot',
        '',
        '---',
        '',
        '## Summary Statistics',
        '',
        '| Metric | Count |',
        '|--------|-------|',
        f'| **Total Files** | {total:,} |',
        f'| **Python Files (.py)** | {ext_counts.get(".py", 0):,} |',
        f'| **Config Files (json/yml/yaml/toml/ini)** | {ext_counts.get(".json", 0) + ext_counts.get(".yml", 0) + ext_counts.get(".yaml", 0) + ext_counts.get(".toml", 0) + ext_counts.get(".ini", 0):,} |',
        f'| **Shell Scripts (sh/ps1/bat)** | {ext_counts.get(".sh", 0) + ext_counts.get(".ps1", 0) + ext_counts.get(".bat", 0):,} |',
        f'| **Jupyter Notebooks** | {ext_counts.get(".ipynb", 0):,} |',
        f'| **Total Python Lines** | {total_lines_py:,} |',
        '',
        '---',
        '',
        '## File Types by Extension (Top 25)',
        '',
        '| Extension | Count | Percentage |',
        '|-----------|-------|------------|',
    ]

    for ext, count in sorted_exts[:25]:
        pct = (count / total) * 100
        report_lines.append(f'| {ext} | {count:,} | {pct:.1f}% |')

    report_lines.extend([
        '',
        '---',
        '',
        '## Files by Top-Level Directory',
        '',
        '| Directory | File Count |',
        '|-----------|------------|',
    ])

    for d, count in sorted_dirs[:30]:
        report_lines.append(f'| {d}/ | {count:,} |')

    report_lines.extend([
        '',
        '---',
        '',
        '## Top 20 Largest Python Files (by lines)',
        '',
        '| Lines | File |',
        '|-------|------|',
    ])

    for lines, path in largest_files[:20]:
        report_lines.append(f'| {lines:,} | {path} |')

    report_lines.extend([
        '',
        '---',
        '',
        '## Manifest Files Created',
        '',
        '| File | Description |',
        '|------|-------------|',
        f'| `file_manifest_all.txt` | All {total:,} files in repository |',
        f'| `file_manifest_py.txt` | All {ext_counts.get(".py", 0):,} Python files |',
        '| `file_manifest_configs.txt` | All config files (yml, yaml, json, toml, ini, env) |',
        f'| `file_manifest_scripts.txt` | All {ext_counts.get(".sh", 0) + ext_counts.get(".ps1", 0) + ext_counts.get(".bat", 0):,} shell/batch scripts |',
        f'| `file_manifest_notebooks.txt` | All {ext_counts.get(".ipynb", 0):,} Jupyter notebooks |',
        '',
        '---',
        '',
        '## Directory Structure (Top-Level)',
        '',
        '```',
        'kobe81_traderbot/',
    ])

    for d in sorted(top_level_dirs):
        report_lines.append(f'    {d}/')

    report_lines.extend([
        '```',
        '',
        '---',
        '',
        '## Notes',
        '',
        '- Hidden directories (starting with `.`) are excluded',
        '- `__pycache__`, `node_modules`, `venv`, `.venv`, `env` directories are excluded',
        '- File counts include all non-hidden files in the repository',
    ])

    # Write the report
    with open('AUDITS/00_REPO_CENSUS.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print('00_REPO_CENSUS.md created successfully')
    print()
    print('=== SUMMARY ===')
    print(f'Total files: {total:,}')
    print(f'Python files: {ext_counts.get(".py", 0):,}')
    print(f'Total Python lines: {total_lines_py:,}')
    print(f'Top-level directories: {len(top_level_dirs)}')
    print()
    print('Top 10 extensions:')
    for ext, count in sorted_exts[:10]:
        print(f'  {ext}: {count:,}')

if __name__ == '__main__':
    main()
