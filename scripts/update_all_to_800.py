#!/usr/bin/env python3
"""
Update All References from 900 to 800 Stocks
Comprehensive system-wide update to ensure NO confusion

Usage:
    python scripts/update_all_to_800.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def update_file(filepath: Path, dry_run=False):
    """Update a single file's references from 900 to 800."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Count occurrences
        count_900_csv = content.count('optionable_liquid_800.csv')
        count_900_stocks = content.count('800 stocks')
        count_900_symbols = content.count('800 symbols')
        count_900_arrow = content.count('800 →')

        total = count_900_csv + count_900_stocks + count_900_symbols + count_900_arrow

        if total == 0:
            return None  # No changes needed

        if dry_run:
            return {
                'file': str(filepath.relative_to(ROOT)),
                '900.csv': count_900_csv,
                '800 stocks': count_900_stocks,
                '800 symbols': count_900_symbols,
                '800 →': count_900_arrow,
                'total': total
            }

        # Make replacements
        updated = content
        updated = updated.replace('optionable_liquid_800.csv', 'optionable_liquid_800.csv')
        updated = updated.replace('800 stocks', '800 stocks')
        updated = updated.replace('800 symbols', '800 symbols')
        updated = updated.replace('800 →', '800 →')

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(updated)

        return {
            'file': str(filepath.relative_to(ROOT)),
            'total': total,
            'status': 'UPDATED'
        }

    except Exception as e:
        return {
            'file': str(filepath.relative_to(ROOT)),
            'error': str(e),
            'status': 'ERROR'
        }


def main():
    print('[UPDATE ALL REFERENCES: 900 to 800]')
    print()

    # File extensions to check
    extensions = ['.py', '.yaml', '.yml', '.md', '.txt', '.json', '.sh', '.ps1', '.bat']

    # Directories to exclude
    exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv', '.claude'}

    # Find all relevant files
    all_files = []
    for ext in extensions:
        for filepath in ROOT.rglob(f'*{ext}'):
            # Skip excluded directories
            if any(ex in filepath.parts for ex in exclude_dirs):
                continue
            all_files.append(filepath)

    print(f'Found {len(all_files)} files to check')
    print()

    # Dry run first
    print('[DRY RUN] Checking which files need updates...')
    files_to_update = []
    for filepath in all_files:
        result = update_file(filepath, dry_run=True)
        if result and result.get('total', 0) > 0:
            files_to_update.append(result)

    if not files_to_update:
        print('[SUCCESS] All files already updated to 800!')
        return 0

    print(f'\\nFiles needing update: {len(files_to_update)}')
    print()

    # Show top 20
    files_to_update_sorted = sorted(files_to_update, key=lambda x: x['total'], reverse=True)
    print('Top 20 files with most references:')
    for i, item in enumerate(files_to_update_sorted[:20], 1):
        print(f"  {i:2d}. {item['file']:60s} {item['total']:3d} refs")

    print()
    print(f'Total files to update: {len(files_to_update)}')
    print()

    # Auto-proceed (no confusion allowed!)
    print('[AUTO-PROCEEDING] Updating all files to ensure NO confusion...')
    print()

    # Perform actual updates
    print()
    print('[UPDATING FILES...]')
    updated_count = 0
    error_count = 0

    for filepath_str in [item['file'] for item in files_to_update]:
        filepath = ROOT / filepath_str
        result = update_file(filepath, dry_run=False)

        if result and result.get('status') == 'UPDATED':
            updated_count += 1
            print(f"  [OK] {result['file']}")
        elif result and result.get('status') == 'ERROR':
            error_count += 1
            print(f"  [ERROR] {result['file']}: {result.get('error')}")

    print()
    print('=' * 70)
    print('UPDATE COMPLETE')
    print('=' * 70)
    print(f'Updated: {updated_count} files')
    print(f'Errors: {error_count} files')
    print()

    if error_count == 0:
        print('[SUCCESS] All references updated from 900 to 800!')
    else:
        print(f'[WARNING] {error_count} files had errors')

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
