#!/usr/bin/env python3
"""Find directories with Python files but missing __init__.py"""
import os

def find_missing_init(root_dir='.'):
    missing = []
    exclude_patterns = ['.git', '__pycache__', 'wf_outputs', 'smoke_', 'showdown_', 'optimize_outputs']

    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        if any(pattern in root for pattern in exclude_patterns):
            continue

        # Check if directory has .py files
        has_python = any(f.endswith('.py') for f in files)
        has_init = '__init__.py' in files

        if has_python and not has_init:
            missing.append(root)

    return sorted(missing)

if __name__ == '__main__':
    missing = find_missing_init()
    if missing:
        print("Directories with Python files but no __init__.py:")
        for d in missing:
            print(f"  {d}")
    else:
        print("All directories with Python files have __init__.py")
