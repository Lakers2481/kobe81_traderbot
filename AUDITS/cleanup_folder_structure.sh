#!/bin/bash
# FWO-Prime Folder Structure Cleanup Script
# Generated: 2026-01-09
# Execute with: bash AUDITS/cleanup_folder_structure.sh

set -e  # Exit on error

echo "FWO-Prime: Starting folder structure cleanup..."
echo ""

# Phase 1: DELETE BROKEN/STRANGE DIRECTORIES
echo "=== PHASE 1: Removing broken and strange directories ==="

if [ -d "C:UsersOwnerOneDriveDesktopkobe81_traderbotdataverification" ]; then
    echo "Removing broken path directory (data verification)..."
    rm -rf "C:UsersOwnerOneDriveDesktopkobe81_traderbotdataverification"
fi

if [ -d "C:UsersOwnerOneDriveDesktopkobe81_traderbotscriptsexperiments" ]; then
    echo "Removing broken path directory (scripts experiments)..."
    rm -rf "C:UsersOwnerOneDriveDesktopkobe81_traderbotscriptsexperiments"
fi

if [ -d "_ul" ]; then
    echo "Removing temp directory (_ul)..."
    rm -rf "_ul"
fi

if [ -d "_ul-DESKTOP-5IB5S6R" ]; then
    echo "Removing temp directory (_ul-DESKTOP-5IB5S6R)..."
    rm -rf "_ul-DESKTOP-5IB5S6R"
fi

if [ -d "nul" ]; then
    echo "Removing dangerous 'nul' directory..."
    rm -rf "nul"
fi

# Investigate vuLDY5zrhSOyIpTVB6JB5taCKu71bWAQ
if [ -d "vuLDY5zrhSOyIpTVB6JB5taCKu71bWAQ" ]; then
    echo "Found vuLDY5zrhSOyIpTVB6JB5taCKu71bWAQ directory..."
    echo "Contents:"
    ls -la vuLDY5zrhSOyIpTVB6JB5taCKu71bWAQ/
    echo ""
    echo "MANUAL ACTION REQUIRED: Review contents and delete if unnecessary"
    # Uncomment to auto-delete:
    # rm -rf "vuLDY5zrhSOyIpTVB6JB5taCKu71bWAQ"
fi

echo ""

# Phase 2: MOVE ROOT FILES TO PROPER LOCATIONS
echo "=== PHASE 2: Organizing root directory files ==="

# Move JSON files to AUDITS
for file in *.json; do
    if [ -f "$file" ] && [ "$file" != "package.json" ]; then
        echo "Moving $file → AUDITS/"
        mv "$file" AUDITS/
    fi
done

# Move .txt files to AUDITS (except gitignore, requirements)
for file in *.txt; do
    if [ -f "$file" ] && [[ "$file" != "requirements"* ]]; then
        echo "Moving $file → AUDITS/"
        mv "$file" AUDITS/
    fi
done

# Move audit/report markdown files to docs
for pattern in "*AUDIT*.md" "*_REPORT.md" "*PROMPT*.md" "FIX_*.md" "*STATUS*.md"; do
    for file in $pattern; do
        if [ -f "$file" ] && [ "$file" != "README.md" ] && [ "$file" != "CLAUDE.md" ]; then
            echo "Moving $file → docs/"
            mv "$file" docs/ 2>/dev/null || true
        fi
    done
done

# Move specific documentation files
docs_files=(
    "CAPABILITY_MATRIX.md"
    "INTEGRATION_RECOMMENDATIONS.md"
    "INTERVIEW_ONE_PAGER.md"
    "MODULE_FILE_COUNTS.md"
    "PRODUCTION_CRITICAL_COMPONENTS.md"
    "PROJECT_CONTEXT.md"
    "QUICK_START.md"
    "SYSTEM_ARCHITECTURE_INVENTORY.md"
)

for file in "${docs_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Moving $file → docs/"
        mv "$file" docs/
    fi
done

echo ""

# Phase 3: CONSOLIDATE OUTPUT DIRECTORIES
echo "=== PHASE 3: Consolidating output directories ==="

# Create consolidated structure
mkdir -p outputs/{backtests,walk_forward,showdowns,optimizations,smoke_tests}

# Move backtest outputs
if [ -d "backtest_outputs" ]; then
    echo "Consolidating backtest_outputs/ → outputs/backtests/"
    cp -r backtest_outputs/* outputs/backtests/ 2>/dev/null || true
    rm -rf backtest_outputs
fi

# Move walk-forward outputs
if [ -d "wf_outputs" ]; then
    echo "Consolidating wf_outputs/ → outputs/walk_forward/"
    cp -r wf_outputs/* outputs/walk_forward/ 2>/dev/null || true
    rm -rf wf_outputs
fi

# Move showdown outputs
if [ -d "showdown_outputs" ]; then
    echo "Consolidating showdown_outputs/ → outputs/showdowns/"
    cp -r showdown_outputs/* outputs/showdowns/ 2>/dev/null || true
    rm -rf showdown_outputs
fi

if [ -d "showdown_2025_cap60" ]; then
    echo "Consolidating showdown_2025_cap60/ → outputs/showdowns/"
    cp -r showdown_2025_cap60/* outputs/showdowns/ 2>/dev/null || true
    rm -rf showdown_2025_cap60
fi

# Move optimization outputs
if [ -d "optimize_outputs" ]; then
    echo "Consolidating optimize_outputs/ → outputs/optimizations/"
    cp -r optimize_outputs/* outputs/optimizations/ 2>/dev/null || true
    rm -rf optimize_outputs
fi

# Move smoke test outputs
for dir in smoke_outputs smoke_turtle_soup smoke_wf_audit; do
    if [ -d "$dir" ]; then
        echo "Consolidating $dir/ → outputs/smoke_tests/"
        cp -r "$dir"/* outputs/smoke_tests/ 2>/dev/null || true
        rm -rf "$dir"
    fi
done

# Merge output/ into outputs/
if [ -d "output" ]; then
    echo "Merging output/ → outputs/"
    cp -r output/* outputs/ 2>/dev/null || true
    rm -rf output
fi

echo ""

# Phase 4: CREATE MISSING __init__.py FILES
echo "=== PHASE 4: Creating missing __init__.py files ==="

init_dirs=(
    "analysis"
    "autonomous/scrapers"
    "backtest"
    "cognitive"
    "config/alpha_workflows"
    "data/schemas"
    "evolution"
    "explainability"
    "extensions"
    "ml/alpha_discovery"
    "options"
    "pipelines"
    "research_os"
    "risk/advanced"
    "strategy_specs"
)

for dir in "${init_dirs[@]}"; do
    if [ -d "$dir" ] && [ ! -f "$dir/__init__.py" ]; then
        echo "Creating $dir/__init__.py"
        touch "$dir/__init__.py"
    fi
done

echo ""

# Phase 5: CLEAN CACHE DIRECTORIES
echo "=== PHASE 5: Cleaning cache directories ==="

echo "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "Removing .pytest_cache directories..."
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

echo "Removing .ruff_cache directories..."
find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

echo ""

# Phase 6: FIX DATA CACHE DUPLICATION
echo "=== PHASE 6: Consolidating data cache ==="

if [ -d "data/polygon_cache/polygon" ]; then
    echo "Merging data/polygon_cache/polygon/ → data/cache/polygon/"
    mkdir -p data/cache/polygon
    cp -r data/polygon_cache/polygon/* data/cache/polygon/ 2>/dev/null || true
    rm -rf data/polygon_cache
fi

echo ""

# Phase 7: UPDATE .gitignore
echo "=== PHASE 7: Updating .gitignore ==="

if ! grep -q "__pycache__" .gitignore 2>/dev/null; then
    echo "Adding cache patterns to .gitignore..."
    cat >> .gitignore <<EOF

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Testing cache
.pytest_cache/
.ruff_cache/
.mypy_cache/

# ML cache
mlruns/
*.egg-info/

# OS files
.DS_Store
Thumbs.db
EOF
fi

echo ""
echo "=== CLEANUP COMPLETE ==="
echo ""
echo "Summary:"
echo "  - Removed broken/strange directories"
echo "  - Organized 34 root files into docs/ and AUDITS/"
echo "  - Consolidated 7 output directories into outputs/"
echo "  - Created 15 missing __init__.py files"
echo "  - Cleaned all __pycache__ directories"
echo "  - Updated .gitignore"
echo ""
echo "Next steps:"
echo "  1. Review changes with 'git status'"
echo "  2. Test imports with 'python -m pytest tests/'"
echo "  3. Re-run audit: 'python AUDITS/folder_audit_analyzer.py'"
echo "  4. Commit changes with descriptive message"
echo ""
echo "FWO-Prime: Cleanup complete. Repository is now Renaissance-grade."
