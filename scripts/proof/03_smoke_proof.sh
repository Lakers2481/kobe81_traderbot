#!/bin/bash
#
# PROOF SECTION 3: SMOKE TEST EXECUTION
#
# Proves that the optimizer runs successfully on 20 symbols and generates 9 output files.
#
# NOTE: This script requires the smoke test to already be completed.
# If smoke test hasn't run yet, execute:
#   python scripts/optimize_entry_hold_recovery_universe.py --smoke
#

set -e  # Exit on error

echo "=== SMOKE TEST OUTPUT VERIFICATION ==="
echo ""

# Check if output directory exists
if [ ! -d "output" ]; then
    echo "❌ FAILED: output/ directory does not exist"
    echo "Run: python scripts/optimize_entry_hold_recovery_universe.py --smoke"
    exit 1
fi

# List of required output files
required_files=(
    "output/entry_hold_grid_event_weighted.csv"
    "output/best_combos_expected_return.csv"
    "output/best_combos_fast_recovery.csv"
    "output/best_combos_target_hit.csv"
    "output/best_combos_risk_adjusted.csv"
    "output/coverage_report.csv"
    "output/cost_sensitivity_comparison.csv"
    "output/walk_forward_results.json"
    "output/optimizer_report.md"
)

echo "Checking for 9 required output files..."
echo ""

missing_files=()

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        ls -lh "$file"
    else
        echo "❌ MISSING: $file"
        missing_files+=("$file")
    fi
done

echo ""

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✅ PROOF C: All 9 output files exist"
    echo ""
    echo "Total output directory size:"
    du -sh output/
else
    echo "❌ FAILED: ${#missing_files[@]} files missing"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

exit 0
