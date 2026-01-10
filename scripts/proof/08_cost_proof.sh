#!/bin/bash
#
# PROOF SECTION 8: COST SENSITIVITY
#
# Proves that returns decrease monotonically as transaction costs increase.
#

set -e  # Exit on error

echo "=== COST SENSITIVITY PROOF ==="
echo ""

if [ ! -f "output/cost_sensitivity_comparison.csv" ]; then
    echo "❌ FAILED: output/cost_sensitivity_comparison.csv does not exist"
    echo "Run smoke test first: python scripts/optimize_entry_hold_recovery_universe.py --smoke"
    exit 1
fi

echo "Inspecting cost sensitivity results..."
echo ""

# Use Python to verify monotonic decrease
python << 'EOF'
import pandas as pd
import sys

try:
    df = pd.read_csv('output/cost_sensitivity_comparison.csv')

    if 'cost_bps' not in df.columns or 'mean_return' not in df.columns:
        print("[FAILED] Missing required columns (cost_bps, mean_return)")
        sys.exit(1)

    # Group by cost scenario
    cost_summary = df.groupby('cost_bps')['mean_return'].mean().sort_index()

    print("Mean return by cost scenario:")
    for cost_bps, mean_ret in cost_summary.items():
        print(f"  {cost_bps:3d} bps: {mean_ret:+.4f} ({mean_ret*100:+.2f}%)")

    print()

    # Verify monotonic decrease
    cost_levels = sorted(cost_summary.index)
    monotonic = True

    for i in range(len(cost_levels) - 1):
        curr_cost = cost_levels[i]
        next_cost = cost_levels[i + 1]
        curr_ret = cost_summary[curr_cost]
        next_ret = cost_summary[next_cost]

        if curr_ret <= next_ret:
            print(f"[FAILED] Returns did not decrease from {curr_cost} to {next_cost} bps")
            print(f"  {curr_cost} bps: {curr_ret:.4f}")
            print(f"  {next_cost} bps: {next_ret:.4f}")
            monotonic = False

    if monotonic:
        print("[OK] PROOF I: Returns decrease monotonically as costs increase")
        print("  Verified: mean_return(0) > mean_return(5) > mean_return(10)")
    else:
        sys.exit(1)

except Exception as e:
    print(f"[FAILED] {e}")
    sys.exit(1)
EOF

exit 0
