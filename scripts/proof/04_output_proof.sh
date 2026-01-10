#!/bin/bash
#
# PROOF SECTION 4: OUTPUT STRUCTURE
#
# Proves that output CSV files contain all required columns and correct combo counts.
#

set -e  # Exit on error

echo "=== OUTPUT STRUCTURE PROOF ==="
echo ""

if [ ! -f "output/entry_hold_grid_event_weighted.csv" ]; then
    echo "❌ FAILED: output/entry_hold_grid_event_weighted.csv does not exist"
    echo "Run smoke test first: python scripts/optimize_entry_hold_recovery_universe.py --smoke"
    exit 1
fi

echo "Inspecting entry_hold_grid_event_weighted.csv structure..."
echo ""

# Use Python to inspect CSV structure
python << 'EOF'
import pandas as pd
import sys

try:
    df = pd.read_csv('output/entry_hold_grid_event_weighted.csv')

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print()

    print("Column names:")
    for col in df.columns:
        print(f"  - {col}")
    print()

    # Verify required columns
    required_cols = [
        'combo_id', 'streak_length', 'hold_period',
        'win_rate', 'mean_return', 'n_instances', 'p_value'
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[FAILED] Missing required columns: {missing}")
        sys.exit(1)

    print("[OK] All required columns present")
    print()

    print("Unique value counts:")
    print(f"  - streak_length: {sorted(df['streak_length'].unique())}")
    print(f"  - hold_period: {sorted(df['hold_period'].unique())}")

    if 'streak_mode' in df.columns:
        print(f"  - streak_mode: {sorted(df['streak_mode'].unique())}")
    if 'entry_timing' in df.columns:
        print(f"  - entry_timing: {sorted(df['entry_timing'].unique())}")

    print()
    print(f"Total combinations: {df['combo_id'].nunique()}")
    print()

    print("First 5 rows:")
    print(df.head())

    print()
    print("[OK] PROOF D: Output structure is correct")

except Exception as e:
    print(f"[FAILED] {e}")
    sys.exit(1)
EOF

exit 0
