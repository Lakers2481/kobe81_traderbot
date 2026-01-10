#!/bin/bash
#
# PROOF SECTION 2: COMPILATION
#
# Proves that all 3 optimizer files compile without syntax errors.
#

set -e  # Exit on error

echo "=== COMPILATION PROOF ==="
echo ""

echo "Compiling Python files..."
echo ""

# Compile all 3 files
python -m py_compile analytics/statistical_testing.py analytics/recovery_analyzer.py scripts/optimize_entry_hold_recovery_universe.py

EXIT_CODE=$?

echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ PROOF B: All 3 files compile successfully (exit code $EXIT_CODE)"
else
    echo "❌ FAILED: Compilation failed with exit code $EXIT_CODE"
    exit 1
fi

exit 0
