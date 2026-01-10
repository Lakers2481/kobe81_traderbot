#!/bin/bash
#
# PROOF SECTION 6: STATISTICAL CORRECTNESS
#
# Proves that Wilson CI, Binomial test, and BH-FDR are mathematically correct
# by running unit tests with reference-checked values.
#

set -e  # Exit on error

echo "=== STATISTICAL CORRECTNESS PROOF ==="
echo ""

echo "Running statistical tests with reference values..."
echo ""

# Run Wilson CI tests
pytest tests/test_optimizer_proof.py::TestWilsonCI -v

echo ""

# Run Binomial test tests
pytest tests/test_optimizer_proof.py::TestBinomialPValue -v

echo ""

# Run BH-FDR tests
pytest tests/test_optimizer_proof.py::TestBenjaminiHochbergFDR -v

echo ""
echo "✅ PROOF F: Statistical methods are mathematically correct"
echo "  - Wilson CI matches reference values from Brown et al. (2001)"
echo "  - Binomial p-values match scipy.stats.binomtest"
echo "  - BH-FDR follows Benjamini & Hochberg (1995) procedure"

exit 0
