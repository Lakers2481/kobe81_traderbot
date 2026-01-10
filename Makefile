# Makefile for Kobe Trading System
#
# Main targets for development and verification
#

.PHONY: help verify-optimizer test lint format clean

# Default target
help:
	@echo "Kobe Trading System - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make verify-optimizer   - Run comprehensive optimizer proof system (ZERO-TRUST)"
	@echo "  make test               - Run all pytest tests"
	@echo "  make lint               - Run code quality checks"
	@echo "  make format             - Format code with black"
	@echo "  make clean              - Remove cache and temp files"
	@echo ""

# =============================================================================
# OPTIMIZER PROOF SYSTEM (ZERO-TRUST)
# =============================================================================

verify-optimizer:
	@echo "================================================================================"
	@echo "OPTIMIZER PROOF SYSTEM - ZERO-TRUST VERIFICATION"
	@echo "================================================================================"
	@echo ""
	@echo "This will run 10 comprehensive proofs to verify the optimizer:"
	@echo "  A) Files Exist"
	@echo "  B) Code Compiles"
	@echo "  C) Smoke Test Runs"
	@echo "  D) Output Structure"
	@echo "  E) No Lookahead"
	@echo "  F) Stats Correct"
	@echo "  G) Recovery Metrics"
	@echo "  H) Walk-Forward Real"
	@echo "  I) Cost Sensitivity"
	@echo "  J) Reproducible"
	@echo ""
	@echo "================================================================================"
	@echo ""
	@python scripts/proof/generate_verdict.py
	@echo ""
	@echo "================================================================================"
	@echo "Verdict saved to: OPTIMIZER_PROOF_VERDICT.md"
	@echo "================================================================================"

# =============================================================================
# TESTING
# =============================================================================

test:
	@echo "Running all tests..."
	@pytest -v

test-proof:
	@echo "Running optimizer proof tests only..."
	@pytest tests/test_optimizer_proof.py -v

# =============================================================================
# CODE QUALITY
# =============================================================================

lint:
	@echo "Running code quality checks..."
	@echo ""
	@echo "[1/3] Running flake8..."
	@flake8 --max-line-length=120 --extend-ignore=E203,W503 . || true
	@echo ""
	@echo "[2/3] Running mypy..."
	@mypy --ignore-missing-imports scripts/ analytics/ || true
	@echo ""
	@echo "[3/3] Running pylint..."
	@pylint --max-line-length=120 scripts/*.py analytics/*.py || true

format:
	@echo "Formatting code with black..."
	@black --line-length 120 scripts/ analytics/ strategies/ tests/

# =============================================================================
# CLEANUP
# =============================================================================

clean:
	@echo "Cleaning cache and temp files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

# =============================================================================
# SMOKE TESTS
# =============================================================================

smoke-test:
	@echo "Running optimizer smoke test (20 symbols)..."
	@python scripts/optimize_entry_hold_recovery_universe.py --smoke

# =============================================================================
# FULL OPTIMIZATION
# =============================================================================

optimize-full:
	@echo "WARNING: This will run full 900-stock optimization (~2-3 hours)"
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@python scripts/optimize_entry_hold_recovery_universe.py
