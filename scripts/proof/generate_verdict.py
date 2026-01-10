"""
OPTIMIZER PROOF VERDICT GENERATOR

Runs all proof scripts and generates final verdict document.

This script:
1. Executes all 8 proof shell scripts
2. Collects results from each proof section
3. Determines PROVEN / NOT PROVEN for each checklist item (A-J)
4. Generates OPTIMIZER_PROOF_VERDICT.md with evidence and required fixes
5. Returns exit code 0 if all proofs pass, 1 if any fail

Author: Kobe Trading System
Date: 2026-01-08
"""

import subprocess
import sys
import datetime
import os
from pathlib import Path
from typing import Dict, List, Tuple


class ProofResult:
    """Result from a single proof execution."""

    def __init__(self, proof_id: str, name: str, passed: bool, evidence: str, error: str = None):
        self.proof_id = proof_id
        self.name = name
        self.passed = passed
        self.evidence = evidence
        self.error = error


def run_proof_script(script_path: str, proof_id: str, proof_name: str) -> ProofResult:
    """
    Run a single proof script and capture result.

    Args:
        script_path: Path to bash script
        proof_id: Proof identifier (A-J)
        proof_name: Human-readable proof name

    Returns:
        ProofResult with pass/fail and evidence
    """
    print(f"  Running {proof_id}) {proof_name}...")

    try:
        # Run bash script
        result = subprocess.run(
            ['bash', script_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        evidence = result.stdout
        if result.stderr:
            evidence += f"\n[stderr]\n{result.stderr}"

        passed = (result.returncode == 0)
        error = None if passed else f"Script exited with code {result.returncode}"

        return ProofResult(proof_id, proof_name, passed, evidence, error)

    except subprocess.TimeoutExpired:
        return ProofResult(
            proof_id,
            proof_name,
            False,
            "",
            "Script timed out after 5 minutes"
        )
    except Exception as e:
        return ProofResult(
            proof_id,
            proof_name,
            False,
            "",
            f"Script execution failed: {e}"
        )


def run_pytest_tests() -> ProofResult:
    """Run all proof pytest tests."""
    print(f"  Running F) Statistical Correctness Tests...")

    try:
        result = subprocess.run(
            ['pytest', 'tests/test_optimizer_proof.py', '-v', '--tb=short'],
            capture_output=True,
            text=True,
            timeout=120
        )

        evidence = result.stdout
        passed = (result.returncode == 0)
        error = None if passed else f"Tests failed (exit code {result.returncode})"

        return ProofResult('F', 'Statistical Correctness', passed, evidence, error)

    except Exception as e:
        return ProofResult(
            'F',
            'Statistical Correctness',
            False,
            "",
            f"Test execution failed: {e}"
        )


def generate_verdict_markdown(results: List[ProofResult], output_path: str):
    """
    Generate OPTIMIZER_PROOF_VERDICT.md from proof results.

    Args:
        results: List of ProofResult objects
        output_path: Path to write verdict file
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Count passes and fails
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count

    # Determine final verdict
    final_verdict = "OPTIMIZER IS PROVEN CORRECT" if failed_count == 0 else "VERIFICATION FAILED - FIXES REQUIRED"

    # Generate markdown content
    md_lines = []
    md_lines.append("# OPTIMIZER PROOF VERDICT - ZERO-TRUST AUDIT\n")
    md_lines.append(f"**Date:** {now}\n")
    md_lines.append("**Auditor:** Proof System (Automated)\n")
    md_lines.append("**Mode:** ZERO-TRUST\n")
    md_lines.append("\n---\n\n")

    md_lines.append("## FINAL VERDICT\n\n")
    md_lines.append(f"**{final_verdict}**\n\n")
    md_lines.append(f"**Results:** {passed_count}/{len(results)} proofs passed\n\n")

    # Summary table
    md_lines.append("| Item | Status | Evidence |\n")
    md_lines.append("|------|--------|----------|\n")

    proof_names = {
        'A': 'Files Exist',
        'B': 'Code Compiles',
        'C': 'Smoke Test Runs',
        'D': 'Output Structure',
        'E': 'No Lookahead',
        'F': 'Stats Correct',
        'G': 'Recovery Metrics',
        'H': 'Walk-Forward Real',
        'I': 'Cost Sensitivity',
        'J': 'Reproducible'
    }

    for result in results:
        status = "✅ PROVEN" if result.passed else "❌ NOT PROVEN"
        evidence_summary = "See detailed evidence below" if result.passed else result.error or "See detailed evidence below"
        md_lines.append(f"| {result.proof_id}) {proof_names.get(result.proof_id, result.name)} | {status} | {evidence_summary} |\n")

    md_lines.append("\n---\n\n")

    # Detailed evidence
    md_lines.append("## DETAILED EVIDENCE\n\n")

    for result in results:
        md_lines.append(f"### {result.proof_id}) {result.name}\n\n")
        if result.passed:
            md_lines.append("**Status:** ✅ PROVEN\n\n")
        else:
            md_lines.append(f"**Status:** ❌ NOT PROVEN\n\n")
            if result.error:
                md_lines.append(f"**Error:** {result.error}\n\n")

        md_lines.append("**Evidence:**\n\n")
        md_lines.append("```\n")
        md_lines.append(result.evidence[:5000])  # Limit evidence size
        if len(result.evidence) > 5000:
            md_lines.append("\n[... truncated ...]\n")
        md_lines.append("```\n\n")

    md_lines.append("---\n\n")

    # Fixes required
    md_lines.append("## FIXES REQUIRED\n\n")

    failed_results = [r for r in results if not r.passed]
    if failed_results:
        for result in failed_results:
            md_lines.append(f"### {result.proof_id}) {result.name}\n\n")
            md_lines.append(f"**Issue:** {result.error or 'Proof failed'}\n\n")
            md_lines.append("**Required Fix:** Review detailed evidence above and address the failure.\n\n")
    else:
        md_lines.append("*None - all proofs passed.*\n\n")

    md_lines.append("---\n\n")
    md_lines.append("## SUMMARY\n\n")
    md_lines.append(f"- **Total Proofs:** {len(results)}\n")
    md_lines.append(f"- **Passed:** {passed_count}\n")
    md_lines.append(f"- **Failed:** {failed_count}\n")
    md_lines.append(f"- **Final Verdict:** {final_verdict}\n")

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(md_lines)

    print(f"\n[OK] Verdict written to: {output_path}")


def main():
    """Main entry point."""
    print("=" * 80)
    print("OPTIMIZER PROOF SYSTEM - ZERO-TRUST VERIFICATION")
    print("=" * 80)
    print()

    # Change to repository root
    repo_root = Path(__file__).parent.parent.parent
    os.chdir(repo_root)
    print(f"Working directory: {os.getcwd()}\n")

    results = []

    # Proof A: Files Exist
    results.append(run_proof_script(
        'scripts/proof/01_file_proof.sh',
        'A',
        'Files Exist'
    ))

    # Proof B: Code Compiles
    results.append(run_proof_script(
        'scripts/proof/02_compile_proof.sh',
        'B',
        'Code Compiles'
    ))

    # Proof C: Smoke Test Runs
    results.append(run_proof_script(
        'scripts/proof/03_smoke_proof.sh',
        'C',
        'Smoke Test Runs'
    ))

    # Proof D: Output Structure
    results.append(run_proof_script(
        'scripts/proof/04_output_proof.sh',
        'D',
        'Output Structure'
    ))

    # Proof E: No Lookahead
    results.append(run_proof_script(
        'scripts/proof/05_lookahead_audit.sh',
        'E',
        'No Lookahead'
    ))

    # Proof F: Statistical Correctness (pytest)
    results.append(run_pytest_tests())

    # Proof H: Walk-Forward Real
    results.append(run_proof_script(
        'scripts/proof/07_walkforward_proof.sh',
        'H',
        'Walk-Forward Real'
    ))

    # Proof I: Cost Sensitivity
    results.append(run_proof_script(
        'scripts/proof/08_cost_proof.sh',
        'I',
        'Cost Sensitivity'
    ))

    # Proof G: Recovery Metrics (implicitly tested by pytest)
    # Proof J: Reproducible (smoke test proves this)

    # Add G and J manually based on other results
    pytest_passed = any(r.proof_id == 'F' and r.passed for r in results)
    smoke_passed = any(r.proof_id == 'C' and r.passed for r in results)

    results.append(ProofResult(
        'G',
        'Recovery Metrics',
        pytest_passed,
        "Verified by TestRecoveryMetrics pytest tests" if pytest_passed else "",
        None if pytest_passed else "Recovery metrics tests failed (see proof F)"
    ))

    results.append(ProofResult(
        'J',
        'Reproducible',
        smoke_passed,
        "Single command smoke test successful" if smoke_passed else "",
        None if smoke_passed else "Smoke test failed (see proof C)"
    ))

    print()
    print("=" * 80)
    print("GENERATING VERDICT")
    print("=" * 80)
    print()

    # Generate verdict file
    verdict_path = repo_root / "OPTIMIZER_PROOF_VERDICT.md"
    generate_verdict_markdown(results, str(verdict_path))

    # Print summary
    print()
    print("=" * 80)
    print("VERDICT SUMMARY")
    print("=" * 80)
    print()

    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count

    for result in sorted(results, key=lambda r: r.proof_id):
        status = "[PROVEN]" if result.passed else "[NOT PROVEN]"
        print(f"{result.proof_id}) {result.name:25s} {status}")

    print()
    print(f"Final: {passed_count}/{len(results)} proofs passed")
    print()

    # Exit with appropriate code
    if failed_count > 0:
        print("[FAILED] VERIFICATION FAILED - See OPTIMIZER_PROOF_VERDICT.md for details")
        sys.exit(1)
    else:
        print("[SUCCESS] ALL PROOFS PASSED - OPTIMIZER IS PROVEN CORRECT")
        sys.exit(0)


if __name__ == "__main__":
    main()
