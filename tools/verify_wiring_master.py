"""
MASTER WIRING VERIFIER - Evidence-Required Verification (STRICT MODE)

This is the FINAL verification tool that synthesizes all audit evidence
and produces a definitive PASS/FAIL verdict.

RULES:
1. NO HAND-WAVING - Every claim must have evidence
2. EVIDENCE SOURCES:
   - AUDITS/00_REPO_CENSUS.md (file counts, structure)
   - AUDITS/01_ENTRYPOINTS.json (193 entrypoints)
   - AUDITS/02_COMPONENT_INVENTORY.json (1,440 classes, 7,401 functions)
   - AUDITS/TRUTH_TABLE.csv (component verification status)
   - AUDITS/TRACES/*.jsonl (dynamic execution proof)
   - tests/security/test_live_bypass.py results

3. CRITICAL PATH VERIFICATION:
   - Data flow: Polygon -> Strategy -> Backtest -> Results
   - Order flow: Signal -> Risk Gate -> Execution Choke -> Broker
   - Safety flow: Kill Switch -> Policy Gate -> Safety Gate -> Order

4. STRICT FAILURE CONDITIONS (ANY of these = FAIL):
   - SEV-0 bypass paths exist (direct order submission without choke)
   - Trace event count < MIN_TRACE_EVENTS (50)
   - Critical evidence failures

Author: Kobe Trading System
Version: 2.0.0 (STRICT)
Date: 2026-01-06
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# STRICT MODE THRESHOLDS
MIN_TRACE_EVENTS = 50  # Real pipelines generate 50+ events
MIN_TRACED_FUNCTIONS = 5  # Need multiple functions traced

# Known SEV-0 bypass paths to check
# Format: (file_path, line_pattern, description, severity)
KNOWN_BYPASS_PATTERNS = [
    (
        "scripts/position_manager.py",
        'alpaca_request("/v2/orders"',
        "Direct Alpaca order submission bypassing execution choke",
        "SEV-0"
    ),
    (
        "execution/broker_crypto.py",
        "self._exchange.create_order(",
        "Direct CCXT order submission bypassing execution choke",
        "SEV-0"
    ),
    (
        "options/order_router.py",
        "_check_kill_switch()",
        "Only checks kill switch, not full 7-flag safety gate",
        "SEV-1"
    ),
]


@dataclass
class BypassIssue:
    """A detected bypass vulnerability."""
    file_path: str
    line_number: int
    pattern: str
    description: str
    severity: str  # SEV-0, SEV-1


@dataclass
class EvidenceItem:
    """Single piece of evidence."""
    source: str
    claim: str
    evidence: str
    verified: bool
    severity: str = "INFO"  # INFO, WARNING, CRITICAL


@dataclass
class VerificationResult:
    """Result of a verification check."""
    name: str
    passed: bool
    evidence: List[EvidenceItem] = field(default_factory=list)
    notes: str = ""


@dataclass
class WiringReport:
    """Complete wiring verification report."""
    timestamp: str
    verdict: str  # PASS, FAIL, PARTIAL
    grade: str  # A+, A, B, C, D, F
    score: int  # 0-100
    checks: List[VerificationResult] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def load_json_safe(path: Path) -> Optional[Dict]:
    """Load JSON file safely."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def load_csv_rows(path: Path) -> List[Dict]:
    """Load CSV as list of dicts."""
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def count_trace_events(traces_dir: Path) -> Dict[str, int]:
    """Count events in trace files."""
    counts = {"files": 0, "events": 0, "functions": set()}

    if not traces_dir.exists():
        return {"files": 0, "events": 0, "functions": 0}

    for trace_file in traces_dir.glob("*.jsonl"):
        counts["files"] += 1
        with open(trace_file) as f:
            for line in f:
                if line.strip():
                    counts["events"] += 1
                    try:
                        event = json.loads(line)
                        if "function" in event:
                            counts["functions"].add(event["function"])
                    except json.JSONDecodeError:
                        pass

    counts["functions"] = len(counts["functions"])
    return counts


def detect_bypass_issues() -> List[BypassIssue]:
    """
    Scan for SEV-0/SEV-1 bypass vulnerabilities.

    These are paths that submit orders without going through the
    execution choke point (safety/execution_choke.py).
    """
    issues = []

    for file_rel_path, pattern, description, severity in KNOWN_BYPASS_PATTERNS:
        file_path = ROOT / file_rel_path

        if not file_path.exists():
            # File doesn't exist, skip (might be optional like crypto)
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            for line_num, line in enumerate(lines, start=1):
                if pattern in line:
                    # Check if line is commented out
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue  # Skip comments

                    # Check if there's a @require_execution_choke decorator
                    # by looking back a few lines
                    has_decorator = False
                    for back in range(1, min(10, line_num)):
                        check_line = lines[line_num - back - 1].strip()
                        if "@require_execution_choke" in check_line:
                            has_decorator = True
                            break
                        if check_line.startswith("def ") or check_line.startswith("class "):
                            # Reached function/class def without finding decorator
                            break

                    # Check if evaluate_safety_gates is called nearby
                    # Use larger window (40 lines) to catch safety checks in same function
                    context_start = max(0, line_num - 40)
                    context_end = min(len(lines), line_num + 5)
                    context = "\n".join(lines[context_start:context_end])
                    has_safety_check = "evaluate_safety_gates" in context

                    if not has_decorator and not has_safety_check:
                        issues.append(BypassIssue(
                            file_path=file_rel_path,
                            line_number=line_num,
                            pattern=pattern,
                            description=description,
                            severity=severity,
                        ))
                        break  # Only report once per file
        except Exception as e:
            print(f"Warning: Could not scan {file_rel_path}: {e}")

    return issues


def verify_no_bypass_issues() -> VerificationResult:
    """Verify no SEV-0/SEV-1 bypass vulnerabilities exist."""
    evidence = []
    issues = detect_bypass_issues()

    sev_0_count = sum(1 for i in issues if i.severity == "SEV-0")
    sev_1_count = sum(1 for i in issues if i.severity == "SEV-1")

    if sev_0_count > 0:
        for issue in issues:
            if issue.severity == "SEV-0":
                evidence.append(EvidenceItem(
                    source=f"{issue.file_path}:{issue.line_number}",
                    claim=f"No {issue.severity} bypass",
                    evidence=f"BYPASS FOUND: {issue.description}",
                    verified=False,
                    severity="CRITICAL"
                ))
    else:
        evidence.append(EvidenceItem(
            source="Bypass Scanner",
            claim="No SEV-0 bypass paths",
            evidence=f"Scanned {len(KNOWN_BYPASS_PATTERNS)} known patterns, 0 SEV-0 found",
            verified=True
        ))

    if sev_1_count > 0:
        for issue in issues:
            if issue.severity == "SEV-1":
                evidence.append(EvidenceItem(
                    source=f"{issue.file_path}:{issue.line_number}",
                    claim=f"No {issue.severity} bypass",
                    evidence=f"BYPASS FOUND: {issue.description}",
                    verified=False,
                    severity="WARNING"
                ))
    else:
        evidence.append(EvidenceItem(
            source="Bypass Scanner",
            claim="No SEV-1 bypass paths",
            evidence=f"Scanned {len(KNOWN_BYPASS_PATTERNS)} known patterns, 0 SEV-1 found",
            verified=True
        ))

    passed = sev_0_count == 0  # Only SEV-0 causes failure
    return VerificationResult(
        name="Bypass Vulnerability Scan",
        passed=passed,
        evidence=evidence,
        notes=f"SEV-0: {sev_0_count}, SEV-1: {sev_1_count}"
    )


def verify_repo_census() -> VerificationResult:
    """Verify repo census was completed."""
    evidence = []

    census_path = ROOT / "AUDITS" / "00_REPO_CENSUS.md"
    if census_path.exists():
        content = census_path.read_text()
        evidence.append(EvidenceItem(
            source="AUDITS/00_REPO_CENSUS.md",
            claim="Repo census completed",
            evidence=f"File exists, {len(content)} bytes",
            verified=True
        ))

        # Check for key statistics
        if "Python files:" in content:
            evidence.append(EvidenceItem(
                source="AUDITS/00_REPO_CENSUS.md",
                claim="Python file count documented",
                evidence="Contains Python file statistics",
                verified=True
            ))
    else:
        evidence.append(EvidenceItem(
            source="AUDITS/00_REPO_CENSUS.md",
            claim="Repo census completed",
            evidence="FILE NOT FOUND",
            verified=False,
            severity="CRITICAL"
        ))

    passed = all(e.verified for e in evidence)
    return VerificationResult(
        name="Repo Census",
        passed=passed,
        evidence=evidence,
        notes="Comprehensive file inventory"
    )


def verify_entrypoints() -> VerificationResult:
    """Verify entrypoints discovery."""
    evidence = []

    entrypoints = load_json_safe(ROOT / "AUDITS" / "01_ENTRYPOINTS.json")
    if entrypoints:
        count = entrypoints.get("audit_metadata", {}).get("total_entrypoints", 0)
        evidence.append(EvidenceItem(
            source="AUDITS/01_ENTRYPOINTS.json",
            claim="Entrypoints discovered",
            evidence=f"{count} entrypoints documented",
            verified=count >= 100  # Should have at least 100
        ))

        # Check for critical entrypoints
        critical = entrypoints.get("critical_entrypoints", [])
        evidence.append(EvidenceItem(
            source="AUDITS/01_ENTRYPOINTS.json",
            claim="Critical entrypoints identified",
            evidence=f"{len(critical)} critical scripts marked",
            verified=len(critical) >= 5
        ))
    else:
        evidence.append(EvidenceItem(
            source="AUDITS/01_ENTRYPOINTS.json",
            claim="Entrypoints discovered",
            evidence="FILE NOT FOUND",
            verified=False,
            severity="CRITICAL"
        ))

    passed = all(e.verified for e in evidence)
    return VerificationResult(
        name="Entrypoints Discovery",
        passed=passed,
        evidence=evidence
    )


def verify_component_inventory() -> VerificationResult:
    """Verify component auditor ran."""
    evidence = []

    inventory = load_json_safe(ROOT / "AUDITS" / "02_COMPONENT_INVENTORY.json")
    if inventory:
        stats = inventory.get("statistics", {})
        classes = stats.get("total_classes", 0)
        functions = stats.get("total_functions", 0)
        stubs = stats.get("total_stubs", 0)

        evidence.append(EvidenceItem(
            source="AUDITS/02_COMPONENT_INVENTORY.json",
            claim="Classes discovered",
            evidence=f"{classes} classes found",
            verified=classes >= 100
        ))

        evidence.append(EvidenceItem(
            source="AUDITS/02_COMPONENT_INVENTORY.json",
            claim="Functions discovered",
            evidence=f"{functions} functions found",
            verified=functions >= 1000
        ))

        evidence.append(EvidenceItem(
            source="AUDITS/02_COMPONENT_INVENTORY.json",
            claim="Stubs identified",
            evidence=f"{stubs} stub implementations found",
            verified=True  # Any number is fine
        ))
    else:
        evidence.append(EvidenceItem(
            source="AUDITS/02_COMPONENT_INVENTORY.json",
            claim="Component inventory exists",
            evidence="FILE NOT FOUND",
            verified=False,
            severity="CRITICAL"
        ))

    passed = all(e.verified for e in evidence)
    return VerificationResult(
        name="Component Inventory",
        passed=passed,
        evidence=evidence
    )


def verify_truth_table() -> VerificationResult:
    """Verify truth table was generated."""
    evidence = []

    truth_table_path = ROOT / "AUDITS" / "TRUTH_TABLE.csv"
    rows = load_csv_rows(truth_table_path)

    if rows:
        # Count by status
        status_counts = {}
        for row in rows:
            status = row.get("status", "UNKNOWN")
            status_counts[status] = status_counts.get(status, 0) + 1

        total = len(rows)
        real = status_counts.get("REAL", 0)
        partial = status_counts.get("PARTIAL", 0)
        stub = status_counts.get("STUB", 0)

        real_pct = (real / total * 100) if total > 0 else 0

        evidence.append(EvidenceItem(
            source="AUDITS/TRUTH_TABLE.csv",
            claim="Components classified",
            evidence=f"{total} components: {real} REAL, {partial} PARTIAL, {stub} STUB",
            verified=total >= 1000
        ))

        evidence.append(EvidenceItem(
            source="AUDITS/TRUTH_TABLE.csv",
            claim="High REAL percentage",
            evidence=f"{real_pct:.1f}% components are REAL",
            verified=real_pct >= 70
        ))

        evidence.append(EvidenceItem(
            source="AUDITS/TRUTH_TABLE.csv",
            claim="Low STUB percentage",
            evidence=f"{(stub/total*100):.1f}% are STUB",
            verified=(stub / total * 100) < 5 if total > 0 else True
        ))
    else:
        evidence.append(EvidenceItem(
            source="AUDITS/TRUTH_TABLE.csv",
            claim="Truth table exists",
            evidence="FILE NOT FOUND OR EMPTY",
            verified=False,
            severity="CRITICAL"
        ))

    passed = all(e.verified for e in evidence)
    return VerificationResult(
        name="Truth Table",
        passed=passed,
        evidence=evidence
    )


def verify_runtime_traces() -> VerificationResult:
    """Verify runtime traces exist with STRICT thresholds."""
    evidence = []

    traces_dir = ROOT / "AUDITS" / "TRACES"
    trace_stats = count_trace_events(traces_dir)

    if trace_stats["files"] > 0:
        # STRICT: Check minimum event count
        events_ok = trace_stats["events"] >= MIN_TRACE_EVENTS
        evidence.append(EvidenceItem(
            source="AUDITS/TRACES/*.jsonl",
            claim=f"Trace events >= {MIN_TRACE_EVENTS}",
            evidence=f"{trace_stats['events']} events (need {MIN_TRACE_EVENTS}+)",
            verified=events_ok,
            severity="CRITICAL" if not events_ok else "INFO"
        ))

        # STRICT: Check minimum function coverage
        funcs_ok = trace_stats["functions"] >= MIN_TRACED_FUNCTIONS
        evidence.append(EvidenceItem(
            source="AUDITS/TRACES/*.jsonl",
            claim=f"Functions traced >= {MIN_TRACED_FUNCTIONS}",
            evidence=f"{trace_stats['functions']} unique functions (need {MIN_TRACED_FUNCTIONS}+)",
            verified=funcs_ok,
            severity="CRITICAL" if not funcs_ok else "INFO"
        ))
    else:
        evidence.append(EvidenceItem(
            source="AUDITS/TRACES/",
            claim="Runtime traces exist",
            evidence="NO TRACE FILES FOUND",
            verified=False,
            severity="CRITICAL"  # Changed from WARNING to CRITICAL
        ))

    passed = all(e.verified for e in evidence)
    return VerificationResult(
        name="Runtime Traces (STRICT)",
        passed=passed,
        evidence=evidence,
        notes=f"Min events: {MIN_TRACE_EVENTS}, Min functions: {MIN_TRACED_FUNCTIONS}"
    )


def verify_safety_choke() -> VerificationResult:
    """Verify safety execution choke point."""
    evidence = []

    # Check if execution_choke.py exists
    choke_path = ROOT / "safety" / "execution_choke.py"
    if choke_path.exists():
        content = choke_path.read_text()

        # Check for key components
        checks = [
            ("evaluate_safety_gates function", "def evaluate_safety_gates" in content),
            ("SafetyViolationError class", "class SafetyViolationError" in content),
            ("6 required checks", "required_for_live = [" in content),
            ("Runtime token validation", "validate_ack_token" in content),
            ("Kill switch check", "_check_kill_switch" in content),
        ]

        for name, found in checks:
            evidence.append(EvidenceItem(
                source="safety/execution_choke.py",
                claim=name,
                evidence="FOUND" if found else "NOT FOUND",
                verified=found,
                severity="CRITICAL" if not found else "INFO"
            ))
    else:
        evidence.append(EvidenceItem(
            source="safety/execution_choke.py",
            claim="Execution choke point exists",
            evidence="FILE NOT FOUND",
            verified=False,
            severity="CRITICAL"
        ))

    passed = all(e.verified for e in evidence)
    return VerificationResult(
        name="Safety Execution Choke",
        passed=passed,
        evidence=evidence,
        notes="SINGLE enforcement point for all orders"
    )


def verify_bypass_tests() -> VerificationResult:
    """Verify bypass prevention tests exist."""
    evidence = []

    test_path = ROOT / "tests" / "security" / "test_live_bypass.py"
    if test_path.exists():
        content = test_path.read_text()

        # Count test methods
        test_count = content.count("def test_")

        evidence.append(EvidenceItem(
            source="tests/security/test_live_bypass.py",
            claim="Bypass tests exist",
            evidence=f"{test_count} test methods found",
            verified=test_count >= 5
        ))

        # Check for critical test types
        checks = [
            ("Tests live blocking", "test_live_blocked" in content),
            ("Tests wrong token", "wrong" in content.lower() and "token" in content.lower()),
            ("Tests kill switch", "kill_switch" in content.lower()),
            ("Tests all flags", "all" in content.lower() and "flags" in content.lower()),
        ]

        for name, found in checks:
            evidence.append(EvidenceItem(
                source="tests/security/test_live_bypass.py",
                claim=name,
                evidence="FOUND" if found else "NOT FOUND",
                verified=found
            ))
    else:
        evidence.append(EvidenceItem(
            source="tests/security/test_live_bypass.py",
            claim="Bypass tests exist",
            evidence="FILE NOT FOUND",
            verified=False,
            severity="CRITICAL"
        ))

    passed = all(e.verified for e in evidence)
    return VerificationResult(
        name="Bypass Prevention Tests",
        passed=passed,
        evidence=evidence
    )


def verify_critical_paths() -> VerificationResult:
    """Verify critical execution paths exist."""
    evidence = []

    # Check key files in data flow
    data_files = [
        ("data/providers/polygon_eod.py", "Polygon data provider"),
        ("data/universe/loader.py", "Universe loader"),
    ]

    for file_path, desc in data_files:
        full_path = ROOT / file_path
        evidence.append(EvidenceItem(
            source=file_path,
            claim=f"{desc} exists",
            evidence="EXISTS" if full_path.exists() else "NOT FOUND",
            verified=full_path.exists(),
            severity="CRITICAL" if not full_path.exists() else "INFO"
        ))

    # Check strategy files
    strategy_files = [
        ("strategies/dual_strategy/combined.py", "DualStrategyScanner"),
        ("strategies/ibs_rsi/strategy.py", "IBS+RSI Strategy"),
        ("strategies/ict/turtle_soup.py", "Turtle Soup Strategy"),
    ]

    for file_path, desc in strategy_files:
        full_path = ROOT / file_path
        evidence.append(EvidenceItem(
            source=file_path,
            claim=f"{desc} exists",
            evidence="EXISTS" if full_path.exists() else "NOT FOUND",
            verified=full_path.exists(),
            severity="CRITICAL" if not full_path.exists() else "INFO"
        ))

    # Check execution files
    exec_files = [
        ("execution/broker_alpaca.py", "Alpaca broker"),
        ("risk/policy_gate.py", "Policy gate"),
        ("oms/order_state.py", "Order state tracker"),
    ]

    for file_path, desc in exec_files:
        full_path = ROOT / file_path
        evidence.append(EvidenceItem(
            source=file_path,
            claim=f"{desc} exists",
            evidence="EXISTS" if full_path.exists() else "NOT FOUND",
            verified=full_path.exists(),
            severity="CRITICAL" if not full_path.exists() else "INFO"
        ))

    passed = all(e.verified for e in evidence)
    return VerificationResult(
        name="Critical Paths",
        passed=passed,
        evidence=evidence
    )


def calculate_grade(score: int) -> str:
    """Calculate letter grade from score."""
    if score >= 95:
        return "A+"
    elif score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


def calculate_verdict(checks: List[VerificationResult], strict: bool = True) -> tuple[str, int, List[str]]:
    """
    Calculate overall verdict and score with STRICT mode.

    In strict mode (default), ANY of these cause immediate FAIL:
    - SEV-0 bypass paths detected
    - Trace events < MIN_TRACE_EVENTS
    - Any CRITICAL evidence failure

    Returns:
        (verdict, score, failure_reasons)
    """
    total_evidence = 0
    verified_evidence = 0
    critical_failures = 0
    failure_reasons = []

    # Check for SEV-0 bypass issues
    sev_0_issues = [
        c for c in checks
        if c.name == "Bypass Vulnerability Scan" and not c.passed
    ]
    if sev_0_issues:
        failure_reasons.append("SEV-0 bypass vulnerabilities detected")

    # Check for insufficient traces
    trace_check = next(
        (c for c in checks if "Runtime Traces" in c.name),
        None
    )
    if trace_check and not trace_check.passed:
        failure_reasons.append(f"Insufficient runtime traces (need {MIN_TRACE_EVENTS}+ events)")

    # Count evidence
    for check in checks:
        for e in check.evidence:
            total_evidence += 1
            if e.verified:
                verified_evidence += 1
            elif e.severity == "CRITICAL":
                critical_failures += 1
                if strict:
                    failure_reasons.append(f"CRITICAL: {e.claim} - {e.evidence}")

    if total_evidence == 0:
        return "FAIL", 0, ["No evidence found"]

    score = int((verified_evidence / total_evidence) * 100)

    # STRICT MODE: Any failure reason = immediate FAIL
    if strict and failure_reasons:
        return "FAIL", score, failure_reasons

    # Non-strict mode (legacy behavior)
    if critical_failures > 0:
        verdict = "FAIL"
    elif score >= 90:
        verdict = "PASS"
    elif score >= 70:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return verdict, score, failure_reasons


def run_verification(strict: bool = True) -> WiringReport:
    """Run all verification checks with STRICT mode by default."""
    print("=" * 60)
    print("MASTER WIRING VERIFICATION (STRICT MODE)" if strict else "MASTER WIRING VERIFICATION")
    print("=" * 60)

    checks = [
        verify_no_bypass_issues(),  # NEW: Check for SEV-0/SEV-1 bypass paths FIRST
        verify_repo_census(),
        verify_entrypoints(),
        verify_component_inventory(),
        verify_truth_table(),
        verify_runtime_traces(),  # UPDATED: Now requires MIN_TRACE_EVENTS
        verify_safety_choke(),
        verify_bypass_tests(),
        verify_critical_paths(),
    ]

    # Calculate verdict with strict mode
    verdict, score, failure_reasons = calculate_verdict(checks, strict=strict)
    grade = calculate_grade(score)

    # Find critical issues (include failure reasons from strict mode)
    critical_issues = list(failure_reasons)  # Start with failure reasons
    for check in checks:
        for e in check.evidence:
            if not e.verified and e.severity == "CRITICAL":
                issue = f"{check.name}: {e.claim}"
                if issue not in critical_issues:
                    critical_issues.append(issue)

    # Generate recommendations
    recommendations = []
    for check in checks:
        if not check.passed:
            recommendations.append(f"Fix: {check.name}")

    report = WiringReport(
        timestamp=datetime.utcnow().isoformat(),
        verdict=verdict,
        grade=grade,
        score=score,
        checks=checks,
        critical_issues=critical_issues,
        recommendations=recommendations,
    )

    # Print results
    print(f"\nTimestamp: {report.timestamp}")
    print(f"Verdict: {report.verdict}")
    print(f"Grade: {report.grade}")
    print(f"Score: {report.score}/100")

    print("\n" + "-" * 60)
    print("CHECK RESULTS")
    print("-" * 60)

    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"\n[{status}] {check.name}")
        for e in check.evidence:
            mark = "+" if e.verified else "X"
            print(f"  [{mark}] {e.claim}: {e.evidence}")

    if critical_issues:
        print("\n" + "-" * 60)
        print("CRITICAL ISSUES")
        print("-" * 60)
        for issue in critical_issues:
            print(f"  ! {issue}")

    if recommendations:
        print("\n" + "-" * 60)
        print("RECOMMENDATIONS")
        print("-" * 60)
        for rec in recommendations:
            print(f"  * {rec}")

    return report


def save_report(report: WiringReport, output_dir: Optional[Path] = None) -> Path:
    """Save verification report."""
    if output_dir is None:
        output_dir = ROOT / "AUDITS"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to dict for JSON serialization
    report_dict = {
        "timestamp": report.timestamp,
        "verdict": report.verdict,
        "grade": report.grade,
        "score": report.score,
        "critical_issues": report.critical_issues,
        "recommendations": report.recommendations,
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "notes": c.notes,
                "evidence": [
                    {
                        "source": e.source,
                        "claim": e.claim,
                        "evidence": e.evidence,
                        "verified": e.verified,
                        "severity": e.severity,
                    }
                    for e in c.evidence
                ],
            }
            for c in report.checks
        ],
    }

    # Save JSON
    json_path = output_dir / "WIRING_VERIFICATION.json"
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\nReport saved to: {json_path}")
    return json_path


def main():
    """Run master wiring verification."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Master wiring verification (STRICT MODE by default)"
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable strict mode (legacy behavior, NOT recommended)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Enable strict mode (default)"
    )
    args = parser.parse_args()

    strict = not args.no_strict

    print(f"\nMode: {'STRICT' if strict else 'LEGACY'}")
    print(f"Thresholds: MIN_TRACE_EVENTS={MIN_TRACE_EVENTS}, MIN_TRACED_FUNCTIONS={MIN_TRACED_FUNCTIONS}")
    print("")

    report = run_verification(strict=strict)
    save_report(report)

    # Exit with appropriate code
    if report.verdict == "PASS":
        print(f"\n{'=' * 60}")
        print(f"VERDICT: {report.verdict} (Grade {report.grade}, Score {report.score}/100)")
        print(f"{'=' * 60}")
        sys.exit(0)
    else:
        print(f"\n{'=' * 60}")
        print(f"VERDICT: {report.verdict} (Grade {report.grade}, Score {report.score}/100)")
        if report.critical_issues:
            print(f"\nFAILURE REASONS:")
            for issue in report.critical_issues[:5]:  # Show top 5
                print(f"  ! {issue}")
        print(f"{'=' * 60}")
        sys.exit(1 if report.verdict == "FAIL" else 0)


if __name__ == "__main__":
    main()
