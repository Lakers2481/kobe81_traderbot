#!/usr/bin/env python3
"""
FWO-Prime Folder Structure Analyzer
Comprehensive audit of all 1900+ directories in kobe81_traderbot
"""
import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

ROOT = Path(r"C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot")

# Canonical structure from FWO-Prime instructions
CANONICAL_STRUCTURE = {
    "data": ["stocks.csv", "prices/", "metadata/"],
    "strategies": [],
    "backtests": [],
    "reports": [],
    "logs": [],
    "scripts": [],
    "agents": [],
    "_archive": []
}

# Categories for classification
CATEGORIES = {
    "data_related": ["data", "data_exploration", "cache", "polygon_cache"],
    "strategy_related": ["strategies", "strategy_specs"],
    "logs_and_runs": ["logs", "state", "stateguardian"],
    "reports_and_exports": ["reports", "AUDITS", "docs", "RUNBOOKS"],
    "scripts_and_tools": ["scripts", "tools", "ops"],
    "agents_and_configs": ["agents", "config", "cognitive", "autonomous"],
    "ml_and_research": ["ml", "ml_advanced", "ml_features", "ml_meta", "research", "research_os"],
    "execution_and_risk": ["execution", "risk", "portfolio", "oms", "safety"],
    "backtest_and_analysis": ["backtest", "analytics", "analysis", "evaluation", "optimization"],
    "outputs_temp": ["backtest_outputs", "wf_outputs", "showdown_outputs", "optimize_outputs", "smoke_outputs", "output", "outputs"],
    "archive_backup": ["archive", "backups", "RELEASE"],
    "testing": ["tests", "testing", "preflight", "quant_gates"],
    "infrastructure": ["core", "monitor", "observability", "guardian", "compliance"],
    "integrations": ["llm", "altdata", "news", "messaging", "alerts", "web", "dashboard"],
    "specialized": ["options", "bounce", "scanner", "evolution", "experiments", "pipelines", "explainability"],
    "cache_temp": [".pytest_cache", ".ruff_cache", "__pycache__", "mlruns", "cache"],
}

# Naming convention rules
NAMING_VIOLATIONS = {
    "uppercase_module": lambda n: n.isupper() and n not in ["AUDITS", "RUNBOOKS", "RELEASE", "README"],
    "mixed_case": lambda n: any(c.isupper() for c in n[1:]) and "_" not in n and "-" not in n,
    "spaces": lambda n: " " in n,
    "special_chars": lambda n: any(c in n for c in "!@#$%^&*()+=[]{}|\\;:'\",<>?"),
    "ends_with_number": lambda n: n[-1].isdigit() and n not in ["smoke_wf_audit", "showdown_2025_cap60"],
}

def analyze_repository():
    """Comprehensive folder structure analysis"""

    # Statistics
    stats = {
        "total_dirs": 0,
        "total_files": 0,
        "python_dirs": 0,
        "missing_init": [],
        "naming_violations": defaultdict(list),
        "misplaced_files": [],
        "duplicate_dirs": defaultdict(list),
        "oversized_dirs": [],
        "broken_paths": [],
        "orphaned_files": [],
        "category_counts": defaultdict(int),
    }

    # Walk entire tree
    for root, dirs, files in os.walk(ROOT):
        # Skip .git
        if ".git" in root:
            continue

        rel_path = Path(root).relative_to(ROOT)
        stats["total_dirs"] += len(dirs)
        stats["total_files"] += len(files)

        # Check for broken absolute paths
        for d in dirs:
            if d.startswith("C:") or d.startswith("/c/"):
                stats["broken_paths"].append(str(rel_path / d))

        # Check for Python packages missing __init__.py
        has_python = any(f.endswith(".py") for f in files)
        if has_python:
            stats["python_dirs"] += 1
            dir_name = rel_path.name
            # Exclude non-package dirs
            if dir_name not in ["scripts", "tools", "tests", ".pytest_cache", "__pycache__", "mlruns"]:
                if "__init__.py" not in files and rel_path != Path("."):
                    stats["missing_init"].append(str(rel_path))

        # Check naming conventions
        for d in dirs:
            for violation_type, check_func in NAMING_VIOLATIONS.items():
                if check_func(d):
                    stats["naming_violations"][violation_type].append(str(rel_path / d))

        # Detect duplicates (similar names)
        for d in dirs:
            base = d.rstrip("s_0123456789")  # Normalize
            stats["duplicate_dirs"][base].append(str(rel_path / d))

    # Filter duplicates (only keep actual duplicates)
    stats["duplicate_dirs"] = {k: v for k, v in stats["duplicate_dirs"].items() if len(v) > 1}

    # Check directory sizes
    for d in ROOT.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            try:
                size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024*1024)
                if size_mb > 50:  # >50MB
                    stats["oversized_dirs"].append((d.name, f"{size_mb:.1f}MB"))
            except:
                pass

    # Categorize all top-level dirs
    for d in ROOT.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            categorized = False
            for category, patterns in CATEGORIES.items():
                if d.name in patterns or any(p in d.name for p in patterns):
                    stats["category_counts"][category] += 1
                    categorized = True
                    break
            if not categorized:
                stats["category_counts"]["unknown_or_misplaced"] += 1

    # Find loose files in root (should be in docs/)
    root_files = [f for f in ROOT.iterdir() if f.is_file() and f.suffix in [".md", ".json", ".txt"]]
    stats["orphaned_files"] = [f.name for f in root_files if f.name not in ["README.md", "CLAUDE.md", "requirements.txt", "pytest.ini", "Makefile", "Dockerfile", "docker-compose.yml", ".gitignore"]]

    return stats

def generate_manifest():
    """Generate FWO-Prime style manifest"""

    manifest = {
        "agent": "folder_orchestrator_v1",
        "ts_utc": datetime.utcnow().isoformat(),
        "root": str(ROOT),
        "folders": {},
        "files": [],
        "actions": [],
        "summary": {
            "ok": True,
            "problems": 0,
            "suggested_moves": 0
        },
        "problems": [],
        "suggested_moves": [],
        "hints_for_agents": []
    }

    # Scan top-level structure
    for entry in sorted(ROOT.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            manifest["folders"][entry.name] = f"{entry.name}\\"

    # Problems
    stats = analyze_repository()

    # Broken paths
    if stats["broken_paths"]:
        manifest["problems"].append(f"Found {len(stats['broken_paths'])} directories with broken absolute paths")
        for bp in stats["broken_paths"]:
            manifest["suggested_moves"].append({
                "from": str(ROOT / bp),
                "to": "DELETE",
                "reason": "Broken absolute path directory - likely created by error"
            })

    # Orphaned root files
    if stats["orphaned_files"]:
        manifest["problems"].append(f"Found {len(stats['orphaned_files'])} loose documentation files in root")
        for of in stats["orphaned_files"]:
            target = "docs/" if of.endswith(".md") else "AUDITS/"
            manifest["suggested_moves"].append({
                "from": str(ROOT / of),
                "to": str(ROOT / target / of),
                "reason": "Root clutter - move to proper location"
            })

    # Output redundancy
    output_dirs = [d for d in manifest["folders"] if "output" in d.lower()]
    if len(output_dirs) > 2:
        manifest["problems"].append(f"Found {len(output_dirs)} output directories - consolidate")
        manifest["actions"].append("CONSOLIDATE outputs/ and output/ into single outputs/ directory")

    # Missing __init__.py (sample)
    if len(stats["missing_init"]) > 10:
        manifest["problems"].append(f"Found {len(stats['missing_init'])} Python packages missing __init__.py")
        manifest["actions"].append(f"CREATE __init__.py in {len(stats['missing_init'])} directories")

    # Strange directories
    strange = ["vuLDY5zrhSOyIpTVB6JB5taCKu71bWAQ", "_ul", "_ul-DESKTOP-5IB5S6R", "nul"]
    for s in strange:
        if (ROOT / s).exists():
            manifest["problems"].append(f"Found strange directory: {s}")
            manifest["suggested_moves"].append({
                "from": str(ROOT / s),
                "to": "DELETE or INVESTIGATE",
                "reason": "Unknown purpose, possibly temp/garbage"
            })

    # Oversized in OneDrive path
    for dirname, size in stats["oversized_dirs"]:
        if "output" in dirname.lower():
            manifest["problems"].append(f"{dirname}/ is {size} - too large for OneDrive sync")
            manifest["suggested_moves"].append({
                "from": str(ROOT / dirname),
                "to": f"C:\\ICT_OFFLINE_BIGFILES\\{dirname}",
                "reason": f"Size {size} exceeds cloud-safe threshold"
            })

    manifest["summary"]["problems"] = len(manifest["problems"])
    manifest["summary"]["suggested_moves"] = len(manifest["suggested_moves"])
    manifest["summary"]["ok"] = len(manifest["problems"]) == 0

    # Hints for other agents
    manifest["hints_for_agents"] = [
        "data_agent: Primary data location is data/, cache is data/cache/ and cache/",
        "backtest_agent: Backtest outputs are in backtest_outputs/, wf_outputs/, and showdown_outputs/",
        "strategy_agent: Strategies are in strategies/ with dual_strategy/, ibs_rsi/, ict/ subdirs",
        "report_agent: Reports go to reports/, human-readable summaries to reports/tearsheets/",
        "claude_agent: Documentation in docs/, audit reports in AUDITS/",
    ]

    return manifest, stats

if __name__ == "__main__":
    print("FWO-Prime: Analyzing 1900+ directories...")
    manifest, stats = generate_manifest()

    # Save manifest
    manifest_path = ROOT / "AUDITS" / "folder_structure_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved: {manifest_path}")
    print(f"Total directories: {stats['total_dirs']}")
    print(f"Total files: {stats['total_files']}")
    print(f"Python packages: {stats['python_dirs']}")
    print(f"Missing __init__.py: {len(stats['missing_init'])}")
    print(f"Broken paths: {len(stats['broken_paths'])}")
    print(f"Problems found: {manifest['summary']['problems']}")
    print(f"Suggested moves: {manifest['summary']['suggested_moves']}")
