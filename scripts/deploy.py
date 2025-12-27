#!/usr/bin/env python3
"""
Safe deployment for Kobe trading system.
Supports pre-deploy checks, backup, deploy, and rollback.
Usage: python scripts/deploy.py [--check|--backup|--deploy|--rollback]
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Deployment configuration
DEPLOY_DIR = Path("deployments")
BACKUP_DIR = DEPLOY_DIR / "backups"
ROLLBACK_DIR = DEPLOY_DIR / "rollback"

# Critical files that need backup before deploy
CRITICAL_FILES = [
    "config/base.yaml",
    "state/positions.json",
    "state/order_history.json",
    "state/hash_chain.jsonl",
]

# Directories to preserve
PRESERVE_DIRS = [
    "data/cache",
    "data/universe",
    "logs",
    "state",
    "wf_outputs",
]


def ensure_dirs():
    """Ensure deployment directories exist."""
    DEPLOY_DIR.mkdir(exist_ok=True)
    BACKUP_DIR.mkdir(exist_ok=True)
    ROLLBACK_DIR.mkdir(exist_ok=True)


def get_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of file."""
    if not filepath.exists():
        return ""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def run_checks() -> dict:
    """Run pre-deployment checks."""
    print("\n=== Pre-Deployment Checks ===\n")

    results = {
        "timestamp": datetime.now().isoformat(),
        "checks": [],
        "all_passed": True,
    }

    # Check 1: No active positions
    print("1. Checking for active positions...")
    positions_file = Path("state/positions.json")
    if positions_file.exists():
        with open(positions_file) as f:
            positions = json.load(f)
        if positions:
            results["checks"].append({"name": "no_positions", "passed": False, "message": f"{len(positions)} active positions"})
            results["all_passed"] = False
            print(f"   [FAIL] {len(positions)} active positions - close before deploying")
        else:
            results["checks"].append({"name": "no_positions", "passed": True, "message": "No active positions"})
            print("   [OK] No active positions")
    else:
        results["checks"].append({"name": "no_positions", "passed": True, "message": "No positions file"})
        print("   [OK] No positions file")

    # Check 2: Kill switch not active
    print("2. Checking kill switch status...")
    kill_switch = Path("state/KILL_SWITCH")
    if kill_switch.exists():
        results["checks"].append({"name": "kill_switch", "passed": False, "message": "Kill switch is active"})
        results["all_passed"] = False
        print("   [FAIL] Kill switch is active - remove before deploying")
    else:
        results["checks"].append({"name": "kill_switch", "passed": True, "message": "Kill switch not active"})
        print("   [OK] Kill switch not active")

    # Check 3: Git status clean
    print("3. Checking git status...")
    try:
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if result.stdout.strip():
            uncommitted = len(result.stdout.strip().split("\n"))
            results["checks"].append({"name": "git_clean", "passed": False, "message": f"{uncommitted} uncommitted changes"})
            results["all_passed"] = False
            print(f"   [WARN] {uncommitted} uncommitted changes")
        else:
            results["checks"].append({"name": "git_clean", "passed": True, "message": "Working tree clean"})
            print("   [OK] Working tree clean")
    except Exception as e:
        results["checks"].append({"name": "git_clean", "passed": True, "message": "Git not available"})
        print(f"   [SKIP] Git not available: {e}")

    # Check 4: Tests pass
    print("4. Running smoke tests...")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/smoke_test.py"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            results["checks"].append({"name": "tests", "passed": True, "message": "Smoke tests passed"})
            print("   [OK] Smoke tests passed")
        else:
            results["checks"].append({"name": "tests", "passed": False, "message": "Smoke tests failed"})
            results["all_passed"] = False
            print("   [FAIL] Smoke tests failed")
    except subprocess.TimeoutExpired:
        results["checks"].append({"name": "tests", "passed": False, "message": "Tests timed out"})
        results["all_passed"] = False
        print("   [FAIL] Tests timed out")
    except FileNotFoundError:
        results["checks"].append({"name": "tests", "passed": True, "message": "Smoke test not found"})
        print("   [SKIP] Smoke test script not found")

    # Check 5: Config validation
    print("5. Validating configuration...")
    config_file = Path("config/base.yaml")
    if config_file.exists():
        results["checks"].append({"name": "config", "passed": True, "message": "Config file exists"})
        print("   [OK] Configuration file exists")
    else:
        results["checks"].append({"name": "config", "passed": False, "message": "Config file missing"})
        results["all_passed"] = False
        print("   [FAIL] Configuration file missing")

    # Summary
    print("\n" + "=" * 40)
    passed = sum(1 for c in results["checks"] if c["passed"])
    total = len(results["checks"])
    print(f"Checks: {passed}/{total} passed")
    print(f"Status: {'READY FOR DEPLOY' if results['all_passed'] else 'NOT READY'}")

    return results


def create_backup() -> str:
    """Create pre-deployment backup."""
    ensure_dirs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"backup_{timestamp}"
    backup_path.mkdir(parents=True)

    print(f"\n=== Creating Backup: {backup_path.name} ===\n")

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "files": [],
        "directories": [],
    }

    # Backup critical files
    for file_path in CRITICAL_FILES:
        src = Path(file_path)
        if src.exists():
            dst = backup_path / file_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            manifest["files"].append({
                "path": file_path,
                "hash": get_file_hash(src),
            })
            print(f"  Backed up: {file_path}")

    # Backup preserve directories
    for dir_path in PRESERVE_DIRS:
        src = Path(dir_path)
        if src.exists() and src.is_dir():
            dst = backup_path / dir_path
            shutil.copytree(src, dst, dirs_exist_ok=True)
            file_count = len(list(dst.rglob("*")))
            manifest["directories"].append({
                "path": dir_path,
                "file_count": file_count,
            })
            print(f"  Backed up directory: {dir_path} ({file_count} files)")

    # Write manifest
    with open(backup_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nBackup created: {backup_path}")
    return str(backup_path)


def deploy(backup_first: bool = True) -> bool:
    """Perform deployment."""
    print("\n=== Kobe Deployment ===\n")

    # Run checks
    checks = run_checks()
    if not checks["all_passed"]:
        print("\n[ABORT] Pre-deployment checks failed. Fix issues before deploying.")
        return False

    # Create backup
    if backup_first:
        backup_path = create_backup()
        # Save as rollback point
        rollback_marker = ROLLBACK_DIR / "latest"
        with open(rollback_marker, "w") as f:
            f.write(backup_path)

    # Get current git commit
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        commit_before = result.stdout.strip()[:8]
    except:
        commit_before = "unknown"

    print("\n--- Deployment Steps ---\n")

    # Step 1: Pull latest code
    print("1. Pulling latest code...")
    try:
        result = subprocess.run(["git", "pull"], capture_output=True, text=True)
        if result.returncode == 0:
            print("   [OK] Git pull successful")
        else:
            print(f"   [WARN] Git pull: {result.stderr.strip()}")
    except Exception as e:
        print(f"   [SKIP] Git not available: {e}")

    # Step 2: Install dependencies
    print("2. Installing dependencies...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("   [OK] Dependencies installed")
        else:
            print(f"   [WARN] pip install: {result.stderr.strip()}")
    except Exception as e:
        print(f"   [FAIL] Error: {e}")

    # Step 3: Run migrations (if any)
    print("3. Running migrations...")
    print("   [SKIP] No migrations configured")

    # Step 4: Verify deployment
    print("4. Verifying deployment...")
    post_checks = run_checks()

    # Record deployment
    deploy_record = {
        "timestamp": datetime.now().isoformat(),
        "commit_before": commit_before,
        "checks_passed": post_checks["all_passed"],
    }

    deploy_log = DEPLOY_DIR / "deploy_history.jsonl"
    with open(deploy_log, "a") as f:
        f.write(json.dumps(deploy_record) + "\n")

    print("\n" + "=" * 40)
    print(f"Deployment: {'SUCCESS' if post_checks['all_passed'] else 'COMPLETED WITH WARNINGS'}")

    return post_checks["all_passed"]


def rollback():
    """Rollback to last backup."""
    print("\n=== Rollback ===\n")

    rollback_marker = ROLLBACK_DIR / "latest"
    if not rollback_marker.exists():
        print("[ERROR] No rollback point found")
        return False

    with open(rollback_marker) as f:
        backup_path = Path(f.read().strip())

    if not backup_path.exists():
        print(f"[ERROR] Backup not found: {backup_path}")
        return False

    manifest_file = backup_path / "manifest.json"
    if not manifest_file.exists():
        print("[ERROR] Backup manifest not found")
        return False

    with open(manifest_file) as f:
        manifest = json.load(f)

    print(f"Rolling back to: {backup_path.name}")
    print(f"Backup from: {manifest['timestamp']}")

    # Restore files
    for file_info in manifest["files"]:
        src = backup_path / file_info["path"]
        dst = Path(file_info["path"])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  Restored: {file_info['path']}")

    # Restore directories
    for dir_info in manifest["directories"]:
        src = backup_path / dir_info["path"]
        dst = Path(dir_info["path"])
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Restored directory: {dir_info['path']}")

    print("\n[OK] Rollback complete")
    return True


def main():
    parser = argparse.ArgumentParser(description="Safe deployment management")
    parser.add_argument("--check", action="store_true", help="Run pre-deployment checks")
    parser.add_argument("--backup", action="store_true", help="Create backup only")
    parser.add_argument("--deploy", action="store_true", help="Perform deployment")
    parser.add_argument("--rollback", action="store_true", help="Rollback to last backup")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup during deploy")

    args = parser.parse_args()

    if args.rollback:
        success = rollback()
        sys.exit(0 if success else 1)
    elif args.backup:
        create_backup()
    elif args.deploy:
        success = deploy(backup_first=not args.no_backup)
        sys.exit(0 if success else 1)
    elif args.check:
        checks = run_checks()
        sys.exit(0 if checks["all_passed"] else 1)
    else:
        # Default: run checks
        checks = run_checks()
        sys.exit(0 if checks["all_passed"] else 1)


if __name__ == "__main__":
    main()
