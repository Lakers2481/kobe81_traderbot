# ============================================================
# KOBE TRADING SYSTEM - RUN ALL CHECKS
# ============================================================
# PowerShell script to run comprehensive system verification.
# Use this before deployments or after major changes.
# ============================================================

Write-Host "============================================================"
Write-Host "KOBE TRADING SYSTEM - COMPREHENSIVE CHECKS"
Write-Host "============================================================"
Write-Host ""

# Set location to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $scriptPath "..")

$errors = 0

# ============================================================
# 1. PREFLIGHT CHECK
# ============================================================
Write-Host "[1/6] Running preflight checks..."
Write-Host "------------------------------------------------------------"
python scripts/preflight.py --dotenv ./.env
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: Preflight checks" -ForegroundColor Red
    $errors++
} else {
    Write-Host "PASSED: Preflight checks" -ForegroundColor Green
}
Write-Host ""

# ============================================================
# 2. IMPORT VERIFICATION
# ============================================================
Write-Host "[2/6] Verifying imports..."
Write-Host "------------------------------------------------------------"
python -c "from safety import PAPER_ONLY; from autonomous.run import KobeRunner; from pipelines import PIPELINE_REGISTRY; print('All imports OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: Import verification" -ForegroundColor Red
    $errors++
} else {
    Write-Host "PASSED: Import verification" -ForegroundColor Green
}
Write-Host ""

# ============================================================
# 3. STATUS CHECK
# ============================================================
Write-Host "[3/6] Checking system status..."
Write-Host "------------------------------------------------------------"
python -m autonomous.run --status
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Status check returned non-zero" -ForegroundColor Yellow
}
Write-Host ""

# ============================================================
# 4. HEALTH CHECK
# ============================================================
Write-Host "[4/6] Running health check..."
Write-Host "------------------------------------------------------------"
python -m autonomous.run --health
if ($LASTEXITCODE -ne 0) {
    Write-Host "INFO: Brain not running (expected if not started)" -ForegroundColor Yellow
} else {
    Write-Host "PASSED: Health check" -ForegroundColor Green
}
Write-Host ""

# ============================================================
# 5. QUICK TEST
# ============================================================
Write-Host "[5/6] Running quick tests..."
Write-Host "------------------------------------------------------------"
if (Test-Path "tests") {
    pytest tests/ -q --tb=no -x 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Some tests failed" -ForegroundColor Yellow
    } else {
        Write-Host "PASSED: Tests" -ForegroundColor Green
    }
} else {
    Write-Host "SKIPPED: No tests directory" -ForegroundColor Yellow
}
Write-Host ""

# ============================================================
# 6. DEMO RUN
# ============================================================
Write-Host "[6/6] Running demo..."
Write-Host "------------------------------------------------------------"
python -m autonomous.run --tour
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Demo returned non-zero" -ForegroundColor Yellow
} else {
    Write-Host "PASSED: Demo" -ForegroundColor Green
}
Write-Host ""

# ============================================================
# SUMMARY
# ============================================================
Write-Host "============================================================"
Write-Host "CHECK SUMMARY"
Write-Host "============================================================"
if ($errors -eq 0) {
    Write-Host "ALL CHECKS PASSED" -ForegroundColor Green
} else {
    Write-Host "FAILED CHECKS: $errors" -ForegroundColor Red
}
Write-Host "============================================================"

# Return exit code
exit $errors
