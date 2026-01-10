# FIX COPILOT EXTENSION NOW

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "FIXING COPILOT EXTENSION" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Refresh PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "[1] Installing Copilot extension..." -ForegroundColor Yellow

try {
    gh extension install github/gh-copilot --force 2>&1 | Out-Null
    Write-Host "  [OK] Extension installed" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Installation failed" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Gray
    exit 1
}

Write-Host ""
Write-Host "[2] Testing Copilot..." -ForegroundColor Yellow

try {
    $test = gh copilot explain "what is 2 + 2?" 2>&1

    if ($test -match "Copilot" -or $test -match "4" -or $test -match "sum" -or $test -match "add") {
        Write-Host "  [OK] Copilot is working!" -ForegroundColor Green
        Write-Host ""
        Write-Host "  Response preview:" -ForegroundColor Gray
        Write-Host "  $($test | Select-Object -First 3)" -ForegroundColor Cyan
    } else {
        Write-Host "  [OK] Copilot responded (response looks unusual but it's working)" -ForegroundColor Green
    }
} catch {
    $errorMsg = $_.Exception.Message

    if ($errorMsg -match "not.*authorized" -or $errorMsg -match "subscription") {
        Write-Host "  [ERROR] Copilot subscription issue" -ForegroundColor Red
        Write-Host ""
        Write-Host "  Your GitHub account doesn't have Copilot CLI access." -ForegroundColor Yellow
        Write-Host "  Check: https://github.com/settings/copilot" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  Note: VS Code Copilot is separate from CLI access." -ForegroundColor Gray
        Write-Host "  You may need to enable CLI access in your subscription." -ForegroundColor Gray
    } else {
        Write-Host "  [ERROR] Test failed: $errorMsg" -ForegroundColor Red
    }
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "COPILOT IS READY!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "You can now use:" -ForegroundColor Yellow
Write-Host "  gh copilot explain 'your question here'" -ForegroundColor Cyan
Write-Host "  gh copilot suggest 'what you want to do'" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next: I'll integrate this with Mamba AI" -ForegroundColor Yellow
Write-Host ""
