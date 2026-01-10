# Diagnostic script to find what's broken
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host "  MAMBA AI V3 DIAGNOSTIC" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host ""

# Check 1: Does the fixed file exist?
Write-Host "CHECK 1: Does mamba_ai_v3_fixed.ps1 exist?" -ForegroundColor Cyan
$fixedPath = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\mamba_ai_v3_fixed.ps1"
if (Test-Path $fixedPath) {
    Write-Host "  ✅ YES - File exists" -ForegroundColor Green
    $fileSize = (Get-Item $fixedPath).Length
    Write-Host "     Size: $fileSize bytes" -ForegroundColor Gray
} else {
    Write-Host "  ❌ NO - File does NOT exist!" -ForegroundColor Red
    Write-Host "     Expected at: $fixedPath" -ForegroundColor Red
}
Write-Host ""

# Check 2: Try to load V2 only
Write-Host "CHECK 2: Loading V2 (base system)..." -ForegroundColor Cyan
try {
    . "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\mamba_ai_v2.ps1"
    Write-Host "  ✅ V2 loaded successfully" -ForegroundColor Green
} catch {
    Write-Host "  ❌ V2 FAILED to load!" -ForegroundColor Red
    Write-Host "     Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Check 3: Does ai command exist?
Write-Host "CHECK 3: Does 'ai' command exist?" -ForegroundColor Cyan
if (Get-Command "ai" -ErrorAction SilentlyContinue) {
    Write-Host "  ✅ YES - ai command exists (V2 working)" -ForegroundColor Green
} else {
    Write-Host "  ❌ NO - ai command does NOT exist (V2 broken)" -ForegroundColor Red
}
Write-Host ""

# Check 4: Try to load V3 fixed
Write-Host "CHECK 4: Loading V3 fixed..." -ForegroundColor Cyan
if (Test-Path $fixedPath) {
    try {
        . $fixedPath
        Write-Host "  ✅ V3 loaded successfully" -ForegroundColor Green
    } catch {
        Write-Host "  ❌ V3 FAILED to load!" -ForegroundColor Red
        Write-Host "     Error: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "     Line: $($_.InvocationInfo.ScriptLineNumber)" -ForegroundColor Red
        Write-Host ""
        Write-Host "  Full error details:" -ForegroundColor Yellow
        Write-Host $_.Exception.Message -ForegroundColor Red
        Write-Host $_.ScriptStackTrace -ForegroundColor Gray
    }
} else {
    Write-Host "  ⚠️  SKIPPED - File doesn't exist" -ForegroundColor Yellow
}
Write-Host ""

# Check 5: Does ai-autonomous exist?
Write-Host "CHECK 5: Does 'ai-autonomous' command exist?" -ForegroundColor Cyan
if (Get-Command "ai-autonomous" -ErrorAction SilentlyContinue) {
    Write-Host "  ✅ YES - ai-autonomous exists (V3 working)" -ForegroundColor Green
} else {
    Write-Host "  ❌ NO - ai-autonomous does NOT exist (V3 not loaded)" -ForegroundColor Red
}
Write-Host ""

# Check 6: API key set?
Write-Host "CHECK 6: Is OpenAI API key set?" -ForegroundColor Cyan
if ($env:OPENAI_API_KEY) {
    $keyPreview = $env:OPENAI_API_KEY.Substring(0, [Math]::Min(10, $env:OPENAI_API_KEY.Length)) + "..."
    Write-Host "  ✅ YES - Key starts with: $keyPreview" -ForegroundColor Green
} else {
    Write-Host "  ❌ NO - API key not set!" -ForegroundColor Red
    Write-Host "     Set with: `$env:OPENAI_API_KEY = 'sk-your-key'" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host "  SUMMARY" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host ""
Write-Host "Run this from PowerShell:" -ForegroundColor Cyan
Write-Host "  cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot" -ForegroundColor Yellow
Write-Host "  .\diagnose_v3.ps1" -ForegroundColor Yellow
Write-Host ""
