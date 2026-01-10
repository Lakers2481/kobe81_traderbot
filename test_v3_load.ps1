# Test if v3 loads without errors
Write-Host "Testing v3 load..." -ForegroundColor Cyan

try {
    . "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\mamba_ai_v3.ps1"
    Write-Host "✅ V3 loaded successfully!" -ForegroundColor Green

    # Test if v3 functions exist
    if (Get-Command "ai-autonomous" -ErrorAction SilentlyContinue) {
        Write-Host "✅ ai-autonomous command exists" -ForegroundColor Green
    } else {
        Write-Host "❌ ai-autonomous command NOT found" -ForegroundColor Red
    }

    if (Get-Command "ai-scan-issues" -ErrorAction SilentlyContinue) {
        Write-Host "✅ ai-scan-issues command exists" -ForegroundColor Green
    } else {
        Write-Host "❌ ai-scan-issues command NOT found" -ForegroundColor Red
    }

} catch {
    Write-Host "❌ Error loading v3:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor Gray
}
