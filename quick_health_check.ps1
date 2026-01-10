# Quick health check (runs on startup - silent unless there are issues)

$script:ISSUES = @()
$script:KOBE_ROOT = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot"

# Check critical components
if (-not (Test-Path "$script:KOBE_ROOT\.env")) {
    $script:ISSUES += "❌ Missing .env file"
}

if (-not $env:ANTHROPIC_API_KEY) {
    $script:ISSUES += "⚠️  Claude API key not loaded"
}

if (-not $env:POLYGON_API_KEY) {
    $script:ISSUES += "❌ Polygon API key not loaded (CRITICAL)"
}

if (-not (Test-Path "$script:KOBE_ROOT\data\universe\optionable_liquid_800.csv")) {
    $script:ISSUES += "⚠️  Universe file not found"
}

if (Test-Path "$script:KOBE_ROOT\state\KILL_SWITCH") {
    $script:ISSUES += "🔴 KILL SWITCH IS ACTIVE - Trading BLOCKED!"
}

# Display issues if any
if ($script:ISSUES.Count -gt 0) {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "║           ⚠️  SYSTEM HEALTH ISSUES DETECTED ⚠️                   ║" -ForegroundColor Red
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Red
    Write-Host ""
    foreach ($issue in $script:ISSUES) {
        Write-Host "  $issue" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "  Run " -NoNewline -ForegroundColor Gray
    Write-Host ".\verify_complete_system.ps1" -NoNewline -ForegroundColor Cyan
    Write-Host " for full diagnostics" -ForegroundColor Gray
    Write-Host ""
}
