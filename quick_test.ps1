# Quick test to see what's working
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host "  MAMBA AI QUICK TEST" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host ""

# Test 1: Check API Key
Write-Host "TEST 1: Is OpenAI API key set?" -ForegroundColor Cyan
if ($env:OPENAI_API_KEY) {
    $keyLength = $env:OPENAI_API_KEY.Length
    $keyStart = $env:OPENAI_API_KEY.Substring(0, [Math]::Min(7, $keyLength))
    Write-Host "  ✅ YES - Key starts with: $keyStart..." -ForegroundColor Green
    Write-Host "     Length: $keyLength characters" -ForegroundColor Gray
} else {
    Write-Host "  ❌ NO - API key NOT set!" -ForegroundColor Red
    Write-Host "" -ForegroundColor Yellow
    Write-Host "  FIX THIS NOW:" -ForegroundColor Yellow
    Write-Host "  1. Go to: https://platform.openai.com/api-keys" -ForegroundColor White
    Write-Host "  2. Create a key" -ForegroundColor White
    Write-Host "  3. Run this:" -ForegroundColor White
    Write-Host "     `$env:OPENAI_API_KEY = 'sk-your-key-here'" -ForegroundColor Cyan
    Write-Host ""
}
Write-Host ""

# Test 2: Check commands exist
Write-Host "TEST 2: Do commands exist?" -ForegroundColor Cyan
$commands = @('ai', 'talk', 'ai-autonomous', 'ai-scan-issues')
foreach ($cmd in $commands) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        Write-Host "  ✅ $cmd" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $cmd NOT FOUND" -ForegroundColor Red
    }
}
Write-Host ""

# Test 3: Show syntax examples
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host "  HOW TO USE IT" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host ""

Write-Host "METHOD 1: TALK (Natural - EASIEST)" -ForegroundColor Green
Write-Host "  Just type talk followed by your question:" -ForegroundColor Gray
Write-Host ""
Write-Host "  talk what files are in this folder?" -ForegroundColor Cyan
Write-Host "  talk explain this code" -ForegroundColor Cyan
Write-Host "  talk find all bugs" -ForegroundColor Cyan
Write-Host ""

Write-Host "METHOD 2: AI (Traditional)" -ForegroundColor Green
Write-Host "  Use quotes for your question:" -ForegroundColor Gray
Write-Host ""
Write-Host '  ai "what files are in this folder?"' -ForegroundColor Cyan
Write-Host '  ai "explain this code"' -ForegroundColor Cyan
Write-Host ""

Write-Host "METHOD 3: AUTONOMOUS (Multi-Step)" -ForegroundColor Green
Write-Host "  For complex tasks that need multiple steps:" -ForegroundColor Gray
Write-Host ""
Write-Host '  ai-autonomous "find and list all Python files"' -ForegroundColor Cyan
Write-Host '  ai-autonomous "scan for bugs and create report"' -ForegroundColor Cyan
Write-Host ""

Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host "  TRY THIS NOW" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host ""
Write-Host "Copy and paste this line:" -ForegroundColor White
Write-Host 'talk what is 2 plus 2?' -ForegroundColor Cyan
Write-Host ""
Write-Host "If you get an answer, IT WORKS! 🎉" -ForegroundColor Green
Write-Host ""
