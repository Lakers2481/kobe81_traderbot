# Quick test to verify Mamba AI chat is working

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║           🧪 TESTING MAMBA AI CHAT 🧪                           ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Test 1: Check if Claude API key is loaded
Write-Host "TEST 1: Checking Claude API key..." -ForegroundColor Yellow
if ($env:ANTHROPIC_API_KEY) {
    $keyPreview = $env:ANTHROPIC_API_KEY.Substring(0, 10) + "..."
    Write-Host "  ✅ Claude API key loaded: $keyPreview" -ForegroundColor Green
} else {
    Write-Host "  ❌ Claude API key NOT loaded!" -ForegroundColor Red
    Write-Host "     Run: reload" -ForegroundColor Yellow
    exit
}
Write-Host ""

# Test 2: Check if ai command exists
Write-Host "TEST 2: Checking if 'ai' command exists..." -ForegroundColor Yellow
if (Get-Command ai -ErrorAction SilentlyContinue) {
    Write-Host "  ✅ 'ai' command found" -ForegroundColor Green
} else {
    Write-Host "  ❌ 'ai' command NOT found!" -ForegroundColor Red
    Write-Host "     Run: reload" -ForegroundColor Yellow
    exit
}
Write-Host ""

# Test 3: Check if talk command exists
Write-Host "TEST 3: Checking if 'talk' command exists..." -ForegroundColor Yellow
if (Get-Command talk -ErrorAction SilentlyContinue) {
    Write-Host "  ✅ 'talk' command found" -ForegroundColor Green
} else {
    Write-Host "  ❌ 'talk' command NOT found!" -ForegroundColor Red
    Write-Host "     Run: reload" -ForegroundColor Yellow
    exit
}
Write-Host ""

# Test 4: Try a simple AI call
Write-Host "TEST 4: Testing simple AI question (no codebase context)..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  Question: What is 2 + 2?" -ForegroundColor Gray
Write-Host ""

try {
    $result = ai "what is 2 plus 2?"

    if ($result) {
        Write-Host "  ✅ AI responded successfully!" -ForegroundColor Green
        Write-Host "     Response received (length: $($result.Length) chars)" -ForegroundColor Gray
    } else {
        Write-Host "  ⚠️  AI returned empty response" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ❌ Error calling AI: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║           ✅ TEST COMPLETE ✅                                    ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "If all tests passed, try:" -ForegroundColor Cyan
Write-Host "  talk hello, can you help me?" -ForegroundColor Yellow
Write-Host ""
Write-Host "Or start interactive chat:" -ForegroundColor Cyan
Write-Host "  chat" -ForegroundColor Yellow
Write-Host ""
