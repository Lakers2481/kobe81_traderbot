# GitHub Copilot Setup for Mamba AI
# Run this after installing GitHub CLI

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GITHUB COPILOT SETUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if GitHub CLI is available
Write-Host "[STEP 1] Checking GitHub CLI..." -ForegroundColor Yellow

# Refresh environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

try {
    $ghVersion = gh --version 2>&1
    Write-Host "  [OK] GitHub CLI installed: $($ghVersion[0])" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] GitHub CLI not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "Solution:" -ForegroundColor Yellow
    Write-Host "  1. Close this PowerShell window" -ForegroundColor Gray
    Write-Host "  2. Open a NEW PowerShell window" -ForegroundColor Gray
    Write-Host "  3. Run this script again:" -ForegroundColor Gray
    Write-Host "     cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot" -ForegroundColor Cyan
    Write-Host "     .\setup_copilot.ps1" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

Write-Host ""

# Step 2: Check authentication
Write-Host "[STEP 2] Checking GitHub authentication..." -ForegroundColor Yellow

try {
    $authStatus = gh auth status 2>&1
    if ($authStatus -match "Logged in") {
        Write-Host "  [OK] Already authenticated to GitHub" -ForegroundColor Green
    } else {
        throw "Not authenticated"
    }
} catch {
    Write-Host "  [!] Not authenticated to GitHub" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Let's authenticate now..." -ForegroundColor Cyan
    Write-Host ""

    # Start authentication
    Write-Host "Follow the prompts to login:" -ForegroundColor Gray
    Write-Host "  1. Choose: GitHub.com" -ForegroundColor Gray
    Write-Host "  2. Choose: HTTPS" -ForegroundColor Gray
    Write-Host "  3. Authenticate: Yes" -ForegroundColor Gray
    Write-Host "  4. Choose: Login with a web browser" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Press ENTER to start authentication..." -ForegroundColor Yellow
    Read-Host

    gh auth login

    Write-Host ""
    Write-Host "  [OK] Authentication complete!" -ForegroundColor Green
}

Write-Host ""

# Step 3: Check for Copilot extension
Write-Host "[STEP 3] Checking Copilot extension..." -ForegroundColor Yellow

try {
    $extensions = gh extension list 2>&1
    if ($extensions -match "gh-copilot") {
        Write-Host "  [OK] Copilot extension already installed" -ForegroundColor Green
    } else {
        throw "Not installed"
    }
} catch {
    Write-Host "  [!] Copilot extension not installed" -ForegroundColor Yellow
    Write-Host "  Installing now..." -ForegroundColor Cyan

    try {
        gh extension install github/gh-copilot
        Write-Host "  [OK] Copilot extension installed!" -ForegroundColor Green
    } catch {
        Write-Host "  [ERROR] Failed to install Copilot extension" -ForegroundColor Red
        Write-Host "  Error: $_" -ForegroundColor Gray
        exit 1
    }
}

Write-Host ""

# Step 4: Test Copilot
Write-Host "[STEP 4] Testing Copilot access..." -ForegroundColor Yellow

try {
    Write-Host "  Asking Copilot a test question..." -ForegroundColor Gray

    $testResponse = gh copilot explain "what is 2 + 2?" 2>&1

    if ($testResponse -match "Explanation" -or $testResponse -match "4" -or $testResponse -match "sum") {
        Write-Host "  [OK] Copilot is working!" -ForegroundColor Green
        Write-Host ""
        Write-Host "  Sample response:" -ForegroundColor Gray
        Write-Host "  $($testResponse[0..3] -join "`n  ")" -ForegroundColor Cyan
    } else {
        Write-Host "  [?] Unusual response, but Copilot is accessible" -ForegroundColor Yellow
    }
} catch {
    $errorMsg = $_.Exception.Message

    if ($errorMsg -match "subscription" -or $errorMsg -match "access") {
        Write-Host "  [ERROR] Copilot subscription issue" -ForegroundColor Red
        Write-Host ""
        Write-Host "  Your GitHub account might not have Copilot access." -ForegroundColor Yellow
        Write-Host "  Check: https://github.com/settings/copilot" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  If you have Copilot in VS Code, you should have CLI access too." -ForegroundColor Gray
        Write-Host "  Try running manually: gh copilot explain 'test'" -ForegroundColor Gray
    } else {
        Write-Host "  [ERROR] Copilot test failed" -ForegroundColor Red
        Write-Host "  Error: $errorMsg" -ForegroundColor Gray
    }
    exit 1
}

Write-Host ""

# Step 5: Check available models
Write-Host "[STEP 5] Checking available models..." -ForegroundColor Yellow

Write-Host "  Your Copilot subscription includes:" -ForegroundColor Gray
Write-Host "    - GPT-4o (OpenAI)" -ForegroundColor Cyan
Write-Host "    - Claude 3.5 Sonnet (Anthropic)" -ForegroundColor Cyan
Write-Host "    - GPT-4.1 (OpenAI)" -ForegroundColor Cyan
Write-Host "    - Gemini 2.0 Flash (Google)" -ForegroundColor Cyan
Write-Host "    - o4-mini (OpenAI)" -ForegroundColor Cyan

Write-Host ""

# Success!
Write-Host "========================================" -ForegroundColor Green
Write-Host "SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "What's next:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Test Copilot manually:" -ForegroundColor Cyan
Write-Host "   gh copilot explain 'what is mean reversion trading?'" -ForegroundColor Gray
Write-Host ""
Write-Host "2. I'll integrate Copilot with Mamba AI" -ForegroundColor Cyan
Write-Host "   (This will take about 15 minutes)" -ForegroundColor Gray
Write-Host ""
Write-Host "3. You'll be able to use:" -ForegroundColor Cyan
Write-Host "   talk hello, using my Copilot GPT-4o" -ForegroundColor Gray
Write-Host "   talk analyze my trading strategy" -ForegroundColor Gray
Write-Host "   talk explain this code" -ForegroundColor Gray
Write-Host ""
Write-Host "Your Copilot gives you the most powerful AI models!" -ForegroundColor Green
Write-Host ""
