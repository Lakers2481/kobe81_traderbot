# ╔══════════════════════════════════════════════════════════════════╗
# ║    🔒 KOBE TRADING SYSTEM - COMPLETE VERIFICATION 🔒            ║
# ║         Verify ALL Components Before Trading                    ║
# ╚══════════════════════════════════════════════════════════════════╝

$script:KOBE_ROOT = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot"
$script:PASS_COUNT = 0
$script:FAIL_COUNT = 0
$script:WARN_COUNT = 0

function Test-Component {
    param(
        [string]$Name,
        [scriptblock]$Test,
        [string]$Fix = "N/A",
        [switch]$Critical
    )

    Write-Host "  Testing: " -NoNewline -ForegroundColor Gray
    Write-Host $Name.PadRight(50) -NoNewline -ForegroundColor White

    try {
        $result = & $Test
        if ($result) {
            Write-Host "✅ PASS" -ForegroundColor Green
            $script:PASS_COUNT++
        } else {
            if ($Critical) {
                Write-Host "❌ FAIL (CRITICAL)" -ForegroundColor Red
                $script:FAIL_COUNT++
                Write-Host "     Fix: $Fix" -ForegroundColor Yellow
            } else {
                Write-Host "⚠️  WARN" -ForegroundColor Yellow
                $script:WARN_COUNT++
                Write-Host "     Note: $Fix" -ForegroundColor Gray
            }
        }
    } catch {
        if ($Critical) {
            Write-Host "❌ ERROR (CRITICAL)" -ForegroundColor Red
            $script:FAIL_COUNT++
            Write-Host "     Error: $($_.Exception.Message)" -ForegroundColor Red
        } else {
            Write-Host "⚠️  ERROR" -ForegroundColor Yellow
            $script:WARN_COUNT++
        }
    }
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
Write-Host "║           🔒 KOBE TRADING SYSTEM VERIFICATION 🔒                 ║" -ForegroundColor Yellow
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
Write-Host ""

# ══════════════════════════════════════════════════════════════════
# SECTION 1: API KEYS
# ══════════════════════════════════════════════════════════════════

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "📋 SECTION 1: API Keys" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

Test-Component "Polygon API Key" {
    $env:POLYGON_API_KEY -and $env:POLYGON_API_KEY.Length -gt 10
} "Set POLYGON_API_KEY in .env" -Critical

Test-Component "Alpaca API Key ID" {
    $env:ALPACA_API_KEY_ID -and $env:ALPACA_API_KEY_ID.Length -gt 10
} "Set ALPACA_API_KEY_ID in .env" -Critical

Test-Component "Alpaca Secret Key" {
    $env:ALPACA_API_SECRET_KEY -and $env:ALPACA_API_SECRET_KEY.Length -gt 10
} "Set ALPACA_API_SECRET_KEY in .env" -Critical

Test-Component "Alpaca Base URL" {
    $env:ALPACA_BASE_URL -like "https://paper-api.alpaca.markets*"
} "Set ALPACA_BASE_URL in .env" -Critical

Test-Component "Claude (Anthropic) API Key" {
    $env:ANTHROPIC_API_KEY -and $env:ANTHROPIC_API_KEY.StartsWith("sk-ant-")
} "Set ANTHROPIC_API_KEY in .env for Mamba AI"

Test-Component "Telegram Bot Token" {
    $env:TELEGRAM_BOT_TOKEN -and $env:TELEGRAM_BOT_TOKEN.Length -gt 20
} "Set TELEGRAM_BOT_TOKEN in .env for notifications"

Write-Host ""

# ══════════════════════════════════════════════════════════════════
# SECTION 2: CRITICAL FILES
# ══════════════════════════════════════════════════════════════════

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "📁 SECTION 2: Critical Files" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

Test-Component ".env file" {
    Test-Path "$script:KOBE_ROOT\.env"
} "Create .env file with API keys" -Critical

Test-Component "Universe file (800 stocks)" {
    Test-Path "$script:KOBE_ROOT\data\universe\optionable_liquid_800.csv"
} "Run: python scripts/build_universe_polygon.py" -Critical

Test-Component "Backtest script" {
    Test-Path "$script:KOBE_ROOT\scripts\backtest_dual_strategy.py"
} "Critical trading component missing!" -Critical

Test-Component "Scanner script" {
    Test-Path "$script:KOBE_ROOT\scripts\scan.py"
} "Critical trading component missing!" -Critical

Test-Component "Paper trading script" {
    Test-Path "$script:KOBE_ROOT\scripts\run_paper_trade.py"
} "Critical trading component missing!" -Critical

Test-Component "Preflight check script" {
    Test-Path "$script:KOBE_ROOT\scripts\preflight.py"
} "Critical verification component missing!" -Critical

Write-Host ""

# ══════════════════════════════════════════════════════════════════
# SECTION 3: PYTHON ENVIRONMENT
# ══════════════════════════════════════════════════════════════════

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "🐍 SECTION 3: Python Environment" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

Test-Component "Python installed" {
    $null -ne (Get-Command python -ErrorAction SilentlyContinue)
} "Install Python 3.11+" -Critical

Test-Component "Python version 3.11+" {
    $version = python --version 2>&1
    $version -match "Python 3\.(1[1-9]|[2-9]\d)"
} "Upgrade to Python 3.11+"

Test-Component "pip installed" {
    $null -ne (Get-Command pip -ErrorAction SilentlyContinue)
} "Install pip" -Critical

Test-Component "pandas installed" {
    python -c "import pandas" 2>&1 | Out-Null
    $LASTEXITCODE -eq 0
} "pip install pandas"

Test-Component "numpy installed" {
    python -c "import numpy" 2>&1 | Out-Null
    $LASTEXITCODE -eq 0
} "pip install numpy"

Test-Component "alpaca-py installed" {
    python -c "import alpaca" 2>&1 | Out-Null
    $LASTEXITCODE -eq 0
} "pip install alpaca-py"

Write-Host ""

# ══════════════════════════════════════════════════════════════════
# SECTION 4: TRADING STRATEGIES
# ══════════════════════════════════════════════════════════════════

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "📊 SECTION 4: Trading Strategies" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

Test-Component "DualStrategyScanner" {
    Test-Path "$script:KOBE_ROOT\strategies\dual_strategy.py"
} "Missing dual strategy!" -Critical

Test-Component "IBS+RSI Strategy" {
    Test-Path "$script:KOBE_ROOT\strategies\ibs_rsi\strategy.py"
} "Missing IBS+RSI strategy!"

Test-Component "Turtle Soup Strategy" {
    Test-Path "$script:KOBE_ROOT\strategies\ict\turtle_soup.py"
} "Missing Turtle Soup strategy!"

Test-Component "Strategy Registry" {
    Test-Path "$script:KOBE_ROOT\strategies\registry.py"
} "Missing strategy registry!" -Critical

Write-Host ""

# ══════════════════════════════════════════════════════════════════
# SECTION 5: RISK MANAGEMENT
# ══════════════════════════════════════════════════════════════════

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "🛡️  SECTION 5: Risk Management" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

Test-Component "PolicyGate" {
    Test-Path "$script:KOBE_ROOT\risk\policy_gate.py"
} "Missing risk policy gate!" -Critical

Test-Component "Equity Sizer" {
    Test-Path "$script:KOBE_ROOT\risk\equity_sizer.py"
} "Missing position sizer!" -Critical

Test-Component "Kill Zone Gate" {
    Test-Path "$script:KOBE_ROOT\risk\kill_zone_gate.py"
} "Missing kill zone protection!"

Test-Component "Weekly Exposure Gate" {
    Test-Path "$script:KOBE_ROOT\risk\weekly_exposure_gate.py"
} "Missing exposure limits!"

Test-Component "Kill Switch File Check" {
    -not (Test-Path "$script:KOBE_ROOT\state\KILL_SWITCH")
} "KILL_SWITCH is ACTIVE - remove to trade" -Critical

Write-Host ""

# ══════════════════════════════════════════════════════════════════
# SECTION 6: AI/ML COMPONENTS
# ══════════════════════════════════════════════════════════════════

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "🤖 SECTION 6: AI/ML Components" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

Test-Component "HMM Regime Detector" {
    Test-Path "$script:KOBE_ROOT\ml_advanced\hmm_regime_detector.py"
} "Missing HMM regime detection"

Test-Component "Markov Chain Module" {
    Test-Path "$script:KOBE_ROOT\ml_advanced\markov_chain\predictor.py"
} "Missing Markov chain predictor"

Test-Component "LSTM Confidence Model" {
    Test-Path "$script:KOBE_ROOT\ml_advanced\lstm_confidence\model.py"
} "Missing LSTM confidence model"

Test-Component "Cognitive Brain" {
    Test-Path "$script:KOBE_ROOT\cognitive\cognitive_brain.py"
} "Missing cognitive brain"

Test-Component "Research OS" {
    Test-Path "$script:KOBE_ROOT\research_os\orchestrator.py"
} "Missing research OS"

Write-Host ""

# ══════════════════════════════════════════════════════════════════
# SECTION 7: STATE DIRECTORIES
# ══════════════════════════════════════════════════════════════════

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "📂 SECTION 7: State Directories" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

Test-Component "State directory" {
    Test-Path "$script:KOBE_ROOT\state"
} "Create state directory" -Critical

Test-Component "Logs directory" {
    Test-Path "$script:KOBE_ROOT\logs"
} "Create logs directory" -Critical

Test-Component "Data directory" {
    Test-Path "$script:KOBE_ROOT\data"
} "Create data directory" -Critical

Test-Component "Positions directory" {
    Test-Path "$script:KOBE_ROOT\state\positions"
} "Create state/positions directory"

Test-Component "Watchlist directory" {
    Test-Path "$script:KOBE_ROOT\state\watchlist"
} "Create state/watchlist directory"

Write-Host ""

# ══════════════════════════════════════════════════════════════════
# SECTION 8: MAMBA AI INTEGRATION
# ══════════════════════════════════════════════════════════════════

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "🐍 SECTION 8: Mamba AI Integration" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

Test-Component "Mamba AI v2 (Base System)" {
    Test-Path "$script:KOBE_ROOT\mamba_ai_v2.ps1"
} "Mamba AI v2 missing!" -Critical

Test-Component "Mamba AI v3 (Autonomous)" {
    Test-Path "$script:KOBE_ROOT\mamba_ai_v3_fixed.ps1"
} "Mamba AI v3 missing!" -Critical

Test-Component "Trading Integration" {
    Test-Path "$script:KOBE_ROOT\mamba_trading_integration.ps1"
} "Trading integration missing!" -Critical

Test-Component "Natural Chat Mode" {
    Test-Path "$script:KOBE_ROOT\mamba_natural_chat.ps1"
} "Chat interface missing!"

Test-Component "ai command available" {
    $null -ne (Get-Command ai -ErrorAction SilentlyContinue)
} "Reload PowerShell to load Mamba AI"

Test-Component "talk command available" {
    $null -ne (Get-Command talk -ErrorAction SilentlyContinue)
} "Reload PowerShell to load chat"

Test-Component "chat command available" {
    $null -ne (Get-Command chat -ErrorAction SilentlyContinue)
} "Reload PowerShell to load interactive chat"

Write-Host ""

# ══════════════════════════════════════════════════════════════════
# SECTION 9: BROKER CONNECTION
# ══════════════════════════════════════════════════════════════════

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "🔌 SECTION 9: Broker Connection" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

Test-Component "Broker module exists" {
    Test-Path "$script:KOBE_ROOT\execution\broker_alpaca.py"
} "Missing broker connection module!" -Critical

Test-Component "Can connect to Alpaca (requires keys)" {
    if ($env:ALPACA_API_KEY_ID -and $env:ALPACA_API_SECRET_KEY) {
        Push-Location $script:KOBE_ROOT
        python -c "from execution.broker_alpaca import AlpacaBroker; broker = AlpacaBroker(); broker.get_account()" 2>&1 | Out-Null
        $result = $LASTEXITCODE -eq 0
        Pop-Location
        $result
    } else {
        $false
    }
} "Check API keys and network connection"

Write-Host ""

# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════

Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
Write-Host "║                    📊 VERIFICATION SUMMARY 📊                    ║" -ForegroundColor Yellow
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
Write-Host ""

Write-Host "  ✅ Passed:  " -NoNewline -ForegroundColor Green
Write-Host "$script:PASS_COUNT" -ForegroundColor White

Write-Host "  ⚠️  Warnings: " -NoNewline -ForegroundColor Yellow
Write-Host "$script:WARN_COUNT" -ForegroundColor White

Write-Host "  ❌ Failed:  " -NoNewline -ForegroundColor Red
Write-Host "$script:FAIL_COUNT" -ForegroundColor White

Write-Host ""

if ($script:FAIL_COUNT -eq 0) {
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║           ✅ SYSTEM READY FOR TRADING! ✅                        ║" -ForegroundColor Green
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "🎯 All critical components verified!" -ForegroundColor Green
    Write-Host "🚀 You can now run your trading bot safely" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Run-Scan                    # Daily scan" -ForegroundColor White
    Write-Host "  2. Run-Backtest -Years 5       # Test strategy" -ForegroundColor White
    Write-Host "  3. Show-KobeStatus             # Check status" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "║           ⚠️  CRITICAL ISSUES FOUND! ⚠️                          ║" -ForegroundColor Red
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Red
    Write-Host ""
    Write-Host "❌ $script:FAIL_COUNT critical component(s) failed verification" -ForegroundColor Red
    Write-Host "⚠️  DO NOT TRADE until these issues are resolved!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Review the failures above and apply the suggested fixes." -ForegroundColor Yellow
    Write-Host ""
}

if ($script:WARN_COUNT -gt 0) {
    Write-Host "⚠️  Note: $script:WARN_COUNT warning(s) - system will work but may have reduced functionality" -ForegroundColor Yellow
    Write-Host ""
}
