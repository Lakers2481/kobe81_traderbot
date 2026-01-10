# ╔══════════════════════════════════════════════════════════════════╗
# ║    🎮 KOBE TRADING SYSTEM - MASTER CONTROL PANEL 🎮             ║
# ║         One Command to Rule Them All                            ║
# ╚══════════════════════════════════════════════════════════════════╝

function Show-KobeControlPanel {
    Clear-Host
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║           🎮 KOBE TRADING SYSTEM CONTROL PANEL 🎮                ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""

    $continue = $true
    while ($continue) {
        Write-Host "┌────────────────────────────────────────────────────────────────┐" -ForegroundColor Cyan
        Write-Host "│  🔍 VERIFICATION & HEALTH                                      │" -ForegroundColor Cyan
        Write-Host "├────────────────────────────────────────────────────────────────┤" -ForegroundColor Cyan
        Write-Host "│  [1]  Full System Verification (MUST RUN FIRST!)              │" -ForegroundColor White
        Write-Host "│  [2]  Quick Status Check                                       │" -ForegroundColor White
        Write-Host "│  [3]  Verify Data Quality                                      │" -ForegroundColor White
        Write-Host "│  [4]  Run Preflight Check                                      │" -ForegroundColor White
        Write-Host "└────────────────────────────────────────────────────────────────┘" -ForegroundColor Cyan
        Write-Host ""

        Write-Host "┌────────────────────────────────────────────────────────────────┐" -ForegroundColor Green
        Write-Host "│  📊 BACKTESTING & ANALYSIS                                     │" -ForegroundColor Green
        Write-Host "├────────────────────────────────────────────────────────────────┤" -ForegroundColor Green
        Write-Host "│  [5]  Backtest 5 Years (Quick)                                 │" -ForegroundColor White
        Write-Host "│  [6]  Backtest 10 Years (Full)                                 │" -ForegroundColor White
        Write-Host "│  [7]  Walk-Forward Analysis                                    │" -ForegroundColor White
        Write-Host "│  [8]  Custom Backtest (Choose dates)                           │" -ForegroundColor White
        Write-Host "└────────────────────────────────────────────────────────────────┘" -ForegroundColor Green
        Write-Host ""

        Write-Host "┌────────────────────────────────────────────────────────────────┐" -ForegroundColor Magenta
        Write-Host "│  🔎 SCANNING & SIGNALS                                         │" -ForegroundColor Magenta
        Write-Host "├────────────────────────────────────────────────────────────────┤" -ForegroundColor Magenta
        Write-Host "│  [9]  Daily Scan (200 stocks)                                  │" -ForegroundColor White
        Write-Host "│  [10] Full Scan (800 stocks + Top5)                            │" -ForegroundColor White
        Write-Host "│  [11] Markov-Enhanced Scan                                     │" -ForegroundColor White
        Write-Host "│  [12] View Recent Signals                                      │" -ForegroundColor White
        Write-Host "└────────────────────────────────────────────────────────────────┘" -ForegroundColor Magenta
        Write-Host ""

        Write-Host "┌────────────────────────────────────────────────────────────────┐" -ForegroundColor Yellow
        Write-Host "│  🤖 AI/ML COMPONENTS                                           │" -ForegroundColor Yellow
        Write-Host "├────────────────────────────────────────────────────────────────┤" -ForegroundColor Yellow
        Write-Host "│  [13] HMM Regime Detection                                     │" -ForegroundColor White
        Write-Host "│  [14] Markov Chain Analysis                                    │" -ForegroundColor White
        Write-Host "│  [15] LSTM Confidence Model                                    │" -ForegroundColor White
        Write-Host "│  [16] Cognitive Brain Status                                   │" -ForegroundColor White
        Write-Host "└────────────────────────────────────────────────────────────────┘" -ForegroundColor Yellow
        Write-Host ""

        Write-Host "┌────────────────────────────────────────────────────────────────┐" -ForegroundColor Blue
        Write-Host "│  💬 MAMBA AI CHAT                                              │" -ForegroundColor Blue
        Write-Host "├────────────────────────────────────────────────────────────────┤" -ForegroundColor Blue
        Write-Host "│  [17] Start Interactive Chat                                   │" -ForegroundColor White
        Write-Host "│  [18] Ask Quick Question                                       │" -ForegroundColor White
        Write-Host "│  [19] Run Autonomous Task                                      │" -ForegroundColor White
        Write-Host "│  [20] Scan Codebase for Issues                                 │" -ForegroundColor White
        Write-Host "└────────────────────────────────────────────────────────────────┘" -ForegroundColor Blue
        Write-Host ""

        Write-Host "┌────────────────────────────────────────────────────────────────┐" -ForegroundColor Red
        Write-Host "│  🔒 RISK & SAFETY                                              │" -ForegroundColor Red
        Write-Host "├────────────────────────────────────────────────────────────────┤" -ForegroundColor Red
        Write-Host "│  [21] Check Risk Gates                                         │" -ForegroundColor White
        Write-Host "│  [22] View Kill Switch Status                                  │" -ForegroundColor White
        Write-Host "│  [23] Full System Audit                                        │" -ForegroundColor White
        Write-Host "│  [24] Verify Hash Chain (Tamper Detection)                     │" -ForegroundColor White
        Write-Host "└────────────────────────────────────────────────────────────────┘" -ForegroundColor Red
        Write-Host ""

        Write-Host "[0] Exit Control Panel" -ForegroundColor Gray
        Write-Host ""

        Write-Host "Select option: " -NoNewline -ForegroundColor Cyan
        $choice = Read-Host

        Write-Host ""
        Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
        Write-Host ""

        $kobe_root = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot"

        switch ($choice) {
            "1" {
                Write-Host "🔍 Running Full System Verification..." -ForegroundColor Yellow
                & "$kobe_root\verify_complete_system.ps1"
            }
            "2" {
                Write-Host "📊 Quick Status Check..." -ForegroundColor Cyan
                Show-KobeStatus
            }
            "3" {
                Write-Host "🔍 Verifying Data Quality..." -ForegroundColor Cyan
                Verify-Data
            }
            "4" {
                Write-Host "✈️  Running Preflight Check..." -ForegroundColor Cyan
                Push-Location $kobe_root
                python scripts/preflight.py --dotenv ./.env
                Pop-Location
            }
            "5" {
                Write-Host "📊 Running 5-Year Backtest..." -ForegroundColor Green
                Run-Backtest -Years 5
            }
            "6" {
                Write-Host "📊 Running 10-Year Backtest..." -ForegroundColor Green
                Run-Backtest -Years 10
            }
            "7" {
                Write-Host "📈 Running Walk-Forward Analysis..." -ForegroundColor Green
                Run-WalkForward -Years 10
            }
            "8" {
                Write-Host "Start date (YYYY-MM-DD): " -NoNewline
                $start = Read-Host
                Write-Host "End date (YYYY-MM-DD): " -NoNewline
                $end = Read-Host
                Write-Host "📊 Running Custom Backtest..." -ForegroundColor Green
                Run-Backtest -Start $start -End $end
            }
            "9" {
                Write-Host "🔎 Running Daily Scan (200 stocks)..." -ForegroundColor Magenta
                Run-Scan -Cap 200
            }
            "10" {
                Write-Host "🔎 Running Full Scan (800 stocks + Top5)..." -ForegroundColor Magenta
                Run-Scan -Cap 800 -Top5 -Deterministic
            }
            "11" {
                Write-Host "🎲 Running Markov-Enhanced Scan..." -ForegroundColor Magenta
                Run-Scan -Cap 800 -Top5 -WithMarkov -Deterministic
            }
            "12" {
                Write-Host "📋 Recent Signals:" -ForegroundColor Cyan
                if (Test-Path "$kobe_root\logs\signals.jsonl") {
                    Get-Content "$kobe_root\logs\signals.jsonl" -Tail 10 | ForEach-Object {
                        $signal = $_ | ConvertFrom-Json
                        Write-Host "  $($signal.timestamp) | $($signal.symbol) | $($signal.side) | Entry: $$($signal.entry_price)" -ForegroundColor White
                    }
                } else {
                    Write-Host "  No signals found" -ForegroundColor Gray
                }
            }
            "13" {
                Write-Host "🧠 Running HMM Regime Detection..." -ForegroundColor Yellow
                Run-AIComponent -Component hmm
            }
            "14" {
                Write-Host "🎲 Running Markov Chain Analysis..." -ForegroundColor Yellow
                Run-AIComponent -Component markov
            }
            "15" {
                Write-Host "🔮 Running LSTM Confidence Model..." -ForegroundColor Yellow
                Run-AIComponent -Component lstm
            }
            "16" {
                Write-Host "🧠 Checking Cognitive Brain Status..." -ForegroundColor Yellow
                Run-AIComponent -Component brain
            }
            "17" {
                Write-Host "💬 Starting Interactive Chat..." -ForegroundColor Blue
                Start-Chat
            }
            "18" {
                Write-Host "Question: " -NoNewline
                $question = Read-Host
                ai $question
            }
            "19" {
                Write-Host "Task: " -NoNewline
                $task = Read-Host
                ai-autonomous $task
            }
            "20" {
                Write-Host "🔍 Scanning Codebase for Issues..." -ForegroundColor Blue
                ai-scan-issues
            }
            "21" {
                Write-Host "🛡️  Checking Risk Gates..." -ForegroundColor Red
                Push-Location $kobe_root
                python -c "from risk.policy_gate import PolicyGate; gate = PolicyGate(); print('✅ Risk gates functional' if gate else '❌ Risk gates failed')"
                Pop-Location
            }
            "22" {
                Write-Host "🔪 Kill Switch Status..." -ForegroundColor Red
                if (Test-Path "$kobe_root\state\KILL_SWITCH") {
                    Write-Host "  ⚠️  KILL SWITCH IS ACTIVE - Trading BLOCKED!" -ForegroundColor Red
                } else {
                    Write-Host "  ✅ Kill Switch is OFF - Trading allowed" -ForegroundColor Green
                }
            }
            "23" {
                Write-Host "🔒 Running Full System Audit..." -ForegroundColor Red
                Run-FullAudit
            }
            "24" {
                Write-Host "🔗 Verifying Hash Chain..." -ForegroundColor Red
                Push-Location $kobe_root
                python scripts/verify_hash_chain.py
                Pop-Location
            }
            "0" {
                Write-Host "👋 Exiting Control Panel..." -ForegroundColor Gray
                $continue = $false
            }
            default {
                Write-Host "⚠️  Invalid option. Please try again." -ForegroundColor Yellow
            }
        }

        if ($continue) {
            Write-Host ""
            Write-Host "Press Enter to continue..." -ForegroundColor Gray
            Read-Host | Out-Null
            Clear-Host
            Write-Host ""
            Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
            Write-Host "║           🎮 KOBE TRADING SYSTEM CONTROL PANEL 🎮                ║" -ForegroundColor Yellow
            Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
            Write-Host ""
        }
    }
}

# Quick alias
Set-Alias -Name kobe -Value Show-KobeControlPanel
Set-Alias -Name control -Value Show-KobeControlPanel

Write-Host "🎮 Control Panel loaded! Type " -NoNewline -ForegroundColor Green
Write-Host "kobe" -NoNewline -ForegroundColor Cyan
Write-Host " or " -NoNewline -ForegroundColor Green
Write-Host "control" -NoNewline -ForegroundColor Cyan
Write-Host " to open" -ForegroundColor Green
