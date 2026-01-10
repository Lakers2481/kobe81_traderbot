# ╔══════════════════════════════════════════════════════════════════╗
# ║    🐍 MAMBA AI + KOBE TRADING BOT INTEGRATION 🐍                ║
# ║         Full Control Over Your Trading System                   ║
# ╚══════════════════════════════════════════════════════════════════╝

$script:KOBE_ROOT = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot"

# ══════════════════════════════════════════════════════════════════
# BACKTESTING COMMANDS
# ══════════════════════════════════════════════════════════════════

function Run-Backtest {
    <#
    .SYNOPSIS
    Run backtest with custom parameters

    .EXAMPLE
    Run-Backtest -Years 5
    Run-Backtest -Years 10 -Cap 200
    Run-Backtest -Start "2020-01-01" -End "2024-12-31"
    #>
    param(
        [int]$Years = 5,
        [string]$Start,
        [string]$End,
        [int]$Cap = 150,
        [string]$Universe = "data/universe/optionable_liquid_800.csv"
    )

    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║           🔬 RUNNING BACKTEST 🔬                                 ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""

    # Calculate dates if Years specified
    if (-not $Start -or -not $End) {
        $End = (Get-Date).ToString("yyyy-MM-dd")
        $Start = (Get-Date).AddYears(-$Years).ToString("yyyy-MM-dd")
    }

    Write-Host "📅 Period: $Start to $End ($Years years)" -ForegroundColor Cyan
    Write-Host "📊 Universe: $Universe" -ForegroundColor Cyan
    Write-Host "🎯 Cap: $Cap stocks" -ForegroundColor Cyan
    Write-Host ""

    Push-Location $script:KOBE_ROOT

    $cmd = "python scripts/backtest_dual_strategy.py --universe $Universe --start $Start --end $End --cap $Cap"
    Write-Host "🚀 Executing: $cmd" -ForegroundColor Gray
    Write-Host ""

    Invoke-Expression $cmd

    Pop-Location

    Write-Host ""
    Write-Host "✅ Backtest complete!" -ForegroundColor Green
    Write-Host ""
}

# ══════════════════════════════════════════════════════════════════
# WALK-FORWARD ANALYSIS
# ══════════════════════════════════════════════════════════════════

function Run-WalkForward {
    param(
        [int]$Years = 10,
        [int]$TrainDays = 252,
        [int]$TestDays = 63,
        [string]$Universe = "data/universe/optionable_liquid_800.csv"
    )

    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║           📈 WALK-FORWARD ANALYSIS 📈                            ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""

    $End = (Get-Date).ToString("yyyy-MM-dd")
    $Start = (Get-Date).AddYears(-$Years).ToString("yyyy-MM-dd")

    Write-Host "📅 Period: $Start to $End" -ForegroundColor Cyan
    Write-Host "🔄 Train: $TrainDays days | Test: $TestDays days" -ForegroundColor Cyan
    Write-Host ""

    Push-Location $script:KOBE_ROOT

    python scripts/run_wf_polygon.py --universe $Universe --start $Start --end $End --train-days $TrainDays --test-days $TestDays

    Write-Host ""
    Write-Host "📊 Generating HTML report..." -ForegroundColor Cyan
    python scripts/aggregate_wf_report.py --wfdir wf_outputs

    Pop-Location

    Write-Host ""
    Write-Host "✅ Walk-forward complete! Check wf_outputs/ for results" -ForegroundColor Green
    Write-Host ""
}

# ══════════════════════════════════════════════════════════════════
# DATA VERIFICATION
# ══════════════════════════════════════════════════════════════════

function Verify-Data {
    <#
    .SYNOPSIS
    Verify data quality and freshness
    #>
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║           🔍 DATA VERIFICATION 🔍                                ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""

    Push-Location $script:KOBE_ROOT

    Write-Host "1️⃣  Running preflight checks..." -ForegroundColor Cyan
    python scripts/preflight.py --dotenv ./.env
    Write-Host ""

    Write-Host "2️⃣  Verifying hash chain (tamper detection)..." -ForegroundColor Cyan
    python scripts/verify_hash_chain.py
    Write-Host ""

    Write-Host "3️⃣  Checking data coverage..." -ForegroundColor Cyan
    python scripts/validate_lake.py
    Write-Host ""

    Pop-Location

    Write-Host "✅ Data verification complete!" -ForegroundColor Green
    Write-Host ""
}

# ══════════════════════════════════════════════════════════════════
# SCANNING COMMANDS
# ══════════════════════════════════════════════════════════════════

function Run-Scan {
    <#
    .SYNOPSIS
    Run daily stock scanner

    .EXAMPLE
    Run-Scan
    Run-Scan -Cap 900 -Top5
    Run-Scan -WithMarkov
    #>
    param(
        [int]$Cap = 200,
        [switch]$Top5,
        [switch]$WithMarkov,
        [switch]$Deterministic
    )

    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║           🔎 DAILY SCANNER 🔎                                    ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""

    Push-Location $script:KOBE_ROOT

    $cmd = "python scripts/scan.py --cap $Cap"
    if ($Deterministic) { $cmd += " --deterministic" }
    if ($Top5) { $cmd += " --top5" }
    if ($WithMarkov) { $cmd += " --markov --markov-prefilter 100" }

    Write-Host "🚀 Executing: $cmd" -ForegroundColor Gray
    Write-Host ""

    Invoke-Expression $cmd

    Pop-Location

    Write-Host ""
    Write-Host "✅ Scan complete! Check logs/daily_top5.csv and logs/tradeable.csv" -ForegroundColor Green
    Write-Host ""
}

# ══════════════════════════════════════════════════════════════════
# SYSTEM AUDITS
# ══════════════════════════════════════════════════════════════════

function Run-FullAudit {
    <#
    .SYNOPSIS
    Run comprehensive system audit
    #>
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║           🔒 FULL SYSTEM AUDIT 🔒                                ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""

    Push-Location $script:KOBE_ROOT

    Write-Host "1️⃣  Code Quality Check..." -ForegroundColor Cyan
    python -m pytest tests/ --tb=short -v
    Write-Host ""

    Write-Host "2️⃣  Type Checking..." -ForegroundColor Cyan
    mypy strategies/ backtest/ risk/ execution/ --ignore-missing-imports
    Write-Host ""

    Write-Host "3️⃣  Security Audit..." -ForegroundColor Cyan
    python scripts/verify_hash_chain.py
    Write-Host ""

    Write-Host "4️⃣  Data Integrity..." -ForegroundColor Cyan
    python scripts/preflight.py --dotenv ./.env
    Write-Host ""

    Write-Host "5️⃣  Risk Gate Verification..." -ForegroundColor Cyan
    python -c "from risk.policy_gate import PolicyGate; print(PolicyGate.check())"
    Write-Host ""

    Pop-Location

    Write-Host "✅ Full audit complete!" -ForegroundColor Green
    Write-Host ""
}

# ══════════════════════════════════════════════════════════════════
# AI/ML COMPONENTS
# ══════════════════════════════════════════════════════════════════

function Run-AIComponent {
    <#
    .SYNOPSIS
    Run AI/ML components

    .EXAMPLE
    Run-AIComponent -Component "hmm"  # HMM regime detection
    Run-AIComponent -Component "markov"  # Markov chain analysis
    Run-AIComponent -Component "lstm"  # LSTM confidence model
    #>
    param(
        [Parameter(Mandatory=$true)]
        [ValidateSet('hmm', 'markov', 'lstm', 'ensemble', 'brain', 'research')]
        [string]$Component
    )

    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║           🤖 AI/ML COMPONENT: $($Component.ToUpper().PadRight(42)) ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""

    Push-Location $script:KOBE_ROOT

    switch ($Component) {
        'hmm' {
            Write-Host "🧠 Running HMM Regime Detection..." -ForegroundColor Cyan
            python -c "from ml_advanced.hmm_regime_detector import HMMRegimeDetector; detector = HMMRegimeDetector(); print(detector.detect_regime())"
        }
        'markov' {
            Write-Host "🎲 Running Markov Chain Analysis..." -ForegroundColor Cyan
            python -c "from ml_advanced.markov_chain.predictor import MarkovPredictor; predictor = MarkovPredictor(); print(predictor.predict())"
        }
        'lstm' {
            Write-Host "🔮 Running LSTM Confidence Model..." -ForegroundColor Cyan
            python -c "from ml_advanced.lstm_confidence.model import LSTMConfidenceModel; model = LSTMConfidenceModel(); print(model.predict_confidence())"
        }
        'ensemble' {
            Write-Host "🎯 Running Ensemble Predictor..." -ForegroundColor Cyan
            python -c "from ml_advanced.ensemble.ensemble_predictor import EnsemblePredictor; ensemble = EnsemblePredictor(); print(ensemble.predict())"
        }
        'brain' {
            Write-Host "🧠 Running Cognitive Brain..." -ForegroundColor Cyan
            python scripts/run_autonomous.py --status
        }
        'research' {
            Write-Host "🔬 Running Research OS..." -ForegroundColor Cyan
            python scripts/research_os_cli.py status
        }
    }

    Pop-Location

    Write-Host ""
    Write-Host "✅ AI component execution complete!" -ForegroundColor Green
    Write-Host ""
}

# ══════════════════════════════════════════════════════════════════
# HISTORICAL DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════

function Analyze-HistoricalData {
    <#
    .SYNOPSIS
    Analyze historical patterns and performance
    #>
    param(
        [string]$Symbol = "AAPL",
        [int]$Years = 5
    )

    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║           📊 HISTORICAL ANALYSIS: $($Symbol.PadRight(38)) ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""

    Push-Location $script:KOBE_ROOT

    Write-Host "📈 Analyzing $Symbol over $Years years..." -ForegroundColor Cyan
    python -c "
from analysis.historical_patterns import analyze_consecutive_days
result = analyze_consecutive_days('$Symbol', years=$Years)
print(result)
"

    Pop-Location

    Write-Host ""
    Write-Host "✅ Historical analysis complete!" -ForegroundColor Green
    Write-Host ""
}

# ══════════════════════════════════════════════════════════════════
# QUICK STATUS
# ══════════════════════════════════════════════════════════════════

function Show-KobeStatus {
    <#
    .SYNOPSIS
    Show comprehensive system status
    #>
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║           📊 KOBE TRADING SYSTEM STATUS 📊                       ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""

    Push-Location $script:KOBE_ROOT

    # Positions
    Write-Host "💼 POSITIONS:" -ForegroundColor Cyan
    if (Test-Path "state/positions/*.json") {
        Get-ChildItem "state/positions/*.json" | ForEach-Object {
            $pos = Get-Content $_.FullName | ConvertFrom-Json
            Write-Host "   $($pos.symbol): $($pos.shares) shares @ $$($pos.entry)" -ForegroundColor White
        }
    } else {
        Write-Host "   No open positions" -ForegroundColor Gray
    }
    Write-Host ""

    # Recent signals
    Write-Host "🎯 RECENT SIGNALS:" -ForegroundColor Cyan
    if (Test-Path "logs/signals.jsonl") {
        Get-Content "logs/signals.jsonl" -Tail 5 | ForEach-Object {
            $signal = $_ | ConvertFrom-Json
            Write-Host "   $($signal.timestamp): $($signal.symbol) $($signal.side)" -ForegroundColor White
        }
    } else {
        Write-Host "   No recent signals" -ForegroundColor Gray
    }
    Write-Host ""

    # System health
    Write-Host "🏥 SYSTEM HEALTH:" -ForegroundColor Cyan
    if (Test-Path "logs/events.jsonl") {
        $errors = Get-Content "logs/events.jsonl" -Tail 100 | Where-Object { $_ -like '*ERROR*' }
        if ($errors.Count -eq 0) {
            Write-Host "   ✅ No errors in last 100 events" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️  $($errors.Count) errors in last 100 events" -ForegroundColor Yellow
        }
    }
    Write-Host ""

    Pop-Location
}

# ══════════════════════════════════════════════════════════════════
# INITIALIZATION
# ══════════════════════════════════════════════════════════════════

Write-Host "🤝 Kobe Trading Integration loaded!" -ForegroundColor Green
Write-Host "   Type " -NoNewline -ForegroundColor Gray
Write-Host "Show-KobeCommands" -NoNewline -ForegroundColor Cyan
Write-Host " to see all commands" -ForegroundColor Gray

function Show-KobeCommands {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║           🐍 KOBE TRADING COMMANDS 🐍                            ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📊 BACKTESTING:" -ForegroundColor Cyan
    Write-Host "   Run-Backtest -Years 5" -ForegroundColor White
    Write-Host "   Run-Backtest -Start '2020-01-01' -End '2024-12-31'" -ForegroundColor White
    Write-Host "   Run-WalkForward -Years 10" -ForegroundColor White
    Write-Host ""
    Write-Host "🔍 SCANNING:" -ForegroundColor Cyan
    Write-Host "   Run-Scan" -ForegroundColor White
    Write-Host "   Run-Scan -Cap 900 -Top5 -WithMarkov" -ForegroundColor White
    Write-Host ""
    Write-Host "🔒 VERIFICATION:" -ForegroundColor Cyan
    Write-Host "   Verify-Data" -ForegroundColor White
    Write-Host "   Run-FullAudit" -ForegroundColor White
    Write-Host ""
    Write-Host "🤖 AI/ML:" -ForegroundColor Cyan
    Write-Host "   Run-AIComponent -Component hmm" -ForegroundColor White
    Write-Host "   Run-AIComponent -Component markov" -ForegroundColor White
    Write-Host "   Run-AIComponent -Component lstm" -ForegroundColor White
    Write-Host "   Run-AIComponent -Component brain" -ForegroundColor White
    Write-Host ""
    Write-Host "📈 ANALYSIS:" -ForegroundColor Cyan
    Write-Host "   Analyze-HistoricalData -Symbol AAPL -Years 5" -ForegroundColor White
    Write-Host "   Show-KobeStatus" -ForegroundColor White
    Write-Host ""
}
