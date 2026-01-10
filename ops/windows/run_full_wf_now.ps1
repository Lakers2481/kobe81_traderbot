param(
  [string]$RepoRoot = (Resolve-Path "..\..").Path,
  [string]$PythonExe = "python",
  [string]$DotEnv = ".\\.env",
  [string]$Universe = "data/universe/optionable_liquid_800.csv",
  [string]$Start = "2020-01-02",
  [string]$End = "2025-12-26",
  [int]$TrainDays = 252,
  [int]$TestDays = 63,
  [int]$Cap = 150,
  [string]$OutDir = "wf_outputs_verify_2020_2025",
  [string]$ConfigPath = "config/base_backtest.yaml"
)

$ErrorActionPreference = 'Stop'
Set-Location $RepoRoot

# Use backtest overrides (disable regime/earnings/quality gates for WF)
$env:KOBE_CONFIG_PATH = (Resolve-Path $ConfigPath).Path

Write-Host "[WF] Using config: $($env:KOBE_CONFIG_PATH)"

# Run both strategies (IBS+RSI + Turtle Soup)
$args = @(
  "scripts/run_wf_polygon.py",
  "--universe", $Universe,
  "--start", $Start,
  "--end", $End,
  "--train-days", $TrainDays,
  "--test-days", $TestDays,
  "--cap", $Cap,
  "--outdir", $OutDir,
  "--fallback-free",
  "--dotenv", $DotEnv
)

& $PythonExe @args

# Summarize metrics to JSON for each strategy (if present)
if (Test-Path $OutDir) {
  $ibsJson = Join-Path $OutDir 'metrics_ibs_rsi.json'
  $tsJson  = Join-Path $OutDir 'metrics_turtle_soup.json'
  try { & $PythonExe "scripts/metrics.py" --wfdir $OutDir --strategy IBS_RSI --json | Set-Content -Encoding UTF8 $ibsJson } catch {}
  try { & $PythonExe "scripts/metrics.py" --wfdir $OutDir --strategy TURTLE_SOUP --json | Set-Content -Encoding UTF8 $tsJson } catch {}
}

# Drop a done marker
$done = Join-Path $OutDir ("WF_DONE_" + (Get-Date).ToString('yyyyMMdd_HHmm') + ".txt")
New-Item -ItemType File -Path $done -Force | Out-Null
Write-Host "[WF] Completed. Artifacts in $OutDir"

