param(
  [string]$Mode = 'paper',
  [int]$Cap = 120,
  [string]$ScanTimes = '09:35,10:30,15:55'
)

$ErrorActionPreference = 'Stop'

# Project root
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

# Ensure config path and env are set
$env:KOBE_CONFIG_PATH = 'config/base.yaml'

# Launch runner
Write-Host "Starting Kobe81 runner..." -ForegroundColor Green
python scripts/runner.py --mode $Mode --universe data/universe/optionable_liquid_800.csv --cap $Cap --scan-times $ScanTimes --lookback-days 540 --dotenv ./.env

