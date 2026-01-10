param(
  [string]$DotEnv = ".\\.env",
  [string]$Universe = "data\universe\optionable_liquid_800.csv",
  [int]$Cap = 50,
  [string]$ScanTimes = "09:35,10:30,15:55",
  [int]$LookbackDays = 540,
  [int]$HealthPort = 8081
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
Set-Location ..

Write-Host "Starting Kobe health server on port $HealthPort..."
Start-Process -WindowStyle Minimized -FilePath python -ArgumentList @('scripts/start_health.py','--port',"$HealthPort")

Start-Sleep -Seconds 2

Write-Host "Starting Kobe runner (paper) with cap=$Cap, scan-times=$ScanTimes..."
Start-Process -WindowStyle Minimized -FilePath python -ArgumentList @('scripts/runner.py','--mode','paper','--universe',"$Universe",'--cap',"$Cap",'--scan-times',"$ScanTimes",'--lookback-days',"$LookbackDays",'--dotenv',"$DotEnv")

Write-Host "Launched. Use Task Scheduler for auto-start at boot (see docs/RUN_24x7.md)."
