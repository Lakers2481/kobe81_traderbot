# ============================================================
# Kobe Trading System - Windows Task Scheduler Setup
# ============================================================
# Run this script as Administrator to create scheduled tasks.
#
# Tasks created:
# 1. KobeRunner - Starts the 24/7 runner at system startup
# 2. KobeHealthCheck - Monitors runner health every 5 minutes
# 3. KobeExitManager - Checks for time-based exits every 30 minutes
# ============================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot"
$PythonPath = "python"  # Or full path like "C:\Python311\python.exe"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Kobe Trading System - Task Scheduler Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: Run this script as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell -> Run as Administrator" -ForegroundColor Yellow
    exit 1
}

# ============================================================
# Task 1: KobeRunner - Main 24/7 Trading Runner
# ============================================================
Write-Host "`n[1/3] Creating KobeRunner task..." -ForegroundColor Yellow

$RunnerAction = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$ProjectRoot\scripts\start_runner.bat`"" `
    -WorkingDirectory $ProjectRoot

# Trigger: At system startup + daily at 9:00 AM ET (in case of restart)
$RunnerTrigger1 = New-ScheduledTaskTrigger -AtStartup
$RunnerTrigger2 = New-ScheduledTaskTrigger -Daily -At "9:00AM"

$RunnerSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0)  # No time limit

$RunnerPrincipal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Highest

# Remove existing task if exists
Unregister-ScheduledTask -TaskName "KobeRunner" -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask `
    -TaskName "KobeRunner" `
    -Action $RunnerAction `
    -Trigger $RunnerTrigger1, $RunnerTrigger2 `
    -Settings $RunnerSettings `
    -Principal $RunnerPrincipal `
    -Description "Kobe 24/7 Trading Runner - Scans and executes trades automatically"

Write-Host "  KobeRunner task created!" -ForegroundColor Green

# ============================================================
# Task 2: KobeHealthCheck - Monitor runner health
# ============================================================
Write-Host "`n[2/3] Creating KobeHealthCheck task..." -ForegroundColor Yellow

$HealthAction = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument "scripts/health_monitor.py --check-runner --alert-on-failure" `
    -WorkingDirectory $ProjectRoot

# Every 5 minutes during market hours (9 AM - 5 PM weekdays)
$HealthTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At "9:00AM" `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Hours 10)

$HealthSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable

Unregister-ScheduledTask -TaskName "KobeHealthCheck" -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask `
    -TaskName "KobeHealthCheck" `
    -Action $HealthAction `
    -Trigger $HealthTrigger `
    -Settings $HealthSettings `
    -Description "Monitors Kobe runner health and sends alerts if down"

Write-Host "  KobeHealthCheck task created!" -ForegroundColor Green

# ============================================================
# Task 3: KobeExitManager - Time-based exit monitoring
# ============================================================
Write-Host "`n[3/3] Creating KobeExitManager task..." -ForegroundColor Yellow

$ExitAction = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument "scripts/exit_manager.py --check-time-exits --execute" `
    -WorkingDirectory $ProjectRoot

# Every 30 minutes during market hours
$ExitTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At "10:00AM" `
    -RepetitionInterval (New-TimeSpan -Minutes 30) `
    -RepetitionDuration (New-TimeSpan -Hours 7)

$ExitSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable

Unregister-ScheduledTask -TaskName "KobeExitManager" -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask `
    -TaskName "KobeExitManager" `
    -Action $ExitAction `
    -Trigger $ExitTrigger `
    -Settings $ExitSettings `
    -Description "Monitors positions for time-based exits (7-bar rule)"

Write-Host "  KobeExitManager task created!" -ForegroundColor Green

# ============================================================
# Summary
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Tasks created:" -ForegroundColor White
Write-Host "  1. KobeRunner      - Starts at boot + 9 AM daily" -ForegroundColor Gray
Write-Host "  2. KobeHealthCheck - Every 5 min (9 AM - 5 PM)" -ForegroundColor Gray
Write-Host "  3. KobeExitManager - Every 30 min (10 AM - 5 PM)" -ForegroundColor Gray
Write-Host ""
Write-Host "To start immediately:" -ForegroundColor Yellow
Write-Host "  Start-ScheduledTask -TaskName 'KobeRunner'" -ForegroundColor White
Write-Host ""
Write-Host "To view tasks:" -ForegroundColor Yellow
Write-Host "  Get-ScheduledTask -TaskName 'Kobe*' | Format-Table" -ForegroundColor White
Write-Host ""
Write-Host "To disable automation:" -ForegroundColor Yellow
Write-Host "  Disable-ScheduledTask -TaskName 'KobeRunner'" -ForegroundColor White
